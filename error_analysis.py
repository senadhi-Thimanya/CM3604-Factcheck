"""
error_analysis.py

Finds the errors of the FINAL entailment model (same threshold and pruning as
factcheck.py) and prepares everything needed for Part 3:

  error_analysis.csv        all errors, with extra evidence columns
  error_sample.csv          10 false positives + 10 false negatives (seed 42),
                            with an empty 'your_category' column for you to fill
  error_feature_summary.csv how often each language pattern appears in correct
                            answers, false positives and false negatives
  run_stats.txt             time and GPU memory for the report's efficiency section

Run:  python error_analysis.py
"""

import csv
import os
import random
import re
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples


# ------------------------------------------------------------------ configuration
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
RANDOM_SEED = 42

PRONOUNS = {"he", "she", "they", "his", "her", "their", "it", "its", "him", "them"}

# subjective or vague wording seen in the facts (e.g. "talented", "various", "legendary")
VAGUE_WORDS = {
    "various", "numerous", "popular", "talented", "accomplished", "impact", "well-known",
    "legendary", "prominent", "passion", "powerful", "dynamic", "simplicity", "clarity",
    "successful", "famous", "notable", "several", "many", "leading", "top",
}


# ------------------------------------------------------------------ language features of a fact
def fact_features(fact):
    words = re.findall(r"[A-Za-z0-9'\-]+", fact)
    lowered = [w.lower() for w in words]
    return {
        "starts_with_pronoun": bool(words) and lowered[0] in PRONOUNS,
        "has_number_or_date": bool(re.search(r"\d", fact)),
        "has_vague_word": any(w in VAGUE_WORDS for w in lowered) or "a number of" in fact.lower(),
        "short_fact_6_words_or_fewer": len(words) <= 6,
    }


FEATURE_NAMES = list(fact_features("x").keys())


# ------------------------------------------------------------------ extra evidence for the errors
def split_passage(checker, passage):
    text = passage.get("text", "")
    if not text or not text.strip():
        return []
    return [s.text.strip() for s in checker.nlp(text).sents if s.text.strip()]


def scored_sentences(checker, fact, passages):
    """All sentences that survive pruning, with entailment probability, best first."""
    out = []
    for passage in passages:
        for sentence in split_passage(checker, passage):
            if checker._overlap(fact, sentence) < checker.overlap_cutoff:
                continue
            probs = checker.ent_model.check_entailment(premise=sentence, hypothesis=fact)
            out.append((probs["entailment"], sentence, probs))
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def best_two_sentence_window(checker, fact, passages):
    """Best entailment score when two neighbouring sentences are read together."""
    best_score, best_text = 0.0, ""
    for passage in passages:
        sentences = split_passage(checker, passage)
        for i in range(len(sentences) - 1):
            window = sentences[i] + " " + sentences[i + 1]
            if checker._overlap(fact, window) < checker.overlap_cutoff:
                continue
            score = checker.ent_model.check_entailment(premise=window, hypothesis=fact)["entailment"]
            if score > best_score:
                best_score, best_text = score, window
    return best_score, best_text


# ------------------------------------------------------------------ load data and model
print("Loading dataset...")
fact_to_passage_dict = read_passages(PASSAGES_PATH)
examples = read_fact_examples(LABELS_PATH, fact_to_passage_dict)
print(f"Loaded {len(examples)} examples.")

print("\nLoading DeBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ent_model = EntailmentModel(model, tokenizer, cuda=(device.type == "cuda"))
print(f"Using device: {device}")

# SAME checker (threshold + pruning) as the final factcheck.py
checker = EntailmentFactChecker(ent_model)
THRESHOLD = checker.entailment_threshold
print(f"Using threshold: {THRESHOLD}  |  pruning cutoff: {checker.overlap_cutoff}")


# ------------------------------------------------------------------ predict every example
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

records = []      # one entry per example (correct or not), for the feature summary
errors = []

print("\nAnalysing predictions...")
start_time = time.time()

for example in tqdm(examples):
    max_score, best_sentence = checker.max_entailment_score(
        example.fact, example.passages, return_sentence=True
    )

    gold_label = "NS" if example.label == "IR" else example.label     # IR is treated as NS
    predicted_label = "S" if max_score >= THRESHOLD else "NS"

    if predicted_label == gold_label:
        group = "Correct"
    elif gold_label == "NS":
        group = "False Positive"
    else:
        group = "False Negative"

    features = fact_features(example.fact)
    records.append({"group": group, **features})

    if group == "Correct":
        continue

    errors.append({
        "fact": example.fact,
        "original_gold_label": example.label,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "error_type": group,
        "max_entailment_score": round(max_score, 6),
        "best_evidence_sentence": best_sentence,
        **features,
        "_passages": example.passages,      # used below, not saved
    })

elapsed = time.time() - start_time
peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None


# ------------------------------------------------------------------ extra evidence for errors only (cheap)
print("\nCollecting extra evidence for the errors...")

for e in tqdm(errors):
    ranked = scored_sentences(checker, e["fact"], e["_passages"])

    if ranked:
        _, _, probs = ranked[0]
        e["p_entailment"] = round(probs["entailment"], 4)
        e["p_neutral"] = round(probs["neutral"], 4)
        e["p_contradiction"] = round(probs["contradiction"], 4)
    else:
        e["p_entailment"] = e["p_neutral"] = e["p_contradiction"] = ""

    e["top3_sentences"] = " || ".join(f"[{s:.2f}] {t}" for s, t, _ in ranked[:3])

    window_score, window_text = best_two_sentence_window(checker, e["fact"], e["_passages"])
    window_prediction = "S" if window_score >= THRESHOLD else "NS"
    e["two_sentence_window_score"] = round(window_score, 4)
    e["two_sentence_window_text"] = window_text
    e["window_would_fix_error"] = window_prediction == e["gold_label"]

    del e["_passages"]


# ------------------------------------------------------------------ save all errors
base_fields = [
    "fact", "original_gold_label", "gold_label", "predicted_label", "error_type",
    "max_entailment_score", "best_evidence_sentence",
]
extra_fields = [
    "p_entailment", "p_neutral", "p_contradiction", "top3_sentences",
    "two_sentence_window_score", "two_sentence_window_text", "window_would_fix_error",
] + FEATURE_NAMES
all_fields = base_fields + extra_fields

with open("error_analysis.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(errors)


# ------------------------------------------------------------------ 10 + 10 random sample
false_pos = [e for e in errors if e["error_type"] == "False Positive"]
false_neg = [e for e in errors if e["error_type"] == "False Negative"]

random.seed(RANDOM_SEED)     # mention this seed in the report
sample = (
    random.sample(false_pos, min(10, len(false_pos)))
    + random.sample(false_neg, min(10, len(false_neg)))
)
for e in sample:
    e["your_category"] = ""      # fill in by hand

sample_file = "error_sample.csv"
if os.path.exists(sample_file):
    with open(sample_file, newline="", encoding="utf-8") as f:
        already_labelled = any(r.get("your_category", "").strip() for r in csv.DictReader(f))
    if already_labelled:
        sample_file = "error_sample_enriched.csv"
        print("\nWARNING: error_sample.csv already has your categories, so it was NOT overwritten.")
        print(f"The new version with extra columns is saved as {sample_file}.")

with open(sample_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_fields + ["your_category"])
    writer.writeheader()
    writer.writerows(sample)


# ------------------------------------------------------------------ language-pattern summary
groups = ["Correct", "False Positive", "False Negative"]
summary_rows = []

print("\nHOW OFTEN EACH LANGUAGE PATTERN APPEARS (% of facts in each group)")
print("=" * 78)
print(f"{'Pattern':<32}" + "".join(f"{g + ' (n=' + str(sum(1 for r in records if r['group'] == g)) + ')':<24}" for g in groups))
print("-" * 78)

for name in FEATURE_NAMES:
    line = f"{name:<32}"
    row = {"pattern": name}
    for g in groups:
        members = [r for r in records if r["group"] == g]
        pct = 100 * sum(r[name] for r in members) / len(members) if members else 0.0
        row[g] = round(pct, 1)
        line += f"{pct:>6.1f}%{'':<17}"
    summary_rows.append(row)
    print(line)

with open("error_feature_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["pattern"] + groups)
    writer.writeheader()
    writer.writerows(summary_rows)


# ------------------------------------------------------------------ report numbers
n_ir = sum(1 for e in errors if e["original_gold_label"] == "IR")
fn_fixed = sum(1 for e in false_neg if e["window_would_fix_error"])
fp_fixed = sum(1 for e in false_pos if e["window_would_fix_error"])
correct_count = len(examples) - len(errors)

print("\nERROR ANALYSIS SUMMARY")
print("=" * 45)
print(f"Total examples: {len(examples)}")
print(f"Correct predictions: {correct_count}")
print(f"Incorrect predictions: {len(errors)}")
print(f"False Positives: {len(false_pos)}")
print(f"False Negatives: {len(false_neg)}")
print(f"Accuracy: {correct_count / len(examples):.4f}")
print(f"Errors whose gold label was IR (treated as NS): {n_ir}")
print(f"False negatives a 2-sentence window would fix: {fn_fixed} of {len(false_neg)}")
print(f"False positives a 2-sentence window would fix: {fp_fixed} of {len(false_pos)}")

stats = [
    f"Device: {device}",
    f"Threshold: {THRESHOLD}",
    f"Pruning cutoff: {checker.overlap_cutoff}",
    f"Time for {len(examples)} facts: {elapsed:.1f} s ({elapsed / len(examples):.2f} s per fact)",
    f"Peak GPU memory: {peak_mb:.0f} MB" if peak_mb is not None else "Peak GPU memory: n/a (CPU run)",
]
with open("run_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(stats) + "\n")

print("\n" + "\n".join(stats))
print("\nSaved: error_analysis.csv, " + sample_file + ", error_feature_summary.csv, run_stats.txt")