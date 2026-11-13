"""
error_analysis.py  (version 2)

Finds the errors of the FINAL entailment model (same threshold and pruning as factcheck.py),
proves that it reproduces the report's numbers, tests the explanations offered in Part 3 on ALL
221 facts (not just on the errors), and prepares everything needed to rewrite Part 3.

factcheck.py is NOT modified. All experiments are done by wrapping the checker's model.

Outputs (experiments/part2/results/):
  error_analysis.csv          all errors with extra evidence columns
  error_sample.csv            10 FP + 10 FN (seed 42, same sample as before) with an empty
                              'your_category' column.  If you already labelled error_sample.csv,
                              it is NOT overwritten: error_sample_enriched.csv is written instead
                              and your existing labels are copied into it.
  error_feature_summary.csv   language/evidence patterns per group + Fisher exact p-values
  per_topic_errors.csv        errors per main passage title (a proxy for "person")
  experiment_summary.csv      net effect of window / title / pronoun fixes on all 221 facts
  experiment_scores.csv       per-fact scores for every experiment
  score_distribution.png      entailment scores of correct / FP / FN facts
  run_stats.txt               time and GPU memory (main pass only)
  error_analysis_summary.txt  every number printed below, ready to copy into the report

Run:  python error_analysis.py                  (from anywhere inside the project)
      python error_analysis.py --no-experiments  (skip the extra model passes, about 4 x 70 s on GPU)
"""

import csv
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# find the project root (the folder that holds factcheck.py), wherever this script is saved
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "factcheck.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                    # data/ paths below are relative to the project root
RESULTS = ROOT / "experiments" / "part2" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)        # all csv/txt/png output goes here

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

from factcheck import EntailmentModel, EntailmentFactChecker, remove_noise
from factchecking_main import read_passages, read_fact_examples


# ------------------------------------------------------------------ configuration
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
RANDOM_SEED = 42
EXPECTED_TOTAL = 221          # numbers in the submitted report; a mismatch prints a loud warning
EXPECTED_CORRECT = 189
MAX_TOKENS = 512              # EntailmentModel truncates to this length
TOP_K_TOPICS = 7              # the report says "22 of 32 errors come from seven people"
N_PERMUTATIONS = 10000
MARGIN_BAND = 0.10            # "borderline" = within this distance of the threshold
RUN_EXPERIMENTS = "--no-experiments" not in sys.argv

LOG = []                      # everything logged here is also written to error_analysis_summary.txt


def log(message=""):
    print(message)
    LOG.append(str(message))


# ---- helpers (no model needed) ----------------------------------------------------------------
PRONOUNS = {"he", "she", "they", "his", "her", "their", "it", "its", "him", "them"}

# 1) The ORIGINAL list from the submitted report (kept so the numbers stay comparable).
ORIGINAL_VAGUE_WORDS = {
    "various", "numerous", "popular", "talented", "accomplished", "impact", "well-known",
    "legendary", "prominent", "passion", "powerful", "dynamic", "simplicity", "clarity",
    "successful", "famous", "notable", "several", "many", "leading", "top",
}
# 2) Two narrower lexicons that separate the two ideas that the single list mixed up.
#    NOTE: like the original list, these were written after reading the errors, so any p-value
#    attached to them is exploratory.  Say so in the report.
GENERIC_QUANTIFIER_WORDS = {"various", "numerous", "several", "many", "multiple", "different"}
GENERIC_QUANTIFIER_PHRASES = ("a number of", "a variety of", "a range of", "a lot of")
EVALUATIVE_WORDS = {
    "popular", "famous", "well-known", "successful", "talented", "accomplished", "legendary",
    "prominent", "notable", "renowned", "acclaimed", "influential", "powerful", "dynamic",
}
EVALUATIVE_PHRASES = ("known for", "best known", "impact on")

_TAG_RE = re.compile(r"</?s>")
_WORD_RE = re.compile(r"[A-Za-z0-9'\-]+")
_LEAD_PRONOUN_RE = re.compile(r"^((?:</?s>\s*)*)(She|He|They|His|Her|Their)\b")


def mentions_named_entity(words):
    """Rough check for a capitalised name (a pronoun or article at position 0 does not count)."""
    skip_first = PRONOUNS | {"the", "a", "an", "in", "on", "as", "at"}
    for i, w in enumerate(words):
        if not w[0].isupper():
            continue
        if i == 0 and w.lower() in skip_first:
            continue
        return True
    return False


def fact_features(fact):
    words = _WORD_RE.findall(fact)
    lowered = [w.lower() for w in words]
    text = fact.lower()
    return {
        "starts_with_pronoun": bool(words) and lowered[0] in PRONOUNS,
        "has_number_or_date": bool(re.search(r"\d", fact)),
        "has_vague_word": any(w in ORIGINAL_VAGUE_WORDS for w in lowered) or "a number of" in text,
        "has_generic_quantifier": any(w in GENERIC_QUANTIFIER_WORDS for w in lowered)
                                  or any(p in text for p in GENERIC_QUANTIFIER_PHRASES),
        "has_evaluative_word": any(w in EVALUATIVE_WORDS for w in lowered)
                               or any(p in text for p in EVALUATIVE_PHRASES),
        "mentions_named_entity": mentions_named_entity(words),
        "short_fact_6_words_or_fewer": len(words) <= 6,
    }


def sentence_starts_with_pronoun(sentence):
    words = _WORD_RE.findall(_TAG_RE.sub(" ", sentence))
    return bool(words) and words[0].lower() in PRONOUNS


def resolve_leading_pronoun(text, name):
    """'<s>She won ...' -> '<s>Jane Doe won ...' (His/Her/Their -> "Jane Doe's"). Leading pronoun only."""
    match = _LEAD_PRONOUN_RE.match(text)
    if not match or not name:
        return text
    pronoun = match.group(2).lower()
    replacement = name + "'s" if pronoun in {"his", "her", "their"} else name
    return match.group(1) + replacement + text[match.end():]


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]] (no scipy needed)."""
    n, row1, col1 = a + b + c + d, a + b, a + c
    denominator = math.comb(n, col1)

    def prob(x):
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / denominator

    low, high = max(0, col1 - (n - row1)), min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(low, high + 1) if prob(x) <= observed * (1 + 1e-9)))


THRESHOLD_GRID = [round(0.10 + 0.01 * i, 2) for i in range(79)]      # 0.10 ... 0.88, as in the report


def main_title(passages):
    """Most common Wikipedia page title among the retrieved passages (proxy for the topic/person)."""
    titles = [p.get("title", "") for p in passages if p.get("title")]
    return Counter(titles).most_common(1)[0][0] if titles else "UNKNOWN"


# ---- end of helpers ----------------------------------------------------------------------------


def split_passage(checker, passage):
    """Exactly the checker's own text handling (same cleaning, same sentencizer)."""
    text = passage.get("text", "")
    if not text or not text.strip():
        return []
    if checker.clean_text:
        text = remove_noise(text)
    return [s.text.strip() for s in checker.nlp(text).sents if s.text.strip()]


def sentence_candidates(checker, example):
    """(sentence, page title) for every sentence that survives the checker's pruning."""
    out = []
    for passage in example.passages:
        title = passage.get("title", "")
        for sentence in split_passage(checker, passage):
            if checker._overlap(example.fact, sentence) >= checker.overlap_cutoff:
                out.append((sentence, title))
    return out


def window_candidates(checker, example):
    """Same, for two neighbouring sentences read together (pruned on the window text)."""
    out = []
    for passage in example.passages:
        title = passage.get("title", "")
        sentences = split_passage(checker, passage)
        for i in range(len(sentences) - 1):
            window = sentences[i] + " " + sentences[i + 1]
            if checker._overlap(example.fact, window) >= checker.overlap_cutoff:
                out.append((window, title))
    return out


def token_count(tokenizer, premise, hypothesis):
    return len(tokenizer(premise, hypothesis, truncation=False)["input_ids"])


def scored_sentences(checker, tokenizer, example):
    """All sentences that survive pruning with probabilities and token length, best first."""
    out = []
    for sentence, _ in sentence_candidates(checker, example):
        probs = checker.ent_model.check_entailment(premise=sentence, hypothesis=example.fact)
        out.append({"score": probs["entailment"], "sentence": sentence, "probs": probs,
                    "n_tokens": token_count(tokenizer, sentence, example.fact)})
    out.sort(key=lambda item: item["score"], reverse=True)
    return out


# --- text transformations tested as possible fixes (the checker's model is only wrapped, never edited)
def t_title_prefix(fact, premise, sent_title, topic):
    return (f"{sent_title}: {premise}" if sent_title else premise), fact


def t_pronoun_premise(fact, premise, sent_title, topic):
    return resolve_leading_pronoun(premise, sent_title), fact


def t_pronoun_fact_and_premise(fact, premise, sent_title, topic):
    return resolve_leading_pronoun(premise, sent_title), resolve_leading_pronoun(fact, topic)


def confusion(scores, golds, threshold):
    tp = sum(1 for s, g in zip(scores, golds) if g == "S" and s >= threshold)
    fn = sum(1 for s, g in zip(scores, golds) if g == "S" and s < threshold)
    fp = sum(1 for s, g in zip(scores, golds) if g == "NS" and s >= threshold)
    tn = sum(1 for s, g in zip(scores, golds) if g == "NS" and s < threshold)
    return tp, fn, fp, tn


def sweep(scores, golds):
    """Best in-sample accuracy over the threshold grid (middle of the plateau, as in the report)."""
    accs = [sum((s >= t) == (g == "S") for s, g in zip(scores, golds)) for t in THRESHOLD_GRID]
    best = max(accs)
    hits = [i for i, a in enumerate(accs) if a == best]
    return best / len(golds), THRESHOLD_GRID[hits[len(hits) // 2]], THRESHOLD_GRID[hits[0]], THRESHOLD_GRID[hits[-1]]


# ------------------------------------------------------------------ load data and model
log("Loading dataset...")
fact_to_passage_dict = read_passages(PASSAGES_PATH)
examples = read_fact_examples(LABELS_PATH, fact_to_passage_dict)
log(f"Loaded {len(examples)} examples.")

log("\nLoading DeBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
hf_logging.set_verbosity_error()          # silences the "sequence longer than 512" tokenizer warning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ent_model = EntailmentModel(model, tokenizer, cuda=(device.type == "cuda"))
log(f"Using device: {device}")

# SAME checker (threshold + pruning) as the final factcheck.py
checker = EntailmentFactChecker(ent_model)
THRESHOLD = checker.entailment_threshold
log(f"Using threshold: {THRESHOLD}  |  pruning cutoff: {checker.overlap_cutoff}")


# ------------------------------------------------------------------ predict every example (timed pass)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

records = []      # one entry per example (correct or not)
errors = []

log("\nAnalysing predictions...")
start_time = time.time()

for idx, example in enumerate(tqdm(examples)):
    checked_before = checker.checked_count
    max_score, best_sentence = checker.max_entailment_score(
        example.fact, example.passages, return_sentence=True
    )
    n_checked = checker.checked_count - checked_before      # sentences that reached the model
    pruned_all = n_checked == 0                             # nothing survived pruning (score 0 is then an artefact)

    gold_label = "NS" if example.label == "IR" else example.label     # IR is treated as NS
    predicted_label = "S" if max_score >= THRESHOLD else "NS"

    if predicted_label == gold_label:
        group = "Correct"
    elif gold_label == "NS":
        group = "False Positive"
    else:
        group = "False Negative"

    features = fact_features(example.fact)
    evidence = {
        "pruned_all": pruned_all,
        "best_sentence_starts_with_pronoun": sentence_starts_with_pronoun(best_sentence),
    }
    topic = main_title(example.passages)
    records.append({"idx": idx, "group": group, "topic": topic, "fact": example.fact, "gold": gold_label,
                    "pred": predicted_label, "score": max_score, **features, **evidence})

    if group != "Correct":
        errors.append({
            "_idx": idx,
            "_passages": example.passages,
            "fact": example.fact,
            "original_gold_label": example.label,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "error_type": group,
            "max_entailment_score": round(max_score, 6),
            "margin_to_threshold": round(max_score - THRESHOLD, 4),
            "best_evidence_sentence": best_sentence,
            "topic": topic,
            "n_sentences_checked": n_checked,
            **evidence,
            **features,
        })

elapsed = time.time() - start_time
peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None

# ------------------------------------------------------------------ does this script reproduce the report?
golds = [r["gold"] for r in records]
base_scores = [r["score"] for r in records]
n_correct = sum(1 for r in records if r["group"] == "Correct")
false_pos = [e for e in errors if e["error_type"] == "False Positive"]
false_neg = [e for e in errors if e["error_type"] == "False Negative"]

log("\nREPRODUCTION CHECK")
log("=" * 45)
log(f"Correct: {n_correct}/{len(examples)}  (report: {EXPECTED_CORRECT}/{EXPECTED_TOTAL})")
if (len(examples), n_correct) != (EXPECTED_TOTAL, EXPECTED_CORRECT):
    log("WARNING: this does NOT match the report. Do not trust the numbers below until it does.")
tp, fn, fp, tn = confusion(base_scores, golds, THRESHOLD)
log(f"Confusion matrix  S->S {tp}  S->NS {fn}  NS->S {fp}  NS->NS {tn}  (report: 92 / 20 / 12 / 97)")
p_s, r_s = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
p_n, r_n = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
log(f"S : precision {p_s:.3f} recall {r_s:.3f} F1 {2 * p_s * r_s / max(p_s + r_s, 1e-9):.3f}")
log(f"NS: precision {p_n:.3f} recall {r_n:.3f} F1 {2 * p_n * r_n / max(p_n + r_n, 1e-9):.3f}")


# ------------------------------------------------------------------ experiments on ALL 221 facts
variants = {}          # name -> list of scores (one per fact)
window_texts = []

if RUN_EXPERIMENTS:
    log("\nBuilding candidate sentences and windows (same pruning as the checker)...")
    sent_cands = [sentence_candidates(checker, ex) for ex in tqdm(examples, desc="sentences")]
    win_cands = [window_candidates(checker, ex) for ex in tqdm(examples, desc="windows")]
    topics = [r["topic"] for r in records]

    def run_experiment(name, cands, transform):
        scores, texts = [], []
        for i, ex in enumerate(tqdm(examples, desc=name)):
            best, best_text = 0.0, ""
            for text, sent_title in cands[i]:
                premise, hypothesis = transform(ex.fact, text, sent_title, topics[i])
                s = ent_model.check_entailment(premise=premise, hypothesis=hypothesis)["entailment"]
                if s > best:
                    best, best_text = s, text
            scores.append(best)
            texts.append(best_text)
        return scores, texts

    identity = lambda fact, premise, sent_title, topic: (premise, fact)
    window_scores, window_texts = run_experiment("window_only", win_cands, identity)
    variants["sentence_or_window"] = [max(b, w) for b, w in zip(base_scores, window_scores)]
    variants["title_prefix"], _ = run_experiment("title_prefix", sent_cands, t_title_prefix)
    variants["pronoun_premise"], _ = run_experiment("pronoun_premise", sent_cands, t_pronoun_premise)
    variants["pronoun_fact_and_premise"], _ = run_experiment(
        "pronoun_fact_and_premise", sent_cands, t_pronoun_fact_and_premise)

    log("\nEXPERIMENTS ON ALL FACTS (same threshold, NOT re-tuned; 'best' is in-sample and optimistic)")
    log("=" * 100)
    log(f"{'variant':<28}{'acc@thr':>8}{'FP':>5}{'FN':>5}{'fixed':>7}{'broken':>8}{'best acc':>10}{'best thr':>10}")
    summary_rows = []
    for name, scores in [("baseline (final system)", base_scores)] + list(variants.items()):
        preds = ["S" if s >= THRESHOLD else "NS" for s in scores]
        v_tp, v_fn, v_fp, v_tn = confusion(scores, golds, THRESHOLD)
        fixed = sum(1 for r, p in zip(records, preds) if r["pred"] != r["gold"] and p == r["gold"])
        broken = sum(1 for r, p in zip(records, preds) if r["pred"] == r["gold"] and p != r["gold"])
        best_acc, best_thr, lo, hi = sweep(scores, golds)
        acc = (v_tp + v_tn) / len(golds)
        log(f"{name:<28}{acc:>8.3f}{v_fp:>5}{v_fn:>5}{fixed:>7}{broken:>8}{best_acc:>10.3f}{best_thr:>10}")
        summary_rows.append({"variant": name, "accuracy_at_threshold": round(acc, 4), "TP": v_tp, "FN": v_fn,
                             "FP": v_fp, "TN": v_tn, "errors_fixed": fixed, "correct_broken": broken,
                             "best_accuracy_in_sample": round(best_acc, 4), "best_threshold": best_thr,
                             "plateau_low": lo, "plateau_high": hi})
    with open(RESULTS / "experiment_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(RESULTS / "experiment_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "topic", "fact", "gold", "baseline_score", "window_only_score"] + list(variants))
        for i, r in enumerate(records):
            writer.writerow([i, r["topic"], r["fact"], r["gold"], round(base_scores[i], 4),
                             round(window_scores[i], 4)] + [round(variants[v][i], 4) for v in variants])
    log("Reading: 'fixed' = errors the variant gets right; 'broken' = correct answers it gets wrong.")
    log("A fix only counts if fixed clearly exceeds broken.")


# ------------------------------------------------------------------ extra evidence for the errors
log("\nCollecting extra evidence for the errors...")
mismatches = 0

for e in tqdm(errors):
    example = examples[e["_idx"]]
    ranked = scored_sentences(checker, tokenizer, example)

    if ranked:
        top = ranked[0]
        if abs(top["score"] - e["max_entailment_score"]) > 1e-4:
            mismatches += 1               # script and checker disagree about the best sentence
        e["p_entailment"] = round(top["probs"]["entailment"], 4)
        e["p_neutral"] = round(top["probs"]["neutral"], 4)
        e["p_contradiction"] = round(top["probs"]["contradiction"], 4)
        e["best_sentence_tokens"] = top["n_tokens"]
        e["n_candidates_over_512_tokens"] = sum(1 for item in ranked if item["n_tokens"] > MAX_TOKENS)
        e["max_candidate_tokens"] = max(item["n_tokens"] for item in ranked)
    else:
        if e["max_entailment_score"] != 0:
            mismatches += 1
        e["p_entailment"] = e["p_neutral"] = e["p_contradiction"] = ""
        e["best_sentence_tokens"] = e["max_candidate_tokens"] = ""
        e["n_candidates_over_512_tokens"] = 0

    e["top3_sentences"] = " || ".join(f"[{item['score']:.2f}] {item['sentence']}" for item in ranked[:3])

    if RUN_EXPERIMENTS:
        i = e["_idx"]
        e["window_only_text"] = window_texts[i]
        e["window_only_score"] = round(window_scores[i], 4)
        for name, scores in variants.items():
            pred = "S" if scores[i] >= THRESHOLD else "NS"
            e[f"{name}_score"] = round(scores[i], 4)
            e[f"{name}_would_fix_error"] = pred == e["gold_label"]

log(f"Best-sentence mismatches between this script and the checker: {mismatches} (should be 0)")


# ------------------------------------------------------------------ save all errors
base_fields = [
    "fact", "original_gold_label", "gold_label", "predicted_label", "error_type",
    "max_entailment_score", "margin_to_threshold", "best_evidence_sentence",
]
evidence_fields = [
    "topic", "n_sentences_checked", "pruned_all", "best_sentence_starts_with_pronoun",
    "best_sentence_tokens", "max_candidate_tokens", "n_candidates_over_512_tokens",
    "p_entailment", "p_neutral", "p_contradiction", "top3_sentences",
]
experiment_fields = []
if RUN_EXPERIMENTS:
    experiment_fields = ["window_only_score", "window_only_text"]
    for name in variants:
        experiment_fields += [f"{name}_score", f"{name}_would_fix_error"]
FEATURE_NAMES = list(fact_features("x").keys())
all_fields = base_fields + evidence_fields + experiment_fields + FEATURE_NAMES

with open(RESULTS / "error_analysis.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_fields, extrasaction="ignore", restval="")
    writer.writeheader()
    writer.writerows(errors)


# ------------------------------------------------------------------ 10 + 10 random sample (same as before)
random.seed(RANDOM_SEED)     # mention this seed in the report; keep this order (FP first, then FN)
sample = (
    random.sample(false_pos, min(10, len(false_pos)))
    + random.sample(false_neg, min(10, len(false_neg)))
)

sample_file = RESULTS / "error_sample.csv"
carried = {}
if sample_file.exists():
    with open(sample_file, newline="", encoding="utf-8") as f:
        carried = {r["fact"]: r.get("your_category", "").strip()
                   for r in csv.DictReader(f) if r.get("your_category", "").strip()}
    if carried:
        sample_file = RESULTS / "error_sample_enriched.csv"
        log("\nNOTE: error_sample.csv already has your categories, so it was NOT overwritten.")
        log(f"The version with extra columns is {sample_file.name}; your labels were copied into it.")
for e in sample:
    e["your_category"] = carried.get(e["fact"], "")

with open(sample_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_fields + ["your_category"], extrasaction="ignore", restval="")
    writer.writeheader()
    writer.writerows(sample)


# ------------------------------------------------------------------ language / evidence patterns + Fisher tests
groups = ["Correct", "False Positive", "False Negative"]
pattern_names = FEATURE_NAMES + ["pruned_all", "best_sentence_starts_with_pronoun"]
by_group = {g: [r for r in records if r["group"] == g] for g in groups}
all_errors = by_group["False Positive"] + by_group["False Negative"]


def count_true(rows, name):
    return sum(1 for r in rows if r[name])


summary_rows = []
log("\nHOW OFTEN EACH PATTERN APPEARS (% of facts in each group; p = Fisher exact vs Correct, uncorrected)")
log("=" * 118)
header = f"{'Pattern':<36}" + "".join(f"{g + ' (n=' + str(len(by_group[g])) + ')':<22}" for g in groups)
log(header + f"{'p(FP)':>8}{'p(FN)':>8}{'p(all err)':>12}")
log("-" * 118)

for name in pattern_names:
    row = {"pattern": name}
    line = f"{name:<36}"
    for g in groups:
        members = by_group[g]
        pct = 100 * count_true(members, name) / len(members) if members else 0.0
        row[g] = round(pct, 1)
        line += f"{pct:>6.1f}%{'':<15}"
    c_true, c_n = count_true(by_group["Correct"], name), len(by_group["Correct"])
    for label, members in [("p_FP", by_group["False Positive"]), ("p_FN", by_group["False Negative"]),
                           ("p_all_errors", all_errors)]:
        t = count_true(members, name)
        p = fisher_two_sided(t, len(members) - t, c_true, c_n - c_true) if members else float("nan")
        row[label] = round(p, 4)
    line += f"{row['p_FP']:>8.3f}{row['p_FN']:>8.3f}{row['p_all_errors']:>12.3f}"
    summary_rows.append(row)
    log(line)
log("These lexicon features were written after reading the errors, so treat p-values as exploratory.")

with open(RESULTS / "error_feature_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["pattern"] + groups + ["p_FP", "p_FN", "p_all_errors"])
    writer.writeheader()
    writer.writerows(summary_rows)


# ------------------------------------------------------------------ are the errors clustered by topic?
per_topic = defaultdict(lambda: {"facts": 0, "errors": 0, "FP": 0, "FN": 0})
for r in records:
    t = per_topic[r["topic"]]
    t["facts"] += 1
    if r["group"] != "Correct":
        t["errors"] += 1
        t["FP" if r["group"] == "False Positive" else "FN"] += 1

ranked_topics = sorted(per_topic.items(), key=lambda kv: (-kv[1]["errors"], -kv[1]["facts"]))
with open(RESULTS / "per_topic_errors.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["topic (main passage title)", "facts", "errors", "false_positives", "false_negatives",
                     "error_rate"])
    for name, t in ranked_topics:
        writer.writerow([name, t["facts"], t["errors"], t["FP"], t["FN"], round(t["errors"] / t["facts"], 3)])

k = min(TOP_K_TOPICS, len(ranked_topics))
top_k = ranked_topics[:k]
obs_errors = sum(t["errors"] for _, t in top_k)
obs_facts = sum(t["facts"] for _, t in top_k)

topic_ids = {name: i for i, name in enumerate(per_topic)}
ids = np.array([topic_ids[r["topic"]] for r in records])
is_error = np.array([r["group"] != "Correct" for r in records])
rng = np.random.default_rng(RANDOM_SEED)
null_stats = []
for _ in range(N_PERMUTATIONS):
    shuffled = rng.permutation(is_error)
    counts = np.bincount(ids[shuffled], minlength=len(topic_ids))
    null_stats.append(np.sort(counts)[::-1][:k].sum())
p_cluster = (1 + sum(1 for s in null_stats if s >= obs_errors)) / (1 + N_PERMUTATIONS)

log("\nERROR CLUSTERING BY TOPIC (topic = most common passage title; may differ from the person's name)")
log("=" * 80)
log(f"Distinct topics: {len(per_topic)}")
log(f"Top {k} topics by errors hold {obs_errors} of {len(errors)} errors "
    f"({100 * obs_errors / max(len(errors), 1):.0f}%) but only {obs_facts} of {len(examples)} facts "
    f"({100 * obs_facts / len(examples):.0f}%)")
log(f"Permutation test (errors shuffled across facts, {N_PERMUTATIONS} runs): p = {p_cluster:.4f}")
log("  a small p means errors are more concentrated than chance; compare with per_topic_errors.csv")


# ------------------------------------------------------------------ are the errors borderline or confident?
log("\nSCORE MARGINS")
log("=" * 45)
for label, rows in [("False positives", false_pos), ("False negatives", false_neg)]:
    margins = [abs(e["margin_to_threshold"]) for e in rows]
    scores = sorted(e["max_entailment_score"] for e in rows)
    near = sum(1 for m in margins if m <= MARGIN_BAND)
    if rows:
        log(f"{label}: n={len(rows)}, score min {scores[0]:.2f} / median {scores[len(scores) // 2]:.2f} / "
            f"max {scores[-1]:.2f}; within +-{MARGIN_BAND} of the threshold: {near}")
log("If almost no errors are borderline, re-tuning the threshold cannot fix them.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    jitter = np.random.default_rng(RANDOM_SEED)
    for i, g in enumerate(groups):
        ys = [r["score"] for r in by_group[g]]
        ax.scatter(i + jitter.uniform(-0.18, 0.18, len(ys)), ys, s=14, alpha=0.7)
    ax.axhline(THRESHOLD, linestyle="--", color="grey", label=f"threshold {THRESHOLD}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={len(by_group[g])})" for g in groups])
    ax.set_ylabel("max entailment probability")
    ax.set_title("Entailment score of each fact, by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "score_distribution.png", dpi=200)
    plt.close(fig)
    log("Saved score_distribution.png")
except ImportError:
    log("matplotlib not installed: score_distribution.png skipped")


# ------------------------------------------------------------------ report numbers
n_ir = sum(1 for e in errors if e["original_gold_label"] == "IR")
fp_labels = Counter(e["original_gold_label"] for e in false_pos)
n_pruned_errors = sum(1 for e in errors if e["pruned_all"])
n_pruned_fn = sum(1 for e in false_neg if e["pruned_all"])
n_truncated = sum(1 for e in errors if e.get("n_candidates_over_512_tokens", 0) > 0)
n_truncated_best = sum(1 for e in errors if isinstance(e.get("best_sentence_tokens"), int)
                       and e["best_sentence_tokens"] > MAX_TOKENS)
zero_fn = [e for e in false_neg if e["max_entailment_score"] < 0.005]

log("\nERROR ANALYSIS SUMMARY")
log("=" * 45)
log(f"Total examples: {len(examples)}")
log(f"Correct predictions: {n_correct}")
log(f"Incorrect predictions: {len(errors)}")
log(f"False Positives: {len(false_pos)}   (gold label NS: {fp_labels.get('NS', 0)}, IR: {fp_labels.get('IR', 0)})")
log(f"False Negatives: {len(false_neg)}")
log(f"Accuracy: {n_correct / len(examples):.4f}")
log(f"Errors whose gold label was IR (treated as NS): {n_ir}")
log(f"Errors where EVERY sentence was pruned (score 0 is an artefact): {n_pruned_errors} "
    f"(false negatives: {n_pruned_fn})")
log(f"False negatives with score < 0.005 (model genuinely sees no support): {len(zero_fn)} of {len(false_neg)}")
log(f"Errors with at least one candidate sentence over {MAX_TOKENS} tokens (truncated): {n_truncated}")
log(f"Errors whose BEST sentence is over {MAX_TOKENS} tokens: {n_truncated_best}")
log(f"Errors where the best sentence starts with a pronoun: "
    f"{sum(1 for e in errors if e['best_sentence_starts_with_pronoun'])} of {len(errors)}")

stats = [
    f"Device: {device}",
    f"Threshold: {THRESHOLD}",
    f"Pruning cutoff: {checker.overlap_cutoff}",
    f"Time for {len(examples)} facts: {elapsed:.1f} s ({elapsed / len(examples):.2f} s per fact)  [main pass only]",
    f"Peak GPU memory: {peak_mb:.0f} MB" if peak_mb is not None else "Peak GPU memory: n/a (CPU run)",
]
with open(RESULTS / "run_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(stats) + "\n")

log("\n" + "\n".join(stats))
with open(RESULTS / "error_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
print(f"\nSaved to {RESULTS}: error_analysis.csv, {sample_file.name}, error_feature_summary.csv, "
      "per_topic_errors.csv, experiment_summary.csv, experiment_scores.csv, score_distribution.png, "
      "run_stats.txt, error_analysis_summary.txt")