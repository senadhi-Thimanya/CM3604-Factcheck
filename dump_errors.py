
import csv

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from factcheck import EntailmentModel, EntailmentFactChecker, clean_passage
from factchecking_main import read_passages, read_fact_examples




MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

OUTPUT_PATH = "experiments/entailment_errors.csv"


# 2. Load dataset and model


print("Loading dataset...")

fact_to_passage_dict = read_passages(PASSAGES_PATH)

examples = read_fact_examples(
    LABELS_PATH,
    fact_to_passage_dict
)

print(f"Loaded {len(examples)} examples.")

print("\nLoading DeBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

ent_model = EntailmentModel(
    model,
    tokenizer,
    cuda=(device.type == "cuda")
)

# Same settings as the submitted factcheck.py (threshold, pruning cutoff)
checker = EntailmentFactChecker(ent_model)

print(
    f"Threshold: {checker.entailment_threshold}, "
    f"pruning cutoff: {checker.overlap_cutoff} ({checker.overlap_method}), "
    f"clean text: {checker.clean_text}"
)



# 3. Re-run the checker's logic, keeping the best sentence for each fact


rows = []
confusion = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}

for example in tqdm(examples):

    best_score = 0.0
    best_sentence = ""
    best_probs = None

    for passage in example.passages:

        passage_text = passage.get("text", "")

        if not passage_text or not passage_text.strip():
            continue

        if checker.clean_text:
            passage_text = clean_passage(passage_text)

        doc = checker.nlp(passage_text)

        for sentence in doc.sents:

            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            if checker._overlap(example.fact, sentence_text) < checker.overlap_cutoff:
                continue

            result = ent_model.check_entailment(
                premise=sentence_text,
                hypothesis=example.fact
            )

            if best_probs is None or result["entailment"] > best_score:
                best_score = result["entailment"]
                best_sentence = sentence_text
                best_probs = result

    prediction = "S" if best_score >= checker.entailment_threshold else "NS"
    gold = "NS" if example.label == "IR" else example.label

    if gold == "S":
        outcome = "TP" if prediction == "S" else "FN"
    else:
        outcome = "FP" if prediction == "S" else "TN"

    confusion[outcome] += 1

    if outcome in ("FP", "FN"):
        rows.append({
            "error_type": outcome,
            "fact": example.fact,
            "gold_label": example.label,
            "predicted": prediction,
            "max_entailment": round(best_score, 4),
            "neutral": round(best_probs["neutral"], 4) if best_probs else "",
            "contradiction": round(best_probs["contradiction"], 4) if best_probs else "",
            "best_sentence": best_sentence,
            "best_sentence_overlap": (
                round(checker._overlap(example.fact, best_sentence), 4)
                if best_sentence else ""
            ),
            "your_category": "",
        })



# 4. Save the errors: FPs from most to least confident, then FNs


rows.sort(
    key=lambda row: (
        row["error_type"] != "FP",
        -row["max_entailment"] if row["error_type"] == "FP" else row["max_entailment"]
    )
)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

total = sum(confusion.values())
accuracy = (confusion["TP"] + confusion["TN"]) / total

print(f"\nConfusion: {confusion}")
print(f"Accuracy: {accuracy:.4f}")
print(
    f"Saved {len(rows)} errors "
    f"({confusion['FP']} FP, {confusion['FN']} FN) to {OUTPUT_PATH}"
)
