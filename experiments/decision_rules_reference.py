
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import spacy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from factcheck import EntailmentModel
from factchecking_main import read_passages, read_fact_examples




MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

# Rule 1 scores are probabilities (0..1)
RULE1_THRESHOLDS = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]

# Rule 3 scores are entailment minus contradiction (-1..1)
RULE3_THRESHOLDS = [round(t, 2) for t in np.arange(-0.99, 1.00, 0.01)]

# Optional: python decision_rules.py 10  -> quick test on the first 10 facts
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


# 2. Load dataset


print("Loading dataset...")

fact_to_passage_dict = read_passages(PASSAGES_PATH)

examples = read_fact_examples(
    LABELS_PATH,
    fact_to_passage_dict
)

if LIMIT:
    examples = examples[:LIMIT]

print(f"Loaded {len(examples)} examples.")



# 3. Load DeBERTa model


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

print(f"Using device: {device}")



# 4. Initialise sentence splitter


nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# 5. Calculate the scores for all three rules ONCE


print("\nCalculating entailment scores...")

rule1_scores = []   # max entailment probability
rule2_flags = []    # any sentence whose argmax label is entailment
rule3_scores = []   # max (entailment - contradiction)
gold_labels = []

for example in tqdm(examples):

    max_entailment_score = 0.0
    max_diff_score = -1.0
    any_argmax_entail = False

    for passage in example.passages:

        passage_text = passage.get("text", "")

        if not passage_text or not passage_text.strip():
            continue

        doc = nlp(passage_text)

        for sentence in doc.sents:

            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            result = ent_model.check_entailment(
                premise=sentence_text,
                hypothesis=example.fact
            )

            # Rule 1: maximum entailment probability
            max_entailment_score = max(
                max_entailment_score,
                result["entailment"]
            )

            # Rule 2: discrete argmax decision
            if (
                result["entailment"] > result["neutral"]
                and result["entailment"] > result["contradiction"]
            ):
                any_argmax_entail = True

            # Rule 3: entailment minus contradiction
            max_diff_score = max(
                max_diff_score,
                result["entailment"] - result["contradiction"]
            )

    rule1_scores.append(max_entailment_score)
    rule2_flags.append(any_argmax_entail)
    rule3_scores.append(max_diff_score)

    # Match the original evaluation logic:
    # IR is treated as NS
    gold_label = (
        "NS" if example.label == "IR"
        else example.label
    )

    gold_labels.append(gold_label)



# 6. Evaluate a list of scores over a list of thresholds


def evaluate(scores, gold_labels, thresholds):

    results = []

    for threshold in thresholds:

        confusion = np.zeros((2, 2), dtype=int)

        for score, gold in zip(scores, gold_labels):

            prediction = (
                "S" if score >= threshold
                else "NS"
            )

            gold_index = 0 if gold == "S" else 1
            pred_index = 0 if prediction == "S" else 1

            confusion[gold_index][pred_index] += 1

        # Confusion matrix:
        # [[TP, FN],
        #  [FP, TN]]

        tp = confusion[0][0]
        fn = confusion[0][1]
        fp = confusion[1][0]
        tn = confusion[1][1]

        accuracy = (tp + tn) / len(scores)

        s_precision = tp / (tp + fp) if tp + fp else 0
        s_recall = tp / (tp + fn) if tp + fn else 0

        s_f1 = (
            2 * s_precision * s_recall /
            (s_precision + s_recall)
            if s_precision + s_recall else 0
        )

        ns_precision = tn / (tn + fn) if tn + fn else 0
        ns_recall = tn / (tn + fp) if tn + fp else 0

        ns_f1 = (
            2 * ns_precision * ns_recall /
            (ns_precision + ns_recall)
            if ns_precision + ns_recall else 0
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "s_f1": s_f1,
            "ns_f1": ns_f1,
            "confusion": confusion
        })

    return results


rule1_results = evaluate(rule1_scores, gold_labels, RULE1_THRESHOLDS)

# Rule 2 has no threshold: True -> 1.0, False -> 0.0, then cut at 0.5
rule2_scores = [1.0 if flag else 0.0 for flag in rule2_flags]
rule2_results = evaluate(rule2_scores, gold_labels, [0.5])

rule3_results = evaluate(rule3_scores, gold_labels, RULE3_THRESHOLDS)



# 7. Summary table: best threshold per rule


rules = [
    ("Max entailment prob", rule1_results),
    ("Discrete argmax", rule2_results),
    ("Entail - contradict", rule3_results),
]

print("\nDECISION RULE COMPARISON")
print("=" * 75)

print(
    f"{'Rule':<24}"
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'S F1':<10}"
    f"{'NS F1':<10}"
)

print("-" * 75)

for name, results in rules:

    best = max(
        results,
        key=lambda result: result["accuracy"]
    )

    threshold = "n/a" if name == "Discrete argmax" else f"{best['threshold']:.2f}"

    print(
        f"{name:<24}"
        f"{threshold:<12}"
        f"{best['accuracy']:<12.4f}"
        f"{best['s_f1']:<10.4f}"
        f"{best['ns_f1']:<10.4f}"
    )

print("\nConfusion matrices [[TP, FN], [FP, TN]]:")

for name, results in rules:

    best = max(
        results,
        key=lambda result: result["accuracy"]
    )

    print(f"\n{name}:")
    print(best["confusion"])

print("\nDecision rule comparison completed.")
