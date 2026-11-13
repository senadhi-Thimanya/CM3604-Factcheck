"""
threshold_experiment.py

Runs the final entailment checker once over the development set, then evaluates a sweep of
decision thresholds (accuracy, F1 for S and NS, confusion matrix counts).

Run:  python threshold_experiment.py   (from anywhere inside the project)
Output: experiments/part2/results/threshold_results.csv  (read by make_charts.py)
"""

import csv
import os
import sys
from pathlib import Path

# find the project root (the folder that holds factcheck.py), wherever this script is saved
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "factcheck.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                    # data/ paths below are relative to the project root
RESULTS = ROOT / "experiments" / "part2" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

# Coarse sweep (0.05 to 0.95) plus a fine sweep (0.30 to 0.40)
coarse = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
fine = [round(t, 2) for t in np.arange(0.30, 0.401, 0.01)]
THRESHOLDS = sorted(set(coarse + fine))


# Load dataset
print("Loading dataset...")
fact_to_passage_dict = read_passages(PASSAGES_PATH)
examples = read_fact_examples(LABELS_PATH, fact_to_passage_dict)
print(f"Loaded {len(examples)} examples.")


# Load model
print("\nLoading DeBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ent_model = EntailmentModel(model, tokenizer, cuda=(device.type == "cuda"))
print(f"Using device: {device}")

# Use the SAME checker (and pruning and noise removal) as the final factcheck.py
checker = EntailmentFactChecker(ent_model)


# Calculate maximum entailment scores ONCE
print("\nCalculating entailment scores...")

max_scores = []
gold_labels = []

for example in tqdm(examples):
    score = checker.max_entailment_score(example.fact, example.passages)
    max_scores.append(score)

    # IR is treated as NS
    gold_labels.append("NS" if example.label == "IR" else example.label)


# Evaluate each threshold
print("\nTHRESHOLD EVALUATION RESULTS")
print("=" * 85)
print(
    f"{'Threshold':<12}{'Accuracy':<12}{'S Precision':<14}"
    f"{'S Recall':<12}{'S F1':<12}{'NS F1':<12}"
)
print("-" * 85)

results = []

for threshold in THRESHOLDS:
    tp = fn = fp = tn = 0

    for score, gold in zip(max_scores, gold_labels):
        prediction = "S" if score >= threshold else "NS"

        if gold == "S" and prediction == "S":
            tp += 1
        elif gold == "S" and prediction == "NS":
            fn += 1
        elif gold == "NS" and prediction == "S":
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    s_precision = tp / (tp + fp) if tp + fp else 0
    s_recall = tp / (tp + fn) if tp + fn else 0
    s_f1 = (2 * s_precision * s_recall / (s_precision + s_recall)
            if s_precision + s_recall else 0)

    ns_precision = tn / (tn + fn) if tn + fn else 0
    ns_recall = tn / (tn + fp) if tn + fp else 0
    ns_f1 = (2 * ns_precision * ns_recall / (ns_precision + ns_recall)
             if ns_precision + ns_recall else 0)

    results.append({
        "threshold": threshold, "accuracy": accuracy,
        "s_f1": s_f1, "ns_f1": ns_f1,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    })

    print(
        f"{threshold:<12.2f}{accuracy:<12.4f}{s_precision:<14.4f}"
        f"{s_recall:<12.4f}{s_f1:<12.4f}{ns_f1:<12.4f}"
    )


# Best thresholds
best = max(results, key=lambda r: r["accuracy"])
print("\nBEST THRESHOLD BY ACCURACY")
print("=" * 40)
print("Threshold:", best["threshold"])
print("Accuracy:", round(best["accuracy"], 4))
print(f"TP={best['tp']}  FN={best['fn']}  FP={best['fp']}  TN={best['tn']}")

best_f1 = max(results, key=lambda r: (r["s_f1"] + r["ns_f1"]) / 2)
print("\nBEST THRESHOLD BY MACRO F1")
print("=" * 40)
print("Threshold:", best_f1["threshold"])
print("Macro F1:", round((best_f1["s_f1"] + best_f1["ns_f1"]) / 2, 4))

out_file = RESULTS / "threshold_results.csv"
with open(out_file, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "accuracy", "s_f1", "ns_f1", "tp", "fn", "fp", "tn"])
    for r in results:
        w.writerow([r["threshold"], r["accuracy"], r["s_f1"], r["ns_f1"],
                    r["tp"], r["fn"], r["fp"], r["tn"]])
print(f"\nSaved {out_file}")