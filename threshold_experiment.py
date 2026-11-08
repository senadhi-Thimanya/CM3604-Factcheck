
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

THRESHOLDS = [
    0.30, 0.31, 0.32, 0.33, 0.34,
    0.35, 0.36, 0.37, 0.38, 0.39, 0.40
]


# 2. Load dataset


print("Loading dataset...")

fact_to_passage_dict = read_passages(PASSAGES_PATH)

examples = read_fact_examples(
    LABELS_PATH,
    fact_to_passage_dict
)

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


# 5. Calculate maximum entailment scores ONCE


print("\nCalculating entailment scores...")

max_scores = []
gold_labels = []

for example in tqdm(examples):

    max_entailment_score = 0.0

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

            score = result["entailment"]

            max_entailment_score = max(
                max_entailment_score,
                score
            )

    max_scores.append(max_entailment_score)

    # Match the original evaluation logic:
    # IR is treated as NS
    gold_label = (
        "NS" if example.label == "IR"
        else example.label
    )

    gold_labels.append(gold_label)



# 6. Evaluate different thresholds


print("\nTHRESHOLD EVALUATION RESULTS")
print("=" * 85)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'S Precision':<14}"
    f"{'S Recall':<12}"
    f"{'S F1':<12}"
    f"{'NS F1':<12}"
)

print("-" * 85)

results = []

for threshold in THRESHOLDS:

    confusion = np.zeros((2, 2), dtype=int)

    for score, gold in zip(max_scores, gold_labels):

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

    accuracy = (tp + tn) / len(examples)

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
        "s_precision": s_precision,
        "s_recall": s_recall,
        "s_f1": s_f1,
        "ns_f1": ns_f1,
        "confusion": confusion
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{s_precision:<14.4f}"
        f"{s_recall:<12.4f}"
        f"{s_f1:<12.4f}"
        f"{ns_f1:<12.4f}"
    )



# 7. Identify best threshold


best_result = max(
    results,
    key=lambda result: result["accuracy"]
)

print("\nBEST THRESHOLD BY ACCURACY")
print("=" * 40)

print("Threshold:", best_result["threshold"])
print("Accuracy:", best_result["accuracy"])
print("Confusion matrix:")
print(best_result["confusion"])

# Also identify the best threshold by macro F1
best_macro_f1 = max(
    results,
    key=lambda result: (
        result["s_f1"] + result["ns_f1"]
    ) / 2
)

print("\nBEST THRESHOLD BY MACRO F1")
print("=" * 40)

print("Threshold:", best_macro_f1["threshold"])
print(
    "Macro F1:",
    (best_macro_f1["s_f1"] + best_macro_f1["ns_f1"]) / 2
)

print("\nThreshold experiment completed.")
