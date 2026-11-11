
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from factcheck import EntailmentModel, EntailmentFactChecker, overlap_score
from factchecking_main import read_passages, read_fact_examples




MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

# Pruning cutoffs to try (0.0 = pruning off)
CUTOFFS = [round(c, 2) for c in np.arange(0.0, 0.65, 0.05)]

# Entailment thresholds for re-tuning at each cutoff
THRESHOLDS = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]

# Optional: python pruning_sweep.py 10  -> quick test on the first 10 facts
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

# Reuse the submitted checker's sentence splitter and threshold so the
# sweep matches factcheck.py exactly
checker = EntailmentFactChecker(ent_model)



# 4. Score every sentence ONCE: word overlap, entailment and model time


print("\nScoring every sentence...")

# One list per fact of (overlap, entailment, seconds) for each sentence
fact_sentences = []
gold_labels = []

for example in tqdm(examples):

    sentence_scores = []

    for passage in example.passages:

        passage_text = passage.get("text", "")

        if not passage_text or not passage_text.strip():
            continue

        doc = checker.nlp(passage_text)

        for sentence in doc.sents:

            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            overlap = overlap_score(example.fact, sentence_text)

            start = time.perf_counter()

            result = ent_model.check_entailment(
                premise=sentence_text,
                hypothesis=example.fact
            )

            seconds = time.perf_counter() - start

            sentence_scores.append(
                (overlap, result["entailment"], seconds)
            )

    fact_sentences.append(sentence_scores)

    # IR is treated as NS
    gold_labels.append(
        "NS" if example.label == "IR"
        else example.label
    )



# 5. Accuracy of a list of fact scores at one threshold


def accuracy_at(scores, threshold):

    correct = 0

    for score, gold in zip(scores, gold_labels):

        prediction = "S" if score >= threshold else "NS"

        if prediction == gold:
            correct += 1

    return correct / len(scores)



# 6. Simulate each pruning cutoff from the saved scores


total_sentences = sum(len(s) for s in fact_sentences)
total_seconds = sum(sec for s in fact_sentences for _, _, sec in s)

print("\nPRUNING SWEEP RESULTS")
print("=" * 95)

print(
    f"{'Cutoff':<9}"
    f"{'Checked':<10}"
    f"{'Skipped %':<12}"
    f"{'Model time (s)':<16}"
    f"{'Speed-up':<10}"
    f"{'Acc @ ' + str(checker.entailment_threshold):<12}"
    f"{'Best thr':<10}"
    f"{'Best acc':<10}"
)

print("-" * 95)

for cutoff in CUTOFFS:

    fact_scores = []
    checked = 0
    model_seconds = 0.0

    for sentence_scores in fact_sentences:

        # Pruned sentences never reach the model, so they cannot
        # raise the fact's max entailment score
        max_entailment_score = 0.0

        for overlap, entailment, seconds in sentence_scores:

            if overlap < cutoff:
                continue

            checked += 1
            model_seconds += seconds

            max_entailment_score = max(
                max_entailment_score,
                entailment
            )

        fact_scores.append(max_entailment_score)

    skipped_pct = 100 * (total_sentences - checked) / total_sentences
    speed_up = total_seconds / model_seconds if model_seconds else float("inf")

    current_acc = accuracy_at(fact_scores, checker.entailment_threshold)

    best_threshold = max(
        THRESHOLDS,
        key=lambda t: accuracy_at(fact_scores, t)
    )

    print(
        f"{cutoff:<9.2f}"
        f"{checked:<10}"
        f"{skipped_pct:<12.1f}"
        f"{model_seconds:<16.1f}"
        f"{speed_up:<10.2f}"
        f"{current_acc:<12.4f}"
        f"{best_threshold:<10.2f}"
        f"{accuracy_at(fact_scores, best_threshold):<10.4f}"
    )

print(
    "\nModel time only counts DeBERTa calls (the overlap check is tiny)."
    "\nConfirm the chosen cutoff with a real timed run of factchecking_main.py."
)

print("\nPruning sweep completed.")
