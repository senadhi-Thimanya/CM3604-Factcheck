# Part 2 results summary

All numbers are on the 221 dev facts (S = 112, NS + IR = 109). Runs were on CPU (Apple M1 Pro), with IR counted as NS.

## Final submitted configuration (`factcheck.py`)

| Setting | Value |
|---|---|
| Model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Premise unit | Single sentences (spaCy `sentencizer`), tags kept |
| Pruning | Skip a sentence if A's precision overlap with the fact is < 0.20 |
| Decision rule | Max entailment probability over checked sentences |
| Threshold | 0.34 |
| Accuracy | **0.8462 (187/221)** |
| TP / FN / FP / TN | 88 / 24 / 10 / 99 |
| S precision / recall / F1 | 0.8980 / 0.7857 / 0.8381 |
| NS precision / recall / F1 | 0.8049 / 0.9083 / 0.8534 |
| CPU runtime | 143 s (517 s without pruning) |

## Baselines

| Method | Accuracy |
|---|---|
| Always S | 0.5068 |
| Word overlap (Part 1, precision, threshold 0.64) | 0.8190 |
| Entailment, initial threshold 0.50, no pruning | 0.8462 |

## Threshold sweep (max entailment, no pruning)

| Threshold | Accuracy |
|---|---|
| 0.10 | 0.7964 |
| 0.20 | 0.8281 |
| 0.30 | 0.8416 |
| **0.34-0.38** | **0.8552** (best, flat range) |
| 0.40 | 0.8462 |
| 0.50 | 0.8462 |
| 0.70 | 0.8371 |
| 0.90 | 0.7919 |

Full table: `experiments/threshold_sweep_full.txt`.

## Decision rules (no pruning)

| Rule | Best threshold | Accuracy | S F1 | NS F1 | TP / FN / FP / TN |
|---|---|---|---|---|---|
| Max entailment probability | 0.34 | 0.8552 | 0.8545 | 0.8559 | 94 / 18 / 14 / 95 |
| Discrete argmax (any sentence) | n/a | 0.8462 | 0.8411 | 0.8509 | 90 / 22 / 12 / 97 |
| Max (entailment - contradiction) | 0.34 | 0.8552 | 0.8532 | 0.8571 | 93 / 19 / 13 / 96 |

Source: `experiments/decision_rules.txt`.

## Pruning (max entailment, threshold 0.34)

| Cutoff | Sentences checked | Skipped | Model time (s) | Speed-up | Accuracy |
|---|---|---|---|---|---|
| 0.00 | 5,354 | 0% | 490 | 1.00x | 0.8552 |
| 0.05-0.10 | 1,653 | 69.1% | 153 | 3.20x | 0.8462 |
| 0.15 | 1,570 | 70.7% | 145 | 3.37x | 0.8462 |
| **0.20** | **1,484** | **72.3%** | **137** | **3.57x** | **0.8462** |
| 0.25 | 1,265 | 76.4% | 117 | 4.18x | 0.8371 |
| 0.30 | 926 | 82.7% | 86 | 5.72x | 0.8235 |
| 0.40 | 527 | 90.2% | 49 | 10.01x | 0.8235 |
| 0.60 | 238 | 95.6% | 22 | 22.20x | 0.7919 |

Model time is the summed DeBERTa call time from `pruning_sweep.py`. Real end-to-end runs: 517 s at 0.00 and 143 s at 0.20 (3.6x).

## Passage cleaning (pruning 0.20, threshold 0.34)

| Variant | Accuracy | TP / FN / FP / TN |
|---|---|---|
| `<s>`/`</s>` tags kept (submitted) | 0.8462 | 88 / 24 / 10 / 99 |
| Tags removed before splitting | 0.8235 | 85 / 27 / 12 / 97 |

## Errors of the final model

`experiments/entailment_errors.csv` lists all 34 errors, made by `dump_errors.py` with the submitted settings.

| Type | Count | Gold labels |
|---|---|---|
| False positives (gold NS, predicted S) | 10 | 9 NS, 1 IR |
| False negatives (gold S, predicted NS) | 24 | 24 S |

For each error the CSV gives the fact, the gold label, the best-scoring sentence with its entailment, neutral and contradiction probabilities, its word overlap with the fact, and an empty `your_category` column for labelling. One false negative had every sentence pruned, so it has no best sentence.
