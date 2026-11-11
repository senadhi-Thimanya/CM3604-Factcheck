# Part 2 results summary

All numbers are on the 221 dev facts (S = 112, NS + IR = 109). Runs were on CPU (Apple M1 Pro), with IR counted as NS.

## Final submitted configuration (`factcheck.py`, Thisumi's 11 Nov version)

| Setting | Value |
|---|---|
| Model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Premise unit | Single sentences (spaCy `sentencizer`), tags kept (`clean_text=False`) |
| Pruning | Skip a sentence if < 10% of the fact's lower-cased words appear in it (`overlap_method="simple"`, `overlap_cutoff=0.1`) |
| Sentences checked / skipped | 3,284 / 2,070 (38.7% skipped) |
| Decision rule | Max entailment probability over checked sentences |
| Threshold | 0.36 (middle of the flat 0.34-0.38 range) |
| Accuracy | **0.8552 (189/221)** |
| TP / FN / FP / TN | 92 / 20 / 12 / 97 |
| S precision / recall / F1 | 0.8846 / 0.8214 / 0.8519 |
| NS precision / recall / F1 | 0.8291 / 0.8899 / 0.8584 |
| CPU runtime | 170 s (517 s without pruning, about 3x faster) |

`check_submission.sh` also passed in a fresh Python 3.10 env (on the earlier 10 Nov version).

## Baselines

| Method | Accuracy |
|---|---|
| Always S | 0.5068 |
| Word overlap (Part 1, precision, threshold 0.64) | 0.8190 |
| Entailment, initial threshold 0.50, no pruning | 0.8462 |

## Threshold sweep

**No pruning (max entailment):**

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

**Final pruning (simple overlap, cutoff 0.1):**

| 0.30 | 0.32 | 0.33 | 0.34 | 0.35 | **0.36** | 0.37 | 0.38 | 0.39 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|
| 0.8416 | 0.8416 | 0.8507 | 0.8552 | 0.8552 | **0.8552** | 0.8552 | 0.8552 | 0.8462 | 0.8462 |

The flat range is unchanged by the final pruning.

## Decision rules (no pruning)

| Rule | Best threshold | Accuracy | S F1 | NS F1 | TP / FN / FP / TN |
|---|---|---|---|---|---|
| Max entailment probability | 0.34 | 0.8552 | 0.8545 | 0.8559 | 94 / 18 / 14 / 95 |
| Discrete argmax (any sentence) | n/a | 0.8462 | 0.8411 | 0.8509 | 90 / 22 / 12 / 97 |
| Max (entailment - contradiction) | 0.34 | 0.8552 | 0.8532 | 0.8571 | 93 / 19 / 13 / 96 |

Source: `experiments/decision_rules.txt`.

## Pruning

**Final measure (simple word overlap, stopwords kept):** cutoff 0.1 skips 38.7% of sentences, keeps 0.8552, and runs in 170 s.

**Alternative tested: A's overlap measure (stopwords removed, stemmed), threshold 0.34:**

| Cutoff | Sentences checked | Skipped | Model time (s) | Speed-up | Accuracy |
|---|---|---|---|---|---|
| 0.00 | 5,354 | 0% | 490 | 1.00x | 0.8552 |
| 0.05-0.10 | 1,653 | 69.1% | 153 | 3.20x | 0.8462 |
| 0.15 | 1,570 | 70.7% | 145 | 3.37x | 0.8462 |
| 0.20 | 1,484 | 72.3% | 137 | 3.57x | 0.8462 |
| 0.25 | 1,265 | 76.4% | 117 | 4.18x | 0.8371 |
| 0.30 | 926 | 82.7% | 86 | 5.72x | 0.8235 |
| 0.40 | 527 | 90.2% | 49 | 10.01x | 0.8235 |
| 0.60 | 238 | 95.6% | 22 | 22.20x | 0.7919 |

Model time is the summed DeBERTa call time from `pruning_sweep.py`. A real run at cutoff 0.20 took 143 s for 0.8462 (TP 88 / FN 24 / FP 10 / TN 99).

| Pruning setup | Skipped | CPU time | Accuracy |
|---|---|---|---|
| None | 0% | 517 s | 0.8552 |
| **Simple overlap, 0.1 (final)** | **38.7%** | **170 s** | **0.8552** |
| A's overlap, 0.20 | 72.3% | 143 s | 0.8462 |

## Passage cleaning

| Variant | Accuracy |
|---|---|
| `<s>`/`</s>` tags kept (final) | 0.8552 |
| Tags removed, final settings (Thisumi) | 0.8281 (183/221) |
| Tags removed, A's pruning 0.20, threshold 0.34 (Tharuka) | 0.8235 (182/221) |

## Errors of the final model

`experiments/entailment_errors.csv` lists all 32 errors, made by `dump_errors.py` with the final settings. Thisumi's `error_analysis.py` also draws a 10 FP + 10 FN sample (`error_sample.csv`) for labelling.

| Type | Count | Gold labels |
|---|---|---|
| False positives (gold NS, predicted S) | 12 | 11 NS, 1 IR |
| False negatives (gold S, predicted NS) | 20 | 20 S |

For each error the CSV gives the fact, the gold label, the best-scoring sentence with its entailment, neutral and contradiction probabilities, its word overlap with the fact, and an empty `your_category` column.
