# Experiment log

Record every experiment so the report can justify each design decision. One row per run.

## Part 1: word overlap
| Date | Who | Tokenization | Stopwords | Stemming | Metric | Threshold | Accuracy | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Part 2: entailment
| Date | Who | Sentence unit | Pruning cutoff | Decision rule | Threshold | Accuracy | Runtime (CPU/GPU) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | Tharuka | Single sentences (spaCy sentencizer) | None | Max entailment prob | 0.34 | 0.8552 (189/221) | CPU (M1 Pro) | Reproduced Thisumi's result with `factchecking_main.py --mode entailment`. Output: `experiments/entailment_t034.txt` |
| 2026-11-09 | Tharuka | Single sentences (spaCy sentencizer) | None | Max entailment prob | 0.01-0.99 sweep, best 0.34 | 0.8552 | CPU (M1 Pro) | Same accuracy for 0.34-0.38. Output: `experiments/threshold_sweep_full.txt` |
| 2026-11-09 | Tharuka | Single sentences (spaCy sentencizer) | None | Max entailment prob | 0.34 | 0.8552 | CPU, 7m53s (2.14 s/fact) for all 3 rules | TP 94, FN 18, FP 14, TN 95. Output: `experiments/decision_rules.txt` |
| 2026-11-09 | Tharuka | Single sentences (spaCy sentencizer) | None | Discrete argmax (any sentence) | n/a | 0.8462 | same run | TP 90, FN 22, FP 12, TN 97 |
| 2026-11-09 | Tharuka | Single sentences (spaCy sentencizer) | None | Max (entail - contradict) | 0.34 (swept -0.99 to 0.99) | 0.8552 | same run | TP 93, FN 19, FP 13, TN 96 |
| 2026-11-10 | Tharuka | Single sentences (spaCy sentencizer) | 0.00-0.60 sweep (sentence-level overlap precision) | Max entailment prob | 0.34 | 0.8552 at 0.00; 0.8462 for 0.05-0.20; 0.8371 at 0.25; 0.8235 at 0.30 | Estimated from model time per call: 490 s at 0.00, 137 s at 0.20 (3.57x) | 0.20 skips 72.3% of sentences (1,484 of 5,354 checked). Output: `experiments/pruning_sweep.txt` |
| 2026-11-10 | Tharuka | Single sentences (spaCy sentencizer) | 0.20 (chosen) | Max entailment prob | 0.34 | 0.8462 (187/221) | CPU, 143 s vs 517 s unpruned (3.6x) | Real `factchecking_main.py` run. TP 88, FN 24, FP 10, TN 99. Fewer FP, more FN than unpruned |

## Limitations to mention in the report
- Threshold tuned on the same dev set it is evaluated on (the brief allows this; there is no separate test set).
