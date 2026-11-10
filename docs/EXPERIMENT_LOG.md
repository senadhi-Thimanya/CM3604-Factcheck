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

## Limitations to mention in the report
- Threshold tuned on the same dev set it is evaluated on (the brief allows this; there is no separate test set).
