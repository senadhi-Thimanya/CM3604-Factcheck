# Experiment log

Record every experiment so the report can justify each design decision. One row per run.

## Part 1: word overlap

Scripts: `experiments/part1_ablation.py`, `experiments/part1_extra_experiments.py` (and the metrics harness on the `metrics-and-evaluation-harness` branch).

## Part 1: word overlap
| Date | Who | Tokenization | Stopwords | Stemming | Metric | Threshold | Accuracy | Notes |
|---|---|---|---|---|---|---|---|---|
| Oct-Nov 2026 | A | Letters + digits (regex) | Hard-coded full list | Porter | Precision, best passage | 0.64 | 0.8190 (181/221) | **Final.** Plateau 0.61-0.66. TP 94, FN 18, FP 22, TN 87 |
| | A | same | none | Porter | Precision | 0.75 | 0.792 | Stopwords off: -2.7 points |
| | A | same | full | none | Precision | 0.64 | 0.787 | Stemming off: -3.2 points |
| | A | same, lowercase off | full | Porter | Precision | 0.64 | 0.8190 | No change: the Porter stemmer lowercases words itself |
| | A | same | full | none, lowercase on / off | Precision | n/a | 0.787 / 0.778 | Lowercasing helps by about 1 point once the stemmer is off |
| | A | same | full | Porter | Precision, passages pooled | 0.71 | 0.796 | Pooling is 2.3 points worse than the best passage |
| | A | same | full | Porter | Jaccard | 0.03 | 0.697 | Passage length dominates the union |
| | A | same | full | Porter | Dice | 0.06 | 0.692 | Same reason as Jaccard |
| | A | Letters only (digits dropped) | full | Porter | Precision | n/a | 0.792 | Digits matter: -2.7 points |
| | A | Whitespace + strip punctuation | full | Porter | Precision | n/a | 0.810 | |
| | A | Letters, digits, apostrophe words | full | Porter | Precision | n/a | 0.824 | One fact better: within noise |
| | A | Letters + digits | small (about 25 words) | Porter | Precision | n/a | 0.824 | One fact better: within noise |
| | A | same | full | Snowball / WordNet lemmatizer | Precision | n/a | 0.819 / 0.801 | Stemming beats lemmatization |
| | A | same | full | Porter | IDF-weighted precision | n/a | 0.828 | Two facts better: within noise, not adopted |
| | A | same | full | Porter | Best single sentence / 2-sentence window | n/a | 0.774 / 0.810 | Context is needed |
| | A | same | full | Porter | Bigram / unigram+bigram average | n/a | 0.715 / 0.774 | Too strict for short facts |
| | B | same | full | Porter | ROUGE-L precision | 0.64 | 0.8054 (178/221) | Word order does not help |
| | B | ROUGE tokenizer on raw text | ROUGE default | ROUGE stemmer | ROUGE-1 precision | 0.77 | 0.7919 (175/221) | Our preprocessing helps a standard metric too |
| | B | same | full | Porter | TF-IDF cosine | 0.10 | 0.6606 (146/221) | Length normalisation hurts |
| | A | grouped cross-validation (5 folds by person, 30 repeats) | | | Precision | chosen per fold | 0.808 +- 0.015 | 95% bootstrap interval 0.769-0.869 |

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
| 2026-11-10 | Tharuka | Single sentences, `<s>`/`</s>` tags removed before splitting | 0.20 | Max entailment prob | 0.34 | 0.8235 (182/221) | CPU, 153 s | Rejected: below 83% and 2.3 points worse than keeping tags. TP 85, FN 27, FP 12, TN 97. Threshold not re-tuned for this setting |
| 2026-11-11 | Tharuka | Single sentences (spaCy sentencizer), tags kept | 0.1, simple overlap (lower-cased words, stopwords kept) | Max entailment prob | 0.36 | 0.8552 (189/221) | CPU, 170 s | Verified Thisumi's final version. 2,070 of 5,354 sentences skipped (38.7%). TP 92, FN 20, FP 12, TN 97. Thresholds 0.34-0.38 all give 0.8552 under this pruning |

## Limitations to mention in the report
- Threshold tuned on the same dev set it is evaluated on (the brief allows this; there is no separate test set).
