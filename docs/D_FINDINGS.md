# Part 2 findings: pipeline and decision rules (Member D, Tharuka)

Working notes for the report. All numbers are on the 221 dev facts (S = 112, NS + IR = 109), with IR counted as NS, run on CPU (Apple M1 Pro). Full tables are in `experiments/part2_results_summary.md`. Branch: `pruning` (includes Thisumi's 11 Nov `text-entailment` work).

## 1. What was done

| Date | Who | Step | Result | Evidence |
|---|---|---|---|---|
| 8 Nov | Thisumi | Baseline entailment checker, threshold 0.50 → 0.34 | 0.8462 → 0.8552 | `threshold_experiment.py` |
| 9 Nov | Tharuka | Set up the env and ran the `always_entail` baseline | 0.5068 (112/221) | - |
| 9 Nov | Tharuka | Reproduced the 0.34 result | 0.8552 (189/221) | `experiments/entailment_t034.txt` |
| 9 Nov | Tharuka | Full threshold sweep, 0.01-0.99 | Best 0.34-0.38 (flat) | `experiments/threshold_sweep_full.txt` |
| 9 Nov | Tharuka | Compared three decision rules | Max entailment 0.8552, entail - contradict 0.8552, argmax 0.8462 | `experiments/decision_rules.txt` |
| 10 Nov | Tharuka | Merged A's overlap scorer into the entailment code | Both modes unchanged | - |
| 10 Nov | Tharuka | Pruning sweep with A's overlap, cutoffs 0.00-0.60 | Any cutoff > 0 costs 2 facts; 0.20 → 3.57x | `experiments/pruning_sweep.txt` |
| 10 Nov | Tharuka | Real run, A's overlap at 0.20 | 0.8462, 143 s vs 517 s | `docs/EXPERIMENT_LOG.md` |
| 10 Nov | Tharuka | Removed `<s>` tags before splitting | 0.8235, rejected | `docs/EXPERIMENT_LOG.md` |
| 10 Nov | Tharuka | `check_submission.sh` in a fresh Python 3.10 env | Passed | - |
| 11 Nov | Thisumi | Simple-overlap pruning (cutoff 0.1), threshold 0.36, optional tag cleaning, label mapping built once, `error_analysis.py` | 0.8552, 170 s | `text-entailment` branch |
| 11 Nov | Tharuka | Verified Thisumi's version; threshold check 0.30-0.40 under her pruning | 0.8552 for 0.34-0.38; word overlap 0.8190 | `experiments/part2_results_summary.md` |
| 11 Nov | Tharuka | Error list for the final model | 12 FP, 20 FN | `experiments/entailment_errors.csv` |

## 2. Final design (submitted `factcheck.py`)

| Step | Choice | Alternatives tested |
|---|---|---|
| Passage cleaning | Tags kept | Tags removed: 0.8281 (final settings), 0.8235 (A's pruning) |
| Sentence splitting | spaCy `sentencizer`, single sentences | - |
| Pruning | Skip a sentence if < 10% of the fact's lower-cased words appear in it | A's stemmed, stopword-free overlap at 0.00-0.60 |
| Score per fact | Max entailment probability over checked sentences | Discrete argmax; max (entail - contradict) |
| Threshold | 0.36 | 0.01-0.99 |
| **Result** | **0.8552 (189/221), TP 92 / FN 20 / FP 12 / TN 97, 170 s on CPU** | |

## 3. Key results

**Decision rules (no pruning)**

| Rule | Threshold | Accuracy | TP / FN / FP / TN |
|---|---|---|---|
| Max entailment probability | 0.34 | 0.8552 | 94 / 18 / 14 / 95 |
| Discrete argmax | n/a | 0.8462 | 90 / 22 / 12 / 97 |
| Max (entailment - contradiction) | 0.34 | 0.8552 | 93 / 19 / 13 / 96 |

**Pruning setups (threshold 0.34-0.36)**

| Setup | Skipped | CPU time | Accuracy |
|---|---|---|---|
| None | 0% | 517 s | 0.8552 |
| **Simple overlap, 0.1 (final)** | **38.7%** | **170 s** | **0.8552** |
| A's overlap, 0.05 | 69.1% | (3.20x model time) | 0.8462 |
| A's overlap, 0.20 | 72.3% | 143 s | 0.8462 |
| A's overlap, 0.30 | 82.7% | (5.72x model time) | 0.8235 |

**Observations from the data (facts to build the write-up on):**
- Accuracy is identical for every threshold from 0.34 to 0.38, both without pruning and with the final pruning. 0.36 is the middle of that range.
- With A's overlap, even the smallest cutoff (0.05) skips 69.1% of sentences and loses 2 facts. The simple overlap (stopwords kept, no stemming) at 0.1 skips 38.7% and loses none.
- The final setup is about 3x faster than no pruning with no accuracy loss. A's overlap at 0.20 is faster (3.6x) but 0.9 points less accurate.
- Removing the `<s>` tags lowered accuracy in both setups tested.
- Argmax gives the same predictions as threshold 0.50 (TP 90, FP 12).

## 4. Completion checklist

### Brief (must have)
- [x] `word_overlap` mode runs and reaches ≥ 75% (0.8190)
- [x] `entailment` mode runs and reaches ≥ 83% (0.8552)
- [x] Both run on CPU without errors or out-of-memory problems
- [x] `factchecking_main.py` unchanged
- [x] Python 3.10 + pinned packages only
- [x] `check_submission.sh` passes on the final `factcheck.py` in a fresh Python 3.10 env (11 Nov: 0.8190 / 0.8552)
- [ ] Part 3: 10 FP + 10 FN examined and categorised into 2-4 categories (error lists ready, labelling not done)
- [ ] Part 3: category definitions, aggregate statistics, 3 detailed examples
- [ ] PDF report written
- [ ] PDF and `factcheck.py` submitted separately (deadline 11 Nov, 23:59)

### Rubric, Part 2 (top band)
- [x] Decision strategy implemented, with alternatives evaluated by experiment
- [x] Sentence splitting
- [x] Noise removal tested (tag removal; rejected on accuracy)
- [x] Pruning with measured efficiency gain (about 3x, no accuracy loss)
- [x] Threshold selection analysed
- [ ] Justification written up in the report

### D role tasks
- [x] Passage cleaning and sentence splitting
- [x] Pruning with A's overlap scorer (tested; simple overlap chosen)
- [x] `EntailmentFactChecker` (Thisumi, verified)
- [x] Decision rule comparison and threshold tuning
- [x] Review of B's code (notes in `review_B_notes.md`, not posted yet)
- [ ] Label all 20 errors independently, then agree categories with the team
- [ ] One detailed Part 3 example
- [ ] Part 2 pipeline section and Part 3 reflection
- [ ] Final submission check and upload

### Repo / team
- [x] Branches `word-overlap`, `metrics-and-evaluation-harness`, `text-entailment` and `pruning` are merged into `main`
- [ ] `main`'s `factcheck.py` still has the duplicated Part 1 block (447 lines). Replace it with the clean 347-line version: `git checkout origin/pruning -- factcheck.py`
- [ ] Remove the committed `venv/` (1,432 files) and the duplicate `CM3604-Factcheck-word-overlap/` folder from `main` (the problems flagged earlier on Sanuli's branch)
- [ ] Re-run `scripts/check_submission.ps1` on `main` after the fix: it must print 0.8190 and 0.8552
- [ ] Report: Part 1 section drafted (combined from `word-overlap` and `metrics-and-evaluation-harness`); Parts 2 and 3 and the GenAI acknowledgement still to be added
