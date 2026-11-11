# Interfaces in the final `factcheck.py`

| Function / class | Signature | Used by |
|---|---|---|
| `preprocess` | `preprocess(text, lowercase=True, remove_stopwords=True, stem=True) -> List[str]` | Part 1 scoring, `overlap_score` |
| `overlap_score` | `overlap_score(fact, text, metric="precision", **prep_kwargs) -> float` in [0, 1] | Part 1, optional pruning method `"content"` |
| `WordRecallThresholdFactChecker` | `WordRecallThresholdFactChecker(threshold=0.64, metric="precision", lowercase=True, remove_stopwords=True, stem=True, pool_passages=False)`; `.score(fact, passages)`, `.predict(fact, passages) -> "S" or "NS"` | `--mode word_overlap` |
| `EntailmentModel` | `check_entailment(premise, hypothesis)` returns the entailment, neutral and contradiction probabilities | `EntailmentFactChecker` |
| `EntailmentFactChecker` | `EntailmentFactChecker(ent_model, entailment_threshold=0.36, overlap_cutoff=0.1, overlap_method="simple", clean_text=False)`; `.max_entailment_score(fact, passages, return_sentence=False)`, `.predict(fact, passages) -> "S" or "NS"` | `--mode entailment` |

Notes
- The pruning in `EntailmentFactChecker` defaults to the `"simple"` overlap (lower-cased words, stopwords kept). Part 1's `overlap_score`
  is available as `overlap_method="content"`. It was tested and rejected as the default because it skipped more sentences and lost accuracy
  (see `docs/D_FINDINGS.md`).
- `clean_text=False` keeps the `<s>` and `</s>` tags. Removing them lowered accuracy in every setup tested.
- `max_entailment_score(..., return_sentence=True)` also returns the best-scoring sentence, used by the error-analysis scripts.