# Interface contract (agree before coding)

Fill in the exact signatures at the first team meeting (Sun 25 Oct), then treat them as fixed.
Everyone can then code against the agreed signature without waiting for another member.

| Owner | Function / class | Signature (proposed, edit if the group agrees) | Used by |
|---|---|---|---|
| A | overlap score | `overlap_score(fact: str, text: str) -> float` in range 0..1 | A (word overlap), C (pruning) |
| B | entailment probabilities | `EntailmentModel.check_entailment(premise, hypothesis) -> (p_entail, p_neutral, p_contra)` | C |
| C | passage cleaning / splitting | `split_passage(text: str) -> List[str]` | C, B (timing tests) |
| C | decision rule | `EntailmentFactChecker.predict(fact, passages) -> "S" or "NS"` | main driver |

## Working without waiting on others
- **Stubs first.** A pushes a minimal `overlap_score` by Mon 26 Oct and refines it later without changing the signature.
  B pushes a placeholder `check_entailment` (fixed probabilities) so C can build the full pipeline straight away.
- **Cache entailment scores** to `experiments/cache/` (git-ignored) so threshold tuning does not re-run the model.
- **D's tooling** (script that lists false positives/negatives from any prediction file) is built on Part 1 predictions first,
  then re-run on the final entailment predictions.
