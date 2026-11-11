# Git workflow

## Branches
| Member | Role | Branch |
|---|---|---|
| A | Word overlap lead (Part 1) | `a/word-overlap` |
| B | Entailment model lead (Part 2, model side) | `b/entailment-model` |
| C | Entailment pipeline and tuning (Part 2, data side) | `c/pipeline` |
| D | Analysis and report lead (Part 3) | `d/report` |

`main` always runs. Nobody commits to `main` directly.

## Day to day
```bash
git checkout main && git pull
git checkout -b a/word-overlap          # once
# ... work, then:
git add -p && git commit -m "word_overlap: add Jaccard scorer"
git pull --rebase origin main
git push -u origin a/word-overlap       # open a pull request into main
```
- Every PR needs one review from a teammate (see the review rotation: A reviews C, B reviews A, C reviews B, D reviews all).
- Small PRs, merged often. Write commit messages that say what changed; the history is our evidence of who did what.
- Never commit `.venv/`, model caches, or large generated files.

## Conflict avoidance in `factcheck.py`
Only one file is submitted, so we share it. Agreed ownership by section:
- A owns `WordRecallThresholdFactChecker` and the overlap helpers.
- B owns `EntailmentModel`.
- C owns `EntailmentFactChecker` and the passage cleaning/sentence splitting helpers.
- Do not edit another member's section without telling them first.

## Freeze
`factcheck.py` is frozen on **Sun 8 Nov**. After that only bug fixes found in clean-environment runs.
