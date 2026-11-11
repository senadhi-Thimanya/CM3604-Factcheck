# Git workflow

## Branches (as used)
| Branch | Content |
|---|---|
| `word-overlap` | Part 1: preprocessing, overlap scoring, ablation scripts |
| `metrics-and-evaluation-harness` | Part 1: TF-IDF / ROUGE comparison and the evaluation harness |
| `text-entailment` | Part 2: entailment model inference and decision rule |
| `pruning` | Part 2: pruning, final `EntailmentFactChecker`, experiments |

`main` must always run both modes. The final `factcheck.py` is the version on `pruning` (it equals the one on `text-entailment`).

## Rules
- Work on a branch, merge into `main` through a pull request that a teammate has reviewed.
- Never commit `venv/` or `.venv/`, model caches, or large generated files. Check `git status` before every commit.
- Never merge a version of `factcheck.py` that still contains the duplicated Part 1 block (it should be about 347 lines, not 447).
- After every merge into `main`, run `scripts/check_submission.ps1` (or `.sh`) and confirm 0.8190 and 0.8552.

## Fixing a committed virtual environment
```powershell
git rm -r --cached venv
git commit -m "Stop tracking venv"
```
(`venv/` is already in `.gitignore`; the ignore file does not remove files that were committed earlier.)

## Freeze
`factcheck.py` is frozen once the final fresh-environment check has passed. After that only bug fixes found in clean-environment runs.