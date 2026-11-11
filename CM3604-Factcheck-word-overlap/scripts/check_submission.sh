#!/usr/bin/env bash
# Run BEFORE submitting. Run from the repo root, inside a fresh venv.
set -e
echo "== Python version =="; python --version
echo "== word_overlap =="; python factchecking_main.py --mode word_overlap | tail -n 8
echo "== entailment (CPU) =="; python factchecking_main.py --mode entailment | tail -n 8
echo "== factchecking_main.py unchanged? =="
git diff --quiet HEAD~0 -- factchecking_main.py && echo "OK (no local changes)" || echo "WARNING: factchecking_main.py modified"
echo "Done. Targets: word_overlap >= 0.75, entailment >= 0.83"
