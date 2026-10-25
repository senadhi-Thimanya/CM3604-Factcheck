# CM3604 Deep Learning: Fact-Checking Coursework (Group)

Fact-checks LLM-generated biography facts against retrieved Wikipedia passages (FActScore setup).
Predicts **S** (supported) or **NS** (not supported); the gold label **IR** is treated as **NS**.

**Deadline:** Wed 11 Nov 2026, 23:59 (internal target: submit Tue 10 Nov).
**Submit separately:** (1) PDF report, (2) `factcheck.py`.
**Do not modify** `factchecking_main.py`.

| Part | What | Target |
|---|---|---|
| 1 | Word overlap (`--mode word_overlap`) | accuracy >= 75% |
| 2 | DeBERTa-v3 entailment (`--mode entailment`) | accuracy >= 83% |
| 3 | Error analysis PDF (10 FP + 10 FN, 2-4 categories) | see brief |

## Data facts (checked)
- `data/dev_labeled_ChatGPT.jsonl`: 31 lines -> **221 labelled facts** (S = 112, NS = 93, IR = 16).
- Always predicting S gives about 50.7% (112/221), the number every method must beat.
- `data/passages_bm25_ChatGPT_humfacts.jsonl`: 5,426 facts with passages (26 MB). Only the 221 dev facts are scored.
- Passages contain `<s>` and `</s>` tags and are not sentence-aligned.

## Setup
Python **3.10** is required (the brief says so). From the repo root:

```bash
./scripts/setup_env.sh          # creates .venv, installs pinned requirements, downloads NLTK/spaCy data
source .venv/bin/activate
python factchecking_main.py --mode always_entail      # sanity check, should be about 0.507
python factchecking_main.py --mode entailment --cuda  # only if you have a CUDA GPU
```

The first entailment run downloads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` from Hugging Face (needs internet).

## Repo layout
```
factcheck.py                 SUBMISSION FILE (the only code file we submit)
factchecking_main.py         provided driver (DO NOT EDIT)
requirements.txt             pinned packages
data/                        provided datasets
experiments/                 ablation tables, threshold sweeps, notes (cache/ is git-ignored)
report/                      PDF report sources and final PDF
docs/                        git workflow, interface contract, experiment log template
scripts/                     setup_env.sh, check_submission.sh
```

## Rules
1. **Do not edit `factchecking_main.py`.**
2. `factcheck.py` must work with **only** the pinned packages. Do not rely on runtime downloads other than the HF model
   (the NLTK/spaCy data you downloaded locally may not exist on the marker's machine). If you need a stopword list, hard-code it.
3. Both commands must run end to end without errors or OOM, on CPU as well as GPU:
   `python factchecking_main.py --mode word_overlap` and `python factchecking_main.py --mode entailment`.
4. **GenAI policy: category 3, Editing only.** GenAI may be used only to improve wording or presentation of our own work,
   not to write code, ideas or analysis. Include the acknowledgement statement in the report. Keep records of how the work was produced.
5. Everyone must be able to explain every part of the code and report.

See `docs/GIT_WORKFLOW.md` for branching and `docs/INTERFACES.md` for who owns what.
Before submitting, run `./scripts/check_submission.sh` in a fresh environment.
