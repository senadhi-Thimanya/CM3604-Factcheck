# CM3604 Deep Learning: Fact-Checking Coursework (Group)

Fact-checks LLM-generated biography facts against retrieved Wikipedia passages (FActScore setup).
Predicts **S** (supported) or **NS** (not supported); the gold label **IR** is treated as **NS**.

**Submit separately:** (1) the PDF report, (2) `factcheck.py`. Do not modify `factchecking_main.py`.

## Results (221 development facts)

| Part | System | Command | Accuracy | Target |
|---|---|---|---|---|
| 1 | Word overlap (precision, stopwords + Porter stemming, best passage, threshold 0.64) | `python factchecking_main.py --mode word_overlap` | 0.8190 (181/221) | >= 0.75 |
| 2 | DeBERTa-v3 entailment (best sentence, threshold 0.36, overlap pruning cutoff 0.1) | `python factchecking_main.py --mode entailment` | 0.8552 (189/221) | >= 0.83 |
| 3 | Error analysis of the Part 2 model | see report | n/a | 10 FP + 10 FN, 2-4 categories |

Always predicting S scores 0.507 (112/221), the baseline. Entailment takes about 170 s on a CPU with pruning.
Details: `docs/D_FINDINGS.md`, `docs/EXPERIMENT_LOG.md`, `experiments/`.

## Setup
Python **3.10** is required.

Windows (PowerShell), from the repo root:
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python factchecking_main.py --mode always_entail    # sanity check, about 0.507
```
Mac/Linux: `./scripts/setup_env.sh`, then `source .venv/bin/activate`.

`factcheck.py` needs no downloads other than the Hugging Face model
(`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, downloaded on the first entailment run). The stopword list is hard-coded and
sentence splitting uses a blank spaCy pipeline, so no NLTK or spaCy data is required.

If the model download fails with `401 Unauthorized`, an expired Hugging Face token is saved on your machine: run
`huggingface-cli logout` (or delete `~/.cache/huggingface/token`) and try again. The model is public.

## Repo layout
```
factcheck.py                 SUBMISSION FILE (the only code file we submit)
factchecking_main.py         provided driver (DO NOT EDIT)
requirements.txt             pinned packages
data/                        provided datasets
experiments/                 result tables, threshold sweeps, error lists, Part 1 experiment scripts
docs/                        findings, experiment log, interfaces, git workflow
scripts/                     setup_env.sh, check_submission.sh / check_submission.ps1
report/                      report sources and the final PDF
*.py in the repo root        Part 2 experiment scripts (pruning_sweep.py, threshold_experiment.py, dump_errors.py, ...)
```
Do not commit `venv/` or `.venv/` (they are git-ignored).

## Rules
1. **Do not edit `factchecking_main.py`.**
2. `factcheck.py` must work with the pinned packages only, on CPU as well as GPU, without errors or out-of-memory problems.
3. Before submitting, run `scripts/check_submission.ps1` (Windows) or `scripts/check_submission.sh` in a fresh Python 3.10
   environment. It must print 0.8190 and 0.8552.
4. **GenAI:** the assessment is in the "Editing" category. Each member's use of GenAI is declared in the report's
   acknowledgement section, using the template in the assessment brief.
5. Everyone must be able to explain every part of the code and the report.