# Run BEFORE submitting, from the repo root, inside a fresh Python 3.10 venv (.venv\Scripts\Activate.ps1).
$ver = python --version
Write-Host "== $ver =="
if ($ver -notmatch "3\.10") { Write-Host "WARNING: the coursework requires Python 3.10" }

Write-Host "== word_overlap =="
python factchecking_main.py --mode word_overlap | Select-Object -Last 8
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: word_overlap"; exit 1 }

Write-Host "== entailment (CPU) =="
python factchecking_main.py --mode entailment | Select-Object -Last 8
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: entailment"; exit 1 }

Write-Host "== factchecking_main.py unchanged? =="
git diff --quiet -- factchecking_main.py
if ($LASTEXITCODE -eq 0) { Write-Host "OK (no local changes)" } else { Write-Host "WARNING: factchecking_main.py modified" }

Write-Host "Done. Expected: word_overlap 0.8190, entailment 0.8552 (targets 0.75 and 0.83)"