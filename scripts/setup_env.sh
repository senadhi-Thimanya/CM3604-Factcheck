#!/usr/bin/env bash
# Creates a virtual environment with the pinned requirements. Needs Python 3.10.
set -e
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
echo "Environment ready. Activate with: source .venv/bin/activate"
