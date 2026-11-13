"""
ablation.py

Runs the experiments that justify the design choices in factcheck.py (Part 2).
DeBERTa is run ONCE per (sentence, fact) pair and the probabilities are cached, so every
threshold / pruning / strategy comparison afterwards is instant.

Experiments
  1. Strategy:   how to turn entailment/neutral/contradiction into S / NS
  2. Threshold:  accuracy for every threshold of the final strategy
  3. Pruning:    accuracy, sentences checked and estimated time for several overlap cutoffs
  4. Cleaning:   raw text vs. cleaned text vs. split-on-<s>-tags-then-clean vs. whitespace-only vs. markup-removal
  5. Validation: tune the threshold on one random half, test on the other half (100 splits)

Run:  python ablation.py   (from anywhere inside the project)
Output (in experiments/): ablation_strategy.csv, ablation_threshold.csv, ablation_pruning.csv,
        ablation_cleaning.csv, ablation_split_half.csv, ablation_cache.pkl
(Delete ablation_cache.pkl if you change MODEL_NAME or the data.)
"""

import csv
import os
import pickle
import re
import sys
import time
from pathlib import Path

# find the project root (the folder that holds factcheck.py), wherever this script is saved
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "factcheck.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                    # data/ paths below are relative to the project root
RESULTS = ROOT / "experiments"
RESULTS.mkdir(parents=True, exist_ok=True)        # all csv/txt output goes here

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factcheck import EntailmentModel, EntailmentFactChecker, clean_passage
from factchecking_main import read_passages, read_fact_examples

# ------------------------------------------------------------------ configuration
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
CACHE_FILE = RESULTS / "ablation_cache.pkl"
SEED = 42

FINAL_THRESHOLD = 0.36      # values used in factcheck.py
FINAL_CUTOFF = 0.1
CUTOFFS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

STRATEGIES = {
    "max entailment probability": lambda e, n, c: e,
    "entailment minus contradiction": lambda e, n, c: e - c,
    "one minus neutral": lambda e, n, c: 1.0 - n,
    "discrete (entailment is top class)": lambda e, n, c: 1.0 if (e >= n and e >= c) else 0.0,
}
MAIN_STRATEGY = "max entailment probability"

TAG_SPLIT_RE = re.compile(r"</?s>")
SPACE_RE = re.compile(r"\s+")
MARKUP_RE = re.compile(r"\{\{.*?(?:\}\}|(?=</s>)|$)", re.S)     # leftover wiki markup such as {{cite news|...

# seconds per sentence measured on the first cold (uncached) run: 137.3 s / 5354 sentences.
# Used when the cache is already full and no new timing can be measured.
SEC_PER_SENTENCE_FALLBACK = 137.3 / 5354


# ------------------------------------------------------------------ load data and model
print("Loading dataset...")
examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
gold = np.array([0 if ex.label in ("NS", "IR") else 1 for ex in examples])   # IR counts as NS
print(f"{len(examples)} examples, {gold.sum()} supported")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
ent_model = EntailmentModel(model, tokenizer, cuda=(device.type == "cuda"))
checker = EntailmentFactChecker(ent_model, entailment_threshold=FINAL_THRESHOLD, overlap_cutoff=FINAL_CUTOFF)
print(f"Device: {device}")

model_cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        model_cache = pickle.load(f)
    print(f"Loaded {len(model_cache)} cached model outputs")


# ------------------------------------------------------------------ sentence splitting variants
def sentences_of(text, variant):
    if not text or not text.strip():
        return []
    if variant == "raw":
        chunks = [text]
    elif variant == "clean":
        chunks = [clean_passage(text)]
    elif variant == "tag_split":
        chunks = [clean_passage(c) for c in TAG_SPLIT_RE.split(text)]
    elif variant == "ws_only":                  # collapse spaces/newlines but KEEP the <s> tags
        chunks = [SPACE_RE.sub(" ", text).strip()]
    elif variant == "markup":                   # remove wiki markup, KEEP the <s> tags
        chunks = [SPACE_RE.sub(" ", MARKUP_RE.sub(" ", text)).strip()]
    else:
        raise ValueError(variant)
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        for s in checker.nlp(chunk).sents:
            t = s.text.strip()
            if t:
                out.append(t)
    return out


timing = {"calls": 0, "seconds": 0.0}


def collect(variant, floor):
    """For every example: list of (overlap, p_ent, p_neu, p_con) for sentences with overlap >= floor."""
    all_records = []
    for ex in tqdm(examples, desc=f"{variant} (floor {floor})"):
        recs = []
        for passage in ex.passages:
            for sent in sentences_of(passage.get("text", ""), variant):
                overlap = checker._overlap(ex.fact, sent)
                if overlap < floor:
                    continue
                key = (sent, ex.fact)
                if key not in model_cache:
                    t0 = time.time()
                    p = ent_model.check_entailment(premise=sent, hypothesis=ex.fact)
                    timing["seconds"] += time.time() - t0
                    timing["calls"] += 1
                    model_cache[key] = (p["entailment"], p["neutral"], p["contradiction"])
                recs.append((overlap,) + model_cache[key])
        all_records.append(recs)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(model_cache, f)
    return all_records


# raw text FIRST and with no pruning, so the timing is measured on a cold cache
records = {"raw": collect("raw", 0.0)}
sec_per_sentence = timing["seconds"] / timing["calls"] if timing["calls"] else SEC_PER_SENTENCE_FALLBACK
for v in ("clean", "tag_split", "ws_only", "markup"):
    records[v] = collect(v, FINAL_CUTOFF)


# ------------------------------------------------------------------ scoring helpers
NOTHING = -9.0     # example with no sentence left after pruning -> always NS


def best_scores(recs, fn, cutoff):
    out = []
    for per_example in recs:
        s = [fn(e, n, c) for ov, e, n, c in per_example if ov >= cutoff]
        out.append(max(s) if s else NOTHING)
    return np.array(out)


def acc_at(scores, thr):
    return float(((scores >= thr).astype(int) == gold).mean())


def grid_for(name):
    if name.startswith("entailment minus"):
        return np.round(np.arange(-1.0, 1.0001, 0.01), 2)
    if name.startswith("discrete"):
        return np.array([0.5])
    return np.round(np.arange(0.0, 1.0001, 0.01), 2)


def tune(scores, grid, idx=None):
    s, g = (scores, gold) if idx is None else (scores[idx], gold[idx])
    accs = np.array([((s >= t).astype(int) == g).mean() for t in grid])
    ties = grid[accs == accs.max()]
    return float(ties[len(ties) // 2]), float(accs.max())      # middle of the best plateau


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def show(title, rows):
    print("\n" + title)
    print("=" * len(title))
    keys = list(rows[0].keys())
    print(" | ".join(f"{k:<34}" if i == 0 else f"{k:<14}" for i, k in enumerate(keys)))
    for r in rows:
        print(" | ".join(f"{str(r[k]):<34}" if i == 0 else f"{str(r[k]):<14}" for i, k in enumerate(keys)))


# ------------------------------------------------------------------ 1. strategy
rows = []
for name, fn in STRATEGIES.items():
    scores = best_scores(records["raw"], fn, FINAL_CUTOFF)
    thr, acc = tune(scores, grid_for(name))
    rows.append({"strategy": name, "best_threshold": thr, "best_accuracy": round(acc, 4),
                 "correct_of_total": f"{int(round(acc * len(gold)))}/{len(gold)}"})
write_csv(RESULTS / "ablation_strategy.csv", rows)
show("1. DECISION STRATEGY (raw text, overlap cutoff 0.1)", rows)

# ------------------------------------------------------------------ 2. threshold sweep
main_fn = STRATEGIES[MAIN_STRATEGY]
main_scores = best_scores(records["raw"], main_fn, FINAL_CUTOFF)
rows = [{"threshold": round(float(t), 2), "accuracy": round(acc_at(main_scores, t), 4)}
        for t in np.arange(0.10, 0.90, 0.02)]
rows.append({"threshold": FINAL_THRESHOLD, "accuracy": round(acc_at(main_scores, FINAL_THRESHOLD), 4)})
rows.sort(key=lambda r: r["threshold"])
write_csv(RESULTS / "ablation_threshold.csv", rows)
show(f"2. THRESHOLD SWEEP ({MAIN_STRATEGY}); final threshold = {FINAL_THRESHOLD}", rows)

# ------------------------------------------------------------------ 3. pruning
rows = []
total_sentences = sum(len(r) for r in records["raw"])
for cutoff in CUTOFFS:
    scores = best_scores(records["raw"], main_fn, cutoff)
    checked = sum(1 for per_example in records["raw"] for ov, *_ in per_example if ov >= cutoff)
    thr, best_acc = tune(scores, grid_for(MAIN_STRATEGY))
    rows.append({
        "overlap_cutoff": cutoff,
        f"acc_at_{FINAL_THRESHOLD}": round(acc_at(scores, FINAL_THRESHOLD), 4),
        "best_acc": round(best_acc, 4),
        "sentences_checked": checked,
        "pct_skipped": round(100 * (1 - checked / total_sentences), 1),
        "est_seconds": round(checked * sec_per_sentence, 1),
    })
write_csv(RESULTS / "ablation_pruning.csv", rows)
show(f"3. PRUNING (raw text; {total_sentences} sentences in total; {sec_per_sentence:.3f} s per sentence)", rows)

# ------------------------------------------------------------------ 4. cleaning
rows = []
for variant in ("raw", "clean", "tag_split", "ws_only", "markup"):
    scores = best_scores(records[variant], main_fn, FINAL_CUTOFF)
    thr, best_acc = tune(scores, grid_for(MAIN_STRATEGY))
    rows.append({"text_handling": variant, f"acc_at_{FINAL_THRESHOLD}": round(acc_at(scores, FINAL_THRESHOLD), 4),
                 "best_threshold": thr, "best_acc": round(best_acc, 4)})
write_csv(RESULTS / "ablation_cleaning.csv", rows)
show("4. TEXT CLEANING (overlap cutoff 0.1)", rows)

# ------------------------------------------------------------------ 5. split-half validation
rng = np.random.default_rng(SEED)
test_accs, chosen = [], []
for _ in range(100):
    perm = rng.permutation(len(gold))
    a, b = perm[: len(gold) // 2], perm[len(gold) // 2:]
    for tune_idx, test_idx in ((a, b), (b, a)):
        thr, _ = tune(main_scores, grid_for(MAIN_STRATEGY), tune_idx)
        chosen.append(thr)
        test_accs.append(float((((main_scores[test_idx] >= thr).astype(int)) == gold[test_idx]).mean()))
rows = [{"mean_test_accuracy": round(float(np.mean(test_accs)), 4),
         "std": round(float(np.std(test_accs)), 4),
         "mean_chosen_threshold": round(float(np.mean(chosen)), 3),
         "threshold_std": round(float(np.std(chosen)), 3)}]
write_csv(RESULTS / "ablation_split_half.csv", rows)
show("5. SPLIT-HALF VALIDATION (tune on one half, test on the other; 200 runs)", rows)

print("\nSaved: ablation_strategy.csv, ablation_threshold.csv, ablation_pruning.csv, "
      "ablation_cleaning.csv, ablation_split_half.csv")


# ------------------------------------------------------------------ 6. which facts change prediction?
raw_pred = best_scores(records["raw"], main_fn, FINAL_CUTOFF) >= FINAL_THRESHOLD
for v in ("ws_only", "markup"):
    pred = best_scores(records[v], main_fn, FINAL_CUTOFF) >= FINAL_THRESHOLD
    changed = np.where(raw_pred != pred)[0]
    print(f"\n6. {v}: {len(changed)} facts change prediction compared with raw text")
    for i in changed:
        print(f"   {examples[i].fact} | gold {'S' if gold[i] else 'NS'} | raw {'S' if raw_pred[i] else 'NS'} -> {v} {'S' if pred[i] else 'NS'}")