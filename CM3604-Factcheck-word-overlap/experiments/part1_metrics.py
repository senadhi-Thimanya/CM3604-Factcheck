"""
Steps 4-6 (Member B): evaluation harness + TF-IDF cosine + ROUGE + comparison table.

Run from the repo root (inside the venv):
    python experiments/part1_metrics.py

Writes into experiments/:
    metrics_scores.csv      one row per fact: gold label + the score from every metric
    metrics_sweep.csv       accuracy of every metric at every threshold 0.00 .. 1.00
    metrics_comparison.md   the comparison table for the report (best threshold + accuracy per metric)
    metrics_confusion.txt   confusion matrix of each metric at its best threshold
    metrics_curves.png      accuracy-vs-threshold plot (only if matplotlib is installed)
"""
import csv
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factchecking_main import read_passages, read_fact_examples
from factcheck import (preprocess, overlap_from_tokens, WordRecallThresholdFactChecker,
                       RandomGuessFactChecker, AlwaysEntailedFactChecker)
from rouge_score import rouge_scorer

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
OUT_DIR = "experiments"
THRESHOLDS = [i / 100 for i in range(0, 101)]
PREP = dict(lowercase=True, remove_stopwords=True, stem=True)   # Member A's best preprocessing



#  the harness
def load():
    examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
    gold = [1 if ex.label == "S" else 0 for ex in examples]      # 1 = S ; NS and IR = 0
    return examples, gold


def confusion(gold, preds):
    """Returns [[TP, FN], [FP, TN]] with S as the positive class (same layout idea as the main file)."""
    tp = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 1)
    fn = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 0)
    fp = sum(1 for g, p in zip(gold, preds) if g == 0 and p == 1)
    tn = sum(1 for g, p in zip(gold, preds) if g == 0 and p == 0)
    return tp, fn, fp, tn


def accuracy(gold, preds):
    return sum(1 for g, p in zip(gold, preds) if g == p) / len(gold)


def sweep(scores, gold):
    """Accuracy at every threshold. Predict S when score >= threshold. Scores are computed ONCE, so this is instant."""
    return [accuracy(gold, [1 if s >= t else 0 for s in scores]) for t in THRESHOLDS]


def best_of(accs):
    """Best accuracy, and the MIDDLE of the plateau of thresholds that reach it (more robust than the first one)."""
    best = max(accs)
    plateau = [t for t, a in zip(THRESHOLDS, accs) if abs(a - best) < 1e-12]
    return best, plateau[len(plateau) // 2], plateau[0], plateau[-1]


def run_checker(checker, examples, gold):
    """Harness for any FactChecker object (random, always_entail, ...). Returns (accuracy, confusion)."""
    preds = [1 if checker.predict(ex.fact, ex.passages) == "S" else 0 for ex in examples]
    return accuracy(gold, preds), confusion(gold, preds)



#  TF-IDF cosine 

class TfidfCosine:
    """
    Each distinct passage text in the dataset counts as one 'document'.
    idf(w) = log((1 + N) / (1 + df(w))) + 1   (smoothed, always > 0, so unseen fact words do not crash)
    vector = {word: tf * idf};  cosine = dot / (|a| * |b|)
    """
    def __init__(self, examples):
        docs = {p["text"] for ex in examples for p in ex.passages}
        self.n_docs = len(docs)
        df = Counter()
        self.tokens = {}
        for text in docs:
            toks = preprocess(text, **PREP)
            self.tokens[text] = toks
            df.update(set(toks))
        self.idf = {w: math.log((1 + self.n_docs) / (1 + c)) + 1 for w, c in df.items()}
        self.default_idf = math.log(1 + self.n_docs) + 1           # idf of a word found in no passage
        self.vec_cache = {}

    def _vector(self, tokens):
        tf = Counter(tokens)
        vec = {w: c * self.idf.get(w, self.default_idf) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    def _passage_vec(self, text):
        v = self.vec_cache.get(text)
        if v is None:
            v = self._vector(self.tokens.get(text) or preprocess(text, **PREP))
            self.vec_cache[text] = v
        return v

    def score(self, fact, passages):
        fvec, fnorm = self._vector(preprocess(fact, **PREP))
        if not passages or fnorm == 0:
            return 0.0
        best = 0.0
        for p in passages:
            pvec, pnorm = self._passage_vec(p["text"])
            if pnorm == 0:
                continue
            small, big = (fvec, pvec) if len(fvec) < len(pvec) else (pvec, fvec)
            dot = sum(v * big.get(w, 0.0) for w, v in small.items())
            best = max(best, dot / (fnorm * pnorm))
        return best



#  ROUGE

class RougeMetric:
    def __init__(self, kind="rouge1", use_preprocess=True):
        self.kind = kind
        self.use_preprocess = use_preprocess
        # If we feed already-stemmed tokens we must not stem again; on raw text let ROUGE stem for us.
        self.scorer = rouge_scorer.RougeScorer([kind], use_stemmer=not use_preprocess)
        self.cache = {}

    def _text(self, text):
        if not self.use_preprocess:
            return text
        t = self.cache.get(text)
        if t is None:
            t = " ".join(preprocess(text, **PREP))
            self.cache[text] = t
        return t

    def score(self, fact, passages):
        if not passages:
            return 0.0
        fact_text = self._text(fact)
        if not fact_text.strip():
            return 0.0
        best = 0.0
        for p in passages:
            r = self.scorer.score(self._text(p["text"]), fact_text)[self.kind]   # (target, prediction)
            best = max(best, r.precision)
        return best


# Adapters: every metric becomes "function"

def overlap_metric(metric):
    checker = WordRecallThresholdFactChecker(metric=metric, **PREP)
    return checker.score


def main():
    examples, gold = load()
    print(f"{len(examples)} examples: S={sum(gold)}, NS+IR={len(gold) - sum(gold)}")

    #  sanity check the harness with the trivial checkers  
    acc, cm = run_checker(AlwaysEntailedFactChecker(), examples, gold)
    print(f"[harness check] always_entail accuracy = {acc:.4f}  (TP,FN,FP,TN) = {cm}")
    acc, cm = run_checker(RandomGuessFactChecker(), examples, gold)
    print(f"[harness check] random        accuracy = {acc:.4f}  (TP,FN,FP,TN) = {cm}")

    tfidf = TfidfCosine(examples)
    metrics = {
        "Precision overlap (A)": overlap_metric("precision"),
        "Jaccard (A)": overlap_metric("jaccard"),
        "TF-IDF cosine": tfidf.score,
        "ROUGE-1 precision (preprocessed)": RougeMetric("rouge1", True).score,
        "ROUGE-1 precision (raw text, ROUGE stemmer)": RougeMetric("rouge1", False).score,
        "ROUGE-L precision (preprocessed)": RougeMetric("rougeL", True).score,
    }

    #  compute every score ONCE 
    all_scores = {}
    for name, fn in metrics.items():
        print("scoring:", name)
        all_scores[name] = [fn(ex.fact, ex.passages) for ex in examples]

    #  per-fact scores file 
    with open(os.path.join(OUT_DIR, "metrics_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fact", "gold_label"] + list(metrics))
        for i, ex in enumerate(examples):
            w.writerow([ex.fact, ex.label] + [f"{all_scores[m][i]:.4f}" for m in metrics])

    # threshold sweeps 
    all_accs = {name: sweep(sc, gold) for name, sc in all_scores.items()}
    with open(os.path.join(OUT_DIR, "metrics_sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold"] + list(metrics))
        for i, t in enumerate(THRESHOLDS):
            w.writerow([f"{t:.2f}"] + [f"{all_accs[m][i]:.4f}" for m in metrics])

    #  comparison table + confusion matrices at each best threshold 
    lines = ["| Metric | Best threshold | Plateau | Best accuracy | TP | FN | FP | TN |",
             "|---|---|---|---|---|---|---|---|"]
    conf_lines = []
    for name in metrics:
        best, mid, lo, hi = best_of(all_accs[name])
        preds = [1 if s >= mid else 0 for s in all_scores[name]]
        tp, fn, fp, tn = confusion(gold, preds)
        lines.append(f"| {name} | {mid:.2f} | {lo:.2f}-{hi:.2f} | {best:.4f} | {tp} | {fn} | {fp} | {tn} |")
        conf_lines.append(f"{name} @ threshold {mid:.2f}\n"
                          f"              pred S   pred NS\n"
                          f"  gold S      {tp:6d}   {fn:6d}\n"
                          f"  gold NS     {fp:6d}   {tn:6d}\n")
    table = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "metrics_comparison.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT_DIR, "metrics_confusion.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(conf_lines))
    print("\n" + table + "\n")
    print("\n".join(conf_lines))

    # plot 
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name in metrics:
            plt.plot(THRESHOLDS, all_accs[name], label=f"{name} (best {max(all_accs[name]):.3f})")
        plt.xlabel("threshold")
        plt.ylabel("accuracy")
        plt.legend(fontsize=7)
        plt.savefig(os.path.join(OUT_DIR, "metrics_curves.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed: skipped the plot (pip install matplotlib to enable it)")


if __name__ == "__main__":
    main()