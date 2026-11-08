"""
Part 1 ablation: preprocessing choices x overlap metric x passage handling, each with a threshold sweep.

Run from the repo root (inside the project venv):
    python experiments/part1_ablation.py
Writes experiments/part1_ablation.csv, experiments/part1_ablation.md and experiments/part1_best_errors.csv
(and experiments/part1_threshold_curves.png if matplotlib is installed).
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
THRESHOLDS = [i / 100 for i in range(0, 101)]
OUT_DIR = "experiments"


def load():
    examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
    gold = [0 if ex.label == "NS" or ex.label == "IR" else 1 for ex in examples]  # 1 = S, IR counts as NS
    return examples, gold


def sweep(scores, gold):
    """Accuracy for every threshold (predict S when score >= threshold)."""
    accs = []
    for t in THRESHOLDS:
        correct = sum(1 for s, g in zip(scores, gold) if (1 if s >= t else 0) == g)
        accs.append(correct / len(gold))
    return accs


def best_threshold(accs):
    """Best accuracy, the plateau of thresholds that reach it, and the middle of that plateau (more robust)."""
    best = max(accs)
    plateau = [t for t, a in zip(THRESHOLDS, accs) if abs(a - best) < 1e-12]
    mid = plateau[len(plateau) // 2]
    return best, mid, plateau[0], plateau[-1]


def run_config(examples, gold, metric, lowercase, stopwords, stem, pooled):
    checker = WordRecallThresholdFactChecker(metric=metric, lowercase=lowercase, remove_stopwords=stopwords,
                                             stem=stem, pool_passages=pooled)
    scores = [checker.score(ex.fact, ex.passages) for ex in examples]
    accs = sweep(scores, gold)
    best, mid, lo, hi = best_threshold(accs)
    return dict(metric=metric, lowercase=lowercase, stopwords=stopwords, stem=stem,
                passages="pooled" if pooled else "max", best_acc=best, threshold=mid,
                plateau=f"{lo:.2f}-{hi:.2f}", scores=scores, accs=accs)


def main():
    examples, gold = load()
    print(f"{len(examples)} examples, S={sum(gold)}, NS+IR={len(gold) - sum(gold)}")
    rows = []
    for metric, lc, sw, st, pooled in itertools.product(
            ["precision", "jaccard", "dice"], [True, False], [True, False], [True, False], [False, True]):
        rows.append(run_config(examples, gold, metric, lc, sw, st, pooled))
    rows.sort(key=lambda r: -r["best_acc"])

    fields = ["metric", "lowercase", "stopwords", "stem", "passages", "best_acc", "threshold", "plateau"]
    with open(os.path.join(OUT_DIR, "part1_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # One-factor-at-a-time table around the best configuration (what the report should discuss)
    best = rows[0]
    base = dict(metric=best["metric"], lowercase=best["lowercase"], stopwords=best["stopwords"],
                stem=best["stem"], passages=best["passages"])
    lines = ["| Change from best config | Metric | Lowercase | Stopwords | Stem | Passages | Best acc | Threshold | Plateau |",
             "|---|---|---|---|---|---|---|---|---|"]

    def fmt(label, r):
        return (f"| {label} | {r['metric']} | {r['lowercase']} | {r['stopwords']} | {r['stem']} | {r['passages']} "
                f"| {r['best_acc']:.4f} | {r['threshold']:.2f} | {r['plateau']} |")

    lines.append(fmt("(best)", best))
    for key in ["lowercase", "stopwords", "stem"]:
        alt = dict(base)
        alt[key] = not base[key]
        for r in rows:
            if all(r[k] == alt[k] for k in alt):
                lines.append(fmt(f"{key} -> {alt[key]}", r))
    for metric in ["precision", "jaccard", "dice"]:
        if metric != base["metric"]:
            for r in rows:
                if all(r[k] == (metric if k == "metric" else base[k]) for k in base):
                    lines.append(fmt(f"metric -> {metric}", r))
    other = "pooled" if base["passages"] == "max" else "max"
    for r in rows:
        if all(r[k] == (other if k == "passages" else base[k]) for k in base):
            lines.append(fmt(f"passages -> {other}", r))
    with open(os.path.join(OUT_DIR, "part1_ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nTop 5 of all", len(rows), "configs:")
    for r in rows[:5]:
        print(fmt("", r))

    # Errors of the best configuration (for the error-analysis notes)
    t = best["threshold"]
    with open(os.path.join(OUT_DIR, "part1_best_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fact", "gold_label", "score", "prediction", "error_type"])
        for ex, g, s in zip(examples, gold, best["scores"]):
            pred = 1 if s >= t else 0
            if pred != g:
                w.writerow([ex.fact, ex.label, f"{s:.3f}", "S" if pred else "NS", "FP" if pred else "FN"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric in ["precision", "jaccard", "dice"]:
            r = next(x for x in rows if x["metric"] == metric)
            plt.plot(THRESHOLDS, r["accs"], label=f"{metric} (best {r['best_acc']:.3f})")
        plt.xlabel("threshold")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, "part1_threshold_curves.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed: skipped the plot (pip install matplotlib to enable it)")


if __name__ == "__main__":
    main()
