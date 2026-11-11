"""
Part 1 extra experiments (run after part1_ablation.py), from the repo root:
    python experiments/part1_extra_experiments.py
Tests: tokenization variants, minimum token length, stopword-list size, stemming vs lemmatization (and the
lowercase confound), alternative scoring (IDF-weighted, sentence-level, 2-sentence window, bigram), and a
more honest threshold estimate (grouped cross-validation by person + bootstrap confidence interval).
Writes experiments/part1_extra_results.md
"""
import math
import os
import re
import string
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.stem import PorterStemmer, SnowballStemmer
from part1_ablation import load, sweep, best_threshold, THRESHOLDS, PASSAGES_PATH, OUT_DIR
from factchecking_main import read_passages
from factcheck import STOPWORDS, overlap_from_tokens

TAG_RE = re.compile(r"</?s>")
SMALL_STOPWORDS = frozenset("a an the and or of in on at to is was were are be been by for with as from that this it".split())

try:  # lemmatization needs the WordNet data; skipped (not crashed) if it is not available
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        WordNetLemmatizer().lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    WordNetLemmatizer().lemmatize("tests")
    LEMMA_OK = True
except Exception:
    LEMMA_OK = False

TOKEN_REGEXES = {
    "regex (letters+digits)": re.compile(r"[^\W_]+"),
    "letters only (drop digits)": re.compile(r"[^\W\d_]+"),
    "regex + keep apostrophe words": re.compile(r"[^\W_]+(?:'[^\W_]+)?"),
}


def make_pipeline(tok="regex (letters+digits)", lowercase=True, stop="full", norm="porter", min_len=1):
    stoplist = {"none": frozenset(), "small": SMALL_STOPWORDS, "full": STOPWORDS}[stop]
    porter, snowball = PorterStemmer(), SnowballStemmer("english")
    lemma = WordNetLemmatizer() if LEMMA_OK else None
    normalizers = {"none": lambda w: w, "porter": porter.stem, "snowball": snowball.stem,
                   "lemma": (lemma.lemmatize if lemma else (lambda w: w))}
    normalize = normalizers[norm]
    cache = {}

    def pipeline(text):
        text = TAG_RE.sub(" ", text)
        if lowercase:
            text = text.lower()
        if tok == "whitespace + strip punctuation":
            tokens = [t.strip(string.punctuation) for t in text.split()]
            tokens = [t for t in tokens if t]
        else:
            tokens = TOKEN_REGEXES[tok].findall(text)
        out = []
        for t in tokens:
            if len(t) < min_len or t.lower() in stoplist:
                continue
            n = cache.get(t)
            if n is None:
                n = cache[t] = normalize(t)
            out.append(n)
        return out
    return pipeline


def split_sentences(text):
    text = TAG_RE.sub(" ", text)
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def evaluate(examples, gold, score_fn):
    scores = np.array([score_fn(ex) for ex in examples])
    accs = sweep(scores, gold)
    best, mid, lo, hi = best_threshold(accs)
    return dict(scores=scores, best=best, thr=mid, plateau=f"{lo:.2f}-{hi:.2f}")


def precision_max(pipe):
    cache = {}

    def passage_set(text):
        s = cache.get(text)
        if s is None:
            s = cache[text] = set(pipe(text))
        return s

    def score(ex):
        ft = pipe(ex.fact)
        return max((overlap_from_tokens(ft, passage_set(p["text"]), "precision") for p in ex.passages), default=0.0)
    return score


def idf_precision(pipe, df, n_docs):
    def weight(t):
        return math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1

    def score(ex):
        ft = set(pipe(ex.fact))
        if not ft:
            return 0.0
        total = sum(weight(t) for t in ft)
        best = 0.0
        for p in ex.passages:
            ps = set(pipe(p["text"]))
            best = max(best, sum(weight(t) for t in ft & ps) / total)
        return best
    return score


def sentence_precision(pipe, window=1):
    def score(ex):
        ft = set(pipe(ex.fact))
        if not ft:
            return 0.0
        best = 0.0
        for p in ex.passages:
            sents = [set(pipe(s)) for s in split_sentences(p["text"])]
            units = sents if window == 1 else [set().union(*sents[i:i + window]) for i in range(max(1, len(sents) - window + 1))]
            for u in units:
                best = max(best, len(ft & u) / len(ft))
        return best
    return score


def bigram_precision(pipe, mix=False):
    def score(ex):
        ft = pipe(ex.fact)
        fset, fbi = set(ft), bigrams(ft)
        if not fset:
            return 0.0
        best = 0.0
        for p in ex.passages:
            pt = pipe(p["text"])
            uni = len(fset & set(pt)) / len(fset)
            if fbi:
                bi = len(fbi & bigrams(pt)) / len(fbi)
                s = (uni + bi) / 2 if mix else bi
            else:
                s = uni
            best = max(best, s)
        return best
    return score


def grouped_cv(scores, gold, groups, k=5, repeats=30, seed=0):
    """Choose the threshold on training people only, score on held-out people; repeated with random splits."""
    rng = np.random.default_rng(seed)
    gold = np.array(gold)
    T = np.array(THRESHOLDS)
    correct = (scores[:, None] >= T[None, :]) == gold[:, None].astype(bool)
    uniq = sorted(set(groups))
    g_idx = np.array([uniq.index(g) for g in groups])
    results = []
    for _ in range(repeats):
        order = rng.permutation(len(uniq))
        fold_of = np.empty(len(uniq), dtype=int)
        fold_of[order] = np.arange(len(uniq)) % k
        folds = fold_of[g_idx]
        total = 0
        for f in range(k):
            train, test = folds != f, folds == f
            accs = correct[train].mean(axis=0)
            _, mid, _, _ = best_threshold(list(accs))
            total += correct[test][:, THRESHOLDS.index(mid)].sum()
        results.append(total / len(gold))
    return float(np.mean(results)), float(np.std(results))


def bootstrap_ci(scores, gold, thr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ok = ((scores >= thr) == np.array(gold).astype(bool)).astype(float)
    means = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table(title, rows):
    lines = [f"### {title}", "", "| Setting | Best acc | Threshold | Plateau |", "|---|---|---|---|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['best']:.4f} | {r['thr']:.2f} | {r['plateau']} |")
    return "\n".join(lines) + "\n"


def main():
    examples, gold = load()
    groups = [ex.passages[0]["title"] if ex.passages else ex.fact for ex in examples]
    print(f"{len(examples)} examples from {len(set(groups))} people; lemmatizer available: {LEMMA_OK}")
    out = []

    # 1. tokenization (baseline settings otherwise)
    rows = []
    for tok in list(TOKEN_REGEXES) + ["whitespace + strip punctuation"]:
        rows.append((tok, evaluate(examples, gold, precision_max(make_pipeline(tok=tok)))))
    for m in [2, 3]:
        rows.append((f"regex, min token length {m}", evaluate(examples, gold, precision_max(make_pipeline(min_len=m)))))
    out.append(table("1. Tokenization and minimum token length (stopwords + Porter, precision, max passage)", rows))

    # 2. stopword list size
    rows = [(f"stopwords: {s}", evaluate(examples, gold, precision_max(make_pipeline(stop=s)))) for s in ["none", "small", "full"]]
    out.append(table("2. Stopword list (Porter stemming, precision)", rows))

    # 3. normalization + lowercase confound
    rows = []
    norms = ["none", "porter", "snowball"] + (["lemma"] if LEMMA_OK else [])
    for n in norms:
        rows.append((f"normalizer: {n}", evaluate(examples, gold, precision_max(make_pipeline(norm=n)))))
    for lc in [True, False]:
        rows.append((f"no normalizer, lowercase={lc}", evaluate(examples, gold, precision_max(make_pipeline(norm="none", lowercase=lc)))))
    out.append(table("3. Stemming vs lemmatization, and lowercasing with the stemmer switched off", rows))

    # 4. scoring alternatives (best preprocessing)
    pipe = make_pipeline()
    all_passages = {p["text"] for ps in read_passages(PASSAGES_PATH).values() for p in ps}
    df = Counter()
    for t in all_passages:
        df.update(set(pipe(t)))
    n_docs = len(all_passages)
    candidates = [
        ("precision, best passage (current)", precision_max(pipe)),
        ("IDF-weighted precision", idf_precision(pipe, df, n_docs)),
        ("precision, best single sentence", sentence_precision(pipe, 1)),
        ("precision, best 2-sentence window", sentence_precision(pipe, 2)),
        ("bigram precision", bigram_precision(pipe)),
        ("average of unigram and bigram precision", bigram_precision(pipe, mix=True)),
    ]
    rows = [(name, evaluate(examples, gold, fn)) for name, fn in candidates]
    out.append(table("4. Scoring alternatives (precision-style variants)", rows))

    # 5. honest estimates
    lines = ["### 5. Threshold honesty: in-sample vs grouped cross-validation (5 folds by person, 30 repeats)", "",
             "| Scoring | In-sample best | Grouped CV mean +- sd | 95% bootstrap CI at chosen threshold |", "|---|---|---|---|"]
    for (name, r) in rows:
        cv_mean, cv_sd = grouped_cv(r["scores"], gold, groups)
        lo, hi = bootstrap_ci(r["scores"], gold, r["thr"])
        lines.append(f"| {name} | {r['best']:.4f} | {cv_mean:.4f} +- {cv_sd:.4f} | {lo:.3f}-{hi:.3f} |")
    out.append("\n".join(lines) + "\n")

    text = "\n".join(out)
    with open(os.path.join(OUT_DIR, "part1_extra_results.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
