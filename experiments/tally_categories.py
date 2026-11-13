"""
tally_categories.py  (robust version)

Counts how many of the 20 examined errors fall in each hand-assigned category (split by
false positive / false negative) and saves category_counts.csv for the report table.

Run apply_labels.py first (it creates error_sample_labelled.csv), then:
    python tally_categories.py   (from anywhere inside the project)
Reads and writes in experiments/part2/results/.
"""

import csv
import io
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "factcheck.py").exists())
RESULTS = ROOT / "experiments" / "part2" / "results"

SAMPLE_FILE = RESULTS / "error_sample_labelled.csv"

# --- read the file, coping with Excel's UTF-8 marker and ; or tab delimiters
with open(SAMPLE_FILE, encoding="utf-8-sig", newline="") as f:
    text = f.read()

first_line = text.splitlines()[0]
delimiter = max([",", ";", "\t"], key=first_line.count)
reader = csv.reader(io.StringIO(text), delimiter=delimiter)
table = list(reader)

header = [h.strip().lower().replace(" ", "_") for h in table[0]]
print(f"File: {SAMPLE_FILE} | delimiter: {repr(delimiter)} | {len(header)} columns | {len(table) - 1} data rows")
print(f"Last 3 column names seen: {header[-3:]}\n")

if "your_category" not in header or "error_type" not in header:
    print("PROBLEM: the columns 'error_type' and/or 'your_category' were not found.")
    print("All column names I can see:")
    for i, h in enumerate(header):
        print(f"  {i}: {h}")
    print("\nFix: the header row must contain a column called exactly  your_category")
    print("(right after short_fact_6_words_or_fewer). Add it, paste the 20 labels under it,")
    print("and save again as CSV UTF-8.")
    sys.exit(1)

ci = header.index("your_category")
ti = header.index("error_type")

rows = []
for r in table[1:]:
    if not any(cell.strip() for cell in r):
        continue                                   # skip empty lines
    r = r + [""] * (len(header) - len(r))          # pad short rows
    rows.append({"error_type": r[ti].strip(), "your_category": r[ci].strip()})

unlabelled = [r for r in rows if not r["your_category"]]
if unlabelled:
    print(f"WARNING: {len(unlabelled)} of {len(rows)} rows still have no category.\n")

counts = Counter((r["error_type"], r["your_category"] or "(unlabelled)") for r in rows)
totals = Counter(r["error_type"] for r in rows)
categories = sorted({cat for _, cat in counts})

out = []
print(f"{'Category':<40}{'False Positive':>16}{'False Negative':>16}{'Total':>8}{'% of all':>10}")
print("-" * 90)
for cat in categories:
    fp = counts[("False Positive", cat)]
    fn = counts[("False Negative", cat)]
    total = fp + fn
    pct = 100 * total / len(rows)
    print(f"{cat:<40}{fp:>16}{fn:>16}{total:>8}{pct:>9.0f}%")
    out.append({"category": cat, "false_positives": fp, "false_negatives": fn,
                "total": total, "percent_of_examined": round(pct, 1)})
print("-" * 90)
print(f"{'All examined':<40}{totals['False Positive']:>16}{totals['False Negative']:>16}{len(rows):>8}{'100%':>10}")

with open(RESULTS / "category_counts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    writer.writeheader()
    writer.writerows(out)
print("\nSaved: category_counts.csv")