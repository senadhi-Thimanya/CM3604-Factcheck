"""
apply_labels.py

Rebuilds the labelled error sample WITHOUT Excel. Reads error_analysis.csv (written by
error_analysis.py), finds the 20 examined facts by their text, adds the 'your_category'
column and saves error_sample_labelled.csv.

Run:  python apply_labels.py   (from anywhere inside the project)
Then run tally_categories.py, which reads the labelled file.
Reads and writes in experiments/part2/results/.
"""

import csv
import io
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "factcheck.py").exists())
RESULTS = ROOT / "experiments" / "part2" / "results"

SOURCE = RESULTS / "error_analysis.csv"
OUTPUT = RESULTS / "error_sample_labelled.csv"

LOOSE = "Loose support for vague claim"
DEBATE = "Debatable gold label"
UNUSABLE = "Evidence in unusable form"
INFER = "Needs inference beyond wording"

LABELS = {
    "Furlong worked as a hunting guide.": LOOSE,
    "Cleveland is in Ohio.": DEBATE,
    "Marianne McAndrew is a singer.": LOOSE,
    "She has worked with a number of organizations.": LOOSE,
    "Jean Daullé was a printmaker.": DEBATE,
    "Kerser is a songwriter.": LOOSE,
    "He has worked as a researcher at various organizations.": LOOSE,
    "He served as a sniper.": DEBATE,
    "Mayo Clinic is an organization.": DEBATE,
    "Tusa has been involved in various cultural initiatives.": LOOSE,
    "They divorced in 1964.": UNUSABLE,
    "Gerhard Fischer invented the first handheld, battery-operated metal detector.": INFER,
    "Gerhard Fischer is best known for inventing a metal detector.": INFER,
    "Jean Daullé was known for his portraits.": INFER,
    "Red Widow is a novel by Katsu.": UNUSABLE,
    "The Taker trilogy is a novel by Katsu.": UNUSABLE,
    "She was born on July 18, 1987.": UNUSABLE,
    "With CSKA, he won domestic titles.": UNUSABLE,
    '"New Heart." is popular.': INFER,
    '"Familiar Wife." is popular.': INFER,
}

with open(SOURCE, encoding="utf-8-sig", newline="") as f:
    text = f.read()
delimiter = max([",", ";", "\t"], key=text.splitlines()[0].count)
reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
fieldnames = reader.fieldnames + ["your_category"]

found = {}
for row in reader:
    fact = row["fact"].strip()
    if fact in LABELS and fact not in found:
        row["your_category"] = LABELS[fact]
        found[fact] = row

missing = [fact for fact in LABELS if fact not in found]
print(f"Matched {len(found)} of {len(LABELS)} facts (delimiter {repr(delimiter)}).")
if missing:
    print("NOT FOUND (check spelling / quotes in error_analysis.csv; re-run error_analysis.py if needed):")
    for m in missing:
        print("  -", m)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([found[fact] for fact in LABELS if fact in found])

print(f"Saved {OUTPUT}")