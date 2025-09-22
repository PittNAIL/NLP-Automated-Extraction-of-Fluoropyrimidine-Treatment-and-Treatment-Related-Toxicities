import os
import csv
from pathlib import Path
import re

# Path to your CSV
csv_path = Path(r"D:\Github\...\val.csv")
out_dir = csv_path.parent  # write txt files next to the CSV; change if you want another folder

# Optional: put TXT files into a subfolder instead
# out_dir = csv_path.parent / "handfootpreventative_txt"
# out_dir.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    # Remove characters not allowed on Windows file systems
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip()

# Try UTF-8 with BOM first; fall back to latin-1 if needed
def read_rows(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

rows = read_rows(csv_path)

# Be tolerant to slight header variations
def pick_col(headers, candidates):
    for c in candidates:
        if c in headers:
            return c
    # try case-insensitive match
    low = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Could not find any of columns {candidates} in CSV headers: {headers}")

if not rows:
    raise ValueError("CSV appears to be empty (no data rows).")

headers = rows[0].keys()
sentence_col = pick_col(headers, ["sentence", "sentences", "Sentence", "text"])
target_col = pick_col(headers, ["target", "Target", "label", "Label"])

for idx, row in enumerate(rows, start=1):
    sentence = (row.get(sentence_col) or "").strip()
    target = str(row.get(target_col, "")).strip()
    # Ensure target is 0/1 text; if blank or weird, coerce to 0/1 when possible
    if target not in {"0", "1"}:
        # Attempt to coerce numerics like 0.0 -> 0
        try:
            target = str(int(float(target)))
        except Exception:
            target = sanitize_filename(target) or "NA"

    fname = f"drugsofinterest{idx}_{target}.txt"
    fname = sanitize_filename(fname)
    out_path = out_dir / fname

    with out_path.open("w", encoding="utf-8") as f:
        f.write(sentence)

print(f"Done. Wrote {len(rows)} files to: {out_dir}")
