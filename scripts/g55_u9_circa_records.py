import argparse
import pathlib
import re
import sys

import pandas as pd
import pypdf

ROW = re.compile(
    r"^(\d{1,3}(?:,\d{3})*)(T?)\s*(.+?)\s+(\d+)\s+(\d+)-(\d+)-(\d+)-(\d+)\s+(-?\d+(?:\.\d)?)"
)


def parse(path):
    season, capture = path.stem.split("_")[1:3]
    reader = pypdf.PdfReader(str(path))
    rows, missed = [], 0
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            hit = ROW.match(line.strip())
            if not hit:
                if re.search(r"\d+-\d+-\d+-\d+", line):
                    missed += 1
                continue
            place, _, entry, picks, w, lo, p, x, pts = hit.groups()
            rows.append(
                {
                    "season": int(season),
                    "capture": capture,
                    "expert": entry.strip(),
                    "w": int(w),
                    "l": int(lo),
                    "p": int(p) + int(x),
                    "picks": int(picks),
                    "points": float(pts),
                    "place": int(place.replace(",", "")),
                }
            )
    frame = pd.DataFrame(rows)
    check = frame["w"] + 0.5 * frame["p"] - frame["points"]
    bad = int((check.abs() > 1e-9).sum())
    print(f"{path.name} rows {len(frame)} missed {missed} points_mismatch {bad}", file=sys.stderr)
    return frame[check.abs() <= 1e-9]


def main():
    parser = argparse.ArgumentParser(description="Circa Million season records from PDFs.")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = sorted(pathlib.Path(args.pdf_dir).glob("circa_*.pdf"))
    frame = pd.concat([parse(p) for p in paths], ignore_index=True)
    dup = frame.duplicated(["season", "expert"]).sum()
    print(f"duplicate season-entry rows {dup}", file=sys.stderr)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("season").agg(entries=("expert", "size"), w=("w", "sum"), l=("l", "sum")))


if __name__ == "__main__":
    main()
