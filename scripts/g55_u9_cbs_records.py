import argparse
import csv
import re
from pathlib import Path

HEAD_RE = re.compile(r"<thead.*?</thead>", re.S)
WORD = r">\s*([^<>\s][^<>]{0,39}?)\s*<"
ALT = r'alt="(?:player headshot|headshot-image)"[^>]*>.*?'
NAME_RE = re.compile(ALT + WORD + ".*?" + WORD, re.S)
OVERALL_RE = re.compile(r"Overall:\s*(\d+)-(\d+)-(\d+)")
FILE_RE = re.compile(r"cbs_ats_(\d{4})_(\d{14})\.html$")
FIELDS = ["season", "capture", "expert", "w", "l", "p"]


def parse(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    head = HEAD_RE.search(text)
    block = head.group(0) if head else text
    names = [f"{a} {b}" for a, b in NAME_RE.findall(block)]
    records = OVERALL_RE.findall(text)
    return names, records


def main():
    parser = argparse.ArgumentParser(description="Parse archived CBS expert ATS season records.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.src).glob("cbs_ats_*.html")):
        match = FILE_RE.search(path.name)
        if not match:
            continue
        names, records = parse(path)
        if len(names) != len(records):
            print(f"{path.name}: names {len(names)} records {len(records)} mismatch")
            continue
        for name, (w, lo, p) in zip(names, records, strict=True):
            rows.append(
                {
                    "season": int(match.group(1)),
                    "capture": match.group(2),
                    "expert": name,
                    "w": int(w),
                    "l": int(lo),
                    "p": int(p),
                }
            )
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        n = row["w"] + row["l"]
        record = f"{row['w']}-{row['l']}-{row['p']}"
        print(row["season"], row["expert"], record, f"{row['w'] / n:.3f}")


if __name__ == "__main__":
    main()
