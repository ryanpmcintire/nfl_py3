"""Remove `#` comments from Python sources (owner ban, 2026-09-09); `--check` only reports."""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

PRAGMA = re.compile(r"^#\s*(noqa|type:|pragma|fmt:|isort:|ruff:|mypy:|pylint:|!)")
ROOTS = ("src", "scripts", "tests", ".claude/hooks")


def comment_tokens(source: str) -> list[tokenize.TokenInfo]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [t for t in tokens if t.type == tokenize.COMMENT and not PRAGMA.match(t.string)]


def strip(source: str) -> str:
    comments = comment_tokens(source)
    if not comments:
        return source
    lines = source.splitlines(keepends=True)
    drop: set[int] = set()
    cut: dict[int, int] = {}
    for tok in comments:
        row, col = tok.start
        before = lines[row - 1][:col]
        if before.strip() == "":
            drop.add(row - 1)
        else:
            cut[row - 1] = min(col, cut.get(row - 1, col))
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in drop:
            continue
        if index in cut:
            head = line[: cut[index]].rstrip()
            newline = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            line = head + newline
        out.append(line)
    return "".join(out)


def python_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py" and path.is_file():
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(ROOTS))
    parser.add_argument(
        "--check", action="store_true", help="report offending files, exit 1 if any"
    )
    args = parser.parse_args(argv)
    offending = 0
    changed = 0
    for path in python_files(args.paths):
        source = path.read_text(encoding="utf-8")
        found = comment_tokens(source)
        if not found:
            continue
        offending += 1
        if args.check:
            for tok in found[:3]:
                print(f"{path}:{tok.start[0]}: {tok.string[:80]}")
            continue
        path.write_text(strip(source), encoding="utf-8")
        changed += 1
    if args.check:
        print(f"{offending} file(s) with comments")
        return 1 if offending else 0
    print(f"stripped comments from {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
