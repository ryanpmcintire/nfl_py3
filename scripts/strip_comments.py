from __future__ import annotations

import argparse
import ast
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


def docstring_nodes(source: str) -> list[tuple[ast.Expr, bool]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[tuple[ast.Expr, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.append((first, len(node.body) == 1 and not isinstance(node, ast.Module)))
    return found


def remove_docstrings(source: str) -> str:
    nodes = docstring_nodes(source)
    if not nodes:
        return source
    lines = source.splitlines(keepends=True)
    for node, only_statement in sorted(nodes, key=lambda item: item[0].lineno, reverse=True):
        start, end = node.lineno - 1, node.end_lineno or node.lineno
        first_line = lines[start]
        indent = first_line[: len(first_line) - len(first_line.lstrip())]
        newline = "\r\n" if lines[end - 1].endswith("\r\n") else "\n"
        replacement = [f"{indent}pass{newline}"] if only_statement else []
        lines[start:end] = replacement
    text = "".join(lines)
    return text.lstrip("\r\n") if nodes and nodes[0][0].lineno == 1 else text


def strip(source: str) -> str:
    comments = comment_tokens(source)
    if comments:
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
        source = "".join(out)
    return remove_docstrings(source)


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
    parser = argparse.ArgumentParser(
        description="Remove # comments and docstrings from Python sources; --check only reports."
    )
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
        docs = docstring_nodes(source)
        if not found and not docs:
            continue
        offending += 1
        if args.check:
            for tok in found[:3]:
                print(f"{path}:{tok.start[0]}: {tok.string[:80]}")
            for node, _ in docs[:3]:
                print(f"{path}:{node.lineno}: docstring")
            continue
        path.write_text(strip(source), encoding="utf-8")
        changed += 1
    if args.check:
        print(f"{offending} file(s) with comments or docstrings")
        return 1 if offending else 0
    print(f"stripped comments and docstrings in {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
