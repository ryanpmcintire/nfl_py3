from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES: tuple[str, ...] = ("data",)
ARTIFACT_SOURCE = "artifacts"

DEFAULT_DESTS: tuple[str, ...] = (r"E:\nfl_data_backup",)

EXCLUDE_NAMES = frozenset({"scheduler_state.json", "scheduler_log.txt"})

MTIME_TOLERANCE_SECONDS = 2.0

HASH_CHUNK_BYTES = 1 << 20
MANIFEST_NAME = "backup_manifest.json"


@dataclass
class TreeReport:
    name: str
    source_files: int = 0
    source_bytes: int = 0
    up_to_date: int = 0
    missing: int = 0
    stale: int = 0
    copied: int = 0
    copied_bytes: int = 0
    verified: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def pending(self) -> int:
        return self.missing + self.stale


@dataclass
class RunReport:
    dest: Path
    trees: list[TreeReport] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [failure for tree in self.trees for failure in tree.failures]

    def total(self, attribute: str) -> int:
        return sum(getattr(tree, attribute) for tree in self.trees)


def iter_source_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in EXCLUDE_NAMES:
            yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def copy_state(source: Path, destination: Path) -> str:
    if not destination.exists():
        return "missing"
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_size != destination_stat.st_size:
        return "stale"
    if abs(source_stat.st_mtime - destination_stat.st_mtime) > MTIME_TOLERANCE_SECONDS:
        return "stale"
    return "ok"


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,.0f} B"
        size /= 1024
    return f"{size:,.1f} TB"


def process_tree(
    name: str,
    dest_root: Path,
    *,
    apply: bool,
    verify_all: bool,
) -> TreeReport:
    report = TreeReport(name=name)
    source_root = REPO / name
    if not source_root.exists():
        return report

    for source in iter_source_files(source_root):
        relative = source.relative_to(REPO)
        destination = dest_root / relative
        size = source.stat().st_size
        report.source_files += 1
        report.source_bytes += size

        state = copy_state(source, destination)
        if state == "ok":
            report.up_to_date += 1
            if verify_all and apply:
                if sha256(source) == sha256(destination):
                    report.verified += 1
                else:
                    report.failures.append(f"content mismatch: {relative}")
            continue

        if state == "missing":
            report.missing += 1
        else:
            report.stale += 1

        if not apply:
            continue

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as error:
            report.failures.append(f"copy failed: {relative}: {error}")
            continue

        if sha256(source) == sha256(destination):
            report.copied += 1
            report.copied_bytes += size
            report.verified += 1
        else:
            report.failures.append(f"verify failed after copy: {relative}")

    return report


def write_manifest(dest_root: Path, report: RunReport, sources: Sequence[str]) -> None:
    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "repo": str(REPO),
        "sources": list(sources),
        "trees": [
            {
                "name": tree.name,
                "source_files": tree.source_files,
                "source_bytes": tree.source_bytes,
                "copied": tree.copied,
                "copied_bytes": tree.copied_bytes,
                "verified": tree.verified,
                "failures": tree.failures,
            }
            for tree in report.trees
        ],
    }
    path = dest_root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def print_report(report: RunReport, *, apply: bool) -> None:
    print(f"\n=== {report.dest} ===")
    covered_label = "covered@start" if apply else "covered"
    pending_label = "to-copy" if apply else "pending"
    width = len(covered_label)

    def row(name: str, files: int, size: int, covered: int, pending: int) -> str:
        share = f"{100.0 * covered / files:.1f}%" if files else "n/a"
        return f"{name:<26} {files:>8,} {human_bytes(size):>12} {share:>{width}} {pending:>8,}"

    header = f"{'tree':<26} {'files':>8} {'size':>12} {covered_label:>{width}} {pending_label:>8}"
    print(header)
    print("-" * len(header))
    for tree in report.trees:
        print(row(tree.name, tree.source_files, tree.source_bytes, tree.up_to_date, tree.pending))

    source_files = report.total("source_files")
    pending = report.total("missing") + report.total("stale")
    print("-" * len(header))
    print(
        row(
            "TOTAL",
            source_files,
            report.total("source_bytes"),
            report.total("up_to_date"),
            pending,
        )
    )

    if apply:
        print(
            f"\ncopied {report.total('copied'):,} files "
            f"({human_bytes(report.total('copied_bytes'))}), "
            f"verified {report.total('verified'):,} by sha256"
        )
    elif pending:
        print(f"\n{pending:,} file(s) not yet mirrored. Run without --status to copy them.")
    else:
        print("\nmirror is current.")

    for failure in report.failures:
        print(f"  FAILURE {failure}")


READ_ONLY_SCRIPT = True
READ_ONLY_EXCEPTIONS: dict[int, str] = {
    194: "destination is under dest_root (DEFAULT_DESTS / --dest), the mirror drive",
    195: "destination is under dest_root (DEFAULT_DESTS / --dest), the mirror drive",
    229: "path == dest_root / MANIFEST_NAME, the mirror drive, never artifacts/",
    324: "dest_root comes from --dest or DEFAULT_DESTS (E:\\nfl_data_backup)",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror data and artifacts to the backup drive")
    parser.add_argument(
        "--dest",
        action="append",
        metavar="PATH",
        help=f"mirror root; repeatable. Default: {', '.join(DEFAULT_DESTS)}",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="report coverage without copying anything",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="re-hash every already-mirrored file, not just the ones copied now",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="also mirror artifacts/, including prospective and research ledgers",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [*DEFAULT_SOURCES]
    if args.include_artifacts:
        sources.append(ARTIFACT_SOURCE)

    destinations = [Path(item) for item in (args.dest or DEFAULT_DESTS)]
    apply = not args.status
    exit_code = 0

    for dest_root in destinations:
        if apply:
            try:
                dest_root.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                print(f"cannot write to {dest_root}: {error}", file=sys.stderr)
                exit_code = 1
                continue
        elif not dest_root.exists():
            print(f"{dest_root} does not exist (never backed up)", file=sys.stderr)

        report = RunReport(dest=dest_root)
        for name in sources:
            print(f"[{dest_root.name}] scanning {name} ...", flush=True)
            report.trees.append(
                process_tree(name, dest_root, apply=apply, verify_all=args.verify_all)
            )

        print_report(report, apply=apply)
        if apply:
            write_manifest(dest_root, report, sources)
        if report.failures:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
