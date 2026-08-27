"""Incremental off-device mirror of the local data trees.

Why this exists
---------------
On 2026-08-27 an audit of the backup state found that ~70% of `data/` by bytes
was copied anywhere, the copy was 11 days stale, and the entire `data/raw/`
tree -- 741 MB across 31 scraped sources -- had never been backed up at all.
The one prior backup was a hand-run `robocopy` from 2026-08-16 that a session
performed once and nobody repeated, because nothing in the repository knew it
was supposed to happen.

The point-in-time captures under `data/market/raw/` and most of `data/raw/`
are scrapes of pages that have since changed. They cannot be re-fetched at
their original timestamps, so losing them does not cost a download -- it costs
the observation permanently. That asymmetry is the whole justification for this
script: everything here is cheap to run and some of what it protects is
impossible to recreate.

Same-disk copies do not count
-----------------------------
The repository lives on F:. The 2026-08-16 backup wrote to BOTH `F:\\` and
`E:\\`, which reads like two copies but is one device plus a folder on the
source disk -- a single SSD failure takes the repo and the F: copy together.
`DEFAULT_DESTS` therefore points at E: only (a separate physical drive). Pass
`--dest` to add more; pass it twice for two drives.

Mirror, not dated snapshots
---------------------------
The 2026-08-16 backups are dated, read-only, verified snapshots. Those stay
exactly as they are -- they are the immutable floor. This script maintains a
separate rolling MIRROR instead, because a 3.5 GB tree that grows every week
cannot afford a fresh full snapshot per run. The tradeoff is deliberate and
worth naming: a mirror can propagate a deletion, a dated snapshot cannot. The
mirror never deletes on its own (there is no `--delete`), so the failure mode
requires someone to remove files from the mirror by hand.

Usage
-----
    python scripts/backup_data.py --status   # what is covered, what is not
    python scripts/backup_data.py            # copy what is new/changed, verify it
    python scripts/backup_data.py --verify-all   # re-hash the whole mirror

`--status` is the command that answers "how much of our data is backed up?"
without an ad-hoc investigation. Every mode is idempotent.
"""

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

# Trees mirrored by default. `artifacts/` is excluded on purpose: at 1.9 GB it
# is the largest tree in the repo and it is fully regenerable from `data/` plus
# the code, so it is opt-in via --include-artifacts rather than paid for weekly.
DEFAULT_SOURCES: tuple[str, ...] = ("data",)
ARTIFACT_SOURCE = "artifacts"

# The owner's off-device drive, in version control on purpose -- same reasoning
# as SCHEDULE in capture_scheduler.py. A machine fact that governs whether data
# survives belongs somewhere reviewable, not in one operator's shell history.
DEFAULT_DESTS: tuple[str, ...] = (r"E:\nfl_data_backup",)

# Machine-local runtime state. Restoring these onto a different machine would
# make the capture scheduler believe it had already run this machine's windows,
# so they are deliberately not mirrored. Both rebuild themselves.
EXCLUDE_NAMES = frozenset({"scheduler_state.json", "scheduler_log.txt"})

# NTFS-to-NTFS preserves mtime exactly through copy2; the tolerance is for
# destinations that round (FAT/exFAT external drives, some network mounts).
MTIME_TOLERANCE_SECONDS = 2.0

HASH_CHUNK_BYTES = 1 << 20
MANIFEST_NAME = "backup_manifest.json"


@dataclass
class TreeReport:
    """Per-source-tree tally for one destination."""

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
    """Everything one destination's run produced, for printing and the manifest."""

    dest: Path
    trees: list[TreeReport] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [failure for tree in self.trees for failure in tree.failures]

    def total(self, attribute: str) -> int:
        return sum(getattr(tree, attribute) for tree in self.trees)


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield every mirrorable file under `root`, skipping machine-local state."""
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
    """Classify one file as ``ok``, ``missing`` or ``stale``.

    Size plus mtime, not content: hashing 3.5 GB to decide what to skip would
    cost more than the copy it saves. Content IS checked, but only on the files
    this run actually wrote (or under --verify-all), which is where a silent
    corruption would have been introduced.
    """
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
    """Mirror (or, when `apply` is False, merely audit) one source tree."""
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

        # Verify what we just wrote. A copy that silently truncated is exactly
        # the failure a backup must not report as success.
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
    # In apply mode these two columns describe the state this run FOUND, before
    # it copied anything -- so they are labelled that way. Without the suffix a
    # weekly log reads "covered 0.0%" directly above "copied 42,839 files" and
    # looks like a contradiction rather than a before/after pair.
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
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
        help="also mirror artifacts/ (1.9 GB, regenerable from data/ plus code)",
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
