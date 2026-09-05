"""Artifact retention measurement and dry-run pruning planner (WP6 / ROADMAP OPS-02).

Why this exists
----------------
`artifacts/` and `data/` are entirely gitignored (see `.gitignore`) except for
`artifacts/prospective/` and the `.gitkeep` markers under `data/raw/` and
`data/processed/`. Everything else -- every experiment screen's timestamped
output, every raw scrape snapshot, every processed feature table -- lives only
on this machine (mirrored to E: by `scripts/backup_data.py`, see its
docstring for exactly what that mirror covers and how stale it is allowed to
get). Nobody has ever measured how big these trees are, which runs are still
load-bearing, and which are safe to eventually compact or prune. This script
is that measurement, plus a DRY-RUN planner. It never deletes, moves, or
renames anything -- there is no delete mode in this file, and none should be
added here without a separate, explicitly-approved pass (see
`docs/artifact_retention.md` for the exact next step).

What counts as protected
-------------------------
A path is protected -- never listed as a prunable candidate, regardless of
age -- if any of the following is true:

1. It, OR ANY ANCESTOR DIRECTORY OF IT, is referenced by a string value
   anywhere inside `registry/weak_signals.json`, `registry/rotation_registry.json`,
   every file under `registry/experiments/**` and `registry/experiment_specs/*`,
   every `docs/*.md` file, `README.md`, `ROADMAP.md`, `HANDOFF.md`, or
   `CURRENT_PREDICTIONS.md`, or the local `artifacts/active_ats_model.json`
   manifest (including its schema's bare `"<family>/<stamp>"` shorthand for
   `historical_evaluation.artifact` / `weekly_forecast.artifact`), or
   `artifacts/prospective/challengers.json`. The ancestor-inclusive check
   (`_protection_for`) is what lets a generic architecture mention -- "outputs
   land under `artifacts/foo/`", with no specific run named -- protect every
   run inside that family, without the reference scan needing to enumerate
   each one; the run-discovery walk still decomposes the family into its
   individual timestamped runs (see COARSE_NO_DESCEND for the one narrow
   exception).
2. It falls under `artifacts/prospective/` (the prospective challenger
   ledgers and paper-decision records -- carved out of `.gitignore` on
   purpose) or `artifacts/clv_ledger/` (the append-only paper-decision ledger
   read/written by `paper_decision_ledger_path()` in `src/nfl_ats/clv.py`,
   which is the concrete file behind `load_paper_decisions()` /
   `record_paper_decisions()` -- exactly the "ledger" AGENTS.md says must
   never be lost), or `artifacts/lockday_packages/` /
   `artifacts/lockday_packages_rehearsal/` (ENG-01's independently
   verifiable lock-day decision packages and their rehearsal counterpart --
   see `src/nfl_ats/lockday_package.py` and `scripts/lockday_package_verify.py`;
   every JSON file under either is also scanned for path references the same
   way `registry/experiments/**` is, so a package manifest citing a specific
   run elsewhere protects that run too), or `data/scheduler_state.json` /
   `data/scheduler_log.txt` (the capture
   scheduler's persisted heartbeat and run log -- ROADMAP ENG-03; currently
   also doc-referenced by `docs/capture_scheduling.md`, but hardcoded here so
   that incidental protection is never the only thing keeping it alive), or
   it *is* `artifacts/active_ats_model.json` itself.
3. It is a raw scrape or odds snapshot -- ENG-19's `point_in_time_capture`
   retention class (see `src/nfl_ats/artifact_retention_policy.py`): every
   run under the `data/raw`, `data/market`, or `data/players` top-level
   trees, plus any run elsewhere whose path contains a literal `raw`
   segment. These cannot be re-fetched at their original timestamps, so
   `build_plan` excludes them unconditionally -- independent of rule 1,
   which happens to also protect most of them today via doc references.
4. It is the single newest run within its own group (same immediate parent
   directory) -- a light extra safety net so a bug in the reference scan
   above never strands a family with zero surviving output.

Nothing else is protected. In particular, an experiment family with no doc
or registry reference and several superseded (non-newest) timestamped runs
older than the age threshold shows up in `--plan` as a candidate -- and in
practice most of the repo's LARGE families (`margins/`, `backtests/`,
`experiments/`, `player_experiments/`, `player_model_selection/`,
`cfb_role_experiments/`, ...) have no such blanket doc mention, so this is
where `--plan` actually finds candidates; the smaller, thoroughly-documented
families mostly protect themselves wholesale via rule 1.

Path-reference discovery is done programmatically, never by a hardcoded
family-name list (seven paths -- `active_ats_model.json`,
`artifacts/prospective/`, `artifacts/clv_ledger/`, `artifacts/lockday_packages/`,
`artifacts/lockday_packages_rehearsal/`, `data/scheduler_state.json`,
`data/scheduler_log.txt` -- are hardcoded because they are read/written by
*path*, not named in any doc; everything else is discovered by scanning file
content). See `collect_protected_refs`.

ENG-19 (`src/nfl_ats/artifact_retention_policy.py`) adds a named retention-
class vocabulary (`evidence` / `point_in_time_capture` / `scratch` /
`reproducible` -- see that module's docstring) on top of the measurement and
planning below, plus a dry-run `--budget-check` that never deletes anything
either.

Usage
-----
    python scripts/artifact_retention.py --report [--json]
    python scripts/artifact_retention.py --plan [--older-than-days N] [--json]
    python scripts/artifact_retention.py --budget-check [--budget-multiplier X] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats import artifact_retention_policy as retention_policy  # noqa: E402

# ---------------------------------------------------------------------------
# Path-reference discovery
# ---------------------------------------------------------------------------

# A "run" boundary: the UTC-stamped directory or file name every capture,
# experiment screen, and forecast artifact in this repo is named with, e.g.
# "20260821T182533Z" or "2026-week-01-20260824T120725Z".
TIMESTAMP_RE = re.compile(r"\d{8}T\d{6}Z")

# Matches an artifacts/... or data/... path reference inside free text or a
# JSON string value, forward-slash or Windows-backslash separated (JSON
# escaping is already undone by json.loads before this ever runs against a
# JSON string; raw doc text may use either separator). \b keeps this from
# matching "database" or "metadata" -- neither has a word boundary before the
# "data" substring.
PATH_REF_RE = re.compile(r"\b(?:artifacts|data)[/\\][A-Za-z0-9_.\\/-]+")

# artifacts/active_ats_model.json's own schema stores some references as a
# bare two-segment "<family>/<stamp>" string (e.g. "margins/20260824T120013Z")
# that is implicitly relative to artifacts/ -- it never spells out the
# "artifacts/" prefix. Recognise that shape specifically for that one file.
BARE_FAMILY_STAMP_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

NAMED_DATA_SUBTREES = ("raw", "processed", "market", "players")

DOC_SOURCES = ("README.md", "ROADMAP.md", "HANDOFF.md", "CURRENT_PREDICTIONS.md")
DOC_GLOBS = ("docs/*.md",)
REGISTRY_JSON_SOURCES = ("registry/weak_signals.json", "registry/rotation_registry.json")
REGISTRY_JSON_GLOBS = ("registry/experiments/**/*.json", "registry/experiment_specs/*.json")

ACTIVE_MODEL_REL = "artifacts/active_ats_model.json"
PROSPECTIVE_REL = "artifacts/prospective"
CLV_LEDGER_REL = "artifacts/clv_ledger"

# ENG-19 gap-close: ENG-01's lock-day decision packages, and its rehearsal
# counterpart (both folders another session is adding concurrently -- see
# src/nfl_ats/lockday_package.py's PACKAGES_DIRNAME). Protected wholesale
# like PROSPECTIVE_REL/CLV_LEDGER_REL (they package exactly the evidence a
# published lock-day forecast, or its rehearsal, depends on); an empty or
# nonexistent directory is a no-op both here and in the glob below, so this
# is safe to wire in before either folder exists on disk.
LOCKDAY_PACKAGES_REL = "artifacts/lockday_packages"
LOCKDAY_PACKAGES_REHEARSAL_REL = "artifacts/lockday_packages_rehearsal"
# Scanned for path references the same way REGISTRY_JSON_GLOBS is: a package
# manifest (src/nfl_ats/lockday_package.py's build_manifest/hashed_files)
# cites specific runs elsewhere by repo-relative path, and those runs must be
# protected too, not just the package that cites them.
LOCKDAY_PACKAGE_JSON_GLOBS = (
    "artifacts/lockday_packages/**/*.json",
    "artifacts/lockday_packages_rehearsal/**/*.json",
)

# Never listed as prunable, full stop, independent of any reference scan.
# Discovered by reading src/nfl_ats/clv.py (paper_decision_ledger_path ->
# artifacts/clv_ledger/decisions.parquet, the append-only paper-decision
# ledger) and src/nfl_ats/prospective_scoring.py (challenger_decisions.parquet
# under artifacts/prospective/), plus the .gitignore carve-out for
# artifacts/prospective/ and the AGENTS.md session-startup instruction to
# always inspect artifacts/active_ats_model.json. `data/scheduler_state.json`
# / `data/scheduler_log.txt` are also currently protected incidentally via
# docs/capture_scheduling.md:345 -- hardcoded here too (ENG-19) so that a
# future doc edit can never silently drop the capture scheduler's heartbeat.
ALWAYS_PROTECTED = (
    ACTIVE_MODEL_REL,
    PROSPECTIVE_REL,
    CLV_LEDGER_REL,
    LOCKDAY_PACKAGES_REL,
    LOCKDAY_PACKAGES_REHEARSAL_REL,
    "data/scheduler_state.json",
    "data/scheduler_log.txt",
)


_ALNUM_RE = re.compile(r"[A-Za-z0-9_]")


def extract_path_refs(text: str) -> list[str]:
    """Pull every artifacts/... or data/... path-looking substring out of text.

    Rejects two shapes of false positive found in real docs: a bare tree
    root with nothing after it ("artifacts", "data" -- never a meaningful
    reference on its own), and a CLI-usage template placeholder such as
    ``<artifacts/.../<run-id>>`` whose literal ".." segments are not real
    directory names (a segment must contain at least one alnum/underscore
    character to count).
    """

    out = []
    for match in PATH_REF_RE.finditer(text):
        normalized = match.group(0).replace("\\", "/")
        normalized = normalized.rstrip("/.,;:)")
        if not normalized or normalized in ("artifacts", "data"):
            continue
        parts = [part for part in normalized.split("/") if part]
        if len(parts) < 2:
            continue
        if any(not _ALNUM_RE.search(part) for part in parts[1:]):
            continue
        out.append(normalized)
    return out


def iter_json_strings(obj: Any) -> Iterator[str]:
    """Yield every string value found anywhere inside a parsed JSON structure."""

    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_json_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_json_strings(value)


def protected_node_for_ref(ref: str) -> str:
    """Collapse a path reference to its run boundary.

    "artifacts/family/20260101T000000Z/results.json" -> the run directory
    "artifacts/family/20260101T000000Z". A reference with no timestamped
    segment (a flat file, or a bare family directory like
    "artifacts/rehearsal_lockday") is returned unchanged -- it protects
    exactly the thing named.
    """

    parts = ref.split("/")
    for index, part in enumerate(parts):
        if TIMESTAMP_RE.search(part):
            return "/".join(parts[: index + 1])
    return ref


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def collect_protected_refs(repo_root: Path) -> dict[str, set[str]]:
    """Discover every artifacts/ or data/ path referenced by a protected source.

    Returns {normalized run-boundary path: {source files that referenced it}}.
    Every entry is either lifted verbatim from a tracked doc/registry file,
    parsed out of the local active-model manifest, or one of the three
    ALWAYS_PROTECTED paths -- never guessed.
    """

    refs: dict[str, set[str]] = {}

    def add(raw_ref: str, source: str) -> None:
        node = protected_node_for_ref(raw_ref)
        refs.setdefault(node, set()).add(source)

    doc_paths = [repo_root / name for name in DOC_SOURCES]
    for pattern in DOC_GLOBS:
        doc_paths.extend(sorted(repo_root.glob(pattern)))
    for doc_path in doc_paths:
        if not doc_path.is_file():
            continue
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        source = doc_path.relative_to(repo_root).as_posix()
        for ref in extract_path_refs(text):
            add(ref, source)

    json_paths = [repo_root / name for name in REGISTRY_JSON_SOURCES]
    for pattern in REGISTRY_JSON_GLOBS:
        json_paths.extend(sorted(repo_root.glob(pattern)))
    for pattern in LOCKDAY_PACKAGE_JSON_GLOBS:
        json_paths.extend(sorted(repo_root.glob(pattern)))
    for json_path in json_paths:
        if not json_path.is_file():
            continue
        data = _load_json(json_path)
        if data is None:
            continue
        source = json_path.relative_to(repo_root).as_posix()
        for value in iter_json_strings(data):
            for ref in extract_path_refs(value):
                add(ref, source)

    active_model_path = repo_root / ACTIVE_MODEL_REL
    if active_model_path.is_file():
        data = _load_json(active_model_path)
        if data is not None:
            for value in iter_json_strings(data):
                for ref in extract_path_refs(value):
                    add(ref, ACTIVE_MODEL_REL)
                if BARE_FAMILY_STAMP_RE.match(value) and TIMESTAMP_RE.search(value):
                    add(f"artifacts/{value}", ACTIVE_MODEL_REL)

    challengers_path = repo_root / PROSPECTIVE_REL / "challengers.json"
    if challengers_path.is_file():
        data = _load_json(challengers_path)
        if data is not None:
            source = challengers_path.relative_to(repo_root).as_posix()
            for value in iter_json_strings(data):
                for ref in extract_path_refs(value):
                    add(ref, source)

    for hardcoded in ALWAYS_PROTECTED:
        add(hardcoded, "hardcoded: runtime ledger/manifest (see ALWAYS_PROTECTED)")

    return refs


# ---------------------------------------------------------------------------
# Filesystem scanning
# ---------------------------------------------------------------------------


@dataclass
class SubtreeStats:
    size_bytes: int = 0
    file_count: int = 0
    largest_file_rel: str | None = None
    largest_file_bytes: int = 0
    max_mtime: float = 0.0
    skipped_reparse_points: list[str] = field(default_factory=list)


def scan_subtree(path: Path, repo_root: Path) -> SubtreeStats:
    """Sum bytes/files under `path`. Never follows a symlink or junction.

    Defensive against the worktree-junction hazard: a reparse point is
    recorded (so it can be surfaced) but never descended into, so a stray
    junction can neither balloon the byte count nor cause a traversal cycle.
    """

    stats = SubtreeStats()
    try:
        is_link = path.is_symlink()
    except OSError:
        is_link = False
    if is_link:
        stats.skipped_reparse_points.append(path.relative_to(repo_root).as_posix())
        return stats
    if path.is_file():
        try:
            st = path.stat()
        except OSError:
            return stats
        stats.size_bytes = st.st_size
        stats.file_count = 1
        stats.largest_file_rel = path.relative_to(repo_root).as_posix()
        stats.largest_file_bytes = st.st_size
        stats.max_mtime = st.st_mtime
        return stats

    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    stats.skipped_reparse_points.append(
                        Path(entry.path).relative_to(repo_root).as_posix()
                    )
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            stats.size_bytes += st.st_size
            stats.file_count += 1
            stats.max_mtime = max(stats.max_mtime, st.st_mtime)
            if st.st_size > stats.largest_file_bytes:
                stats.largest_file_bytes = st.st_size
                stats.largest_file_rel = Path(entry.path).relative_to(repo_root).as_posix()
    return stats


@dataclass
class RunNode:
    tree: str
    rel: str
    is_dir: bool
    stamp: str | None
    size_bytes: int
    file_count: int
    largest_file_rel: str | None
    largest_file_bytes: int
    effective_time: float
    group_key: str
    protected: bool
    protected_by: tuple[str, ...]
    wholesale: bool


def _parse_stamp(token: str) -> float | None:
    match = TIMESTAMP_RE.search(token)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(0), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.timestamp()


def _protection_for(rel: str, protected: dict[str, set[str]]) -> set[str] | None:
    """Ancestor-inclusive protected-source lookup.

    A path is protected if it, or ANY ancestor directory of it, is a
    protected reference. This is what makes a bare, non-timestamped doc
    mention of a whole family ("outputs land under `artifacts/foo/`") protect
    every run inside that family, without requiring the doc to enumerate
    every run -- while still letting the run-discovery walk decompose the
    family into its individual runs for accurate reporting.
    """

    parts = rel.split("/")
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end])
        hit = protected.get(candidate)
        if hit is not None:
            return hit
    return None


# Directories that are never decomposed into individual runs, independent of
# the general reference-scan protection mechanism above. This is a narrow,
# explicit, PERFORMANCE-and-sanity exception, not a protection rule: measured
# 2026-09-01 (`du -sh artifacts/rehearsal_lockday` -> 2.1G), `rehearsal_lockday/
# sim/` is a full hard-link mirror of most of `data/` built by
# `scripts/lockday_rehearsal.py` for a dry-run rehearsal. Decomposing it would
# re-discover thousands of nodes that are byte-for-byte the same on-disk data
# already counted under `data/` (hard links share inodes -- deleting a link
# here would not even free the reported bytes). It is also referenced wholesale
# in `docs/week1_readiness.md`, so it is protected either way; this constant
# only controls whether the walk decomposes it or reports it as one node.
COARSE_NO_DESCEND = frozenset({"artifacts/rehearsal_lockday"})


def discover_runs(
    repo_root: Path,
    tree: str,
    roots: list[Path],
    protected: dict[str, set[str]],
) -> list[RunNode]:
    """Find every "run" node under `roots`.

    A run boundary is the first path segment (walking downward) whose name
    contains a timestamp token, or -- when none exists anywhere below a
    directory -- that directory itself (a flat family), or a loose file with
    no timestamp in its name. See COARSE_NO_DESCEND for the one explicit,
    documented exception that stops the walk early regardless.
    """

    runs: list[RunNode] = []

    def make_run(node_path: Path) -> RunNode:
        rel = node_path.relative_to(repo_root).as_posix()
        stamp_match = TIMESTAMP_RE.search(node_path.name)
        stamp = stamp_match.group(0) if stamp_match else None
        stats = scan_subtree(node_path, repo_root)
        effective_time = _parse_stamp(node_path.name)
        if effective_time is None:
            effective_time = stats.max_mtime
        if not effective_time:
            try:
                effective_time = node_path.stat().st_mtime
            except OSError:
                effective_time = 0.0
        provenance = _protection_for(rel, protected)
        return RunNode(
            tree=tree,
            rel=rel,
            is_dir=node_path.is_dir(),
            stamp=stamp,
            size_bytes=stats.size_bytes,
            file_count=stats.file_count,
            largest_file_rel=stats.largest_file_rel,
            largest_file_bytes=stats.largest_file_bytes,
            effective_time=effective_time,
            group_key=node_path.parent.relative_to(repo_root).as_posix(),
            protected=provenance is not None,
            protected_by=tuple(sorted(provenance)) if provenance else (),
            wholesale=provenance is not None and stamp is None,
        )

    def walk(directory: Path) -> bool:
        rel = directory.relative_to(repo_root).as_posix()
        if rel in COARSE_NO_DESCEND:
            runs.append(make_run(directory))
            return True
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return False
        found_any = False
        for entry in entries:
            if entry.name == ".gitkeep":
                continue
            entry_path = Path(entry.path)
            try:
                is_link = entry.is_symlink()
            except OSError:
                is_link = False
            if is_link:
                # Defensive: never descend into a reparse point (worktree
                # junction hazard) and never treat it as a run either.
                continue
            if TIMESTAMP_RE.search(entry.name):
                runs.append(make_run(entry_path))
                found_any = True
                continue
            if entry.is_dir(follow_symlinks=False):
                if walk(entry_path):
                    found_any = True
                else:
                    runs.append(make_run(entry_path))
                    found_any = True
            else:
                runs.append(make_run(entry_path))
                found_any = True
        return found_any

    for root in roots:
        if not root.exists():
            continue
        try:
            if root.is_symlink():
                continue
        except OSError:
            continue
        if root.is_file():
            runs.append(make_run(root))
            continue
        walk(root)

    return runs


def top_level_tree_specs(repo_root: Path) -> dict[str, list[Path]]:
    specs: dict[str, list[Path]] = {"artifacts": [repo_root / "artifacts"]}
    data_root = repo_root / "data"
    for name in NAMED_DATA_SUBTREES:
        specs[f"data/{name}"] = [data_root / name]
    others: list[Path] = []
    if data_root.is_dir():
        for entry in sorted(data_root.iterdir()):
            if entry.name in NAMED_DATA_SUBTREES or entry.name == ".gitkeep":
                continue
            others.append(entry)
    specs["data/other"] = others
    return specs


def artifact_family_of(run: RunNode) -> str:
    parts = run.rel.split("/")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2 and run.is_dir:
        return parts[1]
    return "(root files)"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class TreeSummary:
    name: str
    run_count: int
    total_bytes: int
    oldest_stamp: str | None
    newest_stamp: str | None
    largest_file_rel: str | None
    largest_file_bytes: int
    protected_run_count: int


def summarize(name: str, runs: list[RunNode]) -> TreeSummary:
    total_bytes = sum(run.size_bytes for run in runs)
    stamped = [run for run in runs if run.stamp]
    oldest = min((run.stamp for run in stamped), default=None)
    newest = max((run.stamp for run in stamped), default=None)
    largest_rel: str | None = None
    largest_bytes = -1
    for run in runs:
        if run.largest_file_bytes > largest_bytes:
            largest_bytes = run.largest_file_bytes
            largest_rel = run.largest_file_rel
    protected_count = sum(1 for run in runs if run.protected)
    return TreeSummary(
        name=name,
        run_count=len(runs),
        total_bytes=total_bytes,
        oldest_stamp=oldest,
        newest_stamp=newest,
        largest_file_rel=largest_rel,
        largest_file_bytes=max(largest_bytes, 0),
        protected_run_count=protected_count,
    )


@dataclass
class ReportData:
    generated_at_utc: str
    protected_refs: dict[str, list[str]]
    tree_rows: list[TreeSummary]
    family_rows: list[TreeSummary]


def build_report(repo_root: Path) -> ReportData:
    protected = collect_protected_refs(repo_root)
    specs = top_level_tree_specs(repo_root)

    tree_rows = []
    artifacts_runs: list[RunNode] = []
    for tree_name, roots in specs.items():
        runs = discover_runs(repo_root, tree_name, roots, protected)
        if tree_name == "artifacts":
            artifacts_runs = runs
        tree_rows.append(summarize(tree_name, runs))

    families: dict[str, list[RunNode]] = {}
    for run in artifacts_runs:
        families.setdefault(artifact_family_of(run), []).append(run)
    family_rows = [summarize(name, runs) for name, runs in sorted(families.items())]

    return ReportData(
        generated_at_utc=datetime.now(UTC).isoformat(),
        protected_refs={key: sorted(value) for key, value in protected.items()},
        tree_rows=tree_rows,
        family_rows=family_rows,
    )


# ---------------------------------------------------------------------------
# Plan (dry run only)
# ---------------------------------------------------------------------------


@dataclass
class PlanCandidate:
    rel: str
    tree: str
    family: str | None
    size_bytes: int
    age_days: float
    stamp: str | None
    retention_class: str


@dataclass
class PlanData:
    generated_at_utc: str
    older_than_days: int
    protected_refs: dict[str, list[str]]
    candidates: list[PlanCandidate]
    total_bytes: int


def build_plan(repo_root: Path, older_than_days: int = 30) -> PlanData:
    protected = collect_protected_refs(repo_root)
    specs = top_level_tree_specs(repo_root)
    now = datetime.now(UTC).timestamp()
    cutoff_seconds = older_than_days * 86400

    runs_by_tree: dict[str, list[RunNode]] = {}
    for tree_name, roots in specs.items():
        runs_by_tree[tree_name] = discover_runs(repo_root, tree_name, roots, protected)

    newest_by_group: dict[str, RunNode] = {}
    for runs in runs_by_tree.values():
        for run in runs:
            current = newest_by_group.get(run.group_key)
            if current is None or run.effective_time > current.effective_time:
                newest_by_group[run.group_key] = run

    candidates: list[PlanCandidate] = []
    for tree_name, runs in runs_by_tree.items():
        for run in runs:
            if run.protected:
                continue
            # ENG-19: point-in-time captures are never a candidate, full
            # stop -- independent of whether a doc reference happens to
            # protect them too (see retention_policy module docstring and
            # docs/artifact_retention.md Safety rule 3, the gap this closes).
            if retention_policy.is_point_in_time_capture(tree_name, run.rel):
                continue
            if newest_by_group.get(run.group_key) is run:
                continue
            age_seconds = now - run.effective_time
            if age_seconds < cutoff_seconds:
                continue
            family = artifact_family_of(run) if tree_name == "artifacts" else None
            retention_class = retention_policy.classify(tree_name, run.rel, protected=run.protected)
            candidates.append(
                PlanCandidate(
                    rel=run.rel,
                    tree=tree_name,
                    family=family,
                    size_bytes=run.size_bytes,
                    age_days=age_seconds / 86400,
                    stamp=run.stamp,
                    retention_class=retention_class.value,
                )
            )

    candidates.sort(key=lambda c: c.size_bytes, reverse=True)

    return PlanData(
        generated_at_utc=datetime.now(UTC).isoformat(),
        older_than_days=older_than_days,
        protected_refs={key: sorted(value) for key, value in protected.items()},
        candidates=candidates,
        total_bytes=sum(c.size_bytes for c in candidates),
    )


# ---------------------------------------------------------------------------
# Budget check (dry run only -- ENG-19)
# ---------------------------------------------------------------------------


@dataclass
class TreeBudgetRow:
    tree: str
    used_bytes: int
    budget_bytes: int | None
    over_budget: bool
    reclaimable_bytes: int


@dataclass
class BudgetCheckData:
    generated_at_utc: str
    multiplier: float
    older_than_days: int
    disk_total_bytes: int | None
    disk_free_bytes: int | None
    disk_path: str
    mirror_generated_utc: str | None
    mirror_manifest_found: bool
    rows: list[TreeBudgetRow]
    any_over_budget: bool


def build_budget_check(
    repo_root: Path,
    *,
    multiplier: float = retention_policy.DEFAULT_BUDGET_MULTIPLIER,
) -> BudgetCheckData:
    """Dry-run disk-budget check. Reads only -- never deletes, moves, or

    renames anything. Reuses `build_report` for per-tree used bytes and
    `build_plan` (at `retention_policy.REPRODUCIBLE_MIN_AGE_DAYS`) for the
    "what would the safe-pruning plan reclaim" figure -- the same dry-run
    candidate list `--plan` already prints, never a new deletion path.
    """

    report = build_report(repo_root)
    plan = build_plan(repo_root, older_than_days=retention_policy.REPRODUCIBLE_MIN_AGE_DAYS)

    reclaimable_by_tree: dict[str, int] = {}
    for candidate in plan.candidates:
        reclaimable_by_tree[candidate.tree] = (
            reclaimable_by_tree.get(candidate.tree, 0) + candidate.size_bytes
        )

    disk = retention_policy.measure_free_space(repo_root)
    mirror = retention_policy.read_mirror_manifest()

    rows: list[TreeBudgetRow] = []
    any_over_budget = False
    for tree_row in report.tree_rows:
        budget = retention_policy.budget_bytes_for_tree(tree_row.name, multiplier)
        over_budget = budget is not None and tree_row.total_bytes > budget
        any_over_budget = any_over_budget or over_budget
        rows.append(
            TreeBudgetRow(
                tree=tree_row.name,
                used_bytes=tree_row.total_bytes,
                budget_bytes=budget,
                over_budget=over_budget,
                reclaimable_bytes=reclaimable_by_tree.get(tree_row.name, 0),
            )
        )

    return BudgetCheckData(
        generated_at_utc=datetime.now(UTC).isoformat(),
        multiplier=multiplier,
        older_than_days=retention_policy.REPRODUCIBLE_MIN_AGE_DAYS,
        disk_total_bytes=disk.total_bytes if disk else None,
        disk_free_bytes=disk.free_bytes if disk else None,
        disk_path=str(repo_root),
        mirror_generated_utc=(mirror.get("generated_utc") if mirror else None),
        mirror_manifest_found=mirror is not None,
        rows=rows,
        any_over_budget=any_over_budget,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _row_to_dict(row: TreeSummary) -> dict[str, Any]:
    return {
        "name": row.name,
        "run_count": row.run_count,
        "total_bytes": row.total_bytes,
        "oldest_stamp": row.oldest_stamp,
        "newest_stamp": row.newest_stamp,
        "largest_file": row.largest_file_rel,
        "largest_file_bytes": row.largest_file_bytes,
        "protected_run_count": row.protected_run_count,
    }


def report_to_json(report: ReportData) -> dict[str, Any]:
    return {
        "generated_at_utc": report.generated_at_utc,
        "mode": "report",
        "protected_ref_count": len(report.protected_refs),
        "protected_refs": report.protected_refs,
        "by_top_level_tree": [_row_to_dict(row) for row in report.tree_rows],
        "by_artifact_family": [_row_to_dict(row) for row in report.family_rows],
    }


def plan_to_json(plan: PlanData) -> dict[str, Any]:
    return {
        "generated_at_utc": plan.generated_at_utc,
        "mode": "plan_dry_run_no_delete",
        "older_than_days": plan.older_than_days,
        "protected_ref_count": len(plan.protected_refs),
        "candidate_count": len(plan.candidates),
        "total_bytes": plan.total_bytes,
        "candidates": [
            {
                "rel": candidate.rel,
                "tree": candidate.tree,
                "family": candidate.family,
                "size_bytes": candidate.size_bytes,
                "age_days": round(candidate.age_days, 1),
                "stamp": candidate.stamp,
                "retention_class": candidate.retention_class,
            }
            for candidate in plan.candidates
        ],
    }


def _render_table(title: str, rows: list[TreeSummary]) -> str:
    header = (
        f"{'name':<30}{'runs':>6}{'bytes':>10}{'oldest':>18}{'newest':>18}"
        f"{'protected':>11}  largest_file (bytes)"
    )
    lines = [title, header, "-" * len(header)]
    for row in rows:
        oldest = row.oldest_stamp or "n/a"
        newest = row.newest_stamp or "n/a"
        largest = row.largest_file_rel or "n/a"
        lines.append(
            f"{row.name:<30}{row.run_count:>6}{_human_bytes(row.total_bytes):>10}"
            f"{oldest:>18}{newest:>18}{row.protected_run_count:>11}  "
            f"{largest} ({_human_bytes(row.largest_file_bytes)})"
        )
    return "\n".join(lines)


def render_report_text(report: ReportData) -> str:
    parts = [
        f"Artifact retention report -- generated {report.generated_at_utc}",
        f"Protected reference nodes discovered: {len(report.protected_refs)}",
        "",
        _render_table("By top-level tree", report.tree_rows),
        "",
        _render_table("By artifact family (artifacts/<family>)", report.family_rows),
    ]
    return "\n".join(parts)


def render_plan_text(plan: PlanData) -> str:
    lines = [
        "Artifact retention PLAN (dry run -- no delete mode exists in this tool)",
        f"generated {plan.generated_at_utc}",
        (
            f"Threshold: older than {plan.older_than_days} days, not referenced by any "
            "protected source, not the newest run of its group."
        ),
        (
            f"Candidates: {len(plan.candidates)}  Total bytes: "
            f"{_human_bytes(plan.total_bytes)} ({plan.total_bytes} bytes)"
        ),
        "",
    ]
    header = f"{'tree':<14}{'family':<30}{'age_days':>9}{'bytes':>10}  {'class':<12}rel"
    lines.append(header)
    lines.append("-" * len(header))
    for candidate in plan.candidates:
        family = candidate.family or ""
        size = _human_bytes(candidate.size_bytes)
        lines.append(
            f"{candidate.tree:<14}{family:<30}{candidate.age_days:>9.1f}{size:>10}  "
            f"{candidate.retention_class:<12}{candidate.rel}"
        )
    return "\n".join(lines)


def budget_check_to_json(check: BudgetCheckData) -> dict[str, Any]:
    return {
        "generated_at_utc": check.generated_at_utc,
        "mode": "budget_check_dry_run_no_delete",
        "multiplier": check.multiplier,
        "reclaimable_older_than_days": check.older_than_days,
        "disk": {
            "path": check.disk_path,
            "total_bytes": check.disk_total_bytes,
            "free_bytes": check.disk_free_bytes,
        },
        "mirror": {
            "manifest_found": check.mirror_manifest_found,
            "generated_utc": check.mirror_generated_utc,
        },
        "any_over_budget": check.any_over_budget,
        "trees": [
            {
                "tree": row.tree,
                "used_bytes": row.used_bytes,
                "budget_bytes": row.budget_bytes,
                "over_budget": row.over_budget,
                "reclaimable_bytes": row.reclaimable_bytes,
            }
            for row in check.rows
        ],
    }


def render_budget_check_text(check: BudgetCheckData) -> str:
    lines = [
        "Artifact retention BUDGET CHECK (dry run -- no delete mode exists in this tool)",
        f"generated {check.generated_at_utc}",
        f"Budget = {check.multiplier}x the 2026-09-04 measured baseline per tree "
        "(see docs/artifact_retention.md, src/nfl_ats/artifact_retention_policy.py).",
        (
            f"Free space at {check.disk_path}: "
            + (
                f"{_human_bytes(check.disk_free_bytes)} free of "
                f"{_human_bytes(check.disk_total_bytes)} total"
                if check.disk_free_bytes is not None and check.disk_total_bytes is not None
                else "unavailable (shutil.disk_usage failed)"
            )
        ),
        (
            f"Off-device mirror manifest: last mirrored {check.mirror_generated_utc}"
            if check.mirror_manifest_found
            else "Off-device mirror manifest: not found -- run "
            "`scripts/backup_data.py --status` to confirm coverage."
        ),
        (
            f"'Reclaimable' below is what a `--plan --older-than-days "
            f"{check.older_than_days}` dry run would list for that tree today "
            "(reproducible/scratch only -- evidence and point-in-time captures "
            "never appear there)."
        ),
        "",
    ]
    header = f"{'tree':<14}{'used':>10}{'budget':>10}{'status':>10}{'reclaimable':>13}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in check.rows:
        budget_str = _human_bytes(row.budget_bytes) if row.budget_bytes is not None else "n/a"
        status = "OVER" if row.over_budget else "under"
        lines.append(
            f"{row.tree:<14}{_human_bytes(row.used_bytes):>10}{budget_str:>10}{status:>10}"
            f"{_human_bytes(row.reclaimable_bytes):>13}"
        )
    lines.append("")
    lines.append(
        "OVER BUDGET (exit code 1)"
        if check.any_over_budget
        else "All trees under budget (exit code 0)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


READ_ONLY_SCRIPT = True
# ENG-29: read-only; the ENG-29 scanner confirms zero write sites -- it only builds a report in
# memory and prints it (--json prints to stdout), never writing under artifacts/ or registry/ (see
# the module's own read-only claim, formerly in tests/test_experiment_registry.py's allowlist).


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure artifacts/ and data/ retention (WP6 / ROADMAP OPS-02). "
            "Report-only by default; --plan lists prunable CANDIDATES but never "
            "deletes, moves, or renames anything -- there is no delete mode in "
            "this tool."
        )
    )
    parser.add_argument(
        "--report", action="store_true", help="Print the size/reference report (default)."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print prunable candidates (dry run only; nothing is deleted).",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=30,
        help="Plan-mode age threshold in days (default 30).",
    )
    parser.add_argument(
        "--budget-check",
        action="store_true",
        help=(
            "Dry-run disk-budget check (ENG-19): report per-tree used vs. budget "
            "bytes and what a --plan run would reclaim. Never deletes anything; "
            "exits non-zero if any tree is over budget."
        ),
    )
    parser.add_argument(
        "--budget-multiplier",
        type=float,
        default=retention_policy.DEFAULT_BUDGET_MULTIPLIER,
        help=(
            "Budget-check multiplier applied to the measured baseline bytes per "
            f"tree (default {retention_policy.DEFAULT_BUDGET_MULTIPLIER}; see "
            "src/nfl_ats/artifact_retention_policy.py for the derivation)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    modes_selected = sum([args.report, args.plan, args.budget_check])
    if modes_selected > 1:
        parser.error("--report, --plan, and --budget-check are mutually exclusive")

    repo_root = args.root.resolve()

    if args.plan:
        plan = build_plan(repo_root, older_than_days=args.older_than_days)
        print(json.dumps(plan_to_json(plan), indent=2) if args.json else render_plan_text(plan))
        return 0

    if args.budget_check:
        check = build_budget_check(repo_root, multiplier=args.budget_multiplier)
        print(
            json.dumps(budget_check_to_json(check), indent=2)
            if args.json
            else render_budget_check_text(check)
        )
        return 1 if check.any_over_budget else 0

    report = build_report(repo_root)
    print(json.dumps(report_to_json(report), indent=2) if args.json else render_report_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
