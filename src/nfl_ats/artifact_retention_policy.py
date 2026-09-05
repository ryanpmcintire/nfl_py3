"""Retention classes and disk-budget policy for artifact_retention.py (ENG-19).

``scripts/artifact_retention.py`` (WP6 / OPS-02) already measures
``artifacts/`` and ``data/`` and plans (dry-run only) which unreferenced,
non-newest runs are old enough to be worth pruning eventually. What it did
not have was an explicit, named retention-CLASS contract, or any disk-budget
signal -- both required by ROADMAP ENG-19 ("Inventory ignored artifact
growth, define retention classes and safe pruning rules, and add a dry-run
budget check that never removes evidence needed by a registry row or
published forecast"). This module is that contract. It is deliberately pure
and side-effect-free apart from two read-only filesystem probes
(:func:`measure_free_space`, :func:`read_mirror_manifest`) -- no
delete/prune function exists here, matching
``scripts/artifact_retention.py``'s own no-delete-mode guarantee
(``tests/test_artifact_retention.py::test_main_has_no_delete_flag``).

The four retention classes
---------------------------
Every run ``scripts/artifact_retention.py`` discovers gets exactly one
class, via :func:`classify`:

- **evidence** -- referenced by a registry row, a doc, the active model
  manifest, ``challengers.json``, or one of the hardcoded always-protected
  paths (see ``ALWAYS_PROTECTED`` in ``scripts/artifact_retention.py``).
  Never prunable, at any age. This is precisely ``RunNode.protected`` from
  that script -- the class exists to give that boolean a name in the wider
  retention-class vocabulary the roadmap item asks for, not to recompute it.
- **point_in_time_capture** -- a raw scrape or odds snapshot: everything
  under the ``data/raw``, ``data/market``, or ``data/players`` top-level
  trees (**read**, 2026-09-04: ``data/players/**`` is entirely
  raw/timestamped-snapshot shaped -- ``participation/raw``,
  ``role_actions/raw``, ``values/raw``, ``referee_assignments/<stamp>`` --
  confirmed by listing the tree directly, no processed/derived subtree lives
  under it), plus any run elsewhere whose path contains a literal ``raw``
  segment (catches ``data/cfb/pbp/raw/...`` inside the mixed ``data/other``
  bucket, and an equivalent ``artifacts/.../raw/...`` shape should one ever
  appear). NEVER prunable, regardless of age or reference count -- these are
  scrapes of pages that have since changed and cannot be re-fetched at their
  original timestamps (see ``scripts/backup_data.py``'s module docstring,
  and ``docs/artifact_retention.md`` Safety rule 3, which flagged this exact
  gap: today's protection for ``data/raw``/``data/market`` is "not because
  of an exemption written into the code" but incidental doc references, and
  asked a future pass to make the policy intent survive a doc edit that
  thinned those references out. This module, plus the
  :func:`is_point_in_time_capture` filter now wired into
  ``scripts/artifact_retention.py``'s ``build_plan``, is that future pass.
- **scratch** -- a stray package-manager cache that ended up inside a
  scanned tree, not a research artifact: any run whose path contains a
  ``tmp``, ``uv-cache``, ``.uv-cache``, ``__pycache__``, or
  ``.pytest_cache`` segment (**measured** 2026-09-04:
  ``artifacts/.uv-cache/**`` and ``data/tmp/uv-cache/**`` are the two real
  instances found by ``--plan --older-than-days 0`` today, 3 files / 43
  bytes total). Prunable at any age -- there is nothing here a re-run of
  ``uv sync`` does not reconstruct.
- **reproducible** -- everything else: deterministic, re-derivable research
  output (margin backtests, experiment screens, ``data/processed/*.parquet``
  feature tables, ...). Prunable, but only once older than
  ``REPRODUCIBLE_MIN_AGE_DAYS`` -- see below for the derivation.

:func:`classify` never needs to know whether a candidate SHOULD be pruned
today, only which of the four buckets it falls in; ``REPRODUCIBLE_MIN_AGE_DAYS``
and the disk-budget multiplier below are the only places an age or size
threshold lives.

Why ``REPRODUCIBLE_MIN_AGE_DAYS`` = 30
----------------------------------------
The one thing that must be true before compacting a reproducible run is that
it has had a real chance to reach the off-device mirror. ``backup_data`` is
scheduled weekly, Sunday 22:00 America/New_York (**read**,
``scripts/capture_scheduler.py``, ``SCHEDULE``, the ``backup_data`` job,
cadence ``"sun"`` / ``"22:00"``), deliberately placed after the week's last
capture job so "a week's point-in-time data or artifact ledger is never left
unmirrored over the following week" (same job's own ``why`` string). One
clean cycle is 7 days; 30 days is a ~4x multiple of that cadence, so a run
stays a candidate only after roughly four weekly mirror windows have had the
chance to run -- enough slack that even two or three consecutive missed
windows (exactly the ``MISSED``-row failure mode ``AGENTS.md`` tells every
session to check for via ``capture_scheduler.py --status``) still leave at
least one completed mirror pass before the run is old enough to plan around.
This also matches the pre-existing, independently-chosen
``--older-than-days`` default already in ``scripts/artifact_retention.py``
-- the two numbers agreeing is a coincidence of both being "about a month,"
not a shared derivation, which is why this module states its own reasoning
rather than importing the script's constant.

Disk budget
-----------
``BUDGET_BASELINE_BYTES`` is the **measured** per-top-level-tree total from
``python scripts/artifact_retention.py --report --json``, run 2026-09-04
(see the dated section in ``docs/artifact_retention.md`` this baseline was
copied from). ``DEFAULT_BUDGET_MULTIPLIER`` (5.0) turns that baseline into a
budget: :func:`budget_bytes_for_tree` returns ``baseline * multiplier``.
Five times today's measured size is deliberately generous -- F: had 1051.46
GB free of 2000.38 GB total at the same measurement (**measured**,
``shutil.disk_usage``), so the six trees' combined ~32 GB budget is under 3%
of free space, and exhaustion is not the near-term risk this guards against.
What it does guard against is *ignored* growth: comparing this measurement
to the prior dated one in ``docs/artifact_retention.md`` (2026-09-01,
``artifacts`` 2.2 GB) against today's 2.53 GB gives a rough **inferred**
slope of roughly +110 MB/day for ``artifacts`` over that one 3-day window
(an extrapolation from three days of history, not a promise -- treat every
lead-time figure derived from it the same way); at that rate the
5x/~12.7 GB ``artifacts`` budget would not be crossed for roughly three
months of sustained growth at today's rate, which is enough lead time for a
``--budget-check`` run (read-only, runnable any time) to catch a runaway
family long before it threatens anything. Recompute
``BUDGET_BASELINE_BYTES`` the next time this policy is revisited; it is a
dated snapshot, not a formula.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Retention classes
# ---------------------------------------------------------------------------


class RetentionClass(StrEnum):
    """The four retention classes ENG-19 asks for. See module docstring."""

    EVIDENCE = "evidence"
    REPRODUCIBLE = "reproducible"
    SCRATCH = "scratch"
    POINT_IN_TIME_CAPTURE = "point_in_time_capture"


@dataclass(frozen=True)
class RetentionClassSpec:
    name: RetentionClass
    description: str
    prunable: bool
    # None: scratch (any age) / evidence & point-in-time-capture (never -- see `prunable`)
    min_age_days: int | None


RETENTION_CLASSES: dict[RetentionClass, RetentionClassSpec] = {
    RetentionClass.EVIDENCE: RetentionClassSpec(
        RetentionClass.EVIDENCE,
        "Referenced by a registry row, a doc, the active model manifest, "
        "challengers.json, or a hardcoded always-protected path.",
        prunable=False,
        min_age_days=None,
    ),
    RetentionClass.POINT_IN_TIME_CAPTURE: RetentionClassSpec(
        RetentionClass.POINT_IN_TIME_CAPTURE,
        "A raw scrape or odds snapshot (data/raw, data/market, data/players, "
        "or any run with a literal 'raw' path segment). Cannot be re-fetched "
        "at its original timestamp.",
        prunable=False,
        min_age_days=None,
    ),
    RetentionClass.SCRATCH: RetentionClassSpec(
        RetentionClass.SCRATCH,
        "A package-manager cache or build temp dir that ended up inside a "
        "scanned tree (tmp/, uv-cache/, .uv-cache/, __pycache__/, "
        ".pytest_cache/ path segments).",
        prunable=True,
        min_age_days=0,
    ),
    RetentionClass.REPRODUCIBLE: RetentionClassSpec(
        RetentionClass.REPRODUCIBLE,
        "Deterministic, re-derivable research output (margin backtests, "
        "experiment screens, processed feature tables) with no evidence "
        "reference.",
        prunable=True,
        min_age_days=30,
    ),
}

SCRATCH_PATH_SEGMENTS = frozenset(
    {"tmp", "uv-cache", ".uv-cache", "__pycache__", ".pytest_cache", ".agent_tmp"}
)

POINT_IN_TIME_TREES = frozenset({"data/raw", "data/market", "data/players"})

# Individual artifacts/ families that are point-in-time-capture even though
# they carry no literal "raw" path segment and live in the mixed `artifacts`
# tree. `refresh_triggers` (ENG-08, src/nfl_ats/refresh_triggers.py, added
# concurrently this session) records exactly when/why a late-week pick
# refresh fired -- that record cannot be reconstructed after the fact any
# more than a market snapshot can, so it gets the same never-prune guarantee
# as data/market/raw. Ancestor-inclusive: any run whose `rel` equals one of
# these or starts with it plus "/" matches, the same shape as the doc-
# reference protection check (`_protection_for`) in scripts/artifact_retention.py.
POINT_IN_TIME_ARTIFACT_PREFIXES = frozenset({"artifacts/refresh_triggers"})

# By contrast, `artifacts/prospective_scorecards/` (ENG-06,
# src/nfl_ats/prospective_scorecard.py, also added concurrently this
# session) is deliberately NOT listed here or in ALWAYS_PROTECTED: it holds
# derived summary scorecards computed FROM the evidence-protected
# `artifacts/prospective/` ledgers, so it is re-derivable from those ledgers
# at any time -- the textbook `reproducible` case. No code change was needed
# for it to classify that way; `classify` already falls through to
# `reproducible` for anything that is not evidence, a point-in-time capture,
# or scratch. Named here only so the omission reads as a decision, not a gap.

_reproducible_min_age = RETENTION_CLASSES[RetentionClass.REPRODUCIBLE].min_age_days
assert _reproducible_min_age is not None
REPRODUCIBLE_MIN_AGE_DAYS: int = _reproducible_min_age


def _path_segments(rel: str) -> list[str]:
    return [part for part in rel.split("/") if part]


def is_scratch(rel: str) -> bool:
    """True if any path segment of `rel` names a package-manager cache/temp dir."""

    return any(segment in SCRATCH_PATH_SEGMENTS for segment in _path_segments(rel))


def is_point_in_time_capture(tree: str, rel: str) -> bool:
    """True if `rel` (within top-level tree `tree`) is a raw scrape/snapshot.

    Whole-tree rule for `data/raw`, `data/market`, `data/players` (measured
    2026-09-04: every run discovered under these three trees today is
    already reference-protected in practice, but that protection is
    incidental -- see module docstring -- so this function makes it
    unconditional). A literal `raw` path segment catches the same shape
    nested inside the mixed `data/other` (e.g. `data/cfb/pbp/raw/...`) or
    `artifacts` buckets. `POINT_IN_TIME_ARTIFACT_PREFIXES` catches specific
    `artifacts/` families that need the same guarantee without a `raw`
    segment (currently just `refresh_triggers` -- see that constant).
    """

    if tree in POINT_IN_TIME_TREES:
        return True
    if any(
        rel == prefix or rel.startswith(prefix + "/") for prefix in POINT_IN_TIME_ARTIFACT_PREFIXES
    ):
        return True
    return "raw" in _path_segments(rel)


def classify(tree: str, rel: str, *, protected: bool) -> RetentionClass:
    """Classify one discovered run into exactly one retention class.

    `protected` is `RunNode.protected` from `scripts/artifact_retention.py`
    -- evidence status is decided there (the reference-scan machinery lives
    in that script, not here) and simply relabeled into the class
    vocabulary. Checked in this order: evidence first (being cited by a doc
    or registry is a stronger claim than a path heuristic, even for a run
    that would otherwise look like a raw capture or a scratch cache), then
    point-in-time-capture, then scratch, then reproducible as the default.
    """

    if protected:
        return RetentionClass.EVIDENCE
    if is_point_in_time_capture(tree, rel):
        return RetentionClass.POINT_IN_TIME_CAPTURE
    if is_scratch(rel):
        return RetentionClass.SCRATCH
    return RetentionClass.REPRODUCIBLE


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------

# Measured 2026-09-04 (`python scripts/artifact_retention.py --report --json`;
# see the dated section in docs/artifact_retention.md this was copied from).
# A dated snapshot, not a formula -- recompute when this policy is revisited.
BUDGET_BASELINE_BYTES: dict[str, int] = {
    "artifacts": 2_532_357_848,
    "data/raw": 841_161_362,
    "data/processed": 203_522_779,
    "data/market": 1_015_744_046,
    "data/players": 49_875_116,
    "data/other": 1_785_502_457,
}

# See module docstring, "Disk budget", for the derivation.
DEFAULT_BUDGET_MULTIPLIER = 5.0


def budget_bytes_for_tree(tree: str, multiplier: float = DEFAULT_BUDGET_MULTIPLIER) -> int | None:
    """The byte budget for one top-level tree, or None if it has no baseline."""

    baseline = BUDGET_BASELINE_BYTES.get(tree)
    if baseline is None:
        return None
    return round(baseline * multiplier)


# ---------------------------------------------------------------------------
# Read-only filesystem probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskUsage:
    total_bytes: int
    used_bytes: int
    free_bytes: int


def measure_free_space(path: Path) -> DiskUsage | None:
    """`shutil.disk_usage` for the drive holding `path`.

    Returns None on any OSError (e.g. an unmounted drive) -- this is a
    best-effort context probe for `--budget-check`, never something that
    should crash a read-only report.
    """

    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return DiskUsage(int(usage.total), int(usage.used), int(usage.free))


# The off-device mirror `scripts/backup_data.py` writes to by default
# (`DEFAULT_DESTS[0]` in that script) and the manifest file name it writes on
# every apply-mode run (`MANIFEST_NAME` in that script). Duplicated here
# rather than imported because `scripts/` is not part of the installed
# `nfl_ats` package `src/` can import from (same constraint documented by
# several `[[tool.mypy.overrides]]` entries in pyproject.toml for the reverse
# direction); kept as two named constants, not inlined, so a future rename in
# `backup_data.py` is a one-line fix here, not a re-derivation.
MIRROR_DEST_DEFAULT = Path(r"E:\nfl_data_backup")
MIRROR_MANIFEST_NAME = "backup_manifest.json"


def read_mirror_manifest(dest: Path = MIRROR_DEST_DEFAULT) -> dict[str, Any] | None:
    """Best-effort, read-only peek at the mirror's own last-run manifest.

    Returns None (never raises) when the drive is unmounted, the manifest is
    absent (mirror never run), or it fails to parse -- exactly the cases
    where `--budget-check` should still print a result, just without mirror
    context. This reads one small JSON file only; it never re-hashes the
    mirror the way `backup_data.py --status` does, so it is cheap enough to
    call on every `--budget-check` run.
    """

    manifest_path = dest / MIRROR_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None
