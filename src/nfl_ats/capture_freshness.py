"""Per-source on-disk freshness for the recurring point-in-time captures (ENG-03).

Why this exists
----------------
``scripts/capture_scheduler.py --status`` answers "did each SCHEDULED
occurrence run" -- it says nothing about whether the DATA those jobs are
supposed to produce actually exists and is recent. A job can show ``OK`` in
scheduler state while its output directory is empty (a bug in the underlying
capture script, an API outage that still exits 0, a wrong path) and
``--status`` would never notice. This module answers the complementary
question directly from disk: for each capture SOURCE, what is the newest
artifact, how old is it, and is that age inside or outside what the schedule
itself implies is normal.

Cadence derivation (not a magic number)
----------------------------------------
Every capture ``Job`` in ``scripts/capture_scheduler.SCHEDULE`` already
declares ``dedupe_dir`` -- the directory its own command writes timestamped
snapshots into, used by the scheduler's own ``already_captured`` /
``snapshot_in_window`` checks. This module groups jobs by that SAME field to
build one "source" per directory (so a schedule edit that adds, removes, or
retimes a job updates the freshness budget for free instead of drifting out
of sync with a hand-maintained table), and derives each source's expected
cadence as::

    budget_minutes = (largest gap, in minutes, between any two of that
                       source's ENABLED job occurrences over one full
                       calendar week, wrapping from the week's last
                       occurrence back to its first the following week)
                      + (the largest grace_minutes among those same jobs)

The gap term is the worst-case time the schedule itself expects this source
to go without a fresh capture if every job runs exactly on time -- the same
arithmetic the ``SCHEDULE`` docstrings already do by hand to justify each
job's own ``dedupe_minutes`` (e.g. "odds: 225 min at Sun 12:30 -> 16:15");
this module just does it once, generally, for every source, from the
schedule itself. The grace term is a lateness buffer: a job still inside its
own declared grace window is not yet a scheduling failure, so flagging its
source stale before that grace closes would be a false alarm.

A source with only one weekly job gets a 7-day gap plus its own grace --
correct, and loose on purpose: a tighter number would false-alarm at the end
of every cycle, and a false "stale" is exactly as corrosive to a monitoring
signal as a missed real one (the same trade `snapshot_in_window`'s docstring
makes in ``capture_scheduler.py`` for ``MISSED``).

Two locators, not one
----------------------
Most sources write a fresh ``YYYYMMDDTHHMMSSZ``-named snapshot directory per
capture; ``newest_snapshot_instant`` reads the newest directory NAME (never
filesystem mtime -- mtime moves for reasons that have nothing to do with
capture time: a backup restore, a file copy, an antivirus touch). One source
today, ``lineups`` (``artifacts/lineups``), instead overwrites one stable
file in place on every run (read: ``src/nfl_ats/lineup_view.py``'s
``STABLE_LINEUP_PATH`` docstring, "REPLACED on every refresh, not
accumulated") and stamps its own ``generated_at`` field inside the JSON
payload; ``newest_json_field_instant`` reads that instead, for the same
directory-name-not-mtime reason. ``JSON_FIELD_LOCATORS`` declares which
sources use which.

No dependency on ``scripts.capture_scheduler``
------------------------------------------------
This module never imports ``scripts.capture_scheduler``: a ``src/nfl_ats``
package reaching into ``scripts/`` would invert the project's layout, and
risks an import cycle the day ``capture_scheduler.py`` imports this module
back (which it does, for ``--health``). Callers pass the schedule data in,
duck-typed against ``ScheduleJob``, instead.

Consumed by ENG-14 (``nfl_ats.source_freshness_policy``)
-----------------------------------------------------------
That module's docstring names this one as its future join point: it computes
its own budgets independently today (mirroring the same gap+grace algorithm
above against a hand-copied ``SCHEDULE`` excerpt) and reads snapshots via
private ``_newest_snapshot_instant`` / ``_json_field_instant`` helpers with
the same directory-name and JSON-field semantics as this module's
``newest_snapshot_instant`` / ``newest_json_field_instant`` below -- those two
are kept public specifically so that module can import them directly instead
of keeping its own private copies, without needing anything else from this
file. This module does not modify that one; wiring is left to that module's
own owner.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

Status = Literal["fresh", "stale", "missing", "disabled"]

_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_MINUTES_PER_WEEK = 7 * 24 * 60
_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# Same pattern as scripts/capture_scheduler.SNAPSHOT_NAME, deliberately
# duplicated rather than imported (see module docstring: no dependency on
# scripts.capture_scheduler).
_SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")

#: dedupe_dir -> a short human-readable source id, used only for display /
#: JSON keys. Any dedupe_dir not listed here falls back to a slug of the path
#: itself (see `_friendly_name`), so a newly added Job with a new dedupe_dir
#: is never silently dropped from the report -- it just gets a less pretty name
#: until this table is updated.
FRIENDLY_NAMES: dict[str, str] = {
    "data/market/raw": "market_odds",
    "data/raw/nflcom_injuries": "nflcom_injuries",
    "data/raw/sportradar_injuries": "sportradar_injuries",
    "data/players/inactives": "inactives",
    "artifacts/lineups": "lineups",
    "data/players/referee_assignments": "referee_assignments",
    "data/raw/pfr_transactions": "pfr_transactions",
    "data/raw/airnow_hourly": "airnow",
    "data/raw/player_arrests": "player_arrests",
    "data/raw/public_betting_live": "public_betting",
}

#: dedupe_dir -> (relative file path under it, JSON field) for sources whose
#: job overwrites one stable artifact in place instead of writing dated
#: snapshot subdirectories -- see the module docstring's "Two locators" section.
JSON_FIELD_LOCATORS: dict[str, tuple[str, str]] = {
    "artifacts/lineups": ("current/lineups.json", "generated_at"),
}


class ScheduleJob(Protocol):
    """The subset of ``scripts.capture_scheduler.Job`` this module reads.

    A ``Protocol``, not an import of the concrete ``Job`` dataclass -- see the
    module docstring for why this module never imports
    ``scripts.capture_scheduler``. ``Job`` satisfies this structurally.

    Declared as read-only ``@property`` members, not plain annotations: mypy
    treats a plain Protocol attribute as requiring a SETTABLE variable, which
    a ``@dataclass(frozen=True)`` (``Job`` is one) never provides -- this
    module only ever reads these fields, so the Protocol should say so.
    """

    @property
    def name(self) -> str: ...
    @property
    def day(self) -> str: ...
    @property
    def at(self) -> str: ...
    @property
    def grace_minutes(self) -> int: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def season_guarded(self) -> bool: ...
    @property
    def dedupe_dir(self) -> str: ...


@dataclass(frozen=True)
class SourceFreshness:
    """One source's freshness verdict: newest artifact, age, and budget status."""

    name: str
    dedupe_dir: str
    enabled_job_count: int
    job_names: tuple[str, ...]
    newest_artifact_at: str | None  # ISO 8601, UTC
    age_minutes: float | None
    budget_minutes: float | None
    status: Status
    expected_active: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dedupe_dir": self.dedupe_dir,
            "enabled_job_count": self.enabled_job_count,
            "job_names": list(self.job_names),
            "newest_artifact_at": self.newest_artifact_at,
            "age_minutes": self.age_minutes,
            "budget_minutes": self.budget_minutes,
            "status": self.status,
            "expected_active": self.expected_active,
            "note": self.note,
        }


def _minute_of_week(day: str, at: str) -> int:
    hour, minute = (int(part) for part in at.split(":", maxsplit=1))
    return _DAY_INDEX[day] * 1440 + hour * 60 + minute


def _friendly_name(dedupe_dir: str) -> str:
    return FRIENDLY_NAMES.get(dedupe_dir, dedupe_dir.strip("/").replace("/", "_"))


def group_by_source(entries: Iterable[ScheduleJob]) -> dict[str, list[ScheduleJob]]:
    """Group schedule jobs by the on-disk directory their output shares.

    Jobs with an empty ``dedupe_dir`` (e.g. ``weekly_lock``, the ``refresh_*``
    passes, ``backup_data``) produce no artifact directory of their own and
    are excluded -- they are orchestration/derivation steps, not capture
    sources.
    """

    groups: dict[str, list[ScheduleJob]] = {}
    for job in entries:
        if not job.dedupe_dir:
            continue
        groups.setdefault(job.dedupe_dir, []).append(job)
    return groups


def derive_budget_minutes(jobs: Iterable[ScheduleJob]) -> float | None:
    """The expected-cadence budget for one source; see the module docstring.

    Returns ``None`` when no job in ``jobs`` is enabled -- there is nothing to
    derive a cadence from, and the caller should treat the source as
    ``disabled`` rather than compute a meaningless budget.
    """

    enabled = [job for job in jobs if job.enabled]
    if not enabled:
        return None
    if len(enabled) == 1:
        gap = float(_MINUTES_PER_WEEK)
    else:
        offsets = sorted(_minute_of_week(job.day, job.at) for job in enabled)
        gaps = [later - earlier for earlier, later in pairwise(offsets)]
        gaps.append(offsets[0] + _MINUTES_PER_WEEK - offsets[-1])
        gap = float(max(gaps))
    grace = float(max(job.grace_minutes for job in enabled))
    return gap + grace


def _parse_timestamp(value: str) -> datetime | None:
    """Parse the project's compact UTC stamp (``YYYYMMDDThhmmssZ``, used both
    by snapshot directory names and by the lineup payload's ``generated_at``),
    falling back to standard ISO 8601 for any other JSON-field locator."""

    match = _SNAPSHOT_NAME.match(value)
    if match:
        try:
            return datetime.strptime(match.group(1), _STAMP_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def newest_snapshot_instant(root: Path) -> datetime | None:
    """Newest ``YYYYMMDDTHHMMSSZ``-named subdirectory instant under ``root``.

    Reads the directory NAME, never filesystem mtime -- see the module
    docstring's "Two locators" section for why. Public: this is the ENG-03
    equivalent of ``nfl_ats.source_freshness_policy``'s private
    ``_newest_snapshot_instant``, kept importable for that module's own future
    join point (see this module's docstring).
    """

    if not root.is_dir():
        return None
    newest: datetime | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _SNAPSHOT_NAME.match(child.name)
        if not match:
            continue
        stamp = _parse_timestamp(match.group(1))
        if stamp is not None and (newest is None or stamp > newest):
            newest = stamp
    return newest


def newest_json_field_instant(path: Path, field: str) -> datetime | None:
    """The timestamp in ``field`` of the JSON file at ``path``, or ``None``.

    Public for the same reason as ``newest_snapshot_instant`` -- the ENG-03
    equivalent of ``nfl_ats.source_freshness_policy``'s private
    ``_json_field_instant``.
    """

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get(field) if isinstance(payload, dict) else None
    if not isinstance(value, str):
        return None
    return _parse_timestamp(value)


def newest_artifact_at(repo_root: Path, dedupe_dir: str) -> datetime | None:
    """The newest artifact instant for one source, whichever locator it uses."""

    if dedupe_dir in JSON_FIELD_LOCATORS:
        relative_path, field = JSON_FIELD_LOCATORS[dedupe_dir]
        return newest_json_field_instant(repo_root / dedupe_dir / relative_path, field)
    return newest_snapshot_instant(repo_root / dedupe_dir)


def compute_freshness(
    entries: Iterable[ScheduleJob],
    *,
    repo_root: Path,
    now: datetime,
    season_active: Callable[[datetime], bool] | None = None,
) -> list[SourceFreshness]:
    """One ``SourceFreshness`` row per distinct ``dedupe_dir`` in ``entries``.

    ``season_active``, if given, is called with ``now`` to decide
    ``expected_active`` for a source whose jobs are ALL season-guarded (e.g.
    the seven ``inactives_*`` jobs) -- a season-guarded source with nothing on
    disk during the offseason is not a scheduling failure, and the caller
    (``capture_scheduler.py``'s ``--health``) should not exit non-zero over it.
    A source with at least one non-season-guarded job (e.g. the six odds
    captures) is always ``expected_active``.
    """

    now_utc = now.astimezone(UTC)
    results: list[SourceFreshness] = []
    for dedupe_dir, jobs in sorted(group_by_source(entries).items()):
        enabled = [job for job in jobs if job.enabled]
        name = _friendly_name(dedupe_dir)
        job_names = tuple(sorted(job.name for job in jobs))

        if not enabled:
            results.append(
                SourceFreshness(
                    name=name,
                    dedupe_dir=dedupe_dir,
                    enabled_job_count=0,
                    job_names=job_names,
                    newest_artifact_at=None,
                    age_minutes=None,
                    budget_minutes=None,
                    status="disabled",
                    expected_active=False,
                    note=(
                        "every job for this source is disabled "
                        "(paused policy or missing credential)"
                    ),
                )
            )
            continue

        season_guarded_only = all(job.season_guarded for job in enabled)
        expected_active = True
        if season_guarded_only and season_active is not None:
            expected_active = season_active(now)

        newest = newest_artifact_at(repo_root, dedupe_dir)
        budget = derive_budget_minutes(jobs)
        age_minutes = (now_utc - newest).total_seconds() / 60.0 if newest is not None else None

        status: Status
        note = ""
        if newest is None:
            status = "missing"
            if not expected_active:
                note = "no artifact yet; source is season-guarded and currently offseason"
        elif budget is not None and age_minutes is not None and age_minutes > budget:
            status = "stale"
        else:
            status = "fresh"

        results.append(
            SourceFreshness(
                name=name,
                dedupe_dir=dedupe_dir,
                enabled_job_count=len(enabled),
                job_names=job_names,
                newest_artifact_at=newest.isoformat() if newest is not None else None,
                age_minutes=round(age_minutes, 1) if age_minutes is not None else None,
                budget_minutes=round(budget, 1) if budget is not None else None,
                status=status,
                expected_active=expected_active,
                note=note,
            )
        )
    return results


def any_unexpected_missing(sources: Iterable[SourceFreshness]) -> bool:
    """True if a source that should currently be producing data has nothing."""

    return any(source.status == "missing" and source.expected_active for source in sources)


def render_table(sources: Iterable[SourceFreshness]) -> str:
    """A fixed-width text table, used by ``capture_scheduler.py --health``."""

    rows = list(sources)
    if not rows:
        return "  (no sources found in the schedule)"
    width = max(len(source.name) for source in rows)
    lines = []
    for source in rows:
        age = f"{source.age_minutes:>9.1f}m" if source.age_minutes is not None else "      n/a"
        budget_text = (
            f"{source.budget_minutes:>9.1f}m" if source.budget_minutes is not None else "      n/a"
        )
        marker = {
            "fresh": "ok ",
            "stale": "!! ",
            "missing": "!! ",
            "disabled": "-- ",
        }[source.status]
        flag = "" if source.expected_active or source.status == "disabled" else "  (offseason)"
        line = (
            f"  {marker}{source.name:<{width}}  age {age}  budget {budget_text}  "
            f"[{source.status}]{flag}"
        )
        if source.note:
            line += f"  ({source.note})"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "FRIENDLY_NAMES",
    "JSON_FIELD_LOCATORS",
    "ScheduleJob",
    "SourceFreshness",
    "any_unexpected_missing",
    "compute_freshness",
    "derive_budget_minutes",
    "group_by_source",
    "newest_artifact_at",
    "newest_json_field_instant",
    "newest_snapshot_instant",
    "render_table",
]
