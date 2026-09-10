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

_SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")

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

JSON_FIELD_LOCATORS: dict[str, tuple[str, str]] = {
    "artifacts/lineups": ("current/lineups.json", "generated_at"),
}


class ScheduleJob(Protocol):
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
    name: str
    dedupe_dir: str
    enabled_job_count: int
    job_names: tuple[str, ...]
    newest_artifact_at: str | None
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

    groups: dict[str, list[ScheduleJob]] = {}
    for job in entries:
        if not job.dedupe_dir:
            continue
        groups.setdefault(job.dedupe_dir, []).append(job)
    return groups


def derive_budget_minutes(jobs: Iterable[ScheduleJob]) -> float | None:

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

    return any(source.status == "missing" and source.expected_active for source in sources)


def render_table(sources: Iterable[SourceFreshness]) -> str:

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
