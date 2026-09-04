"""Week-1 pool observables capture (LEAD-52, POL-05 input).

The pool's field size and prize structure decide every contest-utility
question, and both are observable only once the pool is entered; each
game's pick distribution unlocks at its own kickoff. There is no API
(the pool needs a login), so this module is a MANUAL-entry capture path:
values read off the pool pages are written into immutable UTC-stamped
snapshot directories with SHA-256 manifests, exactly like every other
ingest in this repo. It never touches a ledger, a forecast, or a model.

Two record types, one directory each:

- field/prive facts: ``entries``, ``paid_places``, free-text prize
  notes (recorded once per week, at entry time);
- unlocked distributions: per-game home/away pick shares with the
  unlock instant (recorded any time after each game's kickoff).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.io import atomic_json
from nfl_ats.provenance import sha256_file

#: ``2026_01_MIA_LV``-shaped nflverse game ids, the only game key this
#: capture accepts (a typo'd id would silently orphan the row).
GAME_ID_RE = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")

#: Pick-share rounding tolerance: pool pages show whole percents, so two
#: shares may miss 1.0 by a point of rounding without failing the row.
SHARE_SUM_TOLERANCE = 0.02

OBSERVABLES_DIRNAME = "pool_observables"


class PoolObservableError(ValueError):
    """A refused manual observation (fail closed, never a partial row)."""


@dataclass(frozen=True)
class FieldObservation:
    season: int
    week: int
    entries: int
    paid_places: int
    prize_notes: str
    observed_at_utc: str
    observer: str


@dataclass(frozen=True)
class DistributionObservation:
    season: int
    week: int
    game_id: str
    home_share: float
    away_share: float
    unlocked_at_utc: str
    observed_at_utc: str
    observer: str


def _parse_instant(value: str, *, field: str) -> datetime:
    try:
        instant = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise PoolObservableError(f"{field} is not ISO-8601: {value!r}") from error
    if instant.tzinfo is None:
        raise PoolObservableError(f"{field} needs an explicit timezone: {value!r}")
    return instant.astimezone(UTC)


def _validate_field(observation: FieldObservation) -> None:
    if observation.season < 2000 or observation.week < 1 or observation.week > 22:
        raise PoolObservableError(f"implausible season/week: {observation}")
    if observation.entries < 2:
        raise PoolObservableError(f"entries must be at least 2: {observation.entries}")
    if not 1 <= observation.paid_places <= observation.entries:
        raise PoolObservableError(
            f"paid_places must sit inside [1, entries]: {observation.paid_places}"
        )
    if not observation.prize_notes.strip():
        raise PoolObservableError("prize_notes must say what the payout is")
    if not observation.observer.strip():
        raise PoolObservableError("observer must name who read the pool page")
    _parse_instant(observation.observed_at_utc, field="observed_at_utc")


def _validate_distribution(observation: DistributionObservation) -> None:
    if observation.season < 2000 or observation.week < 1 or observation.week > 22:
        raise PoolObservableError(f"implausible season/week: {observation}")
    if not GAME_ID_RE.match(observation.game_id):
        raise PoolObservableError(f"not a canonical game id: {observation.game_id!r}")
    for side, share in (
        ("home_share", observation.home_share),
        ("away_share", observation.away_share),
    ):
        if not 0.0 <= share <= 1.0:
            raise PoolObservableError(f"{side} outside [0, 1]: {share}")
    total = observation.home_share + observation.away_share
    if abs(total - 1.0) > SHARE_SUM_TOLERANCE:
        raise PoolObservableError(f"shares sum to {total}, outside tolerance of 1.0")
    if not observation.observer.strip():
        raise PoolObservableError("observer must name who read the pool page")
    unlocked = _parse_instant(observation.unlocked_at_utc, field="unlocked_at_utc")
    observed = _parse_instant(observation.observed_at_utc, field="observed_at_utc")
    # A distribution cannot be recorded before the game unlocks it: the
    # whole point of POL-04's closure is that pre-kickoff fields are
    # unobservable, so a row claiming otherwise is rejected, not stored.
    if observed < unlocked:
        raise PoolObservableError("observed_at_utc precedes unlocked_at_utc")


def _write_snapshot(
    data_root: Path,
    *,
    season: int,
    week: int,
    record: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = instant.strftime("%Y%m%dT%H%M%SZ")
    directory = data_root / OBSERVABLES_DIRNAME / stamp
    if directory.exists():
        raise PoolObservableError(f"snapshot directory already exists: {directory}")
    directory.mkdir(parents=True)
    payload = {
        **record,
        "season": season,
        "week": week,
        "recorded_at_utc": instant.isoformat(),
        "source": "manual Splash Sports pool-page entry",
    }
    atomic_json(payload, directory / "observations.json")
    manifest = {
        "created_at_utc": instant.isoformat(),
        "season": season,
        "week": week,
        "files": [
            {"name": "observations.json", "sha256": sha256_file(directory / "observations.json")}
        ],
    }
    atomic_json(manifest, directory / "manifest.json")
    return {"directory": str(directory), "manifest": manifest}


def record_field_observation(
    data_root: Path, observation: FieldObservation, *, now: datetime | None = None
) -> dict[str, Any]:
    """Validate and freeze one week's field/prize facts."""

    _validate_field(observation)
    return _write_snapshot(
        data_root,
        season=observation.season,
        week=observation.week,
        record={"kind": "field", "observation": asdict(observation)},
        now=now,
    )


def record_distribution(
    data_root: Path, observation: DistributionObservation, *, now: datetime | None = None
) -> dict[str, Any]:
    """Validate and freeze one game's unlocked pick distribution."""

    _validate_distribution(observation)
    return _write_snapshot(
        data_root,
        season=observation.season,
        week=observation.week,
        record={"kind": "distribution", "observation": asdict(observation)},
        now=now,
    )


def latest_snapshots(data_root: Path) -> list[dict[str, Any]]:
    """Read-only inventory of captured snapshots (no evaluation)."""

    root = data_root / OBSERVABLES_DIRNAME
    if not root.is_dir():
        return []
    rows = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            import json

            rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return rows


__all__ = [
    "GAME_ID_RE",
    "OBSERVABLES_DIRNAME",
    "DistributionObservation",
    "FieldObservation",
    "PoolObservableError",
    "latest_snapshots",
    "record_distribution",
    "record_field_observation",
]
