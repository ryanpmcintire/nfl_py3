"""Per-family confirmation-window registry (see ``docs/rotation_registry.md``).

The registry is the evaluation substrate for every research family behind it:
it hands each declared hypothesis a block of seasons it has never touched,
logs the assignment, and marks the block spent the moment a look is recorded —
whatever the verdict. Window accounting used to live in prose, which is how a
future session accidentally re-scores a spent window; here it is enforced code.

Nothing in this module scores a model. It only decides which seasons a family
is allowed to look at, and records that the look happened.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import (
    DEFAULT_MIN_CALIBRATION_GAMES,
    EARLY_SEASON_GAME_COUNT,
    MIN_FITTABLE_TRAIN_GAMES,
)
from nfl_ats.io import atomic_json
from nfl_ats.modeling import regular_season_rows

ROTATION_REGISTRY_VERSION = 1
ROTATION_REGISTRY_FILENAME = "rotation_registry.json"

# Grade determines the eligible season pool. Opener-graded confirmations need
# the paired Tuesday-opener archive, which only covers 2020-2025.
GRADE_POOLS: dict[str, tuple[int, int]] = {
    "opener": (2020, 2025),
    "close": (2009, 2025),
    "nflverse_spread": (2009, 2025),
}
DEFAULT_WINDOW_SIZE = {"opener": 2, "close": 3, "nflverse_spread": 3}
MINED_SEASONS = (2018, 2025)

# Warm-up eligibility floor (binding rule 9 in docs/rotation_registry.md).
# The feature table begins with the 2009 season, and a window's first week is
# scorable only once enough completed games sit in front of it: 500 walk-forward
# training games (outcomes.walk_forward_outcomes) before any prediction exists,
# then min_calibration_games further prediction rows before the stream can be
# calibrated (calibration.calibrate_cover_prediction_stream). Without a floor the
# pool's first block silently consumes itself as warm-up: [2009, 2011] yields 17
# scorable weeks, all in 2011, and calibration cannot run at all (the SPEC-5
# incident, 2026-08-17). Enforced at assignment; historical ledger entries are
# never re-judged, so lowering this floor cannot un-spend a window.
#
# The floor is COMPUTED from the thresholds it depends on, never written down
# as a season.
#
# Rule 9 asks one question: can the evaluation substrate SCORE a window's first
# week? That is a feasibility question, so the training term is the smallest
# training set `margin.fit_margin_model` can actually fit -- a value derived
# from that function's own preconditions -- and NOT the conservative reporting
# default in `constants.DEFAULT_MIN_TRAIN_GAMES`. Tying an irreversible
# decision to an underived default was the original defect here: 500 was never
# measured, and when it finally was (2026-08-17) the answer came back that no
# threshold exists at all. Rule 9 must not inherit a number like that.
#
# The two requirements are CHAINED, not additive: out-of-sample prediction rows
# only begin accruing once the training floor is first met, and only at week
# granularity, since a week is scorable only if every game before it clears the
# floor. Summing the two therefore under-counts by up to one partial week at
# each boundary. At the old 500 the slack absorbed that error, which is why the
# naive sum happened to be right; at 50 it does not, and the naive sum returns
# 2010 when the true answer is 2011. The `2 * GAMES_PER_EARLY_WEEK` term pays
# for both boundaries. `earliest_eligible_start_season` below computes the exact
# answer from a real schedule, and a test pins this closed form against it.
FEATURE_TABLE_START_SEASON = 2009
GAMES_PER_EARLY_WEEK = 16
WARMUP_TRAINING_GAMES = MIN_FITTABLE_TRAIN_GAMES
WARMUP_CALIBRATION_ROWS = DEFAULT_MIN_CALIBRATION_GAMES
WARMUP_REQUIRED_GAMES = WARMUP_TRAINING_GAMES + WARMUP_CALIBRATION_ROWS + 2 * GAMES_PER_EARLY_WEEK
WARMUP_PRIOR_SEASONS = math.ceil(WARMUP_REQUIRED_GAMES / EARLY_SEASON_GAME_COUNT)
MIN_ELIGIBLE_START_SEASON = FEATURE_TABLE_START_SEASON + WARMUP_PRIOR_SEASONS


MIN_WINDOW_SIZE = 2
MAX_WINDOW_SIZE = 4

FAMILY_STATUSES = ("open", "confirmed", "closed_negative", "retired")
WINDOW_STATES = ("assigned", "spent")
VERDICTS = ("confirmed", "closed_negative", "unresolved")

_TOP_LEVEL_FIELDS = frozenset({"version", "notes", "families", "season_usage"})
_FAMILY_FIELDS = frozenset(
    {
        "declared_at",
        "description",
        "grade",
        "status",
        "inherits",
        "acknowledges_mined_2018_2025",
        "windows",
    }
)
_WINDOW_FIELDS = frozenset(
    {
        "seasons",
        "state",
        "assigned_at",
        "spent_at",
        "artifact",
        "verdict",
        "probability_positive",
        "notes",
    }
)


class RegistryError(ValueError):
    """Raised when the rotation ledger is invalid or a rule would be violated.

    A ``ValueError`` subclass so the CLI reports it as a user-facing error
    rather than a traceback, matching ``DataContractError``.
    """


def earliest_eligible_start_season(
    features: pd.DataFrame,
    *,
    min_train_games: int = WARMUP_TRAINING_GAMES,
    min_calibration_rows: int = WARMUP_CALIBRATION_ROWS,
) -> int:
    """Return the earliest season whose week 1 is both scorable and calibratable.

    This is the exact form of rule 9, walked week by week over a real schedule
    rather than approximated by the season arithmetic above. A week is scorable
    once every completed game before it reaches ``min_train_games``; only from
    that point do out-of-sample prediction rows accrue, and a season's week 1 is
    eligible only once ``min_calibration_rows`` of them sit behind it.

    Raises ``RegistryError`` if no season in the frame qualifies.
    """

    frame = regular_season_rows(features)
    if frame.empty:
        raise RegistryError("Cannot compute an eligibility floor from an empty feature table")
    weeks = (
        frame.assign(gameday=pd.to_datetime(frame["gameday"], errors="raise"))
        .groupby(["season", "week"], as_index=False)
        .agg(first_day=("gameday", "min"), games=("game_id", "size"))
        .sort_values(["first_day", "season", "week"])
    )
    seasons = [int(value) for value in weeks["season"]]
    week_numbers = [int(value) for value in weeks["week"]]
    game_counts = [int(value) for value in weeks["games"]]

    completed = 0
    predictions = 0
    for season, week, games in zip(seasons, week_numbers, game_counts, strict=True):
        if week == 1 and completed >= min_train_games and predictions >= min_calibration_rows:
            return season
        if completed >= min_train_games:
            predictions += games
        completed += games
    raise RegistryError(
        f"No season satisfies the warm-up requirement "
        f"({min_train_games} training games then {min_calibration_rows} prediction rows)"
    )


@dataclass(frozen=True)
class Window:
    """One block of consecutive seasons drawn by a family."""

    seasons: tuple[int, int]
    state: str
    assigned_at: str
    spent_at: str | None = None
    artifact: str | None = None
    verdict: str | None = None
    probability_positive: float | None = None
    notes: str = ""

    @property
    def season_range(self) -> range:
        return range(self.seasons[0], self.seasons[1] + 1)


@dataclass(frozen=True)
class Family:
    """One declared research hypothesis and its window history."""

    name: str
    declared_at: str
    description: str
    grade: str
    status: str
    inherits: tuple[str, ...] = ()
    acknowledges_mined_2018_2025: bool = False
    windows: tuple[Window, ...] = ()

    @property
    def assigned_window(self) -> Window | None:
        for window in self.windows:
            if window.state == "assigned":
                return window
        return None


@dataclass(frozen=True)
class Registry:
    """The whole ledger: schema version, standing notes, and every family."""

    version: int
    notes: tuple[str, ...]
    families: dict[str, Family]


def default_registry_path() -> Path:
    """Return the tracked ledger path, honouring ``NFL_ATS_REGISTRY_DIR``."""

    return Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) / ROTATION_REGISTRY_FILENAME


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _window_from_payload(family_name: str, payload: Any) -> Window:
    if not isinstance(payload, dict):
        raise RegistryError(f"Family {family_name!r} has a non-object window entry")
    unknown = sorted(set(payload).difference(_WINDOW_FIELDS))
    if unknown:
        raise RegistryError(f"Family {family_name!r} has unknown window fields: {unknown}")
    seasons = payload.get("seasons")
    if (
        not isinstance(seasons, list)
        or len(seasons) != 2
        or not all(isinstance(season, int) for season in seasons)
    ):
        raise RegistryError(f"Family {family_name!r} has a window without [start, end] seasons")
    start, end = int(seasons[0]), int(seasons[1])
    if end < start:
        raise RegistryError(f"Family {family_name!r} has a window ending before it starts")
    state = str(payload.get("state", ""))
    if state not in WINDOW_STATES:
        raise RegistryError(f"Family {family_name!r} has an unknown window state: {state!r}")
    probability_positive = payload.get("probability_positive")
    verdict = payload.get("verdict")
    if verdict is not None and str(verdict) not in VERDICTS:
        raise RegistryError(f"Family {family_name!r} has an unknown verdict: {verdict!r}")
    return Window(
        seasons=(start, end),
        state=state,
        assigned_at=str(payload.get("assigned_at", "")),
        spent_at=None if payload.get("spent_at") is None else str(payload["spent_at"]),
        artifact=None if payload.get("artifact") is None else str(payload["artifact"]),
        verdict=None if verdict is None else str(verdict),
        probability_positive=(
            None if probability_positive is None else float(probability_positive)
        ),
        notes=str(payload.get("notes", "")),
    )


def _family_from_payload(name: str, payload: Any) -> Family:
    if not isinstance(payload, dict):
        raise RegistryError(f"Family {name!r} is not an object")
    unknown = sorted(set(payload).difference(_FAMILY_FIELDS))
    if unknown:
        raise RegistryError(f"Family {name!r} has unknown fields: {unknown}")
    grade = str(payload.get("grade", ""))
    if grade not in GRADE_POOLS:
        raise RegistryError(f"Family {name!r} has an unknown grade: {grade!r}")
    status = str(payload.get("status", ""))
    if status not in FAMILY_STATUSES:
        raise RegistryError(f"Family {name!r} has an unknown status: {status!r}")
    inherits_payload = payload.get("inherits", [])
    if not isinstance(inherits_payload, list):
        raise RegistryError(f"Family {name!r} has a non-list inherits")
    windows_payload = payload.get("windows", [])
    if not isinstance(windows_payload, list):
        raise RegistryError(f"Family {name!r} has a non-list windows")
    return Family(
        name=name,
        declared_at=str(payload.get("declared_at", "")),
        description=str(payload.get("description", "")),
        grade=grade,
        status=status,
        inherits=tuple(str(parent) for parent in inherits_payload),
        acknowledges_mined_2018_2025=bool(payload.get("acknowledges_mined_2018_2025", False)),
        windows=tuple(_window_from_payload(name, window) for window in windows_payload),
    )


def _inherited_names(registry: Registry, name: str) -> tuple[str, ...]:
    """Return the transitive ``inherits`` closure of ``name``, excluding itself."""

    seen: list[str] = []
    pending = list(registry.families[name].inherits)
    while pending:
        parent = pending.pop(0)
        if parent in seen or parent == name:
            continue
        if parent not in registry.families:
            raise RegistryError(f"Family {name!r} inherits unknown family {parent!r}")
        seen.append(parent)
        pending.extend(registry.families[parent].inherits)
    return tuple(seen)


def _chain_windows(registry: Registry, name: str) -> tuple[tuple[str, Window], ...]:
    """Return every window held or spent by ``name`` or anything it inherits."""

    rows: list[tuple[str, Window]] = [(name, window) for window in registry.families[name].windows]
    for parent in _inherited_names(registry, name):
        rows.extend((parent, window) for window in registry.families[parent].windows)
    return tuple(rows)


def _validate(registry: Registry) -> None:
    if registry.version != ROTATION_REGISTRY_VERSION:
        raise RegistryError(f"Unsupported rotation registry version: {registry.version}")
    for name, family in registry.families.items():
        pool = GRADE_POOLS[family.grade]
        assigned = [window for window in family.windows if window.state == "assigned"]
        if len(assigned) > 1:
            raise RegistryError(f"Family {name!r} holds more than one assigned window")
        for window in family.windows:
            if window.state == "spent" and not (window.artifact and window.verdict):
                raise RegistryError(
                    f"Family {name!r} has a spent window {list(window.seasons)} "
                    "without an artifact and verdict"
                )
            if window.seasons[0] < pool[0] or window.seasons[1] > pool[1]:
                raise RegistryError(
                    f"Family {name!r} window {list(window.seasons)} falls outside the "
                    f"{family.grade} pool {list(pool)}"
                )
            if _overlaps(window.seasons, MINED_SEASONS) and not (
                family.acknowledges_mined_2018_2025
            ):
                raise RegistryError(
                    f"Family {name!r} window {list(window.seasons)} intersects the mined "
                    f"{MINED_SEASONS[0]}-{MINED_SEASONS[1]} seasons without "
                    "acknowledges_mined_2018_2025"
                )
        # A family must not re-look at seasons it, or anything it inherits from,
        # has already seen. It must NOT be held responsible for two of its
        # ancestors overlapping each other: rule 4 makes windows retire
        # per-family, so independent families are explicitly allowed to draw the
        # same seasons, and two such families can both legitimately be parents.
        # Checking every pair in the chain conflated those and made an honest
        # declaration impossible -- a candidate genuinely downstream of both
        # `mod07_weak_signal_stack` and `best_pick_ranker_opener` could not name
        # both, because each spent [2020, 2021]. `eligible_blocks` has always
        # treated the chain as a union of blocked seasons; this now matches it.
        own = list(family.windows)
        inherited = [
            window
            for parent in _inherited_names(registry, name)
            for window in registry.families[parent].windows
        ]
        for index, window in enumerate(own):
            for other in own[index + 1 :]:
                if _overlaps(window.seasons, other.seasons):
                    raise RegistryError(
                        f"Family {name!r} has overlapping windows: "
                        f"{list(window.seasons)} and {list(other.seasons)}"
                    )
            for other in inherited:
                if _overlaps(window.seasons, other.seasons):
                    raise RegistryError(
                        f"Family {name!r} window {list(window.seasons)} re-looks at seasons "
                        f"already seen by its inheritance chain: {list(other.seasons)}"
                    )


def season_usage(registry: Registry) -> dict[str, int]:
    """Return the global count of families that have SPENT each season."""

    usage: dict[str, int] = {}
    for family in registry.families.values():
        seasons: set[int] = set()
        for window in family.windows:
            if window.state == "spent":
                seasons.update(window.season_range)
        for season in seasons:
            key = str(season)
            usage[key] = usage.get(key, 0) + 1
    return dict(sorted(usage.items()))


def _window_payload(window: Window) -> dict[str, Any]:
    return {
        "seasons": [window.seasons[0], window.seasons[1]],
        "state": window.state,
        "assigned_at": window.assigned_at,
        "spent_at": window.spent_at,
        "artifact": window.artifact,
        "verdict": window.verdict,
        "probability_positive": window.probability_positive,
        "notes": window.notes,
    }


def registry_payload(registry: Registry) -> dict[str, Any]:
    """Return the JSON payload for ``registry``, with ``season_usage`` recomputed."""

    return {
        "version": registry.version,
        "notes": list(registry.notes),
        "families": {
            name: {
                "declared_at": family.declared_at,
                "description": family.description,
                "grade": family.grade,
                "status": family.status,
                "inherits": list(family.inherits),
                "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
                "windows": [_window_payload(window) for window in family.windows],
            }
            for name, family in registry.families.items()
        },
        "season_usage": season_usage(registry),
    }


def registry_from_payload(payload: Any) -> Registry:
    """Parse and validate a ledger payload."""

    if not isinstance(payload, dict):
        raise RegistryError("Rotation registry must be a JSON object")
    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    if unknown:
        raise RegistryError(f"Rotation registry has unknown top-level fields: {unknown}")
    families_payload = payload.get("families", {})
    if not isinstance(families_payload, dict):
        raise RegistryError("Rotation registry families must be an object")
    notes_payload = payload.get("notes", [])
    if not isinstance(notes_payload, list):
        raise RegistryError("Rotation registry notes must be a list")
    registry = Registry(
        version=int(payload.get("version", 0)),
        notes=tuple(str(note) for note in notes_payload),
        families={
            str(name): _family_from_payload(str(name), family)
            for name, family in families_payload.items()
        },
    )
    _validate(registry)
    return registry


def load_registry(path: Path | None = None) -> Registry:
    """Read and validate the ledger; every rule violation raises ``RegistryError``."""

    destination = path or default_registry_path()
    if not destination.is_file():
        raise RegistryError(f"Rotation registry not found: {destination}")
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryError(f"Rotation registry is not valid JSON: {destination}") from error
    return registry_from_payload(payload)


def save_registry(registry: Registry, path: Path | None = None) -> None:
    """Validate, recompute ``season_usage``, and atomically rewrite the ledger."""

    _validate(registry)
    atomic_json(registry_payload(registry), path or default_registry_path())


def _replace_family(registry: Registry, family: Family) -> Registry:
    families = dict(registry.families)
    families[family.name] = family
    updated = Registry(version=registry.version, notes=registry.notes, families=families)
    _validate(updated)
    return updated


def declare_family(
    registry: Registry,
    name: str,
    *,
    description: str,
    grade: str,
    inherits: tuple[str, ...] = (),
    acknowledges_mined_2018_2025: bool = False,
) -> Registry:
    """Append a new family declaration. Declarations are append-only."""

    if not name:
        raise RegistryError("Family name is required")
    if name in registry.families:
        raise RegistryError(f"Family {name!r} is already declared; declarations are append-only")
    if not description:
        raise RegistryError("Family description is required")
    if grade not in GRADE_POOLS:
        raise RegistryError(f"Unknown grade {grade!r}; choose one of {tuple(GRADE_POOLS)}")
    unknown = sorted(set(inherits).difference(registry.families))
    if unknown:
        raise RegistryError(f"Family {name!r} inherits unknown families: {unknown}")
    family = Family(
        name=name,
        declared_at=_today(),
        description=description,
        grade=grade,
        status="open",
        inherits=tuple(inherits),
        acknowledges_mined_2018_2025=acknowledges_mined_2018_2025,
        windows=(),
    )
    return _replace_family(registry, family)


def _blocked_seasons(registry: Registry, name: str) -> tuple[tuple[int, int], ...]:
    return tuple(window.seasons for _, window in _chain_windows(registry, name))


def eligible_blocks(
    registry: Registry, name: str, *, size: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Return every block of ``size`` consecutive seasons this family may still draw."""

    family = registry.families[name]
    width = DEFAULT_WINDOW_SIZE[family.grade] if size is None else size
    pool_start, pool_end = GRADE_POOLS[family.grade]
    blocked = _blocked_seasons(registry, name)
    blocks: list[tuple[int, int]] = []
    for start in range(max(pool_start, MIN_ELIGIBLE_START_SEASON), pool_end - width + 2):
        block = (start, start + width - 1)
        if any(_overlaps(block, taken) for taken in blocked):
            continue
        if _overlaps(block, MINED_SEASONS) and not family.acknowledges_mined_2018_2025:
            continue
        blocks.append(block)
    return tuple(blocks)


def assign_window(registry: Registry, family: str, *, size: int | None = None) -> Registry:
    """Assign the earliest eligible block to ``family``.

    Deterministic given the ledger: the lowest-starting block of the requested
    size inside the grade's pool that starts at or after the warm-up floor
    (``MIN_ELIGIBLE_START_SEASON``), that neither this family nor anything it
    inherits has held or spent, and that satisfies the mined-season
    acknowledgment rule. There is no hidden choice and nothing to tune.
    """

    if family not in registry.families:
        raise RegistryError(f"Unknown family: {family!r}")
    declared = registry.families[family]
    width = DEFAULT_WINDOW_SIZE[declared.grade] if size is None else size
    if not MIN_WINDOW_SIZE <= width <= MAX_WINDOW_SIZE:
        raise RegistryError(
            f"Window size must be between {MIN_WINDOW_SIZE} and {MAX_WINDOW_SIZE}: {width}"
        )
    if declared.assigned_window is not None:
        raise RegistryError(
            f"Family {family!r} already holds an unspent window "
            f"{list(declared.assigned_window.seasons)}; record that look first"
        )
    blocks = eligible_blocks(registry, family, size=width)
    if not blocks:
        raise RegistryError(
            f"No eligible {width}-season block remains for {family!r} in the "
            f"{declared.grade} pool {list(GRADE_POOLS[declared.grade])}"
        )
    window = Window(seasons=blocks[0], state="assigned", assigned_at=_today())
    return _replace_family(registry, replace(declared, windows=(*declared.windows, window)))


def confirmation_split(
    features: pd.DataFrame, registry: Registry, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``features`` into (training, window) frames for a family's live window.

    The window frame is exactly the assigned seasons' regular-season rows; the
    training frame is every completed regular-season game whose gameday is
    strictly earlier than the window's first gameday. Forward-chaining is not
    optional here — the helper refuses to build a frame that violates it.
    """

    if family not in registry.families:
        raise RegistryError(f"Unknown family: {family!r}")
    declared = registry.families[family]
    window_entry = declared.assigned_window
    if window_entry is None:
        spent = [w for w in declared.windows if w.state == "spent"]
        if spent:
            raise RegistryError(
                f"Family {family!r} has no assigned window; its window "
                f"{list(spent[-1].seasons)} is already spent"
            )
        raise RegistryError(f"Family {family!r} has no assigned window; run `rotation assign`")

    missing_columns = sorted({"season", "gameday", "result"}.difference(features.columns))
    if missing_columns:
        raise RegistryError(f"Feature table is missing columns: {', '.join(missing_columns)}")

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    available = set(frame["season"].astype(int))
    missing_seasons = [s for s in window_entry.season_range if s not in available]
    if missing_seasons:
        raise RegistryError(
            f"Feature table is missing window seasons for {family!r}: {missing_seasons}"
        )

    window = frame.loc[frame["season"].astype(int).isin(list(window_entry.season_range))].copy()
    cutoff = window["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
    if training.empty:
        # Fail-closed backstop for ledgers written outside `assign_window`
        # (rule 9's floor guards assignment, not hand-edited history): a window
        # with nothing completed before it cannot be forward-chain scored.
        raise RegistryError(
            f"Family {family!r} window {list(window_entry.seasons)} has no completed "
            "games before it; the window cannot be scored (warm-up rule 9)"
        )
    return training, window


def record_look(
    registry: Registry,
    family: str,
    *,
    artifact: str,
    verdict: str,
    probability_positive: float | None,
    notes: str = "",
) -> Registry:
    """Mark the family's assigned window spent. A look is one look, always recorded."""

    if family not in registry.families:
        raise RegistryError(f"Unknown family: {family!r}")
    declared = registry.families[family]
    window = declared.assigned_window
    if window is None:
        raise RegistryError(f"Family {family!r} has no assigned window to record")
    if not artifact:
        raise RegistryError("A recorded look requires an artifact path")
    if verdict not in VERDICTS:
        raise RegistryError(f"Unknown verdict {verdict!r}; choose one of {VERDICTS}")
    if probability_positive is not None and not 0.0 <= probability_positive <= 1.0:
        raise RegistryError("probability_positive must lie in [0, 1]")

    spent = replace(
        window,
        state="spent",
        spent_at=_today(),
        artifact=artifact,
        verdict=verdict,
        probability_positive=probability_positive,
        notes=notes,
    )
    windows = tuple(spent if entry is window else entry for entry in declared.windows)
    # "Unresolved at this sample size" spends the window without closing the
    # family; the other two verdicts are terminal statuses.
    status = declared.status if verdict == "unresolved" else verdict
    return _replace_family(registry, replace(declared, windows=windows, status=status))


def grade_pool_capacity(registry: Registry) -> dict[str, dict[str, Any]]:
    """Report remaining unspent default-size blocks per grade pool.

    Capacity is counted globally: a default-size block is unavailable once ANY
    family holds or has spent a window intersecting it. Windows retire
    per-family, so this is a visibility number for accumulating cross-family
    multiplicity, not an eligibility check. The partition starts at the
    warm-up floor (rule 9) — seasons before it are warm-up history, not
    spendable capacity.
    """

    taken = [window.seasons for family in registry.families.values() for window in family.windows]
    capacity: dict[str, dict[str, Any]] = {}
    for grade, (pool_start, pool_end) in GRADE_POOLS.items():
        width = DEFAULT_WINDOW_SIZE[grade]
        blocks = [
            (start, start + width - 1)
            for start in range(max(pool_start, MIN_ELIGIBLE_START_SEASON), pool_end + 1, width)
            if start + width - 1 <= pool_end
        ]
        unspent = [block for block in blocks if not any(_overlaps(block, t) for t in taken)]
        capacity[grade] = {
            "seasons": [pool_start, pool_end],
            "default_window_size": width,
            "total_windows": len(blocks),
            "unspent_windows": len(unspent),
            "unspent_blocks": [list(block) for block in unspent],
        }
    return capacity


def registry_status(registry: Registry) -> dict[str, Any]:
    """Return the full status payload the CLI prints and the dashboard can read."""

    families: list[dict[str, Any]] = []
    for name in sorted(registry.families):
        family = registry.families[name]
        families.append(
            {
                "name": name,
                "grade": family.grade,
                "status": family.status,
                "declared_at": family.declared_at,
                "description": family.description,
                "inherits": list(family.inherits),
                "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
                "windows": [_window_payload(window) for window in family.windows],
                "remaining_eligible_windows": len(eligible_blocks(registry, name)),
            }
        )
    capacity = grade_pool_capacity(registry)
    return {
        "version": registry.version,
        "notes": list(registry.notes),
        "families": families,
        "grade_pools": capacity,
        "season_usage": season_usage(registry),
        "summary": [
            f"{grade} pool: {values['unspent_windows']} windows unspent"
            for grade, values in capacity.items()
        ],
    }
