from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats import weak_signals
from nfl_ats.constants import (
    DEFAULT_MIN_CALIBRATION_GAMES,
    EARLY_SEASON_GAME_COUNT,
    MIN_FITTABLE_TRAIN_GAMES,
)
from nfl_ats.io import atomic_json
from nfl_ats.modeling import regular_season_rows

ROTATION_REGISTRY_VERSION = 1
ROTATION_REGISTRY_FILENAME = "rotation_registry.json"

GRADE_POOLS: dict[str, tuple[int, int]] = {
    "opener": (2020, 2025),
    "close": (2009, 2025),
    "nflverse_spread": (2009, 2025),
}
DEFAULT_WINDOW_SIZE = {"opener": 2, "close": 3, "nflverse_spread": 3}
MINED_SEASONS = (2018, 2025)
_MINED_SEASON_SET = frozenset(range(MINED_SEASONS[0], MINED_SEASONS[1] + 1))

FEATURE_TABLE_START_SEASON = 2009
GAMES_PER_EARLY_WEEK = 16
WARMUP_TRAINING_GAMES = MIN_FITTABLE_TRAIN_GAMES
WARMUP_CALIBRATION_ROWS = DEFAULT_MIN_CALIBRATION_GAMES
WARMUP_REQUIRED_GAMES = WARMUP_TRAINING_GAMES + WARMUP_CALIBRATION_ROWS + 2 * GAMES_PER_EARLY_WEEK
WARMUP_PRIOR_SEASONS = math.ceil(WARMUP_REQUIRED_GAMES / EARLY_SEASON_GAME_COUNT)
MIN_ELIGIBLE_START_SEASON = FEATURE_TABLE_START_SEASON + WARMUP_PRIOR_SEASONS


MIN_WINDOW_SIZE = 2
MAX_WINDOW_SIZE = 4

_WINDOW_KINDS = ("contiguous", "stratified")
STRATIFIED_GRADE = "close"
STRATIFIED_LEG_COUNT = 2

COVERAGE_STUB_STATUS = "declared_for_coverage"
COVERAGE_STUB_GRADE = "close"

FAMILY_STATUSES = ("open", "confirmed", "closed_negative", "retired", COVERAGE_STUB_STATUS)
WINDOW_STATES = ("assigned", "spent")
VERDICTS = ("confirmed", "closed_negative", "unresolved")

_TOP_LEVEL_FIELDS = frozenset(
    {"version", "notes", "families", "season_usage", "no_rotation_needed"}
)
_FAMILY_FIELDS = frozenset(
    {
        "declared_at",
        "description",
        "grade",
        "status",
        "inherits",
        "acknowledges_mined_2018_2025",
        "windows",
        "coverage_weak_signal_family",
        "coverage_league",
        "coverage_effect_units",
        "plain_summary",
    }
)
_WINDOW_FIELDS = frozenset(
    {
        "seasons",
        "state",
        "window_kind",
        "assigned_at",
        "spent_at",
        "artifact",
        "verdict",
        "closing_ground",
        "probability_positive",
        "effect",
        "effect_units",
        "interval",
        "standard_error",
        "sample_blocks",
        "leg_effects",
        "notes",
    }
)

_TERMINAL_VERDICT_GROUNDS = tuple(
    ground for grounds in weak_signals.CLOSING_GROUNDS.values() for ground in grounds
)


NO_ROTATION_FIXED_REASONS = (
    "reliability_measurement",
    "positive_control",
    "oracle",
    "retired_profile",
    "cfb_out_of_scope",
)
_DECOMPOSITION_PARENT_PREFIX = "decomposition_of_parent:"


def _is_admissible_no_rotation_reason(reason: str) -> bool:

    if reason in NO_ROTATION_FIXED_REASONS:
        return True
    return reason.startswith(_DECOMPOSITION_PARENT_PREFIX) and len(reason) > len(
        _DECOMPOSITION_PARENT_PREFIX
    )


def classify_no_rotation_reason(
    weak_signal_family: str, category: str | None, *, league: str = "nfl"
) -> str | None:

    if league != "nfl":
        return "cfb_out_of_scope"
    name = weak_signal_family.lower()
    if "oracle" in name:
        return "oracle"
    if "reliability" in name:
        return "reliability_measurement"
    if "retired" in name:
        return "retired_profile"
    if category == "control":
        return "positive_control"
    return None


@dataclass(frozen=True)
class NoRotationRecord:
    weak_signal_family: str
    league: str
    reason: str
    declared_at: str
    effect_units: tuple[str, ...] = ()
    notes: str = ""


_NO_ROTATION_FIELDS = frozenset(
    {"weak_signal_family", "league", "reason", "declared_at", "effect_units", "notes"}
)


def _no_rotation_record_from_payload(
    key: str, payload: Any, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> NoRotationRecord:
    if not isinstance(payload, dict):
        raise RegistryError(f"no_rotation_needed entry {key!r} is not an object")
    unknown = sorted(set(payload).difference(_NO_ROTATION_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"no_rotation_needed entry {key!r} has unknown fields: {unknown}",
            on_unknown_field=on_unknown_field,
        )
    weak_signal_family = str(payload.get("weak_signal_family", ""))
    if not weak_signal_family:
        raise RegistryError(f"no_rotation_needed entry {key!r} is missing weak_signal_family")
    league = str(payload.get("league", ""))
    if league not in weak_signals.LEAGUES:
        raise RegistryError(f"no_rotation_needed entry {key!r} has unknown league {league!r}")
    reason = str(payload.get("reason", ""))
    if not _is_admissible_no_rotation_reason(reason):
        raise RegistryError(
            f"no_rotation_needed entry {key!r} has inadmissible reason {reason!r}; expected "
            f"one of {NO_ROTATION_FIXED_REASONS} or '{_DECOMPOSITION_PARENT_PREFIX}<family>'"
        )
    effect_units_payload = payload.get("effect_units", [])
    if not isinstance(effect_units_payload, list):
        raise RegistryError(f"no_rotation_needed entry {key!r} has non-list effect_units")
    return NoRotationRecord(
        weak_signal_family=weak_signal_family,
        league=league,
        reason=reason,
        declared_at=str(payload.get("declared_at", "")),
        effect_units=tuple(str(unit) for unit in effect_units_payload),
        notes=str(payload.get("notes", "")),
    )


class RegistryError(ValueError):
    pass


def _handle_unknown_fields(message: str, *, on_unknown_field: weak_signals.OnUnknownField) -> None:

    if on_unknown_field == "raise":
        raise RegistryError(message)
    warnings.warn(
        f"{message} -- ignored so this cannot abort a site build; add the field "
        "to the matching allowlist in rotation.py to stop seeing this warning",
        weak_signals.UnknownRegistryFieldWarning,
        stacklevel=3,
    )


def earliest_eligible_start_season(
    features: pd.DataFrame,
    *,
    min_train_games: int = WARMUP_TRAINING_GAMES,
    min_calibration_rows: int = WARMUP_CALIBRATION_ROWS,
) -> int:

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
class LegResult:
    season: int
    effect: float
    probability_positive: float | None = None
    sample_blocks: int | None = None


@dataclass(frozen=True)
class Window:
    seasons: tuple[int, int]
    state: str
    assigned_at: str
    window_kind: str = "contiguous"
    spent_at: str | None = None
    artifact: str | None = None
    verdict: str | None = None
    closing_ground: str | None = None
    probability_positive: float | None = None
    effect: float | None = None
    effect_units: str | None = None
    interval: tuple[float, float] | None = None
    standard_error: float | None = None
    sample_blocks: int | None = None
    leg_effects: tuple[LegResult, ...] | None = None
    notes: str = ""

    @property
    def season_range(self) -> range:

        if self.window_kind == "stratified":
            raise RegistryError(
                "season_range is undefined for a stratified window; the span "
                "between its legs was never looked at. Use covered_seasons."
            )
        return range(self.seasons[0], self.seasons[1] + 1)

    @property
    def covered_seasons(self) -> tuple[int, ...]:

        if self.window_kind == "stratified":
            return tuple(sorted(self.seasons))
        return tuple(range(self.seasons[0], self.seasons[1] + 1))


@dataclass(frozen=True)
class Family:
    name: str
    declared_at: str
    description: str
    grade: str
    status: str
    inherits: tuple[str, ...] = ()
    acknowledges_mined_2018_2025: bool = False
    windows: tuple[Window, ...] = ()
    coverage_weak_signal_family: str | None = None
    coverage_league: str | None = None
    coverage_effect_units: tuple[str, ...] = ()
    plain_summary: str | None = None

    @property
    def assigned_window(self) -> Window | None:
        for window in self.windows:
            if window.state == "assigned":
                return window
        return None


@dataclass(frozen=True)
class Registry:
    version: int
    notes: tuple[str, ...]
    families: dict[str, Family]
    no_rotation_needed: dict[str, NoRotationRecord] = field(default_factory=dict)


def default_registry_path() -> Path:

    return Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) / ROTATION_REGISTRY_FILENAME


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:

    return left[0] <= right[1] and right[0] <= left[1]


def _windows_overlap(left: Window, right: Window) -> bool:

    return bool(set(left.covered_seasons) & set(right.covered_seasons))


def _validate_closing_ground(
    context: str,
    *,
    verdict: str | None,
    closing_ground: str | None,
    probability_positive: float | None,
) -> None:

    if verdict == "closed_negative":
        if closing_ground not in _TERMINAL_VERDICT_GROUNDS:
            raise RegistryError(
                f"{context}: a closed_negative verdict must name an admissible "
                f"closing_ground ({', '.join(_TERMINAL_VERDICT_GROUNDS)}). An "
                "interval containing zero is NOT one of them; that verdict is "
                "'unresolved' (AGENTS.md, binding)"
            )
        if probability_positive is None:
            raise RegistryError(
                f"{context}: a closed_negative verdict requires "
                "probability_positive — continuous evidence, never bare "
                "pass/fail (AGENTS.md, binding)"
            )
    elif closing_ground is not None:
        raise RegistryError(
            f"{context}: verdict {verdict!r} is not a closure and cannot carry "
            f"closing_ground {closing_ground!r}"
        )


def _validate_effect_fields(
    context: str,
    *,
    effect: Any,
    effect_units: Any,
    interval: Any,
    standard_error: Any,
    sample_blocks: Any,
) -> tuple[float | None, str | None, tuple[float, float] | None, float | None, int | None]:

    resolved_effect = None if effect is None else float(effect)
    resolved_units = None if effect_units is None else str(effect_units)
    if (resolved_effect is None) != (resolved_units is None):
        raise RegistryError(
            f"{context}: effect and effect_units must be given together, or not at all"
        )
    if resolved_effect is not None and not math.isfinite(resolved_effect):
        raise RegistryError(f"{context}: effect must be finite")
    if resolved_units is not None and resolved_units not in weak_signals.EFFECT_UNITS:
        raise RegistryError(
            f"{context}: unknown effect_units {resolved_units!r}; "
            f"expected one of {', '.join(weak_signals.EFFECT_UNITS)}"
        )

    resolved_interval: tuple[float, float] | None = None
    if interval is not None:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise RegistryError(f"{context}: interval must be a two-element [low, high]")
        low, high = float(interval[0]), float(interval[1])
        if low > high:
            raise RegistryError(f"{context}: interval has low > high")
        resolved_interval = (low, high)

    resolved_standard_error = None if standard_error is None else float(standard_error)
    if resolved_standard_error is not None and not resolved_standard_error > 0.0:
        raise RegistryError(f"{context}: standard_error must be positive")

    resolved_sample_blocks = None if sample_blocks is None else int(sample_blocks)

    return (
        resolved_effect,
        resolved_units,
        resolved_interval,
        resolved_standard_error,
        resolved_sample_blocks,
    )


def _validate_plain_summary(context: str, plain_summary: Any) -> str | None:

    if plain_summary is None:
        return None
    if not isinstance(plain_summary, str):
        raise RegistryError(f"{context}: plain_summary must be a string sentence")
    text = plain_summary.strip()
    if not text:
        raise RegistryError(f"{context}: plain_summary, if given, must be a non-empty sentence")
    if " " not in text or text[-1] not in ".!?":
        raise RegistryError(
            f"{context}: plain_summary must read as a sentence (more than one word, ending in "
            f"'.', '!', or '?'); got {plain_summary!r}"
        )
    return text


_LEG_RESULT_FIELDS = frozenset({"season", "effect", "probability_positive", "sample_blocks"})


def _leg_result_from_payload(
    context: str, payload: Any, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> LegResult:
    if not isinstance(payload, dict):
        raise RegistryError(f"{context}: leg_effects entries must be objects")
    unknown = sorted(set(payload).difference(_LEG_RESULT_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"{context}: unknown leg_effects fields: {unknown}",
            on_unknown_field=on_unknown_field,
        )
    if "season" not in payload or "effect" not in payload:
        raise RegistryError(f"{context}: a leg_effects entry requires season and effect")
    season = int(payload["season"])
    effect = float(payload["effect"])
    if not math.isfinite(effect):
        raise RegistryError(f"{context}: leg_effects effect must be finite")
    probability_positive = payload.get("probability_positive")
    if probability_positive is not None:
        probability_positive = float(probability_positive)
        if not 0.0 <= probability_positive <= 1.0:
            raise RegistryError(f"{context}: leg_effects probability_positive must lie in [0, 1]")
    sample_blocks = payload.get("sample_blocks")
    sample_blocks = None if sample_blocks is None else int(sample_blocks)
    return LegResult(
        season=season,
        effect=effect,
        probability_positive=probability_positive,
        sample_blocks=sample_blocks,
    )


def _validate_leg_effects(
    context: str,
    *,
    window_kind: str,
    seasons: tuple[int, ...],
    payload: Any,
    on_unknown_field: weak_signals.OnUnknownField = "raise",
) -> tuple[LegResult, ...] | None:

    if payload is None:
        return None
    if window_kind != "stratified":
        raise RegistryError(f"{context}: leg_effects is only meaningful on a stratified window")
    if not isinstance(payload, list):
        raise RegistryError(f"{context}: leg_effects must be a list")
    results = tuple(
        _leg_result_from_payload(context, entry, on_unknown_field=on_unknown_field)
        for entry in payload
    )
    result_seasons = sorted(result.season for result in results)
    expected_seasons = sorted(seasons)
    if result_seasons != expected_seasons:
        raise RegistryError(
            f"{context}: leg_effects must report exactly one result per leg "
            f"{expected_seasons}, got {result_seasons}"
        )
    return results


def _window_from_payload(
    family_name: str, payload: Any, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> Window:
    if not isinstance(payload, dict):
        raise RegistryError(f"Family {family_name!r} has a non-object window entry")
    unknown = sorted(set(payload).difference(_WINDOW_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Family {family_name!r} has unknown window fields: {unknown}",
            on_unknown_field=on_unknown_field,
        )
    window_kind = str(payload.get("window_kind", "contiguous"))
    if window_kind not in _WINDOW_KINDS:
        raise RegistryError(f"Family {family_name!r} has an unknown window_kind: {window_kind!r}")
    seasons = payload.get("seasons")
    if (
        not isinstance(seasons, list)
        or not seasons
        or not all(isinstance(season, int) for season in seasons)
    ):
        raise RegistryError(f"Family {family_name!r} has a window with a malformed seasons list")
    if window_kind == "stratified":
        if len(seasons) != STRATIFIED_LEG_COUNT or len(set(seasons)) != STRATIFIED_LEG_COUNT:
            raise RegistryError(
                f"Family {family_name!r} has a stratified window that does not name "
                f"exactly {STRATIFIED_LEG_COUNT} distinct leg seasons: {seasons}"
            )
        start, end = sorted(int(season) for season in seasons)
    else:
        if len(seasons) != 2:
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
    closing_ground = payload.get("closing_ground")
    if closing_ground is not None:
        if str(verdict) != "closed_negative":
            raise RegistryError(
                f"Family {family_name!r} window {seasons}: verdict {verdict!r} is "
                f"not a closure and cannot carry closing_ground {closing_ground!r}"
            )
        if str(closing_ground) not in _TERMINAL_VERDICT_GROUNDS:
            raise RegistryError(
                f"Family {family_name!r} window {seasons}: unknown closing_ground "
                f"{closing_ground!r}; choose one of {_TERMINAL_VERDICT_GROUNDS}"
            )
    effect, effect_units, interval, standard_error, sample_blocks = _validate_effect_fields(
        f"Family {family_name!r} window {seasons}",
        effect=payload.get("effect"),
        effect_units=payload.get("effect_units"),
        interval=payload.get("interval"),
        standard_error=payload.get("standard_error"),
        sample_blocks=payload.get("sample_blocks"),
    )
    leg_effects = _validate_leg_effects(
        f"Family {family_name!r} window {seasons}",
        window_kind=window_kind,
        seasons=(start, end),
        payload=payload.get("leg_effects"),
        on_unknown_field=on_unknown_field,
    )
    return Window(
        seasons=(start, end),
        state=state,
        window_kind=window_kind,
        assigned_at=str(payload.get("assigned_at", "")),
        spent_at=None if payload.get("spent_at") is None else str(payload["spent_at"]),
        artifact=None if payload.get("artifact") is None else str(payload["artifact"]),
        verdict=None if verdict is None else str(verdict),
        closing_ground=None if closing_ground is None else str(closing_ground),
        probability_positive=(
            None if probability_positive is None else float(probability_positive)
        ),
        effect=effect,
        effect_units=effect_units,
        interval=interval,
        standard_error=standard_error,
        sample_blocks=sample_blocks,
        leg_effects=leg_effects,
        notes=str(payload.get("notes", "")),
    )


def _family_from_payload(
    name: str, payload: Any, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> Family:
    if not isinstance(payload, dict):
        raise RegistryError(f"Family {name!r} is not an object")
    unknown = sorted(set(payload).difference(_FAMILY_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Family {name!r} has unknown fields: {unknown}",
            on_unknown_field=on_unknown_field,
        )
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
    coverage_family = payload.get("coverage_weak_signal_family")
    coverage_league = payload.get("coverage_league")
    if coverage_league is not None and coverage_league not in weak_signals.LEAGUES:
        raise RegistryError(f"Family {name!r} has unknown coverage_league {coverage_league!r}")
    coverage_effect_units_payload = payload.get("coverage_effect_units", [])
    if not isinstance(coverage_effect_units_payload, list):
        raise RegistryError(f"Family {name!r} has non-list coverage_effect_units")
    plain_summary = _validate_plain_summary(f"Family {name!r}", payload.get("plain_summary"))
    return Family(
        name=name,
        declared_at=str(payload.get("declared_at", "")),
        description=str(payload.get("description", "")),
        grade=grade,
        status=status,
        inherits=tuple(str(parent) for parent in inherits_payload),
        acknowledges_mined_2018_2025=bool(payload.get("acknowledges_mined_2018_2025", False)),
        windows=tuple(
            _window_from_payload(name, window, on_unknown_field=on_unknown_field)
            for window in windows_payload
        ),
        coverage_weak_signal_family=(None if coverage_family is None else str(coverage_family)),
        coverage_league=None if coverage_league is None else str(coverage_league),
        coverage_effect_units=tuple(str(unit) for unit in coverage_effect_units_payload),
        plain_summary=plain_summary,
    )


def _inherited_names(registry: Registry, name: str) -> tuple[str, ...]:

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

    rows: list[tuple[str, Window]] = [(name, window) for window in registry.families[name].windows]
    for parent in _inherited_names(registry, name):
        rows.extend((parent, window) for window in registry.families[parent].windows)
    return tuple(rows)


def _validate(registry: Registry) -> None:
    if registry.version != ROTATION_REGISTRY_VERSION:
        raise RegistryError(f"Unsupported rotation registry version: {registry.version}")
    for name, family in registry.families.items():
        _validate_plain_summary(f"Family {name!r}", family.plain_summary)
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
            if (
                window.window_kind == "stratified"
                and window.state == "spent"
                and not window.leg_effects
            ):
                raise RegistryError(
                    f"Family {name!r} has a spent stratified window "
                    f"{list(window.seasons)} without per-leg magnitudes (leg_effects)"
                )
            if window.seasons[0] < pool[0] or window.seasons[1] > pool[1]:
                raise RegistryError(
                    f"Family {name!r} window {list(window.seasons)} falls outside the "
                    f"{family.grade} pool {list(pool)}"
                )
            if window.window_kind == "stratified" and family.grade != STRATIFIED_GRADE:
                raise RegistryError(
                    f"Family {name!r} window {list(window.seasons)} is stratified but "
                    f"grade is {family.grade!r}; stratified windows are "
                    f"{STRATIFIED_GRADE}-graded only "
                    "(docs/era_stratified_windows_proposal.md scope limit)"
                )
            if set(window.covered_seasons) & _MINED_SEASON_SET and not (
                family.acknowledges_mined_2018_2025
            ):
                raise RegistryError(
                    f"Family {name!r} window {list(window.seasons)} intersects the mined "
                    f"{MINED_SEASONS[0]}-{MINED_SEASONS[1]} seasons without "
                    "acknowledges_mined_2018_2025"
                )
        own = list(family.windows)
        inherited = [
            window
            for parent in _inherited_names(registry, name)
            for window in registry.families[parent].windows
        ]
        for index, window in enumerate(own):
            for other in own[index + 1 :]:
                if _windows_overlap(window, other):
                    raise RegistryError(
                        f"Family {name!r} has overlapping windows: "
                        f"{list(window.seasons)} and {list(other.seasons)}"
                    )
            for other in inherited:
                if _windows_overlap(window, other):
                    raise RegistryError(
                        f"Family {name!r} window {list(window.seasons)} re-looks at seasons "
                        f"already seen by its inheritance chain: {list(other.seasons)}"
                    )


VALIDATOR_INTRODUCED_AT = "2026-09-04"

GRANDFATHERED_WIDTH_VIOLATIONS: dict[str, tuple[int, int]] = {
    "pbp_drive_bundle": (2013, 2017),
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    family: str | None
    message: str


def validate_registry(registry: Registry) -> list[Issue]:

    issues: list[Issue] = []
    for name, family in sorted(registry.families.items()):
        try:
            _validate_plain_summary(f"Family {name!r}", family.plain_summary)
        except RegistryError as exc:
            issues.append(Issue("error", "invalid_plain_summary", name, str(exc)))
        own_windows = list(family.windows)
        for window in own_windows:
            if window.window_kind == "contiguous":
                width = window.seasons[1] - window.seasons[0] + 1
                if not (MIN_WINDOW_SIZE <= width <= MAX_WINDOW_SIZE):
                    grandfathered = (
                        GRANDFATHERED_WIDTH_VIOLATIONS.get(name) == window.seasons
                        and window.assigned_at < VALIDATOR_INTRODUCED_AT
                    )
                    message = (
                        f"window {list(window.seasons)} spans {width} season(s); "
                        f"assign_window only ever draws {MIN_WINDOW_SIZE}-"
                        f"{MAX_WINDOW_SIZE}"
                    )
                    if grandfathered:
                        message += (
                            f" -- grandfathered (ROADMAP.md ENG-37, 2026-09-05): "
                            f"assigned {window.assigned_at}, before this check existed "
                            f"({VALIDATOR_INTRODUCED_AT}); see the window's own notes"
                        )
                    issues.append(
                        Issue(
                            severity="warning" if grandfathered else "error",
                            code="window_width_out_of_range",
                            family=name,
                            message=message,
                        )
                    )
            if (
                set(window.covered_seasons) & _MINED_SEASON_SET
                and not family.acknowledges_mined_2018_2025
            ):
                issues.append(
                    Issue(
                        severity="error",
                        code="missing_mined_acknowledgment",
                        family=name,
                        message=(
                            f"window {list(window.seasons)} intersects the mined "
                            f"{MINED_SEASONS[0]}-{MINED_SEASONS[1]} seasons without "
                            "acknowledges_mined_2018_2025"
                        ),
                    )
                )
        for index, window in enumerate(own_windows):
            for other in own_windows[index + 1 :]:
                if _windows_overlap(window, other):
                    issues.append(
                        Issue(
                            severity="error",
                            code="overlapping_windows_within_family",
                            family=name,
                            message=(
                                f"windows {list(window.seasons)} and {list(other.seasons)} overlap"
                            ),
                        )
                    )
        if family.status in ("confirmed", "closed_negative") and not any(
            window.state == "spent" for window in own_windows
        ):
            issues.append(
                Issue(
                    severity="warning",
                    code="status_look_with_no_window",
                    family=name,
                    message=(
                        f"status {family.status!r} implies a recorded look, but no window "
                        "is in the 'spent' state"
                    ),
                )
            )
    return issues


def season_usage(registry: Registry) -> dict[str, int]:

    usage: dict[str, int] = {}
    for family in registry.families.values():
        seasons: set[int] = set()
        for window in family.windows:
            if window.state == "spent":
                seasons.update(window.covered_seasons)
        for season in seasons:
            key = str(season)
            usage[key] = usage.get(key, 0) + 1
    return dict(sorted(usage.items()))


def _window_payload(window: Window) -> dict[str, Any]:
    return {
        "seasons": [window.seasons[0], window.seasons[1]],
        "state": window.state,
        "window_kind": window.window_kind,
        "assigned_at": window.assigned_at,
        "spent_at": window.spent_at,
        "artifact": window.artifact,
        "verdict": window.verdict,
        "closing_ground": window.closing_ground,
        "probability_positive": window.probability_positive,
        "effect": window.effect,
        "effect_units": window.effect_units,
        "interval": None if window.interval is None else list(window.interval),
        "standard_error": window.standard_error,
        "sample_blocks": window.sample_blocks,
        "leg_effects": (
            None
            if window.leg_effects is None
            else [
                {
                    "season": leg.season,
                    "effect": leg.effect,
                    "probability_positive": leg.probability_positive,
                    "sample_blocks": leg.sample_blocks,
                }
                for leg in window.leg_effects
            ]
        ),
        "notes": window.notes,
    }


def _family_payload(family: Family) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "declared_at": family.declared_at,
        "description": family.description,
        "grade": family.grade,
        "status": family.status,
        "inherits": list(family.inherits),
        "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
        "windows": [_window_payload(window) for window in family.windows],
    }
    if family.coverage_weak_signal_family is not None:
        payload["coverage_weak_signal_family"] = family.coverage_weak_signal_family
        payload["coverage_league"] = family.coverage_league
        payload["coverage_effect_units"] = list(family.coverage_effect_units)
    if family.plain_summary is not None:
        payload["plain_summary"] = family.plain_summary
    return payload


def registry_payload(registry: Registry) -> dict[str, Any]:

    payload: dict[str, Any] = {
        "version": registry.version,
        "notes": list(registry.notes),
        "families": {name: _family_payload(family) for name, family in registry.families.items()},
        "season_usage": season_usage(registry),
    }
    if registry.no_rotation_needed:
        payload["no_rotation_needed"] = {
            key: {
                "weak_signal_family": record.weak_signal_family,
                "league": record.league,
                "reason": record.reason,
                "declared_at": record.declared_at,
                "effect_units": list(record.effect_units),
                "notes": record.notes,
            }
            for key, record in sorted(registry.no_rotation_needed.items())
        }
    return payload


def registry_from_payload(
    payload: Any, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> Registry:

    if not isinstance(payload, dict):
        raise RegistryError("Rotation registry must be a JSON object")
    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Rotation registry has unknown top-level fields: {unknown}",
            on_unknown_field=on_unknown_field,
        )
    families_payload = payload.get("families", {})
    if not isinstance(families_payload, dict):
        raise RegistryError("Rotation registry families must be an object")
    notes_payload = payload.get("notes", [])
    if not isinstance(notes_payload, list):
        raise RegistryError("Rotation registry notes must be a list")
    no_rotation_payload = payload.get("no_rotation_needed", {})
    if not isinstance(no_rotation_payload, dict):
        raise RegistryError("Rotation registry no_rotation_needed must be an object")
    registry = Registry(
        version=int(payload.get("version", 0)),
        notes=tuple(str(note) for note in notes_payload),
        families={
            str(name): _family_from_payload(str(name), family, on_unknown_field=on_unknown_field)
            for name, family in families_payload.items()
        },
        no_rotation_needed={
            str(key): _no_rotation_record_from_payload(
                str(key), record, on_unknown_field=on_unknown_field
            )
            for key, record in no_rotation_payload.items()
        },
    )
    _validate(registry)
    return registry


def load_registry(
    path: Path | None = None, *, on_unknown_field: weak_signals.OnUnknownField = "raise"
) -> Registry:

    destination = path or default_registry_path()
    if not destination.is_file():
        raise RegistryError(f"Rotation registry not found: {destination}")
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryError(f"Rotation registry is not valid JSON: {destination}") from error
    return registry_from_payload(payload, on_unknown_field=on_unknown_field)


def save_registry(registry: Registry, path: Path | None = None) -> None:

    _validate(registry)
    for issue in validate_registry(registry):
        family_note = f" ({issue.family})" if issue.family else ""
        print(
            f"rotation registry warning [{issue.code}]{family_note}: {issue.message}",
            file=sys.stderr,
        )
    atomic_json(registry_payload(registry), path or default_registry_path())


def _replace_family(registry: Registry, family: Family) -> Registry:
    families = dict(registry.families)
    families[family.name] = family
    updated = Registry(
        version=registry.version,
        notes=registry.notes,
        families=families,
        no_rotation_needed=registry.no_rotation_needed,
    )
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
    plain_summary: str | None = None,
) -> Registry:

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
    validated_plain_summary = _validate_plain_summary(f"Family {name!r}", plain_summary)
    family = Family(
        name=name,
        declared_at=_today(),
        description=description,
        grade=grade,
        status="open",
        inherits=tuple(inherits),
        acknowledges_mined_2018_2025=acknowledges_mined_2018_2025,
        windows=(),
        plain_summary=validated_plain_summary,
    )
    return _replace_family(registry, family)


def declare_coverage_stub(
    registry: Registry,
    name: str,
    *,
    weak_signal_family: str,
    league: str,
    effect_units: tuple[str, ...] = (),
) -> Registry:

    if not name:
        raise RegistryError("Family name is required")
    if name in registry.families:
        raise RegistryError(f"Family {name!r} is already declared; declarations are append-only")
    if not weak_signal_family:
        raise RegistryError("weak_signal_family is required")
    if league not in weak_signals.LEAGUES:
        raise RegistryError(f"Unknown league {league!r}; choose one of {weak_signals.LEAGUES}")
    family = Family(
        name=name,
        declared_at=_today(),
        description=(
            f"ENG-27 coverage stub for weak-signal family {weak_signal_family!r} "
            f"(league={league!r}; docs/rotation_registry.md 'Coverage' section). No "
            "window assigned and no research commitment made; grade defaults to "
            f"{COVERAGE_STUB_GRADE!r} pending a real declaration."
        ),
        grade=COVERAGE_STUB_GRADE,
        status=COVERAGE_STUB_STATUS,
        inherits=(),
        acknowledges_mined_2018_2025=False,
        windows=(),
        coverage_weak_signal_family=weak_signal_family,
        coverage_league=league,
        coverage_effect_units=tuple(effect_units),
    )
    return _replace_family(registry, family)


def record_no_rotation_needed(
    registry: Registry,
    weak_signal_family: str,
    *,
    league: str,
    reason: str,
    effect_units: tuple[str, ...] = (),
    notes: str = "",
) -> Registry:

    if not weak_signal_family:
        raise RegistryError("weak_signal_family is required")
    if weak_signal_family in registry.no_rotation_needed:
        raise RegistryError(
            f"{weak_signal_family!r} already has a no_rotation_needed record; "
            "declarations here are append-only"
        )
    if league not in weak_signals.LEAGUES:
        raise RegistryError(f"Unknown league {league!r}; choose one of {weak_signals.LEAGUES}")
    if not _is_admissible_no_rotation_reason(reason):
        raise RegistryError(
            f"Inadmissible reason {reason!r}; expected one of {NO_ROTATION_FIXED_REASONS} or "
            f"'{_DECOMPOSITION_PARENT_PREFIX}<family>'"
        )
    record = NoRotationRecord(
        weak_signal_family=weak_signal_family,
        league=league,
        reason=reason,
        declared_at=_today(),
        effect_units=tuple(effect_units),
        notes=notes,
    )
    no_rotation_needed = dict(registry.no_rotation_needed)
    no_rotation_needed[weak_signal_family] = record
    updated = Registry(
        version=registry.version,
        notes=registry.notes,
        families=registry.families,
        no_rotation_needed=no_rotation_needed,
    )
    _validate(updated)
    return updated


def set_plain_summary(registry: Registry, name: str, *, plain_summary: str) -> Registry:

    if name not in registry.families:
        raise RegistryError(f"Unknown family: {name!r}")
    validated = _validate_plain_summary(f"Family {name!r}", plain_summary)
    if validated is None:
        raise RegistryError(f"Family {name!r}: plain_summary is required")
    family = registry.families[name]
    return _replace_family(registry, replace(family, plain_summary=validated))


def _touched_seasons(registry: Registry, name: str) -> frozenset[int]:

    return frozenset(
        season for _, window in _chain_windows(registry, name) for season in window.covered_seasons
    )


def eligible_blocks(
    registry: Registry, name: str, *, size: int | None = None
) -> tuple[tuple[int, int], ...]:

    family = registry.families[name]
    width = DEFAULT_WINDOW_SIZE[family.grade] if size is None else size
    pool_start, pool_end = GRADE_POOLS[family.grade]
    touched = _touched_seasons(registry, name)
    blocks: list[tuple[int, int]] = []
    for start in range(max(pool_start, MIN_ELIGIBLE_START_SEASON), pool_end - width + 2):
        block = (start, start + width - 1)
        if touched.intersection(range(block[0], block[1] + 1)):
            continue
        if _overlaps(block, MINED_SEASONS) and not family.acknowledges_mined_2018_2025:
            continue
        blocks.append(block)
    return tuple(blocks)


def eligible_stratified_seasons(registry: Registry, name: str) -> tuple[int, ...]:

    family = registry.families[name]
    if family.grade != STRATIFIED_GRADE:
        raise RegistryError(
            f"Stratified confirmation windows are {STRATIFIED_GRADE}-graded only "
            f"(docs/era_stratified_windows_proposal.md scope limit); family "
            f"{name!r} is grade {family.grade!r}. Opener-graded families keep "
            "contiguous windows (the six-season paired archive means "
            "stratification buys little there); nflverse_spread shares the same "
            "numeric season pool as close but is excluded too, as a documented "
            "resolution decision -- the proposal's scope-limit text names only "
            "close-graded families."
        )
    pool_start, pool_end = GRADE_POOLS[family.grade]
    floor = max(pool_start, MIN_ELIGIBLE_START_SEASON)
    touched = _touched_seasons(registry, name)
    return tuple(
        season
        for season in range(floor, pool_end + 1)
        if season not in touched
        and (season not in _MINED_SEASON_SET or family.acknowledges_mined_2018_2025)
    )


def assign_window(registry: Registry, family: str, *, size: int | None = None) -> Registry:

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


def assign_stratified_window(registry: Registry, family: str) -> Registry:

    if family not in registry.families:
        raise RegistryError(f"Unknown family: {family!r}")
    declared = registry.families[family]
    if declared.assigned_window is not None:
        raise RegistryError(
            f"Family {family!r} already holds an unspent window "
            f"{list(declared.assigned_window.seasons)}; record that look first"
        )
    eligible = eligible_stratified_seasons(registry, family)
    if len(eligible) < STRATIFIED_LEG_COUNT:
        raise RegistryError(
            f"Fewer than {STRATIFIED_LEG_COUNT} eligible seasons remain for a "
            f"stratified window for {family!r} in the {declared.grade} pool "
            f"{list(GRADE_POOLS[declared.grade])}"
        )
    leg_a, leg_b = min(eligible), max(eligible)
    window = Window(
        seasons=(leg_a, leg_b),
        state="assigned",
        window_kind="stratified",
        assigned_at=_today(),
    )
    return _replace_family(registry, replace(declared, windows=(*declared.windows, window)))


def confirmation_split(
    features: pd.DataFrame, registry: Registry, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:

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
    if window_entry.window_kind == "stratified":
        raise RegistryError(
            f"Family {family!r}'s assigned window {list(window_entry.seasons)} is "
            "stratified (a leg pair, not a contiguous block); use "
            "confirmation_split_legs, which gives each leg its own forward-chained "
            "training cutoff, instead of confirmation_split"
        )

    missing_columns = sorted({"season", "gameday", "result"}.difference(features.columns))
    if missing_columns:
        raise RegistryError(f"Feature table is missing columns: {', '.join(missing_columns)}")

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    available = set(frame["season"].astype(int))
    missing_seasons = [s for s in window_entry.covered_seasons if s not in available]
    if missing_seasons:
        raise RegistryError(
            f"Feature table is missing window seasons for {family!r}: {missing_seasons}"
        )

    window = frame.loc[frame["season"].astype(int).isin(list(window_entry.covered_seasons))].copy()
    cutoff = window["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
    if training.empty:
        raise RegistryError(
            f"Family {family!r} window {list(window_entry.seasons)} has no completed "
            "games before it; the window cannot be scored (warm-up rule 9)"
        )
    return training, window


@dataclass(frozen=True)
class LegSplit:
    season: int
    training: pd.DataFrame
    scoring: pd.DataFrame


def confirmation_split_legs(
    features: pd.DataFrame, registry: Registry, family: str
) -> tuple[LegSplit, ...]:

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
    if window_entry.window_kind != "stratified":
        raise RegistryError(
            f"Family {family!r}'s assigned window {list(window_entry.seasons)} is "
            "contiguous; use confirmation_split for it, not confirmation_split_legs"
        )

    missing_columns = sorted({"season", "gameday", "result"}.difference(features.columns))
    if missing_columns:
        raise RegistryError(f"Feature table is missing columns: {', '.join(missing_columns)}")

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    available = set(frame["season"].astype(int))
    missing_seasons = [s for s in window_entry.covered_seasons if s not in available]
    if missing_seasons:
        raise RegistryError(
            f"Feature table is missing window seasons for {family!r}: {missing_seasons}"
        )

    splits: list[LegSplit] = []
    for season in window_entry.covered_seasons:
        leg = frame.loc[frame["season"].astype(int) == season].copy()
        cutoff = leg["gameday"].min()
        training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
        if training.empty:
            raise RegistryError(
                f"Family {family!r} leg {season} has no completed games before it; "
                "the leg cannot be scored (warm-up rule 9)"
            )
        splits.append(LegSplit(season=season, training=training, scoring=leg))
    return tuple(splits)


def record_look(
    registry: Registry,
    family: str,
    *,
    artifact: str,
    verdict: str,
    probability_positive: float | None,
    closing_ground: str | None = None,
    effect: float | None = None,
    effect_units: str | None = None,
    interval: tuple[float, float] | None = None,
    standard_error: float | None = None,
    sample_blocks: int | None = None,
    leg_effects: list[dict[str, Any]] | None = None,
    notes: str = "",
    replace_existing: bool = False,
) -> Registry:

    if family not in registry.families:
        raise RegistryError(f"Unknown family: {family!r}")
    declared = registry.families[family]
    if replace_existing:
        if declared.assigned_window is not None:
            raise RegistryError(
                f"Family {family!r} has an assigned window; --replace may only correct "
                "the latest spent window"
            )
        if not declared.windows or declared.windows[-1].state != "spent":
            raise RegistryError(
                f"Family {family!r} has no latest spent window eligible for --replace"
            )
        window = declared.windows[-1]
        if artifact != window.artifact:
            raise RegistryError(
                f"Family {family!r}: --replace requires the exact latest-window artifact "
                f"{window.artifact!r}, not {artifact!r}"
            )
    else:
        assigned_window = declared.assigned_window
        if assigned_window is None:
            raise RegistryError(f"Family {family!r} has no assigned window to record")
        window = assigned_window
    if not artifact:
        raise RegistryError("A recorded look requires an artifact path")
    if verdict not in VERDICTS:
        raise RegistryError(f"Unknown verdict {verdict!r}; choose one of {VERDICTS}")
    if probability_positive is not None and not 0.0 <= probability_positive <= 1.0:
        raise RegistryError("probability_positive must lie in [0, 1]")
    _validate_closing_ground(
        f"Family {family!r}",
        verdict=verdict,
        closing_ground=closing_ground,
        probability_positive=probability_positive,
    )
    (
        resolved_effect,
        resolved_effect_units,
        resolved_interval,
        resolved_standard_error,
        resolved_sample_blocks,
    ) = _validate_effect_fields(
        f"Family {family!r}",
        effect=effect,
        effect_units=effect_units,
        interval=interval,
        standard_error=standard_error,
        sample_blocks=sample_blocks,
    )
    resolved_leg_effects = _validate_leg_effects(
        f"Family {family!r}",
        window_kind=window.window_kind,
        seasons=window.seasons,
        payload=leg_effects,
    )
    if window.window_kind == "stratified" and resolved_leg_effects is None:
        raise RegistryError(
            f"Family {family!r}: a stratified window's recorded look requires "
            f"leg_effects -- one effect magnitude per leg {list(window.seasons)} "
            "(owner's binding refinement: era variation is a change in magnitude, "
            "never collapsed into the pooled read alone)"
        )

    spent = replace(
        window,
        state="spent",
        spent_at=window.spent_at if replace_existing else _today(),
        artifact=artifact,
        verdict=verdict,
        closing_ground=closing_ground,
        probability_positive=probability_positive,
        effect=resolved_effect,
        effect_units=resolved_effect_units,
        interval=resolved_interval,
        standard_error=resolved_standard_error,
        sample_blocks=resolved_sample_blocks,
        leg_effects=resolved_leg_effects,
        notes=notes,
    )
    windows = (
        (*declared.windows[:-1], spent)
        if replace_existing
        else tuple(spent if entry is window else entry for entry in declared.windows)
    )
    status = (
        "open"
        if replace_existing and verdict == "unresolved"
        else (declared.status if verdict == "unresolved" else verdict)
    )
    return _replace_family(registry, replace(declared, windows=windows, status=status))


def grade_pool_capacity(registry: Registry) -> dict[str, dict[str, Any]]:

    taken_seasons = {
        season
        for family in registry.families.values()
        for window in family.windows
        for season in window.covered_seasons
    }
    capacity: dict[str, dict[str, Any]] = {}
    for grade, (pool_start, pool_end) in GRADE_POOLS.items():
        width = DEFAULT_WINDOW_SIZE[grade]
        blocks = [
            (start, start + width - 1)
            for start in range(max(pool_start, MIN_ELIGIBLE_START_SEASON), pool_end + 1, width)
            if start + width - 1 <= pool_end
        ]
        unspent = [
            block
            for block in blocks
            if not taken_seasons.intersection(range(block[0], block[1] + 1))
        ]
        capacity[grade] = {
            "seasons": [pool_start, pool_end],
            "default_window_size": width,
            "total_windows": len(blocks),
            "unspent_windows": len(unspent),
            "unspent_blocks": [list(block) for block in unspent],
        }
    return capacity


def registry_status(registry: Registry) -> dict[str, Any]:

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
                "plain_summary": family.plain_summary,
                "inherits": list(family.inherits),
                "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
                "windows": [_window_payload(window) for window in family.windows],
                "remaining_eligible_windows": len(eligible_blocks(registry, name)),
                "remaining_eligible_stratified_seasons": (
                    len(eligible_stratified_seasons(registry, name))
                    if family.grade == STRATIFIED_GRADE
                    else 0
                ),
            }
        )
    capacity = grade_pool_capacity(registry)
    return {
        "version": registry.version,
        "notes": list(registry.notes),
        "families": families,
        "grade_pools": capacity,
        "season_usage": season_usage(registry),
        "no_rotation_needed_count": len(registry.no_rotation_needed),
        "summary": [
            f"{grade} pool: {values['unspent_windows']} windows unspent"
            for grade, values in capacity.items()
        ],
    }
