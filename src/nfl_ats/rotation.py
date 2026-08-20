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

# Grade determines the eligible season pool. Opener-graded confirmations need
# the paired Tuesday-opener archive, which only covers 2020-2025.
GRADE_POOLS: dict[str, tuple[int, int]] = {
    "opener": (2020, 2025),
    "close": (2009, 2025),
    "nflverse_spread": (2009, 2025),
}
DEFAULT_WINDOW_SIZE = {"opener": 2, "close": 3, "nflverse_spread": 3}
MINED_SEASONS = (2018, 2025)
_MINED_SEASON_SET = frozenset(range(MINED_SEASONS[0], MINED_SEASONS[1] + 1))

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

# Era-stratified confirmation windows (docs/era_stratified_windows_proposal.md,
# owner-approved 2026-08-19). A window is either "contiguous" (the original
# [start, end] block) or "stratified": a pair of non-adjacent single-season
# legs, each scored walk-forward with training strictly prior to that leg.
# Scoped to close-graded families only -- the proposal's own text, verbatim:
# opener-graded families draw from a six-season archive where stratification
# "buys little and is not proposed there." nflverse_spread shares the same
# numeric season pool as close but is never named in that scope-limit
# sentence, so it is excluded too, conservatively, as a resolution decision
# recorded in the proposal doc's "Implemented" section rather than assumed.
# Leg pairs, not larger tuples: the proposal's own worked example and its
# "Same data budget per look (2 seasons)" framing both describe exactly two
# legs, matching MIN_WINDOW_SIZE. Extending to 3+ legs is out of scope here.
_WINDOW_KINDS = ("contiguous", "stratified")
STRATIFIED_GRADE = "close"
STRATIFIED_LEG_COUNT = 2

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

# A closed_negative verdict must stand on one of the admissible closing grounds
# from AGENTS.md's binding taxonomy (shared with the weak-signal registry). An
# interval containing zero is not among them and never will be; that outcome is
# "unresolved", which spends the window without closing the family.
_TERMINAL_VERDICT_GROUNDS = tuple(
    ground for grounds in weak_signals.CLOSING_GROUNDS.values() for ground in grounds
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
class LegResult:
    """One stratified leg's own effect magnitude.

    The owner's binding refinement on the era-stratified proposal
    (docs/era_stratified_windows_proposal.md, 2026-08-19): era variation is
    expected to be a change in effect MAGNITUDE, not presence/absence, so a
    stratified window's per-leg numbers are first-class output and must never
    be collapsed into the pooled read alone. ``effect`` shares the parent
    window's ``effect_units``; ``probability_positive`` and ``sample_blocks``
    are optional, matching the pooled fields' own optionality.
    """

    season: int
    effect: float
    probability_positive: float | None = None
    sample_blocks: int | None = None


@dataclass(frozen=True)
class Window:
    """One confirmation window drawn by a family.

    ``window_kind`` is ``"contiguous"`` (the original block, ``seasons`` is
    ``[start, end]``) or ``"stratified"`` (a leg pair, ``seasons`` is the two
    individual leg seasons, stored ascending -- NOT the endpoints of a range;
    the span between two legs was deliberately never looked at). The two
    interpretations are disambiguated by ``window_kind`` rather than by list
    shape, because a bare 2-element ``[a, b]`` is structurally identical
    either way and guessing from shape alone would be exactly the kind of
    silent ambiguity this registry exists to rule out.
    """

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
        """The contiguous range this window spans. Contiguous windows only.

        Raises for a stratified window: its two endpoints do not denote every
        season in between, so returning a range here would silently misstate
        what was actually looked at. Use ``covered_seasons`` in general code
        that must handle both kinds.
        """

        if self.window_kind == "stratified":
            raise RegistryError(
                "season_range is undefined for a stratified window; the span "
                "between its legs was never looked at. Use covered_seasons."
            )
        return range(self.seasons[0], self.seasons[1] + 1)

    @property
    def covered_seasons(self) -> tuple[int, ...]:
        """Every individual season this window actually touched.

        The full range for a contiguous window; just the two leg seasons for
        a stratified one. This is the abstraction every overlap, usage, and
        capacity computation in this module must use once a window's
        ``[min, max]`` endpoints can no longer be assumed to mean "every
        season in between."
        """

        if self.window_kind == "stratified":
            return tuple(sorted(self.seasons))
        return tuple(range(self.seasons[0], self.seasons[1] + 1))


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
    """Whether two literal [start, end] ranges share a season.

    Only ever called on genuine ranges (a candidate block, or the fixed
    ``MINED_SEASONS`` constant) -- never on a ``Window.seasons`` tuple, which
    may now be a stratified leg pair rather than a range. Use
    ``_windows_overlap`` (below) for two windows, or intersect
    ``covered_seasons``/``_touched_seasons`` directly for a window against a
    candidate block.
    """

    return left[0] <= right[1] and right[0] <= left[1]


def _windows_overlap(left: Window, right: Window) -> bool:
    """Whether two windows share any actual season.

    Computed from each window's real ``covered_seasons``, not its
    ``[min, max]`` endpoints -- load-bearing once a window can be a
    stratified leg pair, where the span between its legs was never looked at
    and must not count as "touched" for overlap purposes.
    """

    return bool(set(left.covered_seasons) & set(right.covered_seasons))


def _validate_closing_ground(
    context: str,
    *,
    verdict: str | None,
    closing_ground: str | None,
    probability_positive: float | None,
) -> None:
    """Enforce AGENTS.md's binding closure taxonomy on a recorded verdict.

    ``closed_negative`` is a terminal claim, and the binding rule allows only
    two grounds for one: a refuted mechanism (a RESOLVED wrong sign, or zero
    split-half reliability) or a positive control proven able to detect an
    effect that size. "The interval contains zero" is on no admissible list —
    that outcome is ``unresolved``, which spends the window without closing
    the family. Enforced here, fail-closed, because the prose version of this
    rule was ignored repeatedly by sessions that never loaded it.
    """

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
    """Validate and coerce the optional effect-size fields on a window.

    Shared by ``_window_from_payload`` (loading a raw JSON payload) and
    ``record_look`` (recording a fresh look), so the two paths cannot drift.
    Mirrors ``weak_signals.signal_from_payload``'s validation of the same
    concepts, and imports ``EFFECT_UNITS`` from there rather than duplicating
    it -- an effect recorded in a unit that module does not recognise could
    never be pooled with anything, in either registry.
    """

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


_LEG_RESULT_FIELDS = frozenset({"season", "effect", "probability_positive", "sample_blocks"})


def _leg_result_from_payload(context: str, payload: Any) -> LegResult:
    if not isinstance(payload, dict):
        raise RegistryError(f"{context}: leg_effects entries must be objects")
    unknown = sorted(set(payload).difference(_LEG_RESULT_FIELDS))
    if unknown:
        raise RegistryError(f"{context}: unknown leg_effects fields: {unknown}")
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
    context: str, *, window_kind: str, seasons: tuple[int, ...], payload: Any
) -> tuple[LegResult, ...] | None:
    """Validate the per-leg magnitude report on a stratified window.

    Shared by ``_window_from_payload`` and ``record_look``, so the two paths
    cannot drift -- the same discipline ``_validate_effect_fields`` already
    follows for the pooled effect fields. Only a stratified window may carry
    ``leg_effects``, and when present it must name exactly one result per leg,
    matching the window's own leg seasons -- neither fewer (a dropped leg)
    nor more (a phantom one).
    """

    if payload is None:
        return None
    if window_kind != "stratified":
        raise RegistryError(f"{context}: leg_effects is only meaningful on a stratified window")
    if not isinstance(payload, list):
        raise RegistryError(f"{context}: leg_effects must be a list")
    results = tuple(_leg_result_from_payload(context, entry) for entry in payload)
    result_seasons = sorted(result.season for result in results)
    expected_seasons = sorted(seasons)
    if result_seasons != expected_seasons:
        raise RegistryError(
            f"{context}: leg_effects must report exactly one result per leg "
            f"{expected_seasons}, got {result_seasons}"
        )
    return results


def _window_from_payload(family_name: str, payload: Any) -> Window:
    if not isinstance(payload, dict):
        raise RegistryError(f"Family {family_name!r} has a non-object window entry")
    unknown = sorted(set(payload).difference(_WINDOW_FIELDS))
    if unknown:
        raise RegistryError(f"Family {family_name!r} has unknown window fields: {unknown}")
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
    )
    # A closed_negative window carrying NO ground is tolerated here, and only
    # here: historical ledger entries are never re-judged on load (the same
    # principle as the warm-up floor). `record_look` refuses to WRITE a new
    # one, and the live-ledger contract test enforces the taxonomy on the
    # tracked registry, so the tolerance covers frozen prose-era history only.
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
            if (
                window.window_kind == "stratified"
                and window.state == "spent"
                and not window.leg_effects
            ):
                # Owner's binding refinement on the era-stratified proposal
                # (docs/era_stratified_windows_proposal.md, 2026-08-19): era
                # variation is a change in MAGNITUDE, so a stratified window's
                # per-leg effect sizes must accompany its pooled read, always
                # -- enforced here the same way "spent-without-artifact" is,
                # a few lines up, rather than left to a write-up's prose.
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
                # Belt-and-braces: assign_stratified_window already refuses a
                # non-close grade, but a hand-edited ledger must not be able
                # to smuggle one past load-time validation either.
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


def season_usage(registry: Registry) -> dict[str, int]:
    """Return the global count of families that have SPENT each season."""

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


def _touched_seasons(registry: Registry, name: str) -> frozenset[int]:
    """Every individual season ``name`` or its inherits chain has drawn.

    Computed from each window's real ``covered_seasons``, not its
    ``[min, max]`` endpoints -- for a stratified window those endpoints are
    just its two legs, and the span between them was never looked at, so
    treating it as "touched" would wrongly shrink a family's eligible pool.
    Replaces the old ``_blocked_seasons`` (a tuple of ``[start, end]``
    ranges), which was exactly right as long as every window was contiguous
    and stopped being right the moment one might not be.
    """

    return frozenset(
        season for _, window in _chain_windows(registry, name) for season in window.covered_seasons
    )


def eligible_blocks(
    registry: Registry, name: str, *, size: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Return every block of ``size`` consecutive seasons this family may still draw."""

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
    """Return every individual season this close-graded family may still draw
    as a stratified leg -- the pool floor and up, minus anything touched by
    this family or its inherits chain, minus mined seasons if unacknowledged.

    Raises if ``name`` is not close-graded (docs/era_stratified_windows_proposal.md
    scope limit: stratified windows are close-graded only).
    """

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


def assign_stratified_window(registry: Registry, family: str) -> Registry:
    """Assign a two-leg era-stratified window to a close-graded family.

    docs/era_stratified_windows_proposal.md (owner-approved 2026-08-19): a
    confirmation window may be composed of two non-adjacent single-season
    legs instead of a contiguous block, so one look spans two regime eras
    instead of one. Close-graded families only -- see
    ``eligible_stratified_seasons`` for the scope-limit error.

    Deterministic leg-pair rule (the proposal's own stated rule, implemented
    exactly): the earliest untouched season, paired with the untouched
    season maximally distant from it. That reduces algebraically to
    ``(min(eligible), max(eligible))``: every remaining eligible season is
    ``>= min(eligible)`` by definition, so distance-from-the-minimum is
    ``season - min(eligible)``, a quantity monotonically increasing in
    ``season`` -- its maximizer is therefore always the largest eligible
    season. No tie can arise (the minimum and maximum coincide only when a
    single season remains, which is refused below as insufficient), so the
    pair is fully determined by the ledger, exactly like the contiguous
    ``assign_window``: no hidden choice, nothing to tune, no window that can
    be cherry-picked.
    """

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
        # Fail-closed backstop for ledgers written outside `assign_window`
        # (rule 9's floor guards assignment, not hand-edited history): a window
        # with nothing completed before it cannot be forward-chain scored.
        raise RegistryError(
            f"Family {family!r} window {list(window_entry.seasons)} has no completed "
            "games before it; the window cannot be scored (warm-up rule 9)"
        )
    return training, window


@dataclass(frozen=True)
class LegSplit:
    """One stratified leg's own forward-chained (training, scoring) frames."""

    season: int
    training: pd.DataFrame
    scoring: pd.DataFrame


def confirmation_split_legs(
    features: pd.DataFrame, registry: Registry, family: str
) -> tuple[LegSplit, ...]:
    """Per-leg walk-forward split for a family's assigned stratified window.

    docs/era_stratified_windows_proposal.md point 2: each leg is evaluated
    walk-forward with training strictly prior to THAT leg -- not a single
    cutoff shared across legs. Concretely, leg B's training frame is every
    completed game before leg B's first gameday, which naturally includes leg
    A's season if leg A is chronologically earlier; what the proposal forbids
    is a leg training on data at or after its OWN scoring season, not on data
    from another leg. Forward-chaining is not optional here, exactly as in
    ``confirmation_split`` -- each leg raises if it has no completed history
    before it.
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
            # Same fail-closed backstop as confirmation_split, applied per leg:
            # rule 9's floor guards assignment, not hand-edited history.
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
) -> Registry:
    """Mark the family's assigned window spent. A look is one look, always recorded.

    ``leg_effects`` is required for a stratified window and rejected for a
    contiguous one: the owner's binding refinement on the era-stratified
    proposal (docs/era_stratified_windows_proposal.md, 2026-08-19) is that era
    variation is a change in effect MAGNITUDE, so a stratified look's per-leg
    numbers must accompany its pooled read every time, not just when someone
    remembers to write them down. One entry per leg
    (``[{"season": ..., "effect": ..., "probability_positive": ..., "sample_blocks": ...}, ...]``);
    ``effect`` shares the window's own ``effect_units``.
    """

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
        spent_at=_today(),
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

    # Season-set based, not a range-overlap on [min, max] endpoints: a
    # stratified window's endpoints are its two legs, and the span between
    # them was never looked at, so it must not count as "taken" capacity.
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
                # Close-graded only (era-stratified proposal scope limit); other
                # grades report 0 rather than raising, since this is a status
                # listing over every family, not a per-family eligibility check.
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
        "summary": [
            f"{grade} pool: {values['unspent_windows']} windows unspent"
            for grade, values in capacity.items()
        ],
    }
