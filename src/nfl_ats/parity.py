"""ENG-17/ENG-28: baseline-parity regression adaptor.

``grade_games`` runs a small frozen fixture (``tests/fixtures/parity/games.csv``)
through the SAME production functions the project's real market-only,
simple-model, active-model, and overlay comparisons use, and returns one
:class:`PathResult` per (path, line) with everything a parity test needs: the
graded game-ID population, which weeks the chronological cutoff skipped, the
push rule and which games it excluded, the accuracy, and a per-game
correctness map (for computing a paired delta on an intersection, per
``ROADMAP.md`` Phase 13's ENG-17 item).

ENG-28 adds two more paths -- ``"active_model_production"`` and
``"overlay_production"`` -- that run through the actual opener-snapshot
machinery the pool card is graded on
(:func:`nfl_ats.clv.opener_pick_evaluation` at the real ``weak_stack``/
ridge-alpha-10 configuration, and
:func:`nfl_ats.four_overlay_composition.apply_four_overlay_composition`)
rather than ``walk_forward_backtest(feature_set="market_context")``'s
stand-in. They require an ``opener_store`` argument: the miniature,
commit-safe market-snapshot store under
``tests/fixtures/parity/opener_store/`` (same directory layout as
``data/market/raw/``, generated deterministically from ``games.csv`` --
see ``docs/baseline_parity.md``).

This module is a pure ADAPTOR. It never recomputes a cover/push rule from
scratch:

* :func:`nfl_ats.features.add_ats_outcomes` computes ``ats_margin``/
  ``home_cover`` (the project's one push definition: ``ats_margin == 0`` is a
  push, excluded from accuracy, never a loss or a half-win) from ``result``
  and ``spread_line`` -- unchanged, imported directly.
* :func:`nfl_ats.clv.pick_correct` turns a pick plus a settle margin into
  correct/incorrect/push (``NaN``) -- the same function every path below uses
  to build its ``correct_by_game`` map, so "identical push handling across
  paths" is not an assertion about parallel code, it is the same function
  called four times.
* :func:`nfl_ats.backtest.walk_forward_backtest` (weekly refit, chronological
  training cutoff, ``fit_cover_model``, ``summarize_predictions``) grades the
  ``simple_model`` and ``active_model`` paths -- the same harness production's
  own backtests use, distinguished only by ``feature_set`` (``"market"`` vs
  ``"market_context"``, both real entries in
  :data:`nfl_ats.constants.FEATURE_SETS`). The project's real weekly-refit
  ``weak_stack``/ridge active model additionally depends on the point-in-time
  Tuesday-opener market-snapshot store (``nfl_ats.clv.opener_pick_evaluation``);
  reproducing that store for a 60-row fixture is out of scope here, so
  ``active_model`` stands in for it with the same walk-forward harness at a
  richer feature set. Nothing about population/cutoff/push parity depends on
  which feature set is fit -- see ``docs/baseline_parity.md``.
* :func:`nfl_ats.odds.no_vig_probabilities` grades ``market_only`` directly
  from spread prices, with no model fit at all.
* :func:`nfl_ats.coach_fade_overlay.apply_coach_fade_overlay` grades
  ``overlay`` by flipping ``active_model``'s own predictions -- the real
  pick-level transform, not a re-implementation.
* :func:`nfl_ats.clv.opener_pick_evaluation` grades
  ``active_model_production`` directly from the miniature opener store (ENG-28)
  -- the real production entry point, not the ``market_context`` stand-in
  ``active_model`` above uses.
* :func:`nfl_ats.four_overlay_composition.apply_four_overlay_composition`
  grades ``overlay_production`` by unioning all four real overlay members'
  flips against ``active_model_production``'s own predictions (ENG-28).

The ONE inlined piece is :func:`_eligible_weeks`, which mirrors (does not
reinvent) the ``cutoff = weekly_games["gameday"].min(); training =
completed.loc[completed["gameday"].lt(cutoff) & completed["home_cover"].notna()]``
pattern that appears verbatim inside both ``backtest.walk_forward_backtest``
and ``clv.opener_pick_evaluation``. It exists because the codebase has no
"market-only, cutoff-aware" function to call (market-only needs no model fit
at all), so ``market_only``'s population is computed independently and then
cross-checked by the test suite against ``walk_forward_backtest``'s own
internal cutoff behaviour -- an agreement between two independent
expressions of the same rule, not a duplicate implementation of grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from nfl_ats.backtest import BacktestResult, walk_forward_backtest
from nfl_ats.clv import opener_pick_evaluation, pick_correct
from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX, apply_coach_fade_overlay
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES, MODEL_FEATURE_COLUMNS
from nfl_ats.features import add_ats_outcomes
from nfl_ats.four_overlay_composition import apply_four_overlay_composition
from nfl_ats.margin import MarginFeatureProfile, margin_feature_columns
from nfl_ats.odds import no_vig_probabilities
from nfl_ats.player_arrests_back_side_overlay import ArrestSnapshot

Line = Literal["opener", "close"]
PathName = Literal[
    "market_only",
    "simple_model",
    "active_model",
    "overlay",
    "active_model_production",
    "overlay_production",
]

PUSH_RULE = (
    "ats_margin == result - line == 0 is a push: excluded from accuracy, "
    "never scored as a loss or a half-win (nfl_ats.features.add_ats_outcomes, "
    "nfl_ats.clv.pick_correct)"
)

#: The hard floor every real fitting path in this codebase enforces
#: (``fit_cover_model``'s literal 50, ``margin.MIN_FITTABLE_TRAIN_GAMES``) --
#: not a suite-local choice. Below this, ``fit_cover_model``/``fit_margin_model``
#: raise regardless of what a caller passes as ``min_train_games``.
MIN_TRAIN_GAMES = MIN_FITTABLE_TRAIN_GAMES

SIMPLE_MODEL_FEATURE_SET = "market"
ACTIVE_MODEL_FEATURE_SET = "market_context"

#: The real production configuration (ENG-28), not a stand-in: matches
#: ``artifacts/active_ats_model.json`` (feature_profile "weak_stack",
#: regressor "ridge", ridge_alpha 10.0, method "market_residual") as read
#: 2026-09-04. Frozen here rather than read live so this suite's claims do
#: not silently drift if the active artifact is retrained -- the module
#: docstring for ``nfl_ats.clv.resolve_active_model_config`` describes the
#: same freeze-on-read pattern for its own fallback.
PRODUCTION_FEATURE_PROFILE: MarginFeatureProfile = "weak_stack"
PRODUCTION_MODEL_CONFIG: dict[str, object] = {
    "feature_profile": PRODUCTION_FEATURE_PROFILE,
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "target": "market_residual",
}

#: The weak_stack profile's feature contract carries 32 columns (injury
#: sub-splits, QB, roster-continuity, bias terms) that are not in
#: ``MODEL_FEATURE_COLUMNS`` -- the 60-row fixture zero-fills them exactly
#: like ``_features_for_line`` already zero-fills ``MODEL_FEATURE_COLUMNS``
#: (see ``_production_features`` below). A constant zero column is a
#: legitimate, if uninformative, ridge input: ``StandardScaler`` defines
#: scale 1.0 for zero variance rather than dividing by it, and the frozen
#: production ridge pipeline (``nfl_ats.margin.make_margin_estimator``) never
#: uses the group-wise penalty path that would otherwise require every
#: column to resolve to a known ``FEATURE_FAMILIES`` block.
PRODUCTION_FEATURE_COLUMNS: tuple[str, ...] = margin_feature_columns(
    "market_residual", PRODUCTION_FEATURE_PROFILE
)

_PATHS: tuple[PathName, ...] = (
    "market_only",
    "simple_model",
    "active_model",
    "overlay",
    "active_model_production",
    "overlay_production",
)

FIXTURE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "home_coach",
    "away_coach",
    "home_score",
    "away_score",
    "result",
    "spread_line_open",
    "spread_line_close",
    "total_line_open",
    "total_line_close",
    "home_spread_odds_open",
    "away_spread_odds_open",
    "home_spread_odds_close",
    "away_spread_odds_close",
    "rest_diff",
    "neutral_site",
    "div_game",
    "temp",
    "wind",
    "week_sin",
    "week_cos",
)


@dataclass(frozen=True)
class PathResult:
    """Everything a parity test needs about one (path, line) grading run."""

    path: str
    line: Line
    min_train_games: int
    push_rule: str
    scored_game_ids: frozenset[str]
    pushed_game_ids: frozenset[str]
    evaluated_game_ids: frozenset[str]
    accuracy: float
    skipped_weeks: tuple[tuple[int, int], ...]
    correct_by_game: dict[str, bool]
    flipped_game_ids: tuple[str, ...] = field(default_factory=tuple)


def load_fixture(path: Path) -> pd.DataFrame:
    """Read the frozen parity fixture CSV, schema-checked."""

    frame = pd.read_csv(path)
    missing = sorted(set(FIXTURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Parity fixture is missing columns: {', '.join(missing)}")
    if frame["game_id"].duplicated().any():
        raise ValueError("Parity fixture contains duplicate game_id values")
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    return frame


def _features_for_line(frame: pd.DataFrame, line: Line) -> pd.DataFrame:
    """The project's own declared approximation: swap the line, keep the rest.

    ``nfl_ats.clv.active_model_residual_at_opener`` documents exactly this
    trick (only ``spread_line`` is swapped to grade a different line; every
    other feature stays at its one recorded value) as the approximation
    behind the real opener-vs-close active-model comparison. Reused here at
    the feature-table level so ``walk_forward_backtest`` needs no opener-aware
    branch of its own.
    """

    if line not in ("opener", "close"):
        raise ValueError(f"line must be 'opener' or 'close', got {line!r}")
    suffix = "open" if line == "opener" else "close"
    working = frame.copy()
    working["spread_line"] = pd.to_numeric(working[f"spread_line_{suffix}"], errors="raise")
    working["total_line"] = pd.to_numeric(working[f"total_line_{suffix}"], errors="raise")
    working["home_spread_odds"] = working[f"home_spread_odds_{suffix}"]
    working["away_spread_odds"] = working[f"away_spread_odds_{suffix}"]
    working = add_ats_outcomes(working)
    for column in MODEL_FEATURE_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
    return working


def _week_keys(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(frame["season"].astype(int), frame["week"].astype(int), strict=True))


def _eligible_weeks(
    frame: pd.DataFrame, min_train_games: int
) -> tuple[frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]:
    """Which (season, week) groups have >= ``min_train_games`` prior non-push games.

    Mirrors (does not call, because no shared function exists) the identical
    inlined cutoff pattern in ``backtest.walk_forward_backtest`` and
    ``clv.opener_pick_evaluation``: strictly-earlier completed, non-push games
    only. See the module docstring's "ONE inlined piece" note.
    """

    completed = frame.loc[frame["ats_margin"].notna()].copy()
    eligible: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    for (season, week), weekly_games in completed.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff) & completed["home_cover"].notna()]
        key = (int(str(season)), int(str(week)))
        (eligible if len(training) >= min_train_games else skipped).append(key)
    return frozenset(eligible), tuple(sorted(skipped))


def _correct_by_game(
    game_ids: pd.Series, pick_home: pd.Series, settle_margin: pd.Series
) -> dict[str, bool]:
    correct = pick_correct(pick_home, settle_margin)
    ids = game_ids.astype(str)
    return {gid: bool(value) for gid, value in zip(ids, correct, strict=True) if pd.notna(value)}


def _accuracy(correct_by_game: dict[str, bool]) -> float:
    if not correct_by_game:
        return float("nan")
    values = list(correct_by_game.values())
    return sum(values) / len(values)


def _grade_market_only(frame: pd.DataFrame, line: Line, min_train_games: int) -> PathResult:
    """No model fit: the pick comes straight from vig-free spread prices."""

    working = _features_for_line(frame, line)
    eligible_weeks, skipped = _eligible_weeks(working, min_train_games)
    mask = pd.Series(_week_keys(working), index=working.index).isin(eligible_weeks)
    scored = working.loc[mask].copy()

    home_probability = pd.Series(
        [
            no_vig_probabilities(home_odds, away_odds)[0]
            for home_odds, away_odds in zip(
                scored["home_spread_odds"], scored["away_spread_odds"], strict=True
            )
        ],
        index=scored.index,
        dtype=float,
    )
    pick_home = home_probability.ge(0.5)
    correct_by_game = _correct_by_game(scored["game_id"], pick_home, scored["ats_margin"])
    scored_ids = frozenset(scored["game_id"].astype(str))
    return PathResult(
        path="market_only",
        line=line,
        min_train_games=min_train_games,
        push_rule=PUSH_RULE,
        scored_game_ids=scored_ids,
        pushed_game_ids=scored_ids - frozenset(correct_by_game),
        evaluated_game_ids=frozenset(correct_by_game),
        accuracy=_accuracy(correct_by_game),
        skipped_weeks=skipped,
        correct_by_game=correct_by_game,
    )


def _grade_walk_forward(
    frame: pd.DataFrame,
    line: Line,
    feature_set: str,
    min_train_games: int,
    path_name: str,
) -> tuple[PathResult, pd.DataFrame]:
    """Delegate fully to the real backtest harness; grade its output with ``pick_correct``."""

    working = _features_for_line(frame, line)
    start_season = int(working["season"].min())
    end_season = int(working["season"].max())
    result: BacktestResult = walk_forward_backtest(
        working,
        start_season=start_season,
        end_season=end_season,
        model_name="logistic",
        feature_set=feature_set,
        min_edge=0.02,
        min_train_games=min_train_games,
    )
    predictions = result.predictions
    pick_home = predictions["home_cover_probability"].ge(0.5)
    correct_by_game = _correct_by_game(predictions["game_id"], pick_home, predictions["ats_margin"])
    scored_ids = frozenset(predictions["game_id"].astype(str))

    graded_weeks = frozenset(_week_keys(predictions))
    all_weeks = frozenset(_week_keys(working))
    skipped = tuple(sorted(all_weeks - graded_weeks))

    path_result = PathResult(
        path=path_name,
        line=line,
        min_train_games=min_train_games,
        push_rule=PUSH_RULE,
        scored_game_ids=scored_ids,
        pushed_game_ids=scored_ids - frozenset(correct_by_game),
        evaluated_game_ids=frozenset(correct_by_game),
        accuracy=_accuracy(correct_by_game),
        skipped_weeks=skipped,
        correct_by_game=correct_by_game,
    )
    return path_result, predictions


def _grade_overlay(frame: pd.DataFrame, line: Line, min_train_games: int) -> PathResult:
    """Flip ``active_model``'s own predictions with the real coach-fade overlay."""

    active_result, predictions = _grade_walk_forward(
        frame, line, ACTIVE_MODEL_FEATURE_SET, min_train_games, "active_model_for_overlay"
    )
    # The FULL fixture (every season), not just the graded weeks: year-1
    # detection needs each team's PRIOR-season modal coach, which for a 2022
    # graded game lives in the 2021 training-only rows.
    schedules = frame[
        ["game_id", "season", "game_type", "home_team", "away_team", "home_coach", "away_coach"]
    ].copy()
    overlay = apply_coach_fade_overlay(
        predictions, schedules, week_max=OVERLAY_WEEK_MAX, enabled=True
    )
    overlaid = overlay.overlaid_predictions
    pick_home = overlaid["home_cover_probability"].ge(0.5)
    correct_by_game = _correct_by_game(overlaid["game_id"], pick_home, overlaid["ats_margin"])
    scored_ids = frozenset(overlaid["game_id"].astype(str))
    return PathResult(
        path="overlay",
        line=line,
        min_train_games=min_train_games,
        push_rule=PUSH_RULE,
        scored_game_ids=scored_ids,
        pushed_game_ids=scored_ids - frozenset(correct_by_game),
        evaluated_game_ids=frozenset(correct_by_game),
        accuracy=_accuracy(correct_by_game),
        skipped_weeks=active_result.skipped_weeks,
        correct_by_game=correct_by_game,
        flipped_game_ids=tuple(str(flip.game_id) for flip in overlay.flips),
    )


def _production_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature table for the real opener-snapshot machinery: CLOSE-era throughout.

    ``nfl_ats.clv.opener_pick_evaluation`` takes ONE features table (matching
    production's ``game_features_weak_stack.parquet``, which carries a single
    close-era ``spread_line`` per game -- see ``docs/baseline_parity.md``) and
    internally overrides ``spread_line`` per scoring row to the opener value
    for one evaluation pass and the close value for another, fitting exactly
    ONE weekly-refit model per week either way. Unlike ``_features_for_line``
    (used by the walk-forward paths, which genuinely need a per-``line``
    feature table because they call ``walk_forward_backtest`` once per line),
    the production path must NOT be built once per line: doing so would fit a
    different model per line from a training target recomputed against
    whichever line happened to be requested, which is not what the real
    weekly-refit job does. So this is called once, and both ``line="opener"``
    and ``line="close"`` reads for ``path="active_model_production"``/
    ``"overlay_production"`` come from the SAME underlying evaluation.
    """

    working = _features_for_line(frame, "close")
    for column in PRODUCTION_FEATURE_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
    return working


def _grade_active_model_production(
    frame: pd.DataFrame, line: Line, min_train_games: int, opener_store: Path
) -> tuple[PathResult, pd.DataFrame]:
    """Grade through the REAL production opener-snapshot machinery.

    Delegates fully to :func:`nfl_ats.clv.opener_pick_evaluation` -- the same
    weekly-refit ``weak_stack``/ridge-alpha-10 ``market_residual`` model,
    evaluated at both the Tuesday-opener and close lines from a paired
    point-in-time snapshot store, that ``docs/opener_evaluation.md``'s
    production archive is built from. Grades with the PROBABILITY pick rule
    (``home_cover_probability >= 0.5``), which ``opener_pick_evaluation``'s
    own docstring identifies as production's actual pick rule
    (``pool.py``/``backtest.py``), not its predeclared sign rule.

    ``skipped_weeks`` is computed independently by :func:`_eligible_weeks`
    (the same cross-checked cutoff helper every other path uses) rather than
    read off ``opener_pick_evaluation``'s output, which reports scored games
    only, not a skip diagnostic -- this fixture's store only ever provides
    snapshot coverage for the same weeks the other four paths' chronological
    cutoff already selects (see ``tests/fixtures/parity/opener_store``'s
    generator), so the two independent computations agree by construction
    and the agreement is exactly what
    ``test_market_only_cutoff_agrees_with_walk_forward_backtest_own_cutoff``-
    style tests check.
    """

    working = _production_features(frame)
    _eligible, skipped = _eligible_weeks(working, min_train_games)
    scored = opener_pick_evaluation(
        opener_store,
        working,
        active_model_config=PRODUCTION_MODEL_CONFIG,
        min_train_games=min_train_games,
    )
    suffix = "open" if line == "opener" else "close"
    pick_column = f"pick_home_at_{suffix}_probability_rule"
    margin_column = f"margin_vs_{suffix}"
    correct_by_game = _correct_by_game(
        scored["game_id"], scored[pick_column], scored[margin_column]
    )
    scored_ids = frozenset(scored["game_id"].astype(str))
    path_result = PathResult(
        path="active_model_production",
        line=line,
        min_train_games=min_train_games,
        push_rule=PUSH_RULE,
        scored_game_ids=scored_ids,
        pushed_game_ids=scored_ids - frozenset(correct_by_game),
        evaluated_game_ids=frozenset(correct_by_game),
        accuracy=_accuracy(correct_by_game),
        skipped_weeks=skipped,
        correct_by_game=correct_by_game,
    )
    return path_result, scored


def _grade_overlay_production(
    frame: pd.DataFrame, line: Line, min_train_games: int, opener_store: Path
) -> PathResult:
    """Flip ``active_model_production``'s own predictions with the REAL four-overlay policy.

    Calls :func:`nfl_ats.four_overlay_composition.apply_four_overlay_composition`
    -- the same joint-OR union of coach-fade, division-revenge-tilt,
    player-arrests-back-side, and spread-gap-zone-fade the played card
    actually composes -- rather than re-implementing any member's rule. The
    player-arrests member needs a live incident feed and a freshness-checked
    snapshot descriptor in production; this commit-safe fixture supplies an
    EMPTY, schema-valid incident table (no arrests in the miniature window)
    and a synthetic in-memory ``ArrestSnapshot`` built directly (bypassing
    ``load_latest_complete_arrest_snapshot``'s filesystem/freshness checks,
    which have nothing to load here) -- ``apply_four_overlay_composition``
    only reads the snapshot's ``snapshot_id``/``fetched_at_utc``/
    ``safe_index_sha256`` for provenance, never for gating, so this changes
    no grading behavior, only which games the arrests member is eligible to
    flip (none, deterministically, since the incident table is empty).
    """

    active_result, scored = _grade_active_model_production(
        frame, line, min_train_games, opener_store
    )
    suffix = "open" if line == "opener" else "close"
    spread_column = "tue_open_home_spread" if line == "opener" else "close_home_spread"
    schedules = frame[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "game_type",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
            "result",
        ]
    ].copy()
    predictions = scored[["game_id", "season", "week"]].copy()
    predictions["game_id"] = predictions["game_id"].astype(str)
    predictions["home_cover_probability"] = scored[f"home_cover_probability_at_{suffix}"].to_numpy()
    predictions["spread_line"] = scored[spread_column].to_numpy()
    # Carried through untouched by the overlay (only home_cover_probability is
    # ever complemented), so it can be read back after composition to grade
    # the overlaid picks against the SAME settle margin the production path
    # used, without a second lookup.
    margin_column = f"margin_vs_{suffix}"
    predictions["settle_margin"] = scored[margin_column].to_numpy()
    predictions = predictions.merge(
        schedules[["game_id", "gameday", "home_team", "away_team"]], on="game_id", how="left"
    )
    incidents = pd.DataFrame(
        {
            "record_id": pd.Series(dtype="string"),
            "incident_date": pd.Series(dtype="datetime64[ns, UTC]"),
            "team": pd.Series(dtype="string"),
        }
    )
    arrest_snapshot = ArrestSnapshot(
        snapshot_id="parity-fixture-no-incidents",
        directory=opener_store,
        manifest_path=opener_store / "manifest.json",
        safe_index_path=opener_store / "safe_index.json",
        fetched_at_utc=pd.Timestamp("2026-09-04T00:00:00Z"),
        age_hours=0.0,
        safe_index_sha256="0" * 64,
        rows_cached=0,
    )
    composed = apply_four_overlay_composition(
        predictions, schedules, incidents, arrest_snapshot=arrest_snapshot
    )
    overlaid = composed.overlaid_predictions
    pick_home = overlaid["home_cover_probability"].ge(0.5)
    correct_by_game = _correct_by_game(overlaid["game_id"], pick_home, overlaid["settle_margin"])
    scored_ids = frozenset(overlaid["game_id"].astype(str))
    return PathResult(
        path="overlay_production",
        line=line,
        min_train_games=min_train_games,
        push_rule=PUSH_RULE,
        scored_game_ids=scored_ids,
        pushed_game_ids=scored_ids - frozenset(correct_by_game),
        evaluated_game_ids=frozenset(correct_by_game),
        accuracy=_accuracy(correct_by_game),
        skipped_weeks=active_result.skipped_weeks,
        correct_by_game=correct_by_game,
        flipped_game_ids=composed.union_flipped_game_ids,
    )


def grade_games(
    frame: pd.DataFrame,
    line: Line,
    path: PathName,
    *,
    min_train_games: int = MIN_TRAIN_GAMES,
    opener_store: Path | None = None,
) -> PathResult:
    """Grade ``frame`` at ``line`` through the real code path named by ``path``.

    ``opener_store`` (the miniature ``tests/fixtures/parity/opener_store``
    market-snapshot store) is required for ``"active_model_production"`` and
    ``"overlay_production"`` -- the two paths that read a real point-in-time
    opener/close pairing rather than a single per-line feature swap -- and
    unused otherwise, mirroring how ``load_fixture`` also takes its path as a
    caller-supplied argument rather than a hardcoded constant.
    """

    if path == "market_only":
        return _grade_market_only(frame, line, min_train_games)
    if path == "simple_model":
        result, _ = _grade_walk_forward(
            frame, line, SIMPLE_MODEL_FEATURE_SET, min_train_games, "simple_model"
        )
        return result
    if path == "active_model":
        result, _ = _grade_walk_forward(
            frame, line, ACTIVE_MODEL_FEATURE_SET, min_train_games, "active_model"
        )
        return result
    if path == "overlay":
        return _grade_overlay(frame, line, min_train_games)
    if path == "active_model_production":
        if opener_store is None:
            raise ValueError("path 'active_model_production' requires opener_store")
        result, _ = _grade_active_model_production(frame, line, min_train_games, opener_store)
        return result
    if path == "overlay_production":
        if opener_store is None:
            raise ValueError("path 'overlay_production' requires opener_store")
        return _grade_overlay_production(frame, line, min_train_games, opener_store)
    raise ValueError(f"Unknown comparison path {path!r}; choose one of {_PATHS}")


def paired_delta(a: PathResult, b: PathResult) -> dict[str, float | int]:
    """Paired candidate-minus-baseline delta on the INTERSECTION of two paths.

    Never compares two paths' bare ``accuracy`` fields directly -- those can
    be computed over different populations. This recomputes both accuracies
    restricted to games both paths actually evaluated (non-push under both),
    and reports the intersection size so a caller can see how much of either
    population the comparison actually rests on.
    """

    intersection = sorted(set(a.correct_by_game) & set(b.correct_by_game))
    n = len(intersection)
    if n == 0:
        return {
            "intersection_size": 0,
            "a_accuracy": float("nan"),
            "b_accuracy": float("nan"),
            "delta_a_minus_b": float("nan"),
        }
    a_accuracy = sum(a.correct_by_game[game_id] for game_id in intersection) / n
    b_accuracy = sum(b.correct_by_game[game_id] for game_id in intersection) / n
    return {
        "intersection_size": n,
        "a_accuracy": a_accuracy,
        "b_accuracy": b_accuracy,
        "delta_a_minus_b": a_accuracy - b_accuracy,
    }


__all__ = [
    "ACTIVE_MODEL_FEATURE_SET",
    "FIXTURE_COLUMNS",
    "MIN_TRAIN_GAMES",
    "PRODUCTION_FEATURE_COLUMNS",
    "PRODUCTION_FEATURE_PROFILE",
    "PRODUCTION_MODEL_CONFIG",
    "PUSH_RULE",
    "SIMPLE_MODEL_FEATURE_SET",
    "Line",
    "PathName",
    "PathResult",
    "grade_games",
    "load_fixture",
    "paired_delta",
]
