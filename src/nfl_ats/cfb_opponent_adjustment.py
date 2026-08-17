"""CFB dimension-neutral opponent-adjustment substitution screen.

PBP-05 built a weekly, time-decayed ridge opponent adjustment for the NFL
(``nfl_ats.opponent_adjustment``) and it was judged "no stable improvement".
Every recorded test of it, however, **added** the adjusted columns on top of
the raw ones, taking the design from 58 to 106-142 columns on ~4,700 rows at
ridge alpha 10 — which confounds "does the adjustment carry signal" with a
straightforward dimension-inflation variance penalty.

This module runs the test that was never run: the **substitution**. The raw
per-play EPA columns come out, the opponent-adjusted equivalents go in, and
the design width, regressor, penalty, walk-forward, and splits are all held
fixed. It runs on the CFB benchmark (``docs/cfb_data.md``), whose clean core
is 8,933 games, because the frozen NFL evaluation cannot resolve an effect
this size — and because CFB costs no confirmation window
(``docs/rotation_registry.md`` rule 8).

The predeclaration is ``docs/cfb_opponent_adjustment.md``, frozen before any
arm was fit.

Design notes
------------
- **One estimator, shared.** The ridge decomposition and its point-in-time
  eligibility rule are imported from ``nfl_ats.opponent_adjustment``; nothing
  is re-implemented here. A second copy of an estimator is a defect waiting
  to drift.
- **Team ratings, not matchup expectations.** The NFL builder emits
  ``intercept + offense(home) + defense(away)`` — a matchup expectation. That
  is the wrong shape for a substitution: ``home_off`` and ``away_def`` would
  be the same number, collapsing four columns into two. The substituted
  columns are the opponent-purged team ratings
  (``intercept + offense(team)`` and ``intercept + defense(team)``), which
  occupy exactly the semantic slot the raw team-state rates vacate.
- **One fit serves both sides.** The CFB team-game table is closed under
  swapping (``def_epa_per_play`` of a team *is* its opponent's
  ``off_epa_per_play`` in the same game, at the same date and therefore the
  same decay weight), so fitting the defensive metric would solve the
  offensive fit's own problem with the two coefficient blocks transposed.
  One fit per week is therefore exact, not an approximation;
  ``tests/test_cfb_opponent_adjustment.py`` asserts the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.cfb_benchmark import (
    _PREDICTION_PASSTHROUGH,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    _score_week,
    cfb_evaluation_window,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MarginModel, fit_market_baseline
from nfl_ats.opponent_adjustment import (
    eligible_opponent_history,
    fit_opponent_effects,
    opponent_adjustment_weeks,
    validate_opponent_adjustment_parameters,
)

# ---------------------------------------------------------------------------
# Frozen configuration (see docs/cfb_opponent_adjustment.md)
# ---------------------------------------------------------------------------

# NFL PBP-05 values, taken verbatim. Nothing here is tuned on CFB: the point
# of the screen is the substitution framing, not a search over penalties.
CFB_OPPONENT_HALF_LIFE_WEEKS: float = 16.0
CFB_OPPONENT_RIDGE_ALPHA: float = 10.0
CFB_OPPONENT_MIN_TEAM_GAMES: int = 64

# The metric the weekly ridge is fit on. The defensive rating falls out of the
# same fit's opposing-defense block (see the module docstring).
CFB_OPPONENT_SOURCE_METRIC: str = "off_epa_per_play"

# The raw team-state metrics that have an opponent-adjusted equivalent, and
# therefore the only columns this screen substitutes.
CFB_ADJUSTED_METRICS: tuple[str, ...] = ("off_epa_per_play", "def_epa_per_play")
CFB_ADJUSTED_SUFFIX = "_adjusted"

CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{metric}{CFB_ADJUSTED_SUFFIX}"
    for metric in CFB_ADJUSTED_METRICS
    for side in ("home", "away", "diff")
)

# POST-HOC CONTROL, not part of the frozen screen (see the results section of
# docs/cfb_opponent_adjustment.md). The substitution changes two things at
# once: it adjusts for opponent, and it swaps the raw columns' span-8
# exponentially weighted game window for a 16-week calendar half-life. These
# columns hold the second change and drop the first, so the two can be told
# apart after the fact instead of argued about.
CFB_TIME_DECAYED_SUFFIX = "_time_decayed"
CFB_TIME_DECAYED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{metric}{CFB_TIME_DECAYED_SUFFIX}"
    for metric in CFB_ADJUSTED_METRICS
    for side in ("home", "away", "diff")
)

OPPONENT_BENCHMARK_BASELINE_ARM = "cfb_benchmark_v1"
OPPONENT_BENCHMARK_CANDIDATE_ARM = "cfb_benchmark_v1_opponent_adjusted"
OPPONENT_BENCHMARK_BASELINE_METHOD = "market_residual"
OPPONENT_BENCHMARK_CANDIDATE_METHOD = "market_residual_opponent_adjusted"
OPPONENT_BENCHMARK_CONTROL_METHOD = "market_residual_time_decayed"

CFB_OPPONENT_BOOTSTRAP_SAMPLES = 2_000
CFB_OPPONENT_BOOTSTRAP_SEED = 20260817

_HISTORY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "team",
    "opponent",
    "gameday",
    *CFB_ADJUSTED_METRICS,
)


# ---------------------------------------------------------------------------
# The dimension-neutral column substitution
# ---------------------------------------------------------------------------


def opponent_adjusted_substitution(suffix: str = CFB_ADJUSTED_SUFFIX) -> dict[str, str]:
    """Map each substituted raw column to its opponent-adjusted equivalent."""

    return {
        f"{side}_{metric}": f"{side}_{metric}{suffix}"
        for metric in CFB_ADJUSTED_METRICS
        for side in ("home", "away", "diff")
    }


def substitute_opponent_adjusted_columns(
    columns: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS,
    suffix: str = CFB_ADJUSTED_SUFFIX,
) -> tuple[str, ...]:
    """Swap the raw per-play EPA columns for the adjusted ones, in place.

    The substitution is positional and total: every raw column named in the
    map must be present, each is replaced where it stands, and the returned
    contract has exactly the same length as the input. A change to the frozen
    CFB feature contract that drops one of these columns fails loudly here
    rather than silently turning the screen into an ablation.
    """

    mapping = opponent_adjusted_substitution(suffix)
    missing = sorted(set(mapping).difference(columns))
    if missing:
        raise DataContractError(
            f"CFB feature contract is missing substitutable columns: {', '.join(missing)}"
        )
    substituted = tuple(mapping.get(column, column) for column in columns)
    if len(substituted) != len(columns) or len(set(substituted)) != len(set(columns)):
        raise DataContractError("Opponent-adjusted substitution changed the design width")
    return substituted


CFB_OPPONENT_ADJUSTED_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    substitute_opponent_adjusted_columns()
)
# Post-hoc control contract; same width, same positions, no opponent block.
CFB_TIME_DECAYED_MODEL_FEATURE_COLUMNS: tuple[str, ...] = substitute_opponent_adjusted_columns(
    suffix=CFB_TIME_DECAYED_SUFFIX
)


# ---------------------------------------------------------------------------
# Point-in-time adjusted columns
# ---------------------------------------------------------------------------


def build_cfb_opponent_history(team_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Turn the canonical CFB team-game table into adjustment history.

    Teams are ESPN team ids rendered as strings; the game date comes from the
    canonical games table, never from the play-by-play, so the cutoff test and
    the decay weights read the same clock the benchmark does.
    """

    require_columns(
        team_games,
        ("game_id", "season", "week", "team_id", "opponent_id", *CFB_ADJUSTED_METRICS),
        "CFB team games",
    )
    require_columns(games, ("game_id", "gameday"), "CFB games")
    if team_games.duplicated(["game_id", "team_id"]).any():
        raise DataContractError("CFB team games contain duplicate game/team rows")

    history = team_games.loc[
        team_games["game_id"].isin(set(games["game_id"])),
        ["game_id", "season", "week", "team_id", "opponent_id", *CFB_ADJUSTED_METRICS],
    ].copy()
    history = history.rename(columns={"team_id": "team", "opponent_id": "opponent"})
    history["team"] = history["team"].astype(str)
    history["opponent"] = history["opponent"].astype(str)
    if history["team"].eq(history["opponent"]).any():
        raise DataContractError("CFB team games contain a team paired with itself")
    for column in ("season", "week"):
        history[column] = pd.to_numeric(history[column], errors="raise").astype("int64")

    dates = games[["game_id", "gameday"]].drop_duplicates("game_id")
    history = history.merge(dates, on="game_id", how="left", validate="many_to_one")
    if history["gameday"].isna().any():
        raise DataContractError("CFB team games contain games absent from the canonical table")
    history["gameday"] = pd.to_datetime(history["gameday"], errors="raise")
    return history.loc[:, list(_HISTORY_COLUMNS)].reset_index(drop=True)


def _weekly_ratings(
    eligible: pd.DataFrame,
    *,
    teams: tuple[str, ...],
    cutoff: pd.Timestamp,
    half_life_weeks: float,
    ridge_alpha: float,
    min_team_games: int,
    include_opponent: bool,
) -> dict[str, dict[str, float]] | None:
    """One week's per-team offensive and defensive ratings.

    With the opponent block in, a single fit yields both: the offense block is
    each team's opponent-purged offensive rate and the opposing-defense block
    is its defensive rate. Without it, the defensive rate has to come from its
    own team-only fit on the defensive metric, because there is no longer a
    second block to read it out of.
    """

    fit = partial(
        fit_opponent_effects,
        eligible,
        teams=teams,
        cutoff=cutoff,
        half_life_weeks=half_life_weeks,
        ridge_alpha=ridge_alpha,
        min_team_games=min_team_games,
        include_opponent=include_opponent,
    )
    offensive = fit(metric=CFB_OPPONENT_SOURCE_METRIC)
    if offensive is None:
        return None
    if include_opponent:
        return {
            "off_epa_per_play": {team: offensive.offense_rating(team) for team in teams},
            "def_epa_per_play": {team: offensive.defense_rating(team) for team in teams},
        }
    defensive = fit(metric="def_epa_per_play")
    if defensive is None:
        return None
    return {
        "off_epa_per_play": {team: offensive.offense_rating(team) for team in teams},
        "def_epa_per_play": {team: defensive.offense_rating(team) for team in teams},
    }


def add_cfb_opponent_adjusted_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    half_life_weeks: float = CFB_OPPONENT_HALF_LIFE_WEEKS,
    ridge_alpha: float = CFB_OPPONENT_RIDGE_ALPHA,
    min_team_games: int = CFB_OPPONENT_MIN_TEAM_GAMES,
    include_opponent: bool = True,
) -> pd.DataFrame:
    """Attach opponent-adjusted per-play EPA ratings to the CFB game table.

    One weighted ridge per week decomposes observed offensive EPA/play into an
    offense effect and an opposing-defense effect, fit only from strictly
    earlier weeks and strictly earlier game dates (the NFL point-in-time
    contract, imported rather than re-stated). Each team's rating is the
    league intercept plus its own effect, so the columns land on the same
    per-play EPA scale as the raw team-state columns they replace. Weeks
    whose eligible history is thinner than ``min_team_games`` are left NaN.

    ``include_opponent=False`` writes the ``_time_decayed`` control columns
    instead: same cutoff, same decay, same penalty, no opponent block. It is a
    post-hoc control, never the screen.
    """

    validate_opponent_adjustment_parameters(
        half_life_weeks=half_life_weeks,
        ridge_alpha=ridge_alpha,
        min_team_games=min_team_games,
    )
    require_columns(
        games,
        ("game_id", "season", "week", "gameday", "home_id", "away_id"),
        "CFB game features",
    )

    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    if result[["home_id", "away_id"]].isna().any(axis=None):
        raise DataContractError("CFB game features contain a null home or away team id")
    home_teams = result["home_id"].astype("int64").astype(str)
    away_teams = result["away_id"].astype("int64").astype(str)

    history = build_cfb_opponent_history(team_games, result)
    teams = tuple(
        sorted(set(history["team"]) | set(history["opponent"]) | set(home_teams) | set(away_teams))
    )
    suffix = CFB_ADJUSTED_SUFFIX if include_opponent else CFB_TIME_DECAYED_SUFFIX
    for metric in CFB_ADJUSTED_METRICS:
        for side in ("home", "away", "diff"):
            result[f"{side}_{metric}{suffix}"] = np.nan

    for week in opponent_adjustment_weeks(result).itertuples(index=False):
        season = int(str(week.season))
        week_number = int(str(week.week))
        cutoff = pd.Timestamp(str(week.cutoff))
        eligible = eligible_opponent_history(
            history, season=season, week=week_number, cutoff=cutoff
        )
        ratings = _weekly_ratings(
            eligible,
            teams=teams,
            cutoff=cutoff,
            half_life_weeks=half_life_weeks,
            ridge_alpha=ridge_alpha,
            min_team_games=min_team_games,
            include_opponent=include_opponent,
        )
        if ratings is None:
            continue
        target = result.index[result["season"].eq(season) & result["week"].eq(week_number)]
        home = home_teams.loc[target]
        away = away_teams.loc[target]
        for metric, values in ratings.items():
            home_values = home.map(values).astype(float)
            away_values = away.map(values).astype(float)
            result.loc[target, f"home_{metric}{suffix}"] = home_values
            result.loc[target, f"away_{metric}{suffix}"] = away_values
            result.loc[target, f"diff_{metric}{suffix}"] = home_values - away_values
    return result


# ---------------------------------------------------------------------------
# Paired margin-error uncertainty (the primary metric)
# ---------------------------------------------------------------------------


def paired_margin_error_comparison(
    predictions: pd.DataFrame,
    *,
    baseline_method: str,
    candidate_method: str,
    samples: int = CFB_OPPONENT_BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    block: Literal["week", "season"] = "week",
    seed: int = CFB_OPPONENT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Blocked paired bootstrap of margin MAE/RMSE improvement.

    ``experiments.paired_feature_comparisons`` covers the probability metrics
    only. Margin error is the primary metric here because it is continuous and
    resolves far smaller effects than accuracy at any realistic n, so it gets
    the same blocked pairing treatment: the two arms keep the exact same games
    and whole weeks (or seasons) are resampled together. Positive values mean
    the candidate is better.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    require_columns(
        predictions,
        ("method", "game_id", "season", "week", "result", "predicted_margin"),
        "paired margin predictions",
    )

    columns = ["game_id", "season", "week", "result", "predicted_margin"]
    frames = {}
    for name in (baseline_method, candidate_method):
        subset = predictions.loc[predictions["method"].eq(name), columns]
        if subset.empty:
            raise ValueError(f"No predictions for method {name!r}")
        frames[name] = subset
    paired = frames[baseline_method].merge(
        frames[candidate_method],
        on="game_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    for column in ("season", "week", "result"):
        if not paired[f"{column}_baseline"].equals(paired[f"{column}_candidate"]):
            raise ValueError(f"Paired {column} values differ between the arms")
    actual = pd.to_numeric(paired["result_baseline"], errors="coerce").to_numpy(dtype=float)
    baseline_error = (
        pd.to_numeric(paired["predicted_margin_baseline"], errors="coerce").to_numpy(dtype=float)
        - actual
    )
    candidate_error = (
        pd.to_numeric(paired["predicted_margin_candidate"], errors="coerce").to_numpy(dtype=float)
        - actual
    )
    finite = np.isfinite(baseline_error) & np.isfinite(candidate_error)
    paired = paired.loc[finite].reset_index(drop=True)
    baseline_error = baseline_error[finite]
    candidate_error = candidate_error[finite]
    if not len(paired):
        raise ValueError("No paired games with a finite margin error in both arms")

    baseline_absolute = np.abs(baseline_error)
    candidate_absolute = np.abs(candidate_error)
    baseline_squared = np.square(baseline_error)
    candidate_squared = np.square(candidate_error)

    group_columns = ["season_baseline", "week_baseline"] if block == "week" else ["season_baseline"]
    blocks = list(paired.groupby(group_columns, sort=False).indices.values())
    generator = np.random.default_rng(seed)
    mae_draws = np.empty(samples, dtype=float)
    rmse_draws = np.empty(samples, dtype=float)
    for draw in range(samples):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[index] for index in selected])
        mae_draws[draw] = float(
            baseline_absolute[positions].mean() - candidate_absolute[positions].mean()
        )
        rmse_draws[draw] = float(
            np.sqrt(baseline_squared[positions].mean())
            - np.sqrt(candidate_squared[positions].mean())
        )

    tail = (1.0 - confidence) / 2.0
    estimates = {
        "margin_mae_improvement": (
            float(baseline_absolute.mean() - candidate_absolute.mean()),
            mae_draws,
        ),
        "margin_rmse_improvement": (
            float(np.sqrt(baseline_squared.mean()) - np.sqrt(candidate_squared.mean())),
            rmse_draws,
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric, (estimate, draws) in estimates.items():
        rows.append(
            {
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "metric": metric,
                "estimate": estimate,
                "lower": float(np.quantile(draws, tail)),
                "upper": float(np.quantile(draws, 1.0 - tail)),
                # Continuous evidence, never a bare pass/fail: the fraction of
                # blocked resamples in which the candidate beats the baseline.
                "probability_positive": float(np.mean(draws > 0.0)),
                "games": len(paired),
                "blocks": len(blocks),
                "block": block,
                "samples": int(samples),
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfbOpponentAdjustmentResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame
    paired_margin: pd.DataFrame
    paired_probability: pd.DataFrame
    diagnostics: dict[str, Any]


def _missing_rates(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    return {
        column: float(pd.to_numeric(frame[column], errors="coerce").isna().mean())
        for column in columns
    }


def _effective_design_width(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    """Input width plus the missing-indicator columns the imputer will add.

    ``make_margin_estimator`` uses ``SimpleImputer(add_indicator=True)``, so a
    column that is missing anywhere in the training frame contributes a second
    column to the fitted design. Identical *input* width does not guarantee
    identical *effective* width, and pretending otherwise would undercut the
    dimension-neutrality claim this screen rests on.
    """

    indicators = sum(
        1 for column in columns if pd.to_numeric(frame[column], errors="coerce").isna().any()
    )
    return len(columns) + indicators


def cfb_opponent_adjustment_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    bootstrap_samples: int = CFB_OPPONENT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = CFB_OPPONENT_BOOTSTRAP_SEED,
    candidate_columns: tuple[str, ...] = CFB_OPPONENT_ADJUSTED_MODEL_FEATURE_COLUMNS,
    candidate_method: str = OPPONENT_BENCHMARK_CANDIDATE_METHOD,
) -> CfbOpponentAdjustmentResult:
    """Walk-forward the baseline and substituted arms on identical weeks.

    Both residual arms use the same regressor, the same alpha, the same
    residual-distribution recipe, the same training rows, and the same scored
    games; only the six substituted columns differ. ``candidate_columns`` and
    ``candidate_method`` default to the frozen screen and are overridden only
    by the post-hoc time-decay control, which reuses the identical harness so
    the two candidates stay comparable.
    """

    baseline_columns = CFB_MODEL_FEATURE_COLUMNS
    if len(baseline_columns) != len(candidate_columns):
        raise DataContractError("The arms do not have identical design widths")
    required = {*_PREDICTION_PASSTHROUGH, *baseline_columns, *candidate_columns}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB opponent-adjusted features are missing columns: {', '.join(missing)}"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    scored_weeks = 0
    last_training: pd.DataFrame | None = None
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models: dict[str, MarginModel] = {
            "market": fit_market_baseline(training),
            OPPONENT_BENCHMARK_BASELINE_METHOD: fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=baseline_columns
            ),
            candidate_method: fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=candidate_columns
            ),
        }
        batches.extend(_score_week(weekly_games, models))
        scored_weeks += 1
        last_training = training
    if not batches or last_training is None:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)

    clean = predictions.loc[predictions["evaluation_window"].eq("clean_core")].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    paired_margin = pd.concat(
        [
            paired_margin_error_comparison(
                clean,
                baseline_method=OPPONENT_BENCHMARK_BASELINE_METHOD,
                candidate_method=candidate_method,
                samples=bootstrap_samples,
                block=block,
                seed=bootstrap_seed,
            )
            for block in ("week", "season")
        ],
        ignore_index=True,
    )

    probability_rows = clean.loc[
        clean["method"].isin((OPPONENT_BENCHMARK_BASELINE_METHOD, candidate_method))
    ].copy()
    probability_rows["feature_set"] = probability_rows["method"].map(
        {
            OPPONENT_BENCHMARK_BASELINE_METHOD: OPPONENT_BENCHMARK_BASELINE_ARM,
            candidate_method: OPPONENT_BENCHMARK_CANDIDATE_ARM,
        }
    )
    paired_probability = pd.concat(
        [
            paired_feature_comparisons(
                probability_rows,
                baseline_feature_set=OPPONENT_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block=block,
                seed=bootstrap_seed,
            ).assign(block=block)
            for block in ("week", "season")
        ],
        ignore_index=True,
    )

    clean_games = completed.loc[completed["game_id"].isin(set(clean["game_id"]))]
    substitution = {
        raw: candidate
        for raw, candidate in zip(baseline_columns, candidate_columns, strict=True)
        if raw != candidate
    }
    diagnostics: dict[str, Any] = {
        "candidate_method": candidate_method,
        "scored_weeks": scored_weeks,
        "scored_games": int(predictions["game_id"].nunique()),
        "clean_core_games": int(clean["game_id"].nunique()),
        "baseline_input_columns": len(baseline_columns),
        "candidate_input_columns": len(candidate_columns),
        "substituted_columns": substitution,
        "final_training_rows": len(last_training),
        "baseline_effective_design_width": _effective_design_width(last_training, baseline_columns),
        "candidate_effective_design_width": _effective_design_width(
            last_training, candidate_columns
        ),
        "clean_core_missing_rate_raw": _missing_rates(clean_games, tuple(substitution)),
        "clean_core_missing_rate_adjusted": _missing_rates(
            clean_games, tuple(substitution.values())
        ),
        "adjusted_column_correlation": _adjusted_correlations(clean_games, substitution),
    }
    return CfbOpponentAdjustmentResult(
        predictions=predictions,
        summary=summary,
        season_summary=season_summary,
        paired_margin=paired_margin,
        paired_probability=paired_probability,
        diagnostics=diagnostics,
    )


def _adjusted_correlations(frame: pd.DataFrame, substitution: dict[str, str]) -> dict[str, float]:
    """Pearson correlation between each raw column and its adjusted twin."""

    correlations: dict[str, float] = {}
    for raw, adjusted in substitution.items():
        left: npt.NDArray[np.float64] = pd.to_numeric(frame[raw], errors="coerce").to_numpy(
            dtype=float
        )
        right: npt.NDArray[np.float64] = pd.to_numeric(frame[adjusted], errors="coerce").to_numpy(
            dtype=float
        )
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 2:
            correlations[raw] = float("nan")
            continue
        correlations[raw] = float(np.corrcoef(left[mask], right[mask])[0, 1])
    return correlations
