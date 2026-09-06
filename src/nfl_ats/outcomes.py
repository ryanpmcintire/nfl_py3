"""Walk-forward comparison of straight-up, ATS, and margin forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from nfl_ats.backtest import summarize_predictions
from nfl_ats.calibration import ResidualSmoothingMethod
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.estimation_variance import MIN_BLOCKS_FOR_INTERVAL, OnDegenerate, guard_block_count
from nfl_ats.key_numbers import DEFAULT_KEY_NUMBERS, implied_key_number_mass
from nfl_ats.margin import (
    DEFAULT_LINE_SWEEP_OFFSETS,
    MARGIN_FEATURE_PROFILES,
    MarginFeatureProfile,
    MarginModel,
    fit_margin_model,
    fit_market_baseline,
    margin_feature_set,
)
from nfl_ats.modeling import (
    CoverModel,
    fit_cover_model,
    regular_season_rows,
    validate_model_frame,
)
from nfl_ats.odds import choose_bet, settle_bet
from nfl_ats.prediction_safety import validate_outcome_prediction_card

OUTCOME_METHODS = (
    "market",
    "fair_margin",
    "market_residual",
    "straight_up",
    "direct_ats",
)
OutcomeMethod = Literal["market", "fair_margin", "market_residual", "straight_up", "direct_ats"]

OUTCOME_UNCERTAINTY_METRICS = (
    "win_accuracy",
    "win_brier_score",
    "margin_mae",
    "margin_rmse",
    "cover_accuracy",
    "cover_brier_score",
    "cover_log_loss",
    "roi",
)

_OUTCOME_BOOTSTRAP_STATS = (
    "win_correct",
    "win_brier",
    "win_count",
    "margin_absolute_error",
    "margin_squared_error",
    "margin_count",
    "cover_correct",
    "cover_brier",
    "cover_log_loss",
    "cover_count",
    "profit_units",
    "bet_count",
)


@dataclass(frozen=True)
class OutcomeBacktestResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame


def _straight_up_training(frame: pd.DataFrame) -> pd.DataFrame:
    training = frame.loc[pd.to_numeric(frame["result"], errors="coerce").notna()].copy()
    training = training.loc[training["result"].ne(0)].copy()
    training["home_cover"] = training["result"].gt(0).astype(float)
    return training


def _fit_week_models(
    training: pd.DataFrame,
    *,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    methods: tuple[OutcomeMethod, ...],
    ridge_alpha: float,
) -> tuple[dict[str, MarginModel], CoverModel | None, CoverModel | None]:
    margin_models: dict[str, MarginModel] = {}
    if "market" in methods:
        margin_models["market"] = fit_market_baseline(training)
    if "fair_margin" in methods:
        margin_models["fair_margin"] = fit_margin_model(
            training,
            target="margin",
            model_name=regressor,
            feature_profile=feature_profile,
            ridge_alpha=ridge_alpha,
        )
    if "market_residual" in methods:
        margin_models["market_residual"] = fit_margin_model(
            training,
            target="market_residual",
            model_name=regressor,
            feature_profile=feature_profile,
            ridge_alpha=ridge_alpha,
        )
    straight_up = (
        fit_cover_model(
            _straight_up_training(training),
            model_name="logistic",
            feature_set=margin_feature_set("margin", feature_profile),
        )
        if "straight_up" in methods
        else None
    )
    direct_ats = (
        fit_cover_model(
            training,
            model_name="logistic",
            feature_set=margin_feature_set("market_residual", feature_profile),
        )
        if "direct_ats" in methods
        else None
    )
    return margin_models, straight_up, direct_ats


def normalize_outcome_methods(methods: tuple[str, ...]) -> tuple[OutcomeMethod, ...]:
    """Validate and canonicalize an outcome-method subset."""

    if not methods:
        raise ValueError("At least one outcome method is required")
    unknown = sorted(set(methods).difference(OUTCOME_METHODS))
    if unknown:
        raise ValueError(f"Unknown outcome methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError("Outcome methods must be unique")
    selected = set(methods)
    return cast(
        tuple[OutcomeMethod, ...], tuple(method for method in OUTCOME_METHODS if method in selected)
    )


def _add_decisions(frame: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    result = frame.copy()
    if result["home_cover_probability"].isna().all():
        result["bet_side"] = "PASS"
        result["edge"] = np.nan
        result["bet_odds"] = np.nan
        result["break_even_probability"] = np.nan
        return result
    decisions = [
        (
            choose_bet(
                float(row["home_cover_probability"]),
                row.get("home_spread_odds"),
                row.get("away_spread_odds"),
                min_edge=min_edge,
            )
            if pd.notna(row["home_cover_probability"])
            else None
        )
        for _, row in result.iterrows()
    ]
    result["bet_side"] = [
        decision.side if decision is not None else "PASS" for decision in decisions
    ]
    result["edge"] = [decision.edge if decision is not None else np.nan for decision in decisions]
    result["bet_odds"] = [
        decision.odds if decision is not None else np.nan for decision in decisions
    ]
    result["break_even_probability"] = [
        decision.break_even_probability if decision is not None else np.nan
        for decision in decisions
    ]
    return result


def _score_methods(
    games: pd.DataFrame,
    margin_models: dict[str, MarginModel],
    straight_up: CoverModel | None,
    direct_ats: CoverModel | None,
    min_edge: float,
    probability_method: ResidualSmoothingMethod = "ecdf",
) -> list[pd.DataFrame]:
    batches: list[pd.DataFrame] = []
    for method, model in margin_models.items():
        batch = games.copy()
        forecasts = model.predict(batch, probability_method=probability_method)
        for column in forecasts:
            batch[column] = forecasts[column]
        batch["method"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        batch["train_max_gameday"] = model.training_max_gameday
        batches.append(_add_decisions(batch, min_edge))

    if straight_up is not None:
        straight = games.copy()
        straight["method"] = "straight_up"
        straight["model_name"] = "logistic"
        straight["home_win_probability"] = straight_up.predict_home_cover(straight)
        straight["home_cover_probability"] = np.nan
        straight["home_cover_probability_excluding_push"] = np.nan
        straight["push_probability"] = np.nan
        straight["home_loss_probability"] = np.nan
        straight["predicted_margin"] = np.nan
        straight["fair_spread"] = np.nan
        straight["predicted_market_residual"] = np.nan
        straight["margin_lower_50"] = np.nan
        straight["margin_upper_50"] = np.nan
        straight["margin_lower_80"] = np.nan
        straight["margin_upper_80"] = np.nan
        straight["train_rows"] = straight_up.training_rows
        straight["distribution_rows"] = 0
        straight["train_max_gameday"] = straight_up.training_max_gameday
        batches.append(_add_decisions(straight, min_edge))

    if direct_ats is not None:
        ats = games.copy()
        ats["method"] = "direct_ats"
        ats["model_name"] = "logistic"
        ats["home_cover_probability"] = direct_ats.predict_home_cover(ats)
        ats["home_cover_probability_excluding_push"] = np.nan
        ats["push_probability"] = np.nan
        ats["home_loss_probability"] = np.nan
        ats["home_win_probability"] = np.nan
        ats["predicted_margin"] = np.nan
        ats["fair_spread"] = np.nan
        ats["predicted_market_residual"] = np.nan
        ats["margin_lower_50"] = np.nan
        ats["margin_upper_50"] = np.nan
        ats["margin_lower_80"] = np.nan
        ats["margin_upper_80"] = np.nan
        ats["train_rows"] = direct_ats.training_rows
        ats["distribution_rows"] = 0
        ats["train_max_gameday"] = direct_ats.training_max_gameday
        batches.append(_add_decisions(ats, min_edge))
    return batches


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, probability >= 0.5)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
    }


def summarize_outcome_method(predictions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": str(predictions["method"].iloc[0]),
        "model_name": str(predictions["model_name"].iloc[0]),
        "games": len(predictions),
    }
    win_rows = predictions.loc[
        predictions["result"].notna()
        & predictions["result"].ne(0)
        & predictions["home_win_probability"].notna()
    ]
    if not win_rows.empty:
        win_actual = win_rows["result"].gt(0).to_numpy(dtype=int)
        win_probability = win_rows["home_win_probability"].to_numpy(dtype=float)
        win_metrics = _probability_metrics(win_actual, win_probability)
        result.update({f"win_{key}": value for key, value in win_metrics.items()})
        result["win_games"] = len(win_rows)

    cover_rows = predictions.loc[
        predictions["home_cover"].notna() & predictions["home_cover_probability"].notna()
    ]
    if not cover_rows.empty:
        cover_metrics = summarize_predictions(cover_rows)
        result.update(
            {
                "cover_games": cover_metrics["games_evaluated"],
                "cover_accuracy": cover_metrics["accuracy"],
                "cover_brier_score": cover_metrics["brier_score"],
                "cover_log_loss": cover_metrics["log_loss"],
                "cover_ece": cover_metrics["expected_calibration_error"],
                "bets": cover_metrics["bets"],
                "roi": cover_metrics["roi"],
                "net_profit_units": cover_metrics["net_profit_units"],
            }
        )

    margin_rows = predictions.loc[
        predictions["result"].notna() & predictions["predicted_margin"].notna()
    ].copy()
    if not margin_rows.empty:
        error = margin_rows["predicted_margin"] - margin_rows["result"]
        result.update(
            {
                "margin_games": len(margin_rows),
                "margin_mae": float(error.abs().mean()),
                "margin_rmse": float(np.sqrt(np.square(error).mean())),
                "margin_bias": float(error.mean()),
                "interval_50_coverage": float(
                    margin_rows["result"]
                    .between(margin_rows["margin_lower_50"], margin_rows["margin_upper_50"])
                    .mean()
                ),
                "interval_80_coverage": float(
                    margin_rows["result"]
                    .between(margin_rows["margin_lower_80"], margin_rows["margin_upper_80"])
                    .mean()
                ),
                "fair_spread_edge_mean": float(margin_rows["predicted_market_residual"].mean()),
                "fair_spread_edge_abs_mean": float(
                    margin_rows["predicted_market_residual"].abs().mean()
                ),
            }
        )
    return result


def _summary_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in predictions.groupby(grouper, sort=True):
        values = (keys,) if not isinstance(keys, tuple) else keys
        prefix = dict(zip(group_columns, values, strict=True))
        rows.append({**prefix, **summarize_outcome_method(group)})
    return pd.DataFrame(rows)


def walk_forward_outcomes(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int | None = None,
    regressor: str = "ridge",
    min_edge: float = 0.02,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    feature_profile: MarginFeatureProfile = "base",
    methods: tuple[str, ...] = OUTCOME_METHODS,
    ridge_alpha: float = 10.0,
    # Default unchanged (2026-08-19, MOD-08 promotion): this walk-forward
    # backs every historical/research backtest (margin-backtest CLI, player
    # ablations, experiment comparisons), so it stays on the raw ECDF unless
    # a caller explicitly asks for "gaussian" -- e.g. to build a matching
    # ``margins/`` evaluation for a Gaussian-mapped weekly forecast to
    # synchronize against. See ``score_outcome_week``, the one caller whose
    # OWN default did change.
    probability_method: ResidualSmoothingMethod = "ecdf",
) -> OutcomeBacktestResult:
    if feature_profile not in MARGIN_FEATURE_PROFILES:
        raise ValueError(f"Unknown outcome feature profile: {feature_profile}")
    if end_season is not None and end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    selected_methods = normalize_outcome_methods(methods)
    validate_model_frame(features)
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()
    test_mask = completed["season"].ge(start_season)
    if end_season is not None:
        test_mask &= completed["season"].le(end_season)
    test = completed.loc[test_mask]
    if test.empty:
        end_label = f" through {end_season}" if end_season is not None else ""
        raise ValueError(f"No completed games found from season {start_season}{end_label}")

    batches: list[pd.DataFrame] = []
    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = _fit_week_models(
            training,
            regressor=regressor,
            feature_profile=feature_profile,
            methods=selected_methods,
            ridge_alpha=ridge_alpha,
        )
        weekly_predictions = pd.concat(
            _score_methods(weekly_games, *models, min_edge, probability_method),
            ignore_index=True,
        )
        validate_outcome_prediction_card(
            weekly_predictions,
            min_edge=min_edge,
            expected_methods=selected_methods,
            expected_season=int(str(season)),
            expected_week=int(str(week)),
        )
        batches.append(weekly_predictions)
    if not batches:
        raise ValueError("No outcome-model window had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True).sort_values(
        ["gameday", "game_id", "method"]
    )
    summary = _summary_table(predictions, ["method"])
    season_summary = _summary_table(predictions, ["method", "season"])
    return OutcomeBacktestResult(
        predictions=predictions.reset_index(drop=True),
        summary=summary.sort_values("method").reset_index(drop=True),
        season_summary=season_summary.sort_values(["method", "season"]).reset_index(drop=True),
    )


def _target_and_models_for_week(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str,
    min_train_games: int,
    feature_profile: MarginFeatureProfile,
    ridge_alpha: float,
    methods: tuple[OutcomeMethod, ...],
) -> tuple[pd.DataFrame, dict[str, MarginModel], CoverModel | None, CoverModel | None]:
    """Leak-safe target games and models trained strictly before the target week.

    Shared by ``score_outcome_week`` and ``score_outcome_week_line_sweep`` so
    both use the exact same training cutoff and fitted models -- the line
    sweep is a re-evaluation of the same walk-forward fit at alternative
    lines, never a fresh fit with a different (potentially leaky) cutoff.
    """

    if feature_profile not in MARGIN_FEATURE_PROFILES:
        raise ValueError(f"Unknown outcome feature profile: {feature_profile}")
    validate_model_frame(features)
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
    if target.empty:
        raise ValueError(f"No games found for {season} week {week}")
    if target["spread_line"].isna().any():
        raise ValueError("Cannot score outcome models while a target spread is missing")
    cutoff = target["gameday"].min()
    training = regular_season_rows(frame)
    training = training.loc[training["gameday"].lt(cutoff) & training["result"].notna()].copy()
    if len(training) < min_train_games:
        raise ValueError(
            f"Only {len(training)} eligible games precede the target; need {min_train_games}"
        )
    models = _fit_week_models(
        training,
        regressor=regressor,
        feature_profile=feature_profile,
        methods=methods,
        ridge_alpha=ridge_alpha,
    )
    return (target, *models)


def score_outcome_week(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str = "ridge",
    min_edge: float = 0.02,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    feature_profile: MarginFeatureProfile = "base",
    ridge_alpha: float = 10.0,
    # PROMOTED DEFAULT (MOD-08, 2026-08-19, docs/smooth_cdf_mapping.md): this
    # is the SOLE production entry point for the weekly forecast card (its
    # only caller is the ``margin-predict`` CLI command, which every real
    # weekly-run and publish-predictions call goes through) -- everywhere
    # else in this module keeps the "ecdf" default so historical backtests
    # never silently move. Opener-grade decision measurement:
    # probability_positive 0.5536 (production pick rule, week-blocked,
    # n=1,503); frozen rule fired PROMOTE. Pinned by
    # tests/test_probability_method_promotion.py so an accidental revert to
    # "ecdf" here is caught by CI, not discovered in production.
    probability_method: ResidualSmoothingMethod = "gaussian",
) -> pd.DataFrame:
    target, margin_models, straight_up, direct_ats = _target_and_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=normalize_outcome_methods(OUTCOME_METHODS),
    )
    predictions = pd.concat(
        _score_methods(
            target, margin_models, straight_up, direct_ats, min_edge, probability_method
        ),
        ignore_index=True,
    ).sort_values(["game_id", "method"])
    validate_outcome_prediction_card(
        predictions,
        min_edge=min_edge,
        expected_methods=OUTCOME_METHODS,
        expected_season=season,
        expected_week=week,
    )
    return predictions


MARGIN_DISTRIBUTION_METHODS: tuple[str, ...] = ("market", "fair_margin", "market_residual")


def fit_margin_models_for_week(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str = "ridge",
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    feature_profile: MarginFeatureProfile = "base",
    ridge_alpha: float = 10.0,
    methods: tuple[str, ...] = MARGIN_DISTRIBUTION_METHODS,
) -> tuple[pd.DataFrame, dict[str, MarginModel]]:
    """Target games and fitted margin-distribution models for one week.

    A thin, public entry point over the same leak-safe cutoff logic
    ``score_outcome_week`` uses, for callers (such as re-scoring at
    externally supplied lines) that need the fitted ``MarginModel`` objects
    themselves rather than a pre-summarized prediction card.
    """

    unknown = sorted(set(methods).difference(MARGIN_DISTRIBUTION_METHODS))
    if unknown:
        raise ValueError(f"Unknown margin-distribution methods: {', '.join(unknown)}")
    selected = tuple(method for method in MARGIN_DISTRIBUTION_METHODS if method in methods)
    if not selected:
        raise ValueError("At least one margin-distribution method is required")
    target, margin_models, _, _ = _target_and_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=normalize_outcome_methods(selected),
    )
    return target, margin_models


def score_outcome_week_line_sweep(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str = "ridge",
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    feature_profile: MarginFeatureProfile = "base",
    ridge_alpha: float = 10.0,
    offsets: Sequence[float] = DEFAULT_LINE_SWEEP_OFFSETS,
    methods: tuple[str, ...] = MARGIN_DISTRIBUTION_METHODS,
    probability_method: ResidualSmoothingMethod = "gaussian",
) -> pd.DataFrame:
    """Line-sweep confidence curves for one week's margin-distribution methods.

    Fits the same walk-forward models ``score_outcome_week`` would (cutoff
    strictly before the target week's earliest kickoff) and evaluates each
    margin-based method's predictive distribution across a grid of
    alternative home spreads. Straight-up and direct-ATS methods have no
    margin distribution to sweep and are excluded even if requested.

    Returns a tidy table with one row per (method, game, alternative line).
    """

    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=methods,
    )
    frames: list[pd.DataFrame] = []
    for method, model in margin_models.items():
        sweep = model.line_sweep(target, offsets=offsets, probability_method=probability_method)
        sweep.insert(0, "method", method)
        frames.append(sweep)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["method", "game_id", "line_offset"])
        .reset_index(drop=True)
    )


def walk_forward_key_number_mass(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int | None = None,
    regressor: str = "ridge",
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    feature_profile: MarginFeatureProfile = "base",
    methods: tuple[str, ...] = MARGIN_DISTRIBUTION_METHODS,
    key_numbers: Sequence[int] = DEFAULT_KEY_NUMBERS,
    ridge_alpha: float = 10.0,
) -> pd.DataFrame:
    """Leak-safe walk-forward implied key-number mass, one row per game/method.

    Mirrors ``walk_forward_outcomes``'s weekly cutoff exactly -- each week's
    models are trained strictly on games before that week's earliest
    kickoff -- but additionally records each game's implied probability mass
    on the key numbers, which the summarized outcome card does not retain.
    Intended for ``key_numbers.summarize_key_number_calibration``, a
    validation report rather than a model-selection signal.
    """

    unknown = sorted(set(methods).difference(MARGIN_DISTRIBUTION_METHODS))
    if unknown:
        raise ValueError(f"Unknown margin-distribution methods: {', '.join(unknown)}")
    selected = normalize_outcome_methods(
        tuple(method for method in MARGIN_DISTRIBUTION_METHODS if method in methods)
    )
    if not selected:
        raise ValueError("At least one margin-distribution method is required")
    if end_season is not None and end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    validate_model_frame(features)
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()
    test_mask = completed["season"].ge(start_season)
    if end_season is not None:
        test_mask &= completed["season"].le(end_season)
    test = completed.loc[test_mask]
    if test.empty:
        end_label = f" through {end_season}" if end_season is not None else ""
        raise ValueError(f"No completed games found from season {start_season}{end_label}")

    batches: list[pd.DataFrame] = []
    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        margin_models, _, _ = _fit_week_models(
            training,
            regressor=regressor,
            feature_profile=feature_profile,
            methods=selected,
            ridge_alpha=ridge_alpha,
        )
        for method, model in margin_models.items():
            mass = implied_key_number_mass(model.distribution(weekly_games), key_numbers)
            mass.insert(0, "method", method)
            mass.insert(1, "game_id", weekly_games["game_id"].to_numpy())
            mass.insert(2, "season", int(str(season)))
            mass.insert(3, "week", int(str(week)))
            mass["result"] = weekly_games["result"].to_numpy()
            mass["spread_line"] = weekly_games["spread_line"].to_numpy()
            batches.append(mass)
    if not batches:
        raise ValueError("No walk-forward window had enough prior training games")
    return (
        pd.concat(batches, ignore_index=True)
        .sort_values(["method", "season", "week", "game_id"])
        .reset_index(drop=True)
    )


def outcome_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    # Was 1,000 -- the noisiest of the three bootstrap paths. This one is
    # already vectorized, so raising it is nearly free. See
    # paired_feature_comparisons for why seed jitter matters here.
    samples: int = 20_000,
    confidence: float = 0.95,
    block: Literal["week", "season"] = "week",
    seed: int = 20260812,
    # D4 guard. Default 'warn' + a flagged output column, never 'raise':
    # refusing would change what existing call sites return, and the point is
    # that the flag TRAVELS with the number into the CSV a registry entry cites.
    # A caller that is about to record a verdict should pass 'raise'.
    on_degenerate: OnDegenerate = "warn",
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
) -> pd.DataFrame:
    """Block-bootstrap methods and market deltas from sufficient statistics.

    Every reported metric is reducible to sums and counts inside a week or
    season. Aggregating those contributions once and matrix-multiplying sampled
    block counts is exactly equivalent to repeatedly materializing sampled
    pandas frames, while avoiding thousands of groupby/metric passes.

    Every row carries ``blocks`` and ``degenerate_blocks``. Below the measured
    floor (``estimation_variance.MIN_BLOCKS_FOR_INTERVAL``) the percentile
    bootstrap's coverage is nowhere near nominal, so ``lower``/``upper`` (and
    ``delta_lower``/``delta_upper``) on a flagged row are not a 95% interval
    and must not be read as one. See ``experiments.paired_feature_comparisons``
    for the same guard on paired deltas.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    group_columns = ["season", "week"] if block == "week" else ["season"]
    if predictions.empty:
        raise ValueError("Predictions contain no bootstrap blocks")

    estimates = {
        method: summarize_outcome_method(group)
        for method, group in predictions.groupby("method", sort=True)
    }
    market = estimates.get("market", {})
    methods: tuple[str, ...] = tuple(str(method) for method in estimates)
    keys = tuple(
        (method, metric)
        for method in methods
        for metric in OUTCOME_UNCERTAINTY_METRICS
        if metric in estimates[method]
    )

    contributions = _outcome_bootstrap_contributions(predictions, group_columns)
    block_count = int(contributions["_bootstrap_block"].max()) + 1
    block_verdict = guard_block_count(
        block_count,
        min_blocks=min_blocks,
        on_degenerate=on_degenerate,
        context=f"outcome_bootstrap_intervals(block={block})",
    )
    method_blocks: dict[str, np.ndarray] = {}
    for method in methods:
        method_rows = contributions.loc[contributions["method"].eq(method)]
        values = np.zeros((block_count, len(_OUTCOME_BOOTSTRAP_STATS)), dtype=float)
        positions = method_rows["_bootstrap_block"].to_numpy(dtype=int)
        values[positions] = method_rows.loc[:, list(_OUTCOME_BOOTSTRAP_STATS)].to_numpy(dtype=float)
        method_blocks[method] = values

    generator = np.random.default_rng(seed)
    selected = generator.integers(0, block_count, size=(samples, block_count))
    sample_rows = np.repeat(np.arange(samples), block_count)
    sample_counts = np.zeros((samples, block_count), dtype=np.int32)
    np.add.at(sample_counts, (sample_rows, selected.ravel()), 1)
    method_draws = {
        method: _metrics_from_bootstrap_totals(sample_counts @ method_blocks[method])
        for method in methods
    }

    tail = (1.0 - confidence) / 2.0
    rows: list[dict[str, Any]] = []
    for key in keys:
        method, metric = key
        row: dict[str, Any] = {
            "method": method,
            "metric": metric,
            "estimate": estimates[method][metric],
            "lower": float(np.quantile(method_draws[method][metric], tail)),
            "upper": float(np.quantile(method_draws[method][metric], 1.0 - tail)),
            "confidence": confidence,
            "block": block,
            "samples": samples,
            "blocks": block_verdict.block_count,
            # True => lower/upper (and delta_lower/delta_upper) are NOT a
            # valid interval at this block count. See the docstring; do not
            # render as one.
            "degenerate_blocks": block_verdict.degenerate,
        }
        if method != "market" and metric in market:
            delta = method_draws[method][metric] - method_draws["market"][metric]
            row.update(
                {
                    "market_estimate": market[metric],
                    "delta_vs_market": float(estimates[method][metric]) - float(market[metric]),
                    "delta_lower": float(np.quantile(delta, tail)),
                    "delta_upper": float(np.quantile(delta, 1.0 - tail)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _outcome_bootstrap_contributions(
    predictions: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Aggregate additive metric components once per block and method."""

    working = predictions.copy()
    if len(group_columns) == 1:
        block_values: Any = working[group_columns[0]]
    else:
        block_values = pd.MultiIndex.from_frame(working[group_columns])
    working["_bootstrap_block"] = pd.factorize(block_values, sort=False)[0]
    for column in _OUTCOME_BOOTSTRAP_STATS:
        working[column] = 0.0

    result_values = pd.to_numeric(working["result"], errors="coerce").to_numpy(dtype=float)
    win_probability = pd.to_numeric(working["home_win_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    win_mask = np.isfinite(result_values) & (result_values != 0.0) & np.isfinite(win_probability)
    win_actual = (result_values[win_mask] > 0.0).astype(float)
    working.loc[win_mask, "win_correct"] = (
        (win_probability[win_mask] >= 0.5) == win_actual
    ).astype(float)
    working.loc[win_mask, "win_brier"] = np.square(win_probability[win_mask] - win_actual)
    working.loc[win_mask, "win_count"] = 1.0

    predicted_margin = pd.to_numeric(working["predicted_margin"], errors="coerce").to_numpy(
        dtype=float
    )
    margin_mask = np.isfinite(result_values) & np.isfinite(predicted_margin)
    margin_error = predicted_margin[margin_mask] - result_values[margin_mask]
    working.loc[margin_mask, "margin_absolute_error"] = np.abs(margin_error)
    working.loc[margin_mask, "margin_squared_error"] = np.square(margin_error)
    working.loc[margin_mask, "margin_count"] = 1.0

    cover_actual = pd.to_numeric(working["home_cover"], errors="coerce").to_numpy(dtype=float)
    cover_probability = pd.to_numeric(working["home_cover_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    cover_mask = np.isfinite(cover_actual) & np.isfinite(cover_probability)
    probability = np.clip(
        cover_probability[cover_mask], np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    actual = cover_actual[cover_mask]
    working.loc[cover_mask, "cover_correct"] = ((probability >= 0.5) == actual).astype(float)
    working.loc[cover_mask, "cover_brier"] = np.square(probability - actual)
    working.loc[cover_mask, "cover_log_loss"] = -(
        actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability)
    )
    working.loc[cover_mask, "cover_count"] = 1.0

    # Outcome summaries compute betting metrics on non-push cover rows. Match
    # that contract exactly here instead of counting wagers attached to pushes.
    wager_mask = cover_mask & working["bet_side"].ne("PASS").to_numpy(dtype=bool)
    wager_positions = np.flatnonzero(wager_mask)
    profits = np.zeros(len(wager_positions), dtype=float)
    bet_sides = working["bet_side"].astype(str).to_numpy()
    bet_odds = pd.to_numeric(working["bet_odds"], errors="coerce").to_numpy(dtype=float)
    for output_index, position in enumerate(wager_positions):
        cover = cover_actual[position]
        profits[output_index] = (
            0.0
            if not np.isfinite(cover)
            else settle_bet(
                str(bet_sides[position]),
                float(cover),
                float(bet_odds[position]),
            )
        )
    working.loc[wager_mask, "profit_units"] = profits
    working.loc[wager_mask, "bet_count"] = 1.0
    return (
        working.groupby(["_bootstrap_block", "method"], sort=False, dropna=False)[
            list(_OUTCOME_BOOTSTRAP_STATS)
        ]
        .sum()
        .reset_index()
    )


def _metrics_from_bootstrap_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Convert sampled sufficient-statistic totals to reported metrics."""

    index = {name: position for position, name in enumerate(_OUTCOME_BOOTSTRAP_STATS)}

    def ratio(numerator: str, denominator: str) -> np.ndarray:
        return np.asarray(
            np.divide(
                totals[:, index[numerator]],
                totals[:, index[denominator]],
                out=np.full(len(totals), np.nan, dtype=float),
                where=totals[:, index[denominator]] > 0.0,
            ),
            dtype=float,
        )

    roi = ratio("profit_units", "bet_count")
    roi = np.where(np.isnan(roi), 0.0, roi)

    return {
        "win_accuracy": ratio("win_correct", "win_count"),
        "win_brier_score": ratio("win_brier", "win_count"),
        "margin_mae": ratio("margin_absolute_error", "margin_count"),
        "margin_rmse": np.sqrt(ratio("margin_squared_error", "margin_count")),
        "cover_accuracy": ratio("cover_correct", "cover_count"),
        "cover_brier_score": ratio("cover_brier", "cover_count"),
        "cover_log_loss": ratio("cover_log_loss", "cover_count"),
        "roi": roi,
    }
