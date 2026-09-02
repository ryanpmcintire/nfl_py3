"""Over/under regime: a ridge model on the market total's residual.

Executes the frozen predeclaration in ``docs/totals_model.md`` (written
2026-09-01, before any totals model had been fit on this data). Every
structural choice below -- target, population, feature allowlist, pipeline,
protocol, metrics -- is that document's contract, not a choice made after
seeing a sign.

The architecture mirrors the ATS side exactly: the market line is the prior,
ridge models only the *residual*, evaluation is chronological walk-forward,
and the model is folded in at a MEASURED blend weight rather than allowed to
override the market. The target is::

    total_residual = (home_score + away_score) - total_line

and the served quantity is ``total_line + k * predicted_residual`` for the
MAE-minimizing ``k`` from the sweep, which is free to be 0.0.

Why the residual and not the total itself: the market total already carries
essentially all of the predictable signal, so regressing the raw total mostly
re-learns the line. Modelling the residual makes the model's whole job the
part the market may have left on the table, and makes ``k = 0`` a meaningful
null rather than a degenerate one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.io import atomic_json, run_id

#: The explicit feature allowlist frozen in ``docs/totals_model.md`` (read
#: 2026-09-01, lines 37-45). NOTHING outside this tuple enters the design
#: matrix -- :func:`design_matrix` selects by this list, so a new column
#: appearing in ``game_features.parquet`` cannot silently join the fit.
#:
#: Deliberately excluded, per the same contract: identifiers and outcomes
#: (``result``, ``ats_margin``, ``home_cover``, the scores); the ``diff_*``
#: columns (a total rides the SUM of the two teams, not their difference --
#: ridge forms whatever sum it wants from the home/away columns directly);
#: and the ``*_ats_residual`` / graph / schedule / bias / surface columns,
#: which are spread-oriented constructs a separately declared second wave may
#: screen as totals features.
TOTALS_FEATURES: tuple[str, ...] = (
    "total_line",
    "spread_line",
    "rest_diff",
    "neutral_site",
    "div_game",
    "temp",
    "wind",
    "week_sin",
    "week_cos",
    "elo_diff",
    "elo_home_win_prob",
    "home_team_games",
    "away_team_games",
    "home_off_epa_per_play",
    "home_off_pass_epa_per_play",
    "home_off_rush_epa_per_play",
    "home_off_cpoe",
    "home_off_yards_per_play",
    "home_off_turnover_rate",
    "home_off_sack_rate",
    "away_off_epa_per_play",
    "away_off_pass_epa_per_play",
    "away_off_rush_epa_per_play",
    "away_off_cpoe",
    "away_off_yards_per_play",
    "away_off_turnover_rate",
    "away_off_sack_rate",
    "home_def_epa_per_play",
    "home_def_pass_epa_per_play",
    "home_def_rush_epa_per_play",
    "home_def_yards_per_play",
    "home_def_takeaway_rate",
    "home_def_sack_rate",
    "away_def_epa_per_play",
    "away_def_pass_epa_per_play",
    "away_def_rush_epa_per_play",
    "away_def_yards_per_play",
    "away_def_takeaway_rate",
    "away_def_sack_rate",
    "home_point_diff",
    "away_point_diff",
)

#: Primary ridge penalty: production's exact constant (``margin.py`` line 366
#: default ``ridge_alpha: float = 10.0``, read 2026-09-01), frozen in the
#: predeclaration so no tuning ever touches the test stream.
TOTALS_RIDGE_ALPHA = 10.0

#: Reported alongside the primary for transparency only -- never used to pick
#: a winner (``docs/totals_model.md`` line 56).
TOTALS_REPORTED_ALPHAS: tuple[float, ...] = (1.0, 100.0)

#: The blend weights swept for the decision. ``0.0`` is the market-alone null
#: and ``1.0`` is the raw model total; the MAE-minimizing entry is the answer.
BLEND_WEIGHTS: tuple[float, ...] = tuple(round(0.1 * step, 1) for step in range(11))

_TARGET = "total_residual"


class TotalsDataError(RuntimeError):
    """The population could not be assembled as the contract specifies."""


@dataclass(frozen=True)
class TotalsView:
    """One upcoming game's model opinion on the total.

    Mirrors :class:`nfl_ats.tiebreaker.ModelView` on the margin side: the
    market's line, the model's residual against it, and where the number came
    from -- so the tiebreaker can acknowledge a disagreement instead of
    silently ignoring it.
    """

    predicted_total: float
    market_total: float
    residual: float  # predicted_total - market_total
    train_games: int
    source: str


def newest_schedules_path(data_root: Path) -> Path:
    """The newest ``data/raw/*/schedules.parquet`` -- same resolution rule
    :mod:`nfl_ats.tiebreaker` uses, so both read one market truth."""

    hits = sorted((data_root / "raw").glob("*/schedules.parquet"))
    if not hits:
        raise FileNotFoundError(f"no schedules.parquet under {data_root / 'raw'}")
    return hits[-1]


def design_matrix(frame: pd.DataFrame, features: Sequence[str] = TOTALS_FEATURES) -> pd.DataFrame:
    """The allowlist, and only the allowlist, in a fixed column order.

    This is the single gate the predeclaration's "nothing outside this list
    enters the fit" clause runs through: an extra or renamed column in the
    source table is simply not selected, and a MISSING allowlist column is a
    hard error rather than a silent substitution.
    """

    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise TotalsDataError(f"allowlist columns absent from the feature table: {missing}")
    return frame.loc[:, list(features)].astype(float)


def make_totals_estimator(*, ridge_alpha: float = TOTALS_RIDGE_ALPHA) -> BaseEstimator:
    """Production's exact recipe (``margin.py`` lines 377-387, read
    2026-09-01): median imputation with missingness indicators, standardize,
    ridge. Reused verbatim rather than re-derived so the totals arm cannot
    win or lose on a pipeline difference."""

    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=ridge_alpha)),
        ]
    )


def load_population(
    data_root: Path,
    features_path: Path,
    *,
    schedules_path: Path | None = None,
) -> pd.DataFrame:
    """The frozen population: every newest-schedules game with a non-null
    ``home_score``, ``away_score`` and ``total_line``, inner-joined to the
    canonical feature table on ``game_id``.

    The target and the market baseline come from SCHEDULES (the population's
    defining source) and are carried as ``market_total``; the predictors come
    from the feature table, whose own ``total_line``/``spread_line`` columns
    are the allowlist entries. The two sources agree exactly (measured
    2026-09-01: max absolute difference 0.0 over all 4,630 joined games), so
    keeping both under distinct names costs nothing and keeps each number's
    provenance unambiguous instead of relying on a silent merge suffix.
    """

    path = schedules_path if schedules_path is not None else newest_schedules_path(data_root)
    schedules = pd.read_parquet(path)
    lined = schedules.loc[
        schedules["home_score"].notna()
        & schedules["away_score"].notna()
        & schedules["total_line"].notna()
    ].copy()
    if lined.empty:
        raise TotalsDataError(f"no lined finals in {path}")

    keep = ["game_id", "season", "week", "game_type", "home_score", "away_score", "total_line"]
    if "gameday" in lined.columns:
        keep.append("gameday")
    if "home_team" in lined.columns:
        keep += ["home_team", "away_team"]
    lined = lined.loc[:, keep].rename(columns={"total_line": "market_total"})

    features = pd.read_parquet(features_path)
    feature_columns = ["game_id"] + [
        column for column in TOTALS_FEATURES if column in features.columns
    ]
    joined = lined.merge(features.loc[:, feature_columns], on="game_id", how="inner")
    if joined.empty:
        raise TotalsDataError("no lined finals survived the join to the feature table")

    joined["actual_total"] = joined["home_score"].astype(float) + joined["away_score"].astype(float)
    joined["market_total"] = joined["market_total"].astype(float)
    joined[_TARGET] = joined["actual_total"] - joined["market_total"]
    joined["game_type"] = joined["game_type"].astype(str)
    joined["season"] = joined["season"].astype(int)
    joined["week"] = joined["week"].astype(int)
    return joined.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def chronological_blocks(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Ordered ``(season, week)`` prediction blocks.

    Plain ``(season, week)`` ordering is already chronologically honest in
    this data: no ``(season, week)`` block mixes game types, and the wild-card
    round sits at week 18 through 2020 and week 19 from 2021 -- always AFTER
    that season's last regular week (measured 2026-09-01 from the newest
    schedules: 0 blocks with more than one ``game_type``; WC week is 18 for
    2009-2020 and 19 for 2021-2025). So no postseason game can ever enter a
    regular-season week's training pool.
    """

    pairs = frame.loc[:, ["season", "week"]].drop_duplicates()
    return sorted({(int(season), int(week)) for season, week in pairs.itertuples(index=False)})


def walk_forward_predictions(
    population: pd.DataFrame,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    features: Sequence[str] = TOTALS_FEATURES,
) -> pd.DataFrame:
    """Expanding-window walk-forward, one fit per ``(season, week)`` block.

    The guard the contract names: a block's training pool is every game
    STRICTLY BEFORE that block in ``(season, week)`` order. Not "before or
    equal" -- a row from the target week itself, or any later week, is never
    fitted on. Blocks whose pool holds fewer than ``min_train_games`` games
    (500, ``constants.DEFAULT_MIN_TRAIN_GAMES``) are warm-up and produce no
    predictions at all rather than predictions from a thin model.
    """

    blocks = chronological_blocks(population)
    keys = list(zip(population["season"], population["week"], strict=True))
    order = np.array([blocks.index((int(season), int(week))) for season, week in keys])
    design = design_matrix(population, features)
    target = population[_TARGET].astype(float).to_numpy()

    chunks: list[pd.DataFrame] = []
    for position, (season, week) in enumerate(blocks):
        train_mask = order < position
        train_count = int(train_mask.sum())
        if train_count < min_train_games:
            continue
        test_mask = order == position
        estimator = make_totals_estimator(ridge_alpha=ridge_alpha)
        estimator.fit(design.loc[train_mask], target[train_mask])
        predicted = np.asarray(estimator.predict(design.loc[test_mask]), dtype=float)
        block = population.loc[test_mask, :].copy()
        block["predicted_residual"] = predicted
        block["train_games"] = train_count
        block["block_season"] = season
        block["block_week"] = week
        chunks.append(block)

    if not chunks:
        raise TotalsDataError(
            f"no block reached min_train_games={min_train_games}; population has {len(population)}"
        )
    predictions = pd.concat(chunks, ignore_index=True)
    predictions["model_total"] = predictions["market_total"] + predictions["predicted_residual"]
    predictions["market_error"] = predictions["market_total"] - predictions["actual_total"]
    predictions["model_error"] = predictions["model_total"] - predictions["actual_total"]
    return predictions


def blend_total(market_total: pd.Series, predicted_residual: pd.Series, weight: float) -> pd.Series:
    """``total_line + k * predicted_residual`` -- the served quantity.

    ``k = 0`` is the market alone and ``k = 1`` is the raw model total, so the
    two endpoint baselines are the same arithmetic as every interior point
    and cannot drift apart from it.
    """

    return market_total.astype(float) + float(weight) * predicted_residual.astype(float)


def _error_metrics(errors: pd.Series) -> dict[str, float]:
    absolute = errors.abs()
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt((errors.astype(float) ** 2).mean())),
    }


def blend_sweep(
    predictions: pd.DataFrame, weights: Iterable[float] = BLEND_WEIGHTS
) -> pd.DataFrame:
    """MAE and RMSE at every swept blend weight, plus the market and raw-model
    deltas, on whatever subset of predictions is handed in."""

    market = _error_metrics(predictions["market_error"])
    rows: list[dict[str, float]] = []
    for weight in weights:
        blended = blend_total(
            predictions["market_total"], predictions["predicted_residual"], weight
        )
        errors = blended - predictions["actual_total"].astype(float)
        metrics = _error_metrics(errors)
        rows.append(
            {
                "k": float(weight),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mae_improvement_vs_market": market["mae"] - metrics["mae"],
                "rmse_improvement_vs_market": market["rmse"] - metrics["rmse"],
            }
        )
    return pd.DataFrame(rows)


def choose_weight(sweep: pd.DataFrame) -> float:
    """The decision rule the contract fixes: the MAE-minimizing ``k``.

    Ties break toward the SMALLER weight, so an exactly flat sweep serves the
    market alone rather than an arbitrary interior point -- the conservative
    direction for a tie, and the one that keeps ``k`` a derived number.
    """

    ordered = sweep.sort_values(["mae", "k"], kind="mergesort")
    return float(ordered.iloc[0]["k"])


def per_season_deltas(predictions: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Season-by-season MAE for the market and the chosen blend.

    ``mae_improvement`` is POSITIVE when the blend is better (it is
    market MAE minus blend MAE), matching the sign convention used for the
    bootstrap and the registry entry.
    """

    blended = blend_total(predictions["market_total"], predictions["predicted_residual"], weight)
    frame = predictions.assign(
        blend_abs_error=(blended - predictions["actual_total"].astype(float)).abs(),
        market_abs_error=predictions["market_error"].abs(),
    )
    grouped = frame.groupby("season", as_index=False).agg(
        games=("game_id", "size"),
        market_mae=("market_abs_error", "mean"),
        blend_mae=("blend_abs_error", "mean"),
    )
    grouped["mae_improvement"] = grouped["market_mae"] - grouped["blend_mae"]
    return grouped


def paired_error_frame(predictions: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Per-game paired |error| difference, market minus blend.

    POSITIVE = the blend is closer to the actual total on that game. The
    bootstrap runs on this column, so the sign of the reported effect is the
    sign of "the model helped".
    """

    blended = blend_total(predictions["market_total"], predictions["predicted_residual"], weight)
    blend_abs = (blended - predictions["actual_total"].astype(float)).abs()
    market_abs = predictions["market_error"].abs()
    return pd.DataFrame(
        {
            "game_id": predictions["game_id"].astype(str),
            "season": predictions["season"].astype(int),
            "week": predictions["week"].astype(int),
            "market_abs_error": market_abs.to_numpy(dtype=float),
            "blend_abs_error": blend_abs.to_numpy(dtype=float),
            "abs_error_improvement": (market_abs - blend_abs).to_numpy(dtype=float),
        }
    )


def _mean_improvement(frame: pd.DataFrame) -> dict[str, float]:
    return {"mae_improvement": float(frame["abs_error_improvement"].mean())}


def bootstrap_improvement(
    paired: pd.DataFrame, *, samples: int = 2_000, seed: int = 20260901
) -> dict[str, float]:
    """Week-blocked bootstrap of the mean paired |error| improvement.

    Uses :func:`nfl_ats.clv.week_blocked_bootstrap` unchanged so the totals
    arm reports the same interval construction every other arm of this
    project reports, and surfaces ``probability_positive`` -- the fraction of
    blocked resamples in which the blend beats the market -- rather than a
    binary read of the endpoints.
    """

    result = week_blocked_bootstrap(
        paired, _mean_improvement, block="week", samples=samples, seed=seed
    )
    row = result.iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "samples": int(row["samples"]),
        "blocks": int(paired.groupby(["season", "week"]).ngroups),
        "games": len(paired),
    }


def _subset_summary(predictions: pd.DataFrame, weight: float) -> dict[str, Any]:
    sweep = blend_sweep(predictions)
    market = _error_metrics(predictions["market_error"])
    model = _error_metrics(predictions["model_error"])
    chosen = sweep.loc[np.isclose(sweep["k"], weight)].iloc[0]
    return {
        "games": len(predictions),
        "seasons": [int(predictions["season"].min()), int(predictions["season"].max())],
        "market": market,
        "raw_model": model,
        "chosen_k": float(weight),
        "chosen_blend": {"mae": float(chosen["mae"]), "rmse": float(chosen["rmse"])},
        "mae_improvement_vs_market": float(chosen["mae_improvement_vs_market"]),
        "sweep": sweep.to_dict(orient="records"),
    }


def run_backtest(
    data_root: Path,
    features_path: Path,
    artifacts_root: Path,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    reported_alphas: Sequence[float] = TOTALS_REPORTED_ALPHAS,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260901,
    stamp: str | None = None,
) -> dict[str, Any]:
    """The whole frozen regime, once: walk-forward, sweep, decide, bootstrap,
    and write prediction-level output.

    The primary read is ``game_type == "REG"`` and the decision ``k`` comes
    from it alone; playoffs are scored with the same models and reported in
    their own block (FND-15 lineage: never silently pooled).
    """

    population = load_population(data_root, features_path)
    predictions = walk_forward_predictions(
        population, ridge_alpha=ridge_alpha, min_train_games=min_train_games
    )
    regular = predictions.loc[predictions["game_type"] == "REG"].reset_index(drop=True)
    postseason = predictions.loc[predictions["game_type"] != "REG"].reset_index(drop=True)
    if regular.empty:
        raise TotalsDataError("walk-forward produced no regular-season predictions")

    sweep = blend_sweep(regular)
    weight = choose_weight(sweep)
    paired = paired_error_frame(regular, weight)
    bootstrap = bootstrap_improvement(paired, samples=bootstrap_samples, seed=bootstrap_seed)

    alternates: dict[str, Any] = {}
    for alternate_alpha in reported_alphas:
        alternate = walk_forward_predictions(
            population, ridge_alpha=float(alternate_alpha), min_train_games=min_train_games
        )
        alternate_regular = alternate.loc[alternate["game_type"] == "REG"]
        alternate_sweep = blend_sweep(alternate_regular)
        alternate_weight = choose_weight(alternate_sweep)
        alternates[f"alpha_{alternate_alpha:g}"] = _subset_summary(
            alternate_regular, alternate_weight
        )

    results: dict[str, Any] = {
        "contract": "docs/totals_model.md (predeclared 2026-09-01)",
        "target": _TARGET,
        "features": list(TOTALS_FEATURES),
        "feature_count": len(TOTALS_FEATURES),
        "pipeline": "SimpleImputer(median, add_indicator) -> StandardScaler -> Ridge",
        "ridge_alpha": float(ridge_alpha),
        "min_train_games": int(min_train_games),
        "population_games": len(population),
        "scored_games": len(predictions),
        "regular_season": _subset_summary(regular, weight),
        "playoffs": (_subset_summary(postseason, weight) if not postseason.empty else {"games": 0}),
        "chosen_k": weight,
        "per_season": per_season_deltas(regular, weight).to_dict(orient="records"),
        "bootstrap": bootstrap,
        "bootstrap_sign_convention": (
            "positive = the blend's absolute error is SMALLER than the market's "
            "(the blend is better)"
        ),
        "reported_alphas": alternates,
    }

    stamp = stamp or run_id()
    output_dir = artifacts_root / "totals_backtest" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    paired.to_parquet(output_dir / "paired_errors.parquet", index=False)
    atomic_json(results, output_dir / "results.json")
    atomic_json(
        {
            "stamp": stamp,
            "generated_by": "nfl-ats totals-backtest",
            "schedules": str(newest_schedules_path(data_root)),
            "features": str(features_path),
            "ridge_alpha": float(ridge_alpha),
            "reported_alphas": [float(value) for value in reported_alphas],
            "min_train_games": int(min_train_games),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "blend_weights": [float(value) for value in BLEND_WEIGHTS],
        },
        output_dir / "metadata.json",
    )
    results["output_dir"] = str(output_dir)
    return results


def model_total_view(
    game_id: str,
    data_root: Path,
    features_path: Path,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> TotalsView | None:
    """The totals model's residual for ONE upcoming game, fit under the same
    walk-forward guard the backtest scores.

    Training is every population game strictly before the target's
    ``(season, week)``, so the number served for a live game is produced the
    same way as the numbers the backtest graded. Returns ``None`` when the
    feature table does not price the game, when the market total is missing,
    or when fewer than ``min_train_games`` prior games exist -- the caller
    then simply uses the market total alone.
    """

    if not features_path.is_file():
        return None
    features = pd.read_parquet(features_path)
    rows = features.loc[features["game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row.get("total_line")):
        return None
    season, week = int(row["season"]), int(row["week"])

    population = load_population(data_root, features_path)
    prior = population.loc[
        (population["season"] < season)
        | ((population["season"] == season) & (population["week"] < week))
    ]
    if len(prior) < min_train_games:
        return None

    estimator = make_totals_estimator(ridge_alpha=ridge_alpha)
    estimator.fit(design_matrix(prior), prior[_TARGET].astype(float).to_numpy())
    target_design = design_matrix(rows.iloc[[0]])
    residual = float(np.asarray(estimator.predict(target_design), dtype=float)[0])
    market_total = float(row["total_line"])
    return TotalsView(
        predicted_total=market_total + residual,
        market_total=market_total,
        residual=residual,
        train_games=len(prior),
        source=(
            f"totals ridge(alpha={ridge_alpha:g}) trained on {len(prior)} games "
            f"before {season} week {week}"
        ),
    )


def format_results(results: dict[str, Any]) -> str:
    """Human-readable summary of :func:`run_backtest` for the CLI."""

    regular = results["regular_season"]
    bootstrap = results["bootstrap"]
    lines = [
        f"totals backtest -- {regular['games']} regular-season games scored "
        f"({regular['seasons'][0]}-{regular['seasons'][1]}), "
        f"{results['population_games']} in the population",
        f"pipeline: {results['pipeline']}(alpha={results['ridge_alpha']:g}), "
        f"{results['feature_count']} allowlist features, "
        f"min_train_games={results['min_train_games']}",
        "",
        f"market total alone : MAE {regular['market']['mae']:.3f}  "
        f"RMSE {regular['market']['rmse']:.3f}",
        f"raw model total    : MAE {regular['raw_model']['mae']:.3f}  "
        f"RMSE {regular['raw_model']['rmse']:.3f}",
        "",
        "blend sweep (total_line + k * predicted_residual):",
    ]
    for entry in regular["sweep"]:
        marker = "  <-- chosen" if np.isclose(entry["k"], results["chosen_k"]) else ""
        lines.append(
            f"  k={entry['k']:.1f}  MAE {entry['mae']:.4f}  RMSE {entry['rmse']:.4f}  "
            f"MAE improvement {entry['mae_improvement_vs_market']:+.4f}{marker}"
        )
    lines += [
        "",
        f"chosen k = {results['chosen_k']:.1f}: MAE improvement over the market "
        f"{regular['mae_improvement_vs_market']:+.4f} total points",
        f"week-blocked bootstrap ({bootstrap['samples']} resamples, "
        f"{bootstrap['blocks']} week blocks): {bootstrap['estimate']:+.4f} "
        f"95% [{bootstrap['lower']:+.4f}, {bootstrap['upper']:+.4f}], "
        f"probability_positive {bootstrap['probability_positive']:.3f}",
        f"  ({results['bootstrap_sign_convention']})",
        "",
        "per-season MAE improvement (positive = blend better):",
    ]
    for entry in results["per_season"]:
        lines.append(
            f"  {int(entry['season'])}: market {entry['market_mae']:.3f}  "
            f"blend {entry['blend_mae']:.3f}  "
            f"delta {entry['mae_improvement']:+.4f}  ({int(entry['games'])} games)"
        )
    playoffs = results["playoffs"]
    if playoffs.get("games"):
        lines += [
            "",
            f"playoffs, reported separately ({playoffs['games']} games): "
            f"market MAE {playoffs['market']['mae']:.3f}, "
            f"blend MAE {playoffs['chosen_blend']['mae']:.3f}, "
            f"delta {playoffs['mae_improvement_vs_market']:+.4f}",
        ]
    lines += ["", "alphas reported alongside (never used to pick):"]
    for label, summary in results["reported_alphas"].items():
        lines.append(
            f"  {label}: best k={summary['chosen_k']:.1f}, "
            f"MAE improvement {summary['mae_improvement_vs_market']:+.4f}"
        )
    if "output_dir" in results:
        lines += ["", f"prediction-level output: {results['output_dir']}"]
    return "\n".join(lines)


def load_results(path: Path) -> dict[str, Any]:
    """Read a previously written ``results.json`` back."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
