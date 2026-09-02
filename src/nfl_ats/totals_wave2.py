"""Over/under regime, wave 2: screens the drive-pace family on top of wave 1.

Executes the frozen predeclaration in ``docs/totals_model_wave2.md`` (written
2026-09-01, before any wave-2 outcome was computed). Every structural choice
here -- the extended allowlist, the comparator, the positive control -- is
that document's contract.

This module deliberately REUSES ``nfl_ats.totals`` rather than reimplementing
it: the pipeline (``make_totals_estimator``), the walk-forward guard
(``walk_forward_predictions``), the blend math (``blend_total``,
``blend_sweep``, ``choose_weight``, ``per_season_deltas``), and the bootstrap
(``bootstrap_improvement``, itself a thin wrapper over
``nfl_ats.clv.week_blocked_bootstrap``) are all called unmodified against the
wider ``WAVE2_FEATURES`` column list. ``nfl_ats.totals.load_population`` is
the one function that cannot be reused directly -- it hardcodes wave 1's
``TOTALS_FEATURES`` allowlist when selecting columns out of the feature
table -- so :func:`load_population_wave2` reimplements that one join,
parameterized over the feature list, with the same population rule (verified
2026-09-01: identical population, identical wave-1-column values, see the
predeclaration's "Verified this session" note).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, DRIVE_STATE_METRICS
from nfl_ats.io import atomic_json, run_id
from nfl_ats.totals import (
    TOTALS_FEATURES,
    TOTALS_RIDGE_ALPHA,
    TotalsDataError,
    TotalsView,
    blend_sweep,
    blend_total,
    choose_weight,
    design_matrix,
    make_totals_estimator,
    newest_schedules_path,
    paired_error_frame,
    per_season_deltas,
    walk_forward_predictions,
)
from nfl_ats.totals import (
    load_population as load_population_wave1,
)

#: The 24 columns this wave adds: ``{home,away}`` crossed with
#: ``nfl_ats.constants.DRIVE_STATE_METRICS`` (12 entries) -- the walk-forward
#: state version of every column ``build_pbp_team_game_metrics``
#: (``src/nfl_ats/pbp.py``) derives from ``build_drive_table``. Frozen order:
#: home metric, away metric, for each of the 12 metrics in
#: ``DRIVE_STATE_METRICS``'s own declared order.
#:
#: Deliberately excluded (``docs/totals_model_wave2.md``, "Deliberately
#: excluded"): ``{home,away}_pbp_drives`` (a drive-count column, not part of
#: ``DRIVE_STATE_METRICS`` and not counted in this work package's frozen "24"),
#: and every ``diff_*`` column (totals ride sums, not differences -- the same
#: reason wave 1 excluded its own ``diff_*`` columns).
WAVE2_DRIVE_FEATURES: tuple[str, ...] = tuple(
    f"{side}_{metric}" for metric in DRIVE_STATE_METRICS for side in ("home", "away")
)

if len(WAVE2_DRIVE_FEATURES) != 24:
    raise AssertionError(
        f"WAVE2_DRIVE_FEATURES must freeze to exactly 24 columns, got {len(WAVE2_DRIVE_FEATURES)}"
    )

#: The full wave-2 candidate allowlist: wave 1's 41 columns, unchanged and in
#: their original order, plus the 24 drive columns above. Nothing outside
#: this tuple enters the wave-2 design matrix.
WAVE2_FEATURES: tuple[str, ...] = tuple(TOTALS_FEATURES) + WAVE2_DRIVE_FEATURES

_TARGET = "total_residual"

#: The frozen comparator operating point: wave 1's own already-chosen blend
#: weight (``docs/totals_model.md`` Results, ``nfl_ats.tiebreaker.
#: TOTALS_RESIDUAL_WEIGHT``). Wave 2 does not get to re-sweep wave 1's k.
WAVE1_CHOSEN_K = 0.1

#: The positive-control column: an arbitrary, explicit, pre-chosen member of
#: ``WAVE2_DRIVE_FEATURES``, frozen before any outcome was computed.
POSITIVE_CONTROL_COLUMN = "home_drive_points_per_drive"


def _feature_table_matches_schedules(
    data_root: Path, features: pd.DataFrame, schedules_path: Path | None = None
) -> bool:
    """Return whether the wave-2 table is still aligned to the schedule.

    The feature build is an optional, generated input.  A table from an older
    schedule snapshot can still contain the target game while carrying an old
    line (or an old season/week assignment), which would make the residual
    silently use a different market than the tiebreaker displays.  Treat that
    condition, duplicate game IDs, and partial joins as unavailable rather
    than serving a residual from mismatched data.

    This is intentionally an identity check, not a freshness timeout: there
    is no frozen wall-clock freshness threshold in the wave-2 contract.  The
    schedule/feature keys and market line are the contract's stable signals of
    staleness.
    """

    required = {"game_id", "season", "week", "total_line"}
    if not required.issubset(features.columns):
        return False
    feature_ids = features["game_id"].astype(str)
    if feature_ids.duplicated().any():
        return False

    try:
        path = schedules_path if schedules_path is not None else newest_schedules_path(data_root)
        schedules = pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):
        return False
    schedule_required = required | {"home_score", "away_score"}
    if not schedule_required.issubset(schedules.columns):
        return False
    # Include all schedule rows (played and upcoming) in the identity check:
    # a stale table missing a current target must not look valid merely because
    # its historical training rows happen to join.
    schedule_ids = schedules["game_id"].astype(str)
    if schedule_ids.duplicated().any() or set(feature_ids) != set(schedule_ids):
        return False

    schedule_index = schedules.assign(_key=schedule_ids).set_index("_key")
    feature_index = features.assign(_key=feature_ids).set_index("_key")
    for column in ("season", "week", "total_line"):
        left = pd.to_numeric(schedule_index[column], errors="coerce")
        right = pd.to_numeric(feature_index[column], errors="coerce").reindex(left.index)
        if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True):
            return False
    return True


def load_population_wave2(
    data_root: Path,
    features_path: Path,
    *,
    features: tuple[str, ...] = WAVE2_FEATURES,
    schedules_path: Path | None = None,
) -> pd.DataFrame:
    """The wave-2 population: identical rows to
    :func:`nfl_ats.totals.load_population`, joined to a wider column list.

    Mirrors ``nfl_ats.totals.load_population`` exactly (same schedules
    filter, same target computation) except the feature-column selection is
    parameterized over ``features`` instead of hardcoding wave 1's
    ``TOTALS_FEATURES`` -- the one piece of wave 1's population loader this
    wave cannot reuse unmodified, because it always projects onto its own
    module-level allowlist.
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

    table = pd.read_parquet(features_path)
    if not _feature_table_matches_schedules(data_root, table, schedules_path=path):
        raise TotalsDataError("wave-2 feature table is stale or misaligned with schedules")
    feature_columns = ["game_id"] + [column for column in features if column in table.columns]
    joined = lined.merge(table.loc[:, feature_columns], on="game_id", how="inner")
    if joined.empty:
        raise TotalsDataError("no lined finals survived the join to the feature table")

    joined["actual_total"] = joined["home_score"].astype(float) + joined["away_score"].astype(float)
    joined["market_total"] = joined["market_total"].astype(float)
    joined[_TARGET] = joined["actual_total"] - joined["market_total"]
    joined["game_type"] = joined["game_type"].astype(str)
    joined["season"] = joined["season"].astype(int)
    joined["week"] = joined["week"].astype(int)
    return joined.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def model_total_view_wave2(
    game_id: str,
    data_root: Path,
    features_path: Path,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> TotalsView | None:
    """Wave 2's totals residual for ONE upcoming game.

    Mirrors :func:`nfl_ats.totals.model_total_view` line for line -- same
    signature shape, same return type, same walk-forward guard (train on
    every population game strictly before the target's ``(season, week)``) --
    built on the 65-column :data:`WAVE2_FEATURES` allowlist and
    :func:`load_population_wave2` against ``features_path`` (normally
    ``data/processed/game_features_pbp.parquet``, the drive-pace-enriched
    table) instead of wave 1's 41-column ``game_features.parquet``.

    Returns ``None`` -- cleanly, never a silent substitution -- when
    ``features_path`` does not exist, when it carries no row for
    ``game_id`` (a game the PBP pipeline has not enriched), when the market
    total is missing, or when fewer than ``min_train_games`` prior games
    exist. That is a DELIBERATE design choice, stated here because the work
    package that added this function asked for it explicitly: a missing
    single-game PBP row falls back to MARKET-ONLY (``None``), the same
    contract wave 1's own ``model_total_view`` already has for a missing row
    -- it does NOT reach across to wave 1's model internally. The
    wave-1-VIEW fallback lives one level up, in
    ``nfl_ats.tiebreaker.tiebreaker_report``, and is scoped narrower: it
    fires only when the whole PBP table file is absent (a fresh clone),
    never merely because one game's row is missing from an existing table.
    Keeping this function's own contract as simple as wave 1's (one table
    in, one view or ``None`` out) is what keeps that higher-level fallback
    decision auditable instead of buried in two different places.
    """

    if not features_path.is_file():
        return None
    features = pd.read_parquet(features_path)
    if not _feature_table_matches_schedules(data_root, features):
        return None
    rows = features.loc[features["game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row.get("total_line")):
        return None
    season, week = int(row["season"]), int(row["week"])

    population = load_population_wave2(data_root, features_path)
    prior = population.loc[
        (population["season"] < season)
        | ((population["season"] == season) & (population["week"] < week))
    ]
    if len(prior) < min_train_games:
        return None

    estimator = make_totals_estimator(ridge_alpha=ridge_alpha)
    estimator.fit(design_matrix(prior, WAVE2_FEATURES), prior[_TARGET].astype(float).to_numpy())
    target_design = design_matrix(rows.iloc[[0]], WAVE2_FEATURES)
    residual = float(np.asarray(estimator.predict(target_design), dtype=float)[0])
    market_total = float(row["total_line"])
    return TotalsView(
        predicted_total=market_total + residual,
        market_total=market_total,
        residual=residual,
        train_games=len(prior),
        source=(
            f"totals ridge wave 2 (65 cols, drive pace, alpha={ridge_alpha:g}) "
            f"trained on {len(prior)} games before {season} week {week}"
        ),
    )


def wave_vs_wave_paired_frame(
    wave1_predictions: pd.DataFrame,
    wave1_weight: float,
    wave2_predictions: pd.DataFrame,
    wave2_weight: float,
) -> pd.DataFrame:
    """Per-game paired |error| difference, wave-1 blend minus wave-2 blend.

    POSITIVE = wave 2 is closer to the actual total on that game -- the same
    sign convention already stored in the registry for
    ``totals_market_residual_blend`` (baseline minus candidate,
    positive-is-better). The two prediction frames must carry the same
    ``game_id`` set (the wave-2 population is a strict feature superset of
    wave 1's, verified in the predeclaration, so this is a hard equality
    check rather than an inner join that could silently drop games).
    """

    left = wave1_predictions.loc[:, ["game_id", "season", "week"]].copy()
    left["game_id"] = left["game_id"].astype(str)
    right_ids = set(wave2_predictions["game_id"].astype(str))
    left_ids = set(left["game_id"])
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)[:5]
        only_right = sorted(right_ids - left_ids)[:5]
        raise TotalsDataError(
            "wave-1 and wave-2 scored game sets differ: "
            f"{len(left_ids)} vs {len(right_ids)} games; "
            f"only-in-wave1={only_left} only-in-wave2={only_right}"
        )

    wave1_blend = blend_total(
        wave1_predictions["market_total"], wave1_predictions["predicted_residual"], wave1_weight
    )
    wave1_abs = (wave1_blend - wave1_predictions["actual_total"].astype(float)).abs()
    wave1_by_game = pd.Series(
        wave1_abs.to_numpy(dtype=float),
        index=wave1_predictions["game_id"].astype(str).to_numpy(),
    )

    wave2_blend = blend_total(
        wave2_predictions["market_total"], wave2_predictions["predicted_residual"], wave2_weight
    )
    wave2_abs = (wave2_blend - wave2_predictions["actual_total"].astype(float)).abs()
    wave2_by_game = pd.Series(
        wave2_abs.to_numpy(dtype=float),
        index=wave2_predictions["game_id"].astype(str).to_numpy(),
    )

    left["wave1_abs_error"] = left["game_id"].map(wave1_by_game).to_numpy(dtype=float)
    left["wave2_abs_error"] = left["game_id"].map(wave2_by_game).to_numpy(dtype=float)
    left["abs_error_improvement"] = left["wave1_abs_error"] - left["wave2_abs_error"]
    return left


def _mean_improvement(frame: pd.DataFrame) -> dict[str, float]:
    return {"mae_improvement": float(frame["abs_error_improvement"].mean())}


def bootstrap_wave_vs_wave(
    paired: pd.DataFrame, *, samples: int = 2_000, seed: int = 20260901
) -> dict[str, float]:
    """Week-blocked bootstrap of wave 2's paired improvement over wave 1.

    Reuses ``nfl_ats.clv.week_blocked_bootstrap`` unmodified -- the same
    interval construction every arm of this project uses -- and surfaces
    ``probability_positive`` for "wave 2 beats wave 1," never a binary read
    of whether the interval crosses zero.
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
    chosen = sweep.loc[np.isclose(sweep["k"], weight)].iloc[0]
    return {
        "games": len(predictions),
        "chosen_k": float(weight),
        "chosen_blend_mae": float(chosen["mae"]),
        "chosen_blend_rmse": float(chosen["rmse"]),
        "mae_improvement_vs_market": float(chosen["mae_improvement_vs_market"]),
        "sweep": sweep.to_dict(orient="records"),
    }


def run_screen(
    data_root: Path,
    wave1_features_path: Path,
    wave2_features_path: Path,
    artifacts_root: Path,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260901,
    stamp: str | None = None,
) -> dict[str, Any]:
    """The frozen wave-2 screen: reproduce wave 1 fresh, run wave 2, pair them.

    Both arms use the identical guarded ``walk_forward_predictions`` from
    ``nfl_ats.totals``; only the feature table and column list differ.
    """

    wave1_population = load_population_wave1(data_root, wave1_features_path)
    wave1_predictions = walk_forward_predictions(
        wave1_population, ridge_alpha=ridge_alpha, min_train_games=min_train_games
    )
    wave1_regular = wave1_predictions.loc[wave1_predictions["game_type"] == "REG"].reset_index(
        drop=True
    )
    wave1_postseason = wave1_predictions.loc[wave1_predictions["game_type"] != "REG"].reset_index(
        drop=True
    )

    wave2_population = load_population_wave2(data_root, wave2_features_path)
    wave2_predictions = walk_forward_predictions(
        wave2_population,
        ridge_alpha=ridge_alpha,
        min_train_games=min_train_games,
        features=WAVE2_FEATURES,
    )
    wave2_regular = wave2_predictions.loc[wave2_predictions["game_type"] == "REG"].reset_index(
        drop=True
    )
    wave2_postseason = wave2_predictions.loc[wave2_predictions["game_type"] != "REG"].reset_index(
        drop=True
    )
    if wave1_regular.empty or wave2_regular.empty:
        raise TotalsDataError("walk-forward produced no regular-season predictions")

    wave2_sweep = blend_sweep(wave2_regular)
    wave2_weight = choose_weight(wave2_sweep)

    paired_regular = wave_vs_wave_paired_frame(
        wave1_regular, WAVE1_CHOSEN_K, wave2_regular, wave2_weight
    )
    bootstrap = bootstrap_wave_vs_wave(
        paired_regular, samples=bootstrap_samples, seed=bootstrap_seed
    )

    wave2_vs_market_paired = paired_error_frame(wave2_regular, wave2_weight)

    results: dict[str, Any] = {
        "contract": "docs/totals_model_wave2.md (predeclared 2026-09-01)",
        "target": _TARGET,
        "wave1_features": list(TOTALS_FEATURES),
        "wave2_drive_features": list(WAVE2_DRIVE_FEATURES),
        "wave2_features": list(WAVE2_FEATURES),
        "feature_count_wave1": len(TOTALS_FEATURES),
        "feature_count_wave2": len(WAVE2_FEATURES),
        "pipeline": "SimpleImputer(median, add_indicator) -> StandardScaler -> Ridge",
        "ridge_alpha": float(ridge_alpha),
        "min_train_games": int(min_train_games),
        "wave1_chosen_k": float(WAVE1_CHOSEN_K),
        "wave1_regular_season": _subset_summary(wave1_regular, WAVE1_CHOSEN_K),
        "wave2_regular_season": _subset_summary(wave2_regular, wave2_weight),
        "wave2_chosen_k": float(wave2_weight),
        "wave1_playoffs": (
            _subset_summary(wave1_postseason, WAVE1_CHOSEN_K)
            if not wave1_postseason.empty
            else {"games": 0}
        ),
        "wave2_playoffs": (
            _subset_summary(wave2_postseason, wave2_weight)
            if not wave2_postseason.empty
            else {"games": 0}
        ),
        "per_season_wave1": per_season_deltas(wave1_regular, WAVE1_CHOSEN_K).to_dict(
            orient="records"
        ),
        "per_season_wave2": per_season_deltas(wave2_regular, wave2_weight).to_dict(
            orient="records"
        ),
        "primary_bootstrap_wave2_vs_wave1": bootstrap,
        "primary_bootstrap_sign_convention": (
            "positive = wave 2's absolute error is SMALLER than wave 1's "
            "(wave 2 is better); wave 1 graded at its frozen k=0.1, wave 2 at "
            "its own MAE-minimizing k"
        ),
        "secondary_wave2_vs_market": {
            "mae_improvement_vs_market": float(
                wave2_sweep.loc[
                    np.isclose(wave2_sweep["k"], wave2_weight), "mae_improvement_vs_market"
                ].iloc[0]
            ),
            "games": len(wave2_vs_market_paired),
        },
    }

    if not wave1_postseason.empty and not wave2_postseason.empty:
        playoff_paired = wave_vs_wave_paired_frame(
            wave1_postseason, WAVE1_CHOSEN_K, wave2_postseason, wave2_weight
        )
        results["playoffs_wave2_vs_wave1_mean_improvement"] = float(
            playoff_paired["abs_error_improvement"].mean()
        )
        results["playoffs_games"] = len(playoff_paired)

    stamp = stamp or run_id()
    output_dir = artifacts_root / "totals_backtest_wave2" / stamp / "screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    wave1_predictions.to_parquet(output_dir / "wave1_predictions.parquet", index=False)
    wave2_predictions.to_parquet(output_dir / "wave2_predictions.parquet", index=False)
    paired_regular.to_parquet(output_dir / "paired_wave2_vs_wave1.parquet", index=False)
    atomic_json(results, output_dir / "results.json")
    results["output_dir"] = str(output_dir)
    return results


def run_positive_control(
    data_root: Path,
    wave2_features_path: Path,
    artifacts_root: Path,
    *,
    ridge_alpha: float = TOTALS_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260901,
    control_column: str = POSITIVE_CONTROL_COLUMN,
    stamp: str | None = None,
) -> dict[str, Any]:
    """Instrument-sanity check: inject the target into one drive column.

    Method frozen in ``docs/totals_model_wave2.md``: replace
    ``control_column`` with the row's own ``total_residual`` (unit slope,
    zero noise) for every row, run the SAME walk-forward + sweep + bootstrap
    pipeline the real screen uses, and confirm the machinery registers a
    large, unambiguous effect. This is a check on the pipeline, not a claim
    about the real screen's own (separately computed) result.
    """

    if control_column not in WAVE2_DRIVE_FEATURES:
        raise ValueError(f"control_column must be one of {WAVE2_DRIVE_FEATURES}")

    population = load_population_wave2(data_root, wave2_features_path)
    contaminated = population.copy()
    contaminated[control_column] = contaminated[_TARGET].astype(float)

    predictions = walk_forward_predictions(
        contaminated,
        ridge_alpha=ridge_alpha,
        min_train_games=min_train_games,
        features=WAVE2_FEATURES,
    )
    regular = predictions.loc[predictions["game_type"] == "REG"].reset_index(drop=True)
    if regular.empty:
        raise TotalsDataError("positive control produced no regular-season predictions")

    sweep = blend_sweep(regular)
    weight = choose_weight(sweep)
    paired = paired_error_frame(regular, weight)
    bootstrap_result = week_blocked_bootstrap(
        paired,
        lambda frame: {"mae_improvement": float(frame["abs_error_improvement"].mean())},
        block="week",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    row = bootstrap_result.iloc[0]

    results: dict[str, Any] = {
        "contract": "docs/totals_model_wave2.md (predeclared 2026-09-01), positive control",
        "control_column": control_column,
        "control_method": (
            "column replaced by its row's own total_residual (unit slope, zero noise)"
        ),
        "games": len(regular),
        "chosen_k": float(weight),
        "sweep": sweep.to_dict(orient="records"),
        "mae_improvement_vs_market": float(
            sweep.loc[np.isclose(sweep["k"], weight), "mae_improvement_vs_market"].iloc[0]
        ),
        "bootstrap": {
            "estimate": float(row["estimate"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
            "samples": int(row["samples"]),
        },
        "expected_shape": (
            "k near 1.0, large positive MAE improvement, probability_positive "
            "near 1.0 -- frozen in docs/totals_model_wave2.md before this ran"
        ),
        "shape_matches_expectation": bool(
            weight >= 0.8 and float(row["probability_positive"]) >= 0.95
        ),
    }

    stamp = stamp or run_id()
    output_dir = artifacts_root / "totals_backtest_wave2" / stamp / "positive_control"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    atomic_json(results, output_dir / "results.json")
    results["output_dir"] = str(output_dir)
    return results


def format_screen_results(results: dict[str, Any]) -> str:
    """Human-readable summary of :func:`run_screen`."""

    bootstrap = results["primary_bootstrap_wave2_vs_wave1"]
    wave1 = results["wave1_regular_season"]
    wave2 = results["wave2_regular_season"]
    lines = [
        f"wave-2 totals screen -- {wave2['games']} regular-season games",
        f"wave 1 (k={results['wave1_chosen_k']:.1f}, frozen): MAE {wave1['chosen_blend_mae']:.4f}",
        f"wave 2 (k={results['wave2_chosen_k']:.1f}, own sweep minimum): MAE "
        f"{wave2['chosen_blend_mae']:.4f}",
        f"wave 2 vs market: MAE improvement "
        f"{results['secondary_wave2_vs_market']['mae_improvement_vs_market']:+.4f}",
        "",
        f"primary -- wave2 vs wave1 paired |error| improvement: "
        f"{bootstrap['estimate']:+.4f} 95% [{bootstrap['lower']:+.4f}, {bootstrap['upper']:+.4f}], "
        f"probability_positive {bootstrap['probability_positive']:.3f} "
        f"({bootstrap['blocks']} week blocks, {bootstrap['samples']} resamples)",
    ]
    if "output_dir" in results:
        lines += ["", f"prediction-level output: {results['output_dir']}"]
    return "\n".join(lines)


def format_positive_control_results(results: dict[str, Any]) -> str:
    """Human-readable summary of :func:`run_positive_control`."""

    bootstrap = results["bootstrap"]
    lines = [
        f"positive control ({results['control_column']} <- total_residual) -- "
        f"{results['games']} regular-season games",
        f"chosen k = {results['chosen_k']:.1f}, MAE improvement vs market "
        f"{results['mae_improvement_vs_market']:+.4f}",
        f"bootstrap: {bootstrap['estimate']:+.4f} 95% [{bootstrap['lower']:+.4f}, "
        f"{bootstrap['upper']:+.4f}], probability_positive {bootstrap['probability_positive']:.3f}",
        f"expected shape: {results['expected_shape']}",
        f"shape_matches_expectation: {results['shape_matches_expectation']}",
    ]
    if "output_dir" in results:
        lines += ["", f"prediction-level output: {results['output_dir']}"]
    return "\n".join(lines)
