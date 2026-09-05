"""Part 1 -- the learning-curve diagnostic: data-limited or model-limited?

Nobody had plotted this project's learning curve before. This script does,
on two substrates:

1. **CFB (primary, well-powered, free under rotation-registry rule 8).**
   Holds a fixed 2023-2025 test window and refits the frozen XLG-03 ridge
   recipe (``nfl_ats.cfb_benchmark.fit_cfb_residual_model``, byte-identical
   configuration) with training TRUNCATED to the most recent K games before
   each test week's cutoff, for a doubling grid of K. Same point-in-time
   contract as the frozen benchmark; only the amount of training history
   differs between arms.
2. **NFL (secondary, read-only, no fresh window spent).** Re-slices the
   ALREADY-FROZEN weak_stack backtest artifact
   (``artifacts/margins/20260818T012407Z/predictions.parquet``, the active
   model's own historical evaluation) by its already-recorded ``train_rows``
   column -- pure re-aggregation of predictions that were fit and scored
   before this session started, never a new walk-forward.

Fits a power-law curve ``metric(N) = a + b * N^-c`` to the CFB points (via
nonlinear least squares) and extrapolates. Writes everything to
``--output`` (a scratchpad directory) as parquet/csv/json; never touches
``artifacts/`` or the rotation registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from nfl_ats.cfb_benchmark import _score_week, fit_cfb_residual_model
from nfl_ats.margin import fit_market_baseline
from nfl_ats.outcomes import outcome_bootstrap_intervals, summarize_outcome_method

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
DEFAULT_NFL_FROZEN_PREDICTIONS = REPO / "artifacts/margins/20260818T012407Z/predictions.parquet"

# Doubling grid: the standard scaling-law convention, easy to reason about on
# a log axis. "full" means no truncation -- every completed game strictly
# before the cutoff, i.e. the frozen benchmark's own behavior.
CFB_TRAIN_SIZE_GRID: tuple[int | str, ...] = (100, 200, 400, 800, 1600, 3200, 6400, "full")
CFB_TEST_START_SEASON = 2023
CFB_TEST_END_SEASON = 2025
CFB_RIDGE_ALPHA = 10.0
CFB_MIN_TRAIN_GAMES = 50
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260818


def _method_label(k: int | str) -> str:
    return f"k_{k:05d}" if isinstance(k, int) else "k_full"


def cfb_learning_curve_predictions(
    features: pd.DataFrame,
    *,
    grid: tuple[int | str, ...] = CFB_TRAIN_SIZE_GRID,
    test_start: int = CFB_TEST_START_SEASON,
    test_end: int = CFB_TEST_END_SEASON,
    ridge_alpha: float = CFB_RIDGE_ALPHA,
    min_train_games: int = CFB_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = (
        frame.loc[
            pd.to_numeric(frame["result"], errors="coerce").notna()
            & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
        ]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    test = completed.loc[completed["season"].between(test_start, test_end)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {test_start} to {test_end}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        eligible = completed.loc[completed["gameday"].lt(cutoff)]
        models = {"market": fit_market_baseline(eligible)}
        for k in grid:
            training = eligible if k == "full" else eligible.tail(int(k))
            if len(training) < min_train_games:
                continue
            models[_method_label(k)] = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)
        batches.extend(_score_week(weekly_games, models))
    predictions = pd.concat(batches, ignore_index=True)
    return predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)


def _power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.asarray(a + b * np.power(n, -c), dtype=np.float64)


def fit_power_law(train_rows: np.ndarray, metric: np.ndarray) -> dict[str, Any]:
    """Fit ``metric(n) = a + b * n^-c`` by nonlinear least squares.

    Initial guess: ``a`` at the smallest observed metric value (a floor),
    ``b`` at the observed range, ``c=0.3`` (a mild, typical decay). Bounds
    keep ``b, c >= 0`` (error must not increase with more data) and ``a`` free
    but real.
    """

    lower = np.array([-np.inf, 0.0, 0.0])
    upper = np.array([np.inf, np.inf, 5.0])
    initial = np.array([float(np.min(metric)), float(np.ptp(metric)) + 1e-6, 0.3])
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
    try:
        params, covariance = curve_fit(
            _power_law, train_rows, metric, p0=initial, bounds=(lower, upper), maxfev=20000
        )
    except RuntimeError as error:
        return {"converged": False, "error": str(error)}
    residuals = metric - _power_law(train_rows, *params)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((metric - np.mean(metric)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "converged": True,
        "a_asymptote": float(params[0]),
        "b_amplitude": float(params[1]),
        "c_decay": float(params[2]),
        "r_squared": r_squared,
        "param_std_errors": (
            [float(value) for value in np.sqrt(np.diag(covariance))]
            if np.all(np.isfinite(covariance))
            else None
        ),
    }


def extrapolate_power_law(fit: dict[str, Any], *, targets: tuple[float, ...]) -> dict[str, float]:
    """For a fitted ``a + b*n^-c``, the N required to reach ``a + target`` above the floor."""

    if not fit.get("converged"):
        return {}
    b, c = fit["b_amplitude"], fit["c_decay"]
    result: dict[str, float] = {}
    for target in targets:
        if target <= 0 or b <= 0 or c <= 0:
            result[f"n_for_gap_{target:g}"] = float("inf")
            continue
        result[f"n_for_gap_{target:g}"] = float((b / target) ** (1.0 / c))
    return result


def nfl_frozen_reslice(
    predictions_path: Path, *, method: str = "market_residual", bins: int = 8
) -> pd.DataFrame:
    """Re-aggregate an ALREADY-FROZEN NFL backtest by its own recorded ``train_rows``.

    No new fit, no new NFL walk-forward: this groups predictions that were
    already made (strictly-earlier-training, already scored against real
    outcomes) by the training-set size they were made with, and computes
    accuracy / Brier / margin MAE per bin. Pure re-aggregation.
    """

    predictions = pd.read_parquet(predictions_path)
    subset = predictions.loc[predictions["method"].eq(method)].copy()
    subset["train_rows_bin"] = pd.qcut(subset["train_rows"], q=bins, duplicates="drop")
    rows = []
    for bin_label, group in subset.groupby("train_rows_bin", sort=True, observed=True):
        summary = summarize_outcome_method(group)
        rows.append(
            {
                "train_rows_bin": str(bin_label),
                "train_rows_min": int(group["train_rows"].min()),
                "train_rows_max": int(group["train_rows"].max()),
                "train_rows_mean": float(group["train_rows"].mean()),
                "games": summary.get("games"),
                "cover_accuracy": summary.get("cover_accuracy"),
                "cover_brier_score": summary.get("cover_brier_score"),
                "margin_mae": summary.get("margin_mae"),
                "margin_rmse": summary.get("margin_rmse"),
            }
        )
    return pd.DataFrame(rows).sort_values("train_rows_mean").reset_index(drop=True)


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument(
        "--nfl-frozen-predictions", type=Path, default=DEFAULT_NFL_FROZEN_PREDICTIONS
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    print("CFB learning curve: walking forward the doubling training-size grid")
    features = pd.read_parquet(arguments.cfb_features)
    predictions = cfb_learning_curve_predictions(features)
    predictions.to_parquet(output / "cfb_learning_curve_predictions.parquet", index=False)

    summary_rows = []
    for method, group in predictions.groupby("method", sort=True):
        summary = summarize_outcome_method(group)
        summary_rows.append({"method": method, **summary})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "cfb_learning_curve_summary.csv", index=False)
    print(summary.loc[:, ["method", "games", "cover_accuracy", "cover_brier_score", "margin_mae"]])

    print("\nBootstrapping week-blocked intervals (this can take a minute)...")
    intervals = outcome_bootstrap_intervals(
        predictions, samples=BOOTSTRAP_SAMPLES, block="week", seed=BOOTSTRAP_SEED
    )
    intervals.to_csv(output / "cfb_learning_curve_intervals.csv", index=False)

    # Actual mean train_rows realized per arm (may be < the nominal K for the
    # earliest scored weeks if fewer than K games were available yet).
    train_rows_by_method = predictions.groupby("method")["train_rows"].mean()
    k_methods = [m for m in summary["method"] if m.startswith("k_")]
    curve_points = (
        pd.DataFrame(
            {
                "method": k_methods,
                "train_rows": [train_rows_by_method[m] for m in k_methods],
                "accuracy": [
                    summary.set_index("method").loc[m, "cover_accuracy"] for m in k_methods
                ],
                "brier": [
                    summary.set_index("method").loc[m, "cover_brier_score"] for m in k_methods
                ],
                "margin_mae": [summary.set_index("method").loc[m, "margin_mae"] for m in k_methods],
            }
        )
        .sort_values("train_rows")
        .reset_index(drop=True)
    )
    curve_points.to_csv(output / "cfb_curve_points.csv", index=False)

    n = curve_points["train_rows"].to_numpy(dtype=float)
    mae_fit = fit_power_law(n, curve_points["margin_mae"].to_numpy(dtype=float))
    brier_fit = fit_power_law(n, curve_points["brier"].to_numpy(dtype=float))
    mae_extrapolation = extrapolate_power_law(mae_fit, targets=(0.1, 0.05, 0.0129, 0.01))
    brier_extrapolation = extrapolate_power_law(brier_fit, targets=(0.005, 0.001))

    print("\nPower-law fit, margin MAE:", json.dumps(mae_fit, indent=2))
    print(
        "Extrapolation (games needed to close a given MAE gap):",
        json.dumps(mae_extrapolation, indent=2),
    )
    print("\nPower-law fit, Brier:", json.dumps(brier_fit, indent=2))

    print("\nNFL frozen-artifact re-slice (read-only, no new window)")
    nfl_curve = nfl_frozen_reslice(arguments.nfl_frozen_predictions)
    nfl_curve.to_csv(output / "nfl_frozen_reslice.csv", index=False)
    print(nfl_curve)

    diagnostics: dict[str, Any] = {
        "cfb_test_window": [CFB_TEST_START_SEASON, CFB_TEST_END_SEASON],
        "cfb_train_size_grid": list(CFB_TRAIN_SIZE_GRID),
        "cfb_ridge_alpha": CFB_RIDGE_ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "mae_power_law_fit": mae_fit,
        "mae_extrapolation_games_needed": mae_extrapolation,
        "brier_power_law_fit": brier_fit,
        "brier_extrapolation_games_needed": brier_extrapolation,
        "nfl_frozen_predictions_path": str(arguments.nfl_frozen_predictions),
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float)
    )
    print(f"\nartifacts written to {output}")


if __name__ == "__main__":
    main()
