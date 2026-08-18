"""Empirical validation for ``nfl_ats.purged_cv``: negative control, positive control,
leak-sensitivity break test, and the headline effective-sample quantification.

CFB-only (rule 8, no rotation cost), read-only against
``data/processed/cfb_game_features.parquet``. Writes every intermediate and
summary result as JSON into the scratchpad -- nothing here writes to
``artifacts/``, ``registry/``, or any tracked doc. Run with the project's uv
environment:

    ./.tools/uv.exe run --no-sync python scripts/purged_validate.py
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import CFB_BENCHMARK_MIN_TRAIN_GAMES, CFB_BENCHMARK_RIDGE_ALPHA
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.outcomes import outcome_bootstrap_intervals
from nfl_ats.purged_cv import (
    DEFAULT_EMBARGO_WEEKS,
    DEFAULT_PURGE_WEEKS,
    inject_synthetic_signal,
    permute_target,
    purged_cv_backtest,
    team_persistent_null,
)

SCRATCH = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\purged_cv"
)
DATA_PATH = Path("data/processed/cfb_game_features.parquet")


def _load() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def _accuracy_and_ci(predictions: pd.DataFrame, *, samples: int, seed: int) -> dict[str, float]:
    market_residual = predictions.loc[predictions["method"].eq("market_residual")]
    mask = market_residual["home_cover"].notna()
    n = int(mask.sum())
    point = float(
        (
            market_residual["home_cover_probability"].ge(0.5).astype(float)
            == market_residual["home_cover"]
        )
        .loc[mask]
        .mean()
    )
    # A separate diagnostic, not the headline number: the forced pick using
    # ONLY sign(predicted_margin - spread_line), bypassing the out-of-time
    # calibration/residual sample entirely. Comparing this to `accuracy`
    # isolates whether the calibration step (leakage channel #2 in
    # docs/purged_cv.md) is adding noise on top of the purge/embargo split
    # itself -- see the positive-control section of that doc.
    raw = market_residual["predicted_margin"] - market_residual["spread_line"]
    raw_sign_point = float(
        ((raw.gt(0)).astype(float) == market_residual["home_cover"]).loc[mask].mean()
    )
    intervals = outcome_bootstrap_intervals(predictions, samples=samples, block="week", seed=seed)
    row = intervals.loc[
        intervals["method"].eq("market_residual") & intervals["metric"].eq("cover_accuracy")
    ]
    lower = float(row["lower"].iloc[0]) if len(row) else float("nan")
    upper = float(row["upper"].iloc[0]) if len(row) else float("nan")
    return {
        "n": n,
        "accuracy": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "raw_sign_accuracy": raw_sign_point,
    }


def headline_run() -> dict[str, Any]:
    print("=== Headline purged CV run (measured defaults) ===", flush=True)
    df = _load()
    t0 = time.time()
    result = purged_cv_backtest(df, min_train_games=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    elapsed = time.time() - t0
    stats = _accuracy_and_ci(result.predictions, samples=500, seed=20260818)
    payload = {
        "elapsed_seconds": elapsed,
        "config": result.config,
        "stats": stats,
        "total_games_in_dataset": int(
            pd.to_numeric(df["ats_margin"], errors="coerce").notna().sum()
        ),
    }
    result.predictions.to_parquet(SCRATCH / "headline_predictions.parquet")
    result.fold_summary.to_csv(SCRATCH / "headline_fold_summary.csv", index=False)
    print(json.dumps(payload, indent=2, default=str))
    return payload


def negative_control(
    *,
    n_blocks: int,
    replicates: int,
    purge_weeks: int,
    embargo_weeks: int,
    label: str,
    null_fn: Callable[..., pd.DataFrame] = permute_target,
) -> dict[str, Any]:
    print(f"=== Negative control ({label}) ===", flush=True)
    df = _load()
    accuracies: list[float] = []
    excludes_half: list[bool] = []
    t0 = time.time()
    for replicate in range(replicates):
        permuted = null_fn(df, seed=1000 + replicate)
        result = purged_cv_backtest(
            permuted,
            n_blocks=n_blocks,
            purge_weeks=purge_weeks,
            embargo_weeks=embargo_weeks,
            min_train_games=CFB_BENCHMARK_MIN_TRAIN_GAMES,
            include_market_baseline=False,
        )
        stats = _accuracy_and_ci(result.predictions, samples=200, seed=2000 + replicate)
        accuracies.append(stats["accuracy"])
        excludes_half.append(not (stats["ci_lower"] <= 0.5 <= stats["ci_upper"]))
    elapsed = time.time() - t0
    payload = {
        "label": label,
        "n_blocks": n_blocks,
        "purge_weeks": purge_weeks,
        "embargo_weeks": embargo_weeks,
        "replicates": replicates,
        "elapsed_seconds": elapsed,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies, ddof=1)),
        "false_positive_rate": float(np.mean(excludes_half)),
        "accuracies": accuracies,
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "accuracies"}, indent=2))
    return payload


def positive_control(*, target_accuracy: float, n_blocks: int, label: str) -> dict[str, Any]:
    print(f"=== Positive control ({label}, target={target_accuracy}) ===", flush=True)
    df = _load()
    injected = inject_synthetic_signal(df, target_accuracy=target_accuracy, seed=42)
    # Population-truth check: the generative process's OWN accuracy, measured
    # directly on the 12,500-row sample with no model in between. Confirms
    # ``inject_synthetic_signal`` actually delivers what it claims before
    # asking whether purged CV can recover it.
    population_accuracy = float(
        (np.sign(injected["synthetic_signal"]) == np.sign(injected["ats_margin"])).mean()
    )
    feature_columns = (*CFB_MODEL_FEATURE_COLUMNS, "synthetic_signal")
    result = purged_cv_backtest(
        injected,
        n_blocks=n_blocks,
        purge_weeks=DEFAULT_PURGE_WEEKS,
        embargo_weeks=DEFAULT_EMBARGO_WEEKS,
        min_train_games=CFB_BENCHMARK_MIN_TRAIN_GAMES,
        feature_columns=feature_columns,
        include_market_baseline=False,
    )
    stats = _accuracy_and_ci(result.predictions, samples=500, seed=9999)
    payload = {
        "label": label,
        "target_accuracy": target_accuracy,
        "population_accuracy_realized": population_accuracy,
        "n_blocks": n_blocks,
        "recovered": stats,
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
    }
    print(json.dumps(payload, indent=2, default=str))
    return payload


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    results["headline"] = headline_run()

    results["negative_control_purged"] = negative_control(
        n_blocks=20,
        replicates=40,
        purge_weeks=DEFAULT_PURGE_WEEKS,
        embargo_weeks=DEFAULT_EMBARGO_WEEKS,
        label="measured_purge_embargo",
    )
    results["negative_control_zero"] = negative_control(
        n_blocks=20,
        replicates=40,
        purge_weeks=0,
        embargo_weeks=0,
        label="zero_purge_embargo",
    )

    results["positive_control_1_3pt"] = positive_control(
        target_accuracy=0.513, n_blocks=40, label="1.3pt_injury_scale"
    )
    results["positive_control_3pt"] = positive_control(
        target_accuracy=0.53, n_blocks=40, label="3pt_clear_signal"
    )

    # permute_target destroys every team-persistent structure in the target,
    # so it cannot tell a leaky split from a clean one (see purged_cv.py's
    # docstring). team_persistent_null keeps the one property real team
    # strength has that permute_target throws away -- a persistent
    # per-(team, season) component -- while still being independent of every
    # real feature (true population accuracy is exactly 50%). This is the
    # control that actually exercises the shared-team feature-proximity
    # channel purge/embargo are meant to guard.
    team_null = partial(team_persistent_null, team_sigma=8.0, noise_sigma=13.0)
    results["team_persistent_purged"] = negative_control(
        n_blocks=20,
        replicates=30,
        purge_weeks=DEFAULT_PURGE_WEEKS,
        embargo_weeks=DEFAULT_EMBARGO_WEEKS,
        label="team_persistent_measured_purge_embargo",
        null_fn=team_null,
    )
    results["team_persistent_zero"] = negative_control(
        n_blocks=20,
        replicates=30,
        purge_weeks=0,
        embargo_weeks=0,
        label="team_persistent_zero_purge_embargo",
        null_fn=team_null,
    )

    with (SCRATCH / "validation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {SCRATCH / 'validation_results.json'}")


if __name__ == "__main__":
    main()
