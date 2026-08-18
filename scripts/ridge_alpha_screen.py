"""Derive the global ridge penalty by leak-safe walk-forward selection on CFB.

The active NFL model (``market_residual`` / ``weak_stack`` / ``ridge``) has run
at ``ridge_alpha = 10.0`` since it was introduced, inherited rather than
derived. A companion diagnostic (``docs/ridge_alpha.md``) shows that value
shrinks the median principal direction of the fitted design by only ~0.27% at
the frozen alpha -- i.e. the model is unregularised least squares on every
direction that carries real signal, while the design also carries a genuine
null space (missing-indicator duplication dominates it; the ``diff = home -
away`` identity contributes a smaller, partly-approximate share -- see the
writeup).

This screen answers the derivation question directly: sweep alpha over
several orders of magnitude on the CFB benchmark (12,500 games, free under
rotation rule 8 -- no NFL confirmation window is spent) and read off the
walk-forward objective as a curve, with uncertainty. Everything here mirrors
``cfb_benchmark.cfb_walk_forward_benchmark`` exactly (same weekly cutoffs,
same feature contract, same distribution-fraction split); only the global
``ridge_alpha`` passed to ``fit_cfb_residual_model`` varies.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/ridge_alpha_screen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_RIDGE_ALPHA,
    cfb_walk_forward_benchmark,
)

# Predeclared grid: log-spaced from 1e-3 (essentially unregularised beyond
# what the null space forces) to 1e5 (heavy global shrinkage), plus the
# frozen 10.0 itself so it is scored on the identical instrument rather than
# only read off the neighbouring points.
ALPHA_GRID: tuple[float, ...] = (
    1e-3,
    1e-2,
    1e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    100_000.0,
)
REFERENCE_ALPHA = CFB_BENCHMARK_RIDGE_ALPHA  # 10.0, the frozen NFL/CFB default

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260818
FEATURES_PATH = Path("data/processed/cfb_game_features.parquet")
OUTPUT_DIR = Path("artifacts/ridge_alpha_screen")


def _components(frame: pd.DataFrame) -> pd.DataFrame:
    evaluated = frame.loc[frame["home_cover"].notna()].copy()
    actual = evaluated["home_cover"].astype(int).to_numpy()
    probability = evaluated["home_cover_probability"].to_numpy(dtype=float)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    evaluated["_correct"] = ((probability >= 0.5).astype(int) == actual).astype(float)
    evaluated["_brier"] = np.square(probability - actual)
    evaluated["_logloss"] = -(actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped))
    return evaluated


def _window_summary(frame: pd.DataFrame, alpha: float, window: str) -> dict[str, Any]:
    subset = frame if window == "all" else frame.loc[frame["evaluation_window"].eq(window)]
    parts = _components(subset)
    return {
        "ridge_alpha": alpha,
        "evaluation_window": window,
        "games": len(subset),
        "evaluated": len(parts),
        "accuracy": float(parts["_correct"].mean()),
        "brier": float(parts["_brier"].mean()),
        "log_loss": float(parts["_logloss"].mean()),
        "mean_abs_residual": float(subset["predicted_market_residual"].abs().mean()),
    }


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    window: str,
    block: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Blocked paired bootstrap of candidate-minus-baseline on the same games."""

    base = _components(
        baseline if window == "all" else baseline.loc[baseline["evaluation_window"].eq(window)]
    )
    cand = _components(
        candidate if window == "all" else candidate.loc[candidate["evaluation_window"].eq(window)]
    )
    merged = base.merge(cand, on="game_id", suffixes=("_b", "_c"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(cand):
        raise ValueError("Baseline and candidate do not cover the same games")

    group_columns = ["season_b", "week_b"] if block == "week" else ["season_b"]
    indices = list(merged.groupby(group_columns, sort=False).indices.values())
    stats = {
        "accuracy": (merged["_correct_c"].to_numpy(), merged["_correct_b"].to_numpy()),
        "brier": (merged["_brier_b"].to_numpy(), merged["_brier_c"].to_numpy()),
        "log_loss": (merged["_logloss_b"].to_numpy(), merged["_logloss_c"].to_numpy()),
    }
    generator = np.random.default_rng(seed)
    draws = {metric: np.empty(samples, dtype=float) for metric in stats}
    for draw in range(samples):
        selected = generator.integers(0, len(indices), size=len(indices))
        positions = np.concatenate([indices[index] for index in selected])
        for metric, (better, worse) in stats.items():
            draws[metric][draw] = float(better[positions].mean() - worse[positions].mean())

    rows: list[dict[str, Any]] = []
    for metric, (better, worse) in stats.items():
        sample = draws[metric]
        rows.append(
            {
                "ridge_alpha": float(candidate["ridge_alpha"].iloc[0]),
                "baseline_ridge_alpha": float(baseline["ridge_alpha"].iloc[0]),
                "evaluation_window": window,
                "block": block,
                "metric": metric,
                # Oriented so positive always means the swept alpha is better
                # than the frozen 10.0 reference (Brier/log-loss: lower is
                # better, so "better" there is baseline-minus-candidate).
                "improvement": float(better.mean() - worse.mean()),
                "lower": float(np.quantile(sample, 0.025)),
                "upper": float(np.quantile(sample, 0.975)),
                "probability_positive": float(np.mean(sample > 0.0)),
                "games": len(merged),
                "blocks": len(indices),
                "samples": int(samples),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHA_GRID))
    args = parser.parse_args()

    features = pd.read_parquet(args.features)

    results: dict[float, pd.DataFrame] = {}
    for alpha in args.alphas:
        print(f"[alpha={alpha:g}] running walk-forward benchmark ...", flush=True)
        outcome = cfb_walk_forward_benchmark(features, ridge_alpha=alpha)
        predictions = outcome.predictions.loc[outcome.predictions["method"].eq("market_residual")]
        predictions = predictions.assign(ridge_alpha=alpha).reset_index(drop=True)
        results[alpha] = predictions

    if REFERENCE_ALPHA not in results:
        print(
            f"[alpha={REFERENCE_ALPHA:g}] running reference walk-forward benchmark ...", flush=True
        )
        outcome = cfb_walk_forward_benchmark(features, ridge_alpha=REFERENCE_ALPHA)
        predictions = outcome.predictions.loc[outcome.predictions["method"].eq("market_residual")]
        results[REFERENCE_ALPHA] = predictions.assign(ridge_alpha=REFERENCE_ALPHA).reset_index(
            drop=True
        )

    reference = results[REFERENCE_ALPHA]

    summaries = [
        _window_summary(results[alpha], alpha, window)
        for alpha in args.alphas
        for window in ("clean_core", "all")
    ]
    comparisons: list[dict[str, Any]] = []
    for alpha in args.alphas:
        if alpha == REFERENCE_ALPHA:
            continue
        for window in ("clean_core", "all"):
            for block in ("week", "season"):
                comparisons.extend(
                    paired_bootstrap(reference, results[alpha], window=window, block=block)
                )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["evaluation_window", "ridge_alpha"], ignore_index=True
    )
    comparison_frame = pd.DataFrame(comparisons).sort_values(
        ["evaluation_window", "block", "metric", "ridge_alpha"], ignore_index=True
    )

    pd.set_option("display.width", 200)
    print("\n=== Alpha sweep summary ===")
    print(summary_frame.to_string(index=False))
    print("\n=== Paired blocked comparisons vs alpha=10 (reference) ===")
    print(comparison_frame.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_dir / "alpha_summary.csv", index=False)
    comparison_frame.to_csv(args.output_dir / "alpha_comparison.csv", index=False)
    (args.output_dir / "grid.json").write_text(
        json.dumps(
            {"alphas": list(args.alphas), "reference_alpha": REFERENCE_ALPHA},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
