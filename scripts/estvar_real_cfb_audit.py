"""Refit-aware audit of real CFB candidate-vs-baseline comparisons (estvar).

Research item: ``docs/estimation_variance.md``. Runs two real comparisons
this project already cares about, once with the CURRENTLY REPORTED style
interval (one fit per season, game-block bootstrap only) and once with the
HONEST refit-aware interval (training-row bootstrap + refit per season,
combined with an independent game-block resample), on identical CFB
clean-core weeks so the two intervals are directly comparable.

Comparisons:

- **A: market vs market_residual** -- XLG-03's own flagship arm. The market
  arm (``margin.fit_market_baseline``) fits no estimator, so it carries zero
  refit variance by construction; only market_residual's ridge fit
  contributes. Candidate columns are the full frozen
  ``CFB_MODEL_FEATURE_COLUMNS`` contract, unchanged.
- **B: "thin" vs "full"** -- both ridge market_residual fits on
  ``fit_cfb_residual_model``, differing only in feature columns: "thin" is
  market + context + experience (11 columns, no team-state), "full" is the
  complete ``CFB_MODEL_FEATURE_COLUMNS`` (35 columns). Both arms carry refit
  variance, and each bootstrap draw resamples the SAME training rows for
  both arms (paired), which is the correct, conservative choice for a paired
  comparison.

Uses ANNUAL refit cadence (one fit per test season) rather than the
production weekly cadence, purely for compute -- refitting a bootstrap
ensemble every week across 200+ season-week cells would be roughly 15x the
cost. The real weekly-cadence XLG-03 benchmark
(``cfb_benchmark.cfb_walk_forward_benchmark``) is also run once, unmodified,
to confirm cadence alone does not explain most of the width difference this
script reports for comparison A.

The refit bootstrap resamples the FINAL-fit training rows and refits the
mean model; it does NOT also resample ``fit_cfb_residual_model``'s internal
80/20 residual-calibration split, so each season's out-of-time residual
sample is held fixed across that season's refit draws. Only the predicted
CENTRE varies -- exactly the "model-estimation variance" source the task
names (predicted-margin bootstrap SD), not residual-calibration noise (a
smaller, different source already partly explored in
``docs/residual_location.md``).

Also screens bagging, center-shrinkage, and the f-lever disagreement gate on
comparison B, reusing the bootstrap refits already computed for the interval
-- no extra fitting.

Not a rotation-registry look: CFB is free per rule 8 of
``registry/rotation_registry.json``, which this script only reads (never
writes). Writes JSON to the scratchpad only; never touches ``artifacts/`` or
the active model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_CLEAN_CORE_SEASONS,
    cfb_benchmark_uncertainty,
    cfb_walk_forward_benchmark,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import (
    CFB_CONTEXT_FEATURES,
    CFB_EXPERIENCE_FEATURES,
    CFB_MARKET_FEATURES,
    CFB_MODEL_FEATURE_COLUMNS,
)
from nfl_ats.estimation_variance import (
    gate_by_disagreement,
    home_cover_probability_from_center,
    mde80,
    naive_block_bootstrap_interval,
    picks_differ_fraction,
    refit_aware_paired_interval,
    refit_pick_flip_rate,
    refit_predicted_values,
)
from nfl_ats.margin import MarginModel, fit_market_baseline

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\estvar"
)

THIN_COLUMNS: tuple[str, ...] = (
    *CFB_MARKET_FEATURES,
    *CFB_CONTEXT_FEATURES,
    *CFB_EXPERIENCE_FEATURES,
)
FULL_COLUMNS: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS

N_BOOT = 120
RIDGE_ALPHA = CFB_BENCHMARK_RIDGE_ALPHA
MIN_TRAIN_GAMES = CFB_BENCHMARK_MIN_TRAIN_GAMES
SEASONS: tuple[int, ...] = tuple(sorted(CFB_CLEAN_CORE_SEASONS))


def _load_completed(features_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(features_path)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    return completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _season_cutoffs(
    completed: pd.DataFrame, seasons: tuple[int, ...]
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    out = []
    for season in seasons:
        test = completed.loc[completed["season"].eq(season)]
        if test.empty:
            continue
        cutoff = test["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < MIN_TRAIN_GAMES:
            continue
        test = test.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        keep = pd.to_numeric(test["home_cover"], errors="coerce").notna()
        out.append((season, training, test.loc[keep].reset_index(drop=True)))
    return out


def _refit_probabilities(
    raw_refits: np.ndarray, spread: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """Cover probability per (draw, game), looping over draws to bound memory.

    A single flattened call would need ``n_boot * n_test * len(residuals)``
    scratch floats -- gigabytes once ``len(residuals)`` reaches the
    thousands in the later, larger-training seasons.
    """

    n_boot = raw_refits.shape[0]
    out = np.empty_like(raw_refits)
    for draw in range(n_boot):
        centers = spread + raw_refits[draw]
        out[draw] = home_cover_probability_from_center(centers, spread, residuals)
    return out


@dataclass
class SeasonFit:
    season: int
    test: pd.DataFrame
    spread: np.ndarray
    actual: np.ndarray
    candidate_model: MarginModel
    candidate_raw: np.ndarray  # point raw prediction (n_test,)
    candidate_refits_raw: np.ndarray  # (n_boot, n_test)
    baseline_model: MarginModel | None  # None => unconditional market arm
    baseline_prob_point: np.ndarray
    baseline_refits_raw: np.ndarray | None  # (n_boot, n_test) or None


def fit_seasons(
    completed: pd.DataFrame,
    *,
    baseline_columns: tuple[str, ...] | None,
    candidate_columns: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> list[SeasonFit]:
    fits: list[SeasonFit] = []
    for season, training, test in _season_cutoffs(completed, SEASONS):
        spread = pd.to_numeric(test["spread_line"], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(test["home_cover"], errors="raise").to_numpy(dtype=float)
        n_test = len(test)

        candidate_model = fit_cfb_residual_model(
            training, ridge_alpha=RIDGE_ALPHA, feature_columns=candidate_columns
        )
        candidate_pred = candidate_model.predict(test)
        candidate_raw = candidate_pred["predicted_market_residual"].to_numpy(dtype=float)
        candidate_refits_raw = refit_predicted_values(
            training,
            test,
            feature_columns=candidate_columns,
            target_column="ats_margin",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=seed + season,
        )

        if baseline_columns is None:
            baseline_model = fit_market_baseline(training)
            baseline_prob_point = baseline_model.predict(test)["home_cover_probability"].to_numpy(
                dtype=float
            )
            baseline_refits_raw = None
        else:
            baseline_model = fit_cfb_residual_model(
                training, ridge_alpha=RIDGE_ALPHA, feature_columns=baseline_columns
            )
            baseline_pred = baseline_model.predict(test)
            baseline_prob_point = baseline_pred["home_cover_probability"].to_numpy(dtype=float)
            baseline_refits_raw = refit_predicted_values(
                training,
                test,
                feature_columns=baseline_columns,
                target_column="ats_margin",
                ridge_alpha=RIDGE_ALPHA,
                n_boot=n_boot,
                seed=seed + season,
            )

        fits.append(
            SeasonFit(
                season=season,
                test=test,
                spread=spread,
                actual=actual,
                candidate_model=candidate_model,
                candidate_raw=candidate_raw,
                candidate_refits_raw=candidate_refits_raw,
                baseline_model=baseline_model,
                baseline_prob_point=baseline_prob_point,
                baseline_refits_raw=baseline_refits_raw,
            )
        )
        print(
            f"  season {season}: train={candidate_model.training_rows} "
            f"test={n_test} residual_n={candidate_model.distribution_rows}",
            flush=True,
        )
    return fits


def pooled_arrays(fits: list[SeasonFit]) -> dict[str, np.ndarray]:
    """Point probabilities, block ids, and refit-probability arrays pooled across seasons."""

    actual = np.concatenate([f.actual for f in fits])
    block_ids = np.concatenate(
        [f.season * 100 + f.test["week"].to_numpy(dtype=np.int64) for f in fits]
    )
    candidate_prob = np.concatenate(
        [
            f.candidate_model.predict(f.test)["home_cover_probability"].to_numpy(dtype=float)
            for f in fits
        ]
    )
    baseline_prob = np.concatenate([f.baseline_prob_point for f in fits])

    candidate_refit_prob = np.concatenate(
        [
            _refit_probabilities(f.candidate_refits_raw, f.spread, f.candidate_model.residuals)
            for f in fits
        ],
        axis=1,
    )
    if fits[0].baseline_refits_raw is None:
        baseline_refit_prob = baseline_prob[np.newaxis, :]
    else:
        baseline_refit_prob = np.concatenate(
            [
                _refit_probabilities(
                    f.baseline_refits_raw,
                    f.spread,
                    f.baseline_model.residuals,  # type: ignore[arg-type]
                )
                for f in fits
            ],
            axis=1,
        )
    return {
        "actual": actual,
        "block_ids": block_ids,
        "baseline_prob": baseline_prob,
        "candidate_prob": candidate_prob,
        "baseline_refit_prob": baseline_refit_prob,
        "candidate_refit_prob": candidate_refit_prob,
    }


def candidate_flip_rate(fits: list[SeasonFit]) -> float:
    rates = []
    weights = []
    for f in fits:
        rates.append(refit_pick_flip_rate(f.candidate_raw, f.candidate_refits_raw))
        weights.append(len(f.test))
    return float(np.average(rates, weights=weights))


def bagging_report(fits: list[SeasonFit], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Split-half stability: bag the first/second half of refit draws, compare signs.

    Reuses the AGENTS.md vocabulary of split-half reliability: does the
    bagged predictor's sign survive using only half the bootstrap draws to
    build it, versus a single individual refit draw's sign against another
    individual refit draw's sign (the single-fit analogue).
    """

    single_pairwise_flip = []
    bagged_split_half_flip = []
    bagged_prob_chunks = []
    single_correct = []
    bagged_correct = []
    for f in fits:
        raw = f.candidate_refits_raw
        b = raw.shape[0]
        half = b // 2
        first_half = raw[:half].mean(axis=0)
        second_half = raw[half:].mean(axis=0)
        bagged_split_half_flip.append(float(np.mean(np.sign(first_half) != np.sign(second_half))))
        # Pairwise: compare consecutive individual draws (b0 vs b1, b2 vs b3, ...)
        even = raw[0 : 2 * (b // 2) : 2]
        odd = raw[1 : 2 * (b // 2) : 2]
        single_pairwise_flip.append(float(np.mean(np.sign(even) != np.sign(odd))))

        bagged_raw = raw.mean(axis=0)
        bagged_prob = home_cover_probability_from_center(
            f.spread + bagged_raw, f.spread, f.candidate_model.residuals
        )
        bagged_prob_chunks.append(bagged_prob)
        point_correct = (
            (f.candidate_model.predict(f.test)["home_cover_probability"] >= 0.5).to_numpy()
            == f.actual
        ).astype(float)
        bag_correct = ((bagged_prob >= 0.5) == f.actual).astype(float)
        single_correct.append(point_correct.mean())
        bagged_correct.append(bag_correct.mean())

    bagged_prob = np.concatenate(bagged_prob_chunks)
    interval = naive_block_bootstrap_interval(
        arrays["actual"],
        arrays["baseline_prob"],
        bagged_prob,
        arrays["block_ids"],
        samples=2_000,
        seed=20260818,
    )
    return {
        "single_fit_pairwise_flip_rate": float(np.mean(single_pairwise_flip)),
        "bagged_split_half_flip_rate": float(np.mean(bagged_split_half_flip)),
        "single_fit_mean_accuracy": float(np.mean(single_correct)),
        "bagged_mean_accuracy": float(np.mean(bagged_correct)),
        "bagged_vs_baseline_naive_interval": vars(interval),
    }


def shrinkage_report(fits: list[SeasonFit], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    grid = (1.0, 0.75, 0.5, 0.25, 0.0)
    rows = []
    for shrink_fraction in grid:
        prob_chunks = []
        flip_rates = []
        weights = []
        for f in fits:
            shrunk_center = f.spread + shrink_fraction * f.candidate_raw
            prob = home_cover_probability_from_center(
                shrunk_center, f.spread, f.candidate_model.residuals
            )
            prob_chunks.append(prob)
            shrunk_refits = shrink_fraction * f.candidate_refits_raw
            refit_prob = _refit_probabilities(shrunk_refits, f.spread, f.candidate_model.residuals)
            point_pick = prob >= 0.5
            flip_rates.append(float(np.mean((refit_prob >= 0.5) != point_pick[np.newaxis, :])))
            weights.append(len(f.test))
        candidate_prob = np.concatenate(prob_chunks)
        interval = naive_block_bootstrap_interval(
            arrays["actual"],
            arrays["baseline_prob"],
            candidate_prob,
            arrays["block_ids"],
            samples=1_500,
            seed=20260818,
        )
        rows.append(
            {
                "shrink_fraction": shrink_fraction,
                "estimate": interval.estimate,
                "probability_positive": interval.probability_positive,
                "flip_rate": float(np.average(flip_rates, weights=weights)),
            }
        )
    return {"rows": rows}


def f_lever_report(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    tau_grid = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25)
    rows = []
    n = len(arrays["actual"])
    for tau in tau_grid:
        gated = gate_by_disagreement(
            arrays["baseline_prob"], arrays["candidate_prob"], threshold=tau
        )
        f = picks_differ_fraction(arrays["baseline_prob"], gated)
        interval = naive_block_bootstrap_interval(
            arrays["actual"],
            arrays["baseline_prob"],
            gated,
            arrays["block_ids"],
            samples=2_000,
            seed=20260818,
        )
        rows.append(
            {
                "tau": tau,
                "f": f,
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "probability_positive": interval.probability_positive,
                "mde80": mde80(max(f, 1e-9), n),
            }
        )
    return {"rows": rows}


def main() -> None:
    started = time.time()
    completed = _load_completed(DEFAULT_CFB_FEATURES)
    print(f"Loaded {len(completed)} completed CFB games", flush=True)

    results: dict[str, Any] = {"seasons": list(SEASONS), "n_boot": N_BOOT}

    print("\n=== Comparison A: market vs market_residual (full columns) ===", flush=True)
    fits_a = fit_seasons(
        completed,
        baseline_columns=None,
        candidate_columns=FULL_COLUMNS,
        n_boot=N_BOOT,
        seed=1,
    )
    arrays_a = pooled_arrays(fits_a)
    report_a = {
        "label": "market vs market_residual (full)",
        "n_games": len(arrays_a["actual"]),
        "f": picks_differ_fraction(arrays_a["baseline_prob"], arrays_a["candidate_prob"]),
        "candidate_flip_rate": candidate_flip_rate(fits_a),
    }
    report_a["mde80"] = mde80(report_a["f"], report_a["n_games"])
    naive_a = naive_block_bootstrap_interval(
        arrays_a["actual"],
        arrays_a["baseline_prob"],
        arrays_a["candidate_prob"],
        arrays_a["block_ids"],
        samples=3_000,
        seed=20260818,
    )
    honest_a = refit_aware_paired_interval(
        arrays_a["actual"],
        arrays_a["baseline_refit_prob"],
        arrays_a["candidate_refit_prob"],
        arrays_a["block_ids"],
        seed=20260818,
    )
    report_a["naive"] = vars(naive_a)
    report_a["honest"] = vars(honest_a)
    report_a["width_inflation_ratio"] = (honest_a.upper - honest_a.lower) / (
        naive_a.upper - naive_a.lower
    )
    results["comparison_a"] = report_a
    print(json.dumps(report_a, indent=2, default=str), flush=True)

    print("\n=== Comparison A cadence check: real weekly XLG-03 benchmark ===", flush=True)
    weekly = cfb_walk_forward_benchmark(completed)
    weekly_uncertainty = cfb_benchmark_uncertainty(weekly.predictions)
    weekly_week = weekly_uncertainty.loc[
        weekly_uncertainty["block"].eq("week") & weekly_uncertainty["method"].eq("market_residual")
    ]
    results["comparison_a_weekly_cadence_real_benchmark"] = weekly_week.to_dict(orient="records")
    print(weekly_week.to_string(index=False), flush=True)

    print("\n=== Comparison B: thin vs full ===", flush=True)
    fits_b = fit_seasons(
        completed,
        baseline_columns=THIN_COLUMNS,
        candidate_columns=FULL_COLUMNS,
        n_boot=N_BOOT,
        seed=2,
    )
    arrays_b = pooled_arrays(fits_b)
    report_b = {
        "label": "thin vs full",
        "n_games": len(arrays_b["actual"]),
        "f": picks_differ_fraction(arrays_b["baseline_prob"], arrays_b["candidate_prob"]),
        "candidate_flip_rate": candidate_flip_rate(fits_b),
    }
    report_b["mde80"] = mde80(report_b["f"], report_b["n_games"])
    naive_b = naive_block_bootstrap_interval(
        arrays_b["actual"],
        arrays_b["baseline_prob"],
        arrays_b["candidate_prob"],
        arrays_b["block_ids"],
        samples=3_000,
        seed=20260818,
    )
    honest_b = refit_aware_paired_interval(
        arrays_b["actual"],
        arrays_b["baseline_refit_prob"],
        arrays_b["candidate_refit_prob"],
        arrays_b["block_ids"],
        seed=20260818,
    )
    report_b["naive"] = vars(naive_b)
    report_b["honest"] = vars(honest_b)
    report_b["width_inflation_ratio"] = (honest_b.upper - honest_b.lower) / (
        naive_b.upper - naive_b.lower
    )
    results["comparison_b"] = report_b
    print(json.dumps(report_b, indent=2, default=str), flush=True)

    print("\n=== Bagging (comparison B candidate) ===", flush=True)
    results["bagging"] = bagging_report(fits_b, arrays_b)
    print(json.dumps(results["bagging"], indent=2, default=str), flush=True)

    print("\n=== Center shrinkage (comparison B candidate) ===", flush=True)
    results["shrinkage"] = shrinkage_report(fits_b, arrays_b)
    print(json.dumps(results["shrinkage"], indent=2, default=str), flush=True)

    print("\n=== f lever (comparison B) ===", flush=True)
    results["f_lever"] = f_lever_report(arrays_b)
    print(json.dumps(results["f_lever"], indent=2, default=str), flush=True)

    out_path = OUT_DIR / "real_cfb_audit_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
