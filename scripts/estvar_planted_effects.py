"""Planted-effect validation for ``nfl_ats.estimation_variance`` (synthetic).

Research item: ``docs/estimation_variance.md``. Real CFB data never gives us
the true effect to check interval coverage against, so this is a self-
contained simulation study with a KNOWN ground truth. Answers three
questions:

1. Coverage: does the new refit-aware interval
   (``estimation_variance.refit_aware_paired_interval``) achieve ~95%
   coverage of a KNOWN true effect -- defined as the expectation over BOTH
   training-refit and test-game draws, "would a model fit this way beat one
   fit that way in general" -- while the currently-reported naive (game
   -block-only) interval under-covers it?
2. Power: do bagging and center-shrinkage actually raise detection power at a
   KNOWN planted effect, not just look plausible?
3. The f lever: does gating a candidate's influence to where it disagrees
   with the baseline reduce ``f`` faster than it destroys the true effect,
   raising detectability (``mde80 = 280 * sqrt(f / n)``)?

Every model is fit with the project's own ``margin.make_margin_estimator``
pipeline via ``nfl_ats.estimation_variance``, so the study exercises the
exact estimator whose variance this module measures, not a toy
re-implementation. Not a rotation-registry look (no real data, no picks);
writes JSON to the scratchpad, not ``artifacts/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.estimation_variance import (
    bagged_values,
    gate_by_disagreement,
    home_cover_probability_from_center,
    mde80,
    naive_block_bootstrap_interval,
    picks_differ_fraction,
    point_predicted_values,
    refit_aware_paired_interval,
    refit_pick_flip_rate,
    refit_predicted_values,
)
from nfl_ats.provenance import write_stamped_artifact

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\estvar"
)
RIDGE_ALPHA = 10.0
NOISE_SD = 13.0
BASELINE_COLUMNS = [f"b{i}" for i in range(4)]
EXTRA_COLUMNS = [f"e{i}" for i in range(3)]
ALL_COLUMNS = BASELINE_COLUMNS + EXTRA_COLUMNS
BASELINE_COEF = np.array([2.0, -1.5, 1.0, 0.8])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def make_frame(n: int, *, seed: int, extra_coef: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(n, len(BASELINE_COLUMNS)))
    x_extra = rng.normal(size=(n, len(EXTRA_COLUMNS)))
    mean = x_base @ BASELINE_COEF + x_extra @ extra_coef
    noise = rng.normal(scale=NOISE_SD, size=n)
    target = mean + noise
    frame = pd.DataFrame(x_base, columns=BASELINE_COLUMNS)
    for index, column in enumerate(EXTRA_COLUMNS):
        frame[column] = x_extra[:, index]
    frame["target"] = target
    frame["actual"] = (target > 0.0).astype(float)
    return frame


def block_ids_for(n: int, *, block_size: int = 20) -> np.ndarray:
    return (np.arange(n) // block_size).astype(np.int64)


def _accuracy_improvement_point(actual, baseline_prob, candidate_prob) -> float:
    base_correct = ((baseline_prob >= 0.5) == actual).astype(float)
    cand_correct = ((candidate_prob >= 0.5) == actual).astype(float)
    return float(np.mean(cand_correct - base_correct))


# ---------------------------------------------------------------------------
# Part 1: coverage of naive vs. refit-aware intervals against a KNOWN true effect
# ---------------------------------------------------------------------------


def true_effect_mc(
    *, extra_coef: np.ndarray, n_train: int, n_test_huge: int, mc_replicates: int, seed: int
) -> tuple[float, float]:
    """Monte-Carlo estimate of Delta_true = E_train[accuracy(candidate) - accuracy(baseline)].

    Each MC draw fits fresh baseline/candidate models on a fresh training set
    and scores them on a large fresh test set (so test-sampling noise is
    negligible and the draw-to-draw variance is almost entirely refit
    variance). Returns (mean, standard_error).
    """

    values = np.empty(mc_replicates, dtype=np.float64)
    for draw in range(mc_replicates):
        train = make_frame(n_train, seed=seed + 2 * draw, extra_coef=extra_coef)
        test = make_frame(n_test_huge, seed=seed + 2 * draw + 1, extra_coef=extra_coef)
        baseline_center = point_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        candidate_center = point_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        values[draw] = _accuracy_improvement_point(
            test["actual"].to_numpy(), sigmoid(baseline_center), sigmoid(candidate_center)
        )
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(mc_replicates))


def coverage_study(
    *,
    label: str,
    extra_coef: np.ndarray,
    n_train: int,
    n_test: int,
    replicates: int,
    n_boot: int,
    seed: int,
) -> dict:
    true_mean, true_se = true_effect_mc(
        extra_coef=extra_coef, n_train=n_train, n_test_huge=20_000, mc_replicates=300, seed=seed
    )

    naive_covers = 0
    honest_covers = 0
    naive_widths = []
    honest_widths = []
    naive_pp = []
    honest_pp = []
    flip_rates = []
    for draw in range(replicates):
        rep_seed = seed + 1_000 + draw
        train = make_frame(n_train, seed=rep_seed, extra_coef=extra_coef)
        test = make_frame(n_test, seed=rep_seed + 500_000, extra_coef=extra_coef)
        actual = test["actual"].to_numpy()
        blocks = block_ids_for(n_test)

        baseline_point = point_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        candidate_point = point_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        baseline_prob = sigmoid(baseline_point)
        candidate_prob = sigmoid(candidate_point)

        naive = naive_block_bootstrap_interval(
            actual, baseline_prob, candidate_prob, blocks, samples=1_500, seed=rep_seed
        )
        naive_covers += int(naive.lower <= true_mean <= naive.upper)
        naive_widths.append(naive.upper - naive.lower)
        naive_pp.append(naive.probability_positive)

        baseline_refits = refit_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=rep_seed + 1,
        )
        candidate_refits = refit_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=rep_seed + 1,
        )
        flip_rates.append(refit_pick_flip_rate(candidate_point, candidate_refits))
        honest = refit_aware_paired_interval(
            actual, sigmoid(baseline_refits), sigmoid(candidate_refits), blocks, seed=rep_seed
        )
        honest_covers += int(honest.lower <= true_mean <= honest.upper)
        honest_widths.append(honest.upper - honest.lower)
        honest_pp.append(honest.probability_positive)

    return {
        "label": label,
        "true_effect_mean": true_mean,
        "true_effect_mc_se": true_se,
        "n_train": n_train,
        "n_test": n_test,
        "replicates": replicates,
        "n_boot": n_boot,
        "naive_coverage": naive_covers / replicates,
        "honest_coverage": honest_covers / replicates,
        "naive_mean_width": float(np.mean(naive_widths)),
        "honest_mean_width": float(np.mean(honest_widths)),
        "width_inflation_ratio": float(np.mean(honest_widths) / np.mean(naive_widths)),
        "naive_mean_probability_positive": float(np.mean(naive_pp)),
        "honest_mean_probability_positive": float(np.mean(honest_pp)),
        "mean_candidate_flip_rate": float(np.mean(flip_rates)),
    }


# ---------------------------------------------------------------------------
# Part 2a: bagging -- power and own-model flip-rate reduction
# ---------------------------------------------------------------------------


def bagging_study(
    *, extra_coef: np.ndarray, n_train: int, n_test: int, replicates: int, n_boot: int, seed: int
) -> dict:
    single_correct = []
    bagged_correct = []
    single_flip = []
    bagged_flip = []
    single_pp_hits = 0
    bagged_pp_hits = 0
    detection_threshold = 0.80

    for draw in range(replicates):
        rep_seed = seed + 2_000 + draw
        train = make_frame(n_train, seed=rep_seed, extra_coef=extra_coef)
        test = make_frame(n_test, seed=rep_seed + 500_000, extra_coef=extra_coef)
        actual = test["actual"].to_numpy()
        blocks = block_ids_for(n_test)

        baseline_point = point_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        candidate_point = point_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        baseline_refits = refit_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=rep_seed + 1,
        )
        candidate_refits = refit_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=rep_seed + 1,
        )
        baseline_bag = bagged_values(baseline_refits)
        candidate_bag = bagged_values(candidate_refits)

        single_correct.append(
            _accuracy_improvement_point(actual, sigmoid(baseline_point), sigmoid(candidate_point))
        )
        bagged_correct.append(
            _accuracy_improvement_point(actual, sigmoid(baseline_bag), sigmoid(candidate_bag))
        )

        # Own-model instability: resample the training set ONCE MORE (a fresh
        # "different history" draw) and see whether the single-fit's sign and
        # the bagged predictor's sign move relative to their ORIGINAL values.
        fresh_train_indices = np.random.default_rng(rep_seed + 777).integers(
            0, n_train, size=n_train
        )
        fresh_train = train.iloc[fresh_train_indices].reset_index(drop=True)
        fresh_single = point_predicted_values(
            fresh_train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        single_flip.append(float(np.mean(np.sign(fresh_single) != np.sign(candidate_point))))

        fresh_bag_refits = refit_predicted_values(
            fresh_train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=max(20, n_boot // 2),
            seed=rep_seed + 999,
        )
        fresh_bag = bagged_values(fresh_bag_refits)
        bagged_flip.append(float(np.mean(np.sign(fresh_bag) != np.sign(candidate_bag))))

        naive = naive_block_bootstrap_interval(
            actual,
            sigmoid(baseline_point),
            sigmoid(candidate_point),
            blocks,
            samples=1_000,
            seed=rep_seed,
        )
        bagged_interval = naive_block_bootstrap_interval(
            actual,
            sigmoid(baseline_bag),
            sigmoid(candidate_bag),
            blocks,
            samples=1_000,
            seed=rep_seed,
        )
        single_pp_hits += int(naive.probability_positive >= detection_threshold)
        bagged_pp_hits += int(bagged_interval.probability_positive >= detection_threshold)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "replicates": replicates,
        "n_boot": n_boot,
        "detection_threshold": detection_threshold,
        "single_fit_mean_accuracy_improvement": float(np.mean(single_correct)),
        "bagged_mean_accuracy_improvement": float(np.mean(bagged_correct)),
        "single_fit_own_model_flip_rate": float(np.mean(single_flip)),
        "bagged_own_model_flip_rate": float(np.mean(bagged_flip)),
        "single_fit_detection_rate": single_pp_hits / replicates,
        "bagged_detection_rate": bagged_pp_hits / replicates,
    }


# ---------------------------------------------------------------------------
# Part 2b: center shrinkage -- uses a LOCATION-BIASED residual sample so a
# uniform positive scalar is NOT sign-invariant. Unlike naive coefficient
# scaling (which cannot change any sign-based pick, see MOD-06), the forced
# pick here reads a fixed threshold off a nonzero-location empirical sample,
# exactly like production's home_cover_probability -- so shrinking the CENTRE
# toward the market line moves the threshold relative to that fixed sample
# and genuinely can change picks.
# ---------------------------------------------------------------------------


def shrinkage_study(
    *,
    extra_coef: np.ndarray,
    n_train: int,
    n_test: int,
    replicates: int,
    n_boot: int,
    seed: int,
    residual_location: float,
    shrink_grid: tuple[float, ...],
) -> dict:
    rows = []
    for shrink_fraction in shrink_grid:
        accuracy_improvements = []
        flip_rates = []
        pp_values = []
        for draw in range(replicates):
            rep_seed = seed + 3_000 + draw
            train = make_frame(n_train, seed=rep_seed, extra_coef=extra_coef)
            test = make_frame(n_test, seed=rep_seed + 500_000, extra_coef=extra_coef)
            actual = test["actual"].to_numpy()
            blocks = block_ids_for(n_test)
            rng = np.random.default_rng(rep_seed + 4)
            residual_sample = rng.normal(loc=residual_location, scale=NOISE_SD, size=800)

            baseline_point = point_predicted_values(
                train,
                test,
                feature_columns=BASELINE_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
            candidate_point = point_predicted_values(
                train,
                test,
                feature_columns=ALL_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
            shrunk_candidate_center = shrink_fraction * candidate_point
            baseline_prob = home_cover_probability_from_center(
                baseline_point, np.zeros(n_test), residual_sample
            )
            candidate_prob = home_cover_probability_from_center(
                shrunk_candidate_center, np.zeros(n_test), residual_sample
            )
            accuracy_improvements.append(
                _accuracy_improvement_point(actual, baseline_prob, candidate_prob)
            )

            if n_boot > 0:
                candidate_refits = refit_predicted_values(
                    train,
                    test,
                    feature_columns=ALL_COLUMNS,
                    target_column="target",
                    ridge_alpha=RIDGE_ALPHA,
                    n_boot=n_boot,
                    seed=rep_seed + 1,
                )
                shrunk_refits = shrink_fraction * candidate_refits
                shrunk_refit_prob = home_cover_probability_from_center(
                    shrunk_refits.reshape(-1), np.zeros(n_boot * n_test), residual_sample
                ).reshape(n_boot, n_test)
                point_pick = candidate_prob >= 0.5
                refit_pick = shrunk_refit_prob >= 0.5
                flip_rates.append(float(np.mean(refit_pick != point_pick[np.newaxis, :])))

            naive = naive_block_bootstrap_interval(
                actual, baseline_prob, candidate_prob, blocks, samples=1_000, seed=rep_seed
            )
            pp_values.append(naive.probability_positive)

        rows.append(
            {
                "shrink_fraction": shrink_fraction,
                "mean_accuracy_improvement": float(np.mean(accuracy_improvements)),
                "mean_flip_rate": float(np.mean(flip_rates)) if flip_rates else None,
                "mean_probability_positive": float(np.mean(pp_values)),
            }
        )
    return {
        "residual_location": residual_location,
        "n_train": n_train,
        "n_test": n_test,
        "replicates": replicates,
        "n_boot": n_boot,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Part 3: the f lever -- gate a candidate to where it disagrees with the baseline
# ---------------------------------------------------------------------------


def make_subset_frame(n: int, *, seed: int, coef: float, threshold: float) -> pd.DataFrame:
    """Candidate's extra column ``z`` only carries real signal where |z| > threshold.

    A plain ridge on ``z`` is misspecified (it can't see the threshold), so it
    predicts roughly linearly in z everywhere -- but its DISAGREEMENT with the
    baseline still grows with |z|, which is exactly what correlates with the
    truly informative games. This is the mechanism the f-lever gate exploits.
    """

    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(n, len(BASELINE_COLUMNS)))
    z = rng.normal(size=n)
    informative = (np.abs(z) > threshold).astype(float)
    mean = x_base @ BASELINE_COEF + coef * z * informative
    noise = rng.normal(scale=NOISE_SD, size=n)
    target = mean + noise
    frame = pd.DataFrame(x_base, columns=BASELINE_COLUMNS)
    frame["z"] = z
    frame["target"] = target
    frame["actual"] = (target > 0.0).astype(float)
    return frame


def f_lever_study(
    *,
    coef: float,
    threshold: float,
    n_train: int,
    n_test: int,
    replicates: int,
    seed: int,
    tau_grid: tuple[float, ...],
) -> dict:
    subset_columns = [*BASELINE_COLUMNS, "z"]
    rows_by_tau: dict[float, list[dict]] = {tau: [] for tau in tau_grid}
    for draw in range(replicates):
        rep_seed = seed + 4_000 + draw
        train = make_subset_frame(n_train, seed=rep_seed, coef=coef, threshold=threshold)
        test = make_subset_frame(n_test, seed=rep_seed + 500_000, coef=coef, threshold=threshold)
        actual = test["actual"].to_numpy()
        blocks = block_ids_for(n_test)

        baseline_point = point_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        candidate_point = point_predicted_values(
            train,
            test,
            feature_columns=subset_columns,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        baseline_prob = sigmoid(baseline_point)
        candidate_prob = sigmoid(candidate_point)

        for tau in tau_grid:
            gated_prob = gate_by_disagreement(baseline_prob, candidate_prob, threshold=tau)
            f = picks_differ_fraction(baseline_prob, gated_prob)
            estimate = _accuracy_improvement_point(actual, baseline_prob, gated_prob)
            interval = naive_block_bootstrap_interval(
                actual, baseline_prob, gated_prob, blocks, samples=800, seed=rep_seed
            )
            rows_by_tau[tau].append(
                {
                    "f": f,
                    "estimate": estimate,
                    "probability_positive": interval.probability_positive,
                    "mde80": mde80(max(f, 1e-9), n_test),
                }
            )

    summary = []
    for tau in tau_grid:
        records = rows_by_tau[tau]
        summary.append(
            {
                "tau": tau,
                "mean_f": float(np.mean([r["f"] for r in records])),
                "mean_estimate": float(np.mean([r["estimate"] for r in records])),
                "mean_probability_positive": float(
                    np.mean([r["probability_positive"] for r in records])
                ),
                "mean_mde80": float(np.mean([r["mde80"] for r in records])),
            }
        )
    return {
        "coef": coef,
        "threshold": threshold,
        "n_train": n_train,
        "n_test": n_test,
        "replicates": replicates,
        "tau_grid": list(tau_grid),
        "summary": summary,
    }


def main() -> None:
    started = time.time()
    results: dict = {}

    null_coef = np.array([0.0, 0.0, 0.0])
    effect_coef = np.array([1.5, 1.2, 0.9])  # calibrated: true effect ~1.6 accuracy points

    print("=== Part 1: coverage (null DGP) ===", flush=True)
    results["coverage_null"] = coverage_study(
        label="null (candidate carries no true information)",
        extra_coef=null_coef,
        n_train=1_200,
        n_test=400,
        replicates=200,
        n_boot=80,
        seed=101,
    )
    print(json.dumps(results["coverage_null"], indent=2), flush=True)

    print("=== Part 1: coverage (real-effect DGP) ===", flush=True)
    results["coverage_effect"] = coverage_study(
        label="real effect (candidate carries genuine information)",
        extra_coef=effect_coef,
        n_train=1_200,
        n_test=400,
        replicates=200,
        n_boot=80,
        seed=202,
    )
    print(json.dumps(results["coverage_effect"], indent=2), flush=True)

    print("=== Part 2a: bagging ===", flush=True)
    results["bagging"] = bagging_study(
        extra_coef=effect_coef,
        n_train=800,
        n_test=400,
        replicates=120,
        n_boot=60,
        seed=303,
    )
    print(json.dumps(results["bagging"], indent=2), flush=True)

    print("=== Part 2b: center shrinkage ===", flush=True)
    results["shrinkage"] = shrinkage_study(
        extra_coef=effect_coef,
        n_train=1_200,
        n_test=400,
        replicates=100,
        n_boot=50,
        seed=404,
        residual_location=0.8,
        shrink_grid=(1.0, 0.75, 0.5, 0.25, 0.0),
    )
    print(json.dumps(results["shrinkage"], indent=2), flush=True)

    print("=== Part 3: f lever ===", flush=True)
    results["f_lever"] = f_lever_study(
        coef=3.2,
        threshold=1.0,
        n_train=1_500,
        n_test=900,
        replicates=40,
        seed=505,
        tau_grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35),
    )
    print(json.dumps(results["f_lever"], indent=2), flush=True)

    out_path = OUT_DIR / "planted_effects_results.json"
    write_stamped_artifact(results, out_path)  # ENG-38
    print(f"\nWrote {out_path} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
