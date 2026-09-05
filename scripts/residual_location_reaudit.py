"""Re-measurement of the two ``refuted_mechanism`` residual_location recency arms.

Research item: ``docs/revisit_list.md`` Tier 1 (last row), ``docs/residual_location.md``
sec 4. ``residual_location_recency_hl200_cfb`` and ``residual_location_recency_hl400_cfb``
are recorded ``refuted_mechanism`` -- a TERMINAL verdict -- in
``registry/weak_signals.json`` on ``probability_positive`` 0.014 / 0.0005. The
2026-08-18 instrument audit found the block bootstrap these numbers came from
under-covers (D2, ``docs/estimation_variance.md``) and raised the default
bootstrap sample count from 2,000 to 20,000 (D3) because seed-to-seed jitter at
2,000 was 0.02-0.03 points. This script settles whether the terminal verdict
survives honest treatment. CFB only, free and unlimited per rotation-registry
rule 8; no NFL window touched; does not write ``registry/*.json``.

Five things happen here, reusing one CFB walk-forward pass
(``residual_location_screen.run_cfb``, extended with an optional
``weekly_capture`` hook rather than duplicated):

1. **Reproduce** ``docs/residual_location.md`` sec 4 bit-for-bit
   (``samples=2000, seed=20260818`` -- the script's own recorded defaults).
2. **D3 check.** Re-run the identical predictions at ``samples=20000`` (the
   corrected default), plus three more 20,000-sample seeds, to measure
   seed-to-seed jitter at the NEW sample count on these two specific rows --
   particularly ``hl200``, whose recorded upper bound (-0.06 pts) sits closest
   to the zero gate of any row in the family.
3. **Honest, mechanism-targeted interval.** ``docs/estimation_variance.md``
   sec 7 explicitly disclaims its refit-aware bootstrap (resample TRAINING
   ROWS, refit the mean Ridge model) for this family: these nine arms share
   one mean model and differ ONLY in how a fixed out-of-time residual sample
   is READ, so resampling training rows and refitting contributes ~nothing to
   what separates them -- sec 8's declared limitation #2 names the actual
   unaddressed source: "it does NOT also resample fit_cfb_residual_model's
   internal 80/20 residual-calibration split. Each season's out-of-time
   residual sample is held fixed across... refit draws." That IS what these
   two arms' comparison turns on. This script builds that bootstrap instead:
   for every week, independently resample (with replacement, chronological
   order preserved so recency weighting stays meaningful) that week's own
   out-of-time residual sample, recompute BOTH readers' probabilities from the
   resampled sample with the mean model's centres held fixed, and combine with
   an independent block resample of games in the same outer loop --
   generalizing ``estimation_variance.refit_aware_paired_interval``'s design
   from "refit the model" to "resample the calibration draws," the correct
   substitution for a reader-only family. As a second, cruder comparator, the
   blanket 17-58% naive-interval-width inflation ``docs/estimation_variance.md``
   sec 3 measured on (different-model) comparisons is also applied, so the two
   can be read side by side with the mismatch stated plainly.
4. **D1 cross-check.** ``sign(predicted_margin - spread_line)`` never touches
   ``model.residuals`` -- it is IDENTICAL across all 9 arms by construction,
   confirmed numerically below, not assumed. A literal sign-only paired
   contrast between any two of these arms is therefore fully degenerate
   (``f=0``). The closest valid substitute is measured instead: how far each
   reader's forced pick departs from that same sign-only rule (a synthetic
   10th arm), and whether recency weighting departs MORE or LESS from it than
   the production ``ecdf`` baseline does.
5. **Power arithmetic.** ``f`` = fraction of games ``hl200``/``hl400`` pick
   differently from the ``ecdf`` baseline, ``MDE80 = 280*sqrt(f/n)``
   (``nfl_ats.estimation_variance.DEFAULT_MDE80_COEFFICIENT``), reported for
   both the full pipeline and the (degenerate) sign-only estimator.

Writes JSON + CSV to the scratchpad only; never touches ``artifacts/`` or the
active model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import residual_location_screen as rls  # noqa: E402  (sibling script reuse)

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS  # noqa: E402
from nfl_ats.estimation_variance import (  # noqa: E402
    PairedInterval,
    mde80,
    picks_differ_fraction,
    refit_aware_paired_interval,
)
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402
from nfl_ats.residual_location import recency_weights  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\d08576b1-0b16-4ebc-9ebc-f0bc62b943d3\scratchpad\residual_location_reaudit"
)
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

FOCAL_ARMS: dict[str, float] = {"recency_hl200": 200.0, "recency_hl400": 400.0}
SIGN_ONLY_ARM = "sign_only"

# The exact recorded configuration behind registry/weak_signals.json's two
# refuted_mechanism entries; source: docs/residual_location.md sec 4 and
# artifacts/residual_location/20260818T115234Z/cfb_paired_comparisons.csv.
RECORDED = {
    "recency_hl200": {
        "estimate": -0.005485,
        "lower": -0.010774,
        "upper": -0.000562,
        "probability_positive": 0.014,
    },
    "recency_hl400": {
        "estimate": -0.005597,
        "lower": -0.009143,
        "upper": -0.001928,
        "probability_positive": 0.0005,
    },
}

# D2's measured naive-interval-width-inflation range (docs/estimation_variance.md
# sec 2-3), applied only as a crude sensitivity comparator -- sec 7 of that same
# document states it does not model this family's actual variance source.
D2_WIDTH_INFLATION_LOW = 1.037
D2_WIDTH_INFLATION_HIGH = 1.575


def _paired_frame(predictions: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core") & predictions["feature_set"].isin(arms)
    ].copy()
    return clean


def d1_degeneracy_check(predictions: pd.DataFrame) -> dict[str, Any]:
    """Confirm sign(predicted_margin - spread_line) is identical across all 9 arms."""

    clean = predictions.loc[predictions["evaluation_window"].eq("clean_core")].copy()
    clean["sign_pick"] = (clean["predicted_margin"] > clean["spread_line"]).astype(float)
    pivot = clean.pivot(index="game_id", columns="feature_set", values="sign_pick")
    reference = pivot[rls.BASELINE_ARM]
    disagreements = {
        column: int((pivot[column] != reference).sum())
        for column in pivot.columns
        if column != rls.BASELINE_ARM
    }
    return {
        "n_games": len(pivot),
        "n_arms_checked": int(pivot.shape[1]),
        "max_disagreement_vs_ecdf": max(disagreements.values()),
        "all_arms_identical_sign_pick": all(count == 0 for count in disagreements.values()),
        "disagreements_by_arm": disagreements,
    }


def d1_closest_valid_contrast(predictions: pd.DataFrame) -> dict[str, Any]:
    """Each reader vs. a synthetic sign-only arm.

    The substitute for a degenerate reader-vs-reader sign test.
    """

    arms = (rls.BASELINE_ARM, *FOCAL_ARMS)
    clean = _paired_frame(predictions, arms).copy()
    sign_only = clean.loc[clean["feature_set"].eq(rls.BASELINE_ARM)].copy()
    sign_only["feature_set"] = SIGN_ONLY_ARM
    sign_only["home_cover_probability"] = (
        sign_only["predicted_margin"] > sign_only["spread_line"]
    ).astype(float)
    frame = pd.concat([clean, sign_only], ignore_index=True)

    results: dict[str, Any] = {}
    for block in ("season", "week"):
        paired = paired_feature_comparisons(
            frame, baseline_feature_set=SIGN_ONLY_ARM, samples=20_000, block=block, seed=20260818
        )
        rows = paired.loc[paired["metric"].eq("accuracy_improvement")]
        results[block] = {
            str(row["candidate_feature_set"]): {
                "estimate_pts": float(row["estimate"]) * 100.0,
                "lower_pts": float(row["lower"]) * 100.0,
                "upper_pts": float(row["upper"]) * 100.0,
                "probability_positive": float(row["probability_positive"]),
            }
            for _, row in rows.iterrows()
        }
    return results


def power_arithmetic(predictions: pd.DataFrame) -> dict[str, Any]:
    clean = _paired_frame(predictions, (rls.BASELINE_ARM, *FOCAL_ARMS))
    pivot = clean.pivot(index="game_id", columns="feature_set", values="home_cover_probability")
    n = len(pivot)
    baseline = pivot[rls.BASELINE_ARM].to_numpy(dtype=float)
    out: dict[str, Any] = {"n_games": int(n)}
    for arm in FOCAL_ARMS:
        f = picks_differ_fraction(baseline, pivot[arm].to_numpy(dtype=float))
        out[arm] = {"f": f, "mde80_pts": mde80(f, n)}

    # Sign-only power: f is 0 by construction (d1_degeneracy_check confirms this
    # directly); mde80 is undefined at f=0 and reported as such, not as 0.0.
    sign_pick = (pivot[rls.BASELINE_ARM].to_numpy(dtype=float) >= 0.5).astype(float)
    out["sign_only_f_vs_ecdf"] = 0.0
    out["sign_only_mde80_pts"] = None
    del sign_pick
    return out


# ---------------------------------------------------------------------------
# Honest, mechanism-targeted interval: resample the residual calibration
# sample itself (order-preserving), hold the mean model's centres fixed.
# ---------------------------------------------------------------------------


class WeeklyCache:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def capture(self, season: int, week: int, weekly_games: pd.DataFrame, model: Any) -> None:
        if season not in CFB_CLEAN_CORE_SEASONS:
            return
        predicted = model.predict(weekly_games)
        centers = predicted["predicted_margin"].to_numpy(dtype=float)
        spread = weekly_games["spread_line"].to_numpy(dtype=float)
        actual = pd.to_numeric(weekly_games["home_cover"], errors="raise").to_numpy(dtype=float)
        # experiments.paired_feature_comparisons drops NaN-home_cover (push) rows
        # before computing anything (src/nfl_ats/experiments.py: paired.loc[...
        # .notna() & ...notna()]) -- clean_core carries exactly 160 such games
        # (9,093 raw vs. the recorded 8,933 paired_games). Match that filter here
        # so this honest interval is built on the identical game set the naive
        # recorded interval was, not a slightly different (and NaN-diluted) one.
        keep = ~np.isnan(actual)
        self.records.append(
            {
                "season": season,
                "week": week,
                "game_id": weekly_games["game_id"].to_numpy()[keep],
                "centers": centers[keep],
                "spread": spread[keep],
                "actual": actual[keep],
                # The out-of-time residual/calibration sample is untouched by
                # this filter: it is a separate trailing-20% draw, never the
                # test games being scored, so push games in the TEST week have
                # no bearing on it.
                "residuals": np.asarray(model.residuals, dtype=np.float64).copy(),
            }
        )


def _week_bootstrap_probabilities(
    record: dict[str, Any],
    half_lives: dict[str, float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """(n_boot, n_games_week) probability draws for ecdf + every named half-life.

    Each draw resamples the week's OWN out-of-time residual sample with
    replacement, keeping the resample sorted back into its original
    chronological order (index 0 = oldest, index n-1 = most recent) so a
    duplicated draw still occupies a sensible recency position -- the natural
    generalization of a block bootstrap to an ordered series, analogous to how
    ``experiments.paired_feature_comparisons`` resamples whole weeks rather
    than shuffling games within them. The mean model's predicted centres are
    held fixed (docs/estimation_variance.md sec 7 scope note: this family
    holds the mean model fixed by construction; only the reader varies).
    """

    residuals = record["residuals"]
    n = len(residuals)
    thresholds = record["spread"] - record["centers"]
    weight_by_label = {
        label: recency_weights(n, half_life_games=hl) for label, hl in half_lives.items()
    }
    total_weight_by_label = {label: float(w.sum()) for label, w in weight_by_label.items()}

    m = len(thresholds)
    out: dict[str, np.ndarray] = {"ecdf": np.empty((n_boot, m), dtype=np.float64)}
    for label in half_lives:
        out[label] = np.empty((n_boot, m), dtype=np.float64)

    for draw in range(n_boot):
        idx = rng.integers(0, n, size=n)
        idx.sort(kind="stable")
        resampled = residuals[idx]

        # Unweighted ecdf reader: order-invariant, so a plain value sort suffices.
        sorted_vals = np.sort(resampled)
        pos = np.searchsorted(sorted_vals, thresholds, side="right")
        successes = (n - pos).astype(np.float64)
        out["ecdf"][draw] = (successes + 0.5) / (float(n) + 1.0)

        # Recency-weighted readers: weight[i] is tied to POSITION i in the
        # resampled (still-chronological) array, so sort (value, weight) pairs
        # together and read a suffix sum of weight for "> threshold".
        order = np.argsort(resampled, kind="stable")
        sorted_by_value = resampled[order]
        pos_weighted = np.searchsorted(sorted_by_value, thresholds, side="right")
        for label, weights in weight_by_label.items():
            sorted_weights = weights[order]
            suffix = np.concatenate([np.cumsum(sorted_weights[::-1])[::-1], [0.0]])
            success_weight = suffix[pos_weighted]
            out[label][draw] = (success_weight + 0.5) / (total_weight_by_label[label] + 1.0)

    return out


def build_residual_resample_draws(
    weekly: list[dict[str, Any]], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Generate the (n_boot, n_games) draw arrays ONCE; interval-building (any
    block convention) reuses them without regenerating -- the expensive part
    is the per-week resample-and-reread loop, not the final bootstrap-of-blocks
    pass, so this is computed a single time regardless of how many block
    conventions are read off it below."""

    half_lives = FOCAL_ARMS
    rng_master = np.random.default_rng(seed)

    baseline_chunks: list[np.ndarray] = []
    candidate_chunks: dict[str, list[np.ndarray]] = {label: [] for label in half_lives}
    actual_chunks: list[np.ndarray] = []
    season_block_chunks: list[np.ndarray] = []
    week_block_chunks: list[np.ndarray] = []

    for record in weekly:
        week_rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        draws = _week_bootstrap_probabilities(record, half_lives, n_boot=n_boot, rng=week_rng)
        baseline_chunks.append(draws["ecdf"])
        for label in half_lives:
            candidate_chunks[label].append(draws[label])
        actual_chunks.append(record["actual"])
        n_games_week = len(record["actual"])
        season_block_chunks.append(np.full(n_games_week, record["season"], dtype=np.int64))
        week_block_chunks.append(
            np.full(n_games_week, record["season"] * 100 + record["week"], dtype=np.int64)
        )

    return {
        "actual": np.concatenate(actual_chunks),
        "season_block_ids": np.concatenate(season_block_chunks),
        "week_block_ids": np.concatenate(week_block_chunks),
        "baseline_refit": np.concatenate(baseline_chunks, axis=1),
        "candidate_refit": {
            label: np.concatenate(chunks, axis=1) for label, chunks in candidate_chunks.items()
        },
    }


def honest_intervals_from_draws(
    draws: dict[str, Any], *, block: str, seed: int
) -> dict[str, PairedInterval]:
    block_ids = draws["season_block_ids"] if block == "season" else draws["week_block_ids"]
    out: dict[str, PairedInterval] = {}
    for label, candidate_refit in draws["candidate_refit"].items():
        out[label] = refit_aware_paired_interval(
            draws["actual"], draws["baseline_refit"], candidate_refit, block_ids, seed=seed
        )
    return out


def _validate_zero_resample_matches_point(weekly: list[dict[str, Any]]) -> dict[str, float]:
    """Sanity check: an IDENTITY resample (idx = arange(n)) must reproduce the
    exact production reader functions. Uses one week to keep this cheap."""

    from nfl_ats.residual_location import (
        recency_weighted_home_cover_probability,
        shrunk_home_cover_probability,  # noqa: F401  (import parity, unused here)
    )

    record = weekly[len(weekly) // 2]
    expected_ecdf = (
        np.sum(
            record["residuals"][np.newaxis, :]
            > (record["spread"] - record["centers"])[:, np.newaxis],
            axis=1,
        )
        + 0.5
    ) / (len(record["residuals"]) + 1.0)
    expected_hl200 = recency_weighted_home_cover_probability(
        record["residuals"], record["centers"], record["spread"], half_life_games=200.0
    )

    class _IdentityRNG:
        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            return np.arange(size, dtype=np.int64)

    draws = _week_bootstrap_probabilities(
        record,
        FOCAL_ARMS,
        n_boot=1,
        rng=_IdentityRNG(),  # type: ignore[arg-type]
    )
    return {
        "max_abs_diff_ecdf": float(np.max(np.abs(draws["ecdf"][0] - expected_ecdf))),
        "max_abs_diff_hl200": float(np.max(np.abs(draws["recency_hl200"][0] - expected_hl200))),
    }


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print(
        "=== Step 0: one CFB walk-forward pass, capturing clean-core weekly residual draws ===",
        flush=True,
    )
    cfb_features = pd.read_parquet(DEFAULT_CFB_FEATURES)
    cache = WeeklyCache()
    predictions, _location = rls.run_cfb(cfb_features, weekly_capture=cache.capture)
    print(
        f"predictions rows={len(predictions)} weekly cache entries={len(cache.records)}", flush=True
    )
    results["n_weekly_cache_entries"] = len(cache.records)
    results["n_clean_core_games"] = int(
        predictions.loc[predictions["evaluation_window"].eq("clean_core"), "game_id"].nunique()
    )

    print(
        "\n=== Sanity check: identity resample reproduces production readers exactly ===",
        flush=True,
    )
    validation = _validate_zero_resample_matches_point(cache.records)
    print(json.dumps(validation, indent=2), flush=True)
    results["identity_resample_validation"] = validation
    assert validation["max_abs_diff_ecdf"] < 1e-9
    assert validation["max_abs_diff_hl200"] < 1e-9

    print("\n=== Step 1: reproduce sec 4 (samples=2000, seed=20260818) ===", flush=True)
    reproduced = rls.paired_evidence(predictions, samples=2_000, seed=20260818)
    reproduced_rows = reproduced.loc[
        reproduced["evaluation_window"].eq("clean_core")
        & reproduced["block"].eq("season")
        & reproduced["metric"].eq("accuracy_improvement")
        & reproduced["candidate_feature_set"].isin(FOCAL_ARMS)
    ]
    results["reproduction"] = {}
    for _, row in reproduced_rows.iterrows():
        arm = str(row["candidate_feature_set"])
        rec = RECORDED[arm]
        results["reproduction"][arm] = {
            "estimate": float(row["estimate"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
            "paired_games": int(row["paired_games"]),
            "matches_recorded_estimate": bool(
                np.isclose(row["estimate"], rec["estimate"], atol=1e-6)
            ),
            "matches_recorded_bounds": bool(
                np.isclose(row["lower"], rec["lower"], atol=1e-6)
                and np.isclose(row["upper"], rec["upper"], atol=1e-6)
            ),
        }
    print(json.dumps(results["reproduction"], indent=2), flush=True)

    print(
        "\n=== Step 2: D3 check -- samples=20000 at the recorded seed, then 3 more seeds ===",
        flush=True,
    )
    d3_seeds = [20260818, 1, 2, 3]
    d3_rows: list[dict[str, Any]] = []
    for seed in d3_seeds:
        paired = rls.paired_evidence(predictions, samples=20_000, seed=seed)
        subset = paired.loc[
            paired["evaluation_window"].eq("clean_core")
            & paired["block"].eq("season")
            & paired["metric"].eq("accuracy_improvement")
            & paired["candidate_feature_set"].isin(FOCAL_ARMS)
        ]
        for _, row in subset.iterrows():
            d3_rows.append(
                {
                    "seed": seed,
                    "arm": str(row["candidate_feature_set"]),
                    "estimate": float(row["estimate"]),
                    "lower": float(row["lower"]),
                    "upper": float(row["upper"]),
                    "probability_positive": float(row["probability_positive"]),
                }
            )
    d3_frame = pd.DataFrame(d3_rows)
    results["d3_seed_sweep_samples20000"] = d3_rows
    for arm in FOCAL_ARMS:
        arm_rows = d3_frame.loc[d3_frame["arm"].eq(arm)]
        results.setdefault("d3_jitter_summary", {})[arm] = {
            "lower_sd": float(arm_rows["lower"].std(ddof=1)),
            "upper_sd": float(arm_rows["upper"].std(ddof=1)),
            "probability_positive_sd": float(arm_rows["probability_positive"].std(ddof=1)),
            "probability_positive_range": [
                float(arm_rows["probability_positive"].min()),
                float(arm_rows["probability_positive"].max()),
            ],
        }
    print(d3_frame.to_string(index=False), flush=True)
    print(json.dumps(results["d3_jitter_summary"], indent=2), flush=True)

    print(
        "\n=== Step 3: honest, mechanism-targeted interval "
        "(residual-resample, season-blocked primary) ===",
        flush=True,
    )
    draws = build_residual_resample_draws(cache.records, n_boot=2_000, seed=20260818)
    honest_season = honest_intervals_from_draws(draws, block="season", seed=20260818)
    honest_week = honest_intervals_from_draws(draws, block="week", seed=20260818)
    results["honest_interval"] = {
        "n_boot": 2_000,
        "season_blocked_primary": {arm: vars(iv) for arm, iv in honest_season.items()},
        "week_blocked_corroborating": {arm: vars(iv) for arm, iv in honest_week.items()},
    }
    for arm in FOCAL_ARMS:
        naive_lower = RECORDED[arm]["lower"]
        naive_upper = RECORDED[arm]["upper"]
        naive_width = naive_upper - naive_lower
        honest_width = honest_season[arm].upper - honest_season[arm].lower
        results["honest_interval"].setdefault("width_inflation_vs_recorded", {})[arm] = {
            "measured_residual_resample": honest_width / naive_width,
            "d2_blanket_low": D2_WIDTH_INFLATION_LOW,
            "d2_blanket_high": D2_WIDTH_INFLATION_HIGH,
        }
    print(json.dumps(results["honest_interval"], indent=2, default=str), flush=True)

    print("\n=== D2 blanket-widening comparator (crude, disclaimed) ===", flush=True)
    d2_comparator: dict[str, Any] = {}
    for arm in FOCAL_ARMS:
        rec = RECORDED[arm]
        center = rec["estimate"]
        for label, factor in (("low", D2_WIDTH_INFLATION_LOW), ("high", D2_WIDTH_INFLATION_HIGH)):
            half_width_lower = center - rec["lower"]
            half_width_upper = rec["upper"] - center
            d2_comparator.setdefault(arm, {})[label] = {
                "lower": center - half_width_lower * factor,
                "upper": center + half_width_upper * factor,
            }
    results["d2_blanket_widened_interval"] = d2_comparator
    print(json.dumps(d2_comparator, indent=2), flush=True)

    print("\n=== Step 4: D1 -- sign-only degeneracy check + closest valid contrast ===", flush=True)
    results["d1_degeneracy_check"] = d1_degeneracy_check(predictions)
    print(json.dumps(results["d1_degeneracy_check"], indent=2), flush=True)
    results["d1_closest_valid_contrast_vs_sign_only"] = d1_closest_valid_contrast(predictions)
    print(json.dumps(results["d1_closest_valid_contrast_vs_sign_only"], indent=2), flush=True)

    print("\n=== Step 5: power arithmetic (f, MDE80) ===", flush=True)
    results["power_arithmetic"] = power_arithmetic(predictions)
    print(json.dumps(results["power_arithmetic"], indent=2), flush=True)

    results["recorded_for_reference"] = RECORDED
    results["timing_seconds"] = time.time() - started
    out_path = OUT_DIR / "residual_location_reaudit.json"
    write_stamped_artifact(results, out_path)  # ENG-38
    print(f"\nWrote {out_path} in {results['timing_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
