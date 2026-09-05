"""Re-measurement of ``cfb_role_continuity`` against the 2026-08-18 instrument audit.

Research item: ``docs/revisit_list.md`` Tier 1, ``docs/cfb_role_features.md``.
``cfb_role_continuity`` is ``closed_negative`` in ``registry/rotation_registry.json``
on a measured -0.67 accuracy points (week-blocked [-1.33, +0.01]) on 8,933 clean-core
CFB games. Two instrument defects flagged in the same audit made a terminal
verdict on a small negative worth re-checking before it is trusted: D1, the
trailing residual-calibration step can attenuate or invert small planted
effects (per ``docs/purged_cv.md`` Sec.3 -- since disproved as a single-seed
artifact, see ``calibration_distortion.py``'s module docstring and
``docs/calibration_distortion.md``), and D2, reported intervals were once
claimed 17-58% too narrow because the block bootstrap never refits. That D2
headline is RETRACTED (``docs/estimation_variance.md`` Part II): it
double-counted a training-by-game interaction the game bootstrap already
carries, and the honest refit factor is 1.003x (one-sided 95% upper bound
1.099x), not 1.037x-1.575x. The real defect behind the original
under-coverage measurement was D4 (too few blocks), not D2. This script's own
step 5 below measures its OWN width-inflation ratio for this feature family
rather than borrowing the retracted headline, so that step's result is
unaffected by the retraction; it is cited here only as background for why a
re-check was warranted in the first place.

This script, in one pass over the frozen feature build (the expensive step is run
once):

1. Reproduces the recorded weekly-cadence -0.67 accuracy-point result bit-for-bit
   (``bootstrap_samples=2000, bootstrap_seed=20260817``, the exact recorded
   configuration).
2. Recomputes the same weekly-cadence comparison at ``samples=20000`` (the D3 fix
   already default in ``experiments.paired_feature_comparisons``, applied here
   explicitly since ``cfb_role_features.cfb_role_benchmark`` still hardcodes 2,000).
3. D1 cross-check: recomputes the identical paired comparison (same folds, same
   games, same weeks) under the sign-only estimator
   ``sign(predicted_margin - spread_line)`` -- the exact ablation ``docs/purged_cv.md``
   used to show the calibration step can distort small planted effects. If the
   full-pipeline and sign-only estimates agree closely, D1 is not live for this
   result.
4. Reports the paired power arithmetic (``f`` = fraction of games the two arms
   picked differently, ``MDE80 = 280*sqrt(f/n)``) for both estimators.
5. Honest interval: an annual-refit-cadence refit-aware bootstrap (reusing
   ``scripts/estvar_real_cfb_audit.py``'s machinery; N_BOOT=120, the same recipe
   documented in ``docs/estimation_variance.md``), specific to THIS feature family
   (candidate = frozen CFB columns + the six role-continuity columns; baseline =
   the frozen CFB columns alone) rather than borrowed from a different family. The
   measured width-inflation ratio from that run is then applied to the recorded
   WEEKLY-cadence interval's width, since a full weekly-cadence refit-aware
   bootstrap across ~14 seasons is not affordable this session (~15x the annual
   cost per ``docs/estimation_variance.md`` Sec.3's own cadence note).
6. Split-half reliability of the role-continuity TRAIT itself (not the accuracy
   effect): the same recipe ``docs/injury_value_lost.md`` Sec.3.1 used -- split
   each team-season into odd/even weeks, correlate the two halves' mean
   continuity, Spearman-Brown-correct to a full-length reliability, bootstrap a
   CI and ``probability_positive``. Run separately for dropback and carry
   continuity.

CFB only, per rotation-registry rule 8 (free, unlimited). Never touches
``registry/*.json`` or an NFL rotation window. Writes JSON to the scratchpad only.
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

import estvar_real_cfb_audit as estvar  # noqa: E402  (sibling script reuse)

import nfl_ats.cli as cli  # noqa: E402
from nfl_ats.cfb_benchmark import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.cfb_role_features import (  # noqa: E402
    CFB_ROLE_FEATURE_COLUMNS,
    ROLE_BENCHMARK_BASELINE_ARM,
    ROLE_BENCHMARK_CANDIDATE_ARM,
    attach_role_continuity,
    build_role_continuity,
    cfb_role_benchmark,
)
from nfl_ats.estimation_variance import (  # noqa: E402
    mde80,
    naive_block_bootstrap_interval,
    picks_differ_fraction,
    refit_aware_paired_interval,
)
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\d08576b1-0b16-4ebc-9ebc-f0bc62b943d3\scratchpad\role_continuity_remeasurement"
)
CFB_FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"

RECORDED = {
    "accuracy_improvement_estimate": -0.006716668532407925,
    "accuracy_improvement_week_lower": -0.013330806116650032,
    "accuracy_improvement_week_upper": 0.00011418170713983402,
    "accuracy_improvement_season_lower": -0.016801681550700935,
    "accuracy_improvement_season_upper": 0.004016198843631899,
    "artifact": "artifacts/cfb_role_experiments/20260817T110541Z",
    "paired_games": 8933,
}

N_BOOT = 120


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_with_role_columns, raw_continuity_frame)."""

    actions, team_games, team_ids = cli._load_cfb_role_inputs(CFB_FEATURES_PATH)
    canonical_games = pd.read_parquet(CFB_FEATURES_PATH)
    continuity = build_role_continuity(actions, team_games)
    features = attach_role_continuity(canonical_games, continuity, team_ids)
    return features, continuity


def extract_clean(predictions: pd.DataFrame) -> pd.DataFrame:
    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].isin(("market_residual", "market_residual_roles"))
    ].copy()
    clean["feature_set"] = clean["method"].map(
        {
            "market_residual": ROLE_BENCHMARK_BASELINE_ARM,
            "market_residual_roles": ROLE_BENCHMARK_CANDIDATE_ARM,
        }
    )
    return clean


def sign_only_frame(clean: pd.DataFrame) -> pd.DataFrame:
    signed = clean.copy()
    signed["home_cover_probability"] = (signed["predicted_margin"] > signed["spread_line"]).astype(
        float
    )
    return signed


def paired_f_and_mde(clean: pd.DataFrame) -> dict[str, float]:
    pivot = clean.pivot(index="game_id", columns="feature_set", values="home_cover_probability")
    baseline = pivot[ROLE_BENCHMARK_BASELINE_ARM].to_numpy(dtype=float)
    candidate = pivot[ROLE_BENCHMARK_CANDIDATE_ARM].to_numpy(dtype=float)
    n = len(pivot)
    f = picks_differ_fraction(baseline, candidate)
    return {"n": n, "f": f, "mde80": mde80(f, n)}


def _load_completed_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    return completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def honest_interval_report(features: pd.DataFrame) -> dict[str, Any]:
    completed = _load_completed_frame(features)
    candidate_columns = (*CFB_MODEL_FEATURE_COLUMNS, *CFB_ROLE_FEATURE_COLUMNS)
    fits = estvar.fit_seasons(
        completed,
        baseline_columns=CFB_MODEL_FEATURE_COLUMNS,
        candidate_columns=candidate_columns,
        n_boot=N_BOOT,
        seed=42,
    )
    arrays = estvar.pooled_arrays(fits)
    n_games = len(arrays["actual"])
    f_annual = picks_differ_fraction(arrays["baseline_prob"], arrays["candidate_prob"])
    naive = naive_block_bootstrap_interval(
        arrays["actual"],
        arrays["baseline_prob"],
        arrays["candidate_prob"],
        arrays["block_ids"],
        samples=20_000,
        seed=20260818,
    )
    honest = refit_aware_paired_interval(
        arrays["actual"],
        arrays["baseline_refit_prob"],
        arrays["candidate_refit_prob"],
        arrays["block_ids"],
        seed=20260818,
    )
    flip_rate = estvar.candidate_flip_rate(fits)
    naive_width = naive.upper - naive.lower
    honest_width = honest.upper - honest.lower
    return {
        "n_games_annual_cadence": n_games,
        "f_annual_cadence": f_annual,
        "candidate_flip_rate": flip_rate,
        "naive_annual": vars(naive),
        "honest_annual": vars(honest),
        "width_inflation_ratio": honest_width / naive_width,
    }


def split_half_reliability(
    continuity: pd.DataFrame, action_type: str, *, seed: int, n_boot: int = 4000
) -> dict[str, Any]:
    subset = continuity.loc[continuity["action_type"].eq(action_type)].copy()
    subset["half"] = np.where(subset["week"] % 2 == 0, "even", "odd")
    means = subset.groupby(["team", "season", "half"])["continuity"].mean().unstack("half")
    counts = subset.groupby(["team", "season", "half"]).size().unstack("half")
    valid = (counts.get("odd", 0) >= 2) & (counts.get("even", 0) >= 2)
    means = means.dropna()
    valid = valid.reindex(means.index).fillna(False)
    means = means.loc[valid]

    odd_vals = means["odd"].to_numpy(dtype=float)
    even_vals = means["even"].to_numpy(dtype=float)
    n = len(means)
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    from scipy.stats import spearmanr

    rho = float(spearmanr(odd_vals, even_vals).correlation)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sel = rng.integers(0, n, size=n)
        boots[i] = np.corrcoef(odd_vals[sel], even_vals[sel])[0, 1]
    sb = (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")
    return {
        "action_type": action_type,
        "n_team_seasons": n,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_rho": rho,
        "spearman_brown_full_length_reliability": sb,
        "probability_positive": float(np.mean(boots > 0.0)),
    }


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("=== Step 1: build role features (one pass) ===", flush=True)
    features, continuity = build_features()
    print(f"features rows={len(features)} continuity rows={len(continuity)}", flush=True)

    print("\n=== Step 2: reproduce recorded run (samples=2000, seed=20260817) ===", flush=True)
    result = cfb_role_benchmark(features, bootstrap_samples=2_000, bootstrap_seed=20260817)
    reproduced = result.paired.loc[
        result.paired["metric"].eq("accuracy_improvement") & result.paired["block"].eq("week")
    ].iloc[0]
    results["reproduction"] = {
        "estimate": float(reproduced["estimate"]),
        "lower": float(reproduced["lower"]),
        "upper": float(reproduced["upper"]),
        "paired_games": int(reproduced["paired_games"]),
        "matches_recorded_estimate": bool(
            np.isclose(
                reproduced["estimate"],
                RECORDED["accuracy_improvement_estimate"],
                atol=1e-9,
            )
        ),
        "matches_recorded_bounds": bool(
            np.isclose(reproduced["lower"], RECORDED["accuracy_improvement_week_lower"], atol=1e-9)
            and np.isclose(
                reproduced["upper"], RECORDED["accuracy_improvement_week_upper"], atol=1e-9
            )
        ),
    }
    print(json.dumps(results["reproduction"], indent=2), flush=True)

    clean = extract_clean(result.predictions)
    print(f"clean-core paired games: {clean['game_id'].nunique()}", flush=True)

    print("\n=== Step 3: D3-fixed naive interval (samples=20000) ===", flush=True)
    d3_week = paired_feature_comparisons(
        clean,
        baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
        samples=20_000,
        block="week",
        seed=20260817,
    )
    d3_season = paired_feature_comparisons(
        clean,
        baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
        samples=20_000,
        block="season",
        seed=20260817,
    )
    results["full_pipeline_samples20000"] = {
        "week": d3_week.loc[d3_week["metric"].eq("accuracy_improvement")].iloc[0].to_dict(),
        "season": d3_season.loc[d3_season["metric"].eq("accuracy_improvement")].iloc[0].to_dict(),
    }
    print(json.dumps(results["full_pipeline_samples20000"], indent=2, default=str), flush=True)

    print("\n=== Step 4: D1 cross-check -- sign-only estimator, identical folds ===", flush=True)
    signed = sign_only_frame(clean)
    signonly_week = paired_feature_comparisons(
        signed,
        baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
        samples=20_000,
        block="week",
        seed=20260817,
    )
    signonly_season = paired_feature_comparisons(
        signed,
        baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
        samples=20_000,
        block="season",
        seed=20260817,
    )
    results["sign_only_samples20000"] = {
        "week": signonly_week.loc[signonly_week["metric"].eq("accuracy_improvement")]
        .iloc[0]
        .to_dict(),
        "season": signonly_season.loc[signonly_season["metric"].eq("accuracy_improvement")]
        .iloc[0]
        .to_dict(),
    }
    print(json.dumps(results["sign_only_samples20000"], indent=2, default=str), flush=True)

    print("\n=== Step 5: paired power arithmetic (f, MDE80) ===", flush=True)
    results["power_full_pipeline"] = paired_f_and_mde(clean)
    results["power_sign_only"] = paired_f_and_mde(signed)
    print(
        json.dumps(
            {
                "full_pipeline": results["power_full_pipeline"],
                "sign_only": results["power_sign_only"],
            },
            indent=2,
        ),
        flush=True,
    )

    print("\n=== Step 6: honest refit-aware interval (annual cadence, N_BOOT=120) ===", flush=True)
    results["honest_interval"] = honest_interval_report(features)
    print(json.dumps(results["honest_interval"], indent=2, default=str), flush=True)

    print("\n=== Step 7: split-half reliability of the role-continuity trait ===", flush=True)
    results["split_half_reliability"] = {
        "dropback": split_half_reliability(continuity, "dropback", seed=1),
        "carry": split_half_reliability(continuity, "carry", seed=2),
    }
    print(json.dumps(results["split_half_reliability"], indent=2, default=str), flush=True)

    results["recorded_for_reference"] = RECORDED
    results["timing_seconds"] = time.time() - started
    out_path = OUT_DIR / "role_continuity_remeasurement.json"
    write_stamped_artifact(results, out_path)  # ENG-38
    print(f"\nWrote {out_path} in {results['timing_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
