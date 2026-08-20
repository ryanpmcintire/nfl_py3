"""MEASURE-ONLY: does adding ``surface_switch_flag`` to the active NFL model
(``weak_stack_surface`` vs the production ``weak_stack``) beat the incumbent
at the grade the pool settles on -- the opener?

This script produces numbers only. It does not write the registry, promote
a model, touch ``artifacts/active_ats_model.json``, ``cli.py``, or the
published card. Recording the result to ``registry/weak_signals.json`` is a
SEPARATE step (``scripts/surface_profile_opener_record.py``).

Background (read this session, all three docs):
- ``artifacts/active_ats_model.json``: active production model is
  ``method=market_residual``, ``feature_profile=weak_stack``, ``regressor=
  ridge``, ``ridge_alpha=10.0``.
- ``src/nfl_ats/margin.py`` (``MARGIN_FEATURE_PROFILES``): a prior session
  added ``weak_stack_surface`` -- ``weak_stack`` plus exactly one column,
  ``surface_switch_flag`` -- and built
  ``data/processed/game_features_weak_stack_surface.parquet`` as a pure,
  additive merge-by-``game_id`` enrichment of the existing weak_stack table
  (measured: 274 weak_stack columns are byte-identical between the two
  files; the surface file adds exactly one more, 275 total, same 4,902
  rows). Per ``docs/surface_switch_feature_arm.md``'s own precedent, BOTH
  arms of this comparison are fit from that single enriched file --
  weak_stack's own columns are unaffected by which of the two files backs
  it, so the baseline arm is unaffected by this substitution.
- ``docs/surface_switch_feature_arm.md`` already ran this exact pairing at
  the CLOSE grade via ``feature_arm`` (mined/scored, not opener --
  ``feature_arm`` has no opener implementation, read from that doc): +0.2410
  accuracy points, week-blocked 95% [-1.1594, +1.5988], P+ 0.6181
  (registry entry ``surface_switch_feature_arm``). This script is the
  OPENER-graded look at the identical pairing -- the grade the project's
  primary goal (beat the Splash Sports opener line) actually cares about,
  per AGENTS.md "grade the decision at the OPENER".

Methodology, adapted from ``scripts/ridge_alpha_promotion_eval.py``'s
``run_opener_grade`` path (the pattern that graded the ridge-alpha swap at
the opener; that swap was refused at -1.397 pts, P+ 0.0504) -- itself
adapted from the MOD-07 precedent (52.83% vs 52.50%,
``docs/opener_evaluation.md``). Two deliberate deviations from the
``ridge_alpha_promotion_eval.py`` copy, both declared:

1. The two arms differ by ``feature_profile`` (``weak_stack`` vs
   ``weak_stack_surface``), not ``ridge_alpha``; both hold ``ridge_alpha=10.0``
   (the frozen incumbent value) and ``regressor="ridge"`` fixed.
2. Rather than re-copying ``clv.opener_pick_evaluation`` by hand (as the
   ridge_alpha script did, before the 2026-08-19 addendum), this script
   calls the CURRENT, already-updated ``clv.opener_pick_evaluation``
   directly. That function now natively computes BOTH pick rules
   additively on the same scored frame -- the predeclared **sign rule**
   (``residual_at_open > 0``, the historical protocol,
   ``docs/opener_evaluation.md``) and production's actual **probability
   rule** (``home_cover_probability_at_open >= 0.5``, what ``pool.py``/
   ``backtest.py`` have always played, per that doc's 2026-08-19 addendum).
   Calling the real function instead of a stale hand-copy is more faithful
   to what the task asked ("use BOTH pick rules if the machinery reports
   them"), not a new evaluation design of its own.

Endpoints reported, in the project's declared priority order:
  1. PRIMARY -- paired forced-pick accuracy delta at the OPENER grade under
     the PRODUCTION probability rule, on the paired Tuesday-opener
     population, week-blocked bootstrap (also season-blocked secondary),
     probability_positive.
  2. Also reported at the opener under the sign rule (the historical
     protocol number), for comparability with prior opener-grade runs.
  3. Close-graded accuracy delta on the same paired games, both rules
     (secondary; per AGENTS.md "grade the decision at the OPENER... a
     close-graded number may never veto a play" -- reported, never a gate).
  4. Each arm's absolute opener/close accuracy under both rules (sanity
     anchor: incumbent weak_stack should land near the tracked 53.36%
     production-rule / 52.83% sign-rule opener numbers,
     ``docs/opener_evaluation.md``'s 2026-08-19 addendum).
  5. Brier / log-loss direction (paired, week-blocked; probabilities are
     rule-agnostic, continuous).
  6. Pick-flip count under each rule, and where flips concentrate by
     |opener line| bucket.
  7. ``surface_switch_flag`` prevalence on the paired opener archive
     population itself -- needed to reason about double-counting against
     the live ``surface_switch_tilt_overlay`` pick-level challenger, which
     already flips a subset of these same games post-prediction.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/surface_profile_opener_eval.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import opener_evaluation_metrics, opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.margin import MarginFeatureProfile

REPO = Path(__file__).resolve().parents[1]

# --- Frozen configuration -----------------------------------------------
# Both arms hold ridge_alpha=10.0 and regressor="ridge" fixed at the
# incumbent's own values (artifacts/active_ats_model.json, read this
# session) -- only feature_profile differs. This isolates the
# surface_switch_flag column's effect exactly, matching
# docs/surface_switch_feature_arm.md's own close-graded predeclaration.
BASELINE_PROFILE: MarginFeatureProfile = "weak_stack"
CANDIDATE_PROFILE: MarginFeatureProfile = "weak_stack_surface"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

# Matches docs/opener_evaluation.md's predeclared seed, reused by
# scripts/ridge_alpha_promotion_eval.py for every opener-grade confirmation
# run since -- keeping this run comparable to those, not a fresh choice.
OPENER_BOOTSTRAP_SAMPLES = 20_000
OPENER_BOOTSTRAP_SEED = 20260817

LINE_BUCKET_EDGES: tuple[float, ...] = (0.0, 3.0, 7.0, 10.0, float("inf"))
LINE_BUCKET_LABELS: tuple[str, ...] = ("[0,3)", "[3,7)", "[7,10)", "[10,inf)")


def _config(profile: MarginFeatureProfile) -> dict[str, Any]:
    return {"feature_profile": profile, "regressor": REGRESSOR, "ridge_alpha": RIDGE_ALPHA}


def paired_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """One row per game scored by BOTH arms (mirrors mod07_weak_stack.paired_frame /
    ridge_alpha_promotion_eval.paired_frame, extended with the probability-rule
    columns clv.opener_pick_evaluation now reports natively)."""

    keep = [
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "correct_at_open",
        "correct_at_close",
        "correct_at_open_probability_rule",
        "correct_at_close_probability_rule",
        "pick_home_at_open",
        "pick_home_at_open_probability_rule",
        "home_cover_probability_at_open",
        "home_cover_probability_at_close",
        "margin_vs_open",
        "margin_vs_close",
    ]
    left = baseline[keep].rename(
        columns={
            "correct_at_open": "baseline_correct_open",
            "correct_at_close": "baseline_correct_close",
            "correct_at_open_probability_rule": "baseline_correct_open_pr",
            "correct_at_close_probability_rule": "baseline_correct_close_pr",
            "pick_home_at_open": "baseline_pick_home",
            "pick_home_at_open_probability_rule": "baseline_pick_home_pr",
            "home_cover_probability_at_open": "baseline_prob_open",
            "home_cover_probability_at_close": "baseline_prob_close",
        }
    )
    right = candidate[
        [
            "game_id",
            "correct_at_open",
            "correct_at_close",
            "correct_at_open_probability_rule",
            "correct_at_close_probability_rule",
            "pick_home_at_open",
            "pick_home_at_open_probability_rule",
            "home_cover_probability_at_open",
            "home_cover_probability_at_close",
        ]
    ].rename(
        columns={
            "correct_at_open": "candidate_correct_open",
            "correct_at_close": "candidate_correct_close",
            "correct_at_open_probability_rule": "candidate_correct_open_pr",
            "correct_at_close_probability_rule": "candidate_correct_close_pr",
            "pick_home_at_open": "candidate_pick_home",
            "pick_home_at_open_probability_rule": "candidate_pick_home_pr",
            "home_cover_probability_at_open": "candidate_prob_open",
            "home_cover_probability_at_close": "candidate_prob_close",
        }
    )
    merged = left.merge(right, on="game_id", how="inner")
    for column in (
        "baseline_correct_open",
        "baseline_correct_close",
        "baseline_correct_open_pr",
        "baseline_correct_close_pr",
        "candidate_correct_open",
        "candidate_correct_close",
        "candidate_correct_open_pr",
        "candidate_correct_close_pr",
    ):
        merged[column] = merged[column].astype(float)
    # Binary actual outcome for Brier/log-loss, NaN on a push (excluded),
    # matching nfl_ats.outcomes.summarize_predictions' convention and
    # ridge_alpha_promotion_eval.py's identical construction.
    merged["actual_home_cover_open"] = np.where(
        merged["margin_vs_open"] > 0.0, 1.0, np.where(merged["margin_vs_open"] < 0.0, 0.0, np.nan)
    )
    merged["actual_home_cover_close"] = np.where(
        merged["margin_vs_close"] > 0.0,
        1.0,
        np.where(merged["margin_vs_close"] < 0.0, 0.0, np.nan),
    )
    return merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _accuracy_metric_fn(open_col: str, base_col: str) -> Any:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        return {"candidate_minus_baseline": float(frame[open_col].mean() - frame[base_col].mean())}

    return metric_fn


def _brier_metric_fn(prob_col: str, base_prob_col: str, actual_col: str) -> Any:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        valid = frame.dropna(subset=[actual_col])
        actual = valid[actual_col].to_numpy(dtype=float)
        cand = valid[prob_col].to_numpy(dtype=float)
        base = valid[base_prob_col].to_numpy(dtype=float)
        cand_brier = np.mean(np.square(cand - actual))
        base_brier = np.mean(np.square(base - actual))
        clip = lambda p: np.clip(p, 1e-15, 1.0 - 1e-15)  # noqa: E731
        cand_ll = np.mean(-(actual * np.log(clip(cand)) + (1 - actual) * np.log(1.0 - clip(cand))))
        base_ll = np.mean(-(actual * np.log(clip(base)) + (1 - actual) * np.log(1.0 - clip(base))))
        return {
            "brier_improvement": float(base_brier - cand_brier),
            "log_loss_improvement": float(base_ll - cand_ll),
        }

    return metric_fn


def line_flip_buckets(paired: pd.DataFrame, pick_col_suffix: str = "") -> pd.DataFrame:
    base_col = f"baseline_pick_home{pick_col_suffix}"
    cand_col = f"candidate_pick_home{pick_col_suffix}"
    correct_base_col = f"baseline_correct_open{'_pr' if pick_col_suffix == '_pr' else ''}"
    correct_cand_col = f"candidate_correct_open{'_pr' if pick_col_suffix == '_pr' else ''}"
    flips = paired.loc[paired[base_col].ne(paired[cand_col])].copy()
    abs_line = flips["tue_open_home_spread"].abs()
    flips["bucket"] = pd.cut(
        abs_line, bins=LINE_BUCKET_EDGES, labels=LINE_BUCKET_LABELS, right=False
    )
    rows = []
    for label in LINE_BUCKET_LABELS:
        subset = flips.loc[flips["bucket"].eq(label)]
        rows.append(
            {
                "bucket": label,
                "flips": len(subset),
                "baseline_correct_rate": (
                    float(subset[correct_base_col].mean()) if len(subset) else float("nan")
                ),
                "candidate_correct_rate": (
                    float(subset[correct_cand_col].mean()) if len(subset) else float("nan")
                ),
            }
        )
    total_row = {
        "bucket": "ALL",
        "flips": len(flips),
        "baseline_correct_rate": (
            float(flips[correct_base_col].mean()) if len(flips) else float("nan")
        ),
        "candidate_correct_rate": (
            float(flips[correct_cand_col].mean()) if len(flips) else float("nan")
        ),
    }
    return pd.DataFrame([*rows, total_row])


def run_opener_grade(
    *,
    features: pd.DataFrame,
    market_root: Path,
    min_train_games: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    baseline = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=_config(BASELINE_PROFILE),
        min_train_games=min_train_games,
    )
    t1 = time.perf_counter()
    candidate = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=_config(CANDIDATE_PROFILE),
        min_train_games=min_train_games,
    )
    t2 = time.perf_counter()

    baseline_metrics = opener_evaluation_metrics(baseline)
    candidate_metrics = opener_evaluation_metrics(candidate)

    paired = paired_frame(baseline, candidate)

    def _bootstrap(open_col: str, base_col: str, block: str) -> dict[str, float]:
        return (
            week_blocked_bootstrap(
                paired,
                _accuracy_metric_fn(open_col, base_col),
                block=block,
                samples=samples,
                seed=seed,
            )
            .iloc[0]
            .to_dict()
        )

    open_delta_sign = {
        "week_blocked": _bootstrap("candidate_correct_open", "baseline_correct_open", "week"),
        "season_blocked": _bootstrap("candidate_correct_open", "baseline_correct_open", "season"),
    }
    open_delta_pr = {
        "week_blocked": _bootstrap("candidate_correct_open_pr", "baseline_correct_open_pr", "week"),
        "season_blocked": _bootstrap(
            "candidate_correct_open_pr", "baseline_correct_open_pr", "season"
        ),
    }
    close_delta_sign = {
        "week_blocked": _bootstrap("candidate_correct_close", "baseline_correct_close", "week"),
        "season_blocked": _bootstrap("candidate_correct_close", "baseline_correct_close", "season"),
    }
    close_delta_pr = {
        "week_blocked": _bootstrap(
            "candidate_correct_close_pr", "baseline_correct_close_pr", "week"
        ),
        "season_blocked": _bootstrap(
            "candidate_correct_close_pr", "baseline_correct_close_pr", "season"
        ),
    }

    brier_open_week = week_blocked_bootstrap(
        paired,
        _brier_metric_fn("candidate_prob_open", "baseline_prob_open", "actual_home_cover_open"),
        block="week",
        samples=samples,
        seed=seed,
    )
    brier_open_season = week_blocked_bootstrap(
        paired,
        _brier_metric_fn("candidate_prob_open", "baseline_prob_open", "actual_home_cover_open"),
        block="season",
        samples=samples,
        seed=seed,
    )

    disagreements_sign = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])]
    disagreements_pr = paired.loc[
        paired["baseline_pick_home_pr"].ne(paired["candidate_pick_home_pr"])
    ]
    flip_buckets_sign = line_flip_buckets(paired, "")
    flip_buckets_pr = line_flip_buckets(paired, "_pr")

    # surface_switch_flag prevalence on the paired opener archive population
    # itself, needed to reason about the surface_switch_tilt_overlay
    # double-counting interaction (task step 5).
    flag_lookup = features[["game_id", "surface_switch_flag"]].drop_duplicates("game_id")
    flagged = paired.merge(flag_lookup, on="game_id", how="left")
    n_flag_missing = int(flagged["surface_switch_flag"].isna().sum())
    n_flagged = int((flagged["surface_switch_flag"] == 1.0).sum())

    return {
        "timing": {
            "baseline_seconds": t1 - t0,
            "candidate_seconds": t2 - t1,
            "total_seconds": t2 - t0,
        },
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "seasons": sorted(int(s) for s in paired["season"].unique()),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "open_accuracy_delta_sign_rule": open_delta_sign,
        "open_accuracy_delta_probability_rule": open_delta_pr,
        "close_accuracy_delta_sign_rule": close_delta_sign,
        "close_accuracy_delta_probability_rule": close_delta_pr,
        "brier_delta_open": {
            "week_blocked": brier_open_week.to_dict(orient="records"),
            "season_blocked": brier_open_season.to_dict(orient="records"),
        },
        "picks_disagreeing_sign_rule": len(disagreements_sign),
        "picks_disagreeing_probability_rule": len(disagreements_pr),
        "flip_buckets_sign_rule": flip_buckets_sign.to_dict(orient="records"),
        "flip_buckets_probability_rule": flip_buckets_pr.to_dict(orient="records"),
        "surface_switch_flag_games_in_archive": n_flagged,
        "surface_switch_flag_missing_in_archive": n_flag_missing,
        "surface_switch_flag_rate_in_archive": (
            n_flagged / len(paired) if len(paired) else float("nan")
        ),
        "paired_frame": paired,
        "baseline_frame": baseline,
        "candidate_frame": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO / "data/processed/game_features_weak_stack_surface.parquet",
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--opener-samples", type=int, default=OPENER_BOOTSTRAP_SAMPLES)
    parser.add_argument("--opener-seed", type=int, default=OPENER_BOOTSTRAP_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    required_cols = {"surface_switch_flag"}
    missing = required_cols.difference(features.columns)
    if missing:
        raise SystemExit(
            f"--features table is missing {sorted(missing)}; use "
            "game_features_weak_stack_surface.parquet, not the plain weak_stack table."
        )

    print(
        f"=== Opener grade (primary): {BASELINE_PROFILE} vs {CANDIDATE_PROFILE}, "
        f"both ridge_alpha={RIDGE_ALPHA:g} ==="
    )
    opener_result = run_opener_grade(
        features=features,
        market_root=args.market_root,
        min_train_games=args.min_train_games,
        samples=args.opener_samples,
        seed=args.opener_seed,
    )
    opener_report = {k: v for k, v in opener_result.items() if not k.endswith("_frame")}
    print(json.dumps(opener_report, indent=2, default=str))

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "artifacts" / "surface_profile_opener_eval" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE_PROFILE,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_samples": args.opener_samples,
        "bootstrap_seed": args.opener_seed,
        "features_path": str(args.features),
        "market_root": str(args.market_root),
        "min_train_games": args.min_train_games,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **opener_report,
    }
    (out_dir / "opener_summary.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    opener_result["paired_frame"].to_parquet(out_dir / "opener_paired.parquet")
    opener_result["baseline_frame"].to_parquet(out_dir / "opener_baseline.parquet")
    opener_result["candidate_frame"].to_parquet(out_dir / "opener_candidate.parquet")

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
