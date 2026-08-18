"""MEASURE-ONLY: does the CFB-localized ridge_alpha optimum beat the ACTIVE
NFL model's frozen alpha=10 at the grade the pool settles on (the opener)?

This script produces numbers only. It does not write the registry, activate
a model, or spend a rotation window (per the task that generated it).

Background (see docs/ridge_alpha.md, read this session): the incumbent
``ridge_alpha=10.0`` on the active ``market_residual`` / ``weak_stack`` /
``ridge`` model has never been derived. A predeclared two-stage CFB screen
(13-point coarse grid, then a 6-point, NOT-blind refinement grid localizing
the coarse curve's turn) found accuracy flat across seven orders of
magnitude (no promotable value) but Brier/log-loss resolvably better in a
broad 300-10,000 band, walk-forward optimum near alpha=2,000-2,500. Section 4
of that document explicitly nominates a single candidate for a future NFL
confirmation look: **ridge_alpha = 2,000**, "the walk-forward Brier optimum
measured here." This script is that one confirmation look -- ONE look at a
value already localized by two predeclared grids on a different (CFB)
instrument, not a search.

Methodology, reproduced from the MOD-07 precedent
---------------------------------------------------
The 52.83% vs 52.50% MOD-07 promotion comparison (AGENTS.md "Grade the
decision at the OPENER", docs/opener_evaluation.md "Results -- weak_stack
profile") was produced by ``nfl_ats.clv.opener_pick_evaluation``: one
weekly-refit ``market_residual`` model per (season, week), trained on
completed games strictly before that week's first kickoff, evaluated twice
-- once with ``spread_line`` overridden to the Tuesday opener, once to the
close -- on every archived game with a paired ``tue_open`` consensus and a
resolvable close (2020-2025 historical snapshot archive, ~1,537 paired
games). ``scripts/mod07_weak_stack.py`` is the paired-arms wrapper around
that machinery: two ``opener_pick_evaluation`` calls (baseline config,
candidate config), paired by ``game_id``, ``candidate_minus_baseline``
accuracy scored with ``clv.week_blocked_bootstrap``.

This script follows that exact pattern with two differences, both declared:

1. It does not route through ``mod07_weak_stack.py``'s ``arm()`` helper,
   which restricts each arm to a specific rotation family's confirmation
   window (``rotation.confirmation_split``). ``ridge_alpha_global`` has no
   assigned window -- docs/ridge_alpha.md section 4 explicitly says not to
   draw one on the accuracy hypothesis alone. So both arms here run over the
   full paired opener archive directly (matching how the 52.83%/52.50%
   headline itself was produced -- that run also was not window-restricted;
   see docs/opener_evaluation.md's 1,537-game count).
2. ``opener_pick_evaluation`` discards ``MarginModel.predict``'s
   ``home_cover_probability`` column, keeping only the residual sign.
   ``evaluate_arm`` below is a line-for-line copy of
   ``opener_pick_evaluation`` (same helper calls: ``build_pairing_table``,
   ``close_reference_table``, ``fit_margin_model``, ``regular_season_rows``,
   ``pick_correct``) that additionally keeps ``home_cover_probability`` at
   both lines, so Brier/log-loss can be scored without editing clv.py. No
   other logic differs; this is not a new evaluation design.

Endpoints reported, in the project's declared priority order:
  1. PRIMARY -- paired forced-pick accuracy delta at the OPENER grade on the
     paired Tuesday-opener population, week-blocked bootstrap (also
     season-blocked), probability_positive.
  2. Close-graded accuracy on the same paired games (secondary; per AGENTS.md
     "grade the decision at the OPENER... a close-graded number may never
     veto a play" -- reported, never a gate).
  3. Brier / log-loss direction (paired, week- and season-blocked).
  4. Pick-flip count, and where flips concentrate by |opener line| bucket.
  5. The same accuracy/Brier/log-loss comparison at the nflverse_spread grade
     (``nfl_ats.outcomes.walk_forward_outcomes``, the machinery behind
     ``artifacts/active_ats_model.json``'s ``historical_evaluation``) across
     the full 2018+ history, for completeness.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/ridge_alpha_promotion_eval.py
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

from nfl_ats.clv import (
    CLOSE_LABEL_PRIORITY,
    build_pairing_table,
    close_reference_table,
    opener_evaluation_metrics,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginFeatureProfile, fit_margin_model, margin_feature_columns
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.outcomes import summarize_outcome_method, walk_forward_outcomes

REPO = Path(__file__).resolve().parents[1]

# --- Frozen configuration -----------------------------------------------
# Active production model, per artifacts/active_ats_model.json (read this
# session): feature_profile="weak_stack", regressor="ridge", ridge_alpha=10.0,
# target="market_residual", calibration="none".
PROFILE: MarginFeatureProfile = "weak_stack"
REGRESSOR = "ridge"
BASELINE_ALPHA = 10.0
# docs/ridge_alpha.md section 4's named candidate -- not chosen by this
# script; read from the document, not re-derived.
CANDIDATE_ALPHA = 2_000.0

OPENER_BOOTSTRAP_SAMPLES = 20_000
OPENER_BOOTSTRAP_SEED = 20260817  # matches docs/opener_evaluation.md's predeclared seed
NFLVERSE_BOOTSTRAP_SAMPLES = 2_000  # matches the ridge_alpha_screen.py CFB precedent
NFLVERSE_BOOTSTRAP_SEED = 20260818

LINE_BUCKET_EDGES: tuple[float, ...] = (0.0, 3.0, 7.0, 10.0, float("inf"))
LINE_BUCKET_LABELS: tuple[str, ...] = ("[0,3)", "[3,7)", "[7,10)", "[10,inf)")


def _config(ridge_alpha: float) -> dict[str, Any]:
    return {
        "feature_profile": PROFILE,
        "regressor": REGRESSOR,
        "ridge_alpha": ridge_alpha,
        "target": "market_residual",
    }


# ---------------------------------------------------------------------------
# Opener-grade evaluation with cover probabilities retained
# ---------------------------------------------------------------------------


def evaluate_arm(
    market_root: Path,
    features: pd.DataFrame,
    *,
    config: dict[str, Any],
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Line-for-line copy of ``clv.opener_pick_evaluation`` that also keeps
    ``home_cover_probability`` at both lines (needed for Brier/log-loss).

    Every helper call, the pairing logic, the weekly-refit loop, and the
    forced-pick settlement are identical to the frozen recipe; see the
    module docstring for why this could not simply call the existing
    function unmodified.
    """

    profile: MarginFeatureProfile = config["feature_profile"]
    feature_columns = margin_feature_columns("market_residual", profile)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        *feature_columns,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Opener evaluation is missing columns: {', '.join(missing)}")
    features = regular_season_rows(features)

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    if pairing.empty:
        raise ValueError(f"No {HISTORICAL_CAPTURE_KIND!r} snapshots with decision quotes")
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread", "spread_books"]
    ].rename(columns={"home_spread": "tue_open_home_spread", "spread_books": "opener_books"})
    paired = tue_open.merge(close, on="game_id", how="inner")

    outcomes = features[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed games have both a Tuesday opener and a close")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    scored_weeks: list[pd.DataFrame] = []
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=config["regressor"],
            feature_profile=profile,
            ridge_alpha=config["ridge_alpha"],
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        at_close = scoring.copy()
        at_close["spread_line"] = at_close["close_home_spread"]
        open_pred = model.predict(at_open)
        close_pred = model.predict(at_close)
        scored = scoring[["game_id"]].copy()
        scored["season"] = int(str(season))
        scored["week"] = int(str(week))
        scored["residual_at_open"] = open_pred["predicted_market_residual"].to_numpy()
        scored["residual_at_close"] = close_pred["predicted_market_residual"].to_numpy()
        scored["prob_at_open"] = open_pred["home_cover_probability"].to_numpy()
        scored["prob_at_close"] = close_pred["home_cover_probability"].to_numpy()
        scored_weeks.append(scored)
    if not scored_weeks:
        raise ValueError("No paired week had at least min_train_games completed training rows")
    residuals = pd.concat(scored_weeks, ignore_index=True)

    result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["margin_vs_close"] = result["result"] - result["close_home_spread"]
    result["open_move"] = result["close_home_spread"] - result["tue_open_home_spread"]

    result["pick_home_at_open"] = result["residual_at_open"].gt(0.0)
    result["pick_home_at_close"] = result["residual_at_close"].gt(0.0)
    result["correct_at_open"] = pick_correct(result["pick_home_at_open"], result["margin_vs_open"])
    result["correct_at_close"] = pick_correct(
        result["pick_home_at_close"], result["margin_vs_close"]
    )
    # Binary actual outcome for Brier/log-loss, NaN on a push (excluded),
    # matching nfl_ats.outcomes.summarize_predictions' convention exactly.
    result["actual_home_cover_open"] = np.where(
        result["margin_vs_open"] > 0.0, 1.0, np.where(result["margin_vs_open"] < 0.0, 0.0, np.nan)
    )
    result["actual_home_cover_close"] = np.where(
        result["margin_vs_close"] > 0.0,
        1.0,
        np.where(result["margin_vs_close"] < 0.0, 0.0, np.nan),
    )
    oracle_correct = pick_correct(result["open_move"].gt(0.0), result["margin_vs_open"])
    result["oracle_correct_at_open"] = oracle_correct.where(result["open_move"].ne(0.0))
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def paired_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """One row per game scored by BOTH arms (mirrors mod07_weak_stack.paired_frame)."""

    keep = [
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "correct_at_open",
        "correct_at_close",
        "pick_home_at_open",
        "prob_at_open",
        "prob_at_close",
        "actual_home_cover_open",
        "actual_home_cover_close",
    ]
    left = baseline[keep].rename(
        columns={
            "correct_at_open": "baseline_correct_open",
            "correct_at_close": "baseline_correct_close",
            "pick_home_at_open": "baseline_pick_home",
            "prob_at_open": "baseline_prob_open",
            "prob_at_close": "baseline_prob_close",
        }
    )
    right = candidate[
        [
            "game_id",
            "correct_at_open",
            "correct_at_close",
            "pick_home_at_open",
            "prob_at_open",
            "prob_at_close",
        ]
    ].rename(
        columns={
            "correct_at_open": "candidate_correct_open",
            "correct_at_close": "candidate_correct_close",
            "pick_home_at_open": "candidate_pick_home",
            "prob_at_open": "candidate_prob_open",
            "prob_at_close": "candidate_prob_close",
        }
    )
    merged = left.merge(right, on="game_id", how="inner")
    for column in (
        "baseline_correct_open",
        "baseline_correct_close",
        "candidate_correct_open",
        "candidate_correct_close",
    ):
        merged[column] = merged[column].astype(float)
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
            # Oriented so positive = candidate better (lower Brier/log-loss).
            "brier_improvement": float(base_brier - cand_brier),
            "log_loss_improvement": float(base_ll - cand_ll),
        }

    return metric_fn


def line_flip_buckets(paired: pd.DataFrame) -> pd.DataFrame:
    flips = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])].copy()
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
                    float(subset["baseline_correct_open"].mean()) if len(subset) else float("nan")
                ),
                "candidate_correct_rate": (
                    float(subset["candidate_correct_open"].mean()) if len(subset) else float("nan")
                ),
            }
        )
    total_row = {
        "bucket": "ALL",
        "flips": len(flips),
        "baseline_correct_rate": (
            float(flips["baseline_correct_open"].mean()) if len(flips) else float("nan")
        ),
        "candidate_correct_rate": (
            float(flips["candidate_correct_open"].mean()) if len(flips) else float("nan")
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
    baseline = evaluate_arm(
        market_root, features, config=_config(BASELINE_ALPHA), min_train_games=min_train_games
    )
    t1 = time.perf_counter()
    candidate = evaluate_arm(
        market_root, features, config=_config(CANDIDATE_ALPHA), min_train_games=min_train_games
    )
    t2 = time.perf_counter()

    baseline_metrics = opener_evaluation_metrics(baseline)
    candidate_metrics = opener_evaluation_metrics(candidate)

    paired = paired_frame(baseline, candidate)

    open_bootstrap = week_blocked_bootstrap(
        paired,
        _accuracy_metric_fn("candidate_correct_open", "baseline_correct_open"),
        block="week",
        samples=samples,
        seed=seed,
    )
    open_bootstrap_season = week_blocked_bootstrap(
        paired,
        _accuracy_metric_fn("candidate_correct_open", "baseline_correct_open"),
        block="season",
        samples=samples,
        seed=seed,
    )
    close_bootstrap = week_blocked_bootstrap(
        paired,
        _accuracy_metric_fn("candidate_correct_close", "baseline_correct_close"),
        block="week",
        samples=samples,
        seed=seed,
    )
    close_bootstrap_season = week_blocked_bootstrap(
        paired,
        _accuracy_metric_fn("candidate_correct_close", "baseline_correct_close"),
        block="season",
        samples=samples,
        seed=seed,
    )
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

    disagreements = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])]
    flip_buckets = line_flip_buckets(paired)

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
        "open_accuracy_delta": {
            "week_blocked": open_bootstrap.iloc[0].to_dict(),
            "season_blocked": open_bootstrap_season.iloc[0].to_dict(),
        },
        "close_accuracy_delta": {
            "week_blocked": close_bootstrap.iloc[0].to_dict(),
            "season_blocked": close_bootstrap_season.iloc[0].to_dict(),
        },
        "brier_delta_open": {
            "week_blocked": brier_open_week.to_dict(orient="records"),
            "season_blocked": brier_open_season.to_dict(orient="records"),
        },
        "picks_disagreeing": len(disagreements),
        "disagreement_baseline_correct_open": (
            float(disagreements["baseline_correct_open"].mean())
            if len(disagreements)
            else float("nan")
        ),
        "disagreement_candidate_correct_open": (
            float(disagreements["candidate_correct_open"].mean())
            if len(disagreements)
            else float("nan")
        ),
        "flip_buckets": flip_buckets.to_dict(orient="records"),
        "paired_frame": paired,
        "baseline_frame": baseline,
        "candidate_frame": candidate,
    }


# ---------------------------------------------------------------------------
# nflverse_spread grade, full history (completeness)
# ---------------------------------------------------------------------------


def run_nflverse_grade(
    *,
    features: pd.DataFrame,
    start_season: int,
    min_train_games: int,
    min_edge: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    baseline_result = walk_forward_outcomes(
        features,
        start_season=start_season,
        regressor=REGRESSOR,
        min_edge=min_edge,
        min_train_games=min_train_games,
        feature_profile=PROFILE,
        methods=("market_residual",),
        ridge_alpha=BASELINE_ALPHA,
    )
    t1 = time.perf_counter()
    candidate_result = walk_forward_outcomes(
        features,
        start_season=start_season,
        regressor=REGRESSOR,
        min_edge=min_edge,
        min_train_games=min_train_games,
        feature_profile=PROFILE,
        methods=("market_residual",),
        ridge_alpha=CANDIDATE_ALPHA,
    )
    t2 = time.perf_counter()

    baseline_pred = baseline_result.predictions
    candidate_pred = candidate_result.predictions
    baseline_summary = summarize_outcome_method(baseline_pred)
    candidate_summary = summarize_outcome_method(candidate_pred)

    base_cover = baseline_pred.loc[
        baseline_pred["home_cover"].notna() & baseline_pred["home_cover_probability"].notna(),
        ["game_id", "season", "week", "home_cover", "home_cover_probability"],
    ].rename(columns={"home_cover_probability": "baseline_prob"})
    cand_cover = candidate_pred.loc[
        candidate_pred["home_cover"].notna() & candidate_pred["home_cover_probability"].notna(),
        ["game_id", "home_cover_probability"],
    ].rename(columns={"home_cover_probability": "candidate_prob"})
    merged = base_cover.merge(cand_cover, on="game_id", how="inner")
    merged["actual"] = merged["home_cover"].astype(float)
    merged["baseline_correct"] = (
        (merged["baseline_prob"] >= 0.5).astype(int) == merged["actual"].astype(int)
    ).astype(float)
    merged["candidate_correct"] = (
        (merged["candidate_prob"] >= 0.5).astype(int) == merged["actual"].astype(int)
    ).astype(float)

    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        actual = frame["actual"].to_numpy(dtype=float)
        cand = frame["candidate_prob"].to_numpy(dtype=float)
        base = frame["baseline_prob"].to_numpy(dtype=float)
        clip = lambda p: np.clip(p, 1e-15, 1.0 - 1e-15)  # noqa: E731
        return {
            "accuracy_delta": float(
                frame["candidate_correct"].mean() - frame["baseline_correct"].mean()
            ),
            "brier_improvement": float(
                np.mean(np.square(base - actual)) - np.mean(np.square(cand - actual))
            ),
            "log_loss_improvement": float(
                np.mean(-(actual * np.log(clip(base)) + (1 - actual) * np.log(1.0 - clip(base))))
                - np.mean(-(actual * np.log(clip(cand)) + (1 - actual) * np.log(1.0 - clip(cand))))
            ),
        }

    week_bootstrap = week_blocked_bootstrap(
        merged, metric_fn, block="week", samples=samples, seed=seed
    )
    season_bootstrap = week_blocked_bootstrap(
        merged, metric_fn, block="season", samples=samples, seed=seed
    )

    return {
        "timing": {
            "baseline_seconds": t1 - t0,
            "candidate_seconds": t2 - t1,
            "total_seconds": t2 - t0,
        },
        "paired_games": len(merged),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "week_blocked": week_bootstrap.to_dict(orient="records"),
        "season_blocked": season_bootstrap.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO / "data/processed/game_features_weak_stack.parquet",
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--opener-samples", type=int, default=OPENER_BOOTSTRAP_SAMPLES)
    parser.add_argument("--opener-seed", type=int, default=OPENER_BOOTSTRAP_SEED)
    parser.add_argument("--nflverse-start-season", type=int, default=2018)
    parser.add_argument("--nflverse-samples", type=int, default=NFLVERSE_BOOTSTRAP_SAMPLES)
    parser.add_argument("--nflverse-seed", type=int, default=NFLVERSE_BOOTSTRAP_SEED)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--skip-nflverse", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)

    print(f"=== Opener grade (primary): alpha={BASELINE_ALPHA:g} vs alpha={CANDIDATE_ALPHA:g} ===")
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
        out_dir = REPO / "artifacts" / "ridge_alpha_promotion" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "opener_summary.json").write_text(
        json.dumps(opener_report, indent=2, default=str), encoding="utf-8"
    )
    opener_result["paired_frame"].to_parquet(out_dir / "opener_paired.parquet")
    opener_result["baseline_frame"].to_parquet(out_dir / "opener_baseline.parquet")
    opener_result["candidate_frame"].to_parquet(out_dir / "opener_candidate.parquet")

    if not args.skip_nflverse:
        print("\n=== nflverse_spread grade (completeness): full history ===")
        nflverse_result = run_nflverse_grade(
            features=features,
            start_season=args.nflverse_start_season,
            min_train_games=args.min_train_games,
            min_edge=args.min_edge,
            samples=args.nflverse_samples,
            seed=args.nflverse_seed,
        )
        print(json.dumps(nflverse_result, indent=2, default=str))
        (out_dir / "nflverse_summary.json").write_text(
            json.dumps(nflverse_result, indent=2, default=str), encoding="utf-8"
        )

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
