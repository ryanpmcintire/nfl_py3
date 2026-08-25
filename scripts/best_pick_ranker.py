"""SPEC-5 Best Pick ranker: screen evaluation machinery.

Built and debugged 2026-08-17 on non-reserved seasons (stage A, plumbing proof
only — see ``docs/best_pick_ranker.md``). Stage B runs exactly once on the
registry-assigned window ([2013, 2015] under rule 9's warm-up floor, with the
raw stream started in 2011 so calibration is warm by the window's first week —
see the revised SPEC-5 in ``docs/archive/opus_execution_specs.md``). Nothing here
writes to the registry — recording the look is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from nfl_ats.best_pick import sweep_robustness
from nfl_ats.calibration import calibrate_cover_prediction_stream
from nfl_ats.clv import (
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    KEY_NUMBERS,
    build_pairing_table,
    close_reference_table,
    key_number_distance,
    opener_pick_evaluation,
    week_blocked_bootstrap,
)
from nfl_ats.margin import DEFAULT_LINE_SWEEP_OFFSETS, fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.outcomes import walk_forward_outcomes

REPO = Path(__file__).resolve().parents[1]

SIGNALS = ("calibrated_probability", "key_number_distance", "sweep_robustness")

# Standard evaluator configuration (the frozen active config, base profile).
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
MIN_EDGE = 0.02
FEATURE_PROFILE = "base"
METHODS = ("market_residual",)


# ---------------------------------------------------------------------------
# 1. Walk-forward predictions + per-week sweeps from the SAME weekly fit
# ---------------------------------------------------------------------------


def sweep_frame(
    frame: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_games: int,
) -> pd.DataFrame:
    """Re-run the walk-forward loop, emitting a line sweep per scored game.

    Mirrors ``walk_forward_outcomes`` exactly (same cutoff, same fit call, same
    hyper-parameters), so the fitted model at each week is the identical object;
    verified in stage A by reproducing ``predicted_margin`` bit-for-bit.
    """

    work = regular_season_rows(frame).copy()
    work["gameday"] = pd.to_datetime(work["gameday"], errors="raise")
    completed = work.loc[work["result"].notna()].copy()
    test = completed.loc[completed["season"].ge(start_season) & completed["season"].le(end_season)]
    rows: list[pd.DataFrame] = []
    for (season, week), weekly in test.groupby(["season", "week"], sort=True):
        cutoff = weekly["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=REGRESSOR,
            feature_profile=FEATURE_PROFILE,
            ridge_alpha=RIDGE_ALPHA,
        )
        sweep = model.line_sweep(weekly, offsets=DEFAULT_LINE_SWEEP_OFFSETS)
        sweep["season"] = int(str(season))
        sweep["week"] = int(str(week))
        # The un-swept prediction from the same fit, for the reproduction check.
        point = model.predict(weekly)
        point["game_id"] = weekly["game_id"].to_numpy()
        rows.append(sweep.merge(point[["game_id", "predicted_margin"]], on="game_id", how="left"))
    if not rows:
        raise ValueError("No week had enough prior training games for the sweep")
    return pd.concat(rows, ignore_index=True)


# The signal itself now lives in ``nfl_ats.best_pick`` so the weekly Best Pick
# the published site shows is computed by the SAME function that was scored here.
# Keeping a second copy in this script is how a confirmed signal silently drifts
# away from the deployed one.


# ---------------------------------------------------------------------------
# 2. Signals
# ---------------------------------------------------------------------------


def build_picks(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_calibration_games: int,
) -> pd.DataFrame:
    picks = predictions.loc[predictions["season"].between(start_season, end_season)].copy()
    picks["pick"] = np.where(picks["home_cover_probability"] >= 0.5, "HOME", "AWAY")
    # Pushes carry no correctness; forced-pick accuracy is defined on resolved games.
    picks = picks.loc[picks["home_cover"].notna()].copy()
    picks["correct"] = np.where(
        picks["pick"].eq("HOME"),
        picks["home_cover"].astype(float),
        1.0 - picks["home_cover"].astype(float),
    )

    # Signal 1: Platt-calibrated pick-side cover probability.
    calibrated = calibrate_cover_prediction_stream(
        predictions,
        method="platt",
        evaluation_start_season=start_season,
        min_calibration_games=min_calibration_games,
        min_edge=MIN_EDGE,
    )
    calibrated_lookup = calibrated.set_index("game_id")["home_cover_probability"]
    home_calibrated = picks["game_id"].map(calibrated_lookup).astype(float)
    picks["calibrated_probability"] = np.where(
        picks["pick"].eq("HOME"), home_calibrated, 1.0 - home_calibrated
    )

    # Signal 2: market key-number distance minus fair key-number distance.
    picks["key_number_distance"] = (
        key_number_distance(picks["spread_line"].astype(float), KEY_NUMBERS).to_numpy()
        - key_number_distance(picks["fair_spread"].astype(float), KEY_NUMBERS).to_numpy()
    )

    # Signal 3: sweep robustness.
    picks["sweep_robustness"] = picks["game_id"].map(sweep_robustness(sweep, picks)).astype(float)
    return picks


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


def _flag_top1(picks: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Deterministically mark the week's highest-signal pick.

    Marked ONCE, before any resampling: the bootstrap resamples whole weeks and
    a week drawn twice must contribute its top-1 pick twice, which recomputing
    the ranking inside the metric function would not do.
    """

    frame = picks.dropna(subset=[signal]).copy()
    # Deterministic tie-break by game_id so the ranking never depends on row order.
    frame = frame.sort_values([signal, "game_id"], ascending=[False, True])
    frame["is_top1"] = ~frame.duplicated(subset=["season", "week"], keep="first")
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    top1 = frame.loc[frame["is_top1"]]
    if top1.empty or frame.empty:
        return {"top1_minus_all": float("nan")}
    return {"top1_minus_all": float(top1["correct"].mean() - frame["correct"].mean())}


def evaluate_signal(picks: pd.DataFrame, signal: str, *, samples: int, seed: int) -> dict[str, Any]:
    frame = _flag_top1(picks, signal)
    top1 = frame.loc[frame["is_top1"]]
    tau, tau_p = kendalltau(
        frame[signal].to_numpy(dtype=float), frame["correct"].to_numpy(dtype=float)
    )
    bootstrap = week_blocked_bootstrap(frame, _metric_fn, block="week", samples=samples, seed=seed)
    row = bootstrap.iloc[0]
    return {
        "signal": signal,
        "weeks": int(frame.groupby(["season", "week"]).ngroups),
        "picks": len(frame),
        "top1_picks": len(top1),
        "top1_accuracy": float(top1["correct"].mean()),
        "all_pick_accuracy": float(frame["correct"].mean()),
        "delta": float(top1["correct"].mean() - frame["correct"].mean()),
        "kendall_tau": float(tau),
        "kendall_tau_p": float(tau_p),
        "bootstrap_lower": float(row["lower"]),
        "bootstrap_upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------


def run(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    raw_start_season: int,
    min_train_games: int,
    min_calibration_games: int,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result = walk_forward_outcomes(
        features,
        start_season=raw_start_season,
        end_season=end_season,
        regressor=REGRESSOR,
        min_edge=MIN_EDGE,
        min_train_games=min_train_games,
        feature_profile=FEATURE_PROFILE,
        methods=METHODS,
        ridge_alpha=RIDGE_ALPHA,
    )
    predictions = result.predictions
    sweep = sweep_frame(
        features,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
    )

    # Proof that the sweep loop reproduces the evaluator's weekly fit exactly.
    check = predictions[["game_id", "predicted_margin"]].merge(
        sweep[["game_id", "predicted_margin"]].drop_duplicates("game_id"),
        on="game_id",
        suffixes=("_eval", "_sweep"),
    )
    reproduction_max_abs_diff = float(
        (check["predicted_margin_eval"] - check["predicted_margin_sweep"]).abs().max()
    )

    picks = build_picks(
        predictions,
        sweep,
        start_season=start_season,
        end_season=end_season,
        min_calibration_games=min_calibration_games,
    )
    rows = [evaluate_signal(picks, signal, samples=samples, seed=seed) for signal in SIGNALS]
    summary: dict[str, Any] = {
        "start_season": start_season,
        "end_season": end_season,
        "raw_start_season": raw_start_season,
        "min_train_games": min_train_games,
        "min_calibration_games": min_calibration_games,
        "scored_games": len(predictions),
        "scored_weeks": int(predictions.groupby(["season", "week"]).ngroups),
        "resolved_picks": len(picks),
        "sweep_reproduction_max_abs_diff": reproduction_max_abs_diff,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "signals": rows,
    }
    return summary, picks


# ---------------------------------------------------------------------------
# 5. Opener-graded confirmation arm (SPEC-5's gate stage)
# ---------------------------------------------------------------------------
#
# The screen grades against the nflverse spread. The confirmation grades the
# same frozen signal against the Tuesday opener the pool actually uses, which
# means re-forming every pick at the opener line. ``opener_pick_evaluation``
# already does that; what it does not emit is a line sweep, so ``sweep_robustness``
# has to come from a parallel loop that refits the identical weekly model and
# sweeps ``at_open``. The reproduction check below proves the two loops share a
# fit: the sweep loop's residual at the opener must equal the evaluator's
# ``residual_at_open`` to 0.0, exactly as stage A proved for the screen.


def opener_sweep_frame(
    features: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    feature_profile: str,
    min_train_games: int,
) -> pd.DataFrame:
    """Line sweeps anchored at the Tuesday opener, from the weekly opener fit."""

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()
    rows: list[pd.DataFrame] = []
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
            model_name=REGRESSOR,
            feature_profile=feature_profile,  # type: ignore[arg-type]
            ridge_alpha=RIDGE_ALPHA,
        )
        # Swap ONLY spread_line to the opener, matching opener_pick_evaluation's
        # declared approximation (every other feature stays close-era).
        at_open = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        sweep = model.line_sweep(at_open, offsets=DEFAULT_LINE_SWEEP_OFFSETS)
        sweep["season"] = int(str(season))
        sweep["week"] = int(str(week))
        point = model.predict(at_open)
        point["game_id"] = at_open["game_id"].to_numpy()
        rows.append(
            sweep.merge(point[["game_id", "predicted_market_residual"]], on="game_id", how="left")
        )
    if not rows:
        raise ValueError("No paired week had enough prior training games for the opener sweep")
    return pd.concat(rows, ignore_index=True)


def run_opener_confirmation(
    features: pd.DataFrame,
    *,
    market_root: Path,
    start_season: int,
    end_season: int,
    feature_profile: str,
    min_train_games: int,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score ``sweep_robustness`` top-1 at the opener grade on one window."""

    config = {
        "feature_profile": feature_profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }
    # Truncate at the window's last season so no unspent window is ever scored,
    # even into a discarded frame. Training is unaffected: the evaluator takes
    # strictly-earlier gamedays only, and everything earlier is still present.
    features = features.loc[features["season"].astype(int).le(end_season)].copy()
    scored = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=config,
        min_train_games=min_train_games,
    )
    window = scored.loc[scored["season"].between(start_season, end_season)].copy()
    if window.empty:
        raise ValueError(f"No opener-paired games in [{start_season}, {end_season}]")

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=regular_season_rows(features),
    )
    close = close_reference_table(pairing, regular_season_rows(features))
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    paired = tue_open.merge(close[["game_id"]], on="game_id", how="inner")
    paired = paired.loc[paired["game_id"].isin(set(window["game_id"]))]

    sweep = opener_sweep_frame(
        features,
        paired,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
    )

    # Same-fit proof: the sweep loop must reproduce the evaluator's opener residual.
    check = window[["game_id", "residual_at_open"]].merge(
        sweep[["game_id", "predicted_market_residual"]].drop_duplicates("game_id"),
        on="game_id",
        how="inner",
    )
    reproduction_max_abs_diff = float(
        (check["residual_at_open"] - check["predicted_market_residual"]).abs().max()
    )

    picks = window.loc[window["correct_at_open"].notna()].copy()
    picks["pick"] = np.where(picks["pick_home_at_open"], "HOME", "AWAY")
    picks["correct"] = picks["correct_at_open"].astype(float)
    picks["sweep_robustness"] = picks["game_id"].map(sweep_robustness(sweep, picks)).astype(float)

    row = evaluate_signal(picks, "sweep_robustness", samples=samples, seed=seed)
    summary: dict[str, Any] = {
        "grade": "opener",
        "start_season": start_season,
        "end_season": end_season,
        "feature_profile": feature_profile,
        "min_train_games": min_train_games,
        "paired_games": len(window),
        "resolved_picks": len(picks),
        "opener_sweep_reproduction_max_abs_diff": reproduction_max_abs_diff,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "signals": [row],
    }
    return summary, picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument(
        "--grade",
        choices=("nflverse_spread", "opener"),
        default="nflverse_spread",
        help="nflverse_spread runs the screen; opener runs SPEC-5's confirmation arm",
    )
    parser.add_argument(
        "--feature-profile",
        default="base",
        help=(
            "margin feature profile for the opener arm. SPEC-5 does not settle whether "
            "the confirmation replicates the screen's signal exactly (base) or ranks the "
            "picks we would actually deploy (player) -- an owner decision, not a default."
        ),
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data" / "market" / "raw")
    parser.add_argument("--raw-start-season", type=int, default=None)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--min-calibration-games", type=int, default=400)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features.parquet"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    if args.grade == "opener":
        summary, picks = run_opener_confirmation(
            features,
            market_root=args.market_root,
            start_season=args.start_season,
            end_season=args.end_season,
            feature_profile=args.feature_profile,
            min_train_games=args.min_train_games,
            samples=args.samples,
            seed=args.seed,
        )
    else:
        summary, picks = run(
            features,
            start_season=args.start_season,
            end_season=args.end_season,
            raw_start_season=args.raw_start_season or args.start_season,
            min_train_games=args.min_train_games,
            min_calibration_games=args.min_calibration_games,
            samples=args.samples,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        picks.to_parquet(args.out.with_suffix(".picks.parquet"))


if __name__ == "__main__":
    main()
