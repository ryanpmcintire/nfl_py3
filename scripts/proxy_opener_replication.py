"""MEASURE-ONLY: score the frozen PRODUCTION model's pick rules on 11 seasons
(2009-2019) of SBR "Open" lines used as a PROXY for the Tuesday opener --
territory no opener-graded look has ever touched (the purchased point-in-time
market archive only reaches back to 2020).

Predeclared in ``docs/proxy_opener_replication.md`` BEFORE this script
produced any accuracy number -- population, grade, model, both pick rules,
bootstrap spec, warm-up-floor consequence (2009-2010 fail the same 500-game
floor the true opener evaluation and the rotation registry both already use,
so the scorable population is 2011-2019, 2,304 games, not the full
2,816-game SBR-matched 2009-2019 set), and a same-instrument calibration arm
on the 2020-2021 overlap. Read that document for the full predeclaration;
this docstring does not repeat it.

Single arm only: production ``market_residual``/``weak_stack``/ridge
alpha=10.0 (``artifacts/active_ats_model.json``). No candidate, no tuning.

Mechanics, adapted from ``nfl_ats.clv.opener_pick_evaluation`` (called
directly, unmodified, for the calibration arm's true-opener side) and
``scripts/surface_profile_opener_eval.py``'s pattern for the harness shape:
one weekly-refit model per (season, week), trained on completed REG games
strictly before that week's first kickoff, evaluated with ``spread_line``
overridden to SBR's ``open_home_spread`` at scoring time only (training rows
keep their native close-era ``spread_line``) -- the same "only the settling
line changes" convention the true-opener evaluation uses.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/proxy_opener_replication.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, pick_correct, week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.margin import MarginFeatureProfile, fit_margin_model
from nfl_ats.modeling import regular_season_rows

REPO = Path(__file__).resolve().parents[1]

# Frozen production configuration (artifacts/active_ats_model.json, read
# 2026-08-19). Single arm -- no candidate, nothing swept.
FEATURE_PROFILE: MarginFeatureProfile = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

# Main population range (docs/proxy_opener_replication.md).
MAIN_SEASON_START = 2009
MAIN_SEASON_END = 2019

# Calibration overlap range: the only two seasons where both a purchased
# Tuesday-opener archive and an SBR Open exist together.
CALIBRATION_SEASONS = (2020, 2021)

# Project's standing opener-bootstrap seed (docs/opener_evaluation.md,
# scripts/surface_profile_opener_eval.py, scripts/ridge_alpha_promotion_eval.py)
# -- reused, not a fresh choice. Task-specified sample count.
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817


def _config() -> dict[str, Any]:
    return {"feature_profile": FEATURE_PROFILE, "regressor": REGRESSOR, "ridge_alpha": RIDGE_ALPHA}


def sbr_proxy_pick_evaluation(
    *,
    features: pd.DataFrame,
    population: pd.DataFrame,
    min_train_games: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score the frozen production model against SBR's Open, substituted as
    the line, on every ``population`` game.

    ``population`` must carry ``game_id``, ``season``, ``week``,
    ``proxy_open_home_spread``. ``features`` is the FULL regular-season
    feature table (training draws from all of it, exactly matching
    ``opener_pick_evaluation``'s convention -- training is never restricted
    to games that happen to have an SBR quote). Mirrors
    ``opener_pick_evaluation``'s internal loop exactly except the settling
    line comes from SBR's Open rather than the store's ``tue_open`` snapshot.

    Returns (scored games, skipped-week log) -- weeks skipped for failing
    ``min_train_games`` are logged, not silently dropped, so warm-up
    consequences are auditable from the artifact rather than only from the
    predeclaration's precomputed table.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    scored_weeks: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for (season, week), group in population.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            skipped.append(
                {"season": int(season), "week": int(week), "reason": "no feature rows", "games": 0}
            )
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            skipped.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "reason": "below min_train_games",
                    "train_games": len(training),
                    "games": len(group),
                }
            )
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=REGRESSOR,
            feature_profile=FEATURE_PROFILE,
            ridge_alpha=RIDGE_ALPHA,
        )
        scoring = week_rows.merge(
            group[["game_id", "proxy_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["proxy_open_home_spread"]
        predicted = model.predict(at_open)
        scored = scoring[["game_id"]].copy()
        scored["season"] = int(season)
        scored["week"] = int(week)
        scored["residual_at_open_proxy"] = predicted["predicted_market_residual"].to_numpy()
        scored["home_cover_probability_at_open_proxy"] = predicted[
            "home_cover_probability"
        ].to_numpy()
        scored_weeks.append(scored)

    if not scored_weeks:
        raise ValueError("No population week cleared min_train_games")
    residuals = pd.concat(scored_weeks, ignore_index=True)

    result = population.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
    result["margin_vs_open_proxy"] = result["result"] - result["proxy_open_home_spread"]
    result["pick_home_at_open_proxy"] = result["residual_at_open_proxy"].gt(0.0)
    result["pick_home_at_open_proxy_pr"] = result["home_cover_probability_at_open_proxy"].ge(0.5)
    result["correct_at_open_proxy"] = pick_correct(
        result["pick_home_at_open_proxy"], result["margin_vs_open_proxy"]
    )
    result["correct_at_open_proxy_pr"] = pick_correct(
        result["pick_home_at_open_proxy_pr"], result["margin_vs_open_proxy"]
    )
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True), skipped


def build_population(
    features: pd.DataFrame, sbr_odds: pd.DataFrame, season_start: int, season_end: int
) -> pd.DataFrame:
    reg = regular_season_rows(features)
    reg = reg.loc[reg["season"].between(season_start, season_end)]
    sbr = sbr_odds.dropna(subset=["game_id", "open_home_spread"])
    pop = (
        reg[["game_id", "season", "week", "gameday", "result"]]
        .merge(
            sbr[["game_id", "open_home_spread", "open_ambiguous"]],
            on="game_id",
            how="inner",
        )
        .rename(columns={"open_home_spread": "proxy_open_home_spread"})
    )
    pop = pop.loc[pd.to_numeric(pop["result"], errors="coerce").notna()].copy()
    pop["season"] = pop["season"].astype(int)
    pop["week"] = pop["week"].astype(int)
    return pop


def _vs_coin_flip_metric_fn(col: str) -> Any:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        return {"accuracy_minus_50": float(frame[col].mean()) - 0.5}

    return metric_fn


def _paired_delta_metric_fn(sbr_col: str, true_col: str) -> Any:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        return {"sbr_minus_true": float(frame[sbr_col].mean() - frame[true_col].mean())}

    return metric_fn


def per_season_table(
    scored: pd.DataFrame, season_start: int, season_end: int
) -> list[dict[str, Any]]:
    rows = []
    scored_by_season = scored.groupby("season")
    for season in range(season_start, season_end + 1):
        if season not in scored_by_season.groups:
            rows.append(
                {
                    "season": season,
                    "games": 0,
                    "scorable": False,
                    "accuracy_sign_rule": float("nan"),
                    "accuracy_probability_rule": float("nan"),
                }
            )
            continue
        group = scored_by_season.get_group(season)
        rows.append(
            {
                "season": season,
                "games": len(group),
                "scorable": True,
                "accuracy_sign_rule": float(group["correct_at_open_proxy"].mean()),
                "accuracy_probability_rule": float(group["correct_at_open_proxy_pr"].mean()),
            }
        )
    return rows


def run_main_arm(
    *, features: pd.DataFrame, sbr_odds: pd.DataFrame, min_train_games: int, samples: int, seed: int
) -> dict[str, Any]:
    population = build_population(features, sbr_odds, MAIN_SEASON_START, MAIN_SEASON_END)
    scored, skipped = sbr_proxy_pick_evaluation(
        features=features, population=population, min_train_games=min_train_games
    )

    def _bootstrap(col: str, block: str) -> dict[str, float]:
        return (
            week_blocked_bootstrap(
                scored, _vs_coin_flip_metric_fn(col), block=block, samples=samples, seed=seed
            )
            .iloc[0]
            .to_dict()
        )

    pooled = {
        "sign_rule": {
            "week_blocked": _bootstrap("correct_at_open_proxy", "week"),
            "season_blocked": _bootstrap("correct_at_open_proxy", "season"),
        },
        "probability_rule": {
            "week_blocked": _bootstrap("correct_at_open_proxy_pr", "week"),
            "season_blocked": _bootstrap("correct_at_open_proxy_pr", "season"),
        },
    }
    return {
        "population_games_2009_2019": len(population),
        "population_ambiguous_open_games": int(population["open_ambiguous"].sum()),
        "scored_games": len(scored),
        "scored_seasons": sorted(int(s) for s in scored["season"].unique()),
        "skipped_weeks": skipped,
        "per_season": per_season_table(scored, MAIN_SEASON_START, MAIN_SEASON_END),
        "pooled_accuracy_vs_coin_flip": pooled,
        "pooled_accuracy_absolute": {
            "sign_rule": float(scored["correct_at_open_proxy"].mean()),
            "probability_rule": float(scored["correct_at_open_proxy_pr"].mean()),
        },
        "scored_frame": scored,
    }


def run_calibration_arm(
    *,
    features: pd.DataFrame,
    sbr_odds: pd.DataFrame,
    market_root: Path,
    min_train_games: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    calib_population = build_population(
        features, sbr_odds, CALIBRATION_SEASONS[0], CALIBRATION_SEASONS[1]
    )
    sbr_scored, sbr_skipped = sbr_proxy_pick_evaluation(
        features=features, population=calib_population, min_train_games=min_train_games
    )

    true_scored = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=_config(),
        min_train_games=min_train_games,
    )
    true_scored = true_scored.loc[true_scored["season"].isin(CALIBRATION_SEASONS)].copy()

    paired = sbr_scored[
        [
            "game_id",
            "season",
            "week",
            "proxy_open_home_spread",
            "correct_at_open_proxy",
            "correct_at_open_proxy_pr",
        ]
    ].merge(
        true_scored[
            [
                "game_id",
                "tue_open_home_spread",
                "correct_at_open",
                "correct_at_open_probability_rule",
            ]
        ],
        on="game_id",
        how="inner",
    )

    def _bootstrap_delta(sbr_col: str, true_col: str, block: str) -> dict[str, float]:
        return (
            week_blocked_bootstrap(
                paired,
                _paired_delta_metric_fn(sbr_col, true_col),
                block=block,
                samples=samples,
                seed=seed,
            )
            .iloc[0]
            .to_dict()
        )

    return {
        "sbr_scored_games": len(sbr_scored),
        "sbr_skipped_weeks": sbr_skipped,
        "true_scored_games_2020_2021": len(true_scored),
        "paired_games": len(paired),
        "absolute_accuracy": {
            "sbr_open_sign_rule": float(paired["correct_at_open_proxy"].mean()),
            "sbr_open_probability_rule": float(paired["correct_at_open_proxy_pr"].mean()),
            "true_open_sign_rule": float(paired["correct_at_open"].mean()),
            "true_open_probability_rule": float(paired["correct_at_open_probability_rule"].mean()),
        },
        "paired_delta_sbr_minus_true": {
            "sign_rule": {
                "week_blocked": _bootstrap_delta(
                    "correct_at_open_proxy", "correct_at_open", "week"
                ),
                "season_blocked": _bootstrap_delta(
                    "correct_at_open_proxy", "correct_at_open", "season"
                ),
            },
            "probability_rule": {
                "week_blocked": _bootstrap_delta(
                    "correct_at_open_proxy_pr", "correct_at_open_probability_rule", "week"
                ),
                "season_blocked": _bootstrap_delta(
                    "correct_at_open_proxy_pr", "correct_at_open_probability_rule", "season"
                ),
            },
        },
        "mean_abs_line_diff": float(
            (paired["proxy_open_home_spread"] - paired["tue_open_home_spread"]).abs().mean()
        ),
        "paired_frame": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_weak_stack.parquet"
    )
    parser.add_argument("--sbr-odds", type=Path, default=REPO / "data/processed/sbr_odds.parquet")
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the 2020-2021 dual-grade calibration arm (main arm only; for chunked runs).",
    )
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    sbr_odds = pd.read_parquet(args.sbr_odds)

    print(f"=== Main arm: proxy-opener grade, {MAIN_SEASON_START}-{MAIN_SEASON_END} ===")
    main_result = run_main_arm(
        features=features,
        sbr_odds=sbr_odds,
        min_train_games=args.min_train_games,
        samples=args.samples,
        seed=args.seed,
    )
    main_report = {k: v for k, v in main_result.items() if k != "scored_frame"}
    print(json.dumps(main_report, indent=2, default=str))

    calibration_result: dict[str, Any] | None = None
    if not args.skip_calibration:
        print(f"\n=== Calibration arm: dual grade, {CALIBRATION_SEASONS} ===")
        calibration_result = run_calibration_arm(
            features=features,
            sbr_odds=sbr_odds,
            market_root=args.market_root,
            min_train_games=args.min_train_games,
            samples=args.samples,
            seed=args.seed,
        )
        calibration_report = {k: v for k, v in calibration_result.items() if k != "paired_frame"}
        print(json.dumps(calibration_report, indent=2, default=str))

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "artifacts" / "proxy_opener_replication" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "feature_profile": FEATURE_PROFILE,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "features_path": str(args.features),
        "sbr_odds_path": str(args.sbr_odds),
        "market_root": str(args.market_root),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "main_arm": main_report,
        "calibration_arm": (
            {k: v for k, v in calibration_result.items() if k != "paired_frame"}
            if calibration_result is not None
            else None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    main_result["scored_frame"].to_parquet(out_dir / "main_scored.parquet")
    if calibration_result is not None:
        calibration_result["paired_frame"].to_parquet(out_dir / "calibration_paired.parquet")

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
