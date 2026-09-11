from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import git_state, sha256_file, utc_now  # noqa: E402
from nfl_ats.tiebreaker import (  # noqa: E402
    last_game_of_week,
    market_implied_scores,
    newest_schedules_path,
)

OUTPUT_ROOT = Path("artifacts/tiebreaker_low_side_shading")
SEASON_START = 2009
SEASON_END = 2025
SHADE_OFFSETS = (0.0, -1.0, -2.0, -3.0, -4.0)
MODEL_LEAN_WEIGHT = 0.1
PUBLIC_FIELD_MEAN_OFFSET = 1.5
PUBLIC_FIELD_SD = 6.0
PUBLIC_DRAWS = 20000
BOOT_SAMPLES = 4000
SEED = 20260910
SWEEP_MEAN_OFFSETS = (0.0, 1.0, 1.5, 2.0, 3.0)
SWEEP_SDS = (4.0, 6.0, 8.0, 10.0)
OOS_SPLIT_SEASON = 2018
TOTALS_PREDICTIONS_PATH = Path("artifacts/totals_backtest/20260901T184010Z/predictions.parquet")


def build_last_game_frame(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    rows: list[dict[str, Any]] = []
    for season in range(SEASON_START, SEASON_END + 1):
        season_games = reg.loc[reg["season"].eq(season)]
        if season_games.empty:
            continue
        weeks = sorted(int(week) for week in season_games["week"].dropna().unique())
        for week in weeks:
            try:
                last = last_game_of_week(reg, season, week)
            except ValueError:
                continue
            if (
                pd.isna(last.get("total_line"))
                or pd.isna(last.get("home_score"))
                or pd.isna(last.get("away_score"))
                or pd.isna(last.get("spread_line"))
            ):
                continue
            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "game_id": str(last["game_id"]),
                    "home_team": str(last["home_team"]),
                    "away_team": str(last["away_team"]),
                    "market_total": float(last["total_line"]),
                    "home_expected_margin": float(last["spread_line"]),
                    "home_score": float(last["home_score"]),
                    "away_score": float(last["away_score"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["actual_total"] = frame["home_score"] + frame["away_score"]
    frame["signed_total_error"] = frame["actual_total"] - frame["market_total"]
    frame["under"] = (frame["actual_total"] < frame["market_total"]).astype(float)
    frame["push"] = (frame["actual_total"] == frame["market_total"]).astype(float)
    frame["over"] = (frame["actual_total"] > frame["market_total"]).astype(float)
    return frame


def under_rate_by_season(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("season", as_index=False).agg(
        games=("game_id", "size"),
        under_rate=("under", "mean"),
        push_rate=("push", "mean"),
        over_rate=("over", "mean"),
    )
    return grouped


def under_rate_by_bucket(frame: pd.DataFrame, n_buckets: int = 4) -> pd.DataFrame:
    working = frame.copy()
    working["total_bucket"] = pd.qcut(working["market_total"], q=n_buckets, duplicates="drop")
    grouped = working.groupby("total_bucket", observed=True).agg(
        games=("game_id", "size"),
        market_total_min=("market_total", "min"),
        market_total_max=("market_total", "max"),
        under_rate=("under", "mean"),
        push_rate=("push", "mean"),
        over_rate=("over", "mean"),
    )
    return grouped.reset_index().rename(columns={"total_bucket": "bucket"})


def _bootstrap_mean(
    frame: pd.DataFrame, column: str, *, block: str, samples: int, seed: int
) -> pd.DataFrame:
    def metric_fn(block_frame: pd.DataFrame) -> dict[str, float]:
        return {column: float(block_frame[column].mean())}

    return week_blocked_bootstrap(frame, metric_fn, block=block, samples=samples, seed=seed)


def under_rate_overall(frame: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    def metric_fn(block_frame: pd.DataFrame) -> dict[str, float]:
        return {"under_minus_half": float(block_frame["under"].mean()) - 0.5}

    result = week_blocked_bootstrap(frame, metric_fn, block="season", samples=samples, seed=seed)
    row = result.iloc[0]
    return {
        "games": len(frame),
        "seasons": int(frame["season"].nunique()),
        "under_rate": float(row["estimate"]) + 0.5,
        "lower": float(row["lower"]) + 0.5,
        "upper": float(row["upper"]) + 0.5,
        "probability_positive_over_half": float(row["probability_positive"]),
    }


def season_pair_reliability(
    season_rates: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    ordered = season_rates.sort_values("season").reset_index(drop=True)
    pairs = []
    seasons = ordered["season"].tolist()
    for season in seasons:
        if season % 2 == 1 and (season + 1) in seasons:
            odd_row = ordered.loc[ordered["season"].eq(season)].iloc[0]
            even_row = ordered.loc[ordered["season"].eq(season + 1)].iloc[0]
            pairs.append(
                {
                    "odd_season": int(season),
                    "even_season": int(season + 1),
                    "odd_under_rate": float(odd_row["under_rate"]),
                    "even_under_rate": float(even_row["under_rate"]),
                }
            )
    pair_frame = pd.DataFrame(pairs)
    n_pairs = len(pair_frame)
    if n_pairs < 3:
        return {
            "n_pairs": n_pairs,
            "pearson_r": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
            "pairs": pairs,
        }
    odd_vals = pair_frame["odd_under_rate"].to_numpy(dtype=float)
    even_vals = pair_frame["even_under_rate"].to_numpy(dtype=float)
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    rng = np.random.default_rng(seed)
    boots = np.empty(samples, dtype=float)
    for draw in range(samples):
        selected = rng.integers(0, n_pairs, size=n_pairs)
        sampled_odd = odd_vals[selected]
        sampled_even = even_vals[selected]
        if sampled_odd.std() > 0 and sampled_even.std() > 0:
            boots[draw] = np.corrcoef(sampled_odd, sampled_even)[0, 1]
        else:
            boots[draw] = np.nan
    return {
        "n_pairs": n_pairs,
        "pearson_r": r,
        "lower": float(np.nanquantile(boots, 0.025)),
        "upper": float(np.nanquantile(boots, 0.975)),
        "probability_positive": float(probability_positive_from_draws(boots, ignore_nan=True)),
        "pairs": pairs,
    }


def half_pooled_under_rate(
    frame: pd.DataFrame, seasons: tuple[int, ...], *, samples: int, seed: int
) -> dict[str, Any]:
    subset = frame.loc[frame["season"].isin(seasons)]
    if subset.empty:
        return {
            "games": 0,
            "under_rate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    result = _bootstrap_mean(subset, "under", block="season", samples=samples, seed=seed)
    row = result.iloc[0]
    return {
        "games": len(subset),
        "seasons": sorted(int(season) for season in subset["season"].unique()),
        "under_rate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
    }


def attach_model_lean(frame: pd.DataFrame, predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.is_file():
        working = frame.copy()
        working["predicted_residual"] = np.nan
        return working
    predictions = pd.read_parquet(predictions_path, columns=["game_id", "predicted_residual"])
    predictions["game_id"] = predictions["game_id"].astype(str)
    merged = frame.merge(predictions, on="game_id", how="left")
    return merged


def generate_public_shocks(
    n_games: int, *, mean_offset: float, sd: float, draws: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean_offset, sd, size=(n_games, draws))


def win_rate_from_shocks(
    offset: np.ndarray, actual_total: np.ndarray, market_total: np.ndarray, shocks: np.ndarray
) -> np.ndarray:
    d = actual_total - market_total
    our_abs_error = np.abs(offset - d)
    public_abs_error = np.abs(shocks - d[:, None])
    wins = (our_abs_error[:, None] < public_abs_error).mean(axis=1)
    ties = (our_abs_error[:, None] == public_abs_error).mean(axis=1)
    return wins + 0.5 * ties


def paired_bootstrap(
    frame: pd.DataFrame, value_column: str, *, samples: int, seed: int
) -> dict[str, Any]:
    def metric_fn(block: pd.DataFrame) -> dict[str, float]:
        return {"delta": float(block[value_column].mean())}

    result = week_blocked_bootstrap(frame, metric_fn, block="season", samples=samples, seed=seed)
    row = result.iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "games": len(frame),
        "blocks": int(frame["season"].nunique()),
    }


def shade_comparison(
    frame: pd.DataFrame,
    *,
    mean_offset: float,
    sd: float,
    draws: int,
    boot_samples: int,
    seed: int,
) -> pd.DataFrame:
    market_total = frame["market_total"].to_numpy(dtype=float)
    actual_total = frame["actual_total"].to_numpy(dtype=float)
    shocks = generate_public_shocks(
        len(frame), mean_offset=mean_offset, sd=sd, draws=draws, seed=seed
    )
    served_offset = np.zeros(len(frame), dtype=float)
    served_abs_error = np.abs(served_offset - (actual_total - market_total))
    served_win_rate = win_rate_from_shocks(served_offset, actual_total, market_total, shocks)
    rows: list[dict[str, Any]] = []
    for offset in SHADE_OFFSETS:
        offset_array = np.full(len(frame), offset, dtype=float)
        abs_error = np.abs(offset_array - (actual_total - market_total))
        win_rate = win_rate_from_shocks(offset_array, actual_total, market_total, shocks)
        working = frame.copy()
        working["ae_improvement"] = served_abs_error - abs_error
        working["win_rate_improvement"] = (win_rate - served_win_rate) * 100.0
        ae_stats = paired_bootstrap(working, "ae_improvement", samples=boot_samples, seed=seed)
        win_stats = paired_bootstrap(
            working, "win_rate_improvement", samples=boot_samples, seed=seed
        )
        rows.append(
            {
                "shade": offset,
                "mean_abs_error": float(abs_error.mean()),
                "served_mean_abs_error": float(served_abs_error.mean()),
                "ae_improvement_estimate": ae_stats["estimate"],
                "ae_improvement_lower": ae_stats["lower"],
                "ae_improvement_upper": ae_stats["upper"],
                "ae_improvement_probability_positive": ae_stats["probability_positive"],
                "mean_win_rate": float(win_rate.mean()),
                "served_mean_win_rate": float(served_win_rate.mean()),
                "win_rate_improvement_pp_estimate": win_stats["estimate"],
                "win_rate_improvement_pp_lower": win_stats["lower"],
                "win_rate_improvement_pp_upper": win_stats["upper"],
                "win_rate_improvement_pp_probability_positive": win_stats["probability_positive"],
                "games": len(frame),
            }
        )
    return pd.DataFrame(rows)


def model_lean_comparison(
    frame_with_lean: pd.DataFrame,
    *,
    mean_offset: float,
    sd: float,
    draws: int,
    boot_samples: int,
    seed: int,
) -> dict[str, Any]:
    subset = frame_with_lean.dropna(subset=["predicted_residual"]).copy()
    if subset.empty:
        return {"games": 0}
    market_total = subset["market_total"].to_numpy(dtype=float)
    actual_total = subset["actual_total"].to_numpy(dtype=float)
    shocks = generate_public_shocks(
        len(subset), mean_offset=mean_offset, sd=sd, draws=draws, seed=seed
    )
    served_offset = np.zeros(len(subset), dtype=float)
    served_abs_error = np.abs(served_offset - (actual_total - market_total))
    served_win_rate = win_rate_from_shocks(served_offset, actual_total, market_total, shocks)
    model_offset = MODEL_LEAN_WEIGHT * subset["predicted_residual"].to_numpy(dtype=float)
    model_abs_error = np.abs(model_offset - (actual_total - market_total))
    model_win_rate = win_rate_from_shocks(model_offset, actual_total, market_total, shocks)
    subset["ae_improvement"] = served_abs_error - model_abs_error
    subset["win_rate_improvement"] = (model_win_rate - served_win_rate) * 100.0
    ae_stats = paired_bootstrap(subset, "ae_improvement", samples=boot_samples, seed=seed)
    win_stats = paired_bootstrap(subset, "win_rate_improvement", samples=boot_samples, seed=seed)
    return {
        "games": len(subset),
        "seasons": sorted(int(season) for season in subset["season"].unique()),
        "weight": MODEL_LEAN_WEIGHT,
        "mean_abs_error": float(model_abs_error.mean()),
        "served_mean_abs_error": float(served_abs_error.mean()),
        "ae_improvement": ae_stats,
        "mean_win_rate": float(model_win_rate.mean()),
        "served_mean_win_rate": float(served_win_rate.mean()),
        "win_rate_improvement_pp": win_stats,
    }


def sensitivity_sweep(
    frame: pd.DataFrame, *, offset: float, draws: int, boot_samples: int, seed: int
) -> pd.DataFrame:
    market_total = frame["market_total"].to_numpy(dtype=float)
    actual_total = frame["actual_total"].to_numpy(dtype=float)
    served_offset = np.zeros(len(frame), dtype=float)
    shade_offset = np.full(len(frame), offset, dtype=float)
    rows: list[dict[str, Any]] = []
    for mean_offset in SWEEP_MEAN_OFFSETS:
        for sd in SWEEP_SDS:
            shocks = generate_public_shocks(
                len(frame), mean_offset=mean_offset, sd=sd, draws=draws, seed=seed
            )
            served_win = win_rate_from_shocks(served_offset, actual_total, market_total, shocks)
            shade_win = win_rate_from_shocks(shade_offset, actual_total, market_total, shocks)
            working = frame.copy()
            working["win_rate_improvement"] = (shade_win - served_win) * 100.0
            stats = paired_bootstrap(
                working, "win_rate_improvement", samples=boot_samples, seed=seed
            )
            rows.append(
                {
                    "public_mean_offset": mean_offset,
                    "public_sd": sd,
                    "shade": offset,
                    "win_rate_improvement_pp_estimate": stats["estimate"],
                    "win_rate_improvement_pp_lower": stats["lower"],
                    "win_rate_improvement_pp_upper": stats["upper"],
                    "win_rate_improvement_pp_probability_positive": stats["probability_positive"],
                }
            )
    return pd.DataFrame(rows)


def exact_score_check(
    frame: pd.DataFrame, *, offset: float, draws: int, boot_samples: int, seed: int
) -> dict[str, Any]:
    market_total = frame["market_total"].to_numpy(dtype=float)
    margin = frame["home_expected_margin"].to_numpy(dtype=float)
    actual_home = frame["home_score"].to_numpy(dtype=float)
    actual_away = frame["away_score"].to_numpy(dtype=float)

    def guess_scores(total_guess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        home_continuous, _away_continuous = market_implied_scores(margin, total_guess)
        home_guess = np.round(home_continuous)
        away_guess = np.round(total_guess) - home_guess
        return home_guess, away_guess

    served_home, served_away = guess_scores(market_total)
    shaded_home, shaded_away = guess_scores(market_total + offset)
    served_manhattan = np.abs(served_home - actual_home) + np.abs(served_away - actual_away)
    shaded_manhattan = np.abs(shaded_home - actual_home) + np.abs(shaded_away - actual_away)

    rng = np.random.default_rng(seed)
    n_games = len(frame)
    served_win = np.empty(n_games, dtype=float)
    shaded_win = np.empty(n_games, dtype=float)
    chunk = max(1, 2_000_000 // max(draws, 1))
    for start in range(0, n_games, chunk):
        end = min(start + chunk, n_games)
        public_total = market_total[start:end, None] + rng.normal(
            PUBLIC_FIELD_MEAN_OFFSET, PUBLIC_FIELD_SD, size=(end - start, draws)
        )
        public_home = np.round((public_total + margin[start:end, None]) / 2.0)
        public_away = np.round(public_total) - public_home
        public_manhattan = np.abs(public_home - actual_home[start:end, None]) + np.abs(
            public_away - actual_away[start:end, None]
        )
        served_win[start:end] = (served_manhattan[start:end, None] < public_manhattan).mean(
            axis=1
        ) + 0.5 * (served_manhattan[start:end, None] == public_manhattan).mean(axis=1)
        shaded_win[start:end] = (shaded_manhattan[start:end, None] < public_manhattan).mean(
            axis=1
        ) + 0.5 * (shaded_manhattan[start:end, None] == public_manhattan).mean(axis=1)

    working = frame.copy()
    working["manhattan_improvement"] = served_manhattan - shaded_manhattan
    working["win_rate_improvement"] = (shaded_win - served_win) * 100.0
    manhattan_stats = paired_bootstrap(
        working, "manhattan_improvement", samples=boot_samples, seed=seed
    )
    win_stats = paired_bootstrap(working, "win_rate_improvement", samples=boot_samples, seed=seed)
    return {
        "shade": offset,
        "games": n_games,
        "served_mean_manhattan": float(served_manhattan.mean()),
        "shaded_mean_manhattan": float(shaded_manhattan.mean()),
        "manhattan_improvement": manhattan_stats,
        "served_mean_win_rate": float(served_win.mean()),
        "shaded_mean_win_rate": float(shaded_win.mean()),
        "win_rate_improvement_pp": win_stats,
    }


def choose_best_shade(comparison: pd.DataFrame, *, metric_column: str) -> float:
    ranked = comparison.sort_values(metric_column, ascending=False)
    return float(ranked.iloc[0]["shade"])


def out_of_sample_check(
    frame: pd.DataFrame, *, best_shade: float, draws: int, boot_samples: int, seed: int
) -> dict[str, Any]:
    later = frame.loc[frame["season"].ge(OOS_SPLIT_SEASON)].copy()
    comparison = shade_comparison(
        later,
        mean_offset=PUBLIC_FIELD_MEAN_OFFSET,
        sd=PUBLIC_FIELD_SD,
        draws=draws,
        boot_samples=boot_samples,
        seed=seed,
    )
    row = comparison.loc[np.isclose(comparison["shade"], best_shade)].iloc[0]
    return {
        "train_seasons": sorted(
            int(s) for s in frame.loc[frame["season"].lt(OOS_SPLIT_SEASON), "season"].unique()
        ),
        "test_seasons": sorted(int(s) for s in later["season"].unique()),
        "test_games": len(later),
        "chosen_shade": best_shade,
        "ae_improvement_estimate": float(row["ae_improvement_estimate"]),
        "ae_improvement_lower": float(row["ae_improvement_lower"]),
        "ae_improvement_upper": float(row["ae_improvement_upper"]),
        "ae_improvement_probability_positive": float(row["ae_improvement_probability_positive"]),
        "win_rate_improvement_pp_estimate": float(row["win_rate_improvement_pp_estimate"]),
        "win_rate_improvement_pp_lower": float(row["win_rate_improvement_pp_lower"]),
        "win_rate_improvement_pp_upper": float(row["win_rate_improvement_pp_upper"]),
        "win_rate_improvement_pp_probability_positive": float(
            row["win_rate_improvement_pp_probability_positive"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=PUBLIC_DRAWS)
    parser.add_argument("--boot-samples", type=int, default=BOOT_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    schedules = pd.read_parquet(newest_schedules_path(args.data_root))
    frame = build_last_game_frame(schedules)

    overall = under_rate_overall(frame, samples=args.boot_samples, seed=args.seed)
    by_season = under_rate_by_season(frame)
    by_bucket = under_rate_by_bucket(frame)

    odd_seasons = tuple(int(s) for s in by_season.loc[by_season["season"] % 2 == 1, "season"])
    even_seasons = tuple(int(s) for s in by_season.loc[by_season["season"] % 2 == 0, "season"])
    reliability = season_pair_reliability(by_season, samples=args.boot_samples, seed=args.seed)
    odd_half = half_pooled_under_rate(frame, odd_seasons, samples=args.boot_samples, seed=args.seed)
    even_half = half_pooled_under_rate(
        frame, even_seasons, samples=args.boot_samples, seed=args.seed
    )

    comparison = shade_comparison(
        frame,
        mean_offset=PUBLIC_FIELD_MEAN_OFFSET,
        sd=PUBLIC_FIELD_SD,
        draws=args.draws,
        boot_samples=args.boot_samples,
        seed=args.seed,
    )

    frame_with_lean = attach_model_lean(frame, TOTALS_PREDICTIONS_PATH)
    model_lean = model_lean_comparison(
        frame_with_lean,
        mean_offset=PUBLIC_FIELD_MEAN_OFFSET,
        sd=PUBLIC_FIELD_SD,
        draws=args.draws,
        boot_samples=args.boot_samples,
        seed=args.seed,
    )

    training_comparison = shade_comparison(
        frame.loc[frame["season"].lt(OOS_SPLIT_SEASON)],
        mean_offset=PUBLIC_FIELD_MEAN_OFFSET,
        sd=PUBLIC_FIELD_SD,
        draws=args.draws,
        boot_samples=args.boot_samples,
        seed=args.seed,
    )
    best_shade_by_win_rate = choose_best_shade(
        training_comparison, metric_column="win_rate_improvement_pp_estimate"
    )
    best_shade_by_ae = choose_best_shade(
        training_comparison, metric_column="ae_improvement_estimate"
    )
    oos_by_win_rate = out_of_sample_check(
        frame,
        best_shade=best_shade_by_win_rate,
        draws=args.draws,
        boot_samples=args.boot_samples,
        seed=args.seed,
    )
    oos_by_ae = out_of_sample_check(
        frame,
        best_shade=best_shade_by_ae,
        draws=args.draws,
        boot_samples=args.boot_samples,
        seed=args.seed,
    )
    oos_predeclared_minus1 = out_of_sample_check(
        frame, best_shade=-1.0, draws=args.draws, boot_samples=args.boot_samples, seed=args.seed
    )

    sweep = sensitivity_sweep(
        frame, offset=-1.0, draws=args.draws, boot_samples=args.boot_samples, seed=args.seed
    )

    exact_score = exact_score_check(
        frame, offset=-1.0, draws=args.draws, boot_samples=args.boot_samples, seed=args.seed
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (OUTPUT_ROOT / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_parquet(frame, out_dir / "last_games.parquet")
    atomic_parquet(comparison, out_dir / "shade_comparison_total_metric.parquet")
    atomic_parquet(sweep, out_dir / "sensitivity_sweep.parquet")
    atomic_parquet(by_season, out_dir / "under_rate_by_season.parquet")
    atomic_parquet(
        by_bucket.assign(bucket=by_bucket["bucket"].astype(str)),
        out_dir / "under_rate_by_bucket.parquet",
    )

    summary = {
        "created_at_utc": utc_now(),
        "git": git_state(REPO),
        "config": {
            "season_start": SEASON_START,
            "season_end": SEASON_END,
            "shade_offsets": SHADE_OFFSETS,
            "model_lean_weight": MODEL_LEAN_WEIGHT,
            "public_field_mean_offset": PUBLIC_FIELD_MEAN_OFFSET,
            "public_field_sd": PUBLIC_FIELD_SD,
            "draws": args.draws,
            "boot_samples": args.boot_samples,
            "seed": args.seed,
            "oos_split_season": OOS_SPLIT_SEASON,
        },
        "overall": overall,
        "by_season": by_season.to_dict(orient="records"),
        "by_bucket": by_bucket.astype(str).to_dict(orient="records"),
        "reliability_odd_even_season_pairs": reliability,
        "odd_half_pooled": odd_half,
        "even_half_pooled": even_half,
        "shade_comparison_total_metric": comparison.to_dict(orient="records"),
        "model_lean_comparison": model_lean,
        "best_shade_trained_pre_2018_by_win_rate": best_shade_by_win_rate,
        "best_shade_trained_pre_2018_by_ae": best_shade_by_ae,
        "out_of_sample_check_win_rate_selected": oos_by_win_rate,
        "out_of_sample_check_ae_selected": oos_by_ae,
        "out_of_sample_check_predeclared_shade_minus1": oos_predeclared_minus1,
        "sensitivity_sweep": sweep.to_dict(orient="records"),
        "exact_score_check": exact_score,
        "totals_predictions_source": {
            "path": str(TOTALS_PREDICTIONS_PATH),
            "exists": TOTALS_PREDICTIONS_PATH.is_file(),
            "sha256": sha256_file(TOTALS_PREDICTIONS_PATH)
            if TOTALS_PREDICTIONS_PATH.is_file()
            else None,
        },
    }
    atomic_json(summary, out_dir / "summary.json")

    print(f"artifact: {out_dir}")
    print(f"games: {len(frame)}  seasons: {frame['season'].nunique()}")
    under_ci = f"[{overall['lower']:.4f}, {overall['upper']:.4f}]"
    print(f"overall under_rate: {overall['under_rate']:.4f}  95% {under_ci}")
    print("by_season:")
    print(by_season.to_string(index=False))
    print("by_bucket:")
    print(by_bucket.to_string(index=False))
    print("reliability (odd/even season pairs):", reliability)
    print("odd_half:", odd_half)
    print("even_half:", even_half)
    print("shade_comparison_total_metric:")
    print(comparison.to_string(index=False))
    print("model_lean_comparison:", model_lean)
    print("best_shade_trained_pre_2018_by_win_rate:", best_shade_by_win_rate)
    print("best_shade_trained_pre_2018_by_ae:", best_shade_by_ae)
    print("out_of_sample_check_win_rate_selected:", oos_by_win_rate)
    print("out_of_sample_check_ae_selected:", oos_by_ae)
    print("out_of_sample_check_predeclared_shade_minus1:", oos_predeclared_minus1)
    print("sensitivity_sweep:")
    print(sweep.to_string(index=False))
    print("exact_score_check:", exact_score)


if __name__ == "__main__":
    main()
