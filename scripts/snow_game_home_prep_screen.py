from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
PRECIP_PROB_THRESHOLD_PCT = 50.0
FREEZING_TEMP_F = 32.0

GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
FORECAST_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "pool_decision_2009_2025" / "forecasts.parquet"
)
MARKET_ROOT = REPO / "data" / "market" / "raw"
ARTIFACTS_ROOT = REPO / "artifacts"
OUT_DIR = REPO / "artifacts" / "snow_game_home_prep"

FAMILY = "snow_game_home_prep_on_production"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260911


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise SystemExit("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def load_snow_table() -> pd.DataFrame:
    feat = pd.read_parquet(
        GAME_FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_cover",
            "spread_line",
            "result",
        ],
    )
    reg = feat.loc[feat["game_type"].eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)

    schedules_path = _latest_schedules()
    sched = pd.read_parquet(schedules_path, columns=["game_id", "roof"])
    sched["game_id"] = sched["game_id"].astype(str)
    reg = reg.merge(sched, on="game_id", how="left", validate="one_to_one")

    archive = pd.read_parquet(
        FORECAST_ARCHIVE,
        columns=[
            "game_id",
            "fetch_status",
            "forecast_temp_f",
            "forecast_precip_prob_pct",
            "actual_temp_f",
        ],
    )
    archive["game_id"] = archive["game_id"].astype(str)
    table = reg.merge(archive, on="game_id", how="inner", validate="one_to_one")
    table = table.loc[table["fetch_status"].eq("ok")].reset_index(drop=True).copy()

    table["outdoor"] = table["roof"].isin(OUTDOOR_ROOFS)
    table["forecast_temp_f"] = pd.to_numeric(table["forecast_temp_f"], errors="coerce")
    table["forecast_precip_prob_pct"] = pd.to_numeric(
        table["forecast_precip_prob_pct"], errors="coerce"
    )
    table["actual_temp_f"] = pd.to_numeric(table["actual_temp_f"], errors="coerce")

    table["snow_flag"] = (
        table["outdoor"]
        & table["forecast_precip_prob_pct"].ge(PRECIP_PROB_THRESHOLD_PCT)
        & table["forecast_temp_f"].le(FREEZING_TEMP_F)
    )
    table["cold_outdoor_other_flag"] = (
        table["outdoor"] & table["forecast_temp_f"].le(FREEZING_TEMP_F) & ~table["snow_flag"]
    )
    table["snow_flag_actual_upper_bound"] = (
        table["outdoor"]
        & table["forecast_precip_prob_pct"].ge(PRECIP_PROB_THRESHOLD_PCT)
        & table["actual_temp_f"].le(FREEZING_TEMP_F)
    )
    table["snow_flag"] = table["snow_flag"].fillna(False)
    table["cold_outdoor_other_flag"] = table["cold_outdoor_other_flag"].fillna(False)
    table["snow_flag_actual_upper_bound"] = table["snow_flag_actual_upper_bound"].fillna(False)
    table["week_block"] = table["season"] * 100 + table["week"]
    return table


def _block_bootstrap_gap(
    frame: pd.DataFrame,
    outcome_column: str,
    group_a_mask: pd.Series,
    *,
    block_column: str,
    samples: int,
    seed: int,
) -> dict:
    valid = frame[outcome_column].notna()
    working = frame.loc[valid].reset_index(drop=True)
    a_mask = group_a_mask.loc[valid].reset_index(drop=True).to_numpy(dtype=bool)
    outcome = working[outcome_column].to_numpy(dtype=float)
    blocks = list(working.groupby(block_column, sort=False).indices.values())

    def _metric(positions: np.ndarray) -> float:
        sub_outcome = outcome[positions]
        sub_a = a_mask[positions]
        if sub_a.sum() == 0 or (~sub_a).sum() == 0:
            return float("nan")
        return float(sub_outcome[sub_a].mean() - sub_outcome[~sub_a].mean())

    point = _metric(np.arange(len(working)))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[j] for j in selected])
        draws[i] = _metric(positions)
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        lower = upper = prob_pos = float("nan")
    else:
        lower, upper = (float(x) for x in np.quantile(finite, [0.025, 0.975]))
        prob_pos = float(probability_positive_from_draws(finite))
    return {
        "point_estimate": point,
        "n_group_a": int(a_mask.sum()),
        "n_group_b": int((~a_mask).sum()),
        "ci95": [lower, upper],
        "probability_positive": prob_pos,
        "samples": samples,
        "nan_draws": int(np.isnan(draws).sum()),
        "block_column": block_column,
    }


def cmd_screen(args: argparse.Namespace) -> None:
    table = load_snow_table()

    by_season = (
        table.groupby("season")
        .agg(
            n_games=("game_id", "size"),
            n_snow_deadline=("snow_flag", "sum"),
            n_snow_actual_upper_bound=("snow_flag_actual_upper_bound", "sum"),
            n_cold_outdoor_other=("cold_outdoor_other_flag", "sum"),
        )
        .reset_index()
    )

    def _rate(mask: pd.Series) -> dict:
        sub = table.loc[mask, "home_cover"].dropna()
        return {
            "n": int(mask.sum()),
            "n_graded": len(sub),
            "cover_rate": float(sub.mean()) if len(sub) else float("nan"),
        }

    population_rate = _rate(pd.Series(True, index=table.index))
    snow_rate = _rate(table["snow_flag"])
    cold_other_rate = _rate(table["cold_outdoor_other_flag"])
    upper_bound_rate = _rate(table["snow_flag_actual_upper_bound"])

    gap_vs_population = _block_bootstrap_gap(
        table,
        "home_cover",
        table["snow_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    gap_vs_population_season_blocked = _block_bootstrap_gap(
        table,
        "home_cover",
        table["snow_flag"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    cold_and_snow = table.loc[table["snow_flag"] | table["cold_outdoor_other_flag"]].reset_index(
        drop=True
    )
    gap_vs_cold_other = _block_bootstrap_gap(
        cold_and_snow,
        "home_cover",
        cold_and_snow["snow_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    gap_vs_cold_other_season_blocked = _block_bootstrap_gap(
        cold_and_snow,
        "home_cover",
        cold_and_snow["snow_flag"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    market_rows = []
    snow_games = table.loc[table["snow_flag"]]
    for row in snow_games.itertuples(index=False):
        controls = table.loc[
            (table["home_team"] == row.home_team)
            & (table["season"] == row.season)
            & (~table["snow_flag"])
            & (table["game_id"] != row.game_id)
        ]
        if controls.empty:
            continue
        market_rows.append(
            {
                "game_id": row.game_id,
                "season": int(row.season),
                "home_team": row.home_team,
                "snow_spread_line": float(row.spread_line),
                "control_mean_spread_line": float(controls["spread_line"].mean()),
                "n_controls": len(controls),
                "diff": float(row.spread_line) - float(controls["spread_line"].mean()),
            }
        )
    market_frame = pd.DataFrame(market_rows)
    if not market_frame.empty:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        blocks = list(market_frame.groupby("season", sort=False).indices.values())
        point = float(market_frame["diff"].mean())
        draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for i in range(BOOTSTRAP_SAMPLES):
            selected = rng.integers(0, len(blocks), size=len(blocks))
            positions = np.concatenate([blocks[j] for j in selected])
            draws[i] = float(market_frame["diff"].to_numpy()[positions].mean())
        lower, upper = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
        market_pricing = {
            "n_snow_games_with_controls": len(market_frame),
            "n_snow_games_no_controls": int(len(snow_games) - len(market_frame)),
            "mean_diff_spread_points": point,
            "ci95": [lower, upper],
            "probability_market_already_shades_home": float(
                probability_positive_from_draws(-draws)
            ),
            "samples": BOOTSTRAP_SAMPLES,
            "sign_convention": (
                "home-perspective spread_line; negative diff means the snow game's line "
                "favoured the home team MORE than that same team's other home games that "
                "season, i.e. the market already shaded toward home"
            ),
        }
    else:
        market_pricing = {
            "n_snow_games_with_controls": 0,
            "note": "no snow game had an eligible control",
        }

    odd = table.loc[table["season"] % 2 == 1].reset_index(drop=True)
    even = table.loc[table["season"] % 2 == 0].reset_index(drop=True)

    def _half(frame: pd.DataFrame) -> dict:
        return _block_bootstrap_gap(
            frame,
            "home_cover",
            frame["snow_flag"],
            block_column="week_block",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )

    reliability = {
        "odd_seasons": _half(odd),
        "even_seasons": _half(even),
        "note": (
            "snow_flag is a per-game situational condition (this week's forecast at an "
            "outdoor venue), not a persistent per-team trait with a year-over-year value; "
            "a formal split-half correlation coefficient does not apply the way it would to "
            "a team-level trait. Reported instead: the home-cover gap measured independently "
            "on odd-numbered and even-numbered seasons."
        ),
    }

    result = {
        "thresholds": {
            "precip_prob_threshold_pct": PRECIP_PROB_THRESHOLD_PCT,
            "freezing_temp_f": FREEZING_TEMP_F,
            "outdoor_roofs": sorted(OUTDOOR_ROOFS),
            "forecast_source": (
                "data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet"
            ),
        },
        "population": {
            "n_games_domestic_reg_2009_2025": len(table),
        },
        "counts_by_season": by_season.to_dict(orient="records"),
        "cover_rates": {
            "population": population_rate,
            "snow_games_deadline_visible": snow_rate,
            "cold_outdoor_other": cold_other_rate,
            "snow_games_actual_upper_bound": upper_bound_rate,
        },
        "home_cover_gap_snow_vs_population": {
            "week_blocked_primary": gap_vs_population,
            "season_blocked_secondary": gap_vs_population_season_blocked,
        },
        "home_cover_gap_snow_vs_cold_outdoor_other": {
            "week_blocked_primary": gap_vs_cold_other,
            "season_blocked_secondary": gap_vs_cold_other_season_blocked,
        },
        "market_pricing_check": market_pricing,
        "split_half_reliability_odd_even_seasons": reliability,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_results.json").write_text(json.dumps(result, indent=2, default=str))
    table.to_parquet(OUT_DIR / "snow_game_table.parquet", index=False)
    if not market_frame.empty:
        market_frame.to_csv(OUT_DIR / "market_pricing_pairs.csv", index=False)
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'screen_results.json'}")


def _accuracy_metric(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def _summarize(frame: pd.DataFrame, samples: int, seed: int) -> dict:
    point = _accuracy_metric(frame)
    week = week_blocked_bootstrap(frame, _accuracy_metric, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(
        frame, _accuracy_metric, block="season", samples=samples, seed=seed
    )
    w = week.loc[week["metric"].eq("accuracy_points")].iloc[0]
    s = season.loc[season["metric"].eq("accuracy_points")].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": len(frame),
        "n_weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(frame["season"].nunique()),
    }


def cmd_production(args: argparse.Namespace) -> None:
    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, FAMILY)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise SystemExit("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(s) for s in window["season"].unique()))
    scoped = pd.concat([training, window], ignore_index=True)

    config = resolve_active_model_config(ARTIFACTS_ROOT)
    scored = opener_pick_evaluation(
        MARKET_ROOT,
        scoped,
        active_model_config=config,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    scored = scored.loc[scored["season"].astype(int).isin(seasons)].reset_index(drop=True)

    snow_table = load_snow_table()[["game_id", "snow_flag"]]
    scored["game_id"] = scored["game_id"].astype(str)
    scored = scored.merge(snow_table, on="game_id", how="left")
    scored["snow_flag"] = scored["snow_flag"].fillna(False)

    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    scored["candidate_pick_home"] = np.where(
        scored["snow_flag"] & ~scored["baseline_pick_home"], True, scored["baseline_pick_home"]
    )
    scored["baseline_correct"] = pick_correct(
        scored["baseline_pick_home"], scored["margin_vs_open"]
    )
    scored["candidate_correct"] = pick_correct(
        pd.Series(scored["candidate_pick_home"], index=scored.index), scored["margin_vs_open"]
    )

    graded = scored.dropna(subset=["baseline_correct", "candidate_correct"])
    picks_changed = int((scored["baseline_pick_home"] != scored["candidate_pick_home"]).sum())
    picks_changed_graded = int(
        (graded["baseline_pick_home"] != graded["candidate_pick_home"]).sum()
    )

    summary = _summarize(graded, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    result = {
        "family": FAMILY,
        "window_seasons": list(seasons),
        "grade": "opener",
        "active_model_id": config.get("model_id"),
        "n_snow_games_in_window": int(scored["snow_flag"].sum()),
        "n_snow_games_graded": int(graded["snow_flag"].sum()),
        "picks_changed": picks_changed,
        "picks_changed_graded": picks_changed_graded,
        "summary": summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT_DIR / "production_paired.parquet", index=False)
    (OUT_DIR / "production_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'production_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAD-38 snow-game home-preparation screen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("screen", help="counts, cover rates, market pricing, split-half reliability")
    sub.add_parser(
        "production", help="production stack: home tilt in snow games on the opener grade"
    )
    args = parser.parse_args()
    if args.command == "screen":
        cmd_screen(args)
    elif args.command == "production":
        cmd_production(args)


if __name__ == "__main__":
    main()
