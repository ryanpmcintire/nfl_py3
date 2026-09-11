from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS
from nfl_ats.clv import pick_correct
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.io import atomic_json, run_id
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]

CFB_RANKED_QUANTILE = 0.80
CFB_TOP10_QUANTILE = 0.92
RIVALRY_MIN_CONSECUTIVE_SEASONS = 8
NFL_HOME_FAVORITE_THRESHOLD = 7.0
NFL_TOP_QUARTILE = 0.75
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260910


def block_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    block_columns: list[str],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:

    valid = frame.dropna(subset=[value_column]).reset_index(drop=True)
    if valid.empty:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
            "n_games": 0,
            "n_blocks": 0,
        }
    blocks = list(valid.groupby(block_columns, sort=False, dropna=False).indices.values())
    values = valid[value_column].to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[index] for index in selected])
        draws[sample_index] = float(values[positions].mean())
    tail = 0.025
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "n_games": len(valid),
        "n_blocks": len(blocks),
    }


def rivalry_pairs(games: pd.DataFrame, min_consecutive: int) -> set[frozenset[str]]:

    frame = games.loc[:, ["season", "home_team", "away_team"]].dropna()
    frame = frame.assign(
        pair=frame.apply(lambda row: frozenset((row["home_team"], row["away_team"])), axis=1)
    )
    rivalries: set[frozenset[str]] = set()
    for pair, group in frame.groupby("pair", sort=False):
        seasons = sorted({int(value) for value in group["season"]})
        if len(seasons) < min_consecutive:
            continue
        best_run = 1
        current_run = 1
        for index in range(1, len(seasons)):
            if seasons[index] == seasons[index - 1] + 1:
                current_run += 1
            else:
                current_run = 1
            best_run = max(best_run, current_run)
        if best_run >= min_consecutive:
            rivalries.add(cast(frozenset[str], pair))
    return rivalries


def build_long_table(base: pd.DataFrame, signed_margin_column: str) -> pd.DataFrame:

    sides = []
    for is_home, team_column, opponent_column, sign in (
        (True, "home_team", "away_team", 1.0),
        (False, "away_team", "home_team", -1.0),
    ):
        side = base.loc[
            :, ["game_id", "season", "week", team_column, opponent_column, signed_margin_column]
        ].rename(columns={team_column: "team", opponent_column: "opponent"})
        side["is_home"] = is_home
        side["signed_margin"] = sign * pd.to_numeric(base[signed_margin_column], errors="coerce")
        sides.append(side.drop(columns=[signed_margin_column]))
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.sort_values(["team", "season", "week", "game_id"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["rating"] = grouped["signed_margin"].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    long_df["next_game_id"] = grouped["game_id"].shift(-1)
    long_df["next_opponent"] = grouped["opponent"].shift(-1)
    return long_df


def attach_opponent_next_rating(long_df: pd.DataFrame) -> pd.DataFrame:

    lookup = long_df.loc[:, ["team", "game_id", "rating"]].rename(
        columns={
            "team": "next_opponent",
            "game_id": "next_game_id",
            "rating": "next_opponent_rating",
        }
    )
    return long_df.merge(lookup, on=["next_opponent", "next_game_id"], how="left")


def score_cfb() -> dict[str, object]:

    cfb = pd.read_parquet(REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet")
    cfb = cfb.loc[cfb["spread_line"].notna() & cfb["result"].notna()].copy()
    cfb["season"] = pd.to_numeric(cfb["season"], errors="raise").astype(int)
    cfb["week"] = pd.to_numeric(cfb["week"], errors="raise").astype(int)

    long_df = build_long_table(cfb, "spread_line")
    rivalries = rivalry_pairs(cfb, RIVALRY_MIN_CONSECUTIVE_SEASONS)
    long_df["is_rivalry_next"] = long_df.apply(
        lambda row: (
            pd.notna(row["next_opponent"])
            and frozenset((row["team"], row["next_opponent"])) in rivalries
        ),
        axis=1,
    )
    long_df["rank_threshold_ranked"] = long_df.groupby(["season", "week"])["rating"].transform(
        lambda series: series.quantile(CFB_RANKED_QUANTILE)
    )
    long_df["rank_threshold_top10"] = long_df.groupby(["season", "week"])["rating"].transform(
        lambda series: series.quantile(CFB_TOP10_QUANTILE)
    )
    long_df = attach_opponent_next_rating(long_df)

    home_rows = long_df.loc[
        long_df["is_home"],
        [
            "game_id",
            "rating",
            "rank_threshold_ranked",
            "rank_threshold_top10",
            "next_opponent",
            "next_opponent_rating",
            "is_rivalry_next",
        ],
    ].rename(
        columns={
            "rating": "home_rating",
            "next_opponent": "home_next_opponent",
            "next_opponent_rating": "home_next_opponent_rating",
            "is_rivalry_next": "home_is_rivalry_next",
        }
    )
    away_rows = long_df.loc[~long_df["is_home"], ["game_id", "rating"]].rename(
        columns={"rating": "away_rating"}
    )
    game_level = cfb.merge(home_rows, on="game_id", how="inner").merge(
        away_rows, on="game_id", how="inner"
    )

    eligible = (
        game_level["home_rating"].notna()
        & game_level["away_rating"].notna()
        & (game_level["home_rating"] >= game_level["rank_threshold_ranked"])
        & (game_level["away_rating"] < game_level["rank_threshold_ranked"])
    )
    has_next = game_level["home_next_opponent"].notna()
    top10_next = has_next & (
        game_level["home_next_opponent_rating"] >= game_level["rank_threshold_top10"]
    )
    marquee = has_next & (game_level["home_is_rivalry_next"] | top10_next)
    game_level["cfb_lead44_sandwich_flag"] = np.where(eligible, marquee.astype(float), np.nan)

    scored = game_level.loc[eligible & game_level["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    home_cover = pd.to_numeric(scored["home_cover"], errors="coerce")
    flag = scored["cfb_lead44_sandwich_flag"]
    value = np.where(flag.eq(1.0), 100.0 * (1.0 - 2.0 * home_cover), 0.0)
    scored["delta_value"] = np.where(home_cover.isna(), np.nan, value)
    scored["home_cover_pct"] = home_cover * 100.0

    pooled_week = block_bootstrap_mean(scored, "delta_value", ["season", "week"])
    pooled_season = block_bootstrap_mean(scored, "delta_value", ["season"])
    treated = scored.loc[scored["cfb_lead44_sandwich_flag"].eq(1.0)]
    control = scored.loc[scored["cfb_lead44_sandwich_flag"].eq(0.0)]
    treated_cover = block_bootstrap_mean(treated, "home_cover_pct", ["season", "week"])
    control_cover = block_bootstrap_mean(control, "home_cover_pct", ["season", "week"])

    odd_seasons = scored.loc[scored["season"] % 2 == 1]
    even_seasons = scored.loc[scored["season"] % 2 == 0]
    reliability_odd = block_bootstrap_mean(odd_seasons, "delta_value", ["season", "week"])
    reliability_even = block_bootstrap_mean(even_seasons, "delta_value", ["season", "week"])

    return {
        "league": "cfb",
        "eligible_population_games": int(eligible.sum()),
        "scored_population_games": len(scored),
        "flagged_games": int((scored["cfb_lead44_sandwich_flag"] == 1.0).sum()),
        "matched_control_games": int((scored["cfb_lead44_sandwich_flag"] == 0.0).sum()),
        "rivalry_pairs_found": len(rivalries),
        "production_screen_delta_accuracy_points": {
            "week_blocked": pooled_week,
            "season_blocked": pooled_season,
        },
        "cover_rate_flagged": treated_cover,
        "cover_rate_control": control_cover,
        "reliability_odd_even_season_split": {
            "odd_seasons": reliability_odd,
            "even_seasons": reliability_even,
        },
    }


def score_nfl() -> dict[str, object]:

    per_game = pd.read_parquet(
        REPO_ROOT / "artifacts" / "opener_evaluation" / "20260910T211255Z" / "per_game.parquet"
    )
    per_game = per_game.copy()
    parts = per_game["game_id"].astype(str).str.split("_")
    per_game["home_team"] = parts.str[-1]
    per_game["away_team"] = parts.str[-2]
    per_game["season"] = pd.to_numeric(per_game["season"], errors="raise").astype(int)
    per_game["week"] = pd.to_numeric(per_game["week"], errors="raise").astype(int)

    schedule = pd.read_parquet(latest_schedules_snapshot(REPO_ROOT))
    schedule = schedule.loc[schedule["game_type"].astype(str) == "REG", ["game_id", "div_game"]]
    per_game = per_game.merge(schedule, on="game_id", how="left")
    per_game["div_game"] = pd.to_numeric(per_game["div_game"], errors="coerce")

    margin_vs_open = pd.to_numeric(per_game["margin_vs_open"], errors="coerce")
    per_game["home_cover"] = np.where(
        margin_vs_open > 0, 1.0, np.where(margin_vs_open < 0, 0.0, np.nan)
    )

    long_df = build_long_table(per_game, "tue_open_home_spread")
    long_df["div_game"] = (
        per_game.set_index("game_id")["div_game"].reindex(long_df["game_id"]).to_numpy()
    )
    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["next_div_game"] = grouped["div_game"].shift(-1)
    long_df["rank_threshold_top_quartile"] = long_df.groupby(["season", "week"])[
        "rating"
    ].transform(lambda series: series.quantile(NFL_TOP_QUARTILE))
    long_df = attach_opponent_next_rating(long_df)

    home_rows = long_df.loc[
        long_df["is_home"],
        [
            "game_id",
            "rank_threshold_top_quartile",
            "next_opponent",
            "next_opponent_rating",
            "next_div_game",
        ],
    ].rename(
        columns={
            "next_opponent": "home_next_opponent",
            "next_opponent_rating": "home_next_opponent_rating",
            "next_div_game": "home_next_div_game",
        }
    )
    game_level = per_game.merge(home_rows, on="game_id", how="inner")

    eligible = pd.to_numeric(game_level["tue_open_home_spread"], errors="coerce").ge(
        NFL_HOME_FAVORITE_THRESHOLD
    )
    has_next = game_level["home_next_opponent"].notna()
    top_quartile_next = has_next & (
        game_level["home_next_opponent_rating"] >= game_level["rank_threshold_top_quartile"]
    )
    division_next = has_next & game_level["home_next_div_game"].eq(1.0)
    marquee = has_next & (division_next | top_quartile_next)
    game_level["nfl_lead44_eligible_flag"] = np.where(eligible, marquee.astype(float), np.nan)
    game_level["nfl_lead44_override_flag"] = np.where(eligible & marquee, 1.0, 0.0)

    baseline_pick_home = game_level["pick_home_at_open_probability_rule"].astype(bool)
    candidate_pick_home = np.where(
        game_level["nfl_lead44_override_flag"] == 1.0, False, baseline_pick_home
    )
    baseline_correct = pd.to_numeric(
        game_level["correct_at_open_probability_rule"], errors="coerce"
    )
    candidate_correct = pick_correct(pd.Series(candidate_pick_home), margin_vs_open)
    game_level["screen_delta_value"] = 100.0 * (candidate_correct - baseline_correct)

    window_population = game_level.loc[game_level["season"].isin((2020, 2021))]
    full_population = game_level

    def per_season_breakdown(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        rows: dict[str, dict[str, float | int]] = {}
        for season, group in frame.groupby("season"):
            rows[str(season)] = block_bootstrap_mean(group, "screen_delta_value", ["week"])
        return rows

    window_week = block_bootstrap_mean(window_population, "screen_delta_value", ["season", "week"])
    window_season = block_bootstrap_mean(window_population, "screen_delta_value", ["season"])
    full_week = block_bootstrap_mean(full_population, "screen_delta_value", ["season", "week"])
    full_season = block_bootstrap_mean(full_population, "screen_delta_value", ["season"])

    eligible_window = window_population.loc[window_population["nfl_lead44_eligible_flag"].notna()]
    eligible_full = full_population.loc[full_population["nfl_lead44_eligible_flag"].notna()]
    flagged_window = eligible_window.loc[eligible_window["nfl_lead44_eligible_flag"].eq(1.0)]
    control_window = eligible_window.loc[eligible_window["nfl_lead44_eligible_flag"].eq(0.0)]
    flagged_full = eligible_full.loc[eligible_full["nfl_lead44_eligible_flag"].eq(1.0)]
    control_full = eligible_full.loc[eligible_full["nfl_lead44_eligible_flag"].eq(0.0)]

    def cover_rate(frame: pd.DataFrame) -> dict[str, float | int]:
        working = frame.copy()
        working["home_cover_pct"] = working["home_cover"] * 100.0
        return block_bootstrap_mean(working, "home_cover_pct", ["season", "week"])

    odd_seasons = full_population.loc[full_population["season"] % 2 == 1]
    even_seasons = full_population.loc[full_population["season"] % 2 == 0]
    reliability_odd = block_bootstrap_mean(odd_seasons, "screen_delta_value", ["season", "week"])
    reliability_even = block_bootstrap_mean(even_seasons, "screen_delta_value", ["season", "week"])

    return {
        "league": "nfl",
        "eligible_population_games_window_2020_2021": len(eligible_window),
        "eligible_population_games_full_2020_2025": len(eligible_full),
        "flagged_games_window": len(flagged_window),
        "flagged_games_full": len(flagged_full),
        "registered_window_production_screen": {
            "week_blocked": window_week,
            "season_blocked": window_season,
            "per_season": per_season_breakdown(window_population),
            "picks_changed": int(window_population["nfl_lead44_override_flag"].sum()),
            "games": len(window_population),
        },
        "disclosed_full_archive_production_screen": {
            "week_blocked": full_week,
            "season_blocked": full_season,
            "per_season": per_season_breakdown(full_population),
            "picks_changed": int(full_population["nfl_lead44_override_flag"].sum()),
            "games": len(full_population),
        },
        "cover_rate_flagged_window": cover_rate(flagged_window),
        "cover_rate_control_window": cover_rate(control_window),
        "cover_rate_flagged_full": cover_rate(flagged_full),
        "cover_rate_control_full": cover_rate(control_full),
        "reliability_odd_even_season_split_full_archive": {
            "odd_seasons": reliability_odd,
            "even_seasons": reliability_even,
        },
    }


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["cfb", "nfl", "both"], default="both")
    args = parser.parse_args()

    result: dict[str, object] = {}
    if args.league in ("cfb", "both"):
        result["cfb"] = score_cfb()
    if args.league in ("nfl", "both"):
        result["nfl"] = score_nfl()

    output_dir = REPO_ROOT / "artifacts" / "lookahead_fade_screen" / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output_dir / "results.json")
    print(json.dumps(result, indent=2, default=str))
    print(str(output_dir))


if __name__ == "__main__":
    main()
