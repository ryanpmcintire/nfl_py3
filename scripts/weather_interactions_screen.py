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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from snow_game_home_prep_screen import (  # noqa: E402
    OUTDOOR_ROOFS,
    _latest_schedules,
    _summarize,
)

from nfl_ats.clv import (  # noqa: E402
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.experiment_runner import (  # noqa: E402
    _base_team_game_table,
    _lag_and_quartile,
    _team_season_pass_rate,
    _year_over_year_reliability,
)
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402
from nfl_ats.surface_switch_tilt_overlay import surface_switch_flag_by_game  # noqa: E402

GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
POOL_DECISION_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "pool_decision_2009_2025" / "forecasts.parquet"
)
KICKOFF_NEAREST_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "kickoff_nearest_2009_2025" / "forecasts.parquet"
)
TEAM_SEASON_STYLE = REPO / "data" / "pbp" / "team_style" / "team_season_style.parquet"
MARKET_ROOT = REPO / "data" / "market" / "raw"
ARTIFACTS_ROOT = REPO / "artifacts"
OUT_DIR = REPO / "artifacts" / "weather_interactions"

WIND_THRESHOLD_MPH = 15.0
HEAT_THRESHOLD_F = 85.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260910

CUTOFFS = {
    "pool_decision": POOL_DECISION_ARCHIVE,
    "kickoff_nearest": KICKOFF_NEAREST_ARCHIVE,
}


def _canonical_team(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _team_season_fg_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["team"] = _canonical_team(plays["posteam"])
    plays["is_fg_attempt"] = plays["play_type"].eq("field_goal").astype(float)
    grouped = plays.groupby(["season", "team"]).agg(
        fg_attempts=("is_fg_attempt", "sum"),
        games=("game_id", "nunique"),
    )
    grouped["rate"] = grouped["fg_attempts"] / grouped["games"]
    return grouped.reset_index()


def _team_season_pace_rate() -> pd.DataFrame:
    style = pd.read_parquet(TEAM_SEASON_STYLE, columns=["season", "team", "seconds_per_play_pace"])
    style = style.rename(columns={"seconds_per_play_pace": "rate"})
    style["team"] = _canonical_team(style["team"])
    style["season"] = style["season"].astype(int)
    return style[["season", "team", "rate"]]


def load_traits() -> dict:
    snapshot = latest_pbp_snapshot(REPO / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)

    pass_rate = _team_season_pass_rate(pbp)
    pass_reliability, pass_pairs = _year_over_year_reliability(pass_rate)
    pass_lag = _lag_and_quartile(pass_rate)

    fg_rate = _team_season_fg_rate(pbp)
    fg_reliability, fg_pairs = _year_over_year_reliability(fg_rate)
    fg_lag = _lag_and_quartile(fg_rate)

    pace_rate = _team_season_pace_rate()
    pace_reliability, pace_pairs = _year_over_year_reliability(pace_rate)
    pace_lag = _lag_and_quartile(pace_rate)

    return {
        "pass": {
            "lag": pass_lag,
            "reliability": pass_reliability,
            "pairs": pass_pairs,
            "snapshot_id": snapshot.snapshot_id,
            "note": "pass_attempt/(pass_attempt+rush_attempt), posteam-grouped, PRIOR season",
        },
        "fg": {
            "lag": fg_lag,
            "reliability": fg_reliability,
            "pairs": fg_pairs,
            "snapshot_id": snapshot.snapshot_id,
            "note": (
                "field_goal ATTEMPTS per team-game (no field_goal_result column exists in "
                "this repo's PBP snapshot; attempt-based proxy, disclosed), PRIOR season"
            ),
        },
        "pace": {
            "lag": pace_lag,
            "reliability": pace_reliability,
            "pairs": pace_pairs,
            "snapshot_id": "data/pbp/team_style/team_season_style.parquet",
            "note": "seconds_per_play_pace (lower=faster), PRIOR season",
        },
    }


def load_base_long_table() -> pd.DataFrame:
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

    sched = pd.read_parquet(_latest_schedules(), columns=["game_id", "roof", "surface"])
    sched["game_id"] = sched["game_id"].astype(str)
    reg = reg.merge(sched, on="game_id", how="left", validate="one_to_one")

    long_df = _base_team_game_table(reg)
    roof_map = reg[["game_id", "roof"]].drop_duplicates("game_id")
    long_df = long_df.merge(roof_map, on="game_id", how="left")
    long_df["outdoor"] = long_df["roof"].isin(OUTDOOR_ROOFS)
    return long_df


def attach_forecast(long_df: pd.DataFrame, archive_path: Path) -> pd.DataFrame:
    archive = pd.read_parquet(
        archive_path,
        columns=["game_id", "fetch_status", "forecast_temp_f", "forecast_wind_mph"],
    )
    archive["game_id"] = archive["game_id"].astype(str)
    archive = archive.loc[archive["fetch_status"].eq("ok")].copy()
    merged = long_df.merge(
        archive[["game_id", "forecast_temp_f", "forecast_wind_mph"]],
        on="game_id",
        how="inner",
    )
    return merged


def attach_trait(long_df: pd.DataFrame, trait_lag: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(
        trait_lag[["team", "season", "quartile"]].rename(columns={"quartile": "_trait_quartile"}),
        on=["team", "season"],
        how="left",
    )
    return merged


def _block_bootstrap_signed_gap(
    frame: pd.DataFrame,
    outcome_column: str,
    flag_mask: pd.Series,
    *,
    block_column: str,
    samples: int,
    seed: int,
    sign: int,
) -> dict:
    valid = frame[outcome_column].notna()
    working = frame.loc[valid].reset_index(drop=True)
    a_mask = flag_mask.loc[valid].reset_index(drop=True).to_numpy(dtype=bool)
    outcome = working[outcome_column].to_numpy(dtype=float)
    blocks = list(working.groupby(block_column, sort=False).indices.values())

    def _metric(positions: np.ndarray) -> float:
        sub_outcome = outcome[positions]
        sub_a = a_mask[positions]
        if sub_a.sum() == 0 or (~sub_a).sum() == 0:
            return float("nan")
        raw = float(sub_outcome[sub_a].mean() - sub_outcome[~sub_a].mean())
        return sign * raw

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
        "point_estimate_accuracy_points": 100.0 * point,
        "n_group_a_flag": int(a_mask.sum()),
        "n_group_b_complement": int((~a_mask).sum()),
        "ci95_accuracy_points": [100.0 * lower, 100.0 * upper],
        "probability_positive": prob_pos,
        "samples": samples,
        "nan_draws": int(np.isnan(draws).sum()),
        "block_column": block_column,
        "sign_convention": (
            "positive means the predicted (fade) direction is confirmed: "
            f"reported effect = {sign:+d} * 100 * (mean(outcome|flag)-mean(outcome|not flag))"
        ),
    }


CELL_SPECS = {
    "wind_pass_heavy": {
        "trait_key": "pass",
        "weather_column": "forecast_wind_mph",
        "threshold": WIND_THRESHOLD_MPH,
        "quartile": 4,
        "family": "weather_interactions_wind_pass_heavy_fade",
        "production_family": "weather_interactions_wind_pass_heavy_on_production",
    },
    "wind_fg_reliant": {
        "trait_key": "fg",
        "weather_column": "forecast_wind_mph",
        "threshold": WIND_THRESHOLD_MPH,
        "quartile": 4,
        "family": "weather_interactions_wind_fg_reliant_fade",
        "production_family": "weather_interactions_wind_fg_reliant_on_production",
    },
    "heat_pace": {
        "trait_key": "pace",
        "weather_column": "forecast_temp_f",
        "threshold": HEAT_THRESHOLD_F,
        "quartile": 1,
        "family": "weather_interactions_heat_pace_fade",
        "production_family": "weather_interactions_heat_pace_on_production",
    },
}


def build_cell_frame(
    cell_key: str, base_long: pd.DataFrame, traits: dict, cutoff_path: Path
) -> pd.DataFrame:
    spec = CELL_SPECS[cell_key]
    merged = attach_forecast(base_long, cutoff_path)
    merged = attach_trait(merged, traits[spec["trait_key"]]["lag"])
    weather_ok = merged["outdoor"] & (merged[spec["weather_column"]] >= spec["threshold"])
    top_quartile = merged["_trait_quartile"] == spec["quartile"]
    merged["_flag"] = (weather_ok & top_quartile).fillna(False)
    merged["_narrow_eligible"] = weather_ok & merged["_trait_quartile"].notna()
    merged["_narrow_control"] = (merged["_narrow_eligible"] & ~top_quartile).fillna(False)
    return merged


def _half(frame: pd.DataFrame) -> pd.DataFrame:
    odd = frame.loc[frame["season"] % 2 == 1].reset_index(drop=True)
    even = frame.loc[frame["season"] % 2 == 0].reset_index(drop=True)
    return odd, even


def cmd_screen(_args: argparse.Namespace) -> None:
    base_long = load_base_long_table()
    traits = load_traits()

    result: dict = {
        "thresholds": {
            "wind_mph": WIND_THRESHOLD_MPH,
            "heat_temp_f": HEAT_THRESHOLD_F,
            "outdoor_roofs": sorted(OUTDOOR_ROOFS),
        },
        "traits": {
            key: {
                "reliability": val["reliability"],
                "reliability_pairs": val["pairs"],
                "note": val["note"],
                "snapshot": val["snapshot_id"],
            }
            for key, val in traits.items()
        },
        "cells": {},
    }

    for cell_key, spec in CELL_SPECS.items():
        cell_result: dict = {"family": spec["family"], "by_cutoff": {}}
        for cutoff_name, cutoff_path in CUTOFFS.items():
            frame = build_cell_frame(cell_key, base_long, traits, cutoff_path)

            n_flag = int(frame["_flag"].sum())
            n_narrow = int(frame["_narrow_control"].sum())
            flag_cover = frame.loc[frame["_flag"], "team_covered"].dropna()
            complement_cover = frame.loc[~frame["_flag"], "team_covered"].dropna()
            narrow_cover = frame.loc[frame["_narrow_control"], "team_covered"].dropna()

            gap_vs_population_week = _block_bootstrap_signed_gap(
                frame,
                "team_covered",
                frame["_flag"],
                block_column="week_block",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )
            gap_vs_population_season = _block_bootstrap_signed_gap(
                frame,
                "team_covered",
                frame["_flag"],
                block_column="season",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )

            narrow_pop = frame.loc[frame["_flag"] | frame["_narrow_control"]].reset_index(drop=True)
            gap_vs_narrow_week = _block_bootstrap_signed_gap(
                narrow_pop,
                "team_covered",
                narrow_pop["_flag"],
                block_column="week_block",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )
            gap_vs_narrow_season = _block_bootstrap_signed_gap(
                narrow_pop,
                "team_covered",
                narrow_pop["_flag"],
                block_column="season",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )

            odd, even = _half(frame)
            odd_gap = _block_bootstrap_signed_gap(
                odd,
                "team_covered",
                odd["_flag"],
                block_column="week_block",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )
            even_gap = _block_bootstrap_signed_gap(
                even,
                "team_covered",
                even["_flag"],
                block_column="week_block",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                sign=-1,
            )

            cell_result["by_cutoff"][cutoff_name] = {
                "n_flag": n_flag,
                "n_narrow_control": n_narrow,
                "flag_cover_rate": float(flag_cover.mean()) if len(flag_cover) else float("nan"),
                "complement_cover_rate": (
                    float(complement_cover.mean()) if len(complement_cover) else float("nan")
                ),
                "narrow_control_cover_rate": (
                    float(narrow_cover.mean()) if len(narrow_cover) else float("nan")
                ),
                "gap_vs_population": {
                    "week_blocked_primary": gap_vs_population_week,
                    "season_blocked_secondary": gap_vs_population_season,
                },
                "gap_vs_narrow_control": {
                    "week_blocked_primary": gap_vs_narrow_week,
                    "season_blocked_secondary": gap_vs_narrow_season,
                },
                "split_half_odd_even_seasons": {"odd": odd_gap, "even": even_gap},
            }
        result["cells"][cell_key] = cell_result

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'screen_results.json'}")


def cmd_screen_surface(_args: argparse.Namespace) -> None:
    traits = load_traits()

    feat = pd.read_parquet(
        GAME_FEATURES,
        columns=["game_id", "season", "week", "game_type", "home_team", "away_team", "home_cover"],
    )
    reg = feat.loc[feat["game_type"].eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["week_block"] = reg["season"] * 100 + reg["week"]

    sched = pd.read_parquet(_latest_schedules(), columns=["game_id", "surface"])
    sched["game_id"] = sched["game_id"].astype(str)

    switch_flags = surface_switch_flag_by_game(
        pd.read_parquet(_latest_schedules()).assign(
            game_id=lambda d: d["game_id"].astype(str),
            home_team=lambda d: _canonical_team(d["home_team"]),
            away_team=lambda d: _canonical_team(d["away_team"]),
        )
    )
    switch_flags["game_id"] = switch_flags["game_id"].astype(str)

    away_pass_lag = traits["pass"]["lag"].rename(
        columns={"team": "away_team", "quartile": "away_pass_rate_quartile"}
    )[["away_team", "season", "away_pass_rate_quartile"]]

    table = reg.merge(switch_flags[["game_id", "surface_switch_flag"]], on="game_id", how="left")
    table = table.merge(away_pass_lag, on=["away_team", "season"], how="left")
    table["surface_switch_flag"] = table["surface_switch_flag"].fillna(False)
    table["_run_heavy_visitor"] = table["away_pass_rate_quartile"] == 1
    table["_flag"] = (table["surface_switch_flag"] & table["_run_heavy_visitor"]).fillna(False)
    table["_narrow_eligible"] = (
        table["surface_switch_flag"] & table["away_pass_rate_quartile"].notna()
    )
    table["_narrow_control"] = (table["_narrow_eligible"] & ~table["_run_heavy_visitor"]).fillna(
        False
    )

    n_flag = int(table["_flag"].sum())
    n_switch = int(table["surface_switch_flag"].sum())
    n_narrow = int(table["_narrow_control"].sum())
    flag_cover = table.loc[table["_flag"], "home_cover"].dropna()
    complement_cover = table.loc[~table["_flag"], "home_cover"].dropna()
    narrow_cover = table.loc[table["_narrow_control"], "home_cover"].dropna()

    gap_vs_population_week = _block_bootstrap_signed_gap(
        table,
        "home_cover",
        table["_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )
    gap_vs_population_season = _block_bootstrap_signed_gap(
        table,
        "home_cover",
        table["_flag"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )
    narrow_pop = table.loc[table["_flag"] | table["_narrow_control"]].reset_index(drop=True)
    gap_vs_narrow_week = _block_bootstrap_signed_gap(
        narrow_pop,
        "home_cover",
        narrow_pop["_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )
    gap_vs_narrow_season = _block_bootstrap_signed_gap(
        narrow_pop,
        "home_cover",
        narrow_pop["_flag"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )
    odd, even = _half(table)
    odd_gap = _block_bootstrap_signed_gap(
        odd,
        "home_cover",
        odd["_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )
    even_gap = _block_bootstrap_signed_gap(
        even,
        "home_cover",
        even["_flag"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        sign=1,
    )

    result = {
        "family": "weather_interactions_surface_switch_rush_fade",
        "n_surface_switch_games": n_switch,
        "n_flag": n_flag,
        "n_narrow_control": n_narrow,
        "flag_cover_rate": float(flag_cover.mean()) if len(flag_cover) else float("nan"),
        "complement_cover_rate": (
            float(complement_cover.mean()) if len(complement_cover) else float("nan")
        ),
        "narrow_control_cover_rate": float(narrow_cover.mean())
        if len(narrow_cover)
        else float("nan"),
        "gap_vs_population": {
            "week_blocked_primary": gap_vs_population_week,
            "season_blocked_secondary": gap_vs_population_season,
        },
        "gap_vs_narrow_control": {
            "week_blocked_primary": gap_vs_narrow_week,
            "season_blocked_secondary": gap_vs_narrow_season,
        },
        "split_half_odd_even_seasons": {"odd": odd_gap, "even": even_gap},
        "pass_rate_trait_reliability": traits["pass"]["reliability"],
        "pass_rate_trait_reliability_pairs": traits["pass"]["pairs"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "surface_switch_table.parquet", index=False)
    (OUT_DIR / "surface_switch_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'surface_switch_results.json'}")


def _run_production_team_long(cell_key: str) -> dict:
    spec = CELL_SPECS[cell_key]
    family = spec["production_family"]

    base_long = load_base_long_table()
    traits = load_traits()
    cell_frame = build_cell_frame(cell_key, base_long, traits, POOL_DECISION_ARCHIVE)

    flags = cell_frame[["game_id", "is_home", "_flag"]]
    home_flag = flags.loc[flags["is_home"], ["game_id", "_flag"]].rename(
        columns={"_flag": "fade_home"}
    )
    away_flag = flags.loc[~flags["is_home"], ["game_id", "_flag"]].rename(
        columns={"_flag": "fade_away"}
    )
    game_flags = home_flag.merge(away_flag, on="game_id", how="outer")
    game_flags["fade_home"] = game_flags["fade_home"].fillna(False).astype(bool)
    game_flags["fade_away"] = game_flags["fade_away"].fillna(False).astype(bool)

    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, family)
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

    scored["game_id"] = scored["game_id"].astype(str)
    scored = scored.merge(game_flags, on="game_id", how="left")
    scored["fade_home"] = scored["fade_home"].fillna(False).astype(bool)
    scored["fade_away"] = scored["fade_away"].fillna(False).astype(bool)

    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    candidate_pick_home = scored["baseline_pick_home"].to_numpy().copy()
    flip_to_away = scored["fade_home"].to_numpy() & scored["baseline_pick_home"].to_numpy()
    flip_to_home = scored["fade_away"].to_numpy() & ~scored["baseline_pick_home"].to_numpy()
    candidate_pick_home[flip_to_away] = False
    candidate_pick_home[flip_to_home] = True
    scored["candidate_pick_home"] = candidate_pick_home

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
        "family": family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "active_model_id": config.get("model_id"),
        "n_fade_home_in_window": int(scored["fade_home"].sum()),
        "n_fade_away_in_window": int(scored["fade_away"].sum()),
        "picks_changed": picks_changed,
        "picks_changed_graded": picks_changed_graded,
        "summary": summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT_DIR / f"production_{cell_key}_paired.parquet", index=False)
    (OUT_DIR / f"production_{cell_key}_results.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / f'production_{cell_key}_results.json'}")
    return result


def _run_production_surface() -> dict:
    family = "weather_interactions_surface_switch_rush_on_production"

    traits = load_traits()
    sched = pd.read_parquet(_latest_schedules())
    sched = sched.assign(
        game_id=lambda d: d["game_id"].astype(str),
        home_team=lambda d: _canonical_team(d["home_team"]),
        away_team=lambda d: _canonical_team(d["away_team"]),
    )
    switch_flags = surface_switch_flag_by_game(sched)
    switch_flags["game_id"] = switch_flags["game_id"].astype(str)

    away_pass_lag = traits["pass"]["lag"].rename(
        columns={"team": "away_team", "quartile": "away_pass_rate_quartile"}
    )[["away_team", "season", "away_pass_rate_quartile"]]

    game_teams = sched[["game_id", "away_team"]].drop_duplicates("game_id")
    flag_table = game_teams.merge(
        switch_flags[["game_id", "surface_switch_flag"]], on="game_id", how="left"
    )
    flag_table["season"] = flag_table["game_id"].map(
        sched.drop_duplicates("game_id").set_index("game_id")["season"]
    )
    flag_table = flag_table.merge(away_pass_lag, on=["away_team", "season"], how="left")
    flag_table["surface_switch_flag"] = flag_table["surface_switch_flag"].fillna(False)
    flag_table["_flag"] = (
        flag_table["surface_switch_flag"] & flag_table["away_pass_rate_quartile"].eq(1)
    ).fillna(False)
    flag_table = flag_table[["game_id", "_flag"]]

    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, family)
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

    scored["game_id"] = scored["game_id"].astype(str)
    scored = scored.merge(flag_table, on="game_id", how="left")
    scored["_flag"] = scored["_flag"].fillna(False)

    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    scored["candidate_pick_home"] = np.where(
        scored["_flag"] & ~scored["baseline_pick_home"], True, scored["baseline_pick_home"]
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
        "family": family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "active_model_id": config.get("model_id"),
        "n_flag_in_window": int(scored["_flag"].sum()),
        "picks_changed": picks_changed,
        "picks_changed_graded": picks_changed_graded,
        "summary": summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT_DIR / "production_surface_switch_rush_paired.parquet", index=False)
    (OUT_DIR / "production_surface_switch_rush_results.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'production_surface_switch_rush_results.json'}")
    return result


def cmd_production(args: argparse.Namespace) -> None:
    if args.cell == "surface_switch_rush":
        _run_production_surface()
    else:
        _run_production_team_long(args.cell)


def main() -> None:
    parser = argparse.ArgumentParser(description="ENV-05 weather-interactions screen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("screen", help="cells 1-3 (wind/heat x team style), both cutoffs")
    sub.add_parser("screen-surface", help="cell 4 (surface switch x visitor run-heaviness)")
    production_parser = sub.add_parser(
        "production", help="production stack: one cell's fade tilt on the opener grade"
    )
    production_parser.add_argument(
        "--cell",
        required=True,
        choices=["wind_pass_heavy", "wind_fg_reliant", "heat_pace", "surface_switch_rush"],
    )
    args = parser.parse_args()
    if args.command == "screen":
        cmd_screen(args)
    elif args.command == "screen-surface":
        cmd_screen_surface(args)
    elif args.command == "production":
        cmd_production(args)


if __name__ == "__main__":
    main()
