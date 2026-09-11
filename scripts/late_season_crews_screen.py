from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.officials_archive import load_officials  # noqa: E402
from nfl_ats.provenance import sha256_file  # noqa: E402
from nfl_ats.public_board import find_matching_opener_evaluation  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_snapshot  # noqa: E402

SEED = 20260910
SAMPLES = 20000
POST_TYPES = ("POST", "WC", "DIV", "CON", "SB")
LATE_WEEKS = (15, 16, 17, 18)
ALL_STAR_POPULATION_SEASON_START = 2016
ALL_STAR_POPULATION_SEASON_END = 2025
RELIABILITY_MIN_GAMES_PER_HALF = 5
PBP_ROOT = REPO / "data" / "pbp" / "raw" / "20260817T184927Z"
OFFICIALS_PATH = REPO / "data" / "raw" / "officials" / "20260819T190537Z" / "officials.parquet"
GAME_PENALTIES_PATH = (
    REPO / "data" / "raw" / "officials" / "20260819T190537Z" / "game_penalties.parquet"
)
GAME_FEATURES_PBP_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
OUTPUT = REPO / "artifacts" / "experiments" / "late_season_crews"


def load_plays_per_game(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = PBP_ROOT / f"season={season}" / "plays.parquet"
        pbp = pd.read_parquet(path, columns=["game_id", "posteam"])
        pbp = pbp.loc[pbp["posteam"].notna()]
        counts = pbp.groupby("game_id").size()
        frames.append(counts)
    combined = pd.concat(frames)
    return combined.rename("plays_from_pbp").reset_index()


def build_crosswalk(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules.loc[schedules["game_type"] == "REG", ["old_game_id", "game_id"]].copy()
    reg = reg.rename(columns={"old_game_id": "legacy_game_id"})
    reg["legacy_game_id"] = reg["legacy_game_id"].astype(str)
    return reg.drop_duplicates("legacy_game_id")


def build_full_referee_table(
    officials: pd.DataFrame, crosswalk: pd.DataFrame, penalties: pd.DataFrame, plays: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref = officials.loc[
        (officials["position"] == "Referee") & (officials["season_type"] == "REG"),
        ["game_id", "official_name", "season", "week"],
    ].copy()
    ref = ref.rename(columns={"game_id": "legacy_game_id"})
    ref["legacy_game_id"] = ref["legacy_game_id"].astype(str)
    merged = ref.merge(crosswalk, on="legacy_game_id", how="left")
    crosswalk_match_rate = float(merged["game_id"].notna().mean())
    merged = merged.loc[merged["game_id"].notna()].copy()
    merged = merged.merge(penalties, on="game_id", how="left")
    merged = merged.merge(plays, on="game_id", how="left")
    merged["penalty_rate"] = merged["penalties_total"] / merged["plays_from_pbp"]
    coverage = {
        "n_referee_reg_rows": len(ref),
        "crosswalk_match_rate": crosswalk_match_rate,
        "n_matched_to_standard_game_id": len(merged),
        "n_with_penalties_total": int(merged["penalties_total"].notna().sum()),
        "n_with_plays_from_pbp": int(merged["plays_from_pbp"].notna().sum()),
    }
    return merged, coverage


def build_playoff_referee_seasons(officials: pd.DataFrame) -> set[tuple[str, int]]:
    playoff = officials.loc[
        (officials["position"] == "Referee") & (officials["season_type"].isin(POST_TYPES)),
        ["official_name", "season"],
    ].drop_duplicates()
    return set(
        zip(
            playoff["official_name"].tolist(),
            playoff["season"].astype(int).tolist(),
            strict=True,
        )
    )


def build_all_star_population(
    full_referee_table: pd.DataFrame, playoff_referee_seasons: set[tuple[str, int]]
) -> pd.DataFrame:
    late = full_referee_table.loc[
        full_referee_table["week"].isin(LATE_WEEKS)
        & full_referee_table["season"].between(
            ALL_STAR_POPULATION_SEASON_START, ALL_STAR_POPULATION_SEASON_END
        )
    ].copy()
    late["all_star_crew"] = [
        (name, int(season) - 1) in playoff_referee_seasons
        for name, season in zip(late["official_name"], late["season"], strict=True)
    ]
    return late


def simple_bootstrap_diff(
    group_a: np.ndarray, group_b: np.ndarray, *, samples: int = SAMPLES, seed: int = SEED
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.full(samples, np.nan)
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
            "n_a": n_a,
            "n_b": n_b,
        }
    for i in range(samples):
        a_sample = group_a[rng.integers(0, n_a, size=n_a)]
        b_sample = group_b[rng.integers(0, n_b, size=n_b)]
        draws[i] = float(np.mean(a_sample) - np.mean(b_sample))
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return {
        "estimate": float(np.mean(group_a) - np.mean(group_b)),
        "lower": float(low),
        "upper": float(high),
        "probability_positive": float(probability_positive_from_draws(draws, ignore_nan=True)),
        "n_a": n_a,
        "n_b": n_b,
    }


def paired_bootstrap_mean(
    diffs: np.ndarray, *, samples: int = SAMPLES, seed: int = SEED
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
            "n_units": 0,
        }
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        sample = diffs[rng.integers(0, n, size=n)]
        draws[i] = float(np.mean(sample))
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return {
        "estimate": float(np.mean(diffs)),
        "lower": float(low),
        "upper": float(high),
        "probability_positive": float(probability_positive_from_draws(draws, ignore_nan=True)),
        "n_units": n,
    }


def tightness_cross_sectional(late_population: pd.DataFrame) -> dict[str, Any]:
    all_star = late_population.loc[late_population["all_star_crew"]]
    other = late_population.loc[~late_population["all_star_crew"]]
    game_diff = simple_bootstrap_diff(
        all_star["penalties_total"].dropna().to_numpy(dtype=float),
        other["penalties_total"].dropna().to_numpy(dtype=float),
        seed=SEED,
    )
    play_diff = simple_bootstrap_diff(
        all_star["penalty_rate"].dropna().to_numpy(dtype=float),
        other["penalty_rate"].dropna().to_numpy(dtype=float),
        seed=SEED + 1,
    )
    return {
        "n_all_star_games": len(all_star),
        "n_other_games": len(other),
        "all_star_mean_penalties_per_game": float(all_star["penalties_total"].mean()),
        "other_mean_penalties_per_game": float(other["penalties_total"].mean()),
        "penalties_per_game_diff_bootstrap": game_diff,
        "all_star_mean_penalty_rate": float(all_star["penalty_rate"].mean()),
        "other_mean_penalty_rate": float(other["penalty_rate"].mean()),
        "penalty_rate_diff_bootstrap": play_diff,
    }


def tightness_own_earlier_season(
    late_population: pd.DataFrame, full_referee_table: pd.DataFrame
) -> dict[str, Any]:
    flagged_units = late_population.loc[
        late_population["all_star_crew"], ["official_name", "season"]
    ].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for official_name, season in zip(
        flagged_units["official_name"], flagged_units["season"], strict=True
    ):
        late_rows = late_population.loc[
            (late_population["official_name"] == official_name)
            & (late_population["season"] == season)
        ]
        early_rows = full_referee_table.loc[
            (full_referee_table["official_name"] == official_name)
            & (full_referee_table["season"] == season)
            & (full_referee_table["week"] < 15)
        ]
        if early_rows.empty or late_rows.empty:
            continue
        rows.append(
            {
                "official_name": official_name,
                "season": int(season),
                "n_late_games": len(late_rows),
                "n_early_games": len(early_rows),
                "late_penalties_per_game": float(late_rows["penalties_total"].mean()),
                "early_penalties_per_game": float(early_rows["penalties_total"].mean()),
                "late_penalty_rate": float(late_rows["penalty_rate"].mean()),
                "early_penalty_rate": float(early_rows["penalty_rate"].mean()),
            }
        )
    paired = pd.DataFrame(rows)
    if paired.empty:
        return {"n_units": 0, "paired_table": paired}
    game_diffs = (paired["late_penalties_per_game"] - paired["early_penalties_per_game"]).to_numpy(
        dtype=float
    )
    rate_diffs = (paired["late_penalty_rate"] - paired["early_penalty_rate"]).to_numpy(dtype=float)
    return {
        "n_units": len(paired),
        "penalties_per_game_diff_bootstrap": paired_bootstrap_mean(game_diffs, seed=SEED + 2),
        "penalty_rate_diff_bootstrap": paired_bootstrap_mean(rate_diffs, seed=SEED + 3),
        "paired_table": paired,
    }


def build_physical_underdog_quartile_cut(
    game_features_pbp: pd.DataFrame, schedules: pd.DataFrame
) -> float:
    reg_ids = set(schedules.loc[schedules["game_type"] == "REG", "game_id"].astype(str))
    reg = game_features_pbp.loc[game_features_pbp["game_id"].astype(str).isin(reg_ids)]
    pooled = pd.concat(
        [reg["home_pbp_off_pass_rate"], reg["away_pbp_off_pass_rate"]], ignore_index=True
    ).dropna()
    return float(np.quantile(pooled.to_numpy(dtype=float), 0.25))


def attach_physical_underdog(
    per_game: pd.DataFrame, game_features_pbp: pd.DataFrame, bottom_quartile_cut: float
) -> pd.DataFrame:
    frame = per_game.merge(
        game_features_pbp[["game_id", "home_pbp_off_pass_rate", "away_pbp_off_pass_rate"]],
        on="game_id",
        how="left",
    )
    frame["underdog_is_home"] = frame["tue_open_home_spread"].lt(0.0)
    frame["underdog_is_away"] = frame["tue_open_home_spread"].gt(0.0)
    frame["valid_underdog"] = frame["underdog_is_home"] | frame["underdog_is_away"]
    frame["underdog_pass_rate"] = np.where(
        frame["underdog_is_home"], frame["home_pbp_off_pass_rate"], frame["away_pbp_off_pass_rate"]
    )
    frame["physical_underdog"] = (
        frame["valid_underdog"]
        & frame["underdog_pass_rate"].notna()
        & frame["underdog_pass_rate"].le(bottom_quartile_cut)
    )
    frame["underdog_covered"] = pick_correct(frame["underdog_is_home"], frame["margin_vs_open"])
    return frame


def attach_all_star_flag(frame: pd.DataFrame, late_population: pd.DataFrame) -> pd.DataFrame:
    flag = late_population[["game_id", "all_star_crew"]].drop_duplicates("game_id")
    merged = frame.merge(flag, on="game_id", how="left")
    merged["all_star_crew"] = merged["all_star_crew"].fillna(False).astype(bool)
    return merged


def underdog_cover_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    physical = frame.loc[frame["physical_underdog"] & frame["underdog_covered"].notna()]
    all_star = physical.loc[physical["all_star_crew"]]
    other = physical.loc[~physical["all_star_crew"]]
    cover_diff = simple_bootstrap_diff(
        all_star["underdog_covered"].to_numpy(dtype=float),
        other["underdog_covered"].to_numpy(dtype=float),
        seed=SEED + 4,
    )
    return {
        "n_physical_underdog_games": len(physical),
        "n_all_star": len(all_star),
        "n_other": len(other),
        "all_star_cover_rate": (
            float(all_star["underdog_covered"].mean()) if len(all_star) else None
        ),
        "other_cover_rate": (float(other["underdog_covered"].mean()) if len(other) else None),
        "cover_rate_diff_bootstrap_accuracy_points": {
            **{
                k: (v * 100.0 if k in ("estimate", "lower", "upper") else v)
                for k, v in cover_diff.items()
            }
        },
    }


def production_screen(frame: pd.DataFrame, season_start: int, season_end: int) -> dict[str, Any]:
    frame = frame.copy()
    frame["favorite_is_home"] = frame["tue_open_home_spread"].gt(0.0)
    frame["trigger"] = frame["all_star_crew"] & frame["physical_underdog"]
    frame["production_pick_home"] = frame["pick_home_at_open_probability_rule"].astype(bool)
    frame["candidate_pick_home"] = np.where(
        frame["trigger"], frame["favorite_is_home"], frame["production_pick_home"]
    )
    frame["production_correct"] = pick_correct(
        frame["production_pick_home"], frame["margin_vs_open"]
    )
    frame["candidate_correct"] = pick_correct(frame["candidate_pick_home"], frame["margin_vs_open"])
    frame["oracle_pick_home"] = frame["margin_vs_open"].gt(0.0)
    frame["oracle_correct"] = pick_correct(frame["oracle_pick_home"], frame["margin_vs_open"])

    def score(subset: pd.DataFrame, correct_col: str, other_col: str) -> pd.DataFrame:
        valid = subset.dropna(subset=[correct_col, other_col]).copy()
        valid["delta"] = (valid[correct_col] - valid[other_col]) * 100.0

        def metric_fn(block: pd.DataFrame) -> dict[str, float]:
            return {"delta": float(block["delta"].mean())}

        return week_blocked_bootstrap(valid, metric_fn, block="week", samples=SAMPLES, seed=SEED)

    full = frame
    window = frame.loc[frame["season"].between(season_start, season_end)]

    results: dict[str, Any] = {}
    for label, subset in (("full_archive", full), ("assigned_window", window)):
        n_games = len(subset)
        n_flips = int((subset["candidate_pick_home"] != subset["production_pick_home"]).sum())
        n_triggers = int(subset["trigger"].sum())
        candidate_bootstrap = score(subset, "candidate_correct", "production_correct")
        control_bootstrap = score(subset, "oracle_correct", "production_correct")
        results[label] = {
            "n_games": n_games,
            "n_triggers": n_triggers,
            "n_flips": n_flips,
            "production_accuracy": float(subset["production_correct"].mean() * 100),
            "candidate_accuracy": float(subset["candidate_correct"].mean() * 100),
            "candidate_effect_accuracy_points": float(candidate_bootstrap.loc[0, "estimate"]),
            "candidate_interval": [
                float(candidate_bootstrap.loc[0, "lower"]),
                float(candidate_bootstrap.loc[0, "upper"]),
            ],
            "candidate_probability_positive": float(
                candidate_bootstrap.loc[0, "probability_positive"]
            ),
            "positive_control_effect_accuracy_points": float(control_bootstrap.loc[0, "estimate"]),
            "positive_control_probability_positive": float(
                control_bootstrap.loc[0, "probability_positive"]
            ),
        }
    return results


def split_half_reliability(full_referee_table: pd.DataFrame) -> dict[str, Any]:
    frame = full_referee_table.dropna(subset=["penalties_total", "plays_from_pbp"]).copy()
    frame["half"] = np.where(frame["season"] % 2 == 0, "even", "odd")
    grouped = frame.groupby(["official_name", "half"]).agg(
        total_penalties=("penalties_total", "sum"),
        total_plays=("plays_from_pbp", "sum"),
        total_games=("penalties_total", "size"),
    )
    grouped["rate"] = grouped["total_penalties"] / grouped["total_plays"]
    grouped["per_game"] = grouped["total_penalties"] / grouped["total_games"]
    rate_wide = grouped["rate"].unstack("half")
    per_game_wide = grouped["per_game"].unstack("half")
    n_wide = grouped["total_games"].unstack("half")
    keep = n_wide["odd"].fillna(0).ge(RELIABILITY_MIN_GAMES_PER_HALF) & n_wide["even"].fillna(0).ge(
        RELIABILITY_MIN_GAMES_PER_HALF
    )

    def reliability_for(wide: pd.DataFrame, seed: int) -> dict[str, Any]:
        paired = wide.loc[keep].dropna(subset=["odd", "even"])
        if len(paired) < 4:
            return {"n_referees": len(paired), "pearson_r": None}
        odd = paired["odd"].to_numpy(dtype=float)
        even = paired["even"].to_numpy(dtype=float)
        pearson_r = (
            float(np.corrcoef(odd, even)[0, 1])
            if odd.std() > 0 and even.std() > 0
            else float("nan")
        )
        rng = np.random.default_rng(seed)
        n = len(paired)
        draws = np.full(SAMPLES, np.nan)
        for i in range(SAMPLES):
            idx = rng.integers(0, n, size=n)
            odd_s, even_s = odd[idx], even[idx]
            if odd_s.std() > 0 and even_s.std() > 0:
                draws[i] = float(np.corrcoef(odd_s, even_s)[0, 1])
        low, high = np.nanquantile(draws, [0.025, 0.975])
        return {
            "n_referees": int(n),
            "pearson_r": pearson_r,
            "interval": [float(low), float(high)],
            "probability_positive": float(probability_positive_from_draws(draws, ignore_nan=True)),
        }

    return {
        "penalty_rate_per_play": reliability_for(rate_wide, SEED + 5),
        "penalties_per_game": reliability_for(per_game_wide, SEED + 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LEAD-33 late-season all-star crews screen")
    parser.add_argument("--season-start", type=int, default=2020)
    parser.add_argument("--season-end", type=int, default=2021)
    args = parser.parse_args()

    officials = load_officials(REPO, include_archive=False)
    snapshot = latest_snapshot(REPO / "data" / "raw")
    schedules, _ = load_snapshot(snapshot)
    crosswalk = build_crosswalk(schedules)
    penalties = pd.read_parquet(GAME_PENALTIES_PATH)[["game_id", "penalties_total"]]
    plays = load_plays_per_game(list(range(2015, 2026)))

    full_referee_table, coverage = build_full_referee_table(officials, crosswalk, penalties, plays)
    playoff_referee_seasons = build_playoff_referee_seasons(officials)
    late_population = build_all_star_population(full_referee_table, playoff_referee_seasons)

    n_late = len(late_population)
    n_all_star = int(late_population["all_star_crew"].sum())
    n_distinct_referees = int(late_population["official_name"].nunique())
    n_distinct_all_star_referees = int(
        late_population.loc[late_population["all_star_crew"], "official_name"].nunique()
    )

    tightness_cross = tightness_cross_sectional(late_population)
    tightness_paired = tightness_own_earlier_season(late_population, full_referee_table)
    reliability = split_half_reliability(full_referee_table)

    match = find_matching_opener_evaluation(REPO / "artifacts")
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    metadata, per_game_dir = match
    per_game_path = per_game_dir / "per_game.parquet"
    per_game = pd.read_parquet(per_game_path)
    per_game["season"] = per_game["season"].astype(int)
    per_game["week"] = per_game["week"].astype(int)

    game_features_pbp = pd.read_parquet(GAME_FEATURES_PBP_PATH)
    bottom_quartile_cut = build_physical_underdog_quartile_cut(game_features_pbp, schedules)

    graded = attach_physical_underdog(per_game, game_features_pbp, bottom_quartile_cut)
    graded = attach_all_star_flag(graded, late_population)

    cover_comparison = underdog_cover_comparison(graded)
    screen = production_screen(graded, args.season_start, args.season_end)

    destination = OUTPUT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metadata_out: dict[str, Any] = {
        "lead": "LEAD-33",
        "active_model_id": metadata.get("active_model_id"),
        "definition": {
            "all_star_crew": (
                "REG-season game in weeks 15-18 whose assigned Referee officiated at least one "
                "POST/WC/DIV/CON/SB game as Referee in the immediately preceding season "
                f"(population restricted to seasons {ALL_STAR_POPULATION_SEASON_START}-"
                f"{ALL_STAR_POPULATION_SEASON_END} to keep a full prior season on record)."
            ),
            "physical_underdog": (
                "the underdog side (by sign of tue_open_home_spread) whose "
                "prior-rolling pregame pbp_off_pass_rate (game_features_pbp.parquet) sits at or "
                f"below the pooled home+away bottom-quartile cut ({bottom_quartile_cut:.4f}) "
                "computed over the full REG-season archive."
            ),
            "predeclared_direction": "BACK the favorite when all_star_crew AND physical_underdog",
        },
        "coverage": coverage,
        "population": {
            "n_late_week_referee_games": n_late,
            "n_all_star_flagged_games": n_all_star,
            "n_distinct_referees": n_distinct_referees,
            "n_distinct_all_star_referees": n_distinct_all_star_referees,
            "season_start": ALL_STAR_POPULATION_SEASON_START,
            "season_end": ALL_STAR_POPULATION_SEASON_END,
        },
        "tightness_cross_sectional": tightness_cross,
        "tightness_own_earlier_season": {
            k: v for k, v in tightness_paired.items() if k != "paired_table"
        },
        "split_half_reliability_odd_even_seasons": reliability,
        "physical_underdog_quartile_cut": bottom_quartile_cut,
        "underdog_cover_comparison": cover_comparison,
        "production_screen": screen,
        "assigned_window": [args.season_start, args.season_end],
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "source_per_game": str(per_game_path),
        "source_per_game_sha256": sha256_file(per_game_path),
        "source_officials": str(OFFICIALS_PATH),
        "source_officials_sha256": sha256_file(OFFICIALS_PATH),
        "source_game_penalties": str(GAME_PENALTIES_PATH),
        "source_game_penalties_sha256": sha256_file(GAME_PENALTIES_PATH),
        "source_game_features_pbp": str(GAME_FEATURES_PBP_PATH),
        "source_game_features_pbp_sha256": sha256_file(GAME_FEATURES_PBP_PATH),
    }
    atomic_json(metadata_out, destination / "results.json")
    atomic_parquet(late_population, destination / "late_population.parquet")
    if not tightness_paired.get("paired_table", pd.DataFrame()).empty:
        atomic_parquet(tightness_paired["paired_table"], destination / "paired_early_late.parquet")
    atomic_parquet(graded, destination / "graded_per_game.parquet")

    print(f"wrote {destination}")
    print(
        {
            "coverage": coverage,
            "population": metadata_out["population"],
            "tightness_cross_sectional": tightness_cross,
            "tightness_own_earlier_season": {
                k: v for k, v in tightness_paired.items() if k != "paired_table"
            },
            "reliability": reliability,
            "underdog_cover_comparison": cover_comparison,
            "production_screen": screen,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
