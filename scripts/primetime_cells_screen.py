"""Primetime situational cells screen: 5 predeclared primetime-window x
role/prior-result interaction cells plus the strongest cell's two era
splits, on REG 2009-2025 NFL team-games, week-blocked bootstrap primary,
season-blocked secondary, full-slate-scaled accuracy_points effects.

Strictly distinct from ``bias_battery_primetime_favorite`` (primetime
FAVORITE, any venue): this screen scores away underdogs, prior-result
response spots, divisional favorites, and post-Monday-night Sunday
follow-ons. Predeclaration frozen in ``docs/primetime_cells_screen.md``
before any cover rate was computed. Measure-only: never writes registry
JSON; stamps a run log to ``registry/experiments/primetime-cells-screen/``.
"""

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

sys.path.append(str(REPO / "scripts"))

from _common import (  # noqa: E402
    block_bootstrap_two_group,
    default_schedules,
)

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2017", 2009, 2017),
    ("2018_2025", 2018, 2025),
)

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "weekday",
    "gametime",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "div_game",
    "location",
    "away_rest",
    "home_rest",
]


def load_population(schedules_path: Path) -> pd.DataFrame:
    available = [c for c in SCHEDULE_COLUMNS if c in pd.read_parquet(schedules_path).columns]
    df = pd.read_parquet(schedules_path, columns=available)
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    for column in ("home_team", "away_team"):
        df[column] = df[column].replace(TEAM_ABBREVIATION_ALIASES)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df["kick_hour"] = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce").dt.hour
    df["neutral"] = df["location"] != "Home"
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    return df


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for is_home in (True, False):
        side = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "season": df["season"],
                "week": df["week"],
                "gameday": df["gameday"],
                "team": df["home_team"] if is_home else df["away_team"],
                "is_home": is_home,
                "team_covered": df["home_cover"] if is_home else 1.0 - df["home_cover"],
                "team_spread": df["spread_line"] if is_home else -df["spread_line"],
                "score_margin": df["result"] if is_home else -df["result"],
                "div_game": df["div_game"],
                "weekday": df["weekday"],
                "kick_hour": df["kick_hour"],
                "neutral": df["neutral"],
                "own_rest": pd.to_numeric(
                    df["home_rest"] if is_home else df["away_rest"], errors="coerce"
                ),
            }
        )
        rows.append(side)
    long_df = pd.concat(rows, ignore_index=True)
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["prior_score_margin"] = grouped["score_margin"].shift(1)
    long_df["prior_weekday"] = grouped["weekday"].shift(1)
    long_df["has_prior"] = grouped.cumcount() > 0
    return long_df


def build_cells(long_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    primetime = (
        long_df["weekday"].isin(["Thursday", "Monday"])
        | (long_df["weekday"].eq("Sunday") & (long_df["kick_hour"] >= 20))
    ) & (~long_df["neutral"])
    true_game = ~long_df["neutral"]
    favorite = long_df["team_spread"] > 0
    underdog = long_df["team_spread"] < 0
    off_loss = long_df["prior_score_margin"] < 0
    off_win = long_df["prior_score_margin"] > 0
    post_mnf_sunday = long_df["weekday"].eq("Sunday") & long_df["prior_weekday"].eq("Monday")

    specs: list[tuple[str, pd.Series, int, str]] = [
        (
            "pt_away_underdog",
            true_game & primetime & (~long_df["is_home"]) & underdog,
            1,
            "Away UNDERDOG in the primetime window (Thu/Mon any time, Sun >=20:00 ET; "
            "Saturday excluded) -- predicted positive team-cover edge (spotlight "
            "premium inflates the favorite's price)",
        ),
        (
            "pt_off_loss",
            true_game & primetime & off_loss,
            1,
            "Primetime team coming OFF A LOSS (own strictly-prior REG game this "
            "season lost) -- predicted positive team-cover edge (response spot)",
        ),
        (
            "pt_off_win",
            true_game & primetime & off_win,
            -1,
            "Primetime team coming OFF A WIN (own strictly-prior REG game this "
            "season won) -- predicted negative team-cover edge (letdown)",
        ),
        (
            "pt_divisional_favorite",
            true_game & primetime & (long_df["div_game"] == 1) & favorite,
            -1,
            "Divisional primetime FAVORITE (hostility-spot hype) -- predicted "
            "negative team-cover edge; DISCLOSED subset of "
            "bias_battery_primetime_favorite, correlated, never pool as independent",
        ),
        (
            "pt_post_mnf_sunday",
            true_game & post_mnf_sunday,
            -1,
            "Team playing SUNDAY after its own Monday game the prior week "
            "(6-day turnaround hangover) -- predicted negative team-cover edge; "
            "disjoint from bias_battery_short_week (rest <=5): measured 0-row overlap",
        ),
    ]

    cells: dict[str, dict[str, Any]] = {}
    for name, flag, sign, description in specs:
        eligible = (
            long_df["has_prior"]
            if name in ("pt_off_loss", "pt_off_win", "pt_post_mnf_sunday")
            else None
        )
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "sign": sign,
            "eligible": None if eligible is None else eligible.fillna(False).astype(bool),
            "description": f"{description} (pregame-safe schedule facts only, no leakage caveat).",
        }
    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    sign: int,
    block_col: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_cover = float(work.loc[work["_flag"], "team_covered"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_covered"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = sign * raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_covered",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    signed_draws = sign * draws
    dropped = samples - len(draws)
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    long_df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    population = long_df
    flag = spec["flag"]
    eligible = spec["eligible"]
    if eligible is not None:
        population = long_df.loc[eligible].reset_index(drop=True)
        flag = flag.loc[eligible].reset_index(drop=True)

    week_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="week_block", samples=samples, seed=seed
    )
    season_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="season", samples=samples, seed=seed
    )
    era_results = {}
    for era_label, start, end in ERA_SPLITS:
        era_mask = population["season"].between(start, end)
        era_results[era_label] = summarize(
            population.loc[era_mask].reset_index(drop=True),
            flag=flag.loc[era_mask].reset_index(drop=True),
            sign=spec["sign"],
            block_col="week_block",
            samples=samples,
            seed=seed,
        )
    return {
        "name": name,
        "sign_dir": spec["sign"],
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
        "era_split": era_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "primetime_cells_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}"
    )
    long_df = build_long_table(df)
    print(f"team-game rows (pushes dropped): {len(long_df)}")

    neutral_pt = int(
        (
            long_df["neutral"]
            & (
                long_df["weekday"].isin(["Thursday", "Monday"])
                | (long_df["weekday"].eq("Sunday") & (long_df["kick_hour"] >= 20))
            )
        ).sum()
    )
    short_week_overlap = int(
        (
            long_df["weekday"].eq("Sunday")
            & long_df["prior_weekday"].eq("Monday")
            & (long_df["own_rest"] <= 5)
        ).sum()
    )
    print(f"international-window team-game rows hitting primetime mask (excluded): {neutral_pt}")
    print(
        f"post-MNF-Sunday rows with own_rest<=5 (short-week battery overlap): {short_week_overlap}"
    )

    cells = build_cells(long_df)

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(long_df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"subset_cover={wb['subset_cover']:.4f} complement_cover={wb['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f} "
            f"full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )
        for era_label, _, _ in ERA_SPLITS:
            era = cell["era_split"][era_label]
            if era.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={era['n_flag']} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"P+={era['probability_positive']:.4f}"
            )

    scored = [r for r in results if not r["week_blocked"].get("insufficient_data")]
    ranked = sorted(
        scored, key=lambda r: abs(r["week_blocked"]["full_slate_effect_pts"]), reverse=True
    )
    print("\n=== ranked by |full-slate effect|, week-blocked primary ===")
    for rank, cell in enumerate(ranked, start=1):
        wb = cell["week_blocked"]
        print(
            f"{rank}. {cell['name']:<28} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    strongest = ranked[0]["name"]
    strongest_sign = next(c["sign_dir"] for c in results if c["name"] == strongest)
    era_cells = []
    for era_label, start, end in ERA_SPLITS:
        source = next(c for c in results if c["name"] == strongest)
        era = source["era_split"][era_label]
        era_cells.append(
            {
                "name": f"{strongest}_era_{era_label}",
                "parent": strongest,
                "sign_dir": strongest_sign,
                "season_start": start,
                "season_end": end,
                "week_blocked": era,
            }
        )
        if not era.get("insufficient_data"):
            print(
                f"\n=== {strongest}_era_{era_label} (item e, parent {strongest}) ===\n"
                f"  n_flag={era['n_flag']} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"95% [{era['ci95_scaled'][0]:+.4f}, {era['ci95_scaled'][1]:+.4f}] "
                f"P+={era['probability_positive']:.4f}"
            )

    configuration = {
        "command": "primetime-cells-screen",
        "schedules": str(args.schedules),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games_before_push_drop": df.attrs["n_before_push_drop"],
        "n_pushes_or_missing_dropped": df.attrs["pushes_or_missing"],
        "n_team_game_rows": len(long_df),
        "n_international_primetime_mask_rows_excluded": neutral_pt,
        "n_post_mnf_short_week_overlap_rows": short_week_overlap,
        "predeclaration": "docs/primetime_cells_screen.md (frozen before scoring)",
        "results": results,
        "strongest_cell": strongest,
        "era_cells_item_e": era_cells,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="primetime-cells-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (5 predeclared primetime situational "
            "cells + strongest cell's 2 era splits); mined family, every cell predeclared "
            "to record unresolved_below_power via nfl-ats weak-signals record regardless "
            "of interval shape (AGENTS.md); pt_divisional_favorite is a disclosed subset "
            "of bias_battery_primetime_favorite and pt_post_mnf_sunday is thematically "
            "adjacent to bias_battery_short_week with measured 0-row overlap -- never "
            "pool any of these as independent."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
