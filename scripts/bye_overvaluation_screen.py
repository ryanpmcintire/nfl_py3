"""Bye-week overvaluation screen (post-2011 CBA): 5 predeclared cells testing
whether the market overprices bye rest in the modern era, against the spread
on REG 2009-2025 NFL games with a week-blocked bootstrap (season-blocked
secondary), full-slate scaled accuracy_points effects, seeded and
deterministic.

Predeclaration frozen in ``docs/bye_overvaluation_screen.md`` BEFORE this
script scored anything. Machinery reused from
``scripts/venue_milestone_screen.py`` (strict >=12-day bye definition kept
verbatim). Measure-only: never writes either registry JSON; every flag is a
schedule fact (gameday gaps plus the pre-release closing spread_line), so all
cells are point-in-time safe by construction.

Writes JSON to ``artifacts/bye_overvaluation_screen/<UTC timestamp>/results.json``
and stamps ``registry/experiments/bye-overvaluation-screen/``.
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

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025
POST_BYE_GAP_DAYS = 12
ERA_POST_MIN_SEASON = 2012


def load_population(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    available = [c for c in DEFAULT_SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)

    df["week_block"] = df["season"] * 100 + df["week"]
    df["gameday_dt"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df


def build_bye_maps(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    long_rows = []
    for _, g in df.iterrows():
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "team": team,
                    "side": side,
                    "gameday_dt": g["gameday_dt"],
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    long_df["gap_days"] = long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days
    long_df["post_bye"] = (long_df["gap_days"] >= POST_BYE_GAP_DAYS).fillna(False).astype(bool)

    def side_map(side: str) -> pd.Series:
        joined = df[["game_id"]].merge(
            long_df.loc[long_df["side"] == side, ["game_id", "post_bye"]],
            on="game_id",
            how="left",
        )
        return joined["post_bye"].fillna(False).astype(bool)

    return side_map("home"), side_map("away")


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    value_col: str,
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
    subset_cover = float(work.loc[work["_flag"], value_col].mean())
    complement_cover = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col=value_col,
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    dropped = samples - len(draws)
    scaled_draws = draws * fraction_of_slate
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
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
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
    output_dir: Path = args.output or (REPO / "artifacts" / "bye_overvaluation_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: scored population {len(df)} "
        f"(pushes/missing dropped upstream of add_ats_outcomes filter)"
    )

    home_pb, away_pb = build_bye_maps(df)
    df["home_off_bye"] = home_pb.to_numpy()
    df["away_off_bye"] = away_pb.to_numpy()

    df["fade_side_cover"] = np.where(
        df["home_off_bye"] & ~df["away_off_bye"],
        1.0 - df["home_cover"],
        np.where(
            df["away_off_bye"] & ~df["home_off_bye"],
            df["home_cover"],
            df["home_cover"],
        ),
    )
    bye_week_blocks = set(
        df.loc[df["home_off_bye"] | df["away_off_bye"], "week_block"].unique().tolist()
    )
    df["in_bye_week_block"] = df["week_block"].isin(bye_week_blocks)

    cells: list[dict[str, Any]] = []

    def run_cell(name: str, description: str, sub: pd.DataFrame, flag: pd.Series, value_col: str):
        flag = flag.reset_index(drop=True)
        print(f"\n=== {name} ===")
        week_blocked = summarize(
            sub.reset_index(drop=True),
            flag=flag,
            value_col=value_col,
            block_col="week_block",
            samples=args.samples,
            seed=args.seed,
        )
        season_blocked = summarize(
            sub.reset_index(drop=True),
            flag=flag,
            value_col=value_col,
            block_col="season",
            samples=args.samples,
            seed=args.seed,
        )
        cell = {
            "name": name,
            "description": description,
            "n_flag": int(flag.sum()),
            "value_column": value_col,
            "week_blocked": week_blocked,
            "season_blocked_secondary": season_blocked,
        }
        cells.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print(f"  insufficient data (n_flag={wb['n_flag']}, n_complement={wb['n_complement']})")
            return
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"subset_cover={wb['subset_cover']:.4f} "
            f"complement_cover={wb['complement_cover']:.4f} "
            f"raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            degenerate = " [DEGENERATE: below block-count floor]" if sb["n_blocks"] < 10 else ""
            print(
                f"  [season-blocked secondary]{degenerate} "
                f"95% [{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
                f"P+={sb['probability_positive']:.4f} n_seasons={sb['n_blocks']}"
            )

    era_mask = {
        "post": df["season"] >= ERA_POST_MIN_SEASON,
        "pre": df["season"] < ERA_POST_MIN_SEASON,
        "all": pd.Series(True, index=df.index),
    }
    home_edge = df["home_off_bye"] & ~df["away_off_bye"]

    run_cell(
        "bye_overval_home_edge_post2011",
        (
            "HOME team off strict bye (>=12-day gap to its immediately preceding "
            "game) AND opponent NOT off bye, seasons 2012-2025. Market overprices "
            "the bye-holding side. Predicted direction NEGATIVE home_cover."
        ),
        df.loc[era_mask["post"]].copy(),
        home_edge.loc[era_mask["post"]],
        "home_cover",
    )
    run_cell(
        "bye_overval_home_edge_pre2011",
        (
            "Identical flag to bye_overval_home_edge_post2011, seasons 2009-2011 "
            "(era control; pre-CBA true bye advantage reportedly real). Predicted "
            "POSITIVE or null-difference versus the post-2011 cell. Only 3 season "
            "blocks exist; season-blocked secondary is DEGENERATE by construction."
        ),
        df.loc[era_mask["pre"]].copy(),
        home_edge.loc[era_mask["pre"]],
        "home_cover",
    )
    run_cell(
        "bye_overval_road_fav_post2011",
        (
            "AWAY team off strict bye AND spread_line < 0 (road favorite, measured "
            "convention positive=home favored), seasons 2012-2025. Modern-era test "
            "of the legacy Sung & Tainsky 73% ATS claim. Overvaluation mechanism "
            "predicts POSITIVE home_cover (fade arm); a surviving legacy effect "
            "would lean negative."
        ),
        df.loc[era_mask["post"]].copy(),
        (
            (df["away_off_bye"] & ~df["home_off_bye"] & (df["spread_line"] < 0)).loc[
                era_mask["post"]
            ]
        ),
        "home_cover",
    )
    run_cell(
        "bye_overval_both_bye_sanity",
        (
            "BOTH teams off strict bye, full window 2009-2025. Rest cancels while "
            "the market reportedly still prices one bye; instrument sanity cell, "
            "predicted NULL (two-sided, no directional claim). A large one-sided "
            "interval here impeaches the instrument, not a confirmation."
        ),
        df.loc[era_mask["all"]].copy(),
        (df["home_off_bye"] & df["away_off_bye"]).loc[era_mask["all"]],
        "home_cover",
    )
    run_cell(
        "bye_overval_fade_full_slate_post2011",
        (
            "Fade arm expressed full-slate: seasons 2012-2025 restricted to "
            "week-blocks containing at least one strictly-off-bye team anywhere in "
            "the league; flag = exactly one of the two teams off strict bye; value "
            "column is the FADE-side cover indicator (home holds the bye edge => "
            "1-home_cover i.e. away covers; away holds the edge => home_cover; "
            "complement rows keep raw home_cover -- disclosed asymmetry). "
            "Predicted direction POSITIVE."
        ),
        df.loc[era_mask["post"] & df["in_bye_week_block"]].copy(),
        (df["home_off_bye"] ^ df["away_off_bye"]).loc[era_mask["post"] & df["in_bye_week_block"]],
        "fade_side_cover",
    )

    assert len(cells) == 5, f"expected 5 predeclared cells, got {len(cells)}"

    diagnostics = {
        "n_home_off_bye_games": int(home_edge.sum()),
        "n_away_only_off_bye_games": int((df["away_off_bye"] & ~df["home_off_bye"]).sum()),
        "n_both_off_bye_games": int((df["home_off_bye"] & df["away_off_bye"]).sum()),
        "n_bye_week_blocks": len(bye_week_blocks),
        "era_boundary": ERA_POST_MIN_SEASON,
    }

    configuration = {
        "command": "bye-overvaluation-screen",
        "schedules": str(args.schedules),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "post_bye_gap_days": POST_BYE_GAP_DAYS,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_scored_population": len(df),
        "post_bye_gap_days": POST_BYE_GAP_DAYS,
        "era_boundary": ERA_POST_MIN_SEASON,
        "diagnostics": diagnostics,
        "spread_convention": "spread_line positive means home favored (measured)",
        "predeclaration": "docs/bye_overvaluation_screen.md (frozen before scoring)",
        "point_in_time_safety": (
            "every flag is a schedule fact derived solely from gameday gaps within "
            "each team's season plus the pre-release closing spread_line"
        ),
        "results": cells,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="bye-overvaluation-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (5 predeclared bye-overvaluation "
            "cells); mined family, every scoreable cell predeclared to record "
            "unresolved_below_power via separate nfl-ats weak-signals record calls "
            "regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
