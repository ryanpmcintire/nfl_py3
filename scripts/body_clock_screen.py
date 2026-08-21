"""ENV-06 circadian body-clock screen: 6 predeclared kickoff-time x
body-clock cells on REG 2009-2025 NFL games, week-blocked bootstrap primary,
season-blocked secondary, full-slate-scaled accuracy_points effects.

Predeclaration frozen in ``docs/body_clock_screen.md`` before any cover rate
was computed. Measure-only: never writes registry JSON; stamps a run log to
``registry/experiments/body-clock-screen/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025

WEST_TZS = {"America/Los_Angeles", "America/Phoenix"}
EAST_TZ = "America/New_York"
EARLY_KICK_MAX_MIN = 14 * 60
LATE_KICK_MIN_MIN = 19 * 60
MIDDAY_MIN_MIN = 14 * 60
MIDDAY_MAX_MIN = 17 * 60
ERA_SPLIT_YEAR = 2017

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "location",
    "stadium",
]

DEFAULT_COORDS_PATH = REPO / "registry" / "stadium_coordinates.json"


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()


def load_coords(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_population(schedules_path: Path, coords: dict[str, dict[str, Any]]) -> pd.DataFrame:
    available = [c for c in SCHEDULE_COLUMNS if c in pd.read_parquet(schedules_path).columns]
    df = pd.read_parquet(schedules_path, columns=available)
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df["week_block"] = df["season"] * 100 + df["week"]

    kick = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce")
    df["kick_min"] = kick.dt.hour * 60 + kick.dt.minute

    home_rows = df.loc[df["location"] == "Home"]
    modal_stadium = (
        home_rows.groupby(["home_team", "season"])["stadium"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("team_home_stadium")
    )
    df = df.merge(modal_stadium, left_on=["away_team", "season"], right_index=True, how="left")

    def tz_of(name: object) -> str | None:
        entry = coords.get(name) if isinstance(name, str) else None
        return entry["tz"] if entry else None

    df["away_body_tz"] = df["team_home_stadium"].map(tz_of)
    df["venue_tz"] = df["stadium"].map(tz_of)

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    df.attrs["n_unresolved_stadium_names"] = int(
        df.loc[df["venue_tz"].isna() | df["away_body_tz"].isna(), "stadium"].nunique()
    )
    return df


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    away_west = df["away_body_tz"].isin(WEST_TZS)
    venue_east = df["venue_tz"] == EAST_TZ
    venue_west = df["venue_tz"].isin(WEST_TZS)
    away_east = df["away_body_tz"] == EAST_TZ
    true_home = df["location"] == "Home"
    early = df["kick_min"] < EARLY_KICK_MAX_MIN
    late = df["kick_min"] >= LATE_KICK_MIN_MIN
    midday = (df["kick_min"] >= MIDDAY_MIN_MIN) & (df["kick_min"] < MIDDAY_MAX_MIN)

    missing = (
        df["kick_min"].isna()
        | df["away_body_tz"].isna()
        | df["venue_tz"].isna()
        | df["location"].isna()
    )

    c1_flag = away_west & true_home & early
    specs = [
        (
            "body_clock_west_road_early",
            c1_flag,
            "Away team's body clock Pacific/Arizona (modal home stadium tz), true road "
            "game, kickoff < 14:00 ET (~10am biological) -- predicted positive "
            "home_cover edge (classic circadian mechanism)",
        ),
        (
            "body_clock_east_host_west_visitor_early",
            c1_flag & venue_east,
            "Cell-1 mirror restricted to Eastern-timezone hosts receiving Western "
            "body-clock visitors at a <14:00 ET kickoff -- predicted positive "
            "home_cover edge",
        ),
        (
            "body_clock_west_host_east_visitor_late",
            venue_west & away_east & true_home & late,
            "CONTROL, expect null: Western venue hosting an Eastern-body-clock visitor "
            "at a >=19:00 ET kickoff (~4pm biological, no circadian disadvantage)",
        ),
        (
            "body_clock_west_road_early_2009_2016",
            c1_flag & (df["season"] < ERA_SPLIT_YEAR),
            "Era stability split of body_clock_west_road_early, seasons 2009-2016 -- "
            "predicted positive home_cover edge",
        ),
        (
            "body_clock_west_road_early_2017_2025",
            c1_flag & (df["season"] >= ERA_SPLIT_YEAR),
            "Era stability split of body_clock_west_road_early, seasons 2017-2025 -- "
            "predicted positive home_cover edge",
        ),
        (
            "body_clock_west_road_midday_control",
            away_west & true_home & midday,
            "Dose-response control: Western body-clock road team at a 14:00-16:59 ET "
            "kickoff (~12-1pm biological) -- predicted positive but WEAKER than "
            "body_clock_west_road_early if the mechanism is kickoff-time-specific",
        ),
    ]

    cells: dict[str, dict[str, Any]] = {}
    for name, flag, mechanism in specs:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "description": f"{mechanism} (pregame-safe schedule fact, no leakage caveat).",
        }
    expected = 6
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
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
    subset_cover = float(work.loc[work["_flag"], "home_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "home_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
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


def score_cell(
    df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    flag = spec["flag"]
    missing_mask = spec["missing_mask"]
    week_blocked = summarize(df, flag=flag, block_col="week_block", samples=samples, seed=seed)
    season_blocked = summarize(df, flag=flag, block_col="season", samples=samples, seed=seed)
    return {
        "name": name,
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing_mask.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "body_clock_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.coords} ===")
    coords = load_coords(args.coords)
    df = load_population(args.schedules, coords)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, "
        f"scored population: {len(df)}, "
        f"unresolved stadium names: {df.attrs['n_unresolved_stadium_names']}"
    )

    cells = build_cells(df)

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"n_missing_required_data={cell['n_missing_required_data']}"
        )
        print(
            f"  subset_cover={wb['subset_cover']:.4f} "
            f"complement_cover={wb['complement_cover']:.4f} "
            f"raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )

    ranked = sorted(
        (r for r in results if not r["week_blocked"].get("insufficient_data")),
        key=lambda r: abs(r["week_blocked"]["full_slate_effect_pts"]),
        reverse=True,
    )
    print("\n=== ranked by |full-slate effect|, week-blocked primary ===")
    for rank, cell in enumerate(ranked, start=1):
        wb = cell["week_blocked"]
        print(
            f"{rank}. {cell['name']:<48} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    configuration = {
        "command": "body-clock-screen",
        "schedules": str(args.schedules),
        "coords": str(args.coords),
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
        "n_scored_population": len(df),
        "n_unresolved_stadium_names": df.attrs["n_unresolved_stadium_names"],
        "predeclaration": "docs/body_clock_screen.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="body-clock-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (6 predeclared ENV-06 circadian "
            "body-clock cells); mined family, every cell predeclared to record "
            "unresolved_below_power via nfl-ats weak-signals record regardless of "
            "interval shape (AGENTS.md); correlated with "
            "bias_battery_west_coast_early_kickoff(_opener), never pool as independent."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
