"""ENV-06 circadian body-clock NIGHT screen: 9 predeclared night-window x
body-clock cells on REG 2009-2025 NFL games, testing the evening half of the
Smith et al. (Sleep 2013) lead that the early-window screen deliberately did
not test. Week-blocked bootstrap primary, season-blocked secondary,
full-slate-scaled accuracy_points effects.

Predeclaration frozen in ``docs/body_clock_night_screen.md`` before any cover
rate was computed. Measure-only: never writes registry JSON; stamps a run log
to ``registry/experiments/body-clock-night-screen/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from body_clock_screen import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_COORDS_PATH,
    DEFAULT_SCHEDULES,
    EAST_TZ,
    SEASON_END,
    SEASON_START,
    WEST_TZS,
    artifact_provenance,
    load_coords,
    load_population,
    score_cell,
    write_experiment_artifact,
)

NIGHT_KICK_MIN_MIN = 20 * 60
TRUE_NIGHT_WEEKDAYS = {"Sunday", "Monday", "Thursday"}
ERA_SPLIT_YEAR = 2017

DOSE_BUCKETS = [
    ("dose_1300", None, 14 * 60),
    ("dose_1400_1659", 14 * 60, 17 * 60),
    ("dose_1700_1959", 17 * 60, 20 * 60),
    ("dose_ge2000", 20 * 60, None),
]


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    away_west = df["away_body_tz"].isin(WEST_TZS)
    away_east = df["away_body_tz"] == EAST_TZ
    true_home = df["location"] == "Home"
    night = df["kick_min"] >= NIGHT_KICK_MIN_MIN
    weekday = pd.to_datetime(df["gameday"]).dt.day_name()
    era_old = df["season"] < ERA_SPLIT_YEAR
    era_new = df["season"] >= ERA_SPLIT_YEAR

    missing = (
        df["kick_min"].isna()
        | df["away_body_tz"].isna()
        | df["venue_tz"].isna()
        | df["location"].isna()
    )

    west_night = away_west & true_home & night
    specs: list[tuple[str, pd.Series, str]] = [
        (
            "body_clock_night_west_road_ge2000et",
            west_night,
            "PRIMARY: away body clock Pacific/Arizona, true road game, kickoff "
            ">= 20:00 ET (SNF/MNF window) -- predicted NEGATIVE home_cover gap "
            "(west side covers; circadian-peak mechanism, Smith et al. Sleep "
            "2013 evening effect)",
        ),
        (
            "body_clock_night_west_road_true_slots",
            west_night & weekday.isin(TRUE_NIGHT_WEEKDAYS),
            "Cell 1 restricted to true night TV slots (Sun/Mon/Thu), excluding "
            "19:xx borderline and Sat/Tue/Fri/Wed late games -- predicted "
            "NEGATIVE home_cover gap",
        ),
        (
            "body_clock_night_east_road_ge2000et",
            away_east & true_home & night,
            "MIRROR CONTROL: Eastern body-clock road team at >= 20:00 ET "
            "kickoff, visitor past biological bedtime -- predicted POSITIVE "
            "home_cover gap (host benefits); includes West-venue games where "
            "the bedtime rationale is diluted, disclosed conservative mixing",
        ),
        (
            "body_clock_night_west_road_ge2000et_2009_2016",
            west_night & era_old,
            "Era stability split of the primary cell, seasons 2009-2016 -- "
            "predicted NEGATIVE home_cover gap",
        ),
        (
            "body_clock_night_west_road_ge2000et_2017_2025",
            west_night & era_new,
            "Era stability split of the primary cell, seasons 2017-2025 -- "
            "predicted NEGATIVE home_cover gap",
        ),
    ]
    for name, lo, hi in DOSE_BUCKETS:
        if lo is None:
            bucket = df["kick_min"] < hi
        elif hi is None:
            bucket = df["kick_min"] >= lo
        else:
            bucket = (df["kick_min"] >= lo) & (df["kick_min"] < hi)
        labels = {
            "dose_1300": "kickoff < 14:00 ET, predicted POSITIVE home_cover gap "
            "(west disadvantaged; replicates the recorded early-screen primary)",
            "dose_1400_1659": "14:00 <= kickoff < 17:00 ET, predicted weaker positive / near null",
            "dose_1700_1959": "17:00 <= kickoff < 20:00 ET, n=3 measured -- no "
            "directional claim possible, ladder completeness only",
            "dose_ge2000": "kickoff >= 20:00 ET, identical flag set to the "
            "primary cell by construction (internal consistency check)",
        }
        specs.append(
            (
                f"body_clock_night_west_road_{name}",
                away_west & true_home & bucket,
                f"DOSE bucket {name}: {labels[name]}",
            )
        )

    cells: dict[str, dict[str, Any]] = {}
    for name, flag, mechanism in specs:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "description": f"{mechanism} (pregame-safe schedule fact).",
        }
    expected = 9
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


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
    output_dir: Path = args.output or (REPO / "artifacts" / "body_clock_night_screen" / timestamp)
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
            f"{rank}. {cell['name']:<52} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    configuration = {
        "command": "body-clock-night-screen",
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
        "predeclaration": "docs/body_clock_night_screen.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="body-clock-night-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (9 predeclared circadian "
            "NIGHT-window body-clock cells, the evening half of the Smith et al. "
            "Sleep 2013 lead the ENV-06 early screen did not test); every cell to "
            "be recorded unresolved_below_power via nfl-ats weak-signals record "
            "regardless of interval shape (AGENTS.md); correlated with all six "
            "body_clock_* entries and bias_battery_west_coast_early_kickoff(_opener), "
            "never pool as independent."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
