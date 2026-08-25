"""Altitude-adaptation lead-generation screen: 6 predeclared game-level
altitude-deficit cells, screened against the spread on 2009-2025 NFL REG
games with a week-blocked bootstrap (season-blocked secondary), a full-slate
scaled effect, seeded and deterministic.

Modeled directly on ``scripts/nfl_travel_rest_battery_screen.py``: same
population construction (REG 2009-2025, ``add_ats_outcomes`` for
``home_cover``, pushes dropped), same ``block_bootstrap_two_group`` joint
week-blocked bootstrap, same full-slate effect scaling, same
``probability_positive`` definition, measure-only posture. The full
predeclaration (exact subset definitions, mechanism, predicted direction,
threshold justification, borderline-classification disclosure) is frozen in
``docs/altitude_screen.md``, written before this script scored anything.

**No leakage caveat applies.** Venue elevation and the visitor's modal
home-stadium elevation are static reference facts about known scheduled
venues; ``div_game`` is a pregame-known scheduling fact. No game-time
actuals are used anywhere.

**New reference data**: ``registry/stadium_elevations.json`` (built this
session; no prior stadium elevation table existed in this repo), keyed by
the exact ``stadium`` string in ``schedules.parquet``. Denver entries are
the documented 5,280 ft figure and Azteca is the Wikipedia-infobox 2,241 m;
all other entries are approximate reported general knowledge (labelled
``inferred`` in the file).

**Measure-only.** Never writes to ``registry/weak_signals.json``; recording
happens via separate explicit ``nfl-ats weak-signals record`` calls against
this script's output JSON, every numeric field passed through unmodified,
every cell predeclared to record ``unresolved_below_power`` regardless of
interval shape (AGENTS.md: an interval containing zero is never a rejection).
Writes an automatic low-stakes provenance stamp to
``registry/experiments/altitude-screen/`` via ``write_experiment_artifact``.

Data: newest ``data/raw/*/schedules.parquet`` snapshot only.

Writes JSON to ``artifacts/altitude_screen/<UTC timestamp>/results.json`` and
prints a summary table to stdout.
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

sys.path.append(str(REPO / "scripts"))

from _common import (  # noqa: E402
    default_schedules,
    summarize,
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
    "location",
    "stadium",
    "div_game",
]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025

ALTITUDE_DEFICIT_FT = 4000.0
M_TO_FT = 100.0 / 30.48

ERA_SPLIT_SEASON = 2018

DENVER_STADIUMS = {
    "Invesco Field at Mile High",
    "Sports Authority Field at Mile High",
    "Empower Field at Mile High",
}

AFC_TEAMS = {
    "BUF",
    "MIA",
    "NE",
    "NYJ",
    "BAL",
    "CIN",
    "CLE",
    "PIT",
    "HOU",
    "IND",
    "JAX",
    "TEN",
    "DEN",
    "KC",
    "OAK",
    "LV",
    "SD",
    "LAC",
}

DEFAULT_ELEVATIONS_PATH = REPO / "registry" / "stadium_elevations.json"


def load_elevations(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_population(schedules_path: Path, elevations: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Load, filter to REG 2009-2025, compute home_cover (pushes dropped),
    and attach every derived altitude column the 6 cells need."""

    raw = pd.read_parquet(schedules_path)
    available = [c for c in DEFAULT_SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df["week_block"] = df["season"] * 100 + df["week"]
    if "div_game" in df.columns:
        df["is_division"] = (
            pd.to_numeric(df["div_game"], errors="coerce").fillna(0).astype(float) > 0
        )
    else:
        df["is_division"] = False

    home_rows = df.loc[df["location"] == "Home"]
    modal_stadium = (
        home_rows.groupby(["home_team", "season"])["stadium"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("team_home_stadium")
    )
    df = df.merge(modal_stadium, left_on=["away_team", "season"], right_index=True, how="left")

    def elev_ft(name: object) -> float:
        entry = elevations.get(name) if isinstance(name, str) else None
        return float(entry["elevation_ft"]) if entry else np.nan

    df["venue_elev_ft"] = df["stadium"].map(elev_ft)
    df["away_home_elev_ft"] = df["team_home_stadium"].map(elev_ft)
    df["altitude_deficit_ft"] = df["venue_elev_ft"] - df["away_home_elev_ft"]

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    df.attrs["n_unresolved_stadium_names"] = int(
        df.loc[df["venue_elev_ft"].isna(), "stadium"].nunique()
    )
    return df


def build_cells(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the 6 predeclared cells. Each entry carries an optional
    ``population`` mask restricting the scored frame (era splits), a boolean
    ``flag`` Series aligned to the FULL frame, and a ``missing_mask``."""
    deficit_missing = df["altitude_deficit_ft"].isna()
    deficit_flag = (df["altitude_deficit_ft"] >= ALTITUDE_DEFICIT_FT).fillna(False)

    cells: list[dict[str, Any]] = []

    cells.append(
        {
            "name": "altitude_deficit_4000ft",
            "population": pd.Series(True, index=df.index),
            "flag": deficit_flag,
            "missing_mask": deficit_missing,
            "description": (
                f"Venue elevation minus away team's modal home-stadium elevation "
                f">= {ALTITUDE_DEFICIT_FT:g} ft -- predicted positive home_cover edge "
                "(acute hypoxic disadvantage for near-sea-level visitors into "
                "Denver/Mexico City)"
            ),
        }
    )

    cells.append(
        {
            "name": "altitude_deficit_4000ft_division",
            "population": pd.Series(True, index=df.index),
            "flag": deficit_flag & df["is_division"],
            "missing_mask": deficit_missing | ~df["is_division"].notna(),
            "description": (
                "Cell-1 flag restricted to division games (repeat exposure: division "
                "visitors face Denver twice/year, chronic adaptation blunts the acute "
                "deficit) -- predicted positive but smaller than cell 1"
            ),
        }
    )

    den_home = df["home_team"] == "DEN"
    afc_visitor = df["away_team"].isin(AFC_TEAMS)
    cells.append(
        {
            "name": "den_home_vs_own_conference",
            "population": pd.Series(True, index=df.index),
            "flag": den_home & afc_visitor,
            "missing_mask": pd.Series(False, index=df.index),
            "description": (
                "Denver home game with an AFC visitor (Denver's own conference; these "
                "opponents visit far more often than NFC ones, so more chronic "
                "adaptation) -- predicted negative home_cover edge relative to complement"
            ),
        }
    )

    azteca = df["stadium"] == "Azteca Stadium"
    cells.append(
        {
            "name": "mexico_city_neutral",
            "population": pd.Series(True, index=df.index),
            "flag": azteca,
            "missing_mask": pd.Series(False, index=df.index),
            "description": (
                "Game at Estadio Azteca (2,241 m; thin by construction, n reported "
                "honestly) -- predicted positive home_cover edge for the designated "
                "home side"
            ),
        }
    )

    early_era = df["season"] < ERA_SPLIT_SEASON
    late_era = ~early_era
    cells.append(
        {
            "name": "altitude_deficit_4000ft_era_2009_2017",
            "population": early_era,
            "flag": deficit_flag & early_era,
            "missing_mask": deficit_missing & early_era,
            "description": (
                "Cell-1 flag restricted to seasons 2009-2017 (early era, plausibly "
                "coarser market pricing of altitude) -- predicted positive"
            ),
        }
    )
    cells.append(
        {
            "name": "altitude_deficit_4000ft_era_2018_2025",
            "population": late_era,
            "flag": deficit_flag & late_era,
            "missing_mask": deficit_missing & late_era,
            "description": (
                "Cell-1 flag restricted to seasons 2018-2025 (attenuated vs the early "
                "era if the market learned; descriptive era split, both windows "
                "recorded)"
            ),
        }
    )

    assert len(cells) == 6, f"expected 6 predeclared cells, got {len(cells)}"
    return cells


def score_cell(
    df: pd.DataFrame, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    pop = spec["population"].to_numpy()
    sub = df.loc[pop].reset_index(drop=True)
    flag = spec["flag"].loc[pop].reset_index(drop=True)
    missing = spec["missing_mask"].loc[pop]

    week_blocked = summarize(sub, flag=flag, block_col="week_block", samples=samples, seed=seed)
    season_blocked = summarize(sub, flag=flag, block_col="season", samples=samples, seed=seed)

    return {
        "name": spec["name"],
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing.sum()),
        "n_population": int(pop.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--elevations", type=Path, default=DEFAULT_ELEVATIONS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "altitude_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.elevations} ===")
    elevations = load_elevations(args.elevations)
    df = load_population(args.schedules, elevations)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, "
        f"scored population: {len(df)}, "
        f"unresolved stadium names: {df.attrs['n_unresolved_stadium_names']}"
    )

    cells = build_cells(df)

    results = []
    for spec in cells:
        print(f"\n=== {spec['name']} ===")
        cell = score_cell(df, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_population={cell['n_population']} "
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
            f"{rank}. {cell['name']:<44} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    configuration = {
        "command": "altitude-screen",
        "schedules": str(args.schedules),
        "elevations": str(args.elevations),
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
        "predeclaration": "docs/altitude_screen.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="altitude-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (6 predeclared altitude-adaptation "
            "cells); mined family, every cell predeclared to record "
            "unresolved_below_power via separate nfl-ats weak-signals record calls "
            "regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
