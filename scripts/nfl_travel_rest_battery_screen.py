"""ENV-03/ENV-04 lead-generation battery: 8 predeclared game-level travel-
geometry and rest-context cells, screened against the spread on 2009-2025
NFL REG games with a week-blocked bootstrap (season-blocked secondary), a
full-slate scaled effect, seeded and deterministic.

Modeled directly on ``scripts/nfl_weather_battery_screen.py`` /
``scripts/nfl_weather_followup_screen.py``: same population construction
(REG 2009-2025, ``add_ats_outcomes`` for ``home_cover``, pushes dropped),
same ``block_bootstrap_two_group`` joint week-blocked bootstrap, same
full-slate effect scaling, same ``probability_positive`` definition, same
measure-only/never-writes-the-registry posture. The full predeclaration
(exact subset definitions, mechanism, predicted direction, threshold
justification, duplicate-avoidance table) is frozen in
``docs/travel_rest_battery.md``, written before this script scored anything.

**Unlike the weather batteries, no leakage caveat applies here.**
``home_rest``/``away_rest``/``location``/``stadium``/``weekday`` are
pregame-known schedule facts (not game-time actuals), and stadium geometry
(lat/lon/timezone) is a static reference fact about a known, scheduled
venue -- every cell in this battery is genuinely pregame-safe on its own
terms.

**New reference data**: ``registry/stadium_coordinates.json`` (built this
session, no prior stadium lat/lon table existed in this repo), keyed by the
exact ``stadium`` string in ``schedules.parquet`` (NOT ``stadium_id``, which
is unreliable for neutral-site games -- see the coordinate file's ``_README``
and ``docs/travel_rest_battery.md``). Great-circle distances use the
haversine formula; timezone offsets are computed per-game-date via stdlib
``zoneinfo`` (DST-aware).

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``) and never edits a tracked doc; recording a
finding there happens via a separate, explicit ``nfl-ats weak-signals
record`` invocation (see ``scripts/record_travel_rest_battery.py``) against
this script's output JSON, with every numeric field passed through
unmodified. It DOES write an automatic, low-stakes experiment-provenance
stamp to ``registry/experiments/`` via ``write_experiment_artifact`` -- a run
log, not a verdict; it carries no closing-ground and asserts nothing about
any cell's status. Per AGENTS.md, an interval containing zero is never a
rejection -- this battery's purpose is to surface category-3 leads, and
every cell here is predeclared to record ``unresolved_below_power``
regardless of shape (mined, uncorrected multiplicity, 8 cells).

Data: newest ``data/raw/*/schedules.parquet`` snapshot only.

Writes JSON to ``artifacts/travel_rest_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "weekday",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "home_rest",
    "away_rest",
    "location",
    "stadium",
    "stadium_id",
]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
SEASON_START = 2009
SEASON_END = 2025

EARTH_RADIUS_MI = 3958.8

# Predeclared, round, externally-justified thresholds (docs/travel_rest_battery.md).
LONG_DISTANCE_MI = 1500.0
EASTBOUND_HOURS = 2.0
RETURN_TRIP_MI = 1500.0
RETURN_TRIP_MAX_HOME_REST = 8
OFF_BYE_REST_DAYS = 13
SHORT_WEEK_REST_DAYS = 5

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


def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def load_population(schedules_path: Path, coords: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Load, filter to REG 2009-2025, compute home_cover (pushes dropped),
    and attach every derived travel/rest column the 8 cells need. One row
    per game.
    """

    raw = pd.read_parquet(schedules_path)
    available = [c for c in DEFAULT_SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)  # adds ats_margin, home_cover (reused verbatim)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    pushes_or_missing = n_before_push_drop - len(df)

    df["home_rest"] = pd.to_numeric(df.get("home_rest"), errors="coerce")
    df["away_rest"] = pd.to_numeric(df.get("away_rest"), errors="coerce")
    df["week_block"] = df["season"] * 100 + df["week"]
    df["gameday_dt"] = pd.to_datetime(df["gameday"], errors="coerce")

    # Away team's own modal home stadium THAT SEASON, same convention as
    # away_modal_roof/away_modal_surface in the weather batteries -- resolves
    # relocations (STL->LA, SD->LAC, OAK->LV) automatically from the schedule.
    home_rows = df.loc[df["location"] == "Home"]
    modal_stadium = (
        home_rows.groupby(["home_team", "season"])["stadium"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("team_home_stadium")
    )
    df = df.merge(modal_stadium, left_on=["away_team", "season"], right_index=True, how="left")

    def coord(name: object) -> dict[str, Any] | None:
        return coords.get(name) if isinstance(name, str) else None

    venue = df["stadium"].map(coord)
    away_home = df["team_home_stadium"].map(coord)

    df["venue_lat"] = venue.map(lambda c: c["lat"] if c else np.nan)
    df["venue_lon"] = venue.map(lambda c: c["lon"] if c else np.nan)
    df["venue_tz"] = venue.map(lambda c: c["tz"] if c else None)
    df["away_home_lat"] = away_home.map(lambda c: c["lat"] if c else np.nan)
    df["away_home_lon"] = away_home.map(lambda c: c["lon"] if c else np.nan)
    df["away_home_tz"] = away_home.map(lambda c: c["tz"] if c else None)

    geo_ok = df["venue_lat"].notna() & df["away_home_lat"].notna()
    df["away_travel_mi"] = np.nan
    df.loc[geo_ok, "away_travel_mi"] = [
        haversine_mi(a, b, c, d)
        for a, b, c, d in zip(
            df.loc[geo_ok, "away_home_lat"],
            df.loc[geo_ok, "away_home_lon"],
            df.loc[geo_ok, "venue_lat"],
            df.loc[geo_ok, "venue_lon"],
            strict=True,
        )
    ]

    tz_ok = df["venue_tz"].notna() & df["away_home_tz"].notna() & df["gameday_dt"].notna()
    venue_offset = pd.Series(np.nan, index=df.index)
    away_offset = pd.Series(np.nan, index=df.index)
    for idx in df.index[tz_ok]:
        gd = df.at[idx, "gameday_dt"].to_pydatetime()
        venue_utc = ZoneInfo(df.at[idx, "venue_tz"]).utcoffset(gd)
        away_utc = ZoneInfo(df.at[idx, "away_home_tz"]).utcoffset(gd)
        assert venue_utc is not None and away_utc is not None
        venue_offset.at[idx] = venue_utc.total_seconds() / 3600.0
        away_offset.at[idx] = away_utc.total_seconds() / 3600.0
    df["tz_delta_eastbound"] = venue_offset - away_offset

    # Team-perspective long table for the return-trip-hangover cell: each
    # team's OWN travel distance in EACH of its games this season, shifted
    # by 1 within (team, season) to get "previous game's own travel", then
    # joined back onto the HOME side of each game.
    def team_home_coord(team: str, season: int) -> dict[str, Any] | None:
        name = modal_stadium.get((team, season))
        return coords.get(name) if isinstance(name, str) else None

    long_rows = []
    for _, g in df.iterrows():
        v = coord(g["stadium"])
        for team in (g["home_team"], g["away_team"]):
            hc = team_home_coord(team, g["season"])
            if v is None or hc is None:
                dist = np.nan
            else:
                dist = haversine_mi(hc["lat"], hc["lon"], v["lat"], v["lon"])
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "team": team,
                    "gameday_dt": g["gameday_dt"],
                    "own_travel_mi": dist,
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    long_df["prev_own_travel_mi"] = long_df.groupby(["team", "season"])["own_travel_mi"].shift(1)
    home_prev = df[["game_id", "home_team"]].merge(
        long_df[["game_id", "team", "prev_own_travel_mi"]],
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )[["game_id", "prev_own_travel_mi"]]
    df = df.merge(home_prev, on="game_id", how="left")

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    df.attrs["n_unresolved_stadium_names"] = int(
        df.loc[df["venue_lat"].isna(), "stadium"].nunique()
    )
    return df


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return {name: {"flag": bool Series, "description": str, "missing_mask":
    bool Series}}. ``missing_mask`` marks rows where a required input for
    THIS cell was missing/undefined (flag is forced False on those rows,
    they are counted and reported, not dropped from the population).
    """

    cells: dict[str, dict[str, Any]] = {}

    def add(name: str, flag: pd.Series, missing: pd.Series, mechanism: str) -> None:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "description": f"{mechanism} (pregame-safe schedule/geometry fact, no leakage caveat).",
        }

    # 1. long_distance_road
    dist_missing = df["away_travel_mi"].isna()
    add(
        "travel_rest_long_distance_road",
        df["away_travel_mi"] >= LONG_DISTANCE_MI,
        dist_missing,
        f"Away team's great-circle travel distance to this game's venue >= "
        f"{LONG_DISTANCE_MI:g} miles (from away team's own modal home stadium that "
        "season) -- predicted positive home_cover edge (away travel fatigue)",
    )

    # 2. eastbound_multizone
    tz_missing = df["tz_delta_eastbound"].isna()
    add(
        "travel_rest_eastbound_multizone",
        df["tz_delta_eastbound"] >= EASTBOUND_HOURS,
        tz_missing,
        f"This game's venue UTC offset minus away team's own home UTC offset >= "
        f"{EASTBOUND_HOURS:g} hours (eastbound body-clock disadvantage, DST-aware via "
        "gameday) -- predicted positive home_cover edge",
    )

    # 3. international_game
    add(
        "travel_rest_international_game",
        df["location"] == "Neutral",
        pd.Series(False, index=df.index),
        "Game played at a neutral/international site (location=='Neutral') -- "
        "designated home team not actually at its true home venue -- predicted "
        "negative home_cover edge",
    )

    # 4. return_trip_hangover
    return_missing = df["prev_own_travel_mi"].isna() | df["home_rest"].isna()
    add(
        "travel_rest_return_trip_hangover",
        (df["prev_own_travel_mi"] >= RETURN_TRIP_MI)
        & (df["home_rest"] <= RETURN_TRIP_MAX_HOME_REST),
        return_missing,
        f"Home team's OWN travel distance in its immediately preceding game this "
        f"season >= {RETURN_TRIP_MI:g} miles AND home_rest <= "
        f"{RETURN_TRIP_MAX_HOME_REST} (excludes bye-reset cases) -- predicted "
        "negative home_cover edge (fatigue hangover from the team's own prior trip)",
    )

    # 5. home_off_bye
    home_bye_missing = df["home_rest"].isna()
    add(
        "travel_rest_home_off_bye",
        df["home_rest"] >= OFF_BYE_REST_DAYS,
        home_bye_missing,
        f"home_rest >= {OFF_BYE_REST_DAYS} days (side-specific absolute threshold, "
        "captures both true byes and MNF-to-Sunday extra-rest turnarounds) -- "
        "predicted positive home_cover edge",
    )

    # 6. away_off_bye
    away_bye_missing = df["away_rest"].isna()
    add(
        "travel_rest_away_off_bye",
        df["away_rest"] >= OFF_BYE_REST_DAYS,
        away_bye_missing,
        f"away_rest >= {OFF_BYE_REST_DAYS} days (side-specific absolute threshold, "
        "mirror of cell 5 on the away side) -- predicted negative home_cover edge",
    )

    # 7. short_week_road
    short_week_missing = df["away_rest"].isna()
    add(
        "travel_rest_short_week_road",
        df["away_rest"] <= SHORT_WEEK_REST_DAYS,
        short_week_missing,
        f"away_rest <= {SHORT_WEEK_REST_DAYS} days (game-level, side-specific: does "
        "short rest cost more specifically when it is the traveling side) -- "
        "predicted positive home_cover edge",
    )

    # 8. thursday_pure
    add(
        "travel_rest_thursday_pure",
        df["weekday"] == "Thursday",
        df["weekday"].isna(),
        "Thursday game, regardless of venue/weather (plain calendar effect, distinct "
        "from weather_battery_thursday_outdoor_cold's outdoor+cold compounding) -- "
        "predicted positive home_cover edge",
    )

    expected = 8
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
    """Vectorized joint block bootstrap of ``100*(subset_mean-complement_mean)``.

    Reused verbatim (same algorithm) from
    ``scripts/nfl_weather_battery_screen.py::block_bootstrap_two_group``.
    """

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
    output_dir: Path = args.output or (REPO / "artifacts" / "travel_rest_battery" / timestamp)
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
        "command": "nfl-travel-rest-battery-screen",
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
        "predeclaration": "docs/travel_rest_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="nfl-travel-rest-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (8 predeclared ENV-03/ENV-04 travel-"
            "geometry and rest-context cells); mined family, every cell predeclared to "
            "record unresolved_below_power via a separate nfl-ats weak-signals record "
            "call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
