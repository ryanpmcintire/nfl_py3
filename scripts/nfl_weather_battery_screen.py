"""Lead-generation weather/environment bias battery: 8 predeclared game-level
weather and environment checks, screened against the spread on 2009-2025 REG
games with a week-blocked bootstrap (season-blocked secondary), a full-slate
scaled effect, seeded and deterministic.

**Critical leakage caveat, true of every cell in this battery**: the
``schedules`` parquet's ``temp``/``wind`` columns are GAME-TIME ACTUALS, not
pregame forecasts. A pregame model may not consume them directly. Every cell
here is a MECHANISM SCREEN -- an upper bound on what a forecast-time feature
(e.g. a future ENV-01) could ever capture, not itself a usable predictor.

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``) and never edits a tracked doc; recording a
finding there happens via a separate, explicit ``nfl-ats weak-signals
record`` invocation against this script's output. It DOES write an
automatic, low-stakes experiment-provenance stamp to
``registry/experiments/`` via ``write_experiment_artifact`` (RWB-09) -- a
run log, not a verdict; it carries no closing-ground and asserts nothing
about any cell's status. Per AGENTS.md, an interval containing zero is never
a rejection -- this battery's purpose is to surface category-3 leads, not to
prove any one of them, and every cell here is predeclared to record
``unresolved_below_power`` regardless of shape (mined, uncorrected
multiplicity, 8 cells).

The full predeclaration (hypothesis, exact subset definition, mechanism,
direction, warm-metro list, surface normalization) is frozen in
``<scratchpad>/agent_weather/predeclaration.json`` and was written before
this script scored anything. This module implements exactly what that
document specifies.

**Method**, reused verbatim from ``scripts/nfl_bias_battery_screen.py``: the
same ``block_bootstrap_two_group`` joint week-blocked bootstrap generalized
to an arbitrary boolean subset flag vs. its complement, the same full-slate
effect scaling (raw gap x fraction-of-slate), and the same
``probability_positive`` definition. This script differs from the precedent
in two ways: (1) it screens at the GAME level (one row per game, home-cover
outcome) rather than the team-side long table, because none of these 8
hypotheses are framed as a team-perspective flag -- they are plain
subset-vs-complement conditions on the game itself; (2) it adds a
season-blocked secondary bootstrap alongside the primary week-blocked one, as
a robustness check reported in notes/console (not the registry interval).

Cover computation is reused verbatim from
``nfl_ats.features.add_ats_outcomes`` (ats_margin = result - spread_line;
home_cover = 1.0/0.0/NaN-on-push); this script does not reimplement ATS
semantics.

Data: newest ``data/raw/*/schedules.parquet`` snapshot only (no
``game_features.parquet`` join needed -- every column used is already in
schedules). REG season only, 2009-2025.

Writes JSON to ``artifacts/nfl_weather_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
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

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "weekday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
    "stadium_id",
    "location",
]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
SEASON_START = 2009
SEASON_END = 2025

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
DOME_CLOSED_ROOFS = frozenset({"dome", "closed"})
GRASS_SURFACES = frozenset({"grass", "dessograss"})
TURF_SURFACES = frozenset(
    {"fieldturf", "sportturf", "matrixturf", "astroturf", "a_turf", "astroplay"}
)
# Documented, static, state-level "warm winter metro" list -- see
# predeclaration.json for the FL/AZ/CA/TX/LA/NV rationale and the SF/OAK
# limitation. Applied to each game's AWAY team's *own per-season* code, so
# STL (2009-2015, Missouri, not warm) is correctly excluded and only the
# post-2016 "LA" code (the Rams' actual relocation) is tagged warm.
WARM_METRO_TEAM_CODES = frozenset(
    {
        "MIA",
        "TB",
        "JAX",
        "ARI",
        "SF",
        "OAK",
        "LA",
        "LAC",
        "SD",
        "HOU",
        "DAL",
        "NO",
        "LV",
    }
)
ALTITUDE_CITY_CODES = frozenset({"DEN"})
LEAKAGE_CAVEAT = (
    "actual-weather mechanism screen, NOT pregame-available; upper bound for a "
    "forecast-time feature"
)


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()


def _normalize_surface(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in GRASS_SURFACES:
        return "grass"
    if value in TURF_SURFACES:
        return "turf"
    return None


def load_population(schedules_path: Path) -> pd.DataFrame:
    """Load, filter to REG 2009-2025, compute home_cover (pushes dropped),
    and attach every derived column the 8 cells need. One row per game.
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

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["roof_norm"] = df["roof"].where(df["roof"].isin(OUTDOOR_ROOFS | DOME_CLOSED_ROOFS))
    df["outdoor"] = df["roof"].isin(OUTDOOR_ROOFS)
    df["surface_norm"] = df["surface"].map(_normalize_surface)
    df["week_block"] = df["season"] * 100 + df["week"]
    df["stadium"] = df["stadium"].fillna("")

    # Per (team, season) modal home roof/surface, computed on the FULL REG
    # 2009-2025 home-game population (roof/surface is a stadium fact, not a
    # cover outcome, so it is unaffected by the push drop above; using the
    # full home-game set avoids thinning an already-small per-season sample).
    modal_roof = (
        df.groupby(["home_team", "season"])["roof"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("away_modal_roof")
    )
    modal_surface = (
        df.assign(surface_norm=df["surface_norm"])
        .groupby(["home_team", "season"])["surface_norm"]
        .agg(lambda s: s.mode().iat[0] if not s.mode(dropna=True).empty else None)
        .rename("away_modal_surface")
    )
    df = df.merge(
        modal_roof,
        left_on=["away_team", "season"],
        right_index=True,
        how="left",
    )
    df = df.merge(
        modal_surface,
        left_on=["away_team", "season"],
        right_index=True,
        how="left",
    )

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
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
            "description": f"{mechanism} ({LEAKAGE_CAVEAT}).",
        }

    outdoor = df["outdoor"]

    # 1. high_wind_outdoor
    wind_missing = df["wind"].isna() | df["roof"].isna()
    add(
        "weather_battery_high_wind_outdoor",
        outdoor & (df["wind"] >= 15),
        wind_missing,
        "Outdoor/open roof AND wind >= 15mph -- plain subset vs. complement, unsigned "
        "(no home/away asymmetry in the flag)",
    )

    # 2. high_wind_road_favorite
    add(
        "weather_battery_high_wind_road_favorite",
        outdoor & (df["wind"] >= 15) & (df["spread_line"] < 0),
        wind_missing | df["spread_line"].isna(),
        "Outdoor/open roof AND wind >= 15mph AND away team favored (spread_line<0, "
        "verified home-favored-when-positive convention) -- predicted home_cover edge",
    )

    # 3. extreme_cold
    temp_missing = df["temp"].isna() | df["roof"].isna()
    add(
        "weather_battery_extreme_cold",
        outdoor & (df["temp"] <= 25),
        temp_missing,
        "Outdoor/open roof AND temp <= 25F -- generic cold home-field-edge lore, "
        "predicted home_cover edge",
    )

    # 4. dome_team_outdoors_cold
    dome_missing = df["away_modal_roof"].isna() | df["temp"].isna() | df["roof"].isna()
    add(
        "weather_battery_dome_team_outdoors_cold",
        df["away_modal_roof"].isin(DOME_CLOSED_ROOFS) & outdoor & (df["temp"] <= 40),
        dome_missing,
        "Away team's modal home roof this season is dome/closed AND this game is "
        "outdoor/open AND temp<=40F -- dome-team cold-mismatch, predicted home_cover edge",
    )

    # 5. warm_team_cold_late
    warm_missing = df["temp"].isna() | df["roof"].isna()
    add(
        "weather_battery_warm_team_cold_late",
        df["away_team"].isin(WARM_METRO_TEAM_CODES)
        & outdoor
        & (df["temp"] <= 35)
        & (df["week"] >= 13),
        warm_missing,
        "Away team's own season code in the static warm-winter-metro list AND outdoor "
        "temp<=35F AND week>=13 -- warm-visitor late-season cold mismatch, predicted "
        "home_cover edge",
    )

    # 6. surface_switch_grass_to_turf
    surface_missing = df["away_modal_surface"].isna() | df["surface_norm"].isna()
    add(
        "weather_battery_surface_switch_grass_to_turf",
        (df["away_modal_surface"] == "grass") & (df["surface_norm"] == "turf"),
        surface_missing,
        "Away team's modal home surface this season normalizes to grass AND this "
        "game's surface normalizes to turf -- footing/speed mismatch, predicted "
        "home_cover edge",
    )

    # 7. high_altitude_road
    at_altitude = (df["home_team"] == "DEN") | df["stadium"].str.contains(
        "Azteca", case=False, na=False
    )
    add(
        "weather_battery_high_altitude_road",
        at_altitude & (~df["away_team"].isin(ALTITUDE_CITY_CODES)),
        pd.Series(False, index=df.index),
        "Game at Denver (or a Mexico City/Azteca neutral-site game) AND away team not "
        "itself from the one altitude-city code (DEN, a no-op gate given NFL "
        "geography, documented) -- altitude-conditioning mismatch, predicted "
        "home_cover edge",
    )

    # 8. thursday_outdoor_cold
    thu_missing = df["temp"].isna() | df["roof"].isna()
    add(
        "weather_battery_thursday_outdoor_cold",
        (df["weekday"] == "Thursday") & outdoor & (df["temp"] <= 35),
        thu_missing,
        "Thursday game AND outdoor/open roof AND temp<=35F -- short-week x cold "
        "compounding, predicted home_cover edge",
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
    ``scripts/nfl_bias_battery_screen.py::block_bootstrap_two_group``: draws
    a multinomial resample of blocks, jointly resamples both arms from the
    same drawn set of blocks each draw, and drops draws where a resampled
    set has zero rows in either arm (reported as ``dropped_draws``).
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "nfl_weather_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, "
        f"scored population: {len(df)}"
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
        "command": "nfl-weather-battery-screen",
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
        "n_scored_population": len(df),
        "predeclaration": "scratchpad/agent_weather/predeclaration.json (frozen before scoring)",
        "leakage_caveat": LEAKAGE_CAVEAT,
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="nfl-weather-battery-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (8 predeclared cells); mined family, "
            "every cell predeclared to record unresolved_below_power via a separate "
            "nfl-ats weak-signals record call regardless of interval shape (AGENTS.md)."
        ),
        # No explicit registry_root: matches nfl_ats.cli._registry_root()'s own
        # convention (Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry"))),
        # which write_experiment_artifact's default already implements -- so a
        # test harness setting NFL_ATS_REGISTRY_DIR still isolates this script.
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
