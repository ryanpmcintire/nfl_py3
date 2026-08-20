"""Second-generation weather/environment follow-up battery: 5 predeclared
game-level cells combining weather with team-habitat and travel context,
screened against the spread on 2009-2025 REG games with a week-blocked
bootstrap (season-blocked secondary), a full-slate scaled effect, seeded and
deterministic.

Modeled directly on ``scripts/nfl_weather_battery_screen.py`` (the
first-generation 8-cell battery): same population construction, same
``block_bootstrap_two_group`` joint week-blocked bootstrap, same full-slate
effect scaling, same ``probability_positive`` definition, same
measure-only/never-writes-the-registry posture. The full predeclaration
(exact subset definitions, mechanism, predicted direction, leakage posture)
is frozen in ``docs/weather_followup.md`` and a machine-readable copy at
``<scratchpad>/agent_weather_followup/predeclaration.json``, written before
this script scored anything.

**Critical leakage caveat, true of every cell that touches THIS game's own
temp/wind/roof** (cells 1, 2, 4, 5; cell 3's wind half): the ``schedules``
parquet's ``temp``/``wind`` columns are GAME-TIME ACTUALS, not pregame
forecasts. Every such cell is a MECHANISM SCREEN -- an upper bound on what a
forecast-time feature (a future ENV-01) could ever capture, not itself a
usable predictor. Cell 3's PRIOR-SEASON pass-rate half is genuinely
season-causal and pregame-safe on its own terms.

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``) and never edits a tracked doc; recording a
finding there happens via a separate, explicit ``nfl-ats weak-signals
record`` invocation (see ``scripts/record_weather_followup.py``) against
this script's output JSON, with every numeric field passed through
unmodified. It DOES write an automatic, low-stakes experiment-provenance
stamp to ``registry/experiments/`` via ``write_experiment_artifact`` -- a run
log, not a verdict; it carries no closing-ground and asserts nothing about
any cell's status. Per AGENTS.md, an interval containing zero is never a
rejection -- this battery's purpose is to surface category-3 leads, and
every cell here is predeclared to record ``unresolved_below_power``
regardless of shape (mined, uncorrected multiplicity, 5 cells).

Data: newest ``data/raw/*/schedules.parquet`` snapshot (weather/roof/surface/
rest) and newest ``data/raw/*/team_stats.parquet`` snapshot (prior-season
team pass rate for cell 3 only).

Writes JSON to ``artifacts/nfl_weather_followup/<UTC timestamp>/results.json``
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
    "away_rest",
    "home_rest",
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
LEAKAGE_CAVEAT_ACTUAL = (
    "actual-weather mechanism screen, NOT pregame-available; upper bound for a "
    "forecast-time feature"
)
LEAKAGE_CAVEAT_MIXED = (
    "actual-weather mechanism screen for the temp/wind half (NOT pregame-available; "
    "upper bound for a forecast-time feature); prior-season pass-rate half is "
    "genuinely season-causal and pregame-safe"
)

# Same-season-aggregate gap thresholds (predeclared, docs/weather_followup.md).
TEMP_GAP_THRESHOLD_F = 25.0
WIND_GAP_TEAM_BASELINE_MAX = 8.0
WIND_GAP_GAME_MIN = 15.0
REST_DISADVANTAGE_COLD_TEMP_MAX = 35.0
SURFACE_SWITCH_COLD_TEMP_MAX = 45.0


def _latest(glob_pattern: str) -> Path:
    candidates = sorted((REPO / "data/raw").glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/{glob_pattern} snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest("*/schedules.parquet")
DEFAULT_TEAM_STATS = _latest("*/team_stats.parquet")


def _normalize_surface(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in GRASS_SURFACES:
        return "grass"
    if value in TURF_SURFACES:
        return "turf"
    return None


def _load_prior_season_pass_rate(team_stats_path: Path) -> pd.DataFrame:
    """Team-season pass rate (attempts / (attempts + carries)), REG games only,
    shifted one full season FORWARD so it is available before the following
    season's games are played (genuinely season-causal, pregame-safe).

    Returns columns [team, season, prior_pass_rate, prior_pass_rate_median]
    where ``season`` is the season the rate APPLIES to (i.e. the prior
    season's stats labeled with next season's number), and
    ``prior_pass_rate_median`` is the cross-team median of that same applied
    season (computed once the shift has happened, so it is also a
    prior-season-only quantity).
    """

    raw = pd.read_parquet(team_stats_path)
    df = raw.loc[raw["season_type"] == "REG", ["team", "season", "attempts", "carries"]].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0.0)
    df["carries"] = pd.to_numeric(df["carries"], errors="coerce").fillna(0.0)
    season_totals = df.groupby(["team", "season"], as_index=False)[["attempts", "carries"]].sum()
    denom = season_totals["attempts"] + season_totals["carries"]
    season_totals["pass_rate"] = np.where(denom > 0, season_totals["attempts"] / denom, np.nan)
    season_totals["applies_to_season"] = season_totals["season"] + 1
    prior = season_totals.loc[:, ["team", "applies_to_season", "pass_rate"]].rename(
        columns={"applies_to_season": "season", "pass_rate": "prior_pass_rate"}
    )
    medians = (
        prior.groupby("season")["prior_pass_rate"]
        .median()
        .rename("prior_pass_rate_median")
        .reset_index()
    )
    return prior.merge(medians, on="season", how="left")


def load_population(schedules_path: Path, team_stats_path: Path) -> pd.DataFrame:
    """Load, filter to REG 2009-2025, compute home_cover (pushes dropped),
    and attach every derived column the 5 cells need. One row per game.
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
    df["away_rest"] = pd.to_numeric(df.get("away_rest"), errors="coerce")
    df["home_rest"] = pd.to_numeric(df.get("home_rest"), errors="coerce")
    df["outdoor"] = df["roof"].isin(OUTDOOR_ROOFS)
    df["surface_norm"] = df["surface"].map(_normalize_surface)
    df["week_block"] = df["season"] * 100 + df["week"]

    # Away team's own same-season climatological-normal temp/wind: mean
    # actual temp/wind across the away team's OWN outdoor home games that
    # season (same full-season-aggregate convention as away_modal_roof /
    # away_modal_surface in the parent script -- disclosed in
    # docs/weather_followup.md, not season-causal, precedented).
    outdoor_home = df.loc[df["outdoor"]]
    team_climate = (
        outdoor_home.groupby(["home_team", "season"])
        .agg(climate_temp=("temp", "mean"), climate_wind=("wind", "mean"))
        .reset_index()
    )
    df = df.merge(
        team_climate,
        left_on=["away_team", "season"],
        right_on=["home_team", "season"],
        how="left",
        suffixes=("", "_climate"),
    )
    if "home_team_climate" in df.columns:
        df = df.drop(columns=["home_team_climate"])

    # away_modal_surface, reused verbatim from the parent script (full REG
    # 2009-2025 home-game population; roof/surface is a stadium fact, not a
    # cover outcome).
    modal_surface = (
        df.assign(surface_norm=df["surface_norm"])
        .groupby(["home_team", "season"])["surface_norm"]
        .agg(lambda s: s.mode().iat[0] if not s.mode(dropna=True).empty else None)
        .rename("away_modal_surface")
    )
    df = df.merge(
        modal_surface,
        left_on=["away_team", "season"],
        right_index=True,
        how="left",
    )

    prior_pass_rate = _load_prior_season_pass_rate(team_stats_path)
    df = df.merge(
        prior_pass_rate,
        left_on=["away_team", "season"],
        right_on=["team", "season"],
        how="left",
    )
    if "team" in df.columns:
        df = df.drop(columns=["team"])

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

    def add(
        name: str, flag: pd.Series, missing: pd.Series, mechanism: str, *, leakage: str
    ) -> None:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "description": f"{mechanism} ({leakage}).",
        }

    outdoor = df["outdoor"]

    # 1. temp_gap_cold_visitor
    temp_gap = df["climate_temp"] - df["temp"]
    temp_gap_missing = df["climate_temp"].isna() | df["temp"].isna() | df["roof"].isna()
    add(
        "weather_followup_temp_gap_cold_visitor",
        outdoor & (temp_gap >= TEMP_GAP_THRESHOLD_F),
        temp_gap_missing,
        "Away team's own climatological-normal outdoor home temp (same-season "
        f"aggregate) minus this game's temp >= {TEMP_GAP_THRESHOLD_F:g}F AND outdoor -- "
        "generalizes dome_team_outdoors_cold + warm_team_cold_late into a continuous "
        "gap, predicted home_cover edge",
        leakage=LEAKAGE_CAVEAT_ACTUAL,
    )

    # 2. wind_gap_visitor
    wind_gap_missing = df["climate_wind"].isna() | df["wind"].isna() | df["roof"].isna()
    add(
        "weather_followup_wind_gap_visitor",
        outdoor
        & (df["climate_wind"] <= WIND_GAP_TEAM_BASELINE_MAX)
        & (df["wind"] >= WIND_GAP_GAME_MIN),
        wind_gap_missing,
        "Away team's own climatological-normal outdoor home wind (same-season "
        f"aggregate) <= {WIND_GAP_TEAM_BASELINE_MAX:g}mph AND this game's wind >= "
        f"{WIND_GAP_GAME_MIN:g}mph AND outdoor -- wind-unaccustomed visitor, predicted "
        "home_cover edge",
        leakage=LEAKAGE_CAVEAT_ACTUAL,
    )

    # 3. high_wind_pass_heavy_visitor
    pass_heavy_missing = df["prior_pass_rate"].isna() | df["wind"].isna() | df["roof"].isna()
    add(
        "weather_followup_high_wind_pass_heavy_visitor",
        outdoor
        & (df["wind"] >= WIND_GAP_GAME_MIN)
        & (df["prior_pass_rate"] > df["prior_pass_rate_median"]),
        pass_heavy_missing,
        f"Outdoor AND wind >= {WIND_GAP_GAME_MIN:g}mph AND away team's PRIOR-season "
        "pass rate (attempts/(attempts+carries)) above that prior season's median -- "
        "pass-heavy visitor, predicted home_cover edge",
        leakage=LEAKAGE_CAVEAT_MIXED,
    )

    # 4. rest_disadvantage_cold
    rest_missing = (
        df["away_rest"].isna() | df["home_rest"].isna() | df["temp"].isna() | df["roof"].isna()
    )
    add(
        "weather_followup_rest_disadvantage_cold",
        (df["away_rest"] < df["home_rest"])
        & outdoor
        & (df["temp"] <= REST_DISADVANTAGE_COLD_TEMP_MAX),
        rest_missing,
        "Away team's rest strictly less than home team's rest AND outdoor AND temp "
        f"<= {REST_DISADVANTAGE_COLD_TEMP_MAX:g}F -- travel/fatigue x cold "
        "compounding, predicted home_cover edge",
        leakage=LEAKAGE_CAVEAT_ACTUAL,
    )

    # 5. surface_switch_x_outdoor_cold
    surf_cold_missing = (
        df["away_modal_surface"].isna()
        | df["surface_norm"].isna()
        | df["temp"].isna()
        | df["roof"].isna()
    )
    add(
        "weather_followup_surface_switch_x_outdoor_cold",
        (df["away_modal_surface"] == "grass")
        & (df["surface_norm"] == "turf")
        & outdoor
        & (df["temp"] <= SURFACE_SWITCH_COLD_TEMP_MAX),
        surf_cold_missing,
        "Away team's modal home surface grass AND this game's surface turf AND "
        f"outdoor AND temp <= {SURFACE_SWITCH_COLD_TEMP_MAX:g}F -- deepens "
        "weather_battery_surface_switch_grass_to_turf with a cold-compounding "
        "interaction, predicted home_cover edge larger than the parent +1.16pt cell",
        leakage=LEAKAGE_CAVEAT_ACTUAL,
    )

    expected = 5
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
    ``scripts/nfl_weather_battery_screen.py::block_bootstrap_two_group``
    (itself reused from ``scripts/nfl_bias_battery_screen.py``): draws a
    multinomial resample of blocks, jointly resamples both arms from the
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
    parser.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "nfl_weather_followup" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.team_stats} ===")
    df = load_population(args.schedules, args.team_stats)
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
        "command": "nfl-weather-followup-screen",
        "schedules": str(args.schedules),
        "team_stats": str(args.team_stats),
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
        "predeclaration": (
            "docs/weather_followup.md + "
            "scratchpad/agent_weather_followup/predeclaration.json (frozen before scoring)"
        ),
        "leakage_caveat_actual": LEAKAGE_CAVEAT_ACTUAL,
        "leakage_caveat_mixed": LEAKAGE_CAVEAT_MIXED,
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="nfl-weather-followup-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation follow-up battery (5 predeclared second-"
            "generation cells combining weather with team-habitat/travel context); "
            "mined family, every cell predeclared to record unresolved_below_power "
            "via a separate nfl-ats weak-signals record call regardless of interval "
            "shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
