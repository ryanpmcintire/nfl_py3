"""ENV-01 payoff screen: forecast-vs-actual weather battery.

Re-measures the four strongest `weather_battery_*`/`weather_followup_*`
mechanisms (each recorded in `registry/weak_signals.json` as an
actual-weather UPPER BOUND on any forecast-time feature) with the
Tuesday-noon GFS-MOS forecast substituted for the game-time actual, on the
population that forecast covers.

**The only changed variable, vs. each cell's actual-weather sibling, is
forecast-vs-actual for the FOCAL game's own temp/wind term.** `roof`,
`surface`, `away_modal_roof`, `away_modal_surface`, and the away team's own
climatological-normal baseline (`climate_temp`, an ACTUAL-weather same-season
aggregate over the away team's OTHER home games) are all reused unchanged
from `scripts/nfl_weather_battery_screen.py` / `nfl_weather_followup_screen.py`.
See `docs/forecast_weather_screen.md` for the full predeclaration (exact
subset definitions, leakage posture, method), frozen before this script
scored anything.

**Population: REG 2020-2025 only** -- the forecast archive's coverage, NOT
the 2009-2025 window the actual-weather originals were recorded on. This is
a disclosed narrower-window re-screen, not a like-for-like replication.

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``); recording happens via the separate
``scripts/record_forecast_weather_screen.py``, which reads this script's
output JSON and passes every numeric field through unmodified. This script
DOES write an automatic experiment-provenance stamp to
``registry/experiments/`` via ``write_experiment_artifact`` -- a run log,
not a verdict.

Per AGENTS.md, an interval containing zero is never a rejection -- every
cell here is predeclared to record ``unresolved_below_power`` regardless of
interval shape unless a RESOLVED wrong sign (whole interval on the wrong
side of the predicted direction) appears, which this script does not itself
adjudicate (the recording script does, against the admissible-grounds rule).

Writes JSON to
``artifacts/forecast_weather_screen/<UTC timestamp>/results.json`` and
prints a summary table to stdout.
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

sys.path.append(str(REPO / "scripts"))

from _common import summarize  # noqa: E402

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

FORECAST_COLUMNS = ["game_id", "forecast_temp_f", "forecast_wind_mph", "fetch_status"]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
SEASON_START = 2020  # forecast archive coverage, narrower than the 2009-2025 originals
SEASON_END = 2025

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
DOME_CLOSED_ROOFS = frozenset({"dome", "closed"})
# Reused verbatim from scripts/nfl_weather_battery_screen.py.
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

TEMP_GAP_THRESHOLD_F = 25.0  # weather_followup_temp_gap_cold_visitor's threshold, reused verbatim

LEAKAGE_CAVEAT_FORECAST = (
    "forecast-based (Tuesday-noon GFS-MOS), genuinely pregame-available at the pool's lock "
    "time, subject only to the inherited roof/away-climate-baseline caveats disclosed in "
    "docs/forecast_weather_screen.md -- NOT the actual-weather upper-bound caveat its sibling "
    "cell carries"
)


def _latest(glob_pattern: str) -> Path:
    candidates = sorted((REPO / "data/raw").glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/{glob_pattern} snapshot found")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest("*/schedules.parquet")


DEFAULT_FORECASTS = REPO / "data/raw/forecast_archive/full_2020_2025/forecasts.parquet"


def load_population(schedules_path: Path, forecasts_path: Path) -> pd.DataFrame:
    """Load, filter to REG 2020-2025, compute home_cover (pushes dropped),
    join the Tuesday-noon forecast archive, and attach every derived column
    the 4 cells (and their actual-weather diagnostic siblings) need. One row
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

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["outdoor"] = df["roof"].isin(OUTDOOR_ROOFS)
    df["week_block"] = df["season"] * 100 + df["week"]

    forecasts = pd.read_parquet(forecasts_path, columns=FORECAST_COLUMNS)
    n_before_forecast_join = len(df)
    df = df.merge(forecasts, on="game_id", how="left", validate="1:1")
    n_forecast_matched = int(df["fetch_status"].notna().sum())
    n_forecast_ok = int((df["fetch_status"] == "ok").sum())

    # away_modal_roof: same convention as scripts/nfl_weather_battery_screen.py
    # (full REG population after push-drop, per (home_team, season)).
    modal_roof = (
        df.groupby(["home_team", "season"])["roof"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("away_modal_roof")
    )
    df = df.merge(
        modal_roof,
        left_on=["away_team", "season"],
        right_index=True,
        how="left",
    )

    # climate_temp: away team's own climatological-normal ACTUAL outdoor home
    # temp this season, same convention as scripts/nfl_weather_followup_screen.py
    # (not season-causal, disclosed, precedented, unchanged here).
    outdoor_home = df.loc[df["outdoor"]]
    team_climate = (
        outdoor_home.groupby(["home_team", "season"])
        .agg(climate_temp=("temp", "mean"))
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

    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = pushes_or_missing
    df.attrs["n_before_forecast_join"] = n_before_forecast_join
    df.attrs["n_forecast_matched"] = n_forecast_matched
    df.attrs["n_forecast_ok"] = n_forecast_ok
    return df


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return {name: {"flag", "missing_mask", "description", "actual_flag",
    "actual_missing_mask", "actual_description"}}. Each forecast cell carries
    its actual-weather diagnostic sibling alongside it (same subset, same
    game, actual temp/wind instead of forecast) for the flag-agreement and
    matched-population comparison. ``missing_mask`` marks rows where a
    required input for THIS flag was missing/undefined (flag forced False on
    those rows, counted and reported, not dropped from the population).
    """

    cells: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        flag: pd.Series,
        missing: pd.Series,
        description: str,
        *,
        actual_flag: pd.Series,
        actual_missing: pd.Series,
        actual_description: str,
    ) -> None:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "description": f"{description} ({LEAKAGE_CAVEAT_FORECAST}).",
            "actual_flag": actual_flag.fillna(False).astype(bool),
            "actual_missing_mask": actual_missing.fillna(False).astype(bool),
            "actual_description": actual_description,
        }

    outdoor = df["outdoor"]
    roof_missing = df["roof"].isna()

    # 1. high_wind_outdoor -- mirrors weather_battery_high_wind_outdoor
    add(
        "forecast_weather_high_wind_outdoor",
        outdoor & (df["forecast_wind_mph"] >= 15),
        df["forecast_wind_mph"].isna() | roof_missing,
        "Outdoor/open roof AND Tuesday-noon forecast wind >= 15mph -- mirrors "
        "weather_battery_high_wind_outdoor with forecast_wind_mph substituted for actual "
        "wind, unsigned (no predicted direction, same as the sibling)",
        actual_flag=outdoor & (df["wind"] >= 15),
        actual_missing=df["wind"].isna() | roof_missing,
        actual_description="weather_battery_high_wind_outdoor's own construction on this "
        "screen's 2020-2025 population (diagnostic, not recorded)",
    )

    # 2. dome_team_outdoors_cold -- mirrors weather_battery_dome_team_outdoors_cold
    dome_missing = df["away_modal_roof"].isna() | df["forecast_temp_f"].isna() | roof_missing
    add(
        "forecast_weather_dome_team_outdoors_cold",
        df["away_modal_roof"].isin(DOME_CLOSED_ROOFS) & outdoor & (df["forecast_temp_f"] <= 40),
        dome_missing,
        "Away team's modal home roof this season is dome/closed AND this game is "
        "outdoor/open AND Tuesday-noon forecast temp<=40F -- mirrors "
        "weather_battery_dome_team_outdoors_cold with forecast_temp_f substituted for "
        "actual temp, predicted home_cover edge",
        actual_flag=df["away_modal_roof"].isin(DOME_CLOSED_ROOFS) & outdoor & (df["temp"] <= 40),
        actual_missing=df["away_modal_roof"].isna() | df["temp"].isna() | roof_missing,
        actual_description="weather_battery_dome_team_outdoors_cold's own construction on "
        "this screen's 2020-2025 population (diagnostic, not recorded)",
    )

    # 3. warm_team_cold_late -- mirrors weather_battery_warm_team_cold_late
    warm_missing = df["forecast_temp_f"].isna() | roof_missing
    add(
        "forecast_weather_warm_team_cold_late",
        df["away_team"].isin(WARM_METRO_TEAM_CODES)
        & outdoor
        & (df["forecast_temp_f"] <= 35)
        & (df["week"] >= 13),
        warm_missing,
        "Away team's own season code in the static warm-winter-metro list AND outdoor AND "
        "Tuesday-noon forecast temp<=35F AND week>=13 -- mirrors "
        "weather_battery_warm_team_cold_late with forecast_temp_f substituted for actual "
        "temp, predicted home_cover edge",
        actual_flag=df["away_team"].isin(WARM_METRO_TEAM_CODES)
        & outdoor
        & (df["temp"] <= 35)
        & (df["week"] >= 13),
        actual_missing=df["temp"].isna() | roof_missing,
        actual_description="weather_battery_warm_team_cold_late's own construction on this "
        "screen's 2020-2025 population (diagnostic, not recorded)",
    )

    # 4. temp_gap_cold_visitor -- mirrors weather_followup_temp_gap_cold_visitor
    forecast_temp_gap = df["climate_temp"] - df["forecast_temp_f"]
    forecast_gap_missing = df["climate_temp"].isna() | df["forecast_temp_f"].isna() | roof_missing
    actual_temp_gap = df["climate_temp"] - df["temp"]
    actual_gap_missing = df["climate_temp"].isna() | df["temp"].isna() | roof_missing
    add(
        "forecast_weather_temp_gap_cold_visitor",
        outdoor & (forecast_temp_gap >= TEMP_GAP_THRESHOLD_F),
        forecast_gap_missing,
        "Away team's own climatological-normal outdoor home temp (ACTUAL, same-season "
        "aggregate) minus this game's Tuesday-noon forecast temp >= "
        f"{TEMP_GAP_THRESHOLD_F:g}F AND outdoor -- mirrors "
        "weather_followup_temp_gap_cold_visitor with forecast_temp_f substituted for actual "
        "temp on the focal-game side only, predicted home_cover edge",
        actual_flag=outdoor & (actual_temp_gap >= TEMP_GAP_THRESHOLD_F),
        actual_missing=actual_gap_missing,
        actual_description="weather_followup_temp_gap_cold_visitor's own construction on "
        "this screen's 2020-2025 population (diagnostic, not recorded)",
    )

    expected = 4
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


def flag_agreement(
    forecast_flag: pd.Series,
    forecast_missing: pd.Series,
    actual_flag: pd.Series,
    actual_missing: pd.Series,
) -> dict[str, Any]:
    both_defined = ~forecast_missing & ~actual_missing
    n_both_defined = int(both_defined.sum())
    if n_both_defined == 0:
        return {"n_both_defined": 0, "agreement_rate": None}
    f = forecast_flag[both_defined]
    a = actual_flag[both_defined]
    agree = f == a
    return {
        "n_both_defined": n_both_defined,
        "agreement_rate": float(agree.mean()),
        "n_both_true": int((f & a).sum()),
        "n_both_false": int((~f & ~a).sum()),
        "n_forecast_only": int((f & ~a).sum()),
        "n_actual_only": int((~f & a).sum()),
    }


def score_cell(
    df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    flag = spec["flag"]
    missing_mask = spec["missing_mask"]
    actual_flag = spec["actual_flag"]
    actual_missing_mask = spec["actual_missing_mask"]

    week_blocked = summarize(df, flag=flag, block_col="week_block", samples=samples, seed=seed)
    season_blocked = summarize(df, flag=flag, block_col="season", samples=samples, seed=seed)
    actual_week_blocked = summarize(
        df, flag=actual_flag, block_col="week_block", samples=samples, seed=seed
    )

    agreement = flag_agreement(flag, missing_mask, actual_flag, actual_missing_mask)

    playable_fraction_matched = None
    if (
        not week_blocked.get("insufficient_data")
        and not actual_week_blocked.get("insufficient_data")
        and actual_week_blocked["full_slate_effect_pts"] != 0
    ):
        playable_fraction_matched = (
            week_blocked["full_slate_effect_pts"] / actual_week_blocked["full_slate_effect_pts"]
        )

    return {
        "name": name,
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing_mask.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
        "actual_sibling_diagnostic": {
            "description": spec["actual_description"],
            "n_flag": int(actual_flag.sum()),
            "n_missing_required_data": int(actual_missing_mask.sum()),
            "week_blocked_same_population": actual_week_blocked,
        },
        "flag_agreement": agreement,
        "playable_fraction_matched_population": playable_fraction_matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--forecasts", type=Path, default=DEFAULT_FORECASTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "forecast_weather_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.forecasts} ===")
    df = load_population(args.schedules, args.forecasts)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}, "
        f"scored population: {len(df)}, forecast matched: {df.attrs['n_forecast_matched']}, "
        f"forecast fetch_status=ok: {df.attrs['n_forecast_ok']}"
    )

    cells = build_cells(df)

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        ad = cell["actual_sibling_diagnostic"]
        aw = ad["week_blocked_same_population"]
        fa = cell["flag_agreement"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"n_missing_required_data={cell['n_missing_required_data']}"
        )
        print(
            f"  [forecast] full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [forecast, season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )
        if not aw.get("insufficient_data"):
            print(
                f"  [actual, same 2020-2025 population] full_slate_effect="
                f"{aw['full_slate_effect_pts']:+.4f}pts 95% [{aw['ci95_scaled'][0]:+.4f}, "
                f"{aw['ci95_scaled'][1]:+.4f}] P+={aw['probability_positive']:.4f}"
            )
        if cell["playable_fraction_matched_population"] is not None:
            print(
                "  playable_fraction (forecast/actual, matched population) = "
                f"{cell['playable_fraction_matched_population']:.4f}"
            )
        if fa.get("n_both_defined"):
            print(
                f"  flag_agreement={fa['agreement_rate']:.4f} "
                f"(n_both_defined={fa['n_both_defined']}, both_true={fa['n_both_true']}, "
                f"both_false={fa['n_both_false']}, forecast_only={fa['n_forecast_only']}, "
                f"actual_only={fa['n_actual_only']})"
            )

    ranked = sorted(
        (r for r in results if not r["week_blocked"].get("insufficient_data")),
        key=lambda r: abs(r["week_blocked"]["full_slate_effect_pts"]),
        reverse=True,
    )
    print("\n=== ranked by |forecast full-slate effect|, week-blocked primary ===")
    for rank, cell in enumerate(ranked, start=1):
        wb = cell["week_blocked"]
        print(
            f"{rank}. {cell['name']:<48} {wb['full_slate_effect_pts']:+.4f}pts "
            f"P+={wb['probability_positive']:.4f} n_flag={cell['n_flag']}"
        )

    configuration = {
        "command": "nfl-forecast-weather-screen",
        "schedules": str(args.schedules),
        "forecasts": str(args.forecasts),
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
        "n_forecast_matched": df.attrs["n_forecast_matched"],
        "n_forecast_ok": df.attrs["n_forecast_ok"],
        "predeclaration": "docs/forecast_weather_screen.md (frozen before scoring)",
        "leakage_caveat_forecast": LEAKAGE_CAVEAT_FORECAST,
        "results": results,
        "ranked_by_abs_forecast_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="nfl-forecast-weather-screen",
        metrics=payload,
        notes=(
            "ENV-01 payoff screen: 4 predeclared cells re-measuring the strongest "
            "weather_battery_*/weather_followup_* mechanisms with the Tuesday-noon GFS-MOS "
            "forecast substituted for the game-time actual on REG 2020-2025 (forecast "
            "archive coverage, narrower than the 2009-2025 originals). Timing re-screen, "
            "not independent discovery; every cell predeclared to record "
            "unresolved_below_power via a separate nfl-ats weak-signals record call "
            "regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
