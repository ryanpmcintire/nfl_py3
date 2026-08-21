"""Weather x market-expectation (betting total) interaction screen: 4
predeclared cells testing whether adverse weather hurts the FAVORITE more in
top-tercile-total (pass-heavy-expected) games than in low-total games where
weather is partially priced in. Outcome is favorite_cover (the favorited team
covers), NOT home_cover -- the predeclared direction ("HIGH-total favorite
covers LESS") is about the favorite, which home_cover cannot express.

Predeclaration frozen before scoring: docs/weather_total_interaction_screen.md.
Cells 1/2/4 predict a NEGATIVE favorite-cover gap; cell 3
(wxtot_wind15_bottom_total) is a control with no predicted direction.

Leakage posture inherited from docs/weather_followup.md: schedules'
temp/wind are GAME-TIME ACTUALS (upper bounds on any forecast-time feature);
the precip cell uses the genuinely pregame-available kickoff-nearest GFS-MOS
forecast archive instead. total_line is the closing total (known at close,
not at a Tuesday opener); tercile cuts are pooled over the full scored
population (market-level constant, outcome-independent, disclosed).

Measure-only. This script never writes registry/weak_signals.json; recording
happens via explicit `nfl-ats weak-signals record` calls against this
script's output JSON. It DOES write an experiment-provenance stamp to
registry/experiments/weather-total-interaction-screen/ via
write_experiment_artifact. Per AGENTS.md every cell is predeclared to record
unresolved_below_power regardless of interval shape.

Method reused verbatim from scripts/nfl_weather_battery_screen.py
(block_bootstrap_two_group, full-slate scaling, week-blocked primary +
season-blocked secondary). Writes JSON to
artifacts/weather_total_interaction_screen/<UTC timestamp>/results.json.
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

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "total_line",
    "roof",
    "temp",
    "wind",
]

FORECAST_COLUMNS = ["game_id", "forecast_precip_prob_pct", "fetch_status"]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
WIND_THRESHOLD_MPH = 15.0
COLD_THRESHOLD_F = 35.0
PRECIP_THRESHOLD_PCT = 60.0

ACTUAL_WEATHER_CAVEAT = (
    "game-time ACTUAL weather mechanism screen, NOT pregame-available; upper "
    "bound for a forecast-time feature"
)
FORECAST_CAVEAT = "kickoff-nearest GFS-MOS forecast trigger, genuinely pregame-available at close"


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()
DEFAULT_FORECASTS = REPO / "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"


def load_population(schedules_path: Path, forecasts_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    available = [c for c in SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    n_reg_games = len(df)
    df = add_ats_outcomes(df)
    n_pushes_or_missing_result = n_reg_games - int(df["home_cover"].notna().sum())

    df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")
    df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    n_missing_spread = int(df["spread_line"].isna().sum())
    n_pickem = int((df["spread_line"] == 0).sum())
    n_missing_total = int(df["total_line"].isna().sum())
    df = df.loc[
        df["spread_line"].notna()
        & (df["spread_line"] != 0)
        & df["total_line"].notna()
        & df["home_cover"].notna()
    ].reset_index(drop=True)

    home_favored = df["spread_line"] > 0
    df["favorite_cover"] = np.where(home_favored, df["ats_margin"], -df["ats_margin"]) > 0
    df["favorite_cover"] = df["favorite_cover"].astype(float)

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["outdoor"] = df["roof"].isin(OUTDOOR_ROOFS)
    df["week_block"] = df["season"] * 100 + df["week"]

    forecasts = pd.read_parquet(forecasts_path, columns=FORECAST_COLUMNS)
    n_before_forecast_join = len(df)
    df = df.merge(forecasts, on="game_id", how="left", validate="1:1")
    n_forecast_matched = int(df["fetch_status"].notna().sum())
    n_forecast_ok = int((df["fetch_status"] == "ok").sum())

    q1, q2 = np.quantile(df["total_line"].to_numpy(dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    df["total_top_tercile"] = df["total_line"] > q2
    df["total_bottom_tercile"] = df["total_line"] < q1

    df.attrs.update(
        n_reg_games=n_reg_games,
        n_pushes_or_missing_result=n_pushes_or_missing_result,
        n_missing_spread=n_missing_spread,
        n_pickem=n_pickem,
        n_missing_total=n_missing_total,
        n_before_forecast_join=n_before_forecast_join,
        n_forecast_matched=n_forecast_matched,
        n_forecast_ok=n_forecast_ok,
        tercile_low_cut=float(q1),
        tercile_high_cut=float(q2),
    )
    return df


def build_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        flag: pd.Series,
        missing: pd.Series,
        predicted: str,
        caveat: str,
    ) -> None:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "missing_mask": missing.fillna(False).astype(bool),
            "predicted_direction": predicted,
            "description": f"{flag_to_text(name)} ({caveat}).",
        }

    def flag_to_text(name: str) -> str:
        return {
            "wxtot_wind15_top_total": (
                "Outdoor/open roof AND game-time actual wind >= 15mph AND top-tercile "
                "betting total -- HIGH-total favorite covers LESS under wind"
            ),
            "wxtot_cold35_top_total": (
                "Outdoor/open roof AND game-time actual temp <= 35F AND top-tercile "
                "betting total -- HIGH-total favorite covers LESS under cold"
            ),
            "wxtot_wind15_bottom_total": (
                "Outdoor/open roof AND game-time actual wind >= 15mph AND bottom-tercile "
                "betting total -- CONTROL, weather partially priced into low-total "
                "games, near null expected"
            ),
            "wxtot_precip60_top_total": (
                "Outdoor/open roof AND kickoff-nearest forecast precip probability >= 60% "
                "AND top-tercile betting total -- HIGH-total favorite covers LESS under precip"
            ),
        }[name]

    roof_missing = df["roof"].isna()

    add(
        "wxtot_wind15_top_total",
        df["outdoor"] & (df["wind"] >= WIND_THRESHOLD_MPH) & df["total_top_tercile"],
        df["wind"].isna() | roof_missing,
        "negative",
        ACTUAL_WEATHER_CAVEAT,
    )
    add(
        "wxtot_cold35_top_total",
        df["outdoor"] & (df["temp"] <= COLD_THRESHOLD_F) & df["total_top_tercile"],
        df["temp"].isna() | roof_missing,
        "negative",
        ACTUAL_WEATHER_CAVEAT,
    )
    add(
        "wxtot_wind15_bottom_total",
        df["outdoor"] & (df["wind"] >= WIND_THRESHOLD_MPH) & df["total_bottom_tercile"],
        df["wind"].isna() | roof_missing,
        "none (control)",
        ACTUAL_WEATHER_CAVEAT,
    )
    add(
        "wxtot_precip60_top_total",
        df["outdoor"]
        & (df["forecast_precip_prob_pct"] >= PRECIP_THRESHOLD_PCT)
        & df["total_top_tercile"],
        df["forecast_precip_prob_pct"].isna() | roof_missing,
        "negative",
        FORECAST_CAVEAT,
    )

    expected = 4
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
    subset_cover = float(work.loc[work["_flag"], "favorite_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "favorite_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="favorite_cover",
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
        "predicted_direction": spec["predicted_direction"],
        "n_flag": int(flag.sum()),
        "n_missing_required_data": int(missing_mask.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--forecasts", type=Path, default=DEFAULT_FORECASTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (
        REPO / "artifacts" / "weather_total_interaction_screen" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} + {args.forecasts} ===")
    df = load_population(args.schedules, args.forecasts)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_reg_games']}, "
        f"pushes/missing result dropped: {df.attrs['n_pushes_or_missing_result']}, "
        f"missing spread_line: {df.attrs['n_missing_spread']}, "
        f"pick'em (spread_line==0): {df.attrs['n_pickem']}, "
        f"missing total_line: {df.attrs['n_missing_total']}, "
        f"scored population: {len(df)}, "
        f"forecast matched: {df.attrs['n_forecast_matched']} "
        f"(fetch ok: {df.attrs['n_forecast_ok']})"
    )
    print(
        f"pooled total_line tercile cuts: low={df.attrs['tercile_low_cut']:.1f} "
        f"high={df.attrs['tercile_high_cut']:.1f}"
    )

    cells = build_cells(df)

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} (predicted: {spec['predicted_direction']}) ===")
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
            f"  subset_fav_cover={wb['subset_cover']:.4f} "
            f"complement_fav_cover={wb['complement_cover']:.4f} "
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

    configuration = {
        "command": "weather-total-interaction-screen",
        "schedules": str(args.schedules),
        "forecasts": str(args.forecasts),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "tercile_low_cut": df.attrs["tercile_low_cut"],
        "tercile_high_cut": df.attrs["tercile_high_cut"],
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games": df.attrs["n_reg_games"],
        "n_pushes_or_missing_result_dropped": df.attrs["n_pushes_or_missing_result"],
        "n_missing_spread_dropped": df.attrs["n_missing_spread"],
        "n_pickem_dropped": df.attrs["n_pickem"],
        "n_missing_total_dropped": df.attrs["n_missing_total"],
        "n_scored_population": len(df),
        "n_forecast_matched": df.attrs["n_forecast_matched"],
        "n_forecast_ok": df.attrs["n_forecast_ok"],
        "tercile_low_cut": df.attrs["tercile_low_cut"],
        "tercile_high_cut": df.attrs["tercile_high_cut"],
        "outcome": "favorite_cover",
        "predeclaration": "docs/weather_total_interaction_screen.md (frozen before scoring)",
        "results": results,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="weather-total-interaction-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation battery (4 predeclared cells, weather x betting-"
            "total tercile interactions on the favorite-cover outcome); mined family, every "
            "cell predeclared to record unresolved_below_power via a separate nfl-ats "
            "weak-signals record call regardless of interval shape (AGENTS.md). Adjacent to "
            "forecast_weather_kn_precip_high_total_* and weather_battery_high_wind_* -- do "
            "not pool as independent."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
