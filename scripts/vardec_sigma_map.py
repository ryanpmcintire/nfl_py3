"""Heteroskedasticity sigma map: does residual ATS sigma vary measurably by
observable pregame conditions, or is the ~13.1-point residual sd effectively a
constant?

Population: REG 2009-2025, newest ``data/raw/*/schedules.parquet`` snapshot,
``add_ats_outcomes`` for ``ats_margin`` (= ``result`` - ``spread_line``), pushes
and missing lines dropped. Statistic per cell: ratio of subset sd to complement
sd of ``ats_margin``, joint week-blocked bootstrap (season-blocked secondary)
with 20,000 resamples, percentile 95% interval, and
``probability_ratio_below_one`` reported instead of zero-crossing language
(per AGENTS.md).

Every condition is pregame-known: venue/stadium facts, kickoff slot, schedule
week, spread magnitude, rest differentials, division status, prior-week-only
standings for the late-season contention cell, and prior-game-only starter
history for the non-incumbent-QB cell. No game-time actuals are used anywhere.

MINED FAMILY DISCLOSURE: the 18 cells below were chosen from the task brief
after the data landscape was known, so roughly one spurious 95% interval
excluding 1.0 is expected by chance alone. Cells are reported regardless of
interval shape; an interval containing 1.0 is never grounds for closing a line.

If any low-sigma cell's week-blocked interval excludes 1.0, the script also
computes an attribution-only Best Pick implication: historical forced-pick
accuracy of always taking the spread favourite inside that bucket versus the
whole slate, plus the analytic Phi(edge/sigma) sensitivity at +1 point mean
edge. This is attribution only, not a validated selection strategy.

Writes JSON to ``artifacts/vardec_sigma/<UTC timestamp>/results.json`` and a
summary table to stdout.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import default_schedules  # noqa: E402

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260822

COLD_TEMP_F = 40.0
HOT_TEMP_F = 82.0
HIGH_WIND_MPH = 13.0
CONTENTION_WEEK_MIN = 14
PRIMETIME_ET = "20:00"
REST_DIFF_ABS_MIN = 4.0
ALTITUDE_STADIUMS = {
    "Invesco Field at Mile High",
    "Sports Authority Field at Mile High",
    "Empower Field at Mile High",
    "Azteca Stadium",
}
GRASS_SURFACES = {"grass", " Grass", "dessograss"}
PLAYOFF_SEEDS_BY_ERA = {
    **dict.fromkeys(range(2009, 2020), 6),
    **dict.fromkeys(range(2020, 2026), 7),
}

DIVISIONS: dict[str, str] = {}
for div, teams in {
    "AFC_East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC_North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC_South": ["HOU", "IND", "JAX", "TEN"],
    "AFC_West": ["DEN", "KC", "OAK", "SD", "LV", "LAC"],
    "NFC_East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC_North": ["CHI", "DET", "GB", "MIN"],
    "NFC_South": ["ATL", "CAR", "NO", "TB"],
    "NFC_West": ["ARI", "STL", "SF", "SEA", "LAR"],
}.items():
    for team in teams:
        DIVISIONS[team] = div


def load_population(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    keep = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "gameday",
            "weekday",
            "gametime",
            "game_type",
            "home_team",
            "away_team",
            "result",
            "spread_line",
            "div_game",
            "roof",
            "surface",
            "temp",
            "wind",
            "away_rest",
            "home_rest",
            "away_qb_name",
            "home_qb_name",
            "stadium",
            "location",
        ]
        if c in raw.columns
    ]
    df = raw.loc[:, keep].copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    n_all = len(df)
    df = add_ats_outcomes(df)
    df = df.loc[
        df["ats_margin"].notna() & df["spread_line"].notna() & df["result"].notna()
    ].reset_index(drop=True)

    df["week_block"] = df["season"] * 100 + df["week"]
    df["is_division"] = (
        pd.to_numeric(df.get("div_game"), errors="coerce").fillna(0).astype(float) > 0
    )
    df["abs_spread"] = df["spread_line"].abs()
    df["rest_diff"] = pd.to_numeric(df.get("home_rest"), errors="coerce") - pd.to_numeric(
        df.get("away_rest"), errors="coerce"
    )
    df["is_dome_or_closed"] = df["roof"].isin(["dome", "closed"])
    df["is_retractable_open"] = df["roof"] == "open"
    df["is_open_air_known"] = df["roof"].isin(["outdoors", "open"])
    outdoor = df["is_open_air_known"]
    temp = pd.to_numeric(df.get("temp"), errors="coerce")
    wind = pd.to_numeric(df.get("wind"), errors="coerce")
    df["outdoor_temp_known"] = outdoor & temp.notna()
    df["is_cold_outdoor"] = (outdoor & temp.notna() & (temp < COLD_TEMP_F)).fillna(False)
    df["is_hot_outdoor"] = (outdoor & temp.notna() & (temp >= HOT_TEMP_F)).fillna(False)
    df["is_high_wind"] = (outdoor & wind.notna() & (wind >= HIGH_WIND_MPH)).fillna(False)
    df["is_primetime"] = (df["gametime"].astype("string").fillna("") >= PRIMETIME_ET).fillna(False)
    surface_norm = df["surface"].astype("string").str.strip().str.lower().fillna("")
    df["is_grass_surface"] = surface_norm.isin({"grass", "dessograss"})
    df["is_altitude_high"] = df["stadium"].isin(ALTITUDE_STADIUMS)
    df["is_big_rest_diff"] = df["rest_diff"].abs().ge(REST_DIFF_ABS_MIN)

    df = add_contention_flag(df)
    df = add_non_incumbent_qb_flag(df)
    df.attrs["n_before_drop"] = n_all
    return df


def add_contention_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Late-season games where either team sits within one game of its
    conference's final playoff seed, computed from PRIOR-GAME records only."""

    ordered = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    wins: dict[tuple[int, str], int] = {}
    losses: dict[tuple[int, str], int] = {}
    ties: dict[tuple[int, str], int] = {}

    def score(season: int, team: str) -> float:
        w = wins.get((season, team), 0)
        t = ties.get((season, team), 0)
        return w + 0.5 * t

    flags: list[bool] = []
    results = ordered["result"].to_numpy()
    homes = ordered["home_team"].to_numpy()
    aways = ordered["away_team"].to_numpy()
    weeks = ordered["week"].to_numpy()
    seasons = ordered["season"].to_numpy()

    def conference_of(team: str) -> str:
        return "AFC" if DIVISIONS.get(team, "").startswith("AFC") else "NFC"

    for i in range(len(ordered)):
        season = int(seasons[i])
        home = str(homes[i])
        away = str(aways[i])
        flagged = False
        if int(weeks[i]) >= CONTENTION_WEEK_MIN:
            seeds = PLAYOFF_SEEDS_BY_ERA.get(season, 7)
            for conf in ("AFC", "NFC"):
                conf_teams = [t for t, d in DIVISIONS.items() if d.startswith(conf)]
                ranked = sorted(
                    range(len(conf_teams)),
                    key=lambda j, _t=conf_teams: -(score(season, _t[j]) + 1e-9 * j),
                )
                if len(ranked) < seeds:
                    continue
                line_score = score(season, conf_teams[ranked[seeds - 1]])
                for team in (home, away):
                    if conference_of(team) == conf and abs(line_score - score(season, team)) <= 1.0:
                        flagged = True
        flags.append(flagged)
        margin = results[i]
        if pd.isna(margin):
            continue
        if margin == 0:
            ties[(season, home)] = ties.get((season, home), 0) + 1
            ties[(season, away)] = ties.get((season, away), 0) + 1
        elif margin > 0:
            wins[(season, home)] = wins.get((season, home), 0) + 1
            losses[(season, away)] = losses.get((season, away), 0) + 1
        else:
            wins[(season, away)] = wins.get((season, away), 0) + 1
            losses[(season, home)] = losses.get((season, home), 0) + 1

    ordered["is_late_contention"] = np.array(flags, dtype=bool)
    return ordered.set_index(df.index)


def add_non_incumbent_qb_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Either team starts someone other than its modal prior starter that
    season (weeks >= 2); prior-game information only."""

    ordered = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    history: dict[tuple[int, str], Counter[str]] = {}
    backup_rows: list[bool] = []
    for row in ordered.itertuples(index=False):
        season = int(row.season)
        flagged = False
        for team, qb in (
            (str(row.home_team), row.home_qb_name),
            (str(row.away_team), row.away_qb_name),
        ):
            key = (season, team)
            counts = history.setdefault(key, Counter())
            if int(row.week) >= 2 and isinstance(qb, str) and qb.strip():
                prior_mode, prior_n = counts.most_common(1)[0] if counts else ("", 0)
                if prior_n > 0 and qb != prior_mode:
                    flagged = True
                counts[qb] += 1
        backup_rows.append(flagged)

    ordered["is_non_incumbent_qb"] = np.array(backup_rows, dtype=bool)
    return ordered.set_index(df.index)


def block_bootstrap_sd_ratio(
    values: np.ndarray,
    flag: np.ndarray,
    block_index: np.ndarray,
    block_count: int,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    sums: dict[bool, np.ndarray] = {}
    sumsqs: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    sq = values * values
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(block_index[mask], weights=values[mask], minlength=block_count)
        sumsqs[group] = np.bincount(block_index[mask], weights=sq[mask], minlength=block_count)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    variances: dict[bool, np.ndarray] = {}
    for group in (True, False):
        n = drawn @ counts[group]
        s = drawn @ sums[group]
        ss = drawn @ sumsqs[group]
        with np.errstate(invalid="ignore", divide="ignore"):
            variances[group] = (ss - s * s / n) / np.maximum(n - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.sqrt(variances[True] / variances[False])
    return ratio


def summarize_cell(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    population: pd.Series,
    block_col: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    sub = df.loc[population.to_numpy()].reset_index(drop=True)
    fl = flag.loc[population.to_numpy()].reset_index(drop=True).to_numpy(dtype=bool)
    values = sub["ats_margin"].to_numpy(dtype=np.float64)
    blocks, block_index = np.unique(sub[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)

    subset_vals = values[fl]
    comp_vals = values[~fl]
    n_flag = int(fl.sum())
    n_comp = int((~fl).sum())
    if n_flag < 30 or n_comp < 30:
        return {
            "n_flag": n_flag,
            "n_complement": n_comp,
            "insufficient_data": True,
        }

    sigma_flag = float(np.std(subset_vals, ddof=1))
    sigma_comp = float(np.std(comp_vals, ddof=1))

    draws = block_bootstrap_sd_ratio(
        values,
        fl,
        block_index,
        len(blocks),
        samples=samples,
        seed=seed,
    )
    valid = draws[np.isfinite(draws)]
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return {
        "n_population": len(sub),
        "n_blocks": len(blocks),
        "n_flag": n_flag,
        "n_complement": n_comp,
        "sigma_subset": sigma_flag,
        "sigma_complement": sigma_comp,
        "sd_ratio_point": sigma_flag / sigma_comp,
        "ci95_ratio": [float(lower), float(upper)],
        "probability_ratio_below_one": float(np.mean(valid < 1.0)),
        "bootstrap_samples": samples,
        "dropped_draws": int(samples - len(valid)),
        "insufficient_data": False,
    }


def build_cells(df: pd.DataFrame) -> list[dict[str, Any]]:
    everyone = pd.Series(True, index=df.index)
    cells: list[tuple[str, str, pd.Series, pd.Series]] = []

    def add(name: str, family: str, flag: pd.Series, pop: pd.Series | None = None) -> None:
        cells.append((name, family, flag.fillna(False), pop if pop is not None else everyone))

    add("dome_or_closed_roof", "roof_type", df["is_dome_or_closed"])
    add("retractable_open_air", "roof_type", df["is_retractable_open"])
    weather_pop = df["outdoor_temp_known"]
    add("cold_outdoor_lt40F", "weather_bands", df["is_cold_outdoor"], weather_pop)
    add("hot_outdoor_ge82F", "weather_bands", df["is_hot_outdoor"], weather_pop)
    add("high_wind_ge13mph", "weather_bands", df["is_high_wind"], weather_pop)
    add("division_game", "matchup", df["is_division"])
    add("late_contention_week14plus", "motivation", df["is_late_contention"])
    add("primetime_kick_ge2000et", "slot", df["is_primetime"])
    add("non_incumbent_qb_start", "qb", df["is_non_incumbent_qb"])
    add("spread_abs_le_3", "spread_buckets", df["abs_spread"] <= 3.0)
    add(
        "spread_abs_3to7",
        "spread_buckets",
        (df["abs_spread"] > 3.0) & (df["abs_spread"] <= 7.0),
    )
    add(
        "spread_abs_7to10",
        "spread_buckets",
        (df["abs_spread"] > 7.0) & (df["abs_spread"] <= 10.0),
    )
    add("spread_abs_gt_10", "spread_buckets", df["abs_spread"] > 10.0)
    add("season_week_1_3", "season_week", df["week"] <= 3)
    add("season_week_15_plus", "season_week", df["week"] >= 15)
    add("turf_surface", "surface", ~df["is_grass_surface"])
    add("altitude_high_denver_azteca", "altitude", df["is_altitude_high"])
    add(f"rest_diff_abs_ge_{REST_DIFF_ABS_MIN:g}", "rest", df["is_big_rest_diff"])

    specs: list[dict[str, Any]] = []
    for name, family, flag, pop in cells:
        specs.append({"name": name, "family": family, "flag": flag, "population": pop})
    assert len(specs) == 18, f"expected 18 mined cells, got {len(specs)}"
    return specs


def best_pick_implication(df: pd.DataFrame, bucket_mask: pd.Series, name: str) -> dict[str, Any]:
    """Attribution-only Best Pick numbers for a low-sigma bucket."""

    bucket = df.loc[bucket_mask.to_numpy()]
    slate_acc = favorite_cover_accuracy(df)
    bucket_acc = favorite_cover_accuracy(bucket)
    sigma_bucket = float(bucket["ats_margin"].std(ddof=1))
    edge_points = 1.0
    phi_edge = 0.5 * (1.0 + math.erf(edge_points / (sigma_bucket * math.sqrt(2))))
    return {
        "bucket": name,
        "n_games": len(bucket),
        "favorite_cover_accuracy_bucket": bucket_acc,
        "favorite_cover_accuracy_slate": slate_acc,
        "accuracy_gap_points": 100.0 * (bucket_acc - slate_acc),
        "analytic_cover_prob_plus1pt_edge": phi_edge,
        "attribution_only_note": (
            "Not a validated selection strategy; single mined look at the "
            "favourite rule inside the bucket."
        ),
    }


def favorite_cover_accuracy(games: pd.DataFrame) -> float:
    spread = games["spread_line"].to_numpy(dtype=np.float64)
    margin = games["ats_margin"].to_numpy(dtype=np.float64)
    fav_home = spread >= 0
    fav_covered = np.where(fav_home, margin > 0, margin < 0)
    return float(np.mean(fav_covered))


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
    output_dir: Path = args.output or (REPO / "artifacts" / "vardec_sigma" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    baseline_sigma = float(df["ats_margin"].std(ddof=1))
    print(
        f"REG {SEASON_START}-{SEASON_END}: {df.attrs['n_before_drop']} games, "
        f"{len(df)} scored, overall residual sd = {baseline_sigma:.3f}"
    )

    cells = build_cells(df)
    results: list[dict[str, Any]] = []
    for spec in cells:
        wb = summarize_cell(
            df,
            flag=spec["flag"],
            population=spec["population"],
            block_col="week_block",
            samples=args.samples,
            seed=args.seed,
        )
        sb = summarize_cell(
            df,
            flag=spec["flag"],
            population=spec["population"],
            block_col="season",
            samples=args.samples,
            seed=args.seed + 1,
        )
        entry = {
            "name": spec["name"],
            "family": spec["family"],
            "week_blocked_primary": wb,
            "season_blocked_secondary": sb,
        }
        results.append(entry)
        if wb.get("insufficient_data"):
            print(f"{spec['name']:32s} insufficient data n={wb['n_flag']}/{wb['n_complement']}")
            continue
        print(
            f"{spec['name']:32s} n={wb['n_flag']:4d} "
            f"sigma={wb['sigma_subset']:.2f} vs {wb['sigma_complement']:.2f} "
            f"ratio={wb['sd_ratio_point']:.3f} "
            f"[{wb['ci95_ratio'][0]:.3f}, {wb['ci95_ratio'][1]:.3f}] "
            f"P(<1)={wb['probability_ratio_below_one']:.3f}"
        )

    low_excluding = [
        r
        for r in results
        if not r["week_blocked_primary"].get("insufficient_data")
        and r["week_blocked_primary"]["ci95_ratio"][1] < 1.0
    ]
    high_excluding = [
        r
        for r in results
        if not r["week_blocked_primary"].get("insufficient_data")
        and r["week_blocked_primary"]["ci95_ratio"][0] > 1.0
    ]
    strongest_low = (
        min(low_excluding, key=lambda r: r["week_blocked_primary"]["ci95_ratio"][0])
        if low_excluding
        else None
    )
    strongest_high = (
        max(high_excluding, key=lambda r: r["week_blocked_primary"]["ci95_ratio"][1])
        if high_excluding
        else None
    )

    best_pick = None
    if strongest_low is not None:
        spec = next(s for s in cells if s["name"] == strongest_low["name"])
        best_pick = best_pick_implication(
            df, spec["population"] & spec["flag"], strongest_low["name"]
        )
        print(
            f"\nlow-sigma bucket {strongest_low['name']}: favourite-cover accuracy "
            f"{best_pick['favorite_cover_accuracy_bucket'] * 100:.2f}% vs slate "
            f"{best_pick['favorite_cover_accuracy_slate'] * 100:.2f}% (attribution only)"
        )

    payload = {
        "schema": 1,
        "generated_at_utc": timestamp,
        "schedules_path": str(args.schedules),
        "population": {
            "game_type": "REG",
            "season_start": SEASON_START,
            "season_end": SEASON_END,
            "n_games_scored": len(df),
            "baseline_sigma_points": baseline_sigma,
        },
        "method": {
            "statistic": "subset sd / complement sd of ats_margin",
            "block_bootstrap": "joint week-block primary, season-block secondary",
            "samples_per_interval": args.samples,
            "seed": args.seed,
            "multiplicity_disclosure": (
                "mined family of 18 cells; roughly one spurious 95% exclusion "
                "of 1.0 expected by chance; no correction applied"
            ),
        },
        "cells": results,
        "strongest_low_sigma_bucket": strongest_low,
        "strongest_high_sigma_bucket": strongest_high,
        "best_pick_implication": best_pick,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    configuration = {
        "schedules_path": str(args.schedules),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "samples": args.samples,
        "seed": args.seed,
    }
    payload["provenance"] = artifact_provenance(configuration, args.schedules, project_root=REPO)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="vardec-sigma-map",
        metrics={
            "baseline_sigma_points": baseline_sigma,
            "n_cells": len(results),
            "low_sigma_bucket": (strongest_low or {}).get("name"),
            "high_sigma_bucket": (strongest_high or {}).get("name"),
        },
        notes=(
            "Measure-only mined heteroskedasticity screen; 18 cells, no "
            "multiplicity correction; nothing recorded automatically."
        ),
        source="scripts/vardec_sigma_map.py",
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
