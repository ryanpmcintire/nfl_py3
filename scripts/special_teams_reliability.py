"""Reliability audit for PBP-06 special-teams dimensions -- run BEFORE any
cell is predeclared, matching the PBP-08 team-style precedent's own ordering
(``scripts/team_style_reliability.py``, ``docs/team_style.md`` "Reliability
gate"): a dimension that doesn't persist year-over-year is not a trait, so
this is the gate that shapes which dimensions enter ``docs/special_teams_battery.md``,
not a footnote after the fact.

Two independent reliability estimates per dimension, both computed on the
LEAGUE-SEASON-CENTERED value (raw values would conflate era-wide rule
changes -- e.g. the 2023 fair-catch-anywhere-inside-25 rule and the 2024
"dynamic kickoff" rule, both of which visibly moved the league-wide kickoff
return rate in this data -- with team/specialist identity):

1. **Year-over-year** (primary): Pearson r between a team's centered value
   in season ``t`` and season ``t+1``, pooled across all same-franchise
   consecutive-season pairs 2009-2025 (``TEAM_ABBREVIATION_ALIASES`` already
   applied upstream in ``special_teams_features.py``), block-bootstrapped
   95% CI over team-season pairs. Method identical to
   ``scripts/team_style_reliability.py::bootstrap_pearson_ci``.
2. **Within-season split-half** (secondary cross-check): odd/even-week
   team-season split, Spearman-Brown corrected, via
   ``nfl_ats.cfb_qb_dependence.split_half_reliability`` REUSED DIRECTLY (not
   reimplemented) -- the same function PBP-05's 0.80/0.46 figure and the
   PBP-08 team-style figures were built on.

Binding taxonomy (own it, do not paraphrase): an interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment --
at this evaluator's ~2-point resolution "contains zero" is the EXPECTED
outcome for a real small signal. Only two closing grounds: (1) refuted
mechanism -- RESOLVED wrong sign (whole interval on the wrong side of zero)
or zero split-half reliability; (2) bounded by a positive control proven
able to detect an effect that size. Everything else is
``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". Reliability is the ONE admissible exclusion criterion for a
cell INPUT (not a whole experiment, per the PBP-08 precedent this task
explicitly reuses): a dimension whose year-over-year 95% CI upper bound sits
at or below zero is not treated as a persistent trait and is excluded from
the predeclared cells on that ground alone, documented here before any cell
exists.

Writes ``artifacts/special_teams_reliability/<UTC timestamp>/results.json``
and prints the reliability table to stdout.
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

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

RELIABILITY_SEED = 20260819
RELIABILITY_EXCLUSION_BAR = 0.0

DIMENSIONS = (
    "fg_oe",
    "punt_net_yards",
    "punt_return_yards",
    "kickoff_return_yards",
    "block_rate",
)
MIN_SEASON_SAMPLE = {
    "fg_oe": 15,  # ~30 FG attempts/team/season typical; 15 is a partial-season floor
    "punt_net_yards": 20,  # ~65-80 punts/team/season typical
    "punt_return_yards": 10,  # returns have fallen with rule changes; keep the floor low
    "kickoff_return_yards": 10,  # same, sharper drop post-2023/2024 rule changes
    "block_rate": 30,  # rare event; needs the combined FG+punt denominator
}
COUNT_COLUMN = {
    "fg_oe": "n_fg_attempts",
    "punt_net_yards": "n_punts",
    "punt_return_yards": "n_punt_returns",
    "kickoff_return_yards": "n_kickoff_returns",
    "block_rate": "n_kicks_total",
}


def _latest_features() -> Path:
    candidates = sorted((REPO / "data" / "raw" / "special_teams").glob("*/team_season.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "no data/raw/special_teams/*/team_season.parquet -- run "
            "scripts/special_teams_features.py first"
        )
    return candidates[-1]


def year_over_year_pairs(team_season: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = team_season[["team", "season", metric]].rename(columns={metric: "value_t"})
    right = team_season[["team", "season", metric]].rename(columns={metric: "value_t1"})
    right["season"] = right["season"] - 1
    pairs = left.merge(right, on=["team", "season"], how="inner")
    return pairs.dropna(subset=["value_t", "value_t1"])


def bootstrap_pearson_ci(
    x: np.ndarray, y: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, float]:
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            draws[i] = np.nan
            continue
        draws[i] = np.corrcoef(xi, yi)[0, 1]
    valid = draws[~np.isnan(draws)]
    lower, upper = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
    return float(lower), float(upper), float(np.mean(valid > 0.0)) if len(valid) else float("nan")


def measure_year_over_year(
    team_season: pd.DataFrame, metric: str, *, seed_offset: int
) -> dict[str, Any]:
    centered = f"{metric}_centered"
    pairs = year_over_year_pairs(team_season, centered)
    n = len(pairs)
    if n < 5:
        return {
            "metric": metric,
            "n_pairs": n,
            "pearson_r": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "probability_positive": float("nan"),
            "spearman_brown_stepped_up": float("nan"),
        }
    x = pairs["value_t"].to_numpy(dtype=float)
    y = pairs["value_t1"].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    lower, upper, prob_pos = bootstrap_pearson_ci(
        x, y, samples=20_000, seed=RELIABILITY_SEED + seed_offset
    )
    sb = (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")
    return {
        "metric": metric,
        "n_pairs": n,
        "pearson_r": r,
        "ci95": [lower, upper],
        "probability_positive": prob_pos,
        "spearman_brown_stepped_up": sb,
    }


def measure_split_half(team_game: pd.DataFrame, metric: str) -> dict[str, Any]:
    centered = f"{metric}_centered"
    long = team_game[["team", "season", "week", centered]].rename(
        columns={"team": "team_id", centered: metric}
    )
    return split_half_reliability(long, metric, seed=RELIABILITY_SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-season", type=Path, default=None)
    parser.add_argument("--team-game", type=Path, default=None)
    args = parser.parse_args()

    started = time.time()
    team_season_path = args.team_season or _latest_features()
    team_game_path = args.team_game or (team_season_path.parent / "team_game.parquet")
    team_season = pd.read_parquet(team_season_path)
    team_game = pd.read_parquet(team_game_path)
    print(f"team_season rows={len(team_season)}  team_game rows={len(team_game)}")

    results: list[dict[str, Any]] = []
    for dim_index, dim in enumerate(DIMENSIONS):
        floor = MIN_SEASON_SAMPLE[dim]
        count_col = COUNT_COLUMN[dim]
        qualified_season = team_season.loc[team_season[count_col].fillna(0) >= floor].copy()
        yoy = measure_year_over_year(qualified_season, dim, seed_offset=dim_index)

        split_half = measure_split_half(team_game, dim)

        reliable = (
            not np.isnan(yoy["pearson_r"])
            and yoy["ci95"][1] is not None
            and not np.isnan(yoy["ci95"][1])
            and yoy["ci95"][1] > RELIABILITY_EXCLUSION_BAR
        )
        results.append(
            {
                "dimension": dim,
                "year_over_year": yoy,
                "within_season_split_half": split_half,
                "min_season_sample_floor": floor,
                "count_column": count_col,
                "n_season_rows_qualified": len(qualified_season),
                "excluded_zero_reliability": not reliable,
            }
        )

    print("\n=== reliability table (centered dimensions) ===")
    header = (
        f"{'dimension':<22} {'YoY r':>8} {'YoY 95% CI':>18} {'YoY P+':>7} "
        f"{'YoY SB-up':>10} {'split-half SB':>14} {'n_pairs':>8} {'excluded':>9}"
    )
    print(header)
    for row in results:
        yoy = row["year_over_year"]
        sh = row["within_season_split_half"]
        ci = yoy["ci95"]
        ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if not np.isnan(ci[0]) else "n/a"
        sh_sb = sh.get("spearman_brown_full_length_reliability", float("nan"))
        excluded_str = "YES" if row["excluded_zero_reliability"] else "no"
        print(
            f"{row['dimension']:<22} {yoy['pearson_r']:+8.3f} {ci_str:>18} "
            f"{yoy['probability_positive']:7.3f} {yoy['spearman_brown_stepped_up']:+10.3f} "
            f"{sh_sb:+14.3f} {yoy['n_pairs']:8d} {excluded_str:>9}"
        )

    output_dir = (
        REPO
        / "artifacts"
        / "special_teams_reliability"
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "special-teams-reliability",
        "team_season_path": str(team_season_path),
        "team_game_path": str(team_game_path),
        "reliability_seed": RELIABILITY_SEED,
        "reliability_exclusion_bar": RELIABILITY_EXCLUSION_BAR,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "dimensions": results,
        "reliability_exclusion_bar": RELIABILITY_EXCLUSION_BAR,
        "note": (
            "Reliability computed on the LEAGUE-SEASON-CENTERED value (era/rule-change drift "
            "removed). Excluded dimensions have YoY 95% CI upper bound at or below zero -- the "
            "ONE admissible reliability-based exclusion for a cell input. An interval crossing "
            "zero on the EXPERIMENT itself (later, at screening time) is never grounds for "
            "exclusion; this is a distinct, narrower gate applied only to whether a dimension "
            "counts as a persistent trait at all."
        ),
        "provenance": artifact_provenance(configuration, team_season_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="special-teams-reliability",
        metrics=payload,
        notes=(
            "PBP-06 special-teams reliability audit -- gates which dimensions enter "
            "predeclared cells."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
