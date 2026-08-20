"""Reliability audit for PBP-08 team "style" dimensions -- run BEFORE any cell
is predeclared, per AGENTS.md label-how-you-know-it discipline and the
task's own ordering: a style dimension that doesn't persist year-over-year
is not a personality, so this is the gate, not a footnote.

Two independent reliability estimates per dimension, both computed on the
LEAGUE-SEASON-CENTERED value (raw values would conflate era-wide drift with
team identity -- PER-07's own precedent: "naive reliability without the
season-norm correction is +0.666" vs a season-normalized +0.320):

1. **Year-over-year** (primary, matches PER-07's own headline framing):
   Pearson r between a team's centered value in season ``t`` and season
   ``t+1``, pooled across all same-franchise consecutive-season pairs
   2009-2025 (``TEAM_ABBREVIATION_ALIASES`` applied upstream in
   ``team_style_features.py``, so OAK->LV/SD->LAC/STL->LA count as one
   continuous franchise). Spearman-Brown stepped up (``2r/(1+r)``) is also
   reported as a secondary "reliability of the two-year-average trait"
   figure, exactly the transform PER-07 and PBP-05 both use, though applying
   it to a year-over-year pair (rather than a true within-season split) is
   noted explicitly as an extrapolation, not a second independent estimate.

2. **Within-season split-half** (secondary cross-check): odd/even-week
   team-season split, Spearman-Brown corrected, using
   ``nfl_ats.cfb_qb_dependence.split_half_reliability`` REUSED DIRECTLY
   (not reimplemented) -- the same function and formula PBP-05's
   0.80-offense/0.46-defense figure and the CFB role-continuity family's
   0.719/0.680 figures were built on.

Binding taxonomy (own it, do not paraphrase): an interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment --
at this evaluator's ~2-point resolution "contains zero" is the EXPECTED
outcome for a real small signal. Only two closing grounds: (1) refuted
mechanism -- RESOLVED wrong sign (whole interval on the wrong side of zero)
or zero split-half reliability; (2) bounded by a positive control proven
able to detect an effect that size. Everything else is
unresolved_below_power: record with probability_positive, never "contains
zero". Reliability is the ONE admissible exclusion criterion for a cell
INPUT (not a whole experiment): a dimension with ~zero year-over-year
reliability is not a personality trait and is excluded from the predeclared
cells on that ground alone, documented here before any cell is written.

Writes ``artifacts/team_style_reliability/<UTC timestamp>/results.json`` and
prints the reliability table to stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from team_style_features import (  # noqa: E402
    STYLE_DIMENSIONS,
    TEAM_GAME_PATH,
    TEAM_SEASON_PATH,
)

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

RELIABILITY_SEED = 20260819
MIN_TEAM_SEASON_PLAYS = {
    "short_pass_share": 100,
    "deep_share": 100,
    "avg_air_yards": 100,
    "screen_rate": 100,
    "proe": 100,
    "shotgun_rate": 100,
    "no_huddle_rate": 100,
    "run_direction_hhi": 100,
    "seconds_per_play_pace": 100,
}
DENOMINATOR_COLUMN = {
    "short_pass_share": "n_pass_plays",
    "deep_share": "n_pass_plays",
    "avg_air_yards": "n_pass_plays",
    "screen_rate": "n_pass_plays",
    "proe": "n_off_plays",
    "shotgun_rate": "n_off_plays",
    "no_huddle_rate": "n_off_plays",
    "run_direction_hhi": "n_run_plays",
    "seconds_per_play_pace": "n_off_plays",
}
# The reliability-based exclusion threshold -- the ONE admissible
# reliability exclusion this task allows. A dimension whose year-over-year
# Pearson r 95% CI (via team-season block bootstrap) sits entirely at or
# below this bar is not treated as a personality trait for cell-building.
# 0.0 (not some higher bar) so the exclusion is "genuinely indistinguishable
# from a random redraw each year", never a precision complaint.
RELIABILITY_EXCLUSION_BAR = 0.0


def year_over_year_pairs(team_season: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = team_season[["team", "season", metric]].rename(columns={metric: "value_t"})
    right = team_season[["team", "season", metric]].rename(columns={metric: "value_t1"})
    right["season"] = right["season"] - 1  # season_t1's season-1 = season_t
    pairs = left.merge(right, on=["team", "season"], how="inner")
    return pairs.dropna(subset=["value_t", "value_t1"])


def bootstrap_pearson_ci(
    x: np.ndarray, y: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, float]:
    """Block bootstrap over TEAM-SEASON PAIRS (each pair is one resample unit)."""

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
    # seed_offset is the dimension's fixed STYLE_DIMENSIONS index, NOT
    # hash(metric) -- Python's string hash is PYTHONHASHSEED-randomized
    # across processes, which would silently break the "fixed seed,
    # deterministic" requirement every dimension here is held to.
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
    result = split_half_reliability(long, metric, seed=RELIABILITY_SEED)
    return result


def main() -> None:
    started = time.time()
    team_game = pd.read_parquet(TEAM_GAME_PATH)
    team_season = pd.read_parquet(TEAM_SEASON_PATH)

    print(f"team_game rows={len(team_game)}  team_season rows={len(team_season)}")

    results: list[dict[str, Any]] = []
    for dim_index, dim in enumerate(STYLE_DIMENSIONS):
        denom_col = DENOMINATOR_COLUMN[dim]
        floor = MIN_TEAM_SEASON_PLAYS[dim]
        qualified_season = team_season.loc[team_season[denom_col].fillna(0) >= floor].copy()
        yoy = measure_year_over_year(qualified_season, dim, seed_offset=dim_index)

        qualified_game = team_game.loc[team_game[denom_col].fillna(0) >= 5].copy()
        split_half = measure_split_half(qualified_game, dim)

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
                "min_plays_floor": floor,
                "denominator_column": denom_col,
                "excluded_zero_reliability": not reliable,
            }
        )

    print("\n=== reliability table (centered dimensions) ===")
    header = (
        f"{'dimension':<24} {'YoY r':>8} {'YoY 95% CI':>18} {'YoY P+':>7} "
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
            f"{row['dimension']:<24} {yoy['pearson_r']:+8.3f} {ci_str:>18} "
            f"{yoy['probability_positive']:7.3f} {yoy['spearman_brown_stepped_up']:+10.3f} "
            f"{sh_sb:+14.3f} {yoy['n_pairs']:8d} {excluded_str:>9}"
        )

    output_dir = (
        REPO
        / "artifacts"
        / "team_style_reliability"
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "team-style-reliability",
        "team_game_path": str(TEAM_GAME_PATH),
        "team_season_path": str(TEAM_SEASON_PATH),
        "reliability_seed": RELIABILITY_SEED,
        "reliability_exclusion_bar": RELIABILITY_EXCLUSION_BAR,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "dimensions": results,
        "reliability_exclusion_bar": RELIABILITY_EXCLUSION_BAR,
        "note": (
            "Reliability computed on the LEAGUE-SEASON-CENTERED value (era-drift removed), "
            "matching PER-07 convention. Excluded dimensions have YoY 95% CI upper bound at or "
            "below zero -- the ONE admissible reliability-based exclusion for a cell input, per "
            "the task predeclaration. An interval crossing zero on the EXPERIMENT itself (later, "
            "at screening time) is never grounds for exclusion; this is a distinct, narrower gate "
            "applied only to whether a dimension counts as a persistent trait at all."
        ),
        "provenance": artifact_provenance(configuration, TEAM_SEASON_PATH, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="team-style-reliability",
        metrics=payload,
        notes=(
            "PBP-08 team style reliability audit -- gates which dimensions enter predeclared cells."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
