"""Re-derive the three NFL-side estimates behind ``DEFAULT_OFFSEASON_RETENTION``.

Predeclaration / context: ``docs/offseason_retention.md``. This script rebuilds,
from local raw data, the three independent estimation routes the 2026-08-17
session ("Close four underived constants by measuring them") reported for the
0.67 constant (0.337 / 0.379 / 0.400, none of whose intervals contain 0.67).
No script from that session survived (it was run ad hoc), so this is a fresh
implementation of the same methodology, not a replay of the same code. Where a
number does not reproduce exactly, that is reported rather than papered over.

Three routes, all using ``current = league_mean + retention**gap * (current -
league_mean)`` as the quantity being estimated:

* **Route 1** -- per-(metric, horizon) regression slope. For every team and
  every consecutive season boundary, regress the next season's within-season
  centered mean (at a given games-into-season horizon) on the prior season's
  full-season centered mean. The slope is a direct estimate of retention for
  that metric/horizon cell. Reported across all ``STATE_METRICS`` and horizons
  1-4 games into the next season (the horizon the constant actually governs
  before in-season EWM data dominates), plus the full next season as a check.
* **Route 2** -- the shipped feature table's own behaviour. Using the already
  built ``data/processed/game_features.parquet`` (built at the current
  DEFAULT_OFFSEASON_RETENTION=0.67), regress ``result ~ diff_point_diff``
  separately for week-1 games (freshly regressed state) and weeks 9-18 games
  (steady-state, in-season EWM dominates). If retention is too high, the
  week-1 feature carries excess unshrunk noise and its slope is attenuated
  relative to the late-season slope; the ratio, applied to 0.67, estimates the
  retention that would remove the attenuation.
* **Route 3** -- plain point-differential regression, both seasons' team means
  centered within season, at two horizons (full next season, first 4 games).

Every slope carries a season-blocked bootstrap interval (resampling season
boundaries with replacement, so no observation is split across resamples).

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/offseason_retention_routes.py \\
        --output <scratch dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION, STATE_METRICS
from nfl_ats.features import add_ats_outcomes, build_team_game_metrics
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260817


def _season_team_means(
    team_games: pd.DataFrame, metric: str, first_n_games: int | None
) -> pd.DataFrame:
    """Mean of ``metric`` per team-season, optionally over only the first N games."""

    frame = team_games[["team", "season", "gameday", "game_id", metric]].dropna(subset=[metric])
    frame = frame.sort_values(["team", "season", "gameday", "game_id"]).copy()
    if first_n_games is not None:
        frame["game_rank"] = frame.groupby(["team", "season"]).cumcount() + 1
        frame = frame.loc[frame["game_rank"] <= first_n_games]
    grouped = frame.groupby(["team", "season"])[metric].mean().reset_index()
    return grouped.rename(columns={metric: "value"})


def _centered(frame: pd.DataFrame) -> pd.DataFrame:
    league = frame.groupby("season")["value"].mean().rename("league_mean").reset_index()
    merged = frame.merge(league, on="season")
    merged["centered"] = merged["value"] - merged["league_mean"]
    return merged


def _transitions(team_games: pd.DataFrame, metric: str, next_horizon: int | None) -> pd.DataFrame:
    """One row per (team, consecutive season boundary): prior full-season state vs.

    the next season's centered mean at ``next_horizon`` games (None = full season).
    """

    prior = _centered(_season_team_means(team_games, metric, first_n_games=None))
    nxt = _centered(_season_team_means(team_games, metric, first_n_games=next_horizon))
    prior = prior[["team", "season", "centered"]].rename(
        columns={"season": "prior_season", "centered": "prior_centered"}
    )
    nxt = nxt[["team", "season", "centered"]].rename(
        columns={"season": "next_season", "centered": "next_centered"}
    )
    prior = prior.copy()
    prior["next_season"] = prior["prior_season"] + 1
    return prior.merge(nxt, on=["team", "next_season"], how="inner")


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _season_block_bootstrap(
    merged: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    seasons = merged["prior_season"].unique()
    rng = np.random.default_rng(seed)
    by_season = dict(iter(merged.groupby("prior_season")))
    slopes = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.choice(seasons, size=len(seasons), replace=True)
        resampled = pd.concat([by_season[season] for season in chosen], ignore_index=True)
        slope, _ = _ols_slope(
            resampled["prior_centered"].to_numpy(), resampled["next_centered"].to_numpy()
        )
        slopes[index] = slope
    return slopes


def route1(team_games: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 4)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in STATE_METRICS:
        for horizon in horizons:
            merged = _transitions(team_games, metric, horizon)
            if len(merged) < 20:
                continue
            slope, intercept = _ols_slope(
                merged["prior_centered"].to_numpy(), merged["next_centered"].to_numpy()
            )
            slopes = _season_block_bootstrap(merged, samples=1000)
            lower, upper = np.quantile(slopes, [0.025, 0.975])
            rows.append(
                {
                    "metric": metric,
                    "horizon_games": horizon,
                    "slope": slope,
                    "intercept": intercept,
                    "n_transitions": len(merged),
                    "n_seasons": int(merged["prior_season"].nunique()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def route3(team_games: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for horizon_name, horizon in (("full_next_season", None), ("first_4_games", 4)):
        merged = _transitions(team_games, "point_diff", horizon)
        slope, intercept = _ols_slope(
            merged["prior_centered"].to_numpy(), merged["next_centered"].to_numpy()
        )
        slopes = _season_block_bootstrap(merged)
        lower, upper = np.quantile(slopes, [0.025, 0.975])
        results[horizon_name] = {
            "slope": slope,
            "intercept": intercept,
            "n_transitions": len(merged),
            "n_seasons": int(merged["prior_season"].nunique()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
    return results


def route2(game_features_path: Path) -> dict[str, Any]:
    features = pd.read_parquet(game_features_path)
    if "game_type" in features.columns:
        features = features.loc[features["game_type"].eq("REG")]
    frame = features.loc[features["result"].notna() & features["diff_point_diff"].notna()].copy()
    week1 = frame.loc[frame["week"].eq(1)]
    late = frame.loc[frame["week"].between(9, 18)]
    slope_w1, intercept_w1 = _ols_slope(
        week1["diff_point_diff"].to_numpy(), week1["result"].to_numpy()
    )
    slope_late, intercept_late = _ols_slope(
        late["diff_point_diff"].to_numpy(), late["result"].to_numpy()
    )
    ratio = slope_w1 / slope_late
    return {
        "week1_slope": slope_w1,
        "week1_intercept": intercept_w1,
        "week1_n": len(week1),
        "late_season_slope": slope_late,
        "late_season_intercept": intercept_late,
        "late_season_n": len(late),
        "ratio": ratio,
        "implied_retention": DEFAULT_OFFSEASON_RETENTION * ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-features", type=Path, default=DEFAULT_GAME_FEATURES)
    parser.add_argument("--raw-root", type=Path, default=REPO / "data" / "raw")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("loading latest NFL raw snapshot ...")
    snapshot = latest_snapshot(args.raw_root)
    schedules, team_stats = load_snapshot(snapshot)
    schedules = add_ats_outcomes(schedules)
    team_games = build_team_game_metrics(schedules, team_stats)
    print(
        f"team_games: {len(team_games)} rows, "
        f"seasons {int(team_games['season'].min())}-{int(team_games['season'].max())}, "
        f"{team_games['team'].nunique()} teams"
    )
    team_games.to_parquet(args.output / "nfl_team_games.parquet", index=False)

    print("\n=== route 3: plain point_diff regression (season-centered) ===")
    r3 = route3(team_games)
    print(json.dumps(r3, indent=2))

    print("\n=== route 1: per-metric/horizon slopes ===")
    r1 = route1(team_games)
    r1.to_csv(args.output / "route1_cells.csv", index=False)
    h4 = r1.loc[r1["horizon_games"].eq(4)]
    print(
        f"horizon=4 games, {len(h4)} metrics: median slope {h4['slope'].median():.3f}, "
        f"range [{h4['slope'].min():.3f}, {h4['slope'].max():.3f}]"
    )
    above = int((r1["ci_upper"] >= DEFAULT_OFFSEASON_RETENTION).sum())
    print(f"cells whose 95% upper bound reaches {DEFAULT_OFFSEASON_RETENTION}: {above} / {len(r1)}")
    print(r1.sort_values(["horizon_games", "metric"]).to_string(index=False))

    print("\n=== route 2: shipped feature table, week1 vs weeks9-18 slope ===")
    r2 = route2(args.game_features)
    print(json.dumps(r2, indent=2))

    summary = {
        "default_offseason_retention": DEFAULT_OFFSEASON_RETENTION,
        "route1_cells": r1.to_dict(orient="records"),
        "route2": r2,
        "route3": r3,
    }
    (args.output / "routes_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
