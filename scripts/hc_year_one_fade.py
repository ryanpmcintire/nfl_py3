"""Reproduce and sanity-check `hc_year_one_fade` (ROADMAP PER-07) before any
confirmation window is declared.

The recorded finding: a first-year head coach's team fades against the
spread in weeks 1-8, ~-1.30 raw points / 46.72% cover on 762 games (104
team-season clusters), independently replicated on CFB, but the effect lives
inside the mined 2018-2025 era (null in 2009-2017). This script rebuilds the
same measurement from scratch, on data already local to this repo:

- Coach identity comes straight from nflverse's raw schedule feed
  (``data/raw/*/schedules.parquet``: ``home_coach``/``away_coach``), not a
  bespoke ingestion this project built -- so nothing about the SOURCE is a
  hidden assumption.
- Franchise relocations (OAK->LV, SD->LAC, STL->LA) are normalized with the
  SAME ``TEAM_ABBREVIATION_ALIASES`` the rest of this project already uses
  (``nfl_ats.constants``), so a relocation is never mistaken for a coaching
  change or a first NFL season.
- "Year 1" requires an OBSERVED prior season for that franchise with a
  DIFFERENT primary coach; a franchise's very first season in the data
  (2009) has unknown tenure and is excluded, not counted as year 1.

This is a research/write-up script, not a registry look: it never calls
``nfl_ats.rotation.assign_window``/``record_look``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULES = REPO / "data/raw/20260817T235649Z/schedules.parquet"
DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"

_TEAM_GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "team",
    "opponent",
    "coach",
    "is_home",
    "team_covered",
    "team_ats_margin",
    "result",
    "spread_line",
]


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def build_team_game_table(schedules: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side): team identity, coach, and ATS outcome."""

    sched = schedules.loc[
        :,
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
        ],
    ].copy()
    feat = features.loc[:, ["game_id", "ats_margin", "home_cover", "spread_line", "result"]].copy()
    merged = sched.merge(feat, on="game_id", how="inner", validate="one_to_one")
    merged["home_team"] = _canonical(merged["home_team"])
    merged["away_team"] = _canonical(merged["away_team"])

    home_rows = merged.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["coach"] = home_rows["home_coach"]
    home_rows["is_home"] = True
    home_rows["team_covered"] = home_rows["home_cover"]
    home_rows["team_ats_margin"] = home_rows["ats_margin"]

    away_rows = merged.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["coach"] = away_rows["away_coach"]
    away_rows["is_home"] = False
    away_rows["team_covered"] = 1.0 - away_rows["home_cover"]
    away_rows["team_ats_margin"] = -away_rows["ats_margin"]

    long = pd.concat(
        [home_rows[_TEAM_GAME_COLUMNS], away_rows[_TEAM_GAME_COLUMNS]], ignore_index=True
    )
    return long


def team_season_primary_coach(long: pd.DataFrame) -> pd.DataFrame:
    """The most common coach name credited to each (team, season) in REG games."""

    reg = long.loc[long["game_type"].eq("REG") & long["coach"].notna()]
    primary = (
        reg.groupby(["team", "season"])["coach"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
        .rename(columns={"coach": "primary_coach"})
    )
    return primary


def flag_year_one(primary: pd.DataFrame) -> pd.DataFrame:
    """Add ``year_one``/``kept_coach`` flags; both require an observed prior season."""

    ordered = primary.sort_values(["team", "season"]).copy()
    ordered["prev_coach"] = ordered.groupby("team")["primary_coach"].shift(1)
    ordered["prev_season"] = ordered.groupby("team")["season"].shift(1)
    ordered["known_tenure"] = ordered["prev_coach"].notna() & (
        ordered["season"] - ordered["prev_season"] == 1
    )
    ordered["year_one"] = ordered["known_tenure"] & (
        ordered["primary_coach"] != ordered["prev_coach"]
    )
    ordered["kept_coach"] = ordered["known_tenure"] & (
        ordered["primary_coach"] == ordered["prev_coach"]
    )
    return ordered


def team_season_quality(long: pd.DataFrame) -> pd.DataFrame:
    """Prior-season average scoring margin per (team, season), as a quality proxy.

    Computed from each team's OWN prior regular season (never the current
    season, so it cannot leak into a same-season quality control), and
    tercile-bucketed WITHIN each season so "quality" is always relative to
    that season's league, not distorted by era-wide scoring drift.
    """

    reg = long.loc[long["game_type"].eq("REG")].copy()
    reg["scoring_margin"] = np.where(
        reg["is_home"], reg["result"], -pd.to_numeric(reg["result"], errors="coerce")
    )
    season_margin = reg.groupby(["team", "season"])["scoring_margin"].mean().reset_index()
    season_margin = season_margin.sort_values(["team", "season"])
    season_margin["prior_season_margin"] = season_margin.groupby("team")["scoring_margin"].shift(1)
    season_margin["prior_season"] = season_margin.groupby("team")["season"].shift(1)
    season_margin = season_margin.loc[season_margin["season"] - season_margin["prior_season"] == 1]

    def _tercile(group: pd.DataFrame) -> pd.Series:
        return pd.qcut(group["prior_season_margin"], 3, labels=["low", "mid", "high"])

    season_margin["quality_tercile"] = season_margin.groupby("season", group_keys=False).apply(
        _tercile
    )
    return season_margin.loc[:, ["team", "season", "prior_season_margin", "quality_tercile"]]


def cluster_bootstrap_mean(
    values: pd.Series, clusters: pd.Series, *, samples: int = 5000, seed: int = 20260818
) -> tuple[float, float, float]:
    """Cluster (team-season) block-bootstrap mean and a 95% percentile interval."""

    frame = pd.DataFrame({"value": values.to_numpy(dtype=float), "cluster": clusters.to_numpy()})
    grouped = frame.groupby("cluster")["value"]
    cluster_means = grouped.mean()
    cluster_sizes = grouped.size()
    point_estimate = float(frame["value"].mean())
    rng = np.random.default_rng(seed)
    cluster_ids = cluster_means.index.to_numpy()
    means_arr = cluster_means.to_numpy()
    sizes_arr = cluster_sizes.to_numpy(dtype=float)
    draws = np.empty(samples, dtype=float)
    n_clusters = len(cluster_ids)
    for sample_index in range(samples):
        selected = rng.integers(0, n_clusters, size=n_clusters)
        draws[sample_index] = np.average(means_arr[selected], weights=sizes_arr[selected])
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point_estimate, float(lower), float(upper)


def headline_measurement(
    long: pd.DataFrame, tenure: pd.DataFrame, *, week_max: int = 8
) -> dict[str, Any]:
    weeks = long.loc[long["game_type"].eq("REG") & long["week"].le(week_max)].copy()
    weeks = weeks.merge(tenure, on=["team", "season"], how="inner")
    weeks = weeks.loc[weeks["team_covered"].notna()]

    year_one_games = weeks.loc[weeks["year_one"]]
    kept_games = weeks.loc[weeks["kept_coach"]]

    year_one_cluster = year_one_games["team"] + "_" + year_one_games["season"].astype(str)
    kept_cluster = kept_games["team"] + "_" + kept_games["season"].astype(str)

    year_one_cover, year_one_lower, year_one_upper = cluster_bootstrap_mean(
        year_one_games["team_covered"], year_one_cluster
    )
    kept_cover, kept_lower, kept_upper = cluster_bootstrap_mean(
        kept_games["team_covered"], kept_cluster
    )

    era_rows = []
    for label, mask in (
        ("2009_2017", year_one_games["season"].between(2009, 2017)),
        ("2018_2025", year_one_games["season"].between(2018, 2025)),
    ):
        subset = year_one_games.loc[mask]
        if subset.empty:
            continue
        cluster = subset["team"] + "_" + subset["season"].astype(str)
        cover, lower, upper = cluster_bootstrap_mean(subset["team_covered"], cluster)
        era_rows.append(
            {
                "era": label,
                "games": len(subset),
                "clusters": int(cluster.nunique()),
                "cover_rate": cover,
                "ci_lower": lower,
                "ci_upper": upper,
                "raw_points": float(subset["team_ats_margin"].mean()),
            }
        )

    return {
        "week_max": week_max,
        "year_one": {
            "games": len(year_one_games),
            "clusters": int(year_one_cluster.nunique()),
            "cover_rate": year_one_cover,
            "ci_lower": year_one_lower,
            "ci_upper": year_one_upper,
            "raw_points": float(year_one_games["team_ats_margin"].mean()),
        },
        "kept_coach": {
            "games": len(kept_games),
            "clusters": int(kept_cluster.nunique()),
            "cover_rate": kept_cover,
            "ci_lower": kept_lower,
            "ci_upper": kept_upper,
            "raw_points": float(kept_games["team_ats_margin"].mean()),
        },
        "era_split": era_rows,
    }


def quality_controlled_comparison(
    long: pd.DataFrame, tenure: pd.DataFrame, quality: pd.DataFrame, *, week_max: int = 8
) -> pd.DataFrame:
    weeks = long.loc[long["game_type"].eq("REG") & long["week"].le(week_max)].copy()
    weeks = weeks.merge(tenure, on=["team", "season"], how="inner")
    weeks = weeks.merge(quality, on=["team", "season"], how="inner")
    weeks = weeks.loc[weeks["team_covered"].notna() & weeks["quality_tercile"].notna()]

    rows = []
    for group_name, mask in (("year_one", weeks["year_one"]), ("kept_coach", weeks["kept_coach"])):
        subset = weeks.loc[mask]
        for tercile, tercile_group in subset.groupby("quality_tercile", observed=True):
            cluster = tercile_group["team"] + "_" + tercile_group["season"].astype(str)
            rows.append(
                {
                    "group": group_name,
                    "quality_tercile": tercile,
                    "games": len(tercile_group),
                    "clusters": cluster.nunique(),
                    "cover_rate": float(tercile_group["team_covered"].mean()),
                }
            )
        cluster = subset["team"] + "_" + subset["season"].astype(str)
        rows.append(
            {
                "group": group_name,
                "quality_tercile": "ALL",
                "games": len(subset),
                "clusters": cluster.nunique(),
                "cover_rate": float(subset["team_covered"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--week-max", type=int, default=8)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    schedules = pd.read_parquet(args.schedules)
    features = pd.read_parquet(args.features)
    long = build_team_game_table(schedules, features)
    primary = team_season_primary_coach(long)
    tenure = flag_year_one(primary)
    quality = team_season_quality(long)

    tenure.to_parquet(output / "team_season_tenure.parquet", index=False)
    quality.to_parquet(output / "team_season_quality.parquet", index=False)

    headline = headline_measurement(long, tenure, week_max=args.week_max)
    controlled = quality_controlled_comparison(long, tenure, quality, week_max=args.week_max)
    controlled.to_csv(output / "quality_controlled_comparison.csv", index=False)

    print(f"=== headline: weeks 1-{args.week_max}, REG only ===")
    for label in ("year_one", "kept_coach"):
        summary = headline[label]
        print(
            f"{label:>11}: {summary['cover_rate']:.4f} cover "
            f"[{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}] "
            f"on {summary['games']} games / {summary['clusters']} team-season clusters, "
            f"{summary['raw_points']:+.3f} raw ATS points"
        )
    print("\n=== era split (year-one only) ===")
    for row in headline["era_split"]:
        print(
            f"{row['era']}: {row['cover_rate']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] "
            f"on {row['games']} games / {row['clusters']} clusters, {row['raw_points']:+.3f} pts"
        )
    print("\n=== quality-controlled comparison (prior-season scoring-margin tercile) ===")
    print(controlled.to_string(index=False))

    (output / "diagnostics.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
