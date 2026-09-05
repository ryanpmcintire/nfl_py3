"""Per-season league-level meta-game trend series.

Read-only, descriptive. Built from two local, gitignored inputs -- nothing
here touches the network and nothing here is committed:

  - ``data/pbp/raw/<latest snapshot>/season=<season>/plays.parquet``
    (nflverse play-by-play, seasons 2009-2025 as of this run; the raw feed is
    deliberately narrowed at ingestion to the columns in
    ``nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS`` -- see the "not computable" section
    below for what that narrowing costs us)
  - ``data/processed/game_features.parquet`` (one row per game, 2009-2026;
    2026 rows are future/unplayed and drop out of every completed-game
    aggregate automatically because ``result`` is null)

This script does not build a model feature, run an experiment, or adjudicate
a weak signal. It is substrate for ``docs/era_events.md``: a set of
descriptive per-season series an era hypothesis can point at.

Binding provenance note, because this script's numbers will get quoted in
commentary on existing weak signals (AGENTS.md): every number in the parquet
this script writes is MEASURED by this run, from the local snapshots named in
its manifest. An interval or CI containing zero is never grounds to reject an
experiment; this script does not compute intervals, run significance tests,
or issue verdicts of any kind, and records nothing to
``registry/weak_signals.json``.

Not computable from the columns this repo stores today (checked dynamically
below, not assumed -- if a future snapshot adds these columns this script
will pick them up automatically):

  - average air yards, short-pass share (air_yards <= 5), deep share
    (air_yards >= 20): the stored snapshot has no ``air_yards`` column.
  - shotgun rate, no-huddle rate: no ``shotgun``/``no_huddle`` boolean, and no
    play-description text column to regex instead -- the stored ``play``
    column is a numeric valid-play flag (0/1), not text.
  - QB scramble rate specifically (as distinct from a called dropback that
    became a scramble): no ``qb_scramble`` flag and no rusher-player id to
    tell a scramble apart from a designed QB run.
  - true touchback rate (a strict boolean): no ``touchback`` column and no
    return-yardage column. A validated proxy is computed instead (see
    ``kickoff_landing_series``) using the next play's starting field
    position, which tracks the known 2011/2016/2024 kickoff rule changes in
    the expected direction and magnitude (checked by hand before writing this
    script; see docs/era_events.md).

Output: ``artifacts/metagame_series/<UTC timestamp>/series.parquet`` (one row
per season) plus a sibling ``manifest.json`` recording the source snapshot id,
season coverage, and the not-computable list actually observed on this run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.io import atomic_parquet, run_id  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402

REQUESTED_PBP_COLUMNS = {
    "air_yards": "average air yards, short-pass share (<=5), deep share (>=20)",
    "shotgun": "shotgun rate",
    "no_huddle": "no-huddle rate",
    "qb_scramble": "QB scramble rate (distinct from qb_hit/dropback)",
}


# Kickoff touchback spot (yardline_100, i.e. yards from the receiving team's
# own goal line subtracted from 100) by rule era. Read from
# registry/reference/nfl_era_events.csv's kickoff rows -- restated here as a
# plain lookup so this script stays a single, runnable file. 2009-2010: spot
# was the 20 (yardline_100=80). 2011-2015: rule moved the *kickoff*, not the
# touchback spot, so it stayed at the 20. 2016-2023: touchback moved to the
# 25 (yardline_100=75). 2024: dynamic kickoff, touchback at the 30
# (yardline_100=70). 2025+: dynamic kickoff made permanent, touchback moved
# to the 35 (yardline_100=65).
def expected_touchback_yardline_100(season: int) -> float:
    if season <= 2015:
        return 80.0
    if season <= 2023:
        return 75.0
    if season == 2024:
        return 70.0
    return 65.0


def latest_pbp_snapshot_dir(repo_root: Path) -> Path:
    manifests = sorted((repo_root / "data" / "pbp" / "raw").glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError("no data/pbp/raw/*/manifest.json snapshot found")
    return manifests[-1].parent


def available_pbp_seasons(snapshot_dir: Path) -> list[int]:
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    return sorted(p["season"] for p in manifest["partitions"])


def load_season_plays(snapshot_dir: Path, season: int) -> pd.DataFrame:
    path = snapshot_dir / f"season={season}" / "plays.parquet"
    frame = pd.read_parquet(path)
    return frame.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def fourth_down_series(df: pd.DataFrame) -> dict[str, float]:
    fourth = df[df["down"] == 4]
    decided = fourth[fourth["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    out: dict[str, float] = {}
    if len(decided) == 0:
        return {
            "fourth_down_go_rate": float("nan"),
            "fourth_down_go_rate_own_territory": float("nan"),
            "fourth_down_go_rate_plus_territory": float("nan"),
            "fourth_down_go_rate_red_zone": float("nan"),
            "fourth_down_decided_plays": 0,
        }
    decided["is_go"] = decided["play_type"].isin(["run", "pass"])
    out["fourth_down_go_rate"] = float(decided["is_go"].mean())
    out["fourth_down_decided_plays"] = len(decided)

    def region_rate(mask: pd.Series) -> float:
        subset = decided[mask]
        return float(subset["is_go"].mean()) if len(subset) else float("nan")

    out["fourth_down_go_rate_own_territory"] = region_rate(decided["yardline_100"] > 50)
    out["fourth_down_go_rate_plus_territory"] = region_rate(
        (decided["yardline_100"] > 10) & (decided["yardline_100"] <= 50)
    )
    out["fourth_down_go_rate_red_zone"] = region_rate(decided["yardline_100"] <= 10)
    return out


def pass_rate_series(df: pd.DataFrame) -> dict[str, float]:
    live = df[(df["play"] == 1) & df["pass_attempt"].notna() & df["rush_attempt"].notna()]
    live = live[(live["pass_attempt"] == 1) | (live["rush_attempt"] == 1)]
    out: dict[str, float] = {}
    out["pass_rate"] = float(live["pass_attempt"].mean()) if len(live) else float("nan")
    out["plays_run_or_pass"] = len(live)
    if "xpass" in df.columns:
        out["mean_xpass"] = float(df["xpass"].mean(skipna=True))
    if "pass_oe" in df.columns:
        # nflverse's pass_oe is a per-play residual on a percentage-point
        # scale ((actual - expected) * 100), so a single play swings from
        # -100 to +100; the *season mean* is the league PROE-style trend.
        out["mean_pass_oe_pct_points"] = float(df["pass_oe"].mean(skipna=True))
    return out


def sack_qb_hit_series(df: pd.DataFrame) -> dict[str, float]:
    dropbacks = df[df["qb_dropback"] == 1]
    out: dict[str, float] = {"dropbacks": len(dropbacks)}
    out["sack_rate"] = float(dropbacks["sack"].mean()) if len(dropbacks) else float("nan")
    out["qb_hit_rate"] = float(dropbacks["qb_hit"].mean()) if len(dropbacks) else float("nan")
    return out


def penalty_series(df: pd.DataFrame, n_games: int) -> dict[str, float]:
    out: dict[str, float] = {}
    out["penalty_rate_per_play"] = float(df["penalty"].mean(skipna=True))
    out["penalties_per_game"] = float(df["penalty"].sum()) / n_games if n_games else float("nan")
    return out


def pace_series(df: pd.DataFrame) -> dict[str, float]:
    """Game-clock seconds elapsed per real (``play`` == 1) scrimmage snap.

    Uses game-clock time (``game_seconds_remaining``), not wall-clock time,
    so this is a pace-of-play proxy (plays per unit of game clock), not a
    broadcast-runtime figure. Handles overtime explicitly: regulation is
    always 3600 game-clock seconds once a game reaches a plotted play, and an
    OT period (``qtr == 5``) counts down its own additional window from 600.
    """

    total_seconds = 0.0
    total_plays = 0
    for _game_id, game in df.groupby("game_id", sort=False):
        live_plays = int((game["play"] == 1).sum())
        if live_plays == 0:
            continue
        elapsed = 3600.0
        ot = game[game["qtr"] == 5]
        if len(ot):
            final_ot_remaining = ot["game_seconds_remaining"].min()
            if pd.notna(final_ot_remaining):
                elapsed += 600.0 - float(final_ot_remaining)
        total_seconds += elapsed
        total_plays += live_plays
    return {
        "seconds_per_play": (total_seconds / total_plays) if total_plays else float("nan"),
        "live_plays_counted": total_plays,
    }


def kickoff_landing_series(df: pd.DataFrame, season: int) -> dict[str, float]:
    """Average post-kickoff starting field position, and a touchback-rate
    proxy: the share of kickoffs whose very next play starts exactly at the
    rule era's expected touchback spot. Validated against known rule-change
    seasons before being trusted here (see the module docstring and
    docs/era_events.md): the mean shifts in the expected direction and a
    plausible magnitude at 2011, 2016, and 2024.
    """

    working = df.copy()
    working["next_yardline_100"] = working.groupby("game_id")["yardline_100"].shift(-1)
    working["next_play_type"] = working.groupby("game_id")["play_type"].shift(-1)
    kicks = working[working["play_type"] == "kickoff"]
    valid = kicks[kicks["next_play_type"].isin(["run", "pass", "no_play"])]
    if len(valid) == 0:
        return {
            "avg_post_kickoff_start_yardline_100": float("nan"),
            "touchback_rate_proxy": float("nan"),
            "kickoffs_counted": 0,
        }
    expected_line = expected_touchback_yardline_100(season)
    return {
        "avg_post_kickoff_start_yardline_100": float(valid["next_yardline_100"].mean()),
        "touchback_rate_proxy": float((valid["next_yardline_100"] == expected_line).mean()),
        "kickoffs_counted": len(valid),
    }


def market_outcome_series(game_features: pd.DataFrame, season: int) -> dict[str, float]:
    season_games = game_features[game_features["season"] == season]
    completed = season_games[season_games["result"].notna()]
    out: dict[str, float] = {"completed_games": len(completed)}
    if len(completed) == 0:
        for key in (
            "home_cover_rate",
            "home_su_win_rate",
            "avg_abs_spread",
            "offensive_ppg",
        ):
            out[key] = float("nan")
        return out
    covers = completed["home_cover"].dropna()
    out["home_cover_rate"] = float(covers.mean()) if len(covers) else float("nan")
    out["home_su_win_rate"] = float((completed["home_score"] > completed["away_score"]).mean())
    out["avg_abs_spread"] = float(completed["spread_line"].abs().mean(skipna=True))
    out["offensive_ppg"] = float(
        pd.concat([completed["home_score"], completed["away_score"]]).mean()
    )
    return out


def detect_missing_columns(sample: pd.DataFrame) -> dict[str, str]:
    return {col: desc for col, desc in REQUESTED_PBP_COLUMNS.items() if col not in sample.columns}


def main() -> int:
    snapshot_dir = latest_pbp_snapshot_dir(REPO_ROOT)
    seasons = available_pbp_seasons(snapshot_dir)
    print(
        f"PBP snapshot: {snapshot_dir.name} (seasons {seasons[0]}-{seasons[-1]}, n={len(seasons)})"
    )

    game_features = pd.read_parquet(
        REPO_ROOT / "data" / "processed" / "game_features.parquet",
        columns=[
            "season",
            "result",
            "home_score",
            "away_score",
            "spread_line",
            "home_cover",
        ],
    )

    missing_columns: dict[str, str] = {}
    rows: list[dict[str, float]] = []
    for season in seasons:
        plays = load_season_plays(snapshot_dir, season)
        if not missing_columns:
            missing_columns = detect_missing_columns(plays)

        row: dict[str, float] = {"season": season, "n_plays_raw": len(plays)}
        row.update(fourth_down_series(plays))
        row.update(pass_rate_series(plays))
        row.update(sack_qb_hit_series(plays))
        n_games = plays["game_id"].nunique()
        row["n_games_pbp"] = int(n_games)
        row.update(penalty_series(plays, n_games))
        row.update(pace_series(plays))
        row.update(kickoff_landing_series(plays, season))
        row.update(market_outcome_series(game_features, season))
        rows.append(row)
        print(f"  season {season}: {n_games} games, {len(plays)} plays -- done")

    series = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    output_id = run_id()
    output_dir = REPO_ROOT / "artifacts" / "metagame_series" / output_id
    atomic_parquet(series, output_dir / "series.parquet")
    stamp_sidecar(output_dir / "series.parquet")  # ENG-38

    manifest = {
        "built_at_utc": output_id,
        "source_pbp_snapshot": snapshot_dir.name,
        "source_game_features": "data/processed/game_features.parquet",
        "seasons": seasons,
        "rows": len(series),
        "columns": list(series.columns),
        "requested_but_not_available_columns": missing_columns,
        "notes": (
            "Descriptive series only. No experiment verdicts, no intervals, "
            "nothing recorded to registry/weak_signals.json. touchback_rate_proxy "
            "and avg_post_kickoff_start_yardline_100 are a validated proxy, not a "
            "stored touchback boolean -- see module docstring."
        ),
    }
    write_stamped_artifact(manifest, output_dir / "manifest.json")  # ENG-38

    print(f"\nWrote {len(series)} season rows to {output_dir / 'series.parquet'}")
    if missing_columns:
        print("Not computable (columns absent from the stored PBP snapshot):")
        for col, desc in missing_columns.items():
            print(f"  - {col}: {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
