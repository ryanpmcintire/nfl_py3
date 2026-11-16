from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_candidate import LIVE_SEASON, LIVE_WEEKS, PICK, kickoff_and_cutoff  # noqa: E402

import nfl_ats.pbp08_matchup_flags as matchup_flags  # noqa: E402
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, load_pbp_snapshot  # noqa: E402
from nfl_ats.pbp08_matchup_flags import build_flag_table  # noqa: E402
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import SCREEN_SEASON_START  # noqa: E402
from nfl_ats.pick_probability import (  # noqa: E402
    COUNTED_FLAG_COLUMNS,
    FLAG_DISCOVERY_DATES,
    signed_composition_flags,
)
from nfl_ats.splash_lines import load_splash_capture, splash_decision_lines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts/flag_timing_parity"
STAMP = "%Y%m%dT%H%M%SZ"
HISTORY_SEASONS = range(2020, 2026)
GAME_COLUMNS = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
PADDED: dict[str, list[str]] = {}


def padded_load(snapshot, **kwargs):
    frame = load_pbp_snapshot(snapshot, **kwargs)
    missing = [column for column in PBP_SNAPSHOT_COLUMNS if column not in frame.columns]
    if missing:
        PADDED[str(getattr(snapshot, "root", snapshot))] = missing
    return frame.assign(**dict.fromkeys(missing, np.nan))


matchup_flags.load_pbp_snapshot = padded_load


def stamped(paths):
    rows = []
    for path in paths:
        name = path.name if path.is_dir() else path.parent.name
        stamp = pd.to_datetime(name, format=STAMP, errors="coerce", utc=True)
        if pd.notna(stamp):
            rows.append((stamp, path))
    return sorted(rows, key=lambda row: row[0])


def as_of(snapshots, moment):
    eligible = [path for stamp, path in snapshots if stamp < moment]
    return eligible[-1] if eligible else None


def reg_schedule(path):
    frame = pd.read_parquet(path)
    return frame.loc[frame["game_type"].astype(str).eq("REG")].copy()


def flags_for(games, schedule_path, pbp_path, cache):
    key = (str(schedule_path), str(pbp_path))
    if key not in cache:
        schedule = reg_schedule(schedule_path)
        scoped = schedule.loc[schedule["season"].between(SCREEN_SEASON_START, LIVE_SEASON)].copy()
        cache[key] = (schedule, build_flag_table(scoped, pbp_path)[["game_id", "back_side"]])
        print(f"built {pbp_path.name} x {schedule_path.parent.name}", flush=True)
    schedule, table = cache[key]
    flags = signed_composition_flags(games, schedule, protection_back_side=table)
    return flags[["game_id", *COUNTED_FLAG_COLUMNS]]


def record(values, outcome):
    signed = np.sign(values) * outcome
    fired = np.sign(values).ne(0)
    return {
        "nonzero": int(fired.sum()),
        "wins": int(signed.gt(0).sum()),
        "losses": int(signed.lt(0).sum()),
        "pushes": int((fired & signed.eq(0)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train/serve parity of the counted composition flags at each pick deadline"
    )
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pbp_snaps = stamped(p for p in (DATA / "pbp/raw").iterdir() if p.is_dir())
    sched_snaps = stamped(DATA.glob("raw/*/schedules.parquet"))
    latest_pbp, latest_sched = pbp_snaps[-1][1], sched_snaps[-1][1]
    cache = {}

    latest_schedule = reg_schedule(latest_sched)
    live = latest_schedule.loc[
        latest_schedule["season"].eq(LIVE_SEASON) & latest_schedule["week"].isin(list(LIVE_WEEKS))
    ].reset_index(drop=True)
    _kick, deadline = kickoff_and_cutoff(live)
    live["deadline_utc"] = deadline.to_numpy()
    live["pbp_asof"] = [as_of(pbp_snaps, d) for d in live["deadline_utc"]]
    live["sched_asof"] = [as_of(sched_snaps, d) for d in live["deadline_utc"]]

    latest_flags = flags_for(live[GAME_COLUMNS], latest_sched, latest_pbp, cache)
    parts = []
    for (sched_path, pbp_path), group in live.groupby(["sched_asof", "pbp_asof"], sort=False):
        part = flags_for(group[GAME_COLUMNS], sched_path, pbp_path, cache)
        part["pbp_asof"] = pbp_path.name
        part["sched_asof"] = sched_path.parent.name
        parts.append(part)
    asof_flags = pd.concat(parts, ignore_index=True)

    lines = {}
    for week in LIVE_WEEKS:
        capture = load_splash_capture(DATA, LIVE_SEASON, int(week))
        if capture is not None:
            lines.update(splash_decision_lines(capture))
    live["game_id"] = live["game_id"].astype(str)
    live["pool_home_spread"] = live["game_id"].map(lines)
    live["outcome"] = np.sign(pd.to_numeric(live["result"], errors="coerce") - live["pool_home_spread"])
    merged = live[["game_id", "week", "gameday", "deadline_utc", "outcome"]].merge(
        asof_flags, on="game_id"
    ).merge(latest_flags, on="game_id", suffixes=("_asof", "_latest"))
    graded = merged.loc[merged["outcome"].notna()]

    summary = {
        "latest_pbp": latest_pbp.name,
        "latest_schedule": latest_sched.parent.name,
        "counted_flags": list(COUNTED_FLAG_COLUMNS),
        "live_2026": {},
        "history": {},
    }
    for column in COUNTED_FLAG_COLUMNS:
        after = pd.to_datetime(graded["gameday"]).gt(pd.Timestamp(FLAG_DISCOVERY_DATES[column]))
        rows = graded.loc[after]
        diff = merged[f"{column}_asof"].ne(merged[f"{column}_latest"])
        summary["live_2026"][column] = {
            "latest_snapshot_replay": record(rows[f"{column}_latest"], rows["outcome"]),
            "deadline_asof_replay": record(rows[f"{column}_asof"], rows["outcome"]),
            "games_differing": int(diff.sum()),
            "differing_by_week": {
                str(int(week)): int(count)
                for week, count in merged.loc[diff].groupby("week").size().items()
            },
        }
    summary["asof_snapshot_by_week"] = {
        str(int(week)): sorted(set(group["pbp_asof"]))
        for week, group in merged.groupby("week")
    }

    per_game = pd.read_parquet(PICK / "per_game.parquet")
    per_game["game_id"] = per_game["game_id"].astype(str)
    history_games = latest_schedule.loc[latest_schedule["season"].isin(list(HISTORY_SEASONS))]
    history_games = history_games.assign(game_id=history_games["game_id"].astype(str))
    history_flags = flags_for(history_games[GAME_COLUMNS], latest_sched, latest_pbp, cache)
    earliest_pbp = pbp_snaps[0][1]
    early_flags = flags_for(history_games[GAME_COLUMNS], latest_sched, earliest_pbp, cache)
    fit = per_game[["game_id", "season", "margin_vs_open"]].merge(history_flags, on="game_id")
    fit["outcome"] = np.sign(pd.to_numeric(fit["margin_vs_open"], errors="coerce"))
    stored = [column for column in COUNTED_FLAG_COLUMNS if column in per_game.columns]
    for column in COUNTED_FLAG_COLUMNS:
        by_season = {
            str(int(season)): record(group[column], group["outcome"])
            for season, group in fit.groupby("season")
        }
        compare = history_flags.merge(early_flags, on="game_id", suffixes=("", "_early"))
        entry = {
            "fit_population_games": len(fit),
            "all": record(fit[column], fit["outcome"]),
            "by_season": by_season,
            "revision_vs_earliest_pbp": {
                "earliest_pbp": earliest_pbp.name,
                "games_differing": int(compare[column].ne(compare[f"{column}_early"]).sum()),
            },
        }
        if column in stored:
            check = per_game[["game_id", column]].merge(
                history_flags, on="game_id", suffixes=("_stored", "")
            )
            entry["stored_fit_mismatch"] = int(check[column].ne(check[f"{column}_stored"]).sum())
        summary["history"][column] = entry

    summary["snapshots_failing_current_column_contract"] = PADDED
    merged.drop(columns=["deadline_utc"]).to_csv(OUT / "live_2026_per_game.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
