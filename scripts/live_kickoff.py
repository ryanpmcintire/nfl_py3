from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
SCHEDULE_ROOT = ROOT / "data" / "raw_schedules"
SCHEDULE_ZONE = "America/New_York"
SCHEDULE_COLUMNS = ["game_id", "season", "week", "home_team", "away_team", "gameday", "gametime"]


def newest_schedule():
    candidates = []
    for root in (RAW_ROOT, SCHEDULE_ROOT):
        folders = [p for p in root.glob("*") if (p / "manifest.json").is_file()]
        folders = [p for p in folders if (p / "schedules.parquet").is_file()]
        if folders:
            newest = max(folders, key=lambda p: p.name)
            candidates.append(newest)
    if not candidates:
        raise FileNotFoundError("No schedule snapshots found")
    return max(candidates, key=lambda p: p.name)


def live_kickoffs():
    schedule = pd.read_parquet(newest_schedule() / "schedules.parquet")
    games = schedule[SCHEDULE_COLUMNS].dropna(subset=["gameday", "gametime"]).copy()
    local = pd.to_datetime(games["gameday"].astype(str) + " " + games["gametime"].astype(str))
    games["live_kickoff"] = local.dt.tz_localize(SCHEDULE_ZONE).dt.tz_convert("UTC")
    return games.drop(columns=["gameday", "gametime"]).reset_index(drop=True)


def earliest_by_game_id(frame, kickoff_col="kickoff", id_col="game_id"):
    live = live_kickoffs().set_index("game_id")["live_kickoff"]
    current = pd.to_datetime(frame[kickoff_col], utc=True)
    mapped = frame[id_col].map(live)
    return pd.concat([current, mapped], axis=1).min(axis=1)


def earliest_by_matchup(frame, kickoff_col="kickoff"):
    keys = ["season", "week", "home_team", "away_team"]
    live = live_kickoffs()[[*keys, "live_kickoff"]]
    merged = frame[keys].merge(live, on=keys, how="left")
    current = pd.to_datetime(frame[kickoff_col], utc=True).reset_index(drop=True)
    out = pd.concat([current, merged["live_kickoff"]], axis=1).min(axis=1)
    out.index = frame.index
    return out
