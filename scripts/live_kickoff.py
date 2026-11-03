from pathlib import Path

import pandas as pd

from nfl_ats.snapshots import latest_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
SCHEDULE_ZONE = "America/New_York"
SCHEDULE_COLUMNS = ["game_id", "season", "week", "home_team", "away_team", "gameday", "gametime"]


def live_kickoffs():
    schedule, _ = load_snapshot(latest_snapshot(RAW_ROOT))
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
