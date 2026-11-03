import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ats import cfb_features as cf  # noqa: E402

OUT = ROOT / "artifacts" / "xlg12_cfb_forward" / "features_2026.parquet"
CFB_ROOT = ROOT / "data" / "cfb"
TARGET = 2026
HISTORY_START = 2019


def main():
    seasons = list(range(HISTORY_START, TARGET + 1))
    present = cf.cfb_season_partitions(CFB_ROOT, "pbp")
    seasons = [s for s in seasons if s in present]
    schedules = cf.load_cfb_seasons(
        CFB_ROOT, "schedules", seasons, columns=list(cf._SCHEDULE_LOAD_COLUMNS)
    )
    pbp = cf.load_cfb_seasons(CFB_ROOT, "pbp", seasons, columns=list(cf._PBP_LOAD_COLUMNS))
    sched = schedules.copy()
    cf.cast_int_columns(sched, ("game_id", "season", "week"))
    regular = sched["season_type"].astype(str).eq("regular")
    fbs = sched["home_division"].astype("string").str.lower().eq("fbs") & sched[
        "away_division"
    ].astype("string").str.lower().eq("fbs")
    sched["kickoff"] = pd.to_datetime(sched["start_date"], utc=True, format="ISO8601")
    sched["gameday"] = sched["kickoff"].dt.tz_localize(None).dt.normalize()
    for c in ("home_id", "away_id"):
        sched[c] = pd.to_numeric(sched[c], errors="coerce")
    sched = sched.loc[regular & sched["home_id"].notna() & sched["away_id"].notna()].copy()
    for c in ("home_id", "away_id"):
        sched[c] = sched[c].astype("int64")
    rest_base = sched[["season", "gameday", "home_id", "away_id"]]
    done = sched["completed"].astype(bool) & fbs.loc[sched.index]
    games = sched.loc[sched["season"].eq(TARGET) & fbs.loc[sched.index]].copy()
    games["neutral_site"] = games["neutral_site"].astype(bool).astype(int)
    games["week_sin"] = np.sin(2.0 * np.pi * games["week"] / cf.CFB_WEEK_PERIOD)
    games["week_cos"] = np.cos(2.0 * np.pi * games["week"] / cf.CFB_WEEK_PERIOD)
    games = cf._add_rest_features(games, rest_base)
    team_games, _ = cf.build_cfb_team_game_metrics(pbp)
    played = sched.loc[done, ["game_id", "gameday"]]
    state_input = team_games.merge(played, on="game_id", how="inner", validate="many_to_one")
    states = cf.build_cfb_team_states(state_input)
    games = cf.attach_cfb_team_states(games, states)
    keep = [
        "game_id", "season", "week", "kickoff", "home_team", "away_team", "home_id", "away_id",
        "completed", "home_points", "away_points", "neutral_site", "rest_diff",
        "week_sin", "week_cos", "home_team_games", "away_team_games",
        "home_off_epa_per_play", "away_off_epa_per_play", "diff_off_epa_per_play",
        "home_def_epa_per_play", "away_def_epa_per_play", "diff_def_epa_per_play",
    ]
    out = games[keep].sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(len(out), "games", out["week"].min(), out["week"].max(), "seasons", seasons)
    print(out.groupby("week")["completed"].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
