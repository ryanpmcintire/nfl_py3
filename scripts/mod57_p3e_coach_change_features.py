from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("HC", "OC", "DC")
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")


def latest_history() -> Path:
    files = sorted(glob.glob(str(ROOT / "data/raw/coordinators/*/coordinator_history.parquet")))
    return Path(files[-1])


def tidy(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    h["team"] = h.team.replace(TEAM_ALIASES)
    h["person"] = h.person.astype(str).map(lambda s: PARENTHETICAL.sub("", s).strip())
    h["at"] = pd.to_datetime(h.effective_observed_at, utc=True)
    return h.sort_values(["season", "team", "role", "at"]).reset_index(drop=True)


def holder_before(group: pd.DataFrame, instant) -> str | None:
    seen = group[group["at"] < instant]
    return None if seen.empty else seen.iloc[-1].person


def build(games: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    h = tidy(history)
    games = games.copy()
    for side in ("home", "away"):
        games[f"{side}_team"] = games[f"{side}_team"].replace(TEAM_ALIASES)
    season_end = games.groupby("season").kickoff.max()
    groups = {k: g for k, g in h.groupby(["season", "team", "role"])}  # noqa: C416
    unresolved = {"no_prior_holder": 0, "no_holder_before_kickoff": 0}
    out = {f"{side}_new_{role.lower()}": [] for side in ("home", "away") for role in ROLES}
    for row in games.itertuples():
        for side in ("home", "away"):
            team = getattr(row, f"{side}_team")
            for role in ROLES:
                prior = groups.get((row.season - 1, team, role))
                now = groups.get((row.season, team, role))
                value = np.nan
                if prior is None or now is None:
                    unresolved["no_prior_holder"] += 1
                else:
                    before = holder_before(prior, season_end.get(row.season - 1))
                    current = holder_before(now, row.kickoff)
                    if before is None:
                        unresolved["no_prior_holder"] += 1
                    elif current is None:
                        unresolved["no_holder_before_kickoff"] += 1
                    else:
                        value = float(current != before)
                out[f"{side}_new_{role.lower()}"].append(value)
    frame = games[["game_id", "season", "week"]].copy()
    for role in ROLES:
        r = role.lower()
        frame[f"coach_new_{r}_diff"] = pd.Series(out[f"home_new_{r}"], index=frame.index).fillna(
            0.0
        ) - pd.Series(out[f"away_new_{r}"], index=frame.index).fillna(0.0)
    frame["coach_new_sum_diff"] = frame[[f"coach_new_{r.lower()}_diff" for r in ROLES]].sum(axis=1)
    print({"unresolved_slots": unresolved, "games": len(frame)}, flush=True)
    return frame


def main() -> None:
    p = argparse.ArgumentParser(
        description="Home-minus-away new HC/OC/DC counts as of each game's kickoff"
    )
    p.add_argument("--out", default=str(ROOT / "artifacts/mod57_p3e/coach_change_features.parquet"))
    args = p.parse_args()
    games = pd.read_parquet(
        ROOT / "data/processed/game_features.parquet",
        columns=["game_id", "season", "week", "home_team", "away_team", "kickoff", "game_type"],
    )
    games = games[games.game_type == "REG"].drop(columns="game_type")
    games["kickoff"] = pd.to_datetime(games.kickoff, utc=True)
    history = pd.read_parquet(latest_history())
    frame = build(games.reset_index(drop=True), history)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    cols = [c for c in frame.columns if c.startswith("coach_")]
    early = frame[frame.week <= 6]
    print(
        {
            "rows": len(frame),
            "nonzero_share_weeks_1_6": {c: float((early[c] != 0).mean()) for c in cols},
            "nonzero_games_by_season_sum": {
                int(s): int((g.coach_new_sum_diff != 0).sum()) for s, g in early.groupby("season")
            },
        }
    )


if __name__ == "__main__":
    main()
