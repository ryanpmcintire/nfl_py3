from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
SCHEDULE_ROOT = REPO / "data" / "cfb" / "schedules" / "raw"
CFB_FEATURES = REPO / "data" / "processed" / "cfb_game_features.parquet"
POLICIES = REPO / "config" / "source_policies.json"
OUT_DIR = REPO / "data" / "external" / "outside_ratings"
ELO_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "home_team",
    "away_team",
    "home_division",
    "away_division",
    "home_pregame_elo",
    "away_pregame_elo",
]
NETWORK_SOURCES = {
    "espn_summary_predictor": ["espn_scoreboard_pickcenter", "espn_nfl_injuries"],
    "fivethirtyeight_nfl_elo": ["fivethirtyeight_nfl_elo"],
    "nfelo_github": ["github_nfelo_public_ratings"],
}
ESPN_BLOCK_FILE = REPO / "data" / "market" / "espn_pickcenter_blocked.json"


def latest_season_files() -> dict[int, Path]:
    chosen: dict[int, Path] = {}
    for snapshot in sorted(SCHEDULE_ROOT.iterdir()):
        for part in snapshot.glob("season=*/schedules.parquet"):
            chosen[int(part.parent.name.split("=")[1])] = part
    return chosen


def cfb_elo(first: int, last: int) -> tuple[pd.DataFrame, list[dict]]:
    files = latest_season_files()
    frames = []
    for season in range(first, last + 1):
        if season not in files:
            continue
        names = set(pq.read_schema(files[season]).names)
        cols = [c for c in ELO_COLUMNS if c in names]
        frame = pd.read_parquet(files[season], columns=cols)
        frame["source_snapshot"] = files[season].parent.parent.name
        frames.append(frame)
    elo = pd.concat(frames, ignore_index=True)
    elo["game_id"] = elo["game_id"].astype(str)
    repo_ids = pd.read_parquet(CFB_FEATURES, columns=["game_id", "season"])
    repo_ids["game_id"] = repo_ids["game_id"].astype(str)
    both = elo["home_pregame_elo"].notna() & elo["away_pregame_elo"].notna()
    elo["has_both_elo"] = both
    rated_ids = set(elo.loc[both, "game_id"])
    rows = []
    for season in range(first, last + 1):
        sub = elo[elo["season"] == season]
        fbs = (
            sub[(sub.get("home_division") == "fbs") | (sub.get("away_division") == "fbs")]
            if "home_division" in sub
            else sub
        )
        repo_season = repo_ids[repo_ids["season"] == season]
        joined = repo_season["game_id"].isin(rated_ids).sum()
        rows.append(
            {
                "season": season,
                "schedule_games": len(sub),
                "fbs_involved_games": len(fbs),
                "fbs_games_with_both_pregame_elo": int(fbs["has_both_elo"].sum())
                if len(fbs)
                else 0,
                "repo_feature_games": len(repo_season),
                "repo_games_joined_with_both_elo": int(joined),
                "join_rate": round(float(joined / len(repo_season)), 4)
                if len(repo_season)
                else None,
            }
        )
    return elo, rows


def network_gate() -> dict:
    policies = json.loads(POLICIES.read_text(encoding="utf-8"))["sources"]
    verdict = {}
    for name, ids in NETWORK_SOURCES.items():
        entries = {
            i: {
                "present": i in policies,
                "acquisition_allowed": policies.get(i, {}).get("acquisition_allowed"),
                "risk": policies.get(i, {}).get("risk"),
            }
            for i in ids
        }
        allowed = all(e["present"] and e["acquisition_allowed"] for e in entries.values())
        if name == "espn_summary_predictor" and ESPN_BLOCK_FILE.exists():
            allowed = False
            entries["espn_pickcenter_blocked_file"] = {"present": True}
        verdict[name] = {"policy_entries": entries, "fetch_permitted": allowed, "requests_made": 0}
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MOD-30 outside-ratings acquisition inventory; reads no outcome columns "
            "and makes no network request unless a source policy permits it"
        )
    )
    parser.add_argument("--first-season", type=int, default=2012)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ".gitignore").write_text("*\n", encoding="utf-8")
    elo, coverage = cfb_elo(args.first_season, args.last_season)
    elo.to_parquet(OUT_DIR / "cfb_pregame_elo.parquet", index=False)
    inventory = {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "cfb_pregame_elo": {"columns_read": ELO_COLUMNS, "coverage": coverage},
        "network_sources": network_gate(),
    }
    (OUT_DIR / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
