"""Build the optional, ignored lineup artifact consumed by This Week.

This is deliberately a separate refresh step: GitHub Pages can only serve
static JSON, and the renderer must never reach out to a live roster or injury
provider.  Injury data is optional and is marked unavailable when no current
source is present.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import pandas as pd

from nfl_ats.public_board import load_public_board_artifacts


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _team_payload(
    depth: pd.DataFrame, team: str, model_qb_id: str | None, qb_probability: float | None
) -> dict[str, Any]:
    rows = depth[(depth["team"] == team) & (depth["pos_rank"] == 1)].copy()
    # nflverse retains a row for each historical depth-chart update.  The
    # public card needs the latest snapshot and one player per slot.
    if "dt" in rows:
        rows["_dt"] = pd.to_datetime(rows["dt"], errors="coerce", utc=True)
    rows = rows.sort_values("_dt", ascending=False, na_position="last")
    rows = rows.drop_duplicates(subset=["pos_grp_id", "pos_slot"], keep="first")
    rows = rows.sort_values(["pos_grp_id", "pos_slot", "player_name"], na_position="last")
    players: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        position = str(row.get("pos_abb") or row.get("pos_name") or "")
        gsis_id = str(row["gsis_id"]) if pd.notna(row.get("gsis_id")) else None
        probability = qb_probability if position == "QB" and gsis_id == model_qb_id else None
        players.append(
            {
                "name": str(row.get("player_name") or "Unknown player"),
                "position": position,
                "slot": str(row.get("pos_slot") or position),
                "depth": 1,
                "gsis_id": gsis_id,
                "play_probability": probability,
                "model_role": "base_model" if gsis_id == model_qb_id else "context_only",
            }
        )
    current_qb = next((player for player in players if player["position"] == "QB"), None)
    note = None
    if model_qb_id and (current_qb is None or current_qb["gsis_id"] != model_qb_id):
        note = (
            "Current depth chart QB differs from forecast input; rerun forecast before treating "
            "this as a model update."
        )
    return {
        "team": team,
        "players": players,
        "as_of": str(rows["dt"].iloc[0]) if not rows.empty else None,
        "source": "nflverse depth charts",
        "injury_status": "unavailable — current injury feed not attached",
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = load_public_board_artifacts(args.artifacts_root)
    season = int(artifacts.metadata.get("season", args.season))
    week = int(artifacts.metadata.get("week", 1))
    depth = nfl.load_depth_charts(season).to_pandas()
    games: dict[str, Any] = {}
    for _, row in artifacts.predictions.iterrows():
        game_id = str(row["game_id"])
        home_qb = (
            str(row["home_projected_qb_id"]) if pd.notna(row.get("home_projected_qb_id")) else None
        )
        away_qb = (
            str(row["away_projected_qb_id"]) if pd.notna(row.get("away_projected_qb_id")) else None
        )
        games[game_id] = {
            "home": _team_payload(
                depth, str(row["home_team"]), home_qb, _number(row.get("home_qb_start_probability"))
            ),
            "away": _team_payload(
                depth, str(row["away_team"]), away_qb, _number(row.get("away_qb_start_probability"))
            ),
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output
        or args.artifacts_root / "lineups" / f"{season}-week-{week}-{stamp}" / "lineups.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"season": season, "week": week, "generated_at": stamp, "games": games}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
