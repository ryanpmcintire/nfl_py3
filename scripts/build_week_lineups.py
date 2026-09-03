"""Build the optional, ignored lineup artifact consumed by This Week.

This is deliberately a separate refresh step: GitHub Pages can only serve
static JSON, and the renderer must never reach out to a live roster or injury
provider.  Injury data is optional and is marked unavailable when no current
source is present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import pandas as pd

from nfl_ats.lineup_view import STABLE_LINEUP_PATH
from nfl_ats.public_board import load_public_board_artifacts
from nfl_ats.quarterbacks import write_depth_snapshot


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _team_payload(
    depth: pd.DataFrame, team: str, model_qb_id: str | None, qb_probability: float | None
) -> dict[str, Any]:
    rows = depth[depth["team"] == team].copy()
    # nflverse retains a row for each historical depth-chart update. Keep the
    # complete latest snapshot, including backups, rather than only starters.
    time_column = "observed_at_utc" if "observed_at_utc" in rows else "dt"
    rows["_dt"] = pd.to_datetime(rows[time_column], errors="coerce", utc=True)
    latest = rows["_dt"].max()
    if pd.notna(latest):
        rows = rows[rows["_dt"] == latest]
    rows = rows.drop_duplicates(subset=["pos_abb", "pos_rank", "player_name"], keep="last")
    unit_order = {"offense": 0, "defense": 1, "special_teams": 2}
    position_order = {
        "QB": 0,
        "RB": 1,
        "FB": 2,
        "WR": 3,
        "TE": 4,
        "LT": 5,
        "LG": 6,
        "C": 7,
        "RG": 8,
        "RT": 9,
        "LDE": 0,
        "LDT": 1,
        "NT": 2,
        "RDT": 3,
        "RDE": 4,
        "WLB": 5,
        "LILB": 6,
        "MLB": 7,
        "RILB": 8,
        "SLB": 9,
        "LCB": 10,
        "SS": 11,
        "FS": 12,
        "RCB": 13,
        "NB": 14,
        "PK": 0,
        "P": 1,
        "H": 2,
        "LS": 3,
        "PR": 4,
        "KR": 5,
    }

    def unit(position: str) -> str:
        if position in {"PK", "P", "H", "LS", "PR", "KR"}:
            return "special_teams"
        if position in {"QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"}:
            return "offense"
        return "defense"

    rows["_unit"] = rows["pos_abb"].fillna("").map(lambda value: unit(str(value)))
    rows["_rank"] = pd.to_numeric(rows["pos_rank"], errors="coerce").fillna(99)
    rows["_position_order"] = rows["pos_abb"].map(position_order).fillna(99)
    rows = rows.sort_values(
        ["_unit", "_position_order", "_rank", "player_name"],
        key=lambda values: values.map(unit_order) if values.name == "_unit" else values,
        na_position="last",
    )
    players: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        position = str(row.get("pos_abb") or row.get("pos_name") or "")
        rank = int(row["_rank"]) if row["_rank"] < 99 else 1
        gsis_id = str(row["gsis_id"]) if pd.notna(row.get("gsis_id")) else None
        probability = qb_probability if position == "QB" and gsis_id == model_qb_id else None
        players.append(
            {
                "name": str(row.get("player_name") or "Unknown player"),
                "position": position,
                "slot": f"{position}{rank}",
                "depth": rank,
                "unit": str(row["_unit"]),
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
        "as_of": str(rows[time_column].iloc[0]) if not rows.empty else None,
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
    display_depth = nfl.load_depth_charts(season).to_pandas()
    depth_snapshot = write_depth_snapshot(
        display_depth, Path("data") / "quarterbacks" / "depth" / "raw", [season]
    )
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
                display_depth,
                str(row["home_team"]),
                home_qb,
                _number(row.get("home_qb_start_probability")),
            ),
            "away": _team_payload(
                display_depth,
                str(row["away_team"]),
                away_qb,
                _number(row.get("away_qb_start_probability")),
            ),
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Replacement, not accumulation: the default target is one stable path that
    # every refresh overwrites. An explicit --output still writes exactly there
    # (and skips legacy cleanup, so ad-hoc exports never delete the live file).
    explicit_output = args.output is not None
    output = args.output or args.artifacts_root / STABLE_LINEUP_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "season": season,
                "week": week,
                "generated_at": stamp,
                "model_id": artifacts.active.get("model_id"),
                "forecast_artifact": artifacts.active.get("weekly_forecast", {}).get("artifact"),
                "depth_snapshot": depth_snapshot.snapshot_id,
                "games": games,
            },
            indent=2,
        )
        + "\n"
    )
    staging = output.with_name(f".{output.name}.{stamp}.tmp")
    staging.write_text(payload, encoding="utf-8")
    os.replace(staging, output)
    _check_artifact_size(output)
    if not explicit_output:
        _remove_legacy_stamped_runs(args.artifacts_root / "lineups", keep=output)
    print(output)


#: Fail-closed ceiling for the display artifact: 16 games of ~140 small
#: player dicts should stay well under a megabyte (measured 674 KB,
#: 2026-09-03); a 2026-09-03 stamped run once reached 37 MB and was deleted
#: unexamined under the replacement policy, so the builder now refuses to
#: publish a bloated artifact silently instead of hoping it was a one-off.
MAX_LINEUP_BYTES = 5 * 1024 * 1024


def _check_artifact_size(path: Path, *, limit: int = MAX_LINEUP_BYTES) -> None:
    size = path.stat().st_size
    if size > limit:
        raise SystemExit(
            f"Refusing to publish {path} at {size} bytes (limit {limit}): "
            "the lineup artifact should stay near a megabyte; inspect the "
            "payload before overriding this guard."
        )


def _remove_legacy_stamped_runs(lineups_root: Path, *, keep: Path) -> None:
    """Delete pre-replacement-policy stamped `*/lineups.json` runs.

    Each stamped run is a ~37 MB display copy superseded by the stable path;
    provenance (model, forecast, depth snapshot) lives inside the payload, and
    the underlying depth snapshots remain in `data/quarterbacks/depth/raw`.
    Only directories directly under the lineups root holding a `lineups.json`
    are touched; anything else is left alone.
    """
    if not lineups_root.is_dir():
        return
    for child in sorted(lineups_root.iterdir()):
        if not child.is_dir() or child.name == keep.parent.name:
            continue
        artifact = child / "lineups.json"
        if not artifact.is_file():
            continue
        shutil.rmtree(child, ignore_errors=True)


if __name__ == "__main__":
    main()
