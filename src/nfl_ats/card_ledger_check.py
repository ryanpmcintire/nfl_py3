from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.board_content import load_board_content
from nfl_ats.clv import current_played_card_view, load_paper_decisions
from nfl_ats.pick_refresh import load_pick_revisions


def _matchup(away: str, home: str) -> str:
    return f"{away} at {home}"


def _latest_revision_side_by_game(revisions: pd.DataFrame) -> dict[str, str]:
    if revisions.empty or "game_id" not in revisions.columns:
        return {}
    if "revision_recorded_at_utc" not in revisions.columns:
        return {}
    ordered = revisions.sort_values("revision_recorded_at_utc")
    latest_side: dict[str, str] = {}
    for _, row in ordered.iterrows():
        side = str(row.get("new_pick_side") or "")
        latest_side[str(row["game_id"])] = side if side in ("HOME", "AWAY") else ""
    return latest_side


def check_card_ledger_consistency(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:

    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    checked_at = instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")

    try:
        played = current_played_card_view(
            artifacts_root, data_root=data_root, now=checked_at.to_pydatetime()
        )
    except Exception as error:
        return {
            "checked_at_utc": checked_at.isoformat(),
            "evaluated": False,
            "ok": True,
            "reason": (
                f"could not resolve the current served card: {type(error).__name__}: {error}"
            ),
            "disagreements": [],
        }

    season, week = played.season, played.week
    card = played.card
    served_pick_side = dict(
        zip(card["game_id"].astype(str), played.final_pick_side.astype(str), strict=False)
    )
    served_matchup = {
        str(game_id): _matchup(str(away), str(home))
        for game_id, away, home in zip(
            card["game_id"].astype(str), card["away_team"], card["home_team"], strict=False
        )
    }
    pre_kickoff_ids = set(card.loc[played.kickoffs.gt(checked_at), "game_id"].astype(str))

    paper = load_paper_decisions(artifacts_root)
    paper_week = paper.loc[
        paper["season"].astype(int).eq(season) & paper["week"].astype(int).eq(week)
    ]

    disagreements: list[dict[str, Any]] = []
    for _, row in paper_week.iterrows():
        game_id = str(row["game_id"])
        if game_id not in pre_kickoff_ids:
            continue
        served_side = served_pick_side.get(game_id)
        if served_side is None:
            continue
        ledger_side = str(row["pick_side"])
        if served_side != ledger_side:
            disagreements.append(
                {
                    "game_id": game_id,
                    "matchup": served_matchup.get(game_id, game_id),
                    "surface": "paper_ledger_vs_served_card",
                    "field": "pick_side",
                    "ledger_value": ledger_side,
                    "served_value": served_side,
                }
            )
        ledger_flip = bool(row["composed_overlay_flip"])
        served_flip = game_id in played.composed_flip_ids
        if ledger_flip != served_flip:
            disagreements.append(
                {
                    "game_id": game_id,
                    "matchup": served_matchup.get(game_id, game_id),
                    "surface": "paper_ledger_vs_served_card",
                    "field": "composed_overlay_flip",
                    "ledger_value": str(ledger_flip),
                    "served_value": str(served_flip),
                }
            )

    revisions = load_pick_revisions(artifacts_root)
    revisions_week = revisions.loc[
        revisions["season"].astype(int).eq(season) & revisions["week"].astype(int).eq(week)
    ]
    latest_revision_side = _latest_revision_side_by_game(revisions_week)

    board_side_by_game: dict[str, str] = {}
    board_content_error = ""
    try:
        board = load_board_content(
            artifacts_root,
            data_root=data_root,
            generated_at=checked_at.to_pydatetime(),
            require_fresh_arrest_overlay=True,
        )
        board_side_by_game = {
            game.game_id: ("HOME" if game.pick_team == game.home else "AWAY")
            for game in board.games
        }
    except Exception as error:
        board_content_error = f"{type(error).__name__}: {error}"

    pick_revision_rows_checked = 0
    for game_id in sorted(pre_kickoff_ids):
        revision_side = latest_revision_side.get(game_id, "")
        if not revision_side:
            continue
        pick_revision_rows_checked += 1
        board_side = board_side_by_game.get(game_id)
        if board_side is not None and board_side != revision_side:
            disagreements.append(
                {
                    "game_id": game_id,
                    "matchup": served_matchup.get(game_id, game_id),
                    "surface": "pick_revision_ledger_vs_board",
                    "field": "pick_side",
                    "ledger_value": revision_side,
                    "served_value": board_side,
                }
            )

    return {
        "checked_at_utc": checked_at.isoformat(),
        "season": season,
        "week": week,
        "model_id": played.model_id,
        "paper_ledger_rows_checked": len(paper_week),
        "pick_revision_rows_checked": pick_revision_rows_checked,
        "board_content_error": board_content_error,
        "evaluated": True,
        "ok": len(disagreements) == 0,
        "disagreements": disagreements,
    }


def render_report(report: dict[str, Any]) -> str:

    lines = [f"card/ledger consistency check  {report.get('season')} week {report.get('week')}"]
    if not report.get("evaluated", True):
        lines.append(f"  could not evaluate: {report.get('reason', '')}")
        return "\n".join(lines)
    lines.append(
        f"  paper ledger rows checked: {report['paper_ledger_rows_checked']}"
        f"   pick-revision rows checked: {report['pick_revision_rows_checked']}"
    )
    if report.get("board_content_error"):
        lines.append(f"  board content could not be loaded: {report['board_content_error']}")
    disagreements = report.get("disagreements") or []
    if not disagreements:
        lines.append(
            "  no disagreements: paper ledger, pick-revision ledger and the served card agree"
        )
        return "\n".join(lines)
    lines.append(f"  {len(disagreements)} disagreement(s):")
    width = max((len(str(row["matchup"])) for row in disagreements), default=10)
    for row in disagreements:
        lines.append(
            f"    !! {row['matchup']:<{width}}  [{row['surface']}] {row['field']}: "
            f"ledger={row['ledger_value']!r} served={row['served_value']!r}"
        )
    return "\n".join(lines)
