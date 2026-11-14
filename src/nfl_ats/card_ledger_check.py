from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.board_content import load_board_content
from nfl_ats.clv import current_played_card_view, load_paper_decisions
from nfl_ats.pick_refresh import (
    load_pick_revisions,
    pick_deadline,
    served_best_pick,
    sunday_pick_lock,
)
from nfl_ats.publishing import BEST_PICK_MARK

DEFAULT_CARD_PATH = Path("CURRENT_PREDICTIONS.md")
CARD_CELL_SEPARATOR = "|"
CARD_MATCHUP_CELL = 2
CARD_PICK_CELL = 3
BEST_PICK_FIELD = "best_pick"
BOARD_STAR_SURFACE = "best_pick_ledger_vs_board"
CARD_FILE_STAR_SURFACE = "best_pick_ledger_vs_card_file"


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


def _flagged_best_pick(paper_week: pd.DataFrame) -> str:
    flagged = paper_week.loc[paper_week["is_best_pick"].fillna(False).astype(bool), "game_id"]
    return str(flagged.iloc[0]) if len(flagged) == 1 else ""


def _card_file_best_pick(card_path: Path, game_id_by_matchup: dict[str, str]) -> str:
    mark = BEST_PICK_MARK.strip()
    starred: list[str] = []
    for line in card_path.read_text(encoding="utf-8").split("\n"):
        if not line.startswith(CARD_CELL_SEPARATOR):
            continue
        cells = line.split(CARD_CELL_SEPARATOR)
        if len(cells) <= CARD_PICK_CELL:
            continue
        game_id = game_id_by_matchup.get(cells[CARD_MATCHUP_CELL].strip())
        if game_id is None or not cells[CARD_PICK_CELL].strip().startswith(mark):
            continue
        if game_id not in starred:
            starred.append(game_id)
    return ", ".join(starred)


def check_card_ledger_consistency(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    card_path: Path | None = DEFAULT_CARD_PATH,
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
            "ok": False,
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

    revisions = load_pick_revisions(artifacts_root)
    revisions_week = revisions.loc[
        revisions["season"].astype(int).eq(season) & revisions["week"].astype(int).eq(week)
    ]
    if not revisions_week.empty:
        recorded_at = pd.to_datetime(
            revisions_week["revision_recorded_at_utc"], utc=True, errors="coerce"
        )
        kickoff_by_game = dict(zip(card["game_id"].astype(str), played.kickoffs, strict=True))
        revision_kickoffs = revisions_week["game_id"].astype(str).map(kickoff_by_game)
        lock = sunday_pick_lock(played.kickoffs)
        deadlines = revision_kickoffs.map(
            lambda kickoff: pick_deadline(kickoff, lock) if pd.notna(kickoff) else pd.NaT
        )
        revisions_week = revisions_week.loc[
            recorded_at.le(checked_at) & recorded_at.lt(deadlines)
        ].copy()
        revisions_week["revision_recorded_at_utc"] = recorded_at.loc[revisions_week.index]
    latest_revision_side = _latest_revision_side_by_game(revisions_week)

    paper_flag = _flagged_best_pick(paper_week)
    revision_star = served_best_pick(artifacts_root, season=season, week=week) or ""
    ledger_star = revision_star or paper_flag
    ledger_star_source = (
        "pick_revision_ledger" if revision_star else "paper_ledger_flag" if paper_flag else ""
    )
    star_compared = ledger_star in pre_kickoff_ids

    disagreements: list[dict[str, Any]] = []
    for _, row in paper_week.iterrows():
        game_id = str(row["game_id"])
        if game_id not in pre_kickoff_ids:
            continue
        served_side = served_pick_side.get(game_id)
        if served_side is None:
            continue
        ledger_side = latest_revision_side.get(game_id, str(row["pick_side"]))
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
        if ledger_flip != served_flip and served_side != ledger_side:
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

    board_side_by_game: dict[str, str] = {}
    board_star: str | None = None
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
        board_star = next((game.game_id for game in board.games if game.is_best), "")
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

    card_star: str | None = None
    card_file_error = ""
    if card_path is not None and card_path.is_file():
        try:
            card_star = _card_file_best_pick(
                card_path, {matchup: game_id for game_id, matchup in served_matchup.items()}
            )
        except (OSError, ValueError) as error:
            card_file_error = f"{type(error).__name__}: {error}"

    if star_compared:
        for surface, surface_star in (
            (BOARD_STAR_SURFACE, board_star),
            (CARD_FILE_STAR_SURFACE, card_star),
        ):
            if surface_star is None or surface_star == ledger_star:
                continue
            disagreements.append(
                {
                    "game_id": ledger_star,
                    "matchup": served_matchup.get(ledger_star, ledger_star),
                    "surface": surface,
                    "field": BEST_PICK_FIELD,
                    "ledger_value": ledger_star,
                    "served_value": surface_star,
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
        "card_file_error": card_file_error,
        "best_pick": {
            "ledger": ledger_star,
            "ledger_source": ledger_star_source,
            "board": board_star,
            "card_file": card_star,
            "paper_ledger_flag": paper_flag,
            "tuesday_card_view": str(played.view.nomination.active_game_id or ""),
            "compared": star_compared,
        },
        "evaluated": True,
        "ok": len(disagreements) == 0 and not board_content_error and not card_file_error,
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
    stars = report.get("best_pick")
    if stars:
        lines.append(
            f"  best pick: ledger={stars['ledger']!r} ({stars['ledger_source']})"
            f"   board={stars['board']!r}   card file={stars['card_file']!r}"
            f"   compared: {stars['compared']}"
        )
        lines.append(
            f"  Tuesday nominee, not compared: paper flag={stars['paper_ledger_flag']!r}"
            f"   served card view={stars['tuesday_card_view']!r}"
        )
    if report.get("board_content_error"):
        lines.append(f"  board content could not be loaded: {report['board_content_error']}")
    if report.get("card_file_error"):
        lines.append(f"  card file could not be read: {report['card_file_error']}")
    disagreements = report.get("disagreements") or []
    if not disagreements:
        if report.get("board_content_error"):
            lines.append("  consistency check incomplete: board comparison unavailable")
            return "\n".join(lines)
        if report.get("card_file_error"):
            lines.append("  consistency check incomplete: card file comparison unavailable")
            return "\n".join(lines)
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
