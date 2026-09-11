from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.market_data_halves import SNAPSHOT_SUFFIX
from nfl_ats.pick_refresh import RefreshResult, original_card

CHALLENGER_ID = "half_line_2h_underdog_refresh_v1"

FULL_GAME_MARKET = "spreads"
SECOND_HALF_MARKET = "spreads_h2"

OVERLAY_STATUS_FIRED = "half_line_2h_underdog_fired"
OVERLAY_STATUS_NO_DISAGREEMENT = "full_and_2h_favorite_agree"
OVERLAY_STATUS_NO_BULK_SNAPSHOT = "no_bulk_snapshot_before_pass"
OVERLAY_STATUS_NO_HALVES_SNAPSHOT = "no_halves_snapshot_before_pass"
OVERLAY_STATUS_NO_MATCHED_QUOTE = "game_absent_from_matched_quotes"
OVERLAY_STATUS_PICKEM = "full_or_2h_line_is_pickem"

_BULK_SNAPSHOT_NAME_RE = re.compile(r"^\d{8}T\d{6}Z$")

HALF_LINE_REFRESH_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "decision_home_spread",
    "played_pick_side",
    "bulk_snapshot_id",
    "bulk_observed_at_utc",
    "halves_snapshot_id",
    "halves_observed_at_utc",
    "matched_book_count",
    "matched_books",
    "full_game_home_spread",
    "half2_home_spread",
    "full_game_favorite_side",
    "half2_favorite_side",
    "disagreement_fired",
    "half_line_would_be_pick_side",
    "overlay_status",
    "model_id",
    "feature_table_sha256",
)


def half_line_refresh_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "half_line_refresh_decisions.parquet"


def load_half_line_refresh_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = half_line_refresh_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(HALF_LINE_REFRESH_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(HALF_LINE_REFRESH_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Second-half-line refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(HALF_LINE_REFRESH_COLUMNS)]


@dataclass(frozen=True)
class MarketSnapshotRef:
    snapshot_id: str
    root: Path
    observed_at_utc: pd.Timestamp
    quotes: pd.DataFrame


def freshest_snapshot_before(
    market_root: Path, *, before: pd.Timestamp, suffix: str
) -> MarketSnapshotRef | None:

    if not market_root.is_dir():
        return None
    before_utc = pd.Timestamp(before)
    before_utc = (
        before_utc.tz_localize("UTC") if before_utc.tzinfo is None else before_utc.tz_convert("UTC")
    )
    best_name: str | None = None
    best_root: Path | None = None
    best_observed: pd.Timestamp | None = None
    for child in market_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if suffix:
            if not name.endswith(suffix):
                continue
        elif not _BULK_SNAPSHOT_NAME_RE.match(name):
            continue
        manifest_path = child / "manifest.json"
        quotes_path = child / "quotes.parquet"
        if not manifest_path.is_file() or not quotes_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        observed_raw = manifest.get("observed_at_utc")
        if observed_raw is None:
            continue
        observed = pd.Timestamp(observed_raw)
        observed = (
            observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
        )
        if observed >= before_utc:
            continue
        if best_observed is None or observed > best_observed:
            best_observed = observed
            best_name = name
            best_root = child
    if best_root is None or best_name is None or best_observed is None:
        return None
    quotes = pd.read_parquet(best_root / "quotes.parquet")
    return MarketSnapshotRef(
        snapshot_id=best_name, root=best_root, observed_at_utc=best_observed, quotes=quotes
    )


def _favorite_side(home_line: float) -> str:
    if home_line < 0:
        return "HOME"
    if home_line > 0:
        return "AWAY"
    return "PICK"


def matched_book_disagreement(
    bulk_quotes: pd.DataFrame, halves_quotes: pd.DataFrame
) -> dict[str, dict[str, Any]]:

    full = bulk_quotes.loc[
        bulk_quotes["market"].eq(FULL_GAME_MARKET)
        & bulk_quotes["outcome_side"].eq("HOME")
        & bulk_quotes["nflverse_game_id"].notna()
    ].copy()
    full["nflverse_game_id"] = full["nflverse_game_id"].astype(str)
    full = full.loc[:, ["nflverse_game_id", "bookmaker_key", "line"]].rename(
        columns={"line": "full_line"}
    )

    half = halves_quotes.loc[
        halves_quotes["market"].eq(SECOND_HALF_MARKET)
        & halves_quotes["outcome_side"].eq("HOME")
        & halves_quotes["nflverse_game_id"].notna()
    ].copy()
    half["nflverse_game_id"] = half["nflverse_game_id"].astype(str)
    half = half.loc[:, ["nflverse_game_id", "bookmaker_key", "line"]].rename(
        columns={"line": "half2_line"}
    )

    matched = full.merge(half, on=["nflverse_game_id", "bookmaker_key"], how="inner")
    matched = matched.dropna(subset=["full_line", "half2_line"])

    result: dict[str, dict[str, Any]] = {}
    for game_id, group in matched.groupby("nflverse_game_id"):
        full_median = float(group["full_line"].median())
        half2_median = float(group["half2_line"].median())
        result[str(game_id)] = {
            "full_line": full_median,
            "half2_line": half2_median,
            "matched_book_count": int(group["bookmaker_key"].nunique()),
            "matched_books": ",".join(sorted(set(group["bookmaker_key"].astype(str)))),
            "full_favorite_side": _favorite_side(full_median),
            "half2_favorite_side": _favorite_side(half2_median),
        }
    return result


def build_half_line_refresh_rows(
    plan: RefreshResult, *, market_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:

    empty = pd.DataFrame(columns=list(HALF_LINE_REFRESH_COLUMNS))
    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}

    pass_instant = pd.Timestamp(plan.computed_at_utc)
    bulk = freshest_snapshot_before(market_root, before=pass_instant, suffix="")
    if bulk is None:
        return empty, {"skipped": True, "reason": OVERLAY_STATUS_NO_BULK_SNAPSHOT}
    halves = freshest_snapshot_before(market_root, before=pass_instant, suffix=SNAPSHOT_SUFFIX)
    if halves is None:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_HALVES_SNAPSHOT,
            "bulk_snapshot_id": bulk.snapshot_id,
        }

    per_game = matched_book_disagreement(bulk.quotes, halves.quotes)

    rows: list[dict[str, Any]] = []
    fired_game_ids: list[str] = []
    for game in eligible_games:
        game_id = str(game.game_id)
        info = per_game.get(game_id)
        if info is None:
            status = OVERLAY_STATUS_NO_MATCHED_QUOTE
            fired = False
            challenger_side = game.new_pick_side
            full_line: float | None = None
            half2_line: float | None = None
            full_fav = "UNKNOWN"
            half2_fav = "UNKNOWN"
            matched_books = ""
            matched_count = 0
        else:
            full_line = info["full_line"]
            half2_line = info["half2_line"]
            matched_count = info["matched_book_count"]
            matched_books = info["matched_books"]
            full_fav = info["full_favorite_side"]
            half2_fav = info["half2_favorite_side"]
            if full_fav == "PICK" or half2_fav == "PICK":
                status = OVERLAY_STATUS_PICKEM
                fired = False
                challenger_side = game.new_pick_side
            elif full_fav != half2_fav:
                status = OVERLAY_STATUS_FIRED
                fired = True
                challenger_side = "AWAY" if full_fav == "HOME" else "HOME"
            else:
                status = OVERLAY_STATUS_NO_DISAGREEMENT
                fired = False
                challenger_side = game.new_pick_side
        if fired:
            fired_game_ids.append(game_id)
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": game.new_pick_side,
                "bulk_snapshot_id": bulk.snapshot_id,
                "bulk_observed_at_utc": bulk.observed_at_utc,
                "halves_snapshot_id": halves.snapshot_id,
                "halves_observed_at_utc": halves.observed_at_utc,
                "matched_book_count": matched_count,
                "matched_books": matched_books,
                "full_game_home_spread": full_line,
                "half2_home_spread": half2_line,
                "full_game_favorite_side": full_fav,
                "half2_favorite_side": half2_fav,
                "disagreement_fired": bool(fired),
                "half_line_would_be_pick_side": challenger_side,
                "overlay_status": status,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(HALF_LINE_REFRESH_COLUMNS))
    diagnostics = {
        "skipped": False,
        "bulk_snapshot_id": bulk.snapshot_id,
        "bulk_observed_at_utc": bulk.observed_at_utc,
        "halves_snapshot_id": halves.snapshot_id,
        "halves_observed_at_utc": halves.observed_at_utc,
        "games_considered": len(frame),
        "games_with_matched_quote": int((frame["matched_book_count"] > 0).sum()),
        "fired_game_ids": fired_game_ids,
        "status_counts": frame["overlay_status"].value_counts().to_dict(),
    }
    return frame, diagnostics


def record_half_line_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:

    market_root = data_root / "market" / "raw"
    rows, diagnostics = build_half_line_refresh_rows(plan, market_root=market_root)

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's would-be picks to the "
                "second-half-line refresh ledger; this preview computed the flags "
                "read-only and wrote nothing"
            ),
            **diagnostics,
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="half-line-refresh-overlay"
    )

    existing = load_half_line_refresh_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(HALF_LINE_REFRESH_COLUMNS)],
        half_line_refresh_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }


__all__ = [
    "CHALLENGER_ID",
    "FULL_GAME_MARKET",
    "HALF_LINE_REFRESH_COLUMNS",
    "SECOND_HALF_MARKET",
    "MarketSnapshotRef",
    "build_half_line_refresh_rows",
    "freshest_snapshot_before",
    "half_line_refresh_ledger_path",
    "load_half_line_refresh_decisions",
    "matched_book_disagreement",
    "record_half_line_refresh_overlay",
]
