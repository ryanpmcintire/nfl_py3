from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

PUBLISHED_PICKS_FILENAME = "published_picks.parquet"
PUBLISHED_PICKS_COLUMNS: tuple[str, ...] = (
    "published_at_utc",
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "kickoff",
    "pick_deadline_utc",
    "pick_team",
    "market_spread",
    "displayed_score",
    "strength_word",
    "source",
)
SOURCE_SITE_PUBLISH = "site_publish"
SCORE_PLACES = 4


@dataclass(frozen=True)
class FrozenPick:
    game_id: str
    pick_team: str
    market_spread: float
    displayed_score: float
    strength_word: str
    published_at_utc: pd.Timestamp
    pick_deadline_utc: pd.Timestamp


def published_picks_path(artifacts_root: Path) -> Path:
    return artifacts_root / "clv_ledger" / PUBLISHED_PICKS_FILENAME


def load_published_picks(artifacts_root: Path) -> pd.DataFrame:
    path = published_picks_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(PUBLISHED_PICKS_COLUMNS))
    frame = pd.read_parquet(path)
    for column in PUBLISHED_PICKS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in ("published_at_utc", "kickoff", "pick_deadline_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame[list(PUBLISHED_PICKS_COLUMNS)]


def game_deadlines(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:

    if frame.empty or not {"game_id", "kickoff"}.issubset(frame.columns):
        return {}
    kickoffs = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce")
    try:
        sunday_lock = sunday_pick_lock(kickoffs)
    except ValueError:
        return {}
    deadlines: dict[str, pd.Timestamp] = {}
    valid = kickoffs.notna()
    for game_id, kickoff in zip(
        frame.loc[valid, "game_id"].astype(str), kickoffs.loc[valid], strict=True
    ):
        deadlines[game_id] = pick_deadline(pd.Timestamp(kickoff), sunday_lock)
    return deadlines


def _same_state(previous: pd.Series, row: dict[str, Any]) -> bool:
    try:
        previous_score = round(float(previous["displayed_score"]), SCORE_PLACES)
        row_score = round(float(row["displayed_score"]), SCORE_PLACES)
    except (TypeError, ValueError):
        return False
    return (
        str(previous["pick_team"]) == str(row["pick_team"])
        and float(previous["market_spread"]) == float(row["market_spread"])
        and previous_score == row_score
        and str(previous["strength_word"] or "") == str(row["strength_word"] or "")
    )


def record_published_picks(
    artifacts_root: Path,
    rows: list[dict[str, Any]],
    *,
    published_at: datetime,
    source: str = SOURCE_SITE_PUBLISH,
) -> int:

    instant = pd.Timestamp(published_at.astimezone(UTC))
    existing = load_published_picks(artifacts_root)
    latest_by_game = (
        existing.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")
        if not existing.empty
        else pd.DataFrame()
    )
    appended: list[dict[str, Any]] = []
    for row in rows:
        deadline = pd.Timestamp(row["pick_deadline_utc"])
        if deadline <= instant:
            continue
        game_id = str(row["game_id"])
        if game_id in latest_by_game.index:
            previous = latest_by_game.loc[game_id]
            if isinstance(previous, pd.DataFrame):
                previous = previous.iloc[-1]
            if _same_state(previous, row):
                continue
        appended.append(
            {
                **{column: row.get(column) for column in PUBLISHED_PICKS_COLUMNS},
                "published_at_utc": instant,
                "source": source,
            }
        )
    if not appended:
        return 0
    new_rows = pd.DataFrame(appended, columns=list(PUBLISHED_PICKS_COLUMNS))
    combined = (
        pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
    )
    path = published_picks_path(artifacts_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(appended)


def frozen_picks(
    artifacts_root: Path,
    *,
    now: datetime,
    season: int | None = None,
    week: int | None = None,
    include_open: bool = False,
) -> dict[str, FrozenPick]:

    frame = load_published_picks(artifacts_root)
    if frame.empty:
        return {}
    if season is not None:
        frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(int(season))]
    if week is not None:
        frame = frame.loc[pd.to_numeric(frame["week"], errors="coerce").eq(int(week))]
    instant = pd.Timestamp(now.astimezone(UTC))
    frozen: dict[str, FrozenPick] = {}
    for game_id, group in frame.groupby("game_id"):
        deadline = group["pick_deadline_utc"].max()
        if pd.isna(deadline):
            continue
        if deadline > instant:
            if not include_open:
                continue
            eligible = group.sort_values("published_at_utc")
        else:
            eligible = group.loc[group["published_at_utc"].le(deadline)].sort_values(
                "published_at_utc"
            )
        if eligible.empty:
            continue
        last = eligible.iloc[-1]
        try:
            score = float(last["displayed_score"])
            spread = float(last["market_spread"])
        except (TypeError, ValueError):
            continue
        frozen[str(game_id)] = FrozenPick(
            game_id=str(game_id),
            pick_team=str(last["pick_team"]),
            market_spread=spread,
            displayed_score=score,
            strength_word=str(last["strength_word"] or ""),
            published_at_utc=pd.Timestamp(last["published_at_utc"]),
            pick_deadline_utc=pd.Timestamp(deadline),
        )
    return frozen
