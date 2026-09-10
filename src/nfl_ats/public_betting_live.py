"""Turn the live public-betting capture store into per-game handle/ticket splits.

The Saturday and Sunday scheduler jobs (``public_betting_sat`` /
``public_betting_sun``, ``scripts/public_betting_capture.ps1``) write one
timestamped snapshot directory per run under
``data/raw/public_betting_live/``, each holding an ``index.parquet`` in the
schema ``docs/public_betting_sourcing.md`` section 6 documents: one row per
game per capture, teams already normalised to nflverse abbreviations, with
``spread_{side}_money_pct`` (handle) and ``spread_{side}_bet_pct`` (tickets)
beside the site's own ``season`` and ``week``.

Nothing in the codebase read that store: the historical handle work
(``docs/handle_follow_on_card.md``) reads the Wayback backfill index instead.
This module is the prospective reader, and it is deliberately narrow -- it
answers one question, "what did the money look like at the latest capture
strictly before this instant", and answers it with an empty mapping rather
than an exception whenever the store is absent, unreadable, or silent about
the week asked for. Callers fail open on the empty mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

LIVE_PUBLIC_BETTING_DIRECTORY = ("raw", "public_betting_live")
SNAPSHOT_DIRECTORY_FORMAT = "%Y%m%dT%H%M%SZ"

SITE_TEAM_ALIASES: dict[str, str] = {
    **TEAM_ABBREVIATION_ALIASES,
    "LAR": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
}

REQUIRED_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "capture_ts",
    "season",
    "week",
    "away_team",
    "home_team",
    "spread_home_money_pct",
    "spread_away_money_pct",
    "spread_home_bet_pct",
    "spread_away_bet_pct",
)


def live_public_betting_root(data_root: Path) -> Path:
    """The capture store the two weekend scheduler jobs write into."""

    return data_root.joinpath(*LIVE_PUBLIC_BETTING_DIRECTORY)


def normalize_site_team(abbreviation: object) -> str:
    """One site-printed abbreviation as its nflverse spelling."""

    text = str(abbreviation or "").strip().upper()
    return SITE_TEAM_ALIASES.get(text, text)


@dataclass(frozen=True)
class HandleReading:
    """One game's spread money and ticket split by side at one capture."""

    game_id: str
    captured_at_utc: pd.Timestamp
    home_money_pct: float
    away_money_pct: float
    home_ticket_pct: float
    away_ticket_pct: float

    @property
    def heavy_side(self) -> str:
        """The side holding the larger share of the money."""

        return "HOME" if self.home_money_pct >= self.away_money_pct else "AWAY"

    @property
    def heavy_money_pct(self) -> float:
        """That side's share of the money, in percent."""

        return max(self.home_money_pct, self.away_money_pct)

    @property
    def heavy_ticket_pct(self) -> float:
        """That same side's share of the tickets, in percent."""

        return self.home_ticket_pct if self.heavy_side == "HOME" else self.away_ticket_pct


def _snapshot_directories(root: Path) -> list[tuple[pd.Timestamp, Path]]:
    found: list[tuple[pd.Timestamp, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or not (candidate / "index.parquet").is_file():
            continue
        try:
            stamp = datetime.strptime(candidate.name, SNAPSHOT_DIRECTORY_FORMAT)
        except ValueError:
            continue
        found.append((pd.Timestamp(stamp, tz="UTC"), candidate))
    return sorted(found, key=lambda entry: entry[0], reverse=True)


def _week_rows(snapshot: Path, *, season: int, week: int, before: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(snapshot / "index.parquet")
    missing = sorted(set(REQUIRED_SNAPSHOT_COLUMNS).difference(frame.columns))
    if missing:
        return pd.DataFrame()
    captured = pd.to_datetime(frame["capture_ts"], utc=True, errors="coerce")
    rows = frame.loc[
        captured.notna()
        & captured.le(before)
        & pd.to_numeric(frame["season"], errors="coerce").eq(season)
        & pd.to_numeric(frame["week"], errors="coerce").eq(week)
    ].copy()
    if rows.empty:
        return rows
    rows["captured_at_utc"] = captured.loc[rows.index]
    for column in (
        "spread_home_money_pct",
        "spread_away_money_pct",
        "spread_home_bet_pct",
        "spread_away_bet_pct",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.dropna(
        subset=[
            "spread_home_money_pct",
            "spread_away_money_pct",
            "spread_home_bet_pct",
            "spread_away_bet_pct",
        ]
    )


def load_latest_public_handle(
    data_root: Path, *, season: int, week: int, before: pd.Timestamp
) -> tuple[dict[str, HandleReading], dict[str, Any]]:
    """This week's money/ticket split per nflverse game id, read-only.

    Uses the single latest capture strictly at or before ``before`` that
    carries rows for ``(season, week)`` -- so a pass run before the weekend
    captures land, or one run on a week the site has not posted yet, gets an
    empty mapping and a named reason rather than a stale reading from another
    week. Never fetches; never raises for a missing or malformed store.
    """

    unavailable: dict[str, Any] = {
        "available": False,
        "reason": "",
        "snapshot": "",
        "captured_at_utc": None,
        "games_with_reading": 0,
    }
    root = live_public_betting_root(data_root)
    if not root.is_dir():
        return {}, {**unavailable, "reason": "no_live_public_betting_store"}
    try:
        snapshots = _snapshot_directories(root)
    except OSError:
        return {}, {**unavailable, "reason": "live_public_betting_store_is_unreadable"}
    eligible = [entry for entry in snapshots if entry[0] <= before]
    if not eligible:
        return {}, {**unavailable, "reason": "no_capture_before_this_pass"}
    for _, snapshot in eligible:
        try:
            rows = _week_rows(snapshot, season=season, week=week, before=before)
        except (OSError, ValueError, KeyError):
            continue
        if rows.empty:
            continue
        readings: dict[str, HandleReading] = {}
        for row in rows.itertuples():
            away = normalize_site_team(row.away_team)
            home = normalize_site_team(row.home_team)
            if not away or not home:
                continue
            game_id = f"{season}_{week:02d}_{away}_{home}"
            readings[game_id] = HandleReading(
                game_id=game_id,
                captured_at_utc=pd.Timestamp(cast(Any, row.captured_at_utc)),
                home_money_pct=float(cast(Any, row.spread_home_money_pct)),
                away_money_pct=float(cast(Any, row.spread_away_money_pct)),
                home_ticket_pct=float(cast(Any, row.spread_home_bet_pct)),
                away_ticket_pct=float(cast(Any, row.spread_away_bet_pct)),
            )
        if not readings:
            continue
        captured_at = max(reading.captured_at_utc for reading in readings.values())
        return readings, {
            "available": True,
            "reason": "",
            "snapshot": snapshot.name,
            "captured_at_utc": captured_at.isoformat(),
            "games_with_reading": len(readings),
        }
    return {}, {**unavailable, "reason": "no_capture_covers_this_week"}
