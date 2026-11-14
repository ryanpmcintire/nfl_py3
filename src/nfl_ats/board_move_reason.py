from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
STATUS_WORDS = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable"}
SNAPSHOT_STAMP = "%Y%m%dT%H%M%SZ"
INJURY_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "report_status"]
MAJORITY_SNAP_SHARE = 0.5
NEWS_RULE = "team_the_line_moved_away_from_then_largest_snap_share_then_latest_change"
TEAM_ALIAS = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "LVR": "LV"}
POSITION_WORDS = {
    "QB": "quarterback",
    "RB": "running back",
    "FB": "running back",
    "WR": "receiver",
    "TE": "tight end",
    "T": "offensive lineman",
    "G": "offensive lineman",
    "C": "offensive lineman",
    "OL": "offensive lineman",
    "OT": "offensive lineman",
    "OG": "offensive lineman",
    "DE": "defensive lineman",
    "DT": "defensive lineman",
    "NT": "defensive lineman",
    "DL": "defensive lineman",
    "LB": "linebacker",
    "ILB": "linebacker",
    "OLB": "linebacker",
    "MLB": "linebacker",
    "CB": "cornerback",
    "DB": "defensive back",
    "S": "safety",
    "SS": "safety",
    "FS": "safety",
}
NO_NEWS_SENTENCE = "No captured news explains this move."


@dataclass(frozen=True)
class MoveReason:
    sentence: str
    evidence: tuple[str, ...]
    toward_pick_points: float = 0.0


@dataclass(frozen=True)
class DesignationChange:
    team: str
    gsis_id: str
    full_name: str
    status: str
    previous_status: str
    observed_at: pd.Timestamp
    source: str


def _snapshot_time(path: Path) -> pd.Timestamp | None:
    try:
        stamp = datetime.strptime(path.parent.name, SNAPSHOT_STAMP)
    except ValueError:
        return None
    return pd.Timestamp(stamp, tz="UTC")


def _week_statuses(path: Path, season: int, week: int) -> dict[tuple[str, str], tuple[str, str]]:
    try:
        frame = pd.read_parquet(path, columns=INJURY_COLUMNS)
    except (OSError, ValueError, KeyError):
        return {}
    frame = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
    frame = frame.dropna(subset=["gsis_id", "team"])
    return {
        (TEAM_ALIAS.get(str(team), str(team)), str(gsis)): (
            "" if pd.isna(status) else str(status),
            str(name),
        )
        for team, gsis, status, name in zip(
            frame["team"],
            frame["gsis_id"],
            frame["report_status"],
            frame["full_name"],
            strict=True,
        )
    }


def designation_changes(
    data_root: Path, *, season: int, week: int, start: pd.Timestamp
) -> list[DesignationChange]:
    stamped = sorted(
        (when, path)
        for path in (data_root / "raw" / "nflverse_injuries").glob("*/injuries.parquet")
        if (when := _snapshot_time(path)) is not None
    )
    earlier = [path for when, path in stamped if when <= start]
    known = _week_statuses(earlier[-1], season, week) if earlier else {}
    changes: list[DesignationChange] = []
    for when, path in stamped:
        if when <= start:
            continue
        statuses = _week_statuses(path, season, week)
        for (team, gsis), (status, name) in statuses.items():
            previous = known.get((team, gsis), ("", ""))[0]
            if status != previous and status in STATUS_WORDS:
                changes.append(
                    DesignationChange(
                        team=team,
                        gsis_id=gsis,
                        full_name=name,
                        status=status,
                        previous_status=previous,
                        observed_at=when,
                        source=path.relative_to(data_root).as_posix(),
                    )
                )
        known.update(statuses)
    return changes


def window_start_utc(first_gameday: date) -> pd.Timestamp:
    tuesday = first_gameday - timedelta(days=(first_gameday.weekday() - 1) % 7)
    local = datetime.combine(tuesday, time(12, 0), tzinfo=EASTERN)
    return pd.Timestamp(local).tz_convert("UTC")


def quote_step(quotes: pd.DataFrame) -> float:
    lines = sorted(set(quotes["home_spread_line"].astype(float).round(4)))
    gaps = [later - earlier for earlier, later in pairwise(lines)]
    return min(gaps) if gaps else 0.0


def _consensus_series(quotes: pd.DataFrame, game_id: str) -> pd.Series:
    rows = quotes.loc[quotes["nflverse_game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby("observed_at_utc")["home_spread_line"].mean().sort_index()


def _move_onset(
    series: pd.Series, *, start_line: float, now_line: float, step: float
) -> pd.Timestamp | None:
    direction = 1.0 if now_line > start_line else -1.0
    hit = series.loc[(series - start_line) * direction >= step - 1e-9]
    return None if hit.empty else pd.Timestamp(hit.index[0])


def move_size_words(size: float) -> str:
    return "half a point" if size == 0.5 else f"{size:g} point" if size == 1 else f"{size:g} points"


def _toward_pick_points(delta: float, step: float, pick_team: str, home: str) -> float:
    size = round(abs(delta) / step) * step
    sign = -1.0 if pick_team == home else 1.0
    return size if delta * sign < 0 else -size


def _move_text(delta: float, step: float, pick_team: str, home: str) -> str:
    points = _toward_pick_points(delta, step, pick_team, home)
    return f"{move_size_words(abs(points))} {'toward' if points > 0 else 'against'} {pick_team}"


def _regulars(data_root: Path, season: int, week: int) -> pd.DataFrame:
    rosters_path = sorted((data_root / "players" / "raw").glob("*/weekly_rosters.parquet"))
    snaps_path = (
        data_root
        / "players"
        / "forward_snaps"
        / f"snap_counts_{season}_before_week_{week:02d}.parquet"
    )
    if not rosters_path or not snaps_path.is_file():
        return pd.DataFrame()
    rosters = pd.read_parquet(
        rosters_path[-1], columns=["season", "gsis_id", "pfr_id", "position", "team"]
    )
    rosters = rosters.loc[rosters["season"].eq(season)].dropna(subset=["gsis_id", "pfr_id"])
    rosters = rosters.drop_duplicates("gsis_id", keep="last")
    snaps = pd.read_parquet(
        snaps_path,
        columns=["season", "game_type", "pfr_player_id", "offense_pct", "defense_pct"],
    )
    snaps = snaps.loc[snaps["season"].eq(season) & snaps["game_type"].eq("REG")]
    share = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    mean_share = share.groupby(snaps["pfr_player_id"]).mean().rename("snap_share").reset_index()
    info = rosters.merge(mean_share, left_on="pfr_id", right_on="pfr_player_id", how="left")
    return info.set_index("gsis_id")


def news_sentence(change: DesignationChange, position_word: str) -> str:
    day = change.observed_at.tz_convert(EASTERN).day_name()
    return (
        f"Injury news since Tuesday: {change.team} {position_word} {change.full_name} was listed "
        f"as {STATUS_WORDS[change.status]} as of {day}."
    )


def build_move_reasons(
    data_root: Path,
    *,
    season: int,
    week: int,
    moves: dict[str, tuple[str, str, str, float]],
    first_gameday: date,
) -> dict[str, MoveReason]:
    from nfl_ats.market_data import load_quote_history

    if not moves:
        return {}
    start = window_start_utc(first_gameday)
    try:
        quotes = load_quote_history(
            data_root / "market" / "raw", since=start - pd.Timedelta(hours=1)
        )
    except Exception:
        quotes = pd.DataFrame()
    if not quotes.empty:
        quotes = quotes.loc[quotes["market"].eq("spreads")].dropna(subset=["home_spread_line"])
        quotes = quotes.loc[pd.to_datetime(quotes["observed_at_utc"], utc=True).ge(start)]
    step = quote_step(quotes) if not quotes.empty else 0.0
    if step <= 0:
        return {}

    changes = designation_changes(data_root, season=season, week=week, start=start)
    info = _regulars(data_root, season, week) if changes else pd.DataFrame()

    shares: dict[str, float] = {}
    positions: dict[str, str] = {}
    if not info.empty:
        measured = info["snap_share"].dropna().astype(float)
        shares = {str(gsis): float(share) for gsis, share in measured.items()}
        positions = {str(gsis): str(position) for gsis, position in info["position"].items()}

    def starter(gsis: str) -> tuple[float, str] | None:
        share = shares.get(gsis)
        word = POSITION_WORDS.get(positions.get(gsis, ""))
        if share is None or share <= MAJORITY_SNAP_SHARE or word is None:
            return None
        return (share, word)

    result: dict[str, MoveReason] = {}
    for game_id, (home, away, pick_team, now_line) in moves.items():
        series = _consensus_series(quotes, game_id)
        if series.empty:
            continue
        tuesday_line = float(series.iloc[0])
        delta = now_line - tuesday_line
        if abs(delta) < step - 1e-9:
            continue
        move_text = _move_text(delta, step, pick_team, home)
        toward_pick = _toward_pick_points(delta, step, pick_team, home)
        onset = _move_onset(series, start_line=tuesday_line, now_line=now_line, step=step)
        lead = f"{game_id} {move_text}; move onset {onset}"
        moved_away_from = away if delta > 0 else home
        ranked = [
            (rank, change)
            for change in changes
            if change.team == moved_away_from and (rank := starter(change.gsis_id)) is not None
        ]
        if not ranked:
            result[game_id] = MoveReason(
                f"The line moved {move_text}. {NO_NEWS_SENTENCE}",
                (f"{lead}; no qualifying news",),
                toward_pick,
            )
            continue
        (share, word), top = max(ranked, key=lambda item: (item[0][0], item[1].observed_at))
        players = len({change.gsis_id for _, change in ranked})
        evidence = (
            f"{lead}; shown {top.team} {top.gsis_id} {top.full_name}"
            f" {top.previous_status or 'no designation'} to {top.status}"
            f" first captured {top.observed_at} in {top.source}; position {word};"
            f" mean snap share {share:.3f}; rule {NEWS_RULE};"
            f" {players} regular players with a designation change since {start}",
        )
        result[game_id] = MoveReason(
            f"The line moved {move_text}. {news_sentence(top, word)}", evidence, toward_pick
        )
    return result
