from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
OUT_CLASSES = frozenset({"ruled_out", "did_not_practice", "absent"})
LIMITED_CLASSES = frozenset({"limited", "questionable", "boot_brace"})
OFFICIAL_OUT = frozenset({"Out", "Doubtful"})
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


def _move_text(delta: float, step: float, pick_team: str, home: str) -> str:
    size = round(abs(delta) / step) * step
    sign = -1.0 if pick_team == home else 1.0
    toward = delta * sign < 0
    word = "half a point" if size == 0.5 else f"{size:g} point" if size == 1 else f"{size:g} points"
    return f"{word} {'toward' if toward else 'against'} {pick_team}"


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

    posts_path = data_root / "processed" / "beat_reporter_posts.parquet"
    posts = pd.DataFrame()
    if posts_path.is_file():
        posts = pd.read_parquet(
            posts_path,
            columns=["team", "created_at", "is_repost", "player_gsis", "avail_classes", "uri"],
        )
        posts = posts.loc[~posts["is_repost"] & posts["avail_classes"].map(len).gt(0)]
        posts = posts.loc[posts["player_gsis"].map(len).gt(0)].copy()
        posts["team"] = posts["team"].map(lambda t: TEAM_ALIAS.get(t, t))
        posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
        posts = posts.loc[posts["created_at"].ge(start)]

    revisions_path = data_root / "processed" / "injury_revisions_2026.parquet"
    revisions = pd.DataFrame()
    if revisions_path.is_file():
        revisions = pd.read_parquet(revisions_path)
        revisions = revisions.loc[revisions["season"].eq(season) & revisions["week"].eq(week)]
        revisions = revisions.loc[~revisions["baseline_snapshot"].astype(bool)].copy()
        revisions["observed_at"] = pd.to_datetime(revisions["observed_at"], utc=True)
        revisions["team"] = revisions["team"].map(lambda t: TEAM_ALIAS.get(t, t))
        revisions = revisions.loc[revisions["observed_at"].ge(start)]

    info = _regulars(data_root, season, week)

    def regular(gsis: str) -> tuple[str, str] | None:
        if info.empty or gsis not in info.index:
            return None
        row = info.loc[gsis]
        share = row["snap_share"]
        if pd.isna(share) or share <= 0.5:
            return None
        word = POSITION_WORDS.get(str(row["position"]))
        return (str(row["position"]), word) if word else None

    result: dict[str, MoveReason] = {}
    step = quote_step(quotes) if not quotes.empty else 0.0
    for game_id, (home, away, pick_team, now_line) in moves.items():
        if step <= 0:
            continue
        series = _consensus_series(quotes, game_id)
        if series.empty:
            continue
        tuesday_line = float(series.iloc[0])
        delta = now_line - tuesday_line
        if abs(delta) < step - 1e-9:
            continue
        move_text = _move_text(delta, step, pick_team, home)
        onset = _move_onset(series, start_line=tuesday_line, now_line=now_line, step=step)
        found: dict[tuple[str, str], tuple[str, pd.Timestamp, str, str]] = {}
        if onset is not None:
            if not posts.empty:
                sel = posts.loc[posts["team"].isin([home, away]) & posts["created_at"].le(onset)]
                for _, post in sel.iterrows():
                    classes = set(post["avail_classes"])
                    level = (
                        "out"
                        if classes & OUT_CLASSES
                        else "limited"
                        if classes & LIMITED_CLASSES
                        else ""
                    )
                    if not level:
                        continue
                    for gsis in post["player_gsis"]:
                        reg = regular(str(gsis))
                        if reg is None:
                            continue
                        key = (str(post["team"]), str(gsis))
                        if key not in found or (level == "out" and found[key][0] != "out"):
                            found[key] = (
                                level,
                                post["created_at"],
                                f"beat_post {post['uri']} {sorted(classes)}",
                                reg[1],
                            )
            if not revisions.empty:
                sel = revisions.loc[
                    revisions["team"].isin([home, away]) & revisions["observed_at"].le(onset)
                ]
                for _, rev in sel.iterrows():
                    status = str(rev["report_status"])
                    if status not in OFFICIAL_OUT and status != "Questionable":
                        continue
                    reg = regular(str(rev["gsis_id"]))
                    if reg is None:
                        continue
                    level = "out" if status in OFFICIAL_OUT else "limited"
                    key = (str(rev["team"]), str(rev["gsis_id"]))
                    if key not in found or (level == "out" and found[key][0] != "out"):
                        found[key] = (level, rev["observed_at"], f"injury_report {status}", reg[1])
        out_items = [(k, v) for k, v in found.items() if v[0] == "out"]
        limited_items = [(k, v) for k, v in found.items() if v[0] == "limited"]
        chosen = out_items if out_items else limited_items
        level_word = "ruled out or out of practice" if out_items else "limited or questionable"
        evidence = tuple(
            f"{game_id} {move_text}; move onset {onset}; {team} {gsis} {v[0]} at {v[1]}"
            f" via {v[2]}; position {v[3]}"
            for (team, gsis), v in chosen
        ) or (f"{game_id} {move_text}; move onset {onset}; no qualifying news",)
        if not chosen:
            result[game_id] = MoveReason(
                f"The line moved {move_text}. {NO_NEWS_SENTENCE}", evidence
            )
            continue
        groups: dict[tuple[str, str], int] = {}
        for (team, _gsis), v in chosen:
            groups[(team, v[3])] = groups.get((team, v[3]), 0) + 1
        parts = []
        for (team, word), count in list(groups.items())[:3]:
            noun = word if count == 1 else f"{word}s"
            lead = "a" if count == 1 else str(count)
            parts.append(f"{lead} {team} {noun}")
        listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
        result[game_id] = MoveReason(
            f"The line moved {move_text}. Earlier captured news: {listed} who plays most of the "
            f"snaps was reported {level_word}.",
            evidence,
        )
    return result
