from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.io import atomic_json, atomic_parquet

TEAM_NICKNAMES: dict[str, tuple[str, ...]] = {
    "ARI": ("cardinals",),
    "ATL": ("falcons",),
    "BAL": ("ravens",),
    "BUF": ("bills",),
    "CAR": ("panthers",),
    "CHI": ("bears",),
    "CIN": ("bengals",),
    "CLE": ("browns",),
    "DAL": ("cowboys",),
    "DEN": ("broncos",),
    "DET": ("lions",),
    "GB": ("packers",),
    "HOU": ("texans",),
    "IND": ("colts",),
    "JAX": ("jaguars",),
    "KC": ("chiefs",),
    "LA": ("rams",),
    "LAC": ("chargers",),
    "LV": ("raiders",),
    "MIA": ("dolphins",),
    "MIN": ("vikings",),
    "NE": ("patriots",),
    "NO": ("saints",),
    "NYG": ("giants",),
    "NYJ": ("jets",),
    "PHI": ("eagles",),
    "PIT": ("steelers",),
    "SEA": ("seahawks",),
    "SF": ("49ers", "niners"),
    "TB": ("buccaneers", "bucs"),
    "TEN": ("titans",),
    "WAS": ("commanders",),
}

_NICKNAME_TO_TEAM: dict[str, str] = {}
for _code, _nicks in TEAM_NICKNAMES.items():
    for _n in _nicks:
        _NICKNAME_TO_TEAM[_n] = _code

CITY_TO_TEAM: dict[str, str | None] = {
    "arizona": "ARI",
    "atlanta": "ATL",
    "baltimore": "BAL",
    "buffalo": "BUF",
    "carolina": "CAR",
    "chicago": "CHI",
    "cincinnati": "CIN",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "green bay": "GB",
    "houston": "HOU",
    "indianapolis": "IND",
    "jacksonville": "JAX",
    "kansas city": "KC",
    "las vegas": "LV",
    "miami": "MIA",
    "minnesota": "MIN",
    "new england": "NE",
    "new orleans": "NO",
    "new york": None,
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "san francisco": "SF",
    "seattle": "SEA",
    "tampa bay": "TB",
    "tennessee": "TEN",
    "washington": "WAS",
}

POSITIONS = frozenset({"qb", "rb", "wr", "te", "ol", "dl", "lb", "cb", "s", "k", "p"})

DESIGNATION_OUT = "out"
DESIGNATION_DOUBTFUL = "doubtful"
DESIGNATION_QUESTIONABLE = "questionable"
DESIGNATION_IR = "ir"
DESIGNATION_ACTIVE = "active"

_OUT_KEYWORDS = re.compile(
    r"\brule out\b|\brules out\b|\bruled out\b|\bis out\b"
    r"|\bout for (?:the )?(?:season|game|week|day)\b"
    r"|\bwill not play\b|\bwont play\b|\bwill miss\b|\bdowngraded to out\b",
)

_FORWARD_RULE_OUT = re.compile(r"\brule out\b|\brules out\b")

_IR_KEYWORDS = re.compile(
    r"\bplaced on (?:injured reserve|ir)\b"
    r"|\bto (?:injured reserve|ir)\b"
    r"|\bon (?:injured reserve|ir)\b",
)

_DOUBTFUL_KEYWORD = re.compile(r"\bdoubtful\b")

_QUESTIONABLE_KEYWORD = re.compile(r"\bquestionable\b")

_ACTIVE_KEYWORDS = re.compile(
    r"\bactivated\b|\bcleared to play\b|\bwill play\b|\bexpected to play\b"
    r"|\bgood to go\b|\bremoved from the injury report\b|\boff injury report\b"
    r"|\bset to play\b",
)

_ACTION_VERBS = frozenset(
    {
        "place",
        "put",
        "add",
        "list",
        "rule",
        "ruled",
        "call",
        "set",
        "sign",
        "claim",
        "release",
        "waive",
        "cut",
        "activate",
        "elevate",
        "designate",
        "is",
    }
)

_PLAYER_SNAPSHOT_ROOT = Path("data") / "players" / "raw"

_TRAILING_STOP_WORDS = frozenset(
    {
        "carted",
        "off",
        "listed",
        "as",
        "is",
        "was",
        "will",
        "has",
        "remains",
        "returns",
        "downgraded",
        "upgraded",
        "ruled",
        "second",
        "round",
        "pick",
        "rounder",
        "rookie",
        "veteran",
    }
)


def _normalize_name(name: str) -> str:
    normalized = name.lower().replace(".", " ").replace("'", " ").replace("-", " ")
    return " ".join(normalized.split())


def _normalize_name_stripped(name: str) -> str:
    return name.lower().replace(".", "").replace("'", "").replace("-", "")


def build_roster_index(
    rosters: pd.DataFrame,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    latest_season = int(rosters["season"].max())
    latest = rosters[rosters["season"] == latest_season]
    max_week = int(latest["week"].max())
    latest_week = latest[latest["week"] == max_week]
    name_to_teams: dict[str, set[str]] = {}
    stripped_to_teams: dict[str, set[str]] = {}
    for _, row in latest_week.iterrows():
        fn = row["full_name"]
        if pd.isna(fn):
            continue
        team = str(row["team"])
        key = _normalize_name(str(fn))
        name_to_teams.setdefault(key, set()).add(team)
        stripped_key = _normalize_name_stripped(str(fn))
        stripped_to_teams.setdefault(stripped_key, set()).add(team)
    return {
        "name_to_teams": name_to_teams,
        "stripped_to_teams": stripped_to_teams,
        "snapshot_id": snapshot_id or "",
    }


def _has_non_forward_designation(headline: str) -> bool:
    for des, start, end in _find_all_designations(headline):
        if des == DESIGNATION_OUT and _FORWARD_RULE_OUT.fullmatch(headline[start:end]):
            continue
        return True
    return False


def _find_designation_and_end(headline: str) -> tuple[str | None, int, int]:
    for pat, des in [
        (_OUT_KEYWORDS, DESIGNATION_OUT),
        (_IR_KEYWORDS, DESIGNATION_IR),
        (_DOUBTFUL_KEYWORD, DESIGNATION_DOUBTFUL),
        (_QUESTIONABLE_KEYWORD, DESIGNATION_QUESTIONABLE),
        (_ACTIVE_KEYWORDS, DESIGNATION_ACTIVE),
    ]:
        m = pat.search(headline)
        if not m:
            continue
        if des == DESIGNATION_OUT and _FORWARD_RULE_OUT.fullmatch(headline[m.start() : m.end()]):
            if _has_non_forward_designation(headline):
                continue
            return des, len(headline), len(headline)
        return des, m.start(), m.end()
    return None, -1, -1


def _find_all_designations(headline: str) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    for pat, des in [
        (_OUT_KEYWORDS, DESIGNATION_OUT),
        (_IR_KEYWORDS, DESIGNATION_IR),
        (_DOUBTFUL_KEYWORD, DESIGNATION_DOUBTFUL),
        (_QUESTIONABLE_KEYWORD, DESIGNATION_QUESTIONABLE),
        (_ACTIVE_KEYWORDS, DESIGNATION_ACTIVE),
    ]:
        for m in pat.finditer(headline):
            results.append((des, m.start(), m.end()))
    results.sort(key=lambda x: x[1])
    return results


_OPPONENT_MARKERS = frozenset(
    {
        "vs",
        "v",
        "at",
        "against",
        "versus",
        "host",
        "hosts",
        "visit",
        "visits",
        "face",
        "faces",
    }
)


def _find_team_from_headline(headline: str) -> str | None:
    tokens = headline.split()
    for i, token in enumerate(tokens):
        if i > 0 and tokens[i - 1] in _OPPONENT_MARKERS:
            continue
        if token in _NICKNAME_TO_TEAM:
            return _NICKNAME_TO_TEAM[token]
    for i in range(len(tokens) - 1):
        if i > 0 and tokens[i - 1] in _OPPONENT_MARKERS:
            continue
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        team = CITY_TO_TEAM.get(bigram)
        if team is not None:
            return team
    return None


def _strip_team_prefix(tokens: list[str]) -> tuple[list[str], str | None]:
    if not tokens:
        return tokens, None
    if tokens[0] in _NICKNAME_TO_TEAM:
        return tokens[1:], _NICKNAME_TO_TEAM[tokens[0]]
    if len(tokens) >= 2:
        bigram = f"{tokens[0]} {tokens[1]}"
        team = CITY_TO_TEAM.get(bigram)
        if team is not None:
            return tokens[2:], team
    return tokens, None


def _strip_position(tokens: list[str]) -> tuple[list[str], str | None]:
    if tokens and tokens[0] in POSITIONS:
        return tokens[1:], tokens[0]
    return tokens, None


def _strip_trailing_stop_words(tokens: list[str]) -> list[str]:
    while tokens and tokens[-1] in _TRAILING_STOP_WORDS:
        tokens = tokens[:-1]
    return tokens


@dataclass(frozen=True)
class ParsedHeadline:
    player_name: str
    position: str | None
    team: str | None
    designation: str | None
    parsed: bool
    team_source: str | None = None


def _try_roster_match(
    tokens: list[str],
    start: int,
    roster_index: dict[str, dict[str, Any]] | None,
) -> tuple[str, int] | None:
    if roster_index is None:
        return None
    name_to_teams: dict[str, set[str]] = roster_index["name_to_teams"]
    stripped_to_teams: dict[str, set[str]] = roster_index.get("stripped_to_teams", {})
    best: tuple[str, int] | None = None
    for length in range(4, 0, -1):
        if start + length > len(tokens):
            continue
        candidate = " ".join(tokens[start : start + length])
        if candidate in name_to_teams:
            candidate_clean = _strip_trailing_stop_words(list(tokens[start : start + length]))
            if candidate_clean:
                clean_name = " ".join(candidate_clean)
                if clean_name in name_to_teams:
                    best = (clean_name, start + len(candidate_clean))
                    break
            best = (candidate, start + length)
            break
        stripped_candidate = _normalize_name_stripped(candidate)
        if stripped_candidate in stripped_to_teams:
            candidate_clean = _strip_trailing_stop_words(list(tokens[start : start + length]))
            if candidate_clean:
                clean_name = " ".join(candidate_clean)
                stripped_clean = _normalize_name_stripped(clean_name)
                if stripped_clean in stripped_to_teams:
                    best = (clean_name, start + len(candidate_clean))
                    break
            best = (candidate, start + length)
            break
    return best


def _resolve_team_and_source(
    headline_team: str | None, roster_team: str | None
) -> tuple[str | None, str | None]:
    if roster_team is not None and headline_team is not None and headline_team != roster_team:
        return roster_team, "roster"
    if headline_team is not None:
        return headline_team, "headline"
    if roster_team is not None:
        return roster_team, "roster"
    return None, None


def _forward_rule_out_windows(
    headline: str, designations: list[tuple[str, int, int]]
) -> list[tuple[int, int]]:
    boundary_words = [m.start() for m in re.finditer(r"\b(?:list|lists|and)\b", headline)]
    forward_spans = [
        (start, end)
        for des, start, end in designations
        if des == DESIGNATION_OUT and _FORWARD_RULE_OUT.fullmatch(headline[start:end])
    ]
    windows: list[tuple[int, int]] = []
    for _f_start, f_end in forward_spans:
        next_boundary = len(headline)
        for _des, start, _end in designations:
            if start > f_end and start < next_boundary:
                next_boundary = start
        for offset in boundary_words:
            if offset > f_end and offset < next_boundary:
                next_boundary = offset
        windows.append((f_end, next_boundary))
    return windows


def _assign_designations(
    matches: list[tuple[str, int, int, str | None]],
    designations: list[tuple[str, int, int]],
    headline: str,
) -> list[str | None]:
    if not designations:
        return [None] * len(matches)
    windows = _forward_rule_out_windows(headline, designations)
    results: list[str | None] = []
    for _name, start, end, _team in matches:
        designated: str | None = None
        for win_start, win_end in windows:
            if win_start <= start < win_end:
                designated = DESIGNATION_OUT
                break
        if designated is not None:
            results.append(designated)
            continue
        for des, des_start, des_end in designations:
            if des == DESIGNATION_OUT and _FORWARD_RULE_OUT.fullmatch(headline[des_start:des_end]):
                continue
            if des_start > end:
                results.append(des)
                break
        else:
            results.append(designations[-1][0])
    return results


def parse_headline(
    headline_guess: str,
    roster_index: dict[str, dict[str, Any]] | None = None,
) -> list[ParsedHeadline]:
    h = headline_guess.strip()
    designation, des_start, _des_end = _find_designation_and_end(h)

    team = _find_team_from_headline(h)

    if designation is None:
        resolved_team, team_src = _resolve_team_and_source(team, None)
        return [
            ParsedHeadline(
                player_name=h,
                position=None,
                team=resolved_team,
                designation=None,
                parsed=False,
                team_source=team_src,
            )
        ]

    pre_des = h[:des_start].strip()
    pre_tokens = pre_des.split()
    pre_tokens, prefix_team = _strip_team_prefix(pre_tokens)
    if prefix_team is not None:
        team = prefix_team
    pre_tokens, position = _strip_position(pre_tokens)

    while pre_tokens and pre_tokens[0] in _ACTION_VERBS:
        pre_tokens = pre_tokens[1:]
    if pre_tokens and position is None:
        pre_tokens, position = _strip_position(pre_tokens)
    while pre_tokens and pre_tokens[-1] in _ACTION_VERBS:
        pre_tokens = pre_tokens[:-1]

    if roster_index is None:
        player_name = " ".join(pre_tokens).strip()
        if not player_name:
            player_name = h
        resolved_team, team_src = _resolve_team_and_source(team, None)
        return [
            ParsedHeadline(
                player_name=player_name,
                position=position,
                team=resolved_team,
                designation=designation,
                parsed=True,
                team_source=team_src,
            )
        ]

    name_to_teams: dict[str, set[str]] = roster_index["name_to_teams"]
    stripped_to_teams: dict[str, set[str]] = roster_index.get("stripped_to_teams", {})

    def _lookup_teams(name: str) -> set[str]:
        teams = name_to_teams.get(name, set())
        if not teams:
            stripped = _normalize_name_stripped(name)
            teams = stripped_to_teams.get(stripped, set())
        return teams

    all_designations = _find_all_designations(h)
    all_designations.sort(key=lambda x: x[1])

    pre_tokens_list = list(pre_tokens)
    token_offsets: list[int] = []
    scan_pos = 0
    for tok in pre_tokens_list:
        idx = h.find(tok, scan_pos)
        token_offsets.append(idx if idx >= 0 else scan_pos)
        scan_pos = token_offsets[-1] + len(tok)

    matches: list[tuple[str, int, int, str | None]] = []
    i = 0
    while i < len(pre_tokens_list):
        result = _try_roster_match(pre_tokens_list, i, roster_index)
        if result is not None:
            clean_name, end_pos = result
            teams = _lookup_teams(clean_name)
            match_team = next(iter(teams)) if len(teams) == 1 else None
            char_end = (
                token_offsets[end_pos - 1] + len(pre_tokens_list[end_pos - 1]) if end_pos > 0 else 0
            )
            matches.append((clean_name, token_offsets[i], char_end, match_team))
            i = end_pos
        else:
            i += 1

    if not matches:
        cleaned = _strip_trailing_stop_words(pre_tokens_list)
        player_name = " ".join(cleaned).strip()
        if not player_name:
            player_name = h
        resolved_team, team_src = _resolve_team_and_source(team, None)
        return [
            ParsedHeadline(
                player_name=player_name,
                position=position,
                team=resolved_team,
                designation=designation,
                parsed=True,
                team_source=team_src,
            )
        ]

    if len(matches) == 1:
        clean_name = matches[0][0]
        match_team = matches[0][3]
        resolved_team, team_src = _resolve_team_and_source(team, match_team)
        des_for_match = _assign_designations(matches, all_designations, h)
        return [
            ParsedHeadline(
                player_name=clean_name,
                position=position,
                team=resolved_team,
                designation=des_for_match[0],
                parsed=True,
                team_source=team_src,
            )
        ]

    des_for_matches = _assign_designations(matches, all_designations, h)
    results: list[ParsedHeadline] = []
    for idx, (clean_name, _start, _end, match_team) in enumerate(matches):
        resolved_team, team_src = _resolve_team_and_source(team, match_team)
        results.append(
            ParsedHeadline(
                player_name=clean_name,
                position=position,
                team=resolved_team,
                designation=des_for_matches[idx],
                parsed=True,
                team_source=team_src,
            )
        )
    return results


def load_all_captures(data_root: Path, since: date) -> pd.DataFrame:
    injury_dir = data_root / "raw" / "injury_news"
    since_dt = datetime(since.year, since.month, since.day, tzinfo=UTC)
    frames: list[pd.DataFrame] = []
    for capture_dir in sorted(injury_dir.iterdir()):
        if not capture_dir.is_dir():
            continue
        try:
            ts = datetime.strptime(capture_dir.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if ts < since_dt:
            continue
        current = capture_dir / "current.parquet"
        if not current.exists():
            continue
        df = pd.read_parquet(current)
        df["_capture_ts"] = ts
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_first_seen(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.groupby("url")["_capture_ts"].idxmin()
    first_seen = df.loc[idx, ["url", "_capture_ts"]].rename(
        columns={"_capture_ts": "first_seen_utc"}
    )
    result = df.drop(columns=["_capture_ts"]).merge(first_seen, on="url", how="left")
    return result


def parse_all_headlines(
    df: pd.DataFrame,
    roster_index: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    results_col = [parse_headline(h, roster_index=roster_index) for h in df["headline_guess"]]
    df["_parse_results"] = pd.Series(results_col, dtype=object)
    df = df.explode("_parse_results", ignore_index=True)
    df["player_name"] = df["_parse_results"].apply(lambda p: p.player_name)
    df["position"] = df["_parse_results"].apply(lambda p: p.position)
    df["team"] = df["_parse_results"].apply(lambda p: p.team)
    df["designation"] = df["_parse_results"].apply(lambda p: p.designation)
    df["parsed"] = df["_parse_results"].apply(lambda p: p.parsed)
    df["team_source"] = df["_parse_results"].apply(lambda p: p.team_source)
    return df.drop(columns=["_parse_results"])


def build_artifacts(
    df: pd.DataFrame,
    capture_dirs: list[str],
    artifacts_root: Path,
    roster_snapshot_id: str | None = None,
) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = artifacts_root / "injury_headline_designations" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = df[
        [
            "url",
            "lastmod",
            "first_seen_utc",
            "player_name",
            "position",
            "team",
            "team_source",
            "designation",
            "parsed",
            "headline_guess",
        ]
    ].copy()
    out_df = out_df.rename(
        columns={
            "lastmod": "filed_at_utc",
            "headline_guess": "headline",
        }
    )

    assert pd.api.types.is_datetime64_any_dtype(out_df["first_seen_utc"]), (
        "first_seen_utc must be datetime"
    )
    assert pd.api.types.is_datetime64_any_dtype(out_df["filed_at_utc"]), (
        "filed_at_utc must be datetime"
    )

    atomic_parquet(out_df, out_dir / "designations.parquet")

    des_counts = out_df["designation"].value_counts().to_dict() if not out_df.empty else {}
    unparsed_count = int((~out_df["parsed"]).sum()) if "parsed" in out_df.columns else 0
    newest_lastmod = out_df["filed_at_utc"].max() if not out_df.empty else None
    summary: dict[str, Any] = {
        "total_rows": len(out_df),
        "designation_counts": {str(k): int(v) for k, v in des_counts.items()},
        "unparsed_count": unparsed_count,
        "source_capture_count": len(capture_dirs),
        "newest_lastmod": newest_lastmod.isoformat() if newest_lastmod is not None else None,
        "roster_snapshot_id": roster_snapshot_id,
    }
    atomic_json(summary, out_dir / "summary.json")

    git_sha: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    manifest: dict[str, Any] = {
        "source_captures": capture_dirs,
        "git_head_short_sha": git_sha,
        "roster_snapshot_id": roster_snapshot_id,
    }
    atomic_json(manifest, out_dir / "manifest.json")

    latest: dict[str, Any] = {
        "run_directory": str(out_dir),
        "total_rows": summary["total_rows"],
        "unparsed_count": summary["unparsed_count"],
        "newest_lastmod": summary["newest_lastmod"],
    }
    atomic_json(latest, artifacts_root / "injury_headline_designations" / "latest.json")

    return out_dir


def run_injury_headlines(
    data_root: Path,
    artifacts_root: Path,
    since: date,
    dry: bool,
) -> dict[str, Any]:
    df = load_all_captures(data_root, since)
    if df.empty:
        return {"status": "empty", "total_rows": 0}

    capture_dirs = sorted(
        {
            str(p.parent.name)
            for p in (data_root / "raw" / "injury_news").glob("*/current.parquet")
            if p.parent.name >= since.strftime("%Y%m%d")[:8]
        }
    )

    roster_index: dict[str, dict[str, Any]] | None = None
    roster_snapshot_id: str | None = None
    try:
        from nfl_ats.players import latest_player_snapshot, load_player_snapshot

        snapshot = latest_player_snapshot(_PLAYER_SNAPSHOT_ROOT)
        _injuries, rosters, _snaps = load_player_snapshot(snapshot, include_postseason=False)
        roster_index = build_roster_index(rosters, snapshot_id=snapshot.snapshot_id)
        roster_snapshot_id = snapshot.snapshot_id
    except Exception:
        pass

    df = build_first_seen(df)
    df = parse_all_headlines(df, roster_index=roster_index)

    out_df = df[
        [
            "url",
            "lastmod",
            "first_seen_utc",
            "player_name",
            "position",
            "team",
            "team_source",
            "designation",
            "parsed",
            "headline_guess",
        ]
    ].copy()
    out_df = out_df.rename(
        columns={
            "lastmod": "filed_at_utc",
            "headline_guess": "headline",
        }
    )

    des_counts = out_df["designation"].value_counts().to_dict() if not out_df.empty else {}
    unparsed_count = int((~out_df["parsed"]).sum()) if "parsed" in out_df.columns else 0
    newest_lastmod = out_df["filed_at_utc"].max() if not out_df.empty else None

    summary: dict[str, Any] = {
        "total_rows": len(out_df),
        "designation_counts": {str(k): int(v) for k, v in des_counts.items()},
        "unparsed_count": unparsed_count,
        "source_capture_count": len(capture_dirs),
        "newest_lastmod": newest_lastmod.isoformat() if newest_lastmod is not None else None,
        "roster_snapshot_id": roster_snapshot_id,
    }

    if dry:
        return {"status": "dry", **summary}

    run_dir = build_artifacts(
        df, capture_dirs, artifacts_root, roster_snapshot_id=roster_snapshot_id
    )
    return {"status": "measured", "run_directory": str(run_dir), **summary}
