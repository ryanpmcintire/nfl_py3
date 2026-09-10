from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

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
    "WAS": ("commanders", "washington", "football team", "redskins"),
}

_NICKNAME_TOKEN_TO_TEAM: dict[str, str] = {}
_NICKNAME_SUBSTRING_TO_TEAM: dict[str, str] = {}
for _team, _nicknames in TEAM_NICKNAMES.items():
    for _nickname in _nicknames:
        _hyphenated = _nickname.replace(" ", "-")
        if " " in _nickname:
            _NICKNAME_SUBSTRING_TO_TEAM[_hyphenated] = _team
        else:
            _NICKNAME_TOKEN_TO_TEAM[_hyphenated] = _team


def canonical_team(code: str) -> str:

    return TEAM_ABBREVIATION_ALIASES.get(str(code), str(code))


def match_transaction_teams(slug: str) -> frozenset[str]:

    tokens = set(slug.split("-"))
    hits: set[str] = set()
    for nickname, team in _NICKNAME_TOKEN_TO_TEAM.items():
        if nickname in tokens:
            hits.add(team)
    for nickname, team in _NICKNAME_SUBSTRING_TO_TEAM.items():
        if nickname in slug:
            hits.add(team)
    return frozenset(hits)


TRANSACTION_CATEGORIES: tuple[str, ...] = (
    "ir_activation",
    "ir_placement",
    "practice_squad_elevation",
    "waiver_claim",
    "release",
    "trade",
    "suspension",
    "signing",
)
OTHER_CATEGORY = "other"
ALL_CATEGORIES: tuple[str, ...] = (*TRANSACTION_CATEGORIES, OTHER_CATEGORY)

_IR_RE = re.compile(r"injured-reserve|-ir-|-on-ir|placed-on-ir|^ir-|-ir$")
_ACTIVATE_RE = re.compile(r"activat")
_ELEVATE_RE = re.compile(r"elevat|practice-squad.*promot|promot.*practice-squad")
_CLAIM_RE = re.compile(r"\bclaim")
_RELEASE_RE = re.compile(r"\bcut-|\bcuts\b|waived|waive-|waives|released|release-|releases")
_TRADE_RE = re.compile(r"\btrade|\btrades\b|\btraded\b|acquir")
_SUSPEND_RE = re.compile(r"suspen")
_SIGN_RE = re.compile(
    r"\bsigns\b|sign-|-sign|\bsigned\b|\bsigning\b|re-signs|resigns|re-sign|"
    r"agree-to-terms|agrees-to-terms|extend|extension|extends|franchise-tag|"
    r"tenders|tendered|\btag-"
)


def classify_transaction_slug(slug: str) -> str:

    lowered = slug.lower()
    if _ELEVATE_RE.search(lowered):
        return "practice_squad_elevation"
    if _IR_RE.search(lowered) and _ACTIVATE_RE.search(lowered):
        return "ir_activation"
    if _IR_RE.search(lowered):
        return "ir_placement"
    if _CLAIM_RE.search(lowered):
        return "waiver_claim"
    if _RELEASE_RE.search(lowered):
        return "release"
    if _TRADE_RE.search(lowered):
        return "trade"
    if _SUSPEND_RE.search(lowered):
        return "suspension"
    if _SIGN_RE.search(lowered):
        return "signing"
    return OTHER_CATEGORY


TEAM_WEEK_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "is_home",
    "kickoff_utc",
    "freeze_utc",
    "window72_start_utc",
)


def kickoff_utc(games: pd.DataFrame) -> pd.Series:

    if "gametime" not in games:
        return pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    date_text = pd.to_datetime(games["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = games["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def own_week_wednesday_freeze_utc(kickoff: pd.Series) -> pd.Series:

    kickoff_et = kickoff.dt.tz_convert("US/Eastern")
    days_since_wednesday = (kickoff_et.dt.weekday - 2) % 7
    wednesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_wednesday, unit="D")
    wednesday_noon_et = wednesday_date_et + pd.Timedelta(hours=12)
    result: pd.Series = wednesday_noon_et.dt.tz_convert("UTC")
    return result


def build_team_week_population(
    schedules: pd.DataFrame, *, season_start: int, season_end: int
) -> pd.DataFrame:

    games = schedules.loc[schedules["game_type"] == "REG"].copy()
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["week"] = pd.to_numeric(games["week"], errors="raise").astype(int)
    games = games.loc[games["season"].between(season_start, season_end)].reset_index(drop=True)
    games["kickoff_utc"] = kickoff_utc(games)
    games = games.loc[games["kickoff_utc"].notna()].reset_index(drop=True)
    games["home_team"] = games["home_team"].map(canonical_team)
    games["away_team"] = games["away_team"].map(canonical_team)

    rows = []
    for side, opponent_side, is_home in (
        ("home_team", "away_team", True),
        ("away_team", "home_team", False),
    ):
        side_frame = games[
            ["season", "week", "game_id", side, opponent_side, "kickoff_utc"]
        ].rename(columns={side: "team", opponent_side: "opponent"})
        side_frame["is_home"] = is_home
        rows.append(side_frame)
    long = pd.concat(rows, ignore_index=True)
    long["freeze_utc"] = own_week_wednesday_freeze_utc(long["kickoff_utc"])
    long["window72_start_utc"] = long["kickoff_utc"] - pd.Timedelta(hours=72)
    return (
        long[list(TEAM_WEEK_COLUMNS)].sort_values(["season", "week", "team"]).reset_index(drop=True)
    )


DATED_TRANSACTION_COLUMNS: tuple[str, ...] = ("slug", "precise_ts", "category", "team")


def explode_dated_transactions(dated: pd.DataFrame) -> pd.DataFrame:

    working = dated.copy()
    working["category"] = working["slug"].map(classify_transaction_slug)
    working["teams"] = working["slug"].map(match_transaction_teams)
    working = working.loc[working["teams"].map(len) > 0]
    exploded = working.explode("teams").rename(columns={"teams": "team"})
    return exploded[["slug", "precise_ts", "category", "team"]].reset_index(drop=True)


def _window_counts(event_ts_sorted: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:

    lower_idx = np.searchsorted(event_ts_sorted, left, side="right")
    upper_idx = np.searchsorted(event_ts_sorted, right, side="left")
    return (upper_idx - lower_idx).astype(np.int64)


def attach_transaction_counts(
    team_week: pd.DataFrame, dated_exploded: pd.DataFrame
) -> pd.DataFrame:

    result = team_week.copy()
    typed = dated_exploded.loc[dated_exploded["category"] != OTHER_CATEGORY]

    by_team: dict[str, np.ndarray] = {}
    by_team_category: dict[tuple[str, str], np.ndarray] = {}
    for team_key, group in typed.groupby("team"):
        by_team[str(team_key)] = np.sort(group["precise_ts"].to_numpy(dtype="datetime64[ns]"))
    for team_category_key, group in typed.groupby(["team", "category"]):
        team_value, category_value = team_category_key
        by_team_category[(str(team_value), str(category_value))] = np.sort(
            group["precise_ts"].to_numpy(dtype="datetime64[ns]")
        )

    empty = np.array([], dtype="datetime64[ns]")
    freeze_left = (
        result["freeze_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    window72_left = (
        result["window72_start_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    right = (
        result["kickoff_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    teams = result["team"].to_numpy(dtype=object)

    def counts_for(bound_left: np.ndarray, lookup: dict) -> np.ndarray:
        out = np.zeros(len(result), dtype=np.int64)
        for team in set(teams):
            mask = teams == team
            ts = lookup.get(team, empty)
            if ts.size == 0:
                continue
            out[mask] = _window_counts(ts, bound_left[mask], right[mask])
        return out

    result["n_events_since_freeze"] = counts_for(freeze_left, by_team)
    result["n_events_72h"] = counts_for(window72_left, by_team)
    for category in TRANSACTION_CATEGORIES:
        cat_lookup = {team: ts for (team, cat), ts in by_team_category.items() if cat == category}
        result[f"n_{category}_since_freeze"] = counts_for(freeze_left, cat_lookup)
        result[f"n_{category}_72h"] = counts_for(window72_left, cat_lookup)
    return result
