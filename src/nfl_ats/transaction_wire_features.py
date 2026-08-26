"""Point-in-time-safe team-week features from the Pro Football Rumors (PFR)
transaction-wire archive (``docs/pfr_transactions_sourcing.md``,
``docs/transaction_wire_battery.md``).

Mechanism (``docs/transaction_wire_battery.md`` section 0): the owner's pool
posts lines Tuesday, revises once Wednesday, then FREEZES them for the week
(``docs/opener_evaluation.md`` line 129, ``docs/observed_movement_channel.md``
line 14). Practice-squad elevations, late signings, and injured-reserve moves
that happen after that freeze are the team publicly announcing which
positions it is scrambling to cover, days after the price stopped moving.
Picks stay editable to kickoff (owner-stated, ``picks-lock-at-kickoff``
memory), so a late-week channel like this one is playable even though the
line itself is frozen.

Every function here is either (a) a pure parse of a PFR URL slug into a
transaction category / set of mentioned teams, with no timing information at
all, or (b) a strict point-in-time window count that only ever admits a
transaction whose OWN precise ``datePublished`` timestamp
(``docs/pfr_transactions_sourcing.md`` section 1 -- the only reliable
day/hour-precision timestamp this source has; sitemap ``<lastmod>`` is
contaminated and never used here) is strictly earlier than the game's own
kickoff. ``tests/test_transaction_wire_features.py`` has a leakage regression
test asserting exactly this: a transaction dated at or after kickoff must
never be countable in any window this module builds.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

# ---------------------------------------------------------------------------
# 1. Team-nickname matching (slug -> canonical team codes)
# ---------------------------------------------------------------------------

# Duplicated from ``src/nfl_ats/injury_signal_refresh_tilt.py``'s
# ``TEAM_NICKNAMES`` (itself extending ``TEAM_ABBREVIATION_ALIASES`` the same
# way ``scripts/movement_attribution.py`` and ``scripts/ingest_public_betting.py``
# do), per this repo's convention of duplicating small cross-module constant
# dicts rather than importing between unrelated feature modules. Nicknames
# are relocation-invariant (the Raiders were "Raiders" in Oakland and are
# "Raiders" in Las Vegas), unlike ``scripts/fluview_battery_ingest.py``'s
# ``STATE_BY_TEAM`` state mapping, which is why this dict only needs the 32
# CURRENT canonical codes -- ``TEAM_ABBREVIATION_ALIASES`` handles the
# historical OAK/SD/STL codes on the schedules side (``canonical_team``
# below), not here.
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

# nickname token (hyphenated form, matching a PFR slug's own hyphen
# delimiter) -> canonical team code. Multi-word nicknames ("football team")
# are matched as a hyphenated substring; single-word nicknames are matched as
# a whole slug TOKEN (split on "-") so "cardinals" never false-positives
# inside an unrelated longer token.
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
    """Map a possibly-historical team code (``OAK``/``SD``/``STL``) to the
    current canonical code, via ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES``
    -- identical to ``nfl_ats.features._canonical_schedules``'s own
    ``home_team``/``away_team`` normalization, duplicated here (not imported;
    it is a private, underscore-prefixed helper) so both sides of every join
    in this module share one canonical code space."""

    return TEAM_ABBREVIATION_ALIASES.get(str(code), str(code))


def match_transaction_teams(slug: str) -> frozenset[str]:
    """Every canonical team code whose nickname appears in ``slug``.

    Zero teams for roundup/link posts with no team name in the slug
    ("minor-nfl-transactions-9-23-15"); one team for the common single-team
    case; two or more for trades and multi-team roundups. Measured on the
    full 29,414-row ``transaction_relevant`` inventory (2026-08-26 session):
    6,778 (23.0%) match zero teams, 20,377 (69.3%) match exactly one,
    2,259 (7.7%) match two or more -- see
    ``docs/transaction_wire_battery.md`` section 2.
    """

    tokens = set(slug.split("-"))
    hits: set[str] = set()
    for nickname, team in _NICKNAME_TOKEN_TO_TEAM.items():
        if nickname in tokens:
            hits.add(team)
    for nickname, team in _NICKNAME_SUBSTRING_TO_TEAM.items():
        if nickname in slug:
            hits.add(team)
    return frozenset(hits)


# ---------------------------------------------------------------------------
# 2. Transaction-type classification (slug -> one of 8 categories, or "other")
# ---------------------------------------------------------------------------

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

# Priority-ordered keyword sets. A slug is classified into the FIRST category
# (in this order) whose pattern matches, so a headline that could plausibly
# match more than one keyword family (e.g. "activated ... from injured
# reserve" contains both "activat" and "injured-reserve") lands in the more
# specific/informative bucket rather than being double counted. Matched
# against the hyphenated ``slug`` (identical token content to
# ``headline_from_slug``, just hyphens instead of spaces).
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
    """One of :data:`TRANSACTION_CATEGORIES`, or ``"other"``.

    Priority order (most specific/rarest first), so a single slug lands in
    exactly one category: practice-squad elevation, IR activation, IR
    placement, waiver claim, release, trade, suspension, signing. Elevation
    is checked BEFORE the IR patterns deliberately: measured this session,
    a large share of real elevation headlines are compound ("49ers-elevate-
    kerryon-johnson-place-jamycal-hasty-on-ir" -- two different players' two
    different transactions in one PFR headline) and would otherwise be
    swallowed by the much larger IR bucket, undercounting the rarer,
    specifically-requested elevation category -- see
    ``docs/transaction_wire_battery.md`` section 2 for the measured effect
    of this ordering (elevation count rose from 40 to 49 once reordered --
    a small, real correction; the category is genuinely thin in this
    corpus, not an artifact of priority order). This is a single-category-per-slug
    approximation throughout: a compound headline naming two events is
    counted once, under its higher-priority category, which is a real,
    disclosed undercount of whichever category sits lower in this order for
    that slug. A slug matching none of these (round-ups like "minor-nfl-
    transactions-9-23-15", "extra-points" link posts, bare "free-agent"/
    "undrafted"/"restructure"/"retirement" mentions) is ``"other"`` -- still
    ``transaction_relevant`` and reported in the coverage table, just not
    one of the 8 typed categories the battery's features are built from.
    """

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


# ---------------------------------------------------------------------------
# 3. Team-week population + point-in-time cutoffs
# ---------------------------------------------------------------------------

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
    """Combine nflverse ``gameday`` + Eastern ``gametime`` into UTC.

    Duplicated (not imported) from ``nfl_ats.features._kickoff_utc``, an
    underscore-prefixed private helper -- same duplication convention this
    module's module docstring and ``TEAM_NICKNAMES`` above already follow.
    """

    if "gametime" not in games:
        return pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    date_text = pd.to_datetime(games["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = games["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def own_week_wednesday_freeze_utc(kickoff: pd.Series) -> pd.Series:
    """Own-week Wednesday noon ET, in UTC -- the pool's line-freeze instant.

    Per ``docs/opener_evaluation.md`` ("posts lines Tuesday morning, revises
    once Wednesday, then freezes them for the week") and
    ``docs/observed_movement_channel.md`` ("line freezes Tuesday noon
    (revised once Wednesday, then frozen for the week)"): the number stops
    moving after the Wednesday revision. Neither document states an exact
    hour for that revision, so noon ET is an INFERRED convention here,
    chosen for consistency with every other noon-anchored cutoff already in
    this repo (``own_week_tuesday_noon_utc`` in
    ``injury_signal_refresh_tilt.py`` / ``movement_attribution.py`` /
    ``injury_tuesday_cutoff_experiment.py``). Same weekday-offset arithmetic
    as those functions, shifted one day later (Wednesday = weekday index 2,
    not Tuesday's index 1).

    Edge case, disclosed rather than special-cased: this returns the MOST
    RECENT Wednesday noon ET at or before kickoff. For the extremely rare
    Tuesday-kickoff game (weather makeup), that week's own Wednesday has not
    happened yet, so the freeze instant this function returns is the PRIOR
    week's Wednesday -- correct under the "most recent Wednesday" definition,
    just worth naming explicitly since it looks surprising at first glance.
    """

    kickoff_et = kickoff.dt.tz_convert("US/Eastern")
    days_since_wednesday = (kickoff_et.dt.weekday - 2) % 7
    wednesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_wednesday, unit="D")
    wednesday_noon_et = wednesday_date_et + pd.Timedelta(hours=12)
    result: pd.Series = wednesday_noon_et.dt.tz_convert("UTC")
    return result


def build_team_week_population(
    schedules: pd.DataFrame, *, season_start: int, season_end: int
) -> pd.DataFrame:
    """One row per (season, week, team) for every REG game in
    ``[season_start, season_end]``, both home and away sides, with
    ``kickoff_utc``, ``freeze_utc`` (own-week Wednesday noon ET), and
    ``window72_start_utc`` (``kickoff_utc - 72h``). Team codes are
    canonicalized via :func:`canonical_team` so a historical OAK/SD/STL row
    in older schedule data joins correctly against the nickname-derived
    (already-canonical) team codes from :func:`match_transaction_teams`.
    """

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


# ---------------------------------------------------------------------------
# 4. Dated, team-attributed transaction rows + point-in-time window counts
# ---------------------------------------------------------------------------

DATED_TRANSACTION_COLUMNS: tuple[str, ...] = ("slug", "precise_ts", "category", "team")


def explode_dated_transactions(dated: pd.DataFrame) -> pd.DataFrame:
    """Expand a ``(slug, precise_ts)`` transaction table into one row per
    ``(team, category, precise_ts)`` -- a trade slug mentioning two teams
    contributes one churn EVENT to each team's count (this module counts
    activity, not signed roster value, so no "traded away" vs. "traded for"
    direction is inferred from slug text alone). Rows matching zero teams
    (round-ups) are dropped here -- they cannot be attributed to a team-week
    from the slug alone; see ``docs/transaction_wire_battery.md`` section 2
    for the measured 23.0% zero-team-match rate this drops.
    """

    working = dated.copy()
    working["category"] = working["slug"].map(classify_transaction_slug)
    working["teams"] = working["slug"].map(match_transaction_teams)
    working = working.loc[working["teams"].map(len) > 0]
    exploded = working.explode("teams").rename(columns={"teams": "team"})
    return exploded[["slug", "precise_ts", "category", "team"]].reset_index(drop=True)


def _window_counts(event_ts_sorted: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Count of ``event_ts_sorted`` entries strictly inside ``(left, right)``
    per row, via two vectorized binary searches. ``right`` is always this
    module's ``kickoff_utc`` -- STRICTLY exclusive, which is exactly the
    leakage boundary ``tests/test_transaction_wire_features.py`` checks: an
    event with ``precise_ts >= kickoff_utc`` can never be counted."""

    lower_idx = np.searchsorted(event_ts_sorted, left, side="right")
    upper_idx = np.searchsorted(event_ts_sorted, right, side="left")
    return (upper_idx - lower_idx).astype(np.int64)


def attach_transaction_counts(
    team_week: pd.DataFrame, dated_exploded: pd.DataFrame
) -> pd.DataFrame:
    """Attach, per team-week row, point-in-time-safe counts:

    - ``n_events_since_freeze``: all 8 typed categories, ``freeze_utc <
      precise_ts < kickoff_utc``.
    - ``n_<category>_since_freeze`` for each of :data:`TRANSACTION_CATEGORIES`.
    - ``n_events_72h``: all 8 typed categories, ``kickoff_utc - 72h <
      precise_ts < kickoff_utc``.
    - ``n_<category>_72h`` for each category.

    Every window is strictly open on both ends and, critically, strictly
    LESS than ``kickoff_utc`` on the right -- no transaction dated at or
    after kickoff can ever reach any of these columns, regardless of window.
    Teams with no dated, team-attributed transactions at all get all-zero
    counts (not missing) for that team; this function does not know which
    team-seasons have complete date-fetch coverage -- that missingness is a
    property of ``dated_exploded``'s own upstream coverage and must be
    reasoned about by the caller (``scripts/transaction_wire_battery_screen.py``
    restricts scoring to seasons with complete date coverage; see
    ``docs/transaction_wire_battery.md`` section 1).
    """

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
