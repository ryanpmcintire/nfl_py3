"""Leakage, join and known-answer tests for the CFB body-clock candidate columns.

Predeclared in ``docs/cfb_body_clock_replication.md``. The leakage regression
(``test_columns_are_invariant_to_every_outcome_column``) is the one this
repository's research invariant requires for every new feature family: each
candidate column must be a pure function of pregame facts (kickoff timestamp,
venue state/city, team identity) and must not move when every outcome column
is shuffled.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_body_clock_feature import (
    CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN,
    CFB_BODY_CLOCK_FEATURE_COLUMNS,
    CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN,
    CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN,
    CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN,
    SPLIT_STATE_CITY_TIMEZONES,
    SPLIT_STATES,
    STATE_TIMEZONES,
    attach_cfb_body_clock_features,
    derive_cfb_body_clock_features,
    eastern_kickoff_minutes,
    load_team_zone_map,
    resolve_timezone,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
TEAM_INFO_GLOB = "data/cfb/team_info/raw/*/season=*/team_info.parquet"

requires_population = pytest.mark.skipif(
    not FEATURES_PATH.is_file() or not list(REPO_ROOT.glob(TEAM_INFO_GLOB)),
    reason="local CFB benchmark table / team_info snapshot not present",
)

WEST = "America/Los_Angeles"
ARIZONA = "America/Phoenix"
EASTERN = "America/New_York"
CENTRAL = "America/Chicago"
MOUNTAIN = "America/Denver"
HAWAII = "Pacific/Honolulu"

# (team_id, school, city, state, zone) -- hand-assigned, every zone checked
# against the school's real venue location.
FIXTURE_TEAMS = [
    (1, "Washington", "Seattle", "WA", WEST),
    (2, "Rutgers", "Piscataway", "NJ", EASTERN),
    (3, "Arizona State", "Tempe", "AZ", ARIZONA),
    (4, "UTEP", "El Paso", "TX", MOUNTAIN),
    (5, "Tennessee", "Knoxville", "TN", EASTERN),
    (6, "Memphis", "Memphis", "TN", CENTRAL),
    (7, "Hawai'i", "Honolulu", "HI", HAWAII),
    (8, "Idaho", "Moscow", "ID", WEST),
]

# game_id, week, home_id, away_id, kickoff (UTC), neutral_site, then the four
# HAND-COMPUTED expected flags in column order.
FIXTURE_GAMES = [
    # 12:00 EDT, Pacific visitor at an Eastern host: early + west + eastern
    # host; venue -4 minus body -7 = 3 hours eastbound.
    ("g1", 5, 2, 1, "2019-09-28T16:00:00Z", 0, (1.0, 1.0, 1.0, 0.0)),
    # 12:00 CDT = 13:00 EDT, Arizona visitor at a Central host: early + west,
    # NOT an Eastern host; venue -5 minus body -7 = 2 hours -> eastbound.
    ("g2", 5, 6, 3, "2019-09-28T17:00:00Z", 0, (1.0, 0.0, 1.0, 0.0)),
    # DST BOUNDARY, same two teams seven weeks later: 12:00 CST = 13:00 EST.
    # Central is now -6, Arizona still -7, so the gap is 1 hour and the
    # eastbound cell switches OFF while the body-clock cells stay ON.
    ("g3", 12, 6, 3, "2019-11-16T18:00:00Z", 0, (1.0, 0.0, 0.0, 0.0)),
    # 20:30 EDT: night window, not early.
    ("g4", 5, 2, 1, "2019-09-29T00:30:00Z", 0, (0.0, 0.0, 1.0, 1.0)),
    # 01:30 EDT the following calendar day at Hawaii: carried forward to
    # minute 1530, so it reads as the previous evening's LATE window and never
    # as an early kickoff. Westbound (-10 minus -7 = -3), so no eastbound flag.
    ("g5", 5, 7, 1, "2019-09-29T05:30:00Z", 0, (0.0, 0.0, 0.0, 1.0)),
    # SPLIT STATE, non-default city: Idaho plays in Moscow, which is Pacific,
    # not the state's Mountain default -- so this IS a west body clock.
    ("g6", 5, 2, 8, "2019-09-28T16:00:00Z", 0, (1.0, 1.0, 1.0, 0.0)),
    # SPLIT STATE, non-default city: UTEP plays in El Paso, which is Mountain,
    # not Texas's Central default -- and Mountain is excluded from the WEST
    # set exactly as Denver is on the NFL side. Still 2 hours eastbound.
    ("g7", 5, 2, 4, "2019-09-28T16:00:00Z", 0, (0.0, 0.0, 1.0, 0.0)),
    # Hawaii body clock is excluded from the WEST set (its dose is 5-6 hours,
    # not the NFL construct's 2-3), but the timezone-general eastbound cell
    # still covers it: -4 minus -10 = 6 hours.
    ("g8", 5, 2, 7, "2019-09-28T16:00:00Z", 0, (0.0, 0.0, 1.0, 0.0)),
    # NEUTRAL SITE: the host's listed venue is not where this is played, so the
    # game's own timezone is unknown and every cell is NaN, never 0.
    ("g9", 5, 2, 1, "2019-09-28T16:00:00Z", 1, (np.nan, np.nan, np.nan, np.nan)),
    # Split state, DEFAULT city, Eastern half of Tennessee: Knoxville host, so
    # this is an Eastern host receiving a Pacific visitor at 12:00 EDT.
    ("g10", 5, 5, 1, "2019-09-28T16:00:00Z", 0, (1.0, 1.0, 1.0, 0.0)),
]

COLUMN_ORDER = (
    CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN,
    CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN,
    CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN,
    CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN,
)


@pytest.fixture
def fixture_zones() -> pd.DataFrame:
    rows = []
    for team_id, school, city, state, zone in FIXTURE_TEAMS:
        rows.append(
            {
                "season": 2019,
                "team_id": team_id,
                "school": school,
                "venue_id": 1000 + team_id,
                "venue_name": f"{school} Stadium",
                "city": city,
                "state": state,
                "body_zone": zone,
                "split_state_fallback": False,
            }
        )
    frame = pd.DataFrame(rows)
    frame["team_id"] = frame["team_id"].astype("Int64")
    return frame


@pytest.fixture
def fixture_games() -> pd.DataFrame:
    rows = []
    for game_id, week, home_id, away_id, kickoff, neutral, _expected in FIXTURE_GAMES:
        rows.append(
            {
                "game_id": game_id,
                "season": 2019,
                "week": week,
                "kickoff": pd.Timestamp(kickoff),
                "gameday": pd.Timestamp(kickoff).tz_convert(None).normalize(),
                "home_id": home_id,
                "away_id": away_id,
                "neutral_site": neutral,
                "result": 7.0,
                "ats_margin": 3.0,
                "home_points": 28,
                "away_points": 21,
                "home_cover": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# (c) known-answer fixtures, one hand-computed expectation per cell
# ---------------------------------------------------------------------------


def test_known_answer_flags_per_cell(
    fixture_games: pd.DataFrame, fixture_zones: pd.DataFrame
) -> None:
    derived, _diagnostics = derive_cfb_body_clock_features(fixture_games, team_zones=fixture_zones)
    derived = derived.set_index("game_id")
    for game_id, _w, _h, _a, _k, _n, expected in FIXTURE_GAMES:
        actual = tuple(float(derived.loc[game_id, column]) for column in COLUMN_ORDER)
        for column, got, want in zip(COLUMN_ORDER, actual, expected, strict=True):
            if np.isnan(want):
                assert np.isnan(got), f"{game_id}/{column}: expected NaN, got {got}"
            else:
                assert got == want, f"{game_id}/{column}: expected {want}, got {got}"


def test_every_cell_is_exercised_in_both_directions() -> None:
    """The fixture must contain at least one 1 and one 0 for each cell."""

    for index, column in enumerate(COLUMN_ORDER):
        values = [expected[index] for *_rest, expected in FIXTURE_GAMES]
        finite = [v for v in values if not np.isnan(v)]
        assert 1.0 in finite, f"{column} is never flagged in the fixture"
        assert 0.0 in finite, f"{column} is never unflagged in the fixture"


def test_dst_boundary_flips_the_eastbound_cell(
    fixture_games: pd.DataFrame, fixture_zones: pd.DataFrame
) -> None:
    """Same two teams, same local kickoff hour, opposite sides of the DST switch."""

    derived, diagnostics = derive_cfb_body_clock_features(fixture_games, team_zones=fixture_zones)
    context = diagnostics["context"].set_index(fixture_games["game_id"].to_numpy())
    # Central host, Arizona visitor: 2 hours apart in CDT, 1 hour apart in CST.
    assert context.loc["g2", "tz_delta_eastbound"] == pytest.approx(2.0)
    assert context.loc["g3", "tz_delta_eastbound"] == pytest.approx(1.0)
    derived = derived.set_index("game_id")
    assert derived.loc["g2", CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN] == 1.0
    assert derived.loc["g3", CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN] == 0.0


def test_past_midnight_kickoff_is_carried_into_the_late_window() -> None:
    """01:30 ET must read as 1530 minutes, not 90 -- otherwise it looks early."""

    kickoffs = pd.Series(
        pd.to_datetime(
            [
                "2019-09-28T16:00:00Z",  # 12:00 EDT
                "2019-09-29T05:30:00Z",  # 01:30 EDT, the previous night's game
                "2019-11-16T18:00:00Z",  # 13:00 EST
            ],
            utc=True,
        )
    )
    minutes = eastern_kickoff_minutes(kickoffs)
    assert minutes.tolist() == [720.0, 1530.0, 780.0]


# ---------------------------------------------------------------------------
# (d) neutral-site handling matches the predeclared rule
# ---------------------------------------------------------------------------


def test_neutral_site_rows_are_nan_on_every_cell(
    fixture_games: pd.DataFrame, fixture_zones: pd.DataFrame
) -> None:
    derived, _diagnostics = derive_cfb_body_clock_features(fixture_games, team_zones=fixture_zones)
    neutral = derived.loc[fixture_games["neutral_site"].to_numpy() == 1]
    assert len(neutral) == 1
    for column in CFB_BODY_CLOCK_FEATURE_COLUMNS:
        assert neutral[column].isna().all(), f"{column} is not NaN at a neutral site"
    non_neutral = derived.loc[fixture_games["neutral_site"].to_numpy() == 0]
    for column in CFB_BODY_CLOCK_FEATURE_COLUMNS:
        assert non_neutral[column].notna().all(), f"{column} has an unexpected NaN"


def test_neutral_site_row_is_kept_not_dropped(
    fixture_games: pd.DataFrame, fixture_zones: pd.DataFrame
) -> None:
    merged, diagnostics = attach_cfb_body_clock_features(fixture_games, team_zones=fixture_zones)
    assert len(merged) == len(fixture_games)
    assert diagnostics["n_neutral_site"] == 1
    assert diagnostics["n_applicable"] == len(fixture_games) - 1
    # additive merge: every pre-existing column comes back bit-identical
    pd.testing.assert_frame_equal(merged.loc[:, fixture_games.columns], fixture_games)


# ---------------------------------------------------------------------------
# (a) LEAKAGE regression
# ---------------------------------------------------------------------------


def _shuffled_outcomes(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    for column in ("result", "ats_margin", "home_points", "away_points", "home_cover"):
        if column in shuffled.columns:
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    return shuffled


def test_columns_are_invariant_to_every_outcome_column(
    fixture_games: pd.DataFrame, fixture_zones: pd.DataFrame
) -> None:
    base, _ = derive_cfb_body_clock_features(fixture_games, team_zones=fixture_zones)
    for seed in (1, 2, 3):
        shuffled, _ = derive_cfb_body_clock_features(
            _shuffled_outcomes(fixture_games, seed), team_zones=fixture_zones
        )
        pd.testing.assert_frame_equal(base, shuffled)


@requires_population
def test_columns_are_invariant_to_outcomes_on_the_real_population() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    zones = load_team_zone_map()
    base, _ = derive_cfb_body_clock_features(features, team_zones=zones)
    shuffled, _ = derive_cfb_body_clock_features(
        _shuffled_outcomes(features, 20260901), team_zones=zones
    )
    pd.testing.assert_frame_equal(base, shuffled)


@requires_population
def test_columns_depend_only_on_declared_pregame_inputs() -> None:
    """Dropping every non-declared column must not change a single flag.

    The declared inputs are the kickoff timestamp, the two team ids, the
    season/week keys and ``neutral_site``. If a column were secretly reading
    anything else, this would fail.
    """

    features = pd.read_parquet(FEATURES_PATH)
    zones = load_team_zone_map()
    base, _ = derive_cfb_body_clock_features(features, team_zones=zones)
    minimal = features.loc[
        :, ["game_id", "season", "week", "kickoff", "home_id", "away_id", "neutral_site"]
    ].copy()
    trimmed, _ = derive_cfb_body_clock_features(minimal, team_zones=zones)
    pd.testing.assert_frame_equal(base, trimmed)


# ---------------------------------------------------------------------------
# (b) JOIN correctness on the real population
# ---------------------------------------------------------------------------


@requires_population
def test_team_info_join_covers_the_whole_population_on_both_sides() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    zones = load_team_zone_map()
    _derived, diagnostics = derive_cfb_body_clock_features(features, team_zones=zones)
    assert diagnostics["n_games"] == len(features)
    assert diagnostics["n_unresolved_zone"] == 0
    for label in ("home_zone_coverage_by_season", "away_zone_coverage_by_season"):
        coverage = diagnostics[label]
        assert coverage, label
        for season, value in coverage.items():
            assert value == 1.0, f"{label}[{season}] = {value}"


@requires_population
def test_no_split_state_school_falls_back_to_a_state_default() -> None:
    """Every split-state school in the population is resolved by its own CITY."""

    features = pd.read_parquet(FEATURES_PATH)
    zones = load_team_zone_map()
    _derived, diagnostics = derive_cfb_body_clock_features(features, team_zones=zones)
    assert diagnostics["n_split_state_city_fallback"] == 0


@requires_population
def test_every_state_in_the_population_has_a_declared_zone() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    zones = load_team_zone_map()
    played = set(
        pd.concat(
            [
                features["home_id"].astype("Int64").astype(str)
                + "|"
                + features["season"].astype(int).astype(str),
                features["away_id"].astype("Int64").astype(str)
                + "|"
                + features["season"].astype(int).astype(str),
            ]
        )
    )
    keyed = zones["team_id"].astype(str) + "|" + zones["season"].astype(str)
    used = zones.loc[keyed.isin(played)]
    assert not used.empty
    assert used["body_zone"].notna().all()
    for state in used["state"].dropna().unique():
        assert str(state) in STATE_TIMEZONES or str(state) in SPLIT_STATES


# ---------------------------------------------------------------------------
# the state -> timezone map itself
# ---------------------------------------------------------------------------


def test_split_state_cities_resolve_away_from_their_state_default() -> None:
    """The rule has teeth: these four schools would be mis-zoned by state alone."""

    assert resolve_timezone("TX", "El Paso") == (MOUNTAIN, False)
    assert resolve_timezone("TX", "Austin") == (CENTRAL, False)
    assert resolve_timezone("ID", "Moscow") == (WEST, False)
    assert resolve_timezone("ID", "Boise") == ("America/Boise", False)
    assert resolve_timezone("TN", "Knoxville") == (EASTERN, False)
    assert resolve_timezone("TN", "Memphis") == (CENTRAL, False)
    assert resolve_timezone("KY", "Bowling Green") == (CENTRAL, False)
    assert resolve_timezone("KY", "Louisville") == (EASTERN, False)


def test_unknown_split_state_city_falls_back_and_says_so() -> None:
    zone, fallback = resolve_timezone("TX", "Nowheresville")
    assert zone == CENTRAL
    assert fallback is True


def test_single_zone_state_ignores_the_city() -> None:
    assert resolve_timezone("CA", "Berkeley") == (WEST, False)
    assert resolve_timezone("ny", " Syracuse ") == (EASTERN, False)


def test_missing_state_resolves_to_none() -> None:
    assert resolve_timezone(None, "Anywhere") == (None, False)
    assert resolve_timezone(float("nan"), "Anywhere") == (None, False)


def test_declared_zone_tables_are_disjoint_and_loadable() -> None:
    assert not SPLIT_STATES.intersection(STATE_TIMEZONES)
    for state, _city in SPLIT_STATE_CITY_TIMEZONES:
        assert state in SPLIT_STATES
    for zone in set(STATE_TIMEZONES.values()) | set(SPLIT_STATE_CITY_TIMEZONES.values()):
        ZoneInfo(zone)
