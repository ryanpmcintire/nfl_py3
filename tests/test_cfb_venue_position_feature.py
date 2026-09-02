"""Leakage, join and known-answer regression tests for the CFB venue-milestone
/ schedule-position candidate columns (``docs/cfb_venue_position_replication.md``).

Three of the four columns are pure schedule facts. The fourth,
``cfb_schedule_revenge_prior_meeting_loss``, deliberately reads a PRIOR game's
final score, which is pregame-legal -- so the leakage test shuffles only the
CURRENT game's outcome columns, and a separate test asserts directly that the
prior-meeting lookup never reaches a game at or after the current kickoff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_venue_position_feature import (
    CFB_HOME_OPENER_COLUMN,
    CFB_NEW_VENUE_DEBUT_COLUMN,
    CFB_REVENGE_PRIOR_MEETING_COLUMN,
    CFB_THREE_PLUS_ROAD_COLUMN,
    CFB_VENUE_POSITION_FEATURE_COLUMNS,
    attach_cfb_venue_position_features,
    attach_prior_meeting,
    build_team_side_sequence,
    declared_home_venues,
    default_schedules_dir,
    default_team_info_dir,
    derive_cfb_venue_position_features,
    flag_three_plus_road,
    load_schedules,
    load_team_own_venues,
    normalize_schedules,
)
from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"


# ---------------------------------------------------------------------------
# Hand-built fixture
# ---------------------------------------------------------------------------

# Teams: 1 = Alpha, 2 = Beta, 3 = Gamma, 4 = Delta, 5 = Epsilon.
# Venues: 10 = Alpha's old barn, 11 = Alpha's new barn, 20 = Beta's field,
#         30 = Gamma's field, 40 = Delta's field, 50 = Epsilon's field,
#         99 = a neutral dome.
#
# Season 2001 is the snapshot's LEFT EDGE. Season 2002 is the scored season.
#: ``(game_id, season, week, home_id, away_id, venue_id, neutral, day, home_pts,
#: away_pts)``. Kept as tuples rather than dicts so the whole hand-computed
#: schedule stays readable as a table.
_FIXTURE_ROWS: tuple[tuple[int, int, int, int, int, int, bool, str, int, int], ...] = (
    # --- 2001, the snapshot's LEFT EDGE -------------------------------------
    (1, 2001, 1, 1, 2, 10, False, "2001-09-01", 30, 10),  # Alpha's 1st ever home game
    (2, 2001, 2, 2, 1, 20, False, "2001-09-08", 21, 24),  # Beta's 1st home game
    (3, 2001, 3, 3, 4, 30, False, "2001-09-15", 7, 35),  # Gamma hosts Delta, LOSES
    (4, 2001, 4, 4, 3, 40, False, "2001-09-22", 14, 28),  # Delta hosts Gamma, LOSES
    (5, 2001, 5, 5, 1, 50, False, "2001-09-29", 17, 14),  # Epsilon's 1st home game
    # --- 2002, the scored season --------------------------------------------
    (10, 2002, 1, 1, 5, 99, True, "2002-08-31", 20, 17),  # (a) Alpha NEUTRAL opener
    (11, 2002, 2, 1, 3, 11, False, "2002-09-07", 31, 3),  # (b) Alpha's NEW venue
    (12, 2002, 3, 1, 4, 11, False, "2002-09-14", 28, 21),  # (c) same new venue again
    (13, 2002, 1, 3, 2, 30, False, "2002-08-31", 10, 13),  # (d) Beta road 1, Beta wins
    (14, 2002, 2, 4, 2, 40, False, "2002-09-07", 17, 20),  # (e) Beta road 2
    (15, 2002, 4, 1, 2, 10, False, "2002-09-21", 24, 14),  # (f) Beta road 3 -> FLAGS
    (16, 2002, 5, 2, 5, 99, True, "2002-09-28", 13, 10),  # (g) Beta NEUTRAL, resets
    (17, 2002, 6, 3, 2, 30, False, "2002-10-05", 21, 7),  # (h) Beta road, streak 1
    (18, 2002, 7, 4, 2, 40, False, "2002-10-12", 14, 10),  # (i) Beta road, streak 2
    (19, 2002, 8, 5, 2, 50, False, "2002-10-19", 30, 20),  # (j) Beta road 3 -> FLAGS
    (20, 2002, 9, 2, 3, 20, False, "2002-10-26", 35, 14),  # (k) Beta HOME, resets
    (21, 2002, 10, 3, 2, 30, False, "2002-11-02", 17, 16),  # (l) Beta road, streak 1
    (22, 2002, 11, 3, 4, 30, False, "2002-11-09", 20, 17),  # (m) cross-season rematch
)


def _fixture_schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "season_type": "regular",
                "start_date": f"{day}T18:00:00.000Z",
                "neutral_site": neutral,
                "venue_id": venue_id,
                "home_id": home_id,
                "away_id": away_id,
                "home_points": home_points,
                "away_points": away_points,
            }
            for (
                game_id,
                season,
                week,
                home_id,
                away_id,
                venue_id,
                neutral,
                day,
                home_points,
                away_points,
            ) in _FIXTURE_ROWS
        ]
    )


def _fixture_features() -> pd.DataFrame:
    schedules = _fixture_schedules()
    scored = schedules.loc[schedules["season"] == 2002].reset_index(drop=True)
    margin = (scored["home_points"] - scored["away_points"]).astype(float)
    return pd.DataFrame(
        {
            "game_id": scored["game_id"],
            "season": scored["season"],
            "week": scored["week"],
            "home_id": scored["home_id"],
            "away_id": scored["away_id"],
            "neutral_site": scored["neutral_site"],
            "home_points": scored["home_points"],
            "away_points": scored["away_points"],
            "result": margin,
            "ats_margin": margin - 3.0,
        }
    )


def _empty_own_venues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": pd.Series(dtype="Int64"),
            "team_id": pd.Series(dtype="Int64"),
            "own_venue_id": pd.Series(dtype="Int64"),
        }
    )


@pytest.fixture(scope="module")
def fixture_derived() -> pd.DataFrame:
    derived, _ = derive_cfb_venue_position_features(
        _fixture_features(), schedules=_fixture_schedules(), team_own_venues=_empty_own_venues()
    )
    return derived.set_index("game_id")


# ---------------------------------------------------------------------------
# (c) KNOWN-ANSWER fixtures, one per cell
# ---------------------------------------------------------------------------


def test_home_opener_known_answers(fixture_derived: pd.DataFrame) -> None:
    """Only a team's first TRUE home game of 2002 flags; the neutral-site
    opener does not flag and does not consume the slot."""

    column = fixture_derived[CFB_HOME_OPENER_COLUMN]
    assert column.loc[11] == 1.0, "Alpha's first true home game, after a neutral opener"
    assert column.loc[10] == 0.0, "a neutral-site game is never a home opener"
    assert column.loc[12] == 0.0, "Alpha's second home game"
    assert column.loc[15] == 0.0, "Alpha's third home game"
    assert column.loc[13] == 1.0, "Gamma's first true home game of 2002"
    assert column.loc[14] == 1.0, "Delta's first true home game of 2002"
    assert column.loc[19] == 1.0, "Epsilon's first true home game of 2002"
    assert column.loc[20] == 1.0, "Beta's first true home game of 2002"
    # One per team-season that hosts at all: Alpha, Beta, Gamma, Delta, Epsilon.
    assert column.sum() == 5.0


def test_new_venue_debut_known_answers(fixture_derived: pd.DataFrame) -> None:
    """Alpha's move to venue 11 is the only debut in the fixture."""

    column = fixture_derived[CFB_NEW_VENUE_DEBUT_COLUMN]
    assert column.loc[11] == 1.0, "first game at the new declared home venue"
    assert column.loc[12] == 0.0, "second game at the same new venue is not a debut"
    assert column.loc[15] == 0.0, "returning to the old venue is not a debut"
    assert column.sum() == 1.0


def test_new_venue_debut_never_flags_the_snapshot_left_edge() -> None:
    """A team's FIRST snapshot season can never be a venue debut."""

    schedules = _fixture_schedules()
    left_edge = schedules.loc[schedules["season"] == 2001]
    features = pd.DataFrame(
        {
            "game_id": left_edge["game_id"],
            "season": left_edge["season"],
            "home_id": left_edge["home_id"],
            "away_id": left_edge["away_id"],
        }
    )
    derived, _ = derive_cfb_venue_position_features(
        features, schedules=schedules, team_own_venues=_empty_own_venues()
    )
    assert derived[CFB_NEW_VENUE_DEBUT_COLUMN].sum() == 0.0


def test_three_plus_road_known_answers(fixture_derived: pd.DataFrame) -> None:
    column = fixture_derived[CFB_THREE_PLUS_ROAD_COLUMN]
    assert column.loc[13] == 0.0, "road game 1"
    assert column.loc[14] == 0.0, "road game 2"
    assert column.loc[15] == 1.0, "road game 3 -> flags"
    assert column.loc[16] == 0.0, "a neutral-site game is not a road game"


def test_revenge_prior_meeting_known_answers(fixture_derived: pd.DataFrame) -> None:
    """Cross-season rivalry rematch, hand-computed.

    Gamma and Delta met twice in 2001: game 3 (Gamma hosts, loses 7-35) and
    game 4 (Delta hosts, loses 14-28). The MOST RECENT prior meeting for both
    sides is game 4, which DELTA lost. In game 22 Gamma hosts Delta, so the
    AWAY side is the revenge side and the signed column is -1.
    """

    column = fixture_derived[CFB_REVENGE_PRIOR_MEETING_COLUMN]
    assert column.loc[22] == -1.0
    # Alpha hosts Gamma (game 11): no prior Alpha-Gamma meeting at all.
    assert column.loc[11] == 0.0
    # Beta at Gamma (game 13): no prior Beta-Gamma meeting before 2002 week 1.
    assert column.loc[13] == 0.0
    # Beta at Gamma again (game 17): Beta WON the week-1 meeting 13-10, so
    # Gamma lost it, and Gamma is the HOME side -> +1.
    assert column.loc[17] == 1.0


# ---------------------------------------------------------------------------
# (d) Road-streak reset rules
# ---------------------------------------------------------------------------


def test_road_streak_resets_after_home_and_after_neutral(fixture_derived: pd.DataFrame) -> None:
    """Declared rule, transcribed from the NFL cell: a neutral-site game is NOT
    a true road game, so it occupies a sequence slot and BREAKS the streak --
    and so does a true home game."""

    column = fixture_derived[CFB_THREE_PLUS_ROAD_COLUMN]
    # Beta: road, road, road(flag), NEUTRAL, road, road, road(flag), HOME, road.
    assert column.loc[15] == 1.0
    assert column.loc[17] == 0.0, "first road game after a neutral site must reset"
    assert column.loc[18] == 0.0, "second road game after a neutral site"
    assert column.loc[19] == 1.0, "third road game after a neutral site"
    assert column.loc[21] == 0.0, "first road game after a true home game must reset"


def test_road_streak_does_not_carry_across_seasons() -> None:
    """The NFL cell groups by ``(team, season)``; so does this one."""

    sequence = build_team_side_sequence(normalize_schedules(_fixture_schedules()))
    flags = flag_three_plus_road(sequence)
    beta_2001 = sequence["team_id"].eq(2) & sequence["season"].eq(2001)
    assert not bool(flags.loc[beta_2001].any())


# ---------------------------------------------------------------------------
# (a) LEAKAGE regression
# ---------------------------------------------------------------------------


def test_columns_are_invariant_to_shuffling_the_current_games_outcome() -> None:
    """Shuffling the CURRENT game's ``result`` / ``ats_margin`` / points columns
    in the features frame cannot move any candidate column.

    Cell 4 legitimately depends on a PRIOR game's result, so only the current
    game's outcome is shuffled here; the prior-meeting lookup is covered by the
    strictly-before assertion below.
    """

    features = _fixture_features()
    schedules = _fixture_schedules()
    baseline, _ = derive_cfb_venue_position_features(
        features, schedules=schedules, team_own_venues=_empty_own_venues()
    )

    rng = np.random.default_rng(20260901)
    shuffled = features.copy()
    order = rng.permutation(len(shuffled))
    for column in ("result", "ats_margin", "home_points", "away_points"):
        shuffled[column] = shuffled[column].to_numpy()[order]
    perturbed, _ = derive_cfb_venue_position_features(
        shuffled, schedules=schedules, team_own_venues=_empty_own_venues()
    )

    pd.testing.assert_frame_equal(baseline, perturbed)


def test_candidate_columns_never_read_the_scored_frames_outcome_columns() -> None:
    """Dropping every outcome column from the features frame changes nothing."""

    features = _fixture_features()
    schedules = _fixture_schedules()
    baseline, _ = derive_cfb_venue_position_features(
        features, schedules=schedules, team_own_venues=_empty_own_venues()
    )
    stripped = features.drop(columns=["result", "ats_margin", "home_points", "away_points"])
    perturbed, _ = derive_cfb_venue_position_features(
        stripped, schedules=schedules, team_own_venues=_empty_own_venues()
    )
    pd.testing.assert_frame_equal(baseline, perturbed)


def _real_schedules() -> pd.DataFrame:
    try:
        return load_schedules()
    except FileNotFoundError:  # pragma: no cover - fresh clone
        pytest.skip("no local CFB schedules snapshot")


def test_prior_meeting_lookup_never_reads_a_game_at_or_after_the_current_kickoff() -> None:
    """The cell-4 lookup is provably strictly-prior, on the fixture AND on the
    real schedules snapshot -- which carries one duplicated matchup record whose
    'prior' meeting shares its kickoff timestamp exactly."""

    for schedules in (normalize_schedules(_fixture_schedules()), _real_schedules()):
        sequence = attach_prior_meeting(build_team_side_sequence(schedules))
        used = sequence.loc[sequence["prior_within_lookback"]]
        assert len(used) > 0
        assert bool((used["prior_kickoff"] < used["kickoff"]).all()), (
            "the prior meeting must kick off STRICTLY before the current game"
        )
        seasons_back = used["season"].astype(float) - used["prior_season"].astype(float)
        assert bool((seasons_back >= 0).all())
        assert bool((seasons_back <= 2).all()), "declared lookback is at most 2 seasons"


# ---------------------------------------------------------------------------
# (b) JOIN tests against the real snapshots
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BENCHMARK.is_file(), reason="no local CFB benchmark table")
def test_every_benchmark_game_id_is_present_in_the_schedules_snapshot() -> None:
    features = pd.read_parquet(BENCHMARK, columns=["game_id", "season", "home_id", "away_id"])
    schedules = _real_schedules()
    known = set(schedules["game_id"].dropna().tolist())
    unknown = [g for g in features["game_id"].tolist() if g not in known]
    assert unknown == [], f"{len(unknown)} benchmark game_ids missing from the schedules snapshot"


@pytest.mark.skipif(not BENCHMARK.is_file(), reason="no local CFB benchmark table")
def test_team_info_join_resolves_every_benchmark_side() -> None:
    """The ``(season, team_id)`` team_info join must resolve both sides of every
    benchmark row -- and the map's own venue_id must be flagged as constant
    across seasons, which is why it is a diagnostic and not the venue source."""

    try:
        own = load_team_own_venues()
    except FileNotFoundError:  # pragma: no cover - fresh clone
        pytest.skip("no local CFB team_info snapshot")
    features = pd.read_parquet(BENCHMARK, columns=["game_id", "season", "home_id", "away_id"])
    lookup = own.loc[:, ["season", "team_id", "own_venue_id"]]
    for side in ("home_id", "away_id"):
        joined = features.assign(**{side: pd.to_numeric(features[side]).astype("Int64")}).merge(
            lookup.rename(columns={"team_id": side}), on=["season", side], how="left"
        )
        assert joined["own_venue_id"].notna().all(), f"team_info join left {side} rows unresolved"

    distinct = own.groupby("team_id")["own_venue_id"].nunique()
    assert int((distinct > 1).sum()) == 0, (
        "team_info venue_id is expected to be constant across seasons (a current-state "
        "snapshot, not a per-season history); if this ever fails, revisit section 4 of "
        "docs/cfb_venue_position_replication.md"
    )


@pytest.mark.skipif(not BENCHMARK.is_file(), reason="no local CFB benchmark table")
def test_attach_is_additive_and_complete_on_the_real_population() -> None:
    features = pd.read_parquet(BENCHMARK)
    merged, diagnostics = attach_cfb_venue_position_features(features)
    assert list(merged.columns[: len(features.columns)]) == list(features.columns)
    for column in CFB_VENUE_POSITION_FEATURE_COLUMNS:
        assert column in merged.columns
        assert merged[column].notna().all(), f"{column} must never be NaN"
    assert diagnostics["n_games"] == len(features)
    assert diagnostics["schedule_seasons"][0] <= 2001


def test_default_snapshot_dirs_resolve() -> None:
    for resolver in (default_schedules_dir, default_team_info_dir):
        try:
            path = resolver()
        except FileNotFoundError:  # pragma: no cover - fresh clone
            pytest.skip("no local CFB snapshot")
        assert path.is_dir()


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def test_attach_rejects_a_column_collision() -> None:
    features = _fixture_features().assign(**{CFB_HOME_OPENER_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=CFB_HOME_OPENER_COLUMN):
        attach_cfb_venue_position_features(
            features, schedules=_fixture_schedules(), team_own_venues=_empty_own_venues()
        )


def test_derive_rejects_a_game_id_absent_from_the_schedules_snapshot() -> None:
    features = _fixture_features()
    features.loc[0, "game_id"] = 999_999
    with pytest.raises(DataContractError, match="absent from the CFB schedules snapshot"):
        derive_cfb_venue_position_features(
            features, schedules=_fixture_schedules(), team_own_venues=_empty_own_venues()
        )


def test_declared_home_venue_is_the_plurality_venue() -> None:
    sequence = build_team_side_sequence(normalize_schedules(_fixture_schedules()))
    declared = declared_home_venues(sequence).set_index(["team_id", "season"])
    # Alpha 2002: two true home games at venue 11, one at venue 10.
    assert int(declared.loc[(1, 2002), "declared_home_venue_id"]) == 11
    # Alpha 2001: its only true home game is at venue 10.
    assert int(declared.loc[(1, 2001), "declared_home_venue_id"]) == 10
