"""Construction, tenure-counting, sign-convention, population-restriction and
leakage contracts for the LEAD-15 backup tenure-gap valuation flag, plus the
on-production confirmation wrapper's duck-typed reuse of
``scripts/on_production_opener_confirmation.py``.

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 9". Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no schedules/weekly_rosters snapshot is ever read).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backup_tenure_flags_on_production as btop  # noqa: E402

from nfl_ats.backup_tenure_flag_features import (  # noqa: E402
    BACKUP_TENURE_GAP_COLUMN,
    BACKUP_TENURE_POPULATION_SEASON_END,
    BACKUP_TENURE_POPULATION_SEASON_START,
    BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS,
    _canonical_team,
    attach_backup_tenure_gap_features,
    derive_backup_tenure_gap_features,
    describe_backup_tenure_population,
)
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402


def _game(
    game_id: str,
    season: int,
    gameday: str,
    home: str,
    away: str,
    home_qb: str | None,
    away_qb: str | None,
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "home_team": home,
        "away_team": away,
        "home_qb_id": home_qb,
        "away_qb_id": away_qb,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rosters(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    """(season, team, gsis_id) rows -- the only columns tenure counting reads."""

    return pd.DataFrame(rows, columns=["season", "team", "gsis_id"])


# ---------------------------------------------------------------------------
# Shared fixture: one schedule covering every scenario, keyed by game_id.
# ---------------------------------------------------------------------------


def _combined_schedule() -> pd.DataFrame:
    return _schedule(
        [
            # --- Group 1: backup detection carries across a season boundary
            # (no reset), sign convention "home starts a NEW-system backup" -> -1.
            _game("bk1", 2013, "2013-09-08", "BKA", "OPP1", "S1", "Z1"),
            _game(
                "bk2", 2013, "2013-09-15", "OPP2", "BKA", "Z2", "S1"
            ),  # same starter S1, not backup
            _game(
                "bk3", 2014, "2014-09-07", "BKA", "OPP3", "S2", "Z3"
            ),  # S2 != S1 -> backup, tenure 0
            # --- Group 2: system-tenured backup (>=2 strictly-prior seasons) -> +1.
            _game("t0", 2013, "2013-09-08", "TEN", "OPP4", "S3", "Z4"),
            _game("t1", 2015, "2015-09-13", "TEN", "OPP5", "S4", "Z5"),  # S4 != S3 -> backup
            # --- Group 3: leakage -- a roster row in the game's OWN season must
            # never be read; S B has exactly 1 STRICTLY PRIOR season (2017) plus
            # a same-season (2018) row that must be ignored -> stays new-system.
            _game("l0", 2016, "2016-09-11", "LEAK", "OPP6", "SA", "Z6"),
            _game("l1", 2018, "2018-09-09", "LEAK", "OPP7", "SB", "Z7"),
            # --- Group 4: unresolved backup identity (never appears in rosters
            # at all) contributes to neither bucket -> 0.
            _game("u0", 2013, "2013-09-08", "UNK", "OPP8", "SC", "Z8"),
            _game("u1", 2015, "2015-09-13", "UNK", "OPP9", "SD", "Z9"),
            # --- Group 5: franchise-relocation continuity on the ROSTER side
            # (old codes SL/STL alias to LA) -- 2 prior seasons under two
            # different literal roster codes still counts as system-tenured.
            _game("r0", 2013, "2013-09-08", "LA", "OPPA", "SE", "ZA"),
            _game("r1", 2016, "2016-09-11", "LA", "OPPB", "SF", "ZB"),
            # --- Group 6: franchise-relocation continuity on the SCHEDULE side
            # (OAK -> LV): the SAME starter across the relocation is NOT a
            # backup start; a later, genuinely different starter IS.
            _game("v0", 2013, "2013-09-08", "OAK", "OPPC", "SG", "ZC"),
            _game("v1", 2020, "2020-09-13", "LV", "OPPD", "SG", "ZD"),  # same SG -> not backup
            _game(
                "v2", 2021, "2021-09-12", "LV", "OPPE", "SH", "ZE"
            ),  # SH != SG -> backup, tenure 0
            # --- Group 7: population restriction -- ingredients for a
            # system-tenured backup, but the game's own season (2011) is
            # outside the declared [2013, 2025] range -> forced to 0.0.
            _game("o0", 2009, "2009-09-13", "OLD", "OPPF", "SI", "ZF"),
            _game("o1", 2011, "2011-09-11", "OLD", "OPPG", "SJ", "ZG"),
            # --- Group 8: both sides independently favour the same direction
            # simultaneously -> cancels to 0.0 (home system-tenured AND away
            # new-system both push toward home, so this is NOT the cancelling
            # case -- see "bothA"/"bothB" below for the two true cancel cases).
            _game("bothA", 2015, "2015-09-13", "TENH", "TENA", "SK", "SL_"),
            _game("bothB", 2015, "2015-09-20", "NEWH", "NEWA", "SM", "SN"),
            # --- Group 9: a not-yet-played future game (both starters
            # missing) must still appear in the output, with flag 0.0 -- never
            # dropped, never NaN.
            _game("future", 2026, "2026-09-10", "FUT1", "FUT2", None, None),
            # Priming rows so bothA/bothB's own starters have a proxy to
            # differ from (each team's OWN first archived game).
            _game("bothA0", 2013, "2013-09-08", "TENH", "X1", "P1", "X1Q"),
            _game("bothA0b", 2013, "2013-09-08", "TENA", "X2", "P2", "X2Q"),
            _game("bothB0", 2013, "2013-09-08", "NEWH", "X3", "P3", "X3Q"),
            _game("bothB0b", 2013, "2013-09-08", "NEWA", "X4", "P4", "X4Q"),
        ]
    )


def _combined_rosters() -> pd.DataFrame:
    return _rosters(
        [
            # Group 1: S2 (BKA backup at bk3, season 2014) has a roster row
            # ONLY in the game's OWN season (2014) -- resolved (a legitimate
            # new signee), but 0 STRICTLY PRIOR seasons -> new-system. (A
            # gsis_id with NO roster row anywhere is UNRESOLVED, not
            # new-system -- see group 4's SD -- so every "new-system"
            # scenario here deliberately carries a current-season-only row.)
            (2014, "BKA", "S2"),
            # Group 2: S4 (TEN backup at t1, season 2015) has 2 strictly-prior
            # seasons with TEN.
            (2013, "TEN", "S4"),
            (2014, "TEN", "S4"),
            # Group 3: SB (LEAK backup at l1, season 2018) has exactly 1
            # strictly-prior season (2017) PLUS a same-season (2018) row that
            # must be ignored.
            (2017, "LEAK", "SB"),
            (2018, "LEAK", "SB"),
            # Group 4: SD (UNK backup at u1) deliberately has NO row anywhere
            # -- the one genuinely UNRESOLVED backup identity in this fixture.
            # Group 5: SF (LA backup at r1, season 2016) has 2 strictly-prior
            # seasons under two different literal (alias) roster codes.
            (2014, "SL", "SF"),
            (2015, "STL", "SF"),
            # Group 6: SH (LV backup at v2, season 2021) has a roster row
            # ONLY in the game's own season (2021) -- resolved, 0 strictly-
            # prior seasons -> new-system.
            (2021, "LV", "SH"),
            # Group 7: SJ (OLD backup at o1, season 2011) has 2 strictly-prior
            # seasons -- would be system-tenured if the population restriction
            # did not zero the flag first.
            (2009, "OLD", "SJ"),
            (2010, "OLD", "SJ"),
            # Group 8a ("bothA"): both SK (home, TENH) and SL_ (away, TENA)
            # are system-tenured backups (>=2 prior seasons each).
            (2013, "TENH", "SK"),
            (2014, "TENH", "SK"),
            (2013, "TENA", "SL_"),
            (2014, "TENA", "SL_"),
            # Group 8b ("bothB"): both SM (home, NEWH) and SN (away, NEWA) are
            # resolved (current-season-only rows), 0-prior-season new-system
            # backups.
            (2015, "NEWH", "SM"),
            (2015, "NEWA", "SN"),
        ]
    )


@pytest.fixture(scope="module")
def combined() -> pd.DataFrame:
    return derive_backup_tenure_gap_features(_combined_schedule(), _combined_rosters()).set_index(
        "game_id"
    )


# ---------------------------------------------------------------------------
# Backup-start detection, season-boundary carry, sign convention
# ---------------------------------------------------------------------------


def test_first_archived_game_has_no_proxy_and_is_never_a_backup(combined: pd.DataFrame) -> None:
    assert combined.loc["bk1", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_same_starter_across_games_is_never_a_backup_start(combined: pd.DataFrame) -> None:
    assert combined.loc["bk2", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_backup_detection_carries_across_a_season_boundary_no_reset(
    combined: pd.DataFrame,
) -> None:
    """A different starter than the team's most recent PRIOR game (even one
    season earlier) is a backup start -- home starts a NEW-system backup
    (S2 has 0 prior seasons with BKA) -> FADE home -> -1.0."""

    assert combined.loc["bk3", BACKUP_TENURE_GAP_COLUMN] == -1.0


def test_home_system_tenured_backup_is_plus_one(combined: pd.DataFrame) -> None:
    assert combined.loc["t1", BACKUP_TENURE_GAP_COLUMN] == 1.0


def test_unresolved_backup_identity_contributes_to_neither_bucket(
    combined: pd.DataFrame,
) -> None:
    assert combined.loc["u1", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_franchise_relocation_continuity_on_roster_side_counts_as_tenured(
    combined: pd.DataFrame,
) -> None:
    """SF has 2 prior seasons with LA's franchise, recorded under the OLD
    literal roster codes SL (2014) and STL (2015) -- both alias to LA."""

    assert combined.loc["r1", BACKUP_TENURE_GAP_COLUMN] == 1.0


def test_franchise_relocation_continuity_on_schedule_side_same_starter_not_backup(
    combined: pd.DataFrame,
) -> None:
    """SG started both the OAK game (2013) and the LV game (2020) -- the SAME
    franchise across a relocation -- so v1 is NOT a backup start."""

    assert combined.loc["v1", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_franchise_relocation_continuity_on_schedule_side_new_starter_is_backup(
    combined: pd.DataFrame,
) -> None:
    """SH != SG (the relocation-continuous proxy) -> backup start; SH has 0
    prior seasons with LV -> new-system -> FADE home -> -1.0."""

    assert combined.loc["v2", BACKUP_TENURE_GAP_COLUMN] == -1.0


def test_population_restriction_zeroes_an_otherwise_qualifying_flag(
    combined: pd.DataFrame,
) -> None:
    """SJ would be a system-tenured backup start (2 strictly-prior seasons),
    but the game's own season (2011) is outside the declared
    [2013, 2025] population -- forced to 0.0 regardless."""

    assert combined.loc["o1", BACKUP_TENURE_GAP_COLUMN] == 0.0
    assert BACKUP_TENURE_POPULATION_SEASON_START == 2013
    assert BACKUP_TENURE_POPULATION_SEASON_END == 2025


def test_both_sides_system_tenured_cancels_to_zero(combined: pd.DataFrame) -> None:
    assert combined.loc["bothA", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_both_sides_new_system_cancels_to_zero(combined: pd.DataFrame) -> None:
    assert combined.loc["bothB", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_every_schedule_game_id_is_present_never_dropped_never_nan(
    combined: pd.DataFrame,
) -> None:
    schedule = _combined_schedule()
    assert set(combined.index) == set(schedule["game_id"])
    assert combined[BACKUP_TENURE_GAP_COLUMN].isna().sum() == 0


def test_future_game_missing_both_starters_reads_zero(combined: pd.DataFrame) -> None:
    assert combined.loc["future", BACKUP_TENURE_GAP_COLUMN] == 0.0


def test_system_tenured_min_seasons_threshold_is_two() -> None:
    assert BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS == 2


# ---------------------------------------------------------------------------
# Team-code canonicalization
# ---------------------------------------------------------------------------


def test_canonical_team_normalizes_relocation_codes() -> None:
    codes = pd.Series(["OAK", "LV", "SD", "LAC", "STL", "SL", "LA", "ARZ", "ARI"])
    canonical = _canonical_team(codes)
    assert list(canonical) == ["LV", "LV", "LAC", "LAC", "LA", "LA", "LA", "ARI", "ARI"]


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_leakage_never_reads_current_season_roster_rows() -> None:
    """Deleting the game's-OWN-season roster row for the identified backup
    (2018, LEAK, SB in the fixture above) must never change the flag -- only
    strictly-prior-season roster rows may ever be read."""

    schedule = _combined_schedule()
    with_current_season_row = _combined_rosters()
    without_current_season_row = with_current_season_row.loc[
        ~(
            (with_current_season_row["season"] == 2018)
            & (with_current_season_row["team"] == "LEAK")
            & (with_current_season_row["gsis_id"] == "SB")
        )
    ]

    baseline = derive_backup_tenure_gap_features(schedule, with_current_season_row).set_index(
        "game_id"
    )
    after = derive_backup_tenure_gap_features(schedule, without_current_season_row).set_index(
        "game_id"
    )
    pd.testing.assert_series_equal(
        baseline[BACKUP_TENURE_GAP_COLUMN], after[BACKUP_TENURE_GAP_COLUMN]
    )
    # And, for the specific game under test, still reads new-system (-1.0),
    # not 0.0 -- proving the assertion above is not vacuously true because
    # the flag collapsed to a default.
    assert baseline.loc["l1", BACKUP_TENURE_GAP_COLUMN] == -1.0


def test_leakage_ignores_unrelated_outcome_columns() -> None:
    """``backup_tenure_gap_flag`` never reads any outcome column at all;
    mutating an unrelated result/score column must never change it."""

    schedule = _combined_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 20
    schedule["away_score"] = 17
    rosters = _combined_rosters()
    baseline = derive_backup_tenure_gap_features(schedule, rosters).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -14.0
    mutated["home_score"] = 3
    mutated["away_score"] = 41
    after = derive_backup_tenure_gap_features(mutated, rosters).set_index("game_id")
    pd.testing.assert_series_equal(
        baseline[BACKUP_TENURE_GAP_COLUMN], after[BACKUP_TENURE_GAP_COLUMN]
    )


# ---------------------------------------------------------------------------
# Population diagnostic
# ---------------------------------------------------------------------------


def test_describe_backup_tenure_population_diagnostic() -> None:
    diagnostic = describe_backup_tenure_population(_combined_schedule(), _combined_rosters())
    # Backup-start SIDES within [2013, 2025]: bk3(BKA), t1(TEN), l1(LEAK),
    # u1(UNK), r1(LA), v2(LV), bothA(TENH+TENA=2 sides), bothB(NEWH+NEWA=2
    # sides) = 6 single-side games + 2 double-side games = 6 + 4 = 10 sides,
    # 8 games. (o1/OLD is season 2011 -- outside the population.)
    assert diagnostic["n_backup_start_games_2013_2025"] == 8
    assert diagnostic["n_backup_start_sides_2013_2025"] == 10
    # System-tenured sides: t1(S4), r1(SF), bothA home(SK), bothA away(SL_) = 4.
    assert diagnostic["n_system_tenured_backup_sides"] == 4
    # New-system sides (resolved, <2 prior seasons): bk3(S2, 0 prior),
    # l1(SB, 1 prior), v2(SH, 0 prior), bothB home(SM, 0 prior), bothB away
    # (SN, 0 prior) = 5.
    assert diagnostic["n_new_system_backup_sides"] == 5
    # Unresolved: u1(SD) only = 1. (l1's SB IS resolved, just new-system.)
    assert diagnostic["n_unresolved_tenure_backup_sides"] == 1
    assert diagnostic["flagged_games_by_season"][2014] == 1  # bk3
    assert diagnostic["flagged_games_by_season"][2015] == 4  # t1, u1, bothA, bothB
    assert diagnostic["flagged_games_by_season"][2016] == 1  # r1
    assert diagnostic["flagged_games_by_season"][2018] == 1  # l1
    assert diagnostic["flagged_games_by_season"][2021] == 1  # v2


# ---------------------------------------------------------------------------
# Additive-merge / data-contract discipline
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    schedule = _combined_schedule()
    rosters = _combined_rosters()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_backup_tenure_gap_features(features, schedule=schedule, rosters=rosters)
    assert sorted(set(widened.columns) - set(features.columns)) == [BACKUP_TENURE_GAP_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_attach_requires_the_join_key() -> None:
    schedule = _combined_schedule()
    rosters = _combined_rosters()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_backup_tenure_gap_features(features, schedule=schedule, rosters=rosters)


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _combined_schedule()
    rosters = _combined_rosters()
    features = pd.DataFrame({"game_id": schedule["game_id"], BACKUP_TENURE_GAP_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=BACKUP_TENURE_GAP_COLUMN):
        attach_backup_tenure_gap_features(features, schedule=schedule, rosters=rosters)


def test_derive_requires_every_schedule_column() -> None:
    schedule = _combined_schedule().drop(columns=["home_qb_id"])
    with pytest.raises(DataContractError, match="home_qb_id"):
        derive_backup_tenure_gap_features(schedule, _combined_rosters())


def test_derive_requires_every_roster_column() -> None:
    schedule = _combined_schedule()
    rosters = _combined_rosters().drop(columns=["team"])
    with pytest.raises(DataContractError, match="team"):
        derive_backup_tenure_gap_features(schedule, rosters)


# ---------------------------------------------------------------------------
# Registered candidate profile: production plus exactly the one column
# ---------------------------------------------------------------------------


def test_registered_profile_is_production_plus_the_declared_one_column() -> None:
    baseline = set(margin_feature_columns("market_residual", btop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", btop.CANDIDATE.profile))
    assert treatment - baseline == {btop.CANDIDATE.column}
    assert baseline - treatment == set()


def test_candidate_duck_types_with_the_template_profile_identity() -> None:
    """``on_production_opener_confirmation.profile_identity`` is reused
    unmodified: our ``_Candidate`` need only carry the same
    ``family``/``profile``/``column`` attribute names."""

    columns = margin_feature_columns("market_residual", btop.CANDIDATE.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = btop.confirmation.profile_identity(btop.CANDIDATE, frame)
    assert observed["only_added_column"] == btop.CANDIDATE.column
