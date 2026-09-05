"""Construction, sign-convention, semantic-trap regression, and leakage
contracts for the three LEAD-12/23/14 transaction-wire flags.

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 6". Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no PFR transaction-wire index, no snap_counts, no schedules
snapshot is ever read from disk except via an explicit ``tmp_path`` parquet
this test suite writes itself).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.transaction_flag_features import (
    ACQUISITION_RE,
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    DRAFT_PICK_RE,
    HOLDOUT_END_RE,
    HOLDOUT_SLOW_START_COLUMN,
    REINSTATED_RE,
    SPECULATIVE_ACQUISITION_RE,
    SUSPENSION_RETURN_RUST_COLUMN,
    _attach,
    _attach_qualifying_sides,
    _confirm_player_team,
    confirmed_acquisition_transactions,
    default_transactions_index,
    derive_deadline_integration_drag_features,
    derive_holdout_slow_start_features,
    derive_suspension_return_rust_features,
    describe_deadline_acquisition_population,
    describe_holdout_population,
    describe_suspension_return_population,
    distinct_player_slugs,
    find_player_in_segment,
    holdout_ending_transactions,
    suspension_category_transactions,
)
from nfl_ats.transaction_wire_features import classify_transaction_slug

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _game(
    game_id: str,
    season: int,
    week: int,
    gameday: str,
    home: str,
    away: str,
    game_type: str = "REG",
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": game_type,
        "gameday": gameday,
        "home_team": home,
        "away_team": away,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _txn_row(slug: str, year: int, month: int) -> dict:
    return {
        "slug": slug,
        "url_year": float(year),
        "url_month": float(month),
        "category": classify_transaction_slug(slug),
    }


def _transactions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _snap_row(player: str, team: str, season: int, week: int, share: float) -> dict:
    return {
        "player": player,
        "team": team,
        "season": season,
        "week": week,
        "offense_pct": share,
        "defense_pct": 0.0,
        "snap_share": share,
    }


_SNAP_COLUMNS = ["player", "team", "season", "week", "offense_pct", "defense_pct", "snap_share"]


def _snaps(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_SNAP_COLUMNS)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared: player-name substring matching
# ---------------------------------------------------------------------------


def test_distinct_player_slugs_sorted_longest_first_and_deduped() -> None:
    snaps = _snaps(
        [
            _snap_row("Ryan Fake", "AAA", 2020, 1, 0.5),
            _snap_row("Bryant Longname", "AAA", 2020, 1, 0.5),
            _snap_row("Ryan Fake", "AAA", 2020, 2, 0.6),  # duplicate name
        ]
    )
    slugs = distinct_player_slugs(snaps)
    assert list(slugs["player"]) == ["Bryant Longname", "Ryan Fake"]
    assert slugs["name_slug"].tolist() == ["bryant-longname", "ryan-fake"]


def test_find_player_in_segment_is_token_anchored_not_substring() -> None:
    """A short name must never match across a token boundary (e.g. "ryan"
    must not match inside "bryant")."""

    snaps = _snaps([_snap_row("Ryan Fake", "AAA", 2020, 1, 0.5)])
    slugs = distinct_player_slugs(snaps)
    assert find_player_in_segment("bryant-longname-signs-extension", slugs) is None
    assert find_player_in_segment("commanders-ryan-fake-reports-to-camp", slugs) == "Ryan Fake"


def test_confirm_player_team_gate() -> None:
    snaps = _snaps([_snap_row("Terry Mclaurin", "WAS", 2024, 17, 0.8)])
    assert _confirm_player_team("Terry Mclaurin", "WAS", snaps)
    assert not _confirm_player_team("Terry Mclaurin", "DAL", snaps)
    assert not _confirm_player_team("Nobody Here", "WAS", snaps)


# ---------------------------------------------------------------------------
# Shared: additive-merge helper
# ---------------------------------------------------------------------------


def test_attach_qualifying_sides_empty_population_is_all_zero() -> None:
    schedule = _schedule([_game("g1", 2020, 1, "2020-09-13", "AAA", "BBB")])
    derived = _attach_qualifying_sides(
        schedule, pd.DataFrame(columns=["season", "week", "team"]), "some_flag"
    )
    assert derived.set_index("game_id").loc["g1", "some_flag"] == 0.0


def test_attach_qualifying_sides_sign_convention() -> None:
    schedule = _schedule(
        [
            _game("g_away", 2020, 2, "2020-09-20", "HHH", "AAA"),  # AAA away qualifies
            _game("g_home", 2020, 2, "2020-09-20", "AAA", "ZZZ"),  # AAA home qualifies
            _game("g_both", 2020, 2, "2020-09-20", "AAA", "BBB"),  # both qualify
            _game("g_neither", 2020, 2, "2020-09-20", "CCC", "DDD"),
        ]
    )
    qualifying = pd.DataFrame(
        [
            {"season": 2020, "week": 2, "team": "AAA"},
            {"season": 2020, "week": 2, "team": "BBB"},
        ]
    )
    derived = _attach_qualifying_sides(schedule, qualifying, "flag").set_index("game_id")
    assert derived.loc["g_away", "flag"] == 1.0
    assert derived.loc["g_home", "flag"] == -1.0
    assert derived.loc["g_both", "flag"] == 0.0
    assert derived.loc["g_neither", "flag"] == 0.0


def test_attach_rejects_missing_join_key_and_collision() -> None:
    derived = pd.DataFrame({"game_id": ["g1"], "flag": [1.0]})
    with pytest.raises(DataContractError):
        _attach(pd.DataFrame({"not_game_id": ["g1"]}), derived, "flag")
    with pytest.raises(DataContractError):
        _attach(pd.DataFrame({"game_id": ["g1"], "flag": [0.0]}), derived, "flag")


def test_attach_preserves_existing_columns_bit_identical() -> None:
    features = pd.DataFrame({"game_id": ["g1", "g2"], "existing": [1.0, 2.0]})
    derived = pd.DataFrame({"game_id": ["g1", "g2"], "flag": [1.0, -1.0]})
    merged = _attach(features, derived, "flag")
    assert merged["existing"].tolist() == [1.0, 2.0]
    assert merged["flag"].tolist() == [1.0, -1.0]
    assert len(merged) == len(features)


# ---------------------------------------------------------------------------
# Retrospective-post exclusion (shared loader)
# ---------------------------------------------------------------------------


def test_default_transactions_index_excludes_retrospective_posts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "slug": "this-date-in-transactions-history-chargers-melvin-gordon-ends-holdout",
                "transaction_relevant": True,
                "url_year": "2021",
                "url_month": "09",
            },
            {
                "slug": "commanders-wr-terry-mclaurin-reports-to-camp-no-extension-in-place",
                "transaction_relevant": True,
                "url_year": "2025",
                "url_month": "07",
            },
            {
                "slug": "irrelevant-roundup",
                "transaction_relevant": False,
                "url_year": "2020",
                "url_month": "01",
            },
        ]
    )
    path = tmp_path / "index.parquet"
    frame.to_parquet(path, index=False)

    result = default_transactions_index(snapshot=path)
    assert len(result) == 1
    assert result.iloc[0]["slug"] == (
        "commanders-wr-terry-mclaurin-reports-to-camp-no-extension-in-place"
    )
    assert "category" in result.columns


# ---------------------------------------------------------------------------
# LEAD-12: holdout slow-start fade
# ---------------------------------------------------------------------------


def test_holdout_end_regex_positive_matches() -> None:
    for slug in (
        "commanders-wr-terry-mclaurin-reports-to-camp-no-extension-in-place",
        "some-player-reported-to-camp-late",
        "some-player-ends-holdout-signs-deal",
        "another-player-ended-holdout-yesterday",
    ):
        assert HOLDOUT_END_RE.search(slug) is not None, slug


def test_holdout_end_regex_rejects_measured_false_positives() -> None:
    """Regression for two real semantic traps measured against the PFR
    corpus: naive substring matching would wrongly fire on both."""

    # "extended" contains "ended" as a raw substring; hyphen-anchoring must
    # reject it since there is no hyphen immediately before "ended" here.
    assert HOLDOUT_END_RE.search("chiefs-dt-chris-jones-hints-at-extended-holdout") is None
    # Bare infinitive "report-to-camp" (a PREDICTION) must not match; only
    # "reports-to-camp"/"reported-to-camp" (confirmed tense) may.
    assert (
        HOLDOUT_END_RE.search(
            "jamal-adams-seahawks-not-close-at-all-on-extension-adams-expected-to-report-to-camp"
        )
        is None
    )


def test_holdout_ending_transactions_filters_correctly() -> None:
    index = _transactions(
        [
            _txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7),
            _txn_row("chiefs-dt-chris-jones-hints-at-extended-holdout", 2023, 8),
            _txn_row("some-team-signs-a-free-agent", 2021, 3),
        ]
    )
    rows = holdout_ending_transactions(index)
    assert rows["slug"].tolist() == ["commanders-wr-terry-mclaurin-reports-to-camp-x"]


def _holdout_snap_counts() -> pd.DataFrame:
    return _snaps(
        [
            # Week-1 fallback: prior season (2024) last recorded week, high share.
            _snap_row("Terry Mclaurin", "WAS", 2024, 17, 0.70),
            # In-season weeks: week W's row is checked as "prior week" for W+1.
            _snap_row("Terry Mclaurin", "WAS", 2025, 1, 0.65),
            _snap_row("Terry Mclaurin", "WAS", 2025, 2, 0.55),
            # Week 3 share drops below threshold -> week 4 should NOT qualify.
            _snap_row("Terry Mclaurin", "WAS", 2025, 3, 0.40),
        ]
    )


def _holdout_schedule(week1_gameday: str = "2025-09-07") -> pd.DataFrame:
    # WAS alternates home/away across weeks 1-4 to exercise the sign convention.
    return _schedule(
        [
            _game("h1", 2025, 1, week1_gameday, "WAS", "OPP1"),
            _game("h2", 2025, 2, "2025-09-14", "OPP2", "WAS"),
            _game("h3", 2025, 3, "2025-09-21", "WAS", "OPP3"),
            _game("h4", 2025, 4, "2025-09-28", "OPP4", "WAS"),
        ]
    )


def test_holdout_slow_start_started_rule_and_sign_convention() -> None:
    index = _transactions([_txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7)])
    derived = derive_holdout_slow_start_features(
        _holdout_schedule(), index, _holdout_snap_counts()
    ).set_index("game_id")

    # Week 1: home WAS qualifies via prior-season (2024) fallback (0.70 >= 0.5) -> -1.
    assert derived.loc["h1", HOLDOUT_SLOW_START_COLUMN] == -1.0
    # Week 2: away WAS qualifies via week-1 share 0.65 >= 0.5 -> +1.
    assert derived.loc["h2", HOLDOUT_SLOW_START_COLUMN] == 1.0
    # Week 3: home WAS qualifies via week-2 share 0.55 >= 0.5 -> -1.
    assert derived.loc["h3", HOLDOUT_SLOW_START_COLUMN] == -1.0
    # Week 4: week-3 share is 0.40 (< 0.5) -> does not qualify -> 0.
    assert derived.loc["h4", HOLDOUT_SLOW_START_COLUMN] == 0.0


def test_holdout_slow_start_unresolved_snap_history_never_guessed() -> None:
    index = _transactions([_txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7)])
    empty_snaps = _snaps([])
    derived = derive_holdout_slow_start_features(_holdout_schedule(), index, empty_snaps).set_index(
        "game_id"
    )
    assert (derived[HOLDOUT_SLOW_START_COLUMN] == 0.0).all()


def test_holdout_slow_start_leakage_guard_per_week() -> None:
    """A report whose latest-possible date is NOT strictly before a given
    week's own kickoff must not flag that week, even though it may still
    flag a later week."""

    # Report is late (September) so its month-end (Sep 30) is not before
    # week 1/2's kickoff (both in September) but IS before week 3/4's
    # kickoff (moved to October here specifically to exercise the guard).
    late_index = _transactions(
        [_txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 9)]
    )
    schedule = _schedule(
        [
            _game("h1", 2025, 1, "2025-09-07", "WAS", "OPP1"),
            _game("h2", 2025, 2, "2025-09-14", "OPP2", "WAS"),
            _game("h3", 2025, 3, "2025-10-05", "WAS", "OPP3"),
            _game("h4", 2025, 4, "2025-10-12", "OPP4", "WAS"),
        ]
    )
    derived = derive_holdout_slow_start_features(
        schedule, late_index, _holdout_snap_counts()
    ).set_index("game_id")
    assert derived.loc["h1", HOLDOUT_SLOW_START_COLUMN] == 0.0  # leakage-guarded out
    assert derived.loc["h2", HOLDOUT_SLOW_START_COLUMN] == 0.0  # leakage-guarded out
    assert derived.loc["h3", HOLDOUT_SLOW_START_COLUMN] == -1.0  # still qualifies


def test_holdout_slow_start_only_one_team_resolution_required() -> None:
    """A slug matching zero or more than one team is never guessed."""

    index = _transactions([_txn_row("bills-and-jets-terry-mclaurin-reports-to-camp-x", 2025, 7)])
    derived = derive_holdout_slow_start_features(
        _holdout_schedule(), index, _holdout_snap_counts()
    ).set_index("game_id")
    assert (derived[HOLDOUT_SLOW_START_COLUMN] == 0.0).all()


def test_attach_holdout_slow_start_features_additive() -> None:
    features = pd.DataFrame({"game_id": ["h1", "h2", "h3", "h4"], "existing": [1, 2, 3, 4]})
    index = _transactions([_txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7)])
    from nfl_ats.transaction_flag_features import attach_holdout_slow_start_features

    merged = attach_holdout_slow_start_features(
        features,
        schedule=_holdout_schedule(),
        transactions_index=index,
        snap_counts=_holdout_snap_counts(),
    )
    assert merged["existing"].tolist() == [1, 2, 3, 4]
    assert HOLDOUT_SLOW_START_COLUMN in merged.columns


def test_describe_holdout_population_diagnostic() -> None:
    index = _transactions(
        [
            _txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7),
            _txn_row("chiefs-dt-chris-jones-hints-at-extended-holdout", 2023, 8),
        ]
    )
    diag = describe_holdout_population(index, _holdout_snap_counts())
    assert diag["n_holdout_ending_slugs"] == 1
    assert diag["n_resolved_player_and_team"] == 1


# ---------------------------------------------------------------------------
# LEAD-23: trade-deadline integration drag
# ---------------------------------------------------------------------------


def test_acquisition_regex_positive_and_exclusions() -> None:
    assert ACQUISITION_RE.search("eagles-to-acquire-desean-jackson-from-buccaneers")
    assert ACQUISITION_RE.search("patriots-acquire-brandin-cooks")
    assert DRAFT_PICK_RE.search("bills-acquire-no-23-select-cb-kaiir-elam")
    assert DRAFT_PICK_RE.search("broncos-acquire-no-42-from-bengals")
    assert SPECULATIVE_ACQUISITION_RE.search("saints-tried-to-acquire-giants-wr-darius-slayton")
    assert SPECULATIVE_ACQUISITION_RE.search(
        "packers-attempted-to-acquire-raiders-te-darren-waller-at-deadline"
    )


def test_confirmed_acquisition_transactions_filters_pick_speculative_and_window() -> None:
    index = _transactions(
        [
            _txn_row("eagles-acquire-fake-player-from-old", 2020, 10),  # confirmed, in-window
            _txn_row("bills-acquire-no-23-select-cb-kaiir-elam", 2022, 4),  # draft pick
            _txn_row("saints-tried-to-acquire-giants-wr-darius-slayton", 2019, 10),  # speculative
            _txn_row("eagles-acquire-offseason-guy-from-old", 2020, 4),  # offseason, out of window
        ]
    )
    confirmed = confirmed_acquisition_transactions(index)
    assert confirmed["slug"].tolist() == ["eagles-acquire-fake-player-from-old"]


def _deadline_snap_counts() -> pd.DataFrame:
    rows = [_snap_row("Fake Player", "OLD", 2020, week, 0.60) for week in range(1, 9)]
    return _snaps(rows)


def test_deadline_integration_drag_sign_and_window() -> None:
    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule(
        [
            _game("d9", 2020, 9, "2020-11-01", "OPP", "PHI"),  # away PHI -> +1
            _game("d10", 2020, 10, "2020-11-08", "PHI", "OPP"),  # home PHI -> -1
            _game("d11", 2020, 11, "2020-11-15", "OPP", "PHI"),  # away PHI -> +1 (3rd game)
            _game("d12", 2020, 12, "2020-11-22", "OPP", "PHI"),  # 4th game, excluded
        ]
    )
    derived = derive_deadline_integration_drag_features(
        schedule, index, _deadline_snap_counts()
    ).set_index("game_id")
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 1.0
    assert derived.loc["d10", DEADLINE_INTEGRATION_DRAG_COLUMN] == -1.0
    assert derived.loc["d11", DEADLINE_INTEGRATION_DRAG_COLUMN] == 1.0
    assert derived.loc["d12", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0  # only first 3 games


def test_deadline_integration_drag_high_snap_gate() -> None:
    """A low-snap acquisition (trailing share < 0.5) is excluded from the
    population -- never a fade candidate."""

    low_snap = _snaps([_snap_row("Fake Player", "OLD", 2020, w, 0.20) for w in range(1, 9)])
    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule([_game("d9", 2020, 9, "2020-11-01", "OPP", "PHI")])
    derived = derive_deadline_integration_drag_features(schedule, index, low_snap).set_index(
        "game_id"
    )
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def test_deadline_integration_drag_no_prior_team_history_never_guessed() -> None:
    """A player with no snap-count rows for any OTHER team this season
    cannot resolve a "previous team" and must be excluded, never guessed."""

    only_phi_team = _snaps([_snap_row("Fake Player", "PHI", 2020, w, 0.90) for w in range(1, 9)])
    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule([_game("d9", 2020, 9, "2020-11-01", "OPP", "PHI")])
    derived = derive_deadline_integration_drag_features(schedule, index, only_phi_team).set_index(
        "game_id"
    )
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def test_deadline_integration_drag_leakage_guard() -> None:
    """A flagged game must kick off strictly after the report's own
    latest-possible (month-end) date."""

    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    # Week 9 game kicks off BEFORE the October report's month-end (Oct 31) --
    # even though week (9) > last_prior_week (8), it must not be flagged.
    schedule = _schedule([_game("d9", 2020, 9, "2020-10-15", "OPP", "PHI")])
    derived = derive_deadline_integration_drag_features(
        schedule, index, _deadline_snap_counts()
    ).set_index("game_id")
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def test_describe_deadline_acquisition_population_diagnostic() -> None:
    index = _transactions(
        [
            _txn_row("eagles-acquire-fake-player-from-old", 2020, 10),
            _txn_row("bills-acquire-no-23-select-cb-kaiir-elam", 2022, 4),
        ]
    )
    diag = describe_deadline_acquisition_population(index, _deadline_snap_counts())
    assert diag["n_confirmed_acquisition_slugs"] == 1
    assert diag["n_resolved_player_and_high_snap"] == 1


# ---------------------------------------------------------------------------
# LEAD-14: suspension-return rust
# ---------------------------------------------------------------------------


def test_reinstated_regex_positive_and_semantic_trap_exclusions() -> None:
    assert REINSTATED_RE.search("aldon-smith-reinstated-suspension")
    assert REINSTATED_RE.search("broncos-dl-x-reinstated-from-gambling-suspension")
    # "reinstatement" (the noun, a petition) must never match "reinstated".
    assert REINSTATED_RE.search("josh-gordon-files-reinstatement-suspension") is None
    # "suspension reinstated" (the SUSPENSION itself reimposed) is the
    # opposite of a player returning.
    assert REINSTATED_RE.search("tom-bradys-suspension-reinstated-by-appeals-court") is None


def test_suspension_category_transactions_uses_shared_classifier() -> None:
    index = _transactions(
        [
            _txn_row("some-player-suspended-six-games", 2020, 3),
            _txn_row("some-player-signs-extension", 2020, 3),
        ]
    )
    rows = suspension_category_transactions(index)
    assert rows["slug"].tolist() == ["some-player-suspended-six-games"]


def _suspension_snap_counts() -> pd.DataFrame:
    return _snaps([_snap_row("Fake Suspend", "STL", 2019, 17, 0.55)])


def _suspension_schedule() -> pd.DataFrame:
    """STL plays one game a month, March 2020 through December 2020.

    The imposed report is month 3 (March), the reinstated report is month
    10 (October): the half-open interval ``[month 3, month 10)`` covers
    months 4-9 inclusive -- exactly 6 team games -- satisfying the >= 6
    threshold. Months 11-12 (November, December) are the "return game plus
    one" candidates (both strictly after October's own month-end).
    """

    rows = []
    for i, month in enumerate(range(3, 13)):  # months 3..12 inclusive
        gameday = f"2020-{month:02d}-15"
        home, away = ("STL", "OPP") if i % 2 == 0 else ("OPP", "STL")
        rows.append(_game(f"s{month}", 2020, i + 1, gameday, home, away))
    return _schedule(rows)


def test_suspension_return_rust_measures_duration_and_flags_return_plus_one() -> None:
    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    schedule = _suspension_schedule()
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    nonzero = derived.loc[derived[SUSPENSION_RETURN_RUST_COLUMN] != 0.0]
    assert len(nonzero) == 2  # return game plus one


def test_suspension_return_rust_below_six_games_excluded() -> None:
    """A bracket measuring fewer than 6 team games must not qualify."""

    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-one-game", 2020, 9),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    schedule = _suspension_schedule()
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    assert (derived[SUSPENSION_RETURN_RUST_COLUMN] == 0.0).all()


def test_suspension_return_rust_no_earlier_imposed_report_excluded() -> None:
    index = _transactions([_txn_row("fake-suspend-reinstated-from-suspension", 2020, 10)])
    schedule = _suspension_schedule()
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    assert (derived[SUSPENSION_RETURN_RUST_COLUMN] == 0.0).all()


def test_suspension_return_rust_unresolved_team_never_guessed() -> None:
    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    schedule = _suspension_schedule()
    derived = derive_suspension_return_rust_features(schedule, index, _snaps([])).set_index(
        "game_id"
    )
    assert (derived[SUSPENSION_RETURN_RUST_COLUMN] == 0.0).all()


def test_suspension_return_rust_leakage_guard() -> None:
    """A flagged game must kick off strictly after the reinstatement
    report's own latest-possible (month-end) date."""

    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    schedule = _schedule(
        [
            *_suspension_schedule().to_dict("records"),
            # A game BEFORE the reinstatement's own month-end must never flag.
            _game("s_early", 2020, 8, "2020-10-05", "STL", "OPP"),
        ]
    )
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    assert derived.loc["s_early", SUSPENSION_RETURN_RUST_COLUMN] == 0.0


def test_describe_suspension_return_population_diagnostic() -> None:
    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    diag = describe_suspension_return_population(
        index, _suspension_snap_counts(), _suspension_schedule()
    )
    assert diag["n_reinstatement_slugs"] == 1
    assert diag["n_resolved_6plus_game_returns"] == 1


def test_team_lookup_ignores_same_week_and_future_second_trade() -> None:
    from nfl_ats.transaction_flag_features import _team_for_player_before

    before = _snaps([_snap_row("Fake Player", "KC", 2020, 4, 0.8)])
    after = _snaps(
        [
            *before.to_dict("records"),
            _snap_row("Fake Player", "BUF", 2020, 17, 0.9),
            _snap_row("Fake Player", "PHI", 2020, 5, 0.9),
        ]
    )
    assert _team_for_player_before("Fake Player", before, 2020, 5) == "KC"
    assert _team_for_player_before("Fake Player", after, 2020, 5) == "KC"


def test_acquisition_features_ignore_future_team_and_usage() -> None:
    from nfl_ats.transaction_flag_features import _acquisition_events

    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule(
        [
            _game("d9", 2020, 9, "2020-11-01", "OPP", "PHI"),
            _game("d10", 2020, 10, "2020-11-08", "PHI", "OPP"),
        ]
    )
    before = _deadline_snap_counts()
    after = _snaps(
        [
            *before.to_dict("records"),
            _snap_row("Fake Player", "BUF", 2020, 17, 0.9),
            _snap_row("Fake Player", "OLD", 2020, 9, 0.0),
        ]
    )
    pd.testing.assert_frame_equal(
        _acquisition_events(index, before, schedule), _acquisition_events(index, after, schedule)
    )
    pd.testing.assert_frame_equal(
        derive_deadline_integration_drag_features(schedule, index, before),
        derive_deadline_integration_drag_features(schedule, index, after),
    )


def test_suspension_team_resolution_ignores_future_trade() -> None:
    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    before = _suspension_snap_counts()
    after = _snaps([*before.to_dict("records"), _snap_row("Fake Suspend", "BUF", 2020, 17, 0.9)])
    pd.testing.assert_frame_equal(
        derive_suspension_return_rust_features(_suspension_schedule(), index, before),
        derive_suspension_return_rust_features(_suspension_schedule(), index, after),
    )
