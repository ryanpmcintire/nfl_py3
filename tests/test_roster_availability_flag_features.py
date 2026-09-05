"""Construction, phrase-discipline, sign-convention, and leakage contracts
for the two LEAD-13/LEAD-17 roster-availability flags.

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 7". Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no PFR transaction-wire index, no snap_counts, no injuries, no
schedules snapshot is ever read from disk except via an explicit
``tmp_path`` parquet this test suite writes itself).
"""

from __future__ import annotations

import pandas as pd

from nfl_ats.roster_availability_flag_features import (
    DESIGNATE_RETURN_RE,
    IR_ACTIVATE_RE,
    IR_PLACE_RE,
    IR_RETURN_REINFORCEMENT_COLUMN,
    SPECIALIST_ABSENCE_FADE_COLUMN,
    _normalize_designate_phrasing,
    _signed_flag_from_qualifying,
    attach_ir_return_reinforcement_features,
    attach_specialist_absence_features,
    derive_ir_return_reinforcement_features,
    derive_specialist_absence_features,
    describe_ir_return_population,
    describe_specialist_population,
    designate_return_events,
    ir_activation_events,
    specialist_ir_placement_events,
    specialist_player_slugs,
    weekly_specialist_out_qualifying,
)
from nfl_ats.transaction_flag_features import distinct_player_slugs
from nfl_ats.transaction_wire_features import classify_transaction_slug

# ---------------------------------------------------------------------------
# Fixture builders (mirrors tests/test_transaction_flag_features.py)
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


_TXN_COLUMNS = ["slug", "url_year", "url_month", "category"]


def _transactions(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_TXN_COLUMNS)
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


def _injury_row(
    season: int,
    week: float,
    team: str,
    full_name: str,
    position: str,
    report_status: str | None,
    game_type: str = "REG",
) -> dict:
    return {
        "season": season,
        "week": week,
        "team": team,
        "full_name": full_name,
        "position": position,
        "report_status": report_status,
        "game_type": game_type,
    }


_INJURY_COLUMNS = ["season", "week", "team", "full_name", "position", "report_status", "game_type"]


def _injuries(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_INJURY_COLUMNS)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frozen phrase discipline: regexes and the designate-phrasing normalizer
# ---------------------------------------------------------------------------


def test_ir_activate_regex_positive_matches_both_prepositions() -> None:
    m = IR_ACTIVATE_RE.search("packers-activate-andrew-quarless-from-ir-dtr")
    assert m is not None and m.group("prefix") == "packers"
    assert m.group("player") == "andrew-quarless"

    m2 = IR_ACTIVATE_RE.search("falcons-activate-p-matt-bosher-off-ir-cut-p-ryan-allen")
    assert m2 is not None and m2.group("prefix") == "falcons"
    assert m2.group("player") == "p-matt-bosher"


def test_ir_activate_regex_compound_slug_isolates_correct_player() -> None:
    """Regression for a real, measured compound headline: one team
    activating one player from IR while separately designating a SECOND
    player for return. The activation clause must only ever capture the
    ACTIVATED player, never the designated one."""

    m = IR_ACTIVATE_RE.search(
        "falcons-activate-ol-elijah-wilkinson-from-ir-designate-ol-matt-hennessy-for-return"
    )
    assert m is not None
    assert m.group("prefix") == "falcons"
    assert m.group("player") == "ol-elijah-wilkinson"


def test_normalize_designate_phrasing_rewrites_for_ir_return_word_order() -> None:
    assert (
        _normalize_designate_phrasing("eagles-designate-te-richard-rodgers-for-ir-return")
        == "eagles-designate-te-richard-rodgers-for-return-from-ir"
    )
    assert (
        _normalize_designate_phrasing("jets-designate-leveon-bell-for-ir-return")
        == "jets-designate-leveon-bell-for-return-from-ir"
    )


def test_designate_return_regex_suffix_capture() -> None:
    normalized = _normalize_designate_phrasing("eagles-designate-te-richard-rodgers-for-ir-return")
    m = DESIGNATE_RETURN_RE.search(normalized)
    assert m is not None
    assert m.group("prefix") == "eagles"
    assert m.group("player") == "te-richard-rodgers"
    assert m.group("suffix") == "-from-ir"

    bare = DESIGNATE_RETURN_RE.search("panthers-designate-adam-thielen-for-return")
    assert bare is not None and bare.group("suffix") is None


def test_ir_place_regex_positive_and_rejects_non_place_phrasing() -> None:
    m = IR_PLACE_RE.search("bears-to-place-ls-patrick-scales-on-ir")
    assert m is not None
    assert m.group("prefix") == "bears"
    assert m.group("player") == "ls-patrick-scales"

    # "waive ... reverts to ir" is a different verb shape and must not match.
    assert IR_PLACE_RE.search("bengals-waive-p-kevin-huber-kr-brandon-wilson-reverts-to-ir") is None


# ---------------------------------------------------------------------------
# Event extraction: PUP/NFI/COVID exclusion, compound-slug clause isolation
# ---------------------------------------------------------------------------


def _full_universe_slugs() -> pd.DataFrame:
    snaps = _snaps(
        [
            _snap_row("Elijah Wilkinson", "ATL", 2022, 1, 0.6),
            _snap_row("Matt Hennessy", "ATL", 2022, 1, 0.6),
            _snap_row("Abraham Lucas", "SEA", 2022, 1, 0.6),
            _snap_row("Adam Thielen", "CAR", 2023, 1, 0.6),
        ]
    )
    return distinct_player_slugs(snaps)


def test_designate_return_events_excludes_pup_list_return() -> None:
    """Measured real-corpus trap: "designated ... for return" is the
    IDENTICAL verb phrase for a PUP-list return, which is NOT an IR
    return and must be excluded."""

    index = _transactions(
        [_txn_row("seahawks-designate-abraham-lucas-for-return-from-pup-list", 2022, 9)]
    )
    events = designate_return_events(index, _full_universe_slugs())
    assert events.empty


def test_designate_return_events_bare_suffix_is_treated_as_ir() -> None:
    index = _transactions([_txn_row("panthers-designate-adam-thielen-for-return", 2023, 10)])
    events = designate_return_events(index, _full_universe_slugs())
    assert events["player"].tolist() == ["Adam Thielen"]
    assert events["team"].tolist() == ["CAR"]


def test_ir_activate_events_compound_slug_never_misattributes() -> None:
    """The activation event must resolve Wilkinson, never Hennessy; a
    SEPARATE designate-return event resolves Hennessy from the same slug."""

    index = _transactions(
        [
            _txn_row(
                "falcons-activate-ol-elijah-wilkinson-from-ir-designate-ol-matt-hennessy-for-return",
                2022,
                11,
            )
        ]
    )
    universe = _full_universe_slugs()
    activated = ir_activation_events(index, universe)
    designated = designate_return_events(index, universe)
    assert activated["player"].tolist() == ["Elijah Wilkinson"]
    assert activated["team"].tolist() == ["ATL"]
    assert designated["player"].tolist() == ["Matt Hennessy"]
    assert designated["team"].tolist() == ["ATL"]


def test_specialist_ir_placement_events_lsp_restricted_universe() -> None:
    """A whole-slug scan over the FULL player universe would wrongly
    attribute a compound "place PLAYER-A on IR, claim PLAYER-B" headline's
    unrelated claimed player if he happened to be a known name; restricting
    the search to the LS/P-only universe prevents that AND correctly
    excludes a non-specialist placed player."""

    injuries = _injuries(
        [
            _injury_row(2020, 3.0, "SF", "Arik Armstead", "DE", None),
            _injury_row(2020, 3.0, "KC", "Chris Jones", "DT", None),
            _injury_row(2020, 2.0, "CAR", "Andy Lee", "P", "Questionable"),
        ]
    )
    lsp_slugs = specialist_player_slugs(injuries)
    index = _transactions([_txn_row("49ers-place-arik-armstead-on-ir-claim-chris-jones", 2020, 10)])
    events = specialist_ir_placement_events(index, lsp_slugs)
    assert events.empty  # neither armstead nor jones is a specialist

    index2 = _transactions(
        [_txn_row("panthers-place-andy-lee-on-ir-sign-michael-palardy", 2020, 10)]
    )
    events2 = specialist_ir_placement_events(index2, lsp_slugs)
    assert events2["player"].tolist() == ["Andy Lee"]
    assert events2["team"].tolist() == ["CAR"]


def test_specialist_player_slugs_restricted_to_ls_and_p() -> None:
    injuries = _injuries(
        [
            _injury_row(2020, 1.0, "NO", "Some Wideout", "WR", "Questionable"),
            _injury_row(2020, 1.0, "NO", "Some Snapper", "LS", "Out"),
            _injury_row(2020, 1.0, "NO", "Some Punter", "P", "Out"),
        ]
    )
    slugs = specialist_player_slugs(injuries)
    assert set(slugs["player"]) == {"Some Snapper", "Some Punter"}


# ---------------------------------------------------------------------------
# LEAD-13: IR-return reinforcement bump
# ---------------------------------------------------------------------------


def _ir_return_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("w1", 2025, 1, "2025-09-07", "OPP1", "WAS"),
            _game("w2", 2025, 2, "2025-09-14", "WAS", "OPP2"),
            _game("w3", 2025, 3, "2025-09-21", "OPP3", "WAS"),
            _game("w4", 2025, 4, "2025-09-28", "WAS", "OPP4"),
            _game("w5", 2025, 5, "2025-10-05", "WAS", "OPP5"),  # home WAS -> +1
            _game("w6", 2025, 6, "2025-10-12", "OPP6", "WAS"),  # away WAS -> -1
            _game("w7", 2025, 7, "2025-10-19", "WAS", "OPP7"),  # home WAS -> +1
            _game("w8", 2025, 8, "2025-10-26", "OPP8", "WAS"),  # away WAS -> -1
        ]
    )


def _ir_return_snap_counts(share: float = 0.6) -> pd.DataFrame:
    # Starter through weeks 1-4 ("before going on IR"); absent thereafter.
    return _snaps([_snap_row("Fake Return", "WAS", 2025, week, share) for week in range(1, 5)])


def test_ir_return_reinforcement_sign_convention_and_week_window() -> None:
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _ir_return_snap_counts()
    ).set_index("game_id")

    assert derived.loc["w5", IR_RETURN_REINFORCEMENT_COLUMN] == 1.0  # home return -> BACK home
    assert derived.loc["w6", IR_RETURN_REINFORCEMENT_COLUMN] == -1.0  # away return -> BACK away
    assert derived.loc["w7", IR_RETURN_REINFORCEMENT_COLUMN] == 1.0
    assert derived.loc["w8", IR_RETURN_REINFORCEMENT_COLUMN] == -1.0
    # Outside the weeks 5-8 window -> never flagged, even though the player
    # was a confirmed starter in those very weeks.
    assert derived.loc["w1", IR_RETURN_REINFORCEMENT_COLUMN] == 0.0
    assert derived.loc["w4", IR_RETURN_REINFORCEMENT_COLUMN] == 0.0


def test_ir_return_reinforcement_starter_gate_below_threshold() -> None:
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    low_share = _ir_return_snap_counts(share=0.3)
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, low_share
    ).set_index("game_id")
    assert (derived[IR_RETURN_REINFORCEMENT_COLUMN] == 0.0).all()


def test_ir_return_reinforcement_no_prior_snap_history_never_guessed() -> None:
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _snaps([])
    ).set_index("game_id")
    assert (derived[IR_RETURN_REINFORCEMENT_COLUMN] == 0.0).all()


def test_ir_return_reinforcement_leakage_guard() -> None:
    """A report whose latest-possible date is NOT strictly before a week
    5-8 game's own kickoff must not flag that game, even though the player
    is otherwise a confirmed returning starter."""

    # October report: month-end (Oct 31) is AFTER every week 5-8 kickoff
    # below (Oct 5 - Oct 26), so none may flag.
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 10)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _ir_return_snap_counts()
    ).set_index("game_id")
    assert (derived[IR_RETURN_REINFORCEMENT_COLUMN] == 0.0).all()


def test_ir_return_reinforcement_dedupes_designated_then_activated() -> None:
    """A player both designated and later activated in the same season
    must contribute exactly one event (the earliest report), not double
    weight."""

    index = _transactions(
        [
            _txn_row("commanders-designate-fake-return-for-return-from-ir", 2025, 9),
            _txn_row("commanders-activate-fake-return-from-ir", 2025, 9),
        ]
    )
    diag = describe_ir_return_population(index, _ir_return_snap_counts())
    assert diag["n_resolved_deduplicated_events"] == 1


def test_attach_ir_return_reinforcement_features_additive() -> None:
    features = pd.DataFrame(
        {"game_id": [f"w{i}" for i in range(1, 9)], "existing": list(range(1, 9))}
    )
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    merged = attach_ir_return_reinforcement_features(
        features,
        schedule=_ir_return_schedule(),
        transactions_index=index,
        snap_counts=_ir_return_snap_counts(),
    )
    assert merged["existing"].tolist() == list(range(1, 9))
    assert IR_RETURN_REINFORCEMENT_COLUMN in merged.columns


# ---------------------------------------------------------------------------
# LEAD-17: specialist absence fade
# ---------------------------------------------------------------------------


def test_weekly_specialist_out_qualifying_position_status_and_season_gates() -> None:
    injuries = _injuries(
        [
            _injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out"),  # qualifies
            _injury_row(2020, 4.0, "NO", "Some Punter", "P", "Questionable"),  # wrong status
            _injury_row(2020, 5.0, "NO", "Some Wideout", "WR", "Out"),  # wrong position
            _injury_row(2020, 6.0, "NO", "Some LS", "LS", "Out", game_type="WC"),  # not REG
            _injury_row(2025, 3.0, "NO", "Future LS", "LS", "Out"),  # season > 2024
        ]
    )
    qualifying = weekly_specialist_out_qualifying(injuries)
    assert list(zip(qualifying["season"], qualifying["week"], qualifying["team"], strict=True)) == [
        (2020, 3, "NO")
    ]


def test_specialist_absence_fade_weekly_out_sign_convention() -> None:
    injuries = _injuries([_injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out")])
    schedule = _schedule(
        [
            _game("g_away", 2020, 3, "2020-09-27", "OPP", "NO"),  # NO away -> +1
            _game("g_home", 2020, 3, "2020-09-27", "NO", "OPP2"),  # NO home -> -1
        ]
    )
    derived = derive_specialist_absence_features(schedule, _transactions([]), injuries).set_index(
        "game_id"
    )
    assert derived.loc["g_away", SPECIALIST_ABSENCE_FADE_COLUMN] == 1.0
    assert derived.loc["g_home", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0


def _specialist_wire_injuries() -> pd.DataFrame:
    # Only seeds the LS/P name universe -- no weekly report rows needed.
    return _injuries([_injury_row(2021, 1.0, "LV", "Fake Snapper", "LS", None)])


def _specialist_wire_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("s_jan", 2021, 1, "2021-01-05", "LV", "OPP1"),  # before placement -> 0
            _game("s_apr", 2021, 4, "2021-04-10", "LV", "OPP2"),  # after placement -> flagged
            _game("s_jul", 2021, 7, "2021-07-10", "LV", "OPP3"),  # after activation -> not flagged
        ]
    )


def test_specialist_absence_fade_wire_placement_window_open_ended() -> None:
    """No activation event confirmed -> the placement window stays open
    through the rest of the season."""

    index = _transactions([_txn_row("raiders-place-ls-fake-snapper-on-ir", 2021, 3)])
    derived = derive_specialist_absence_features(
        _specialist_wire_schedule(), index, _specialist_wire_injuries()
    ).set_index("game_id")
    assert derived.loc["s_jan", SPECIALIST_ABSENCE_FADE_COLUMN] == 0.0
    assert derived.loc["s_apr", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0  # LV home missing
    assert derived.loc["s_jul", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0  # still open


def test_specialist_absence_fade_wire_placement_window_closed_by_activation() -> None:
    index = _transactions(
        [
            _txn_row("raiders-place-ls-fake-snapper-on-ir", 2021, 3),
            _txn_row("raiders-activate-ls-fake-snapper-from-ir", 2021, 6),
        ]
    )
    derived = derive_specialist_absence_features(
        _specialist_wire_schedule(), index, _specialist_wire_injuries()
    ).set_index("game_id")
    assert derived.loc["s_apr", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0  # still out
    assert derived.loc["s_jul", SPECIALIST_ABSENCE_FADE_COLUMN] == 0.0  # closed by activation


def test_specialist_absence_fade_season_cutoff_2024() -> None:
    injuries = _injuries([_injury_row(2025, 3.0, "NO", "Some Punter", "P", "Out")])
    schedule = _schedule([_game("g25", 2025, 3, "2025-09-27", "OPP", "NO")])
    derived = derive_specialist_absence_features(schedule, _transactions([]), injuries).set_index(
        "game_id"
    )
    assert derived.loc["g25", SPECIALIST_ABSENCE_FADE_COLUMN] == 0.0


def test_attach_specialist_absence_features_additive() -> None:
    injuries = _injuries([_injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out")])
    schedule = _schedule([_game("g_away", 2020, 3, "2020-09-27", "OPP", "NO")])
    features = pd.DataFrame({"game_id": ["g_away"], "existing": [7]})
    merged = attach_specialist_absence_features(
        features, schedule=schedule, transactions_index=_transactions([]), injuries=injuries
    )
    assert merged["existing"].tolist() == [7]
    assert merged.loc[0, SPECIALIST_ABSENCE_FADE_COLUMN] == 1.0


def test_describe_specialist_population_diagnostic() -> None:
    injuries = _injuries(
        [
            _injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out"),
            _injury_row(2021, 1.0, "LV", "Fake Snapper", "LS", None),
        ]
    )
    index = _transactions([_txn_row("raiders-place-ls-fake-snapper-on-ir", 2021, 3)])
    diag = describe_specialist_population(injuries, index)
    assert diag["n_weekly_out_team_weeks"] == 1
    assert diag["n_resolved_ir_placement_events"] == 1


# ---------------------------------------------------------------------------
# Shared sign-convention helper
# ---------------------------------------------------------------------------


def test_signed_flag_from_qualifying_sign_convention() -> None:
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
    derived = _signed_flag_from_qualifying(schedule, qualifying, "flag").set_index("game_id")
    assert derived.loc["g_away", "flag"] == 1.0
    assert derived.loc["g_home", "flag"] == -1.0
    assert derived.loc["g_both", "flag"] == 0.0
    assert derived.loc["g_neither", "flag"] == 0.0
