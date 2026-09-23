from __future__ import annotations

import pandas as pd

from nfl_ats.roster_availability_flag_features import (
    IR_RETURN_REINFORCEMENT_COLUMN,
    SPECIALIST_ABSENCE_FADE_COLUMN,
    attach_ir_return_reinforcement_features,
    attach_specialist_absence_features,
    derive_ir_return_reinforcement_features,
    derive_specialist_absence_features,
    describe_ir_return_population,
    designate_return_events,
    ir_activation_events,
    specialist_ir_placement_events,
    specialist_player_slugs,
    weekly_specialist_out_qualifying,
)
from nfl_ats.transaction_flag_features import distinct_player_slugs
from nfl_ats.transaction_wire_features import classify_transaction_slug


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
    assert events.empty

    index2 = _transactions(
        [_txn_row("panthers-place-andy-lee-on-ir-sign-michael-palardy", 2020, 10)]
    )
    events2 = specialist_ir_placement_events(index2, lsp_slugs)
    assert events2["player"].tolist() == ["Andy Lee"]
    assert events2["team"].tolist() == ["CAR"]


def _ir_return_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("w1", 2025, 1, "2025-09-07", "OPP1", "WAS"),
            _game("w2", 2025, 2, "2025-09-14", "WAS", "OPP2"),
            _game("w3", 2025, 3, "2025-09-21", "OPP3", "WAS"),
            _game("w4", 2025, 4, "2025-09-28", "WAS", "OPP4"),
            _game("w5", 2025, 5, "2025-10-05", "WAS", "OPP5"),
            _game("w6", 2025, 6, "2025-10-12", "OPP6", "WAS"),
            _game("w7", 2025, 7, "2025-10-19", "WAS", "OPP7"),
            _game("w8", 2025, 8, "2025-10-26", "OPP8", "WAS"),
        ]
    )


def _ir_return_snap_counts(share: float = 0.6) -> pd.DataFrame:
    return _snaps([_snap_row("Fake Return", "WAS", 2025, week, share) for week in range(1, 5)])


def test_ir_return_reinforcement_sign_convention_and_week_window() -> None:
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _ir_return_snap_counts()
    ).set_index("game_id")

    assert derived.loc["w5", IR_RETURN_REINFORCEMENT_COLUMN] == 1.0
    assert derived.loc["w6", IR_RETURN_REINFORCEMENT_COLUMN] == -1.0
    assert derived.loc["w7", IR_RETURN_REINFORCEMENT_COLUMN] == 1.0
    assert derived.loc["w8", IR_RETURN_REINFORCEMENT_COLUMN] == -1.0
    assert derived.loc["w1", IR_RETURN_REINFORCEMENT_COLUMN] == 0.0
    assert derived.loc["w4", IR_RETURN_REINFORCEMENT_COLUMN] == 0.0


def test_ir_return_reinforcement_no_prior_snap_history_never_guessed() -> None:
    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 9)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _snaps([])
    ).set_index("game_id")
    assert (derived[IR_RETURN_REINFORCEMENT_COLUMN] == 0.0).all()


def test_ir_return_reinforcement_leakage_guard() -> None:

    index = _transactions([_txn_row("commanders-activate-fake-return-from-ir", 2025, 10)])
    derived = derive_ir_return_reinforcement_features(
        _ir_return_schedule(), index, _ir_return_snap_counts()
    ).set_index("game_id")
    assert (derived[IR_RETURN_REINFORCEMENT_COLUMN] == 0.0).all()


def test_ir_return_reinforcement_dedupes_designated_then_activated() -> None:

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


def test_weekly_specialist_out_qualifying_position_status_and_season_gates() -> None:
    injuries = _injuries(
        [
            _injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out"),
            _injury_row(2020, 4.0, "NO", "Some Punter", "P", "Questionable"),
            _injury_row(2020, 5.0, "NO", "Some Wideout", "WR", "Out"),
            _injury_row(2020, 6.0, "NO", "Some LS", "LS", "Out", game_type="WC"),
            _injury_row(2025, 3.0, "NO", "Future LS", "LS", "Out"),
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
            _game("g_away", 2020, 3, "2020-09-27", "OPP", "NO"),
            _game("g_home", 2020, 3, "2020-09-27", "NO", "OPP2"),
        ]
    )
    derived = derive_specialist_absence_features(schedule, _transactions([]), injuries).set_index(
        "game_id"
    )
    assert derived.loc["g_away", SPECIALIST_ABSENCE_FADE_COLUMN] == 1.0
    assert derived.loc["g_home", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0


def _specialist_wire_injuries() -> pd.DataFrame:
    return _injuries([_injury_row(2021, 1.0, "LV", "Fake Snapper", "LS", None)])


def _specialist_wire_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("s_jan", 2021, 1, "2021-01-05", "LV", "OPP1"),
            _game("s_apr", 2021, 4, "2021-04-10", "LV", "OPP2"),
            _game("s_jul", 2021, 7, "2021-07-10", "LV", "OPP3"),
        ]
    )


def test_specialist_absence_fade_wire_placement_window_open_ended() -> None:

    index = _transactions([_txn_row("raiders-place-ls-fake-snapper-on-ir", 2021, 3)])
    derived = derive_specialist_absence_features(
        _specialist_wire_schedule(), index, _specialist_wire_injuries()
    ).set_index("game_id")
    assert derived.loc["s_jan", SPECIALIST_ABSENCE_FADE_COLUMN] == 0.0
    assert derived.loc["s_apr", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0
    assert derived.loc["s_jul", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0


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
    assert derived.loc["s_apr", SPECIALIST_ABSENCE_FADE_COLUMN] == -1.0
    assert derived.loc["s_jul", SPECIALIST_ABSENCE_FADE_COLUMN] == 0.0


def test_attach_specialist_absence_features_additive() -> None:
    injuries = _injuries([_injury_row(2020, 3.0, "NO", "Some Punter", "P", "Out")])
    schedule = _schedule([_game("g_away", 2020, 3, "2020-09-27", "OPP", "NO")])
    features = pd.DataFrame({"game_id": ["g_away"], "existing": [7]})
    merged = attach_specialist_absence_features(
        features, schedule=schedule, transactions_index=_transactions([]), injuries=injuries
    )
    assert merged["existing"].tolist() == [7]
    assert merged.loc[0, SPECIALIST_ABSENCE_FADE_COLUMN] == 1.0
