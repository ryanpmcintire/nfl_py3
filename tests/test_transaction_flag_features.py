from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.transaction_flag_features import (
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    HOLDOUT_SLOW_START_COLUMN,
    SUSPENSION_RETURN_RUST_COLUMN,
    _attach,
    confirmed_acquisition_transactions,
    derive_deadline_integration_drag_features,
    derive_holdout_slow_start_features,
    derive_suspension_return_rust_features,
)
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


def _holdout_snap_counts() -> pd.DataFrame:
    return _snaps(
        [
            _snap_row("Terry Mclaurin", "WAS", 2024, 17, 0.70),
            _snap_row("Terry Mclaurin", "WAS", 2025, 1, 0.65),
            _snap_row("Terry Mclaurin", "WAS", 2025, 2, 0.55),
            _snap_row("Terry Mclaurin", "WAS", 2025, 3, 0.40),
        ]
    )


def _holdout_schedule(week1_gameday: str = "2025-09-07") -> pd.DataFrame:
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

    assert derived.loc["h1", HOLDOUT_SLOW_START_COLUMN] == -1.0
    assert derived.loc["h2", HOLDOUT_SLOW_START_COLUMN] == 1.0
    assert derived.loc["h3", HOLDOUT_SLOW_START_COLUMN] == -1.0
    assert derived.loc["h4", HOLDOUT_SLOW_START_COLUMN] == 0.0


def test_holdout_slow_start_unresolved_snap_history_never_guessed() -> None:
    index = _transactions([_txn_row("commanders-wr-terry-mclaurin-reports-to-camp-x", 2025, 7)])
    empty_snaps = _snaps([])
    derived = derive_holdout_slow_start_features(_holdout_schedule(), index, empty_snaps).set_index(
        "game_id"
    )
    assert (derived[HOLDOUT_SLOW_START_COLUMN] == 0.0).all()


def test_holdout_slow_start_leakage_guard_per_week() -> None:

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
    assert derived.loc["h1", HOLDOUT_SLOW_START_COLUMN] == 0.0
    assert derived.loc["h2", HOLDOUT_SLOW_START_COLUMN] == 0.0
    assert derived.loc["h3", HOLDOUT_SLOW_START_COLUMN] == -1.0


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


def test_confirmed_acquisition_transactions_filters_pick_speculative_and_window() -> None:
    index = _transactions(
        [
            _txn_row("eagles-acquire-fake-player-from-old", 2020, 10),
            _txn_row("bills-acquire-no-23-select-cb-kaiir-elam", 2022, 4),
            _txn_row("saints-tried-to-acquire-giants-wr-darius-slayton", 2019, 10),
            _txn_row("eagles-acquire-offseason-guy-from-old", 2020, 4),
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
            _game("d9", 2020, 9, "2020-11-01", "OPP", "PHI"),
            _game("d10", 2020, 10, "2020-11-08", "PHI", "OPP"),
            _game("d11", 2020, 11, "2020-11-15", "OPP", "PHI"),
            _game("d12", 2020, 12, "2020-11-22", "OPP", "PHI"),
        ]
    )
    derived = derive_deadline_integration_drag_features(
        schedule, index, _deadline_snap_counts()
    ).set_index("game_id")
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 1.0
    assert derived.loc["d10", DEADLINE_INTEGRATION_DRAG_COLUMN] == -1.0
    assert derived.loc["d11", DEADLINE_INTEGRATION_DRAG_COLUMN] == 1.0
    assert derived.loc["d12", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def test_deadline_integration_drag_no_prior_team_history_never_guessed() -> None:

    only_phi_team = _snaps([_snap_row("Fake Player", "PHI", 2020, w, 0.90) for w in range(1, 9)])
    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule([_game("d9", 2020, 9, "2020-11-01", "OPP", "PHI")])
    derived = derive_deadline_integration_drag_features(schedule, index, only_phi_team).set_index(
        "game_id"
    )
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def test_deadline_integration_drag_leakage_guard() -> None:

    index = _transactions([_txn_row("eagles-acquire-fake-player-from-old", 2020, 10)])
    schedule = _schedule([_game("d9", 2020, 9, "2020-10-15", "OPP", "PHI")])
    derived = derive_deadline_integration_drag_features(
        schedule, index, _deadline_snap_counts()
    ).set_index("game_id")
    assert derived.loc["d9", DEADLINE_INTEGRATION_DRAG_COLUMN] == 0.0


def _suspension_snap_counts() -> pd.DataFrame:
    return _snaps([_snap_row("Fake Suspend", "STL", 2019, 17, 0.55)])


def _suspension_schedule() -> pd.DataFrame:

    rows = []
    for i, month in enumerate(range(3, 13)):
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
    assert len(nonzero) == 2


def test_suspension_return_rust_no_earlier_imposed_report_excluded() -> None:
    index = _transactions([_txn_row("fake-suspend-reinstated-from-suspension", 2020, 10)])
    schedule = _suspension_schedule()
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    assert (derived[SUSPENSION_RETURN_RUST_COLUMN] == 0.0).all()


def test_suspension_return_rust_leakage_guard() -> None:

    index = _transactions(
        [
            _txn_row("stl-fake-suspend-suspended-indefinitely", 2020, 3),
            _txn_row("fake-suspend-reinstated-from-suspension", 2020, 10),
        ]
    )
    schedule = _schedule(
        [
            *_suspension_schedule().to_dict("records"),
            _game("s_early", 2020, 8, "2020-10-05", "STL", "OPP"),
        ]
    )
    derived = derive_suspension_return_rust_features(
        schedule, index, _suspension_snap_counts()
    ).set_index("game_id")
    assert derived.loc["s_early", SUSPENSION_RETURN_RUST_COLUMN] == 0.0


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
