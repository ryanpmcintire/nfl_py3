"""Construction, sign-convention and leakage contracts for the three Wave 4
PBP coaching-trait candidates (LEAD-26 ``opening_drive_epa``, LEAD-27
``q3_point_diff``, LEAD-30 ``fourth_down_interaction``).

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 4". Every fixture is
built in memory: these tests must pass in a fresh clone with no local PBP or
market snapshots (no play-by-play parquet and no opener store is ever read).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.margin import margin_feature_columns
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.pbp_trait_on_production_features import (
    FOURTH_DOWN_INTERACTION_COLUMN,
    OPENING_DRIVE_EPA_COLUMN,
    Q3_POINT_DIFF_COLUMN,
    attach_fourth_down_interaction_features,
    attach_opening_drive_epa_features,
    attach_q3_point_diff_features,
    derive_fourth_down_interaction_features,
    derive_opening_drive_epa_features,
    derive_q3_point_diff_features,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
    row.update(
        {
            "season_type": "REG",
            "down": np.nan,
            "ydstogo": np.nan,
            "yardline_100": 50,
            "qtr": 1,
            "play_type": "pass",
            "pass_attempt": 1,
            "rush_attempt": 0,
            "qb_kneel": 0,
            "qb_spike": 0,
            "aborted_play": 0,
            "wp": 0.5,
            "score_differential": 0,
            "fixed_drive_result": "Punt",
            "posteam_score": 0,
            "posteam_score_post": 0,
            "play": 1,
        }
    )
    row.update(overrides)
    return row


def _features(games: list[tuple[str, int, int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": gid, "season": season, "week": week, "home_team": home, "away_team": away}
            for gid, season, week, home, away in games
        ]
    )


# ---------------------------------------------------------------------------
# LEAD-26: opening-drive EPA differential
# ---------------------------------------------------------------------------


def _opening_drive_pbp() -> pd.DataFrame:
    def _drive(game_id, season, week, home, away, home_epa, away_epa, home_play_id_base):
        return [
            _row(
                game_id=game_id,
                season=season,
                week=week,
                home_team=home,
                away_team=away,
                posteam=home,
                defteam=away,
                fixed_drive=1,
                play_id=home_play_id_base,
                epa=home_epa,
            ),
            _row(
                game_id=game_id,
                season=season,
                week=week,
                home_team=home,
                away_team=away,
                posteam=away,
                defteam=home,
                fixed_drive=2,
                play_id=home_play_id_base + 1,
                epa=away_epa,
            ),
        ]

    rows: list[dict[str, object]] = []
    rows += _drive("G1", 2022, 1, "AAA", "BBB", 1.0, 0.5, 1)
    rows += _drive("G2", 2022, 1, "CCC", "DDD", 2.0, 0.2, 11)
    rows += _drive("G3", 2022, 2, "AAA", "CCC", 3.0, 4.0, 21)
    rows += _drive("G4", 2022, 3, "AAA", "BBB", 9.0, 9.0, 31)
    return pd.DataFrame(rows)


def _opening_drive_features() -> pd.DataFrame:
    return _features(
        [
            ("G1", 2022, 1, "AAA", "BBB"),
            ("G2", 2022, 1, "CCC", "DDD"),
            ("G3", 2022, 2, "AAA", "CCC"),
            ("G4", 2022, 3, "AAA", "BBB"),
        ]
    )


def test_opening_drive_epa_nan_before_either_side_has_history() -> None:
    derived = derive_opening_drive_epa_features(
        _opening_drive_features(), pbp=_opening_drive_pbp()
    ).set_index("game_id")
    assert pd.isna(derived.loc["G1", OPENING_DRIVE_EPA_COLUMN])
    assert pd.isna(derived.loc["G2", OPENING_DRIVE_EPA_COLUMN])


def test_opening_drive_epa_differential_sign_and_magnitude() -> None:
    derived = derive_opening_drive_epa_features(
        _opening_drive_features(), pbp=_opening_drive_pbp()
    ).set_index("game_id")
    # AAA's rolling entering G3 = 1.0 (its only prior game, G1); CCC's rolling
    # entering G3 = 2.0 (its only prior game, G2). home - away = 1.0 - 2.0.
    assert derived.loc["G3", OPENING_DRIVE_EPA_COLUMN] == pytest.approx(-1.0)
    # AAA's rolling entering G4 = mean(1.0 [G1], 3.0 [G3]) = 2.0; BBB's rolling
    # entering G4 = 0.5 (its only prior game, G1, as the away side).
    assert derived.loc["G4", OPENING_DRIVE_EPA_COLUMN] == pytest.approx(1.5)


def test_opening_drive_epa_leakage() -> None:
    """Mutating G3's own opening-drive EPA must never change G3's own
    (strictly-prior) differential, but legitimately changes G4's (AAA played
    in both, and G4 rolls up G3's own outcome as history)."""

    features = _opening_drive_features()
    baseline = derive_opening_drive_epa_features(features, pbp=_opening_drive_pbp()).set_index(
        "game_id"
    )

    mutated_pbp = _opening_drive_pbp()
    mutated_pbp.loc[(mutated_pbp["game_id"] == "G3") & (mutated_pbp["posteam"] == "AAA"), "epa"] = (
        30.0
    )
    mutated = derive_opening_drive_epa_features(features, pbp=mutated_pbp).set_index("game_id")

    assert mutated.loc["G3", OPENING_DRIVE_EPA_COLUMN] == pytest.approx(
        baseline.loc["G3", OPENING_DRIVE_EPA_COLUMN]
    )
    assert mutated.loc["G4", OPENING_DRIVE_EPA_COLUMN] != pytest.approx(
        baseline.loc["G4", OPENING_DRIVE_EPA_COLUMN]
    )
    # AAA's rolling entering G4 becomes mean(1.0, 30.0) = 15.5; BBB stays 0.5.
    assert mutated.loc["G4", OPENING_DRIVE_EPA_COLUMN] == pytest.approx(15.0)


def test_opening_drive_epa_attach_is_additive() -> None:
    features = pd.DataFrame(
        {"game_id": _opening_drive_features()["game_id"], "some_existing_feature": 1.0}
    )
    features = features.assign(
        season=_opening_drive_features()["season"].to_numpy(),
        week=_opening_drive_features()["week"].to_numpy(),
        home_team=_opening_drive_features()["home_team"].to_numpy(),
        away_team=_opening_drive_features()["away_team"].to_numpy(),
    )
    widened = attach_opening_drive_epa_features(features, pbp=_opening_drive_pbp())
    assert sorted(set(widened.columns) - set(features.columns)) == [OPENING_DRIVE_EPA_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_opening_drive_epa_attach_refuses_to_overwrite_an_existing_column() -> None:
    features = _opening_drive_features().assign(**{OPENING_DRIVE_EPA_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=OPENING_DRIVE_EPA_COLUMN):
        attach_opening_drive_epa_features(features, pbp=_opening_drive_pbp())


def test_opening_drive_epa_requires_feature_columns() -> None:
    features = _opening_drive_features().drop(columns=["home_team"])
    with pytest.raises(DataContractError, match="home_team"):
        derive_opening_drive_epa_features(features, pbp=_opening_drive_pbp())


# ---------------------------------------------------------------------------
# LEAD-27: third-quarter point-differential differential
# ---------------------------------------------------------------------------


def _q3_pbp() -> pd.DataFrame:
    def _game_rows(game_id, season, week, home, away, q3_lead, q4_lead, play_id_base):
        return [
            _row(
                game_id=game_id,
                season=season,
                week=week,
                home_team=home,
                away_team=away,
                posteam=home,
                defteam=away,
                qtr=3,
                play_id=play_id_base,
                score_differential=q3_lead,
            ),
            _row(
                game_id=game_id,
                season=season,
                week=week,
                home_team=home,
                away_team=away,
                posteam=home,
                defteam=away,
                qtr=4,
                play_id=play_id_base + 1,
                score_differential=q4_lead,
            ),
        ]

    rows: list[dict[str, object]] = []
    rows += _game_rows("G1", 2022, 1, "AAA", "BBB", 2, 5, 1)  # AAA q3_point_diff = 3
    rows += _game_rows("G2", 2022, 1, "CCC", "DDD", 0, -4, 11)  # CCC q3_point_diff = -4
    rows += _game_rows("G3", 2022, 2, "AAA", "CCC", 1, 1, 21)  # AAA(home) own diff this game = 0
    rows += _game_rows("G4", 2022, 3, "AAA", "BBB", 0, 0, 31)
    return pd.DataFrame(rows)


def _q3_features() -> pd.DataFrame:
    return _features(
        [
            ("G1", 2022, 1, "AAA", "BBB"),
            ("G2", 2022, 1, "CCC", "DDD"),
            ("G3", 2022, 2, "AAA", "CCC"),
            ("G4", 2022, 3, "AAA", "BBB"),
        ]
    )


def test_q3_point_diff_nan_before_either_side_has_history() -> None:
    derived = derive_q3_point_diff_features(_q3_features(), pbp=_q3_pbp()).set_index("game_id")
    assert pd.isna(derived.loc["G1", Q3_POINT_DIFF_COLUMN])
    assert pd.isna(derived.loc["G2", Q3_POINT_DIFF_COLUMN])


def test_q3_point_diff_differential_sign_and_magnitude() -> None:
    derived = derive_q3_point_diff_features(_q3_features(), pbp=_q3_pbp()).set_index("game_id")
    # AAA rolling entering G3 = 3 (G1); CCC rolling entering G3 = -4 (G2).
    assert derived.loc["G3", Q3_POINT_DIFF_COLUMN] == pytest.approx(7.0)
    # AAA rolling entering G4 = mean(3 [G1], 0 [G3]) = 1.5; BBB rolling
    # entering G4 = -3 (its own away-perspective value from G1).
    assert derived.loc["G4", Q3_POINT_DIFF_COLUMN] == pytest.approx(4.5)


def test_q3_point_diff_leakage() -> None:
    """Mutating G3's own Q3->Q4 movement must never change G3's own (strictly
    prior) differential, but legitimately changes G4's."""

    features = _q3_features()
    baseline = derive_q3_point_diff_features(features, pbp=_q3_pbp()).set_index("game_id")

    mutated_pbp = _q3_pbp()
    mutated_pbp.loc[
        (mutated_pbp["game_id"] == "G3") & (mutated_pbp["qtr"] == 4), "score_differential"
    ] = 21
    mutated = derive_q3_point_diff_features(features, pbp=mutated_pbp).set_index("game_id")

    assert mutated.loc["G3", Q3_POINT_DIFF_COLUMN] == pytest.approx(
        baseline.loc["G3", Q3_POINT_DIFF_COLUMN]
    )
    assert mutated.loc["G4", Q3_POINT_DIFF_COLUMN] != pytest.approx(
        baseline.loc["G4", Q3_POINT_DIFF_COLUMN]
    )
    # AAA's own G3 diff becomes 21 - 1 = 20; rolling entering G4 = mean(3, 20) = 11.5.
    assert mutated.loc["G4", Q3_POINT_DIFF_COLUMN] == pytest.approx(11.5 - (-3.0))


def test_q3_point_diff_attach_is_additive() -> None:
    base = _q3_features()
    features = pd.DataFrame({"game_id": base["game_id"], "some_existing_feature": 1.0}).assign(
        season=base["season"].to_numpy(),
        week=base["week"].to_numpy(),
        home_team=base["home_team"].to_numpy(),
        away_team=base["away_team"].to_numpy(),
    )
    widened = attach_q3_point_diff_features(features, pbp=_q3_pbp())
    assert sorted(set(widened.columns) - set(features.columns)) == [Q3_POINT_DIFF_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


def test_q3_point_diff_attach_refuses_to_overwrite_an_existing_column() -> None:
    features = _q3_features().assign(**{Q3_POINT_DIFF_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=Q3_POINT_DIFF_COLUMN):
        attach_q3_point_diff_features(features, pbp=_q3_pbp())


# ---------------------------------------------------------------------------
# LEAD-30: fourth-down aggressiveness x opener-spread interaction
# ---------------------------------------------------------------------------


def _fourth_down_pbp() -> pd.DataFrame:
    def _opportunity(game_id, season, week, home, away, team, play_type, play_id):
        opponent = away if team == home else home
        return _row(
            game_id=game_id,
            season=season,
            week=week,
            home_team=home,
            away_team=away,
            posteam=team,
            defteam=opponent,
            down=4,
            ydstogo=2,
            yardline_100=45,
            play_type=play_type,
            play_id=play_id,
        )

    rows = [
        # Week 1 (every team plays exactly once, so week 2's rolling values are
        # unambiguous -- no team below appears twice in the same week).
        # AAA (home) 1 go + 1 kick -> go_rate 0.5; BBB (away) 1 go -> 1.0.
        _opportunity("G1", 2022, 1, "AAA", "BBB", "AAA", "run", 1),
        _opportunity("G1", 2022, 1, "AAA", "BBB", "AAA", "punt", 2),
        _opportunity("G1", 2022, 1, "AAA", "BBB", "BBB", "pass", 3),
        # CCC (home) 2 kicks -> go_rate 0.0; DDD (away) 1 go -> 1.0 (unused week 2).
        _opportunity("G2", 2022, 1, "CCC", "DDD", "CCC", "punt", 11),
        _opportunity("G2", 2022, 1, "CCC", "DDD", "CCC", "field_goal", 12),
        _opportunity("G2", 2022, 1, "CCC", "DDD", "DDD", "run", 13),
        # EEE (home) 1 go -> 1.0; FFF (away) 1 kick -> 0.0.
        _opportunity("G1b", 2022, 1, "EEE", "FFF", "EEE", "pass", 21),
        _opportunity("G1b", 2022, 1, "EEE", "FFF", "FFF", "punt", 22),
        # GGG (home) 1 go -> 1.0; HHH (away) 1 kick -> 0.0.
        _opportunity("G1c", 2022, 1, "GGG", "HHH", "GGG", "pass", 23),
        _opportunity("G1c", 2022, 1, "GGG", "HHH", "HHH", "punt", 24),
        # III (home) 1 go -> 1.0; JJJ (away) 1 kick -> 0.0.
        _opportunity("G1d", 2022, 1, "III", "JJJ", "III", "pass", 25),
        _opportunity("G1d", 2022, 1, "III", "JJJ", "JJJ", "punt", 26),
        # Week 2: AAA (home, rolling 0.5) vs CCC (away, rolling 0.0) -> diff 0.5.
        _opportunity("G3", 2022, 2, "AAA", "CCC", "AAA", "run", 31),
        # Week 2: EEE (home, rolling 1.0) vs FFF (away, rolling 0.0) -> diff 1.0.
        _opportunity("G4", 2022, 2, "EEE", "FFF", "EEE", "run", 41),
        # Week 2: GGG (home, rolling 1.0) vs HHH (away, rolling 0.0) -> diff 1.0, pick'em.
        _opportunity("G5", 2022, 2, "GGG", "HHH", "GGG", "run", 51),
        # Week 2: III (home, rolling 1.0) vs JJJ (away, rolling 0.0) -> diff 1.0, missing opener.
        _opportunity("G6", 2022, 2, "III", "JJJ", "III", "run", 61),
    ]
    return pd.DataFrame(rows)


def _fourth_down_features() -> pd.DataFrame:
    return _features(
        [
            ("G1", 2022, 1, "AAA", "BBB"),
            ("G2", 2022, 1, "CCC", "DDD"),
            ("G1b", 2022, 1, "EEE", "FFF"),
            ("G1c", 2022, 1, "GGG", "HHH"),
            ("G1d", 2022, 1, "III", "JJJ"),
            ("G3", 2022, 2, "AAA", "CCC"),
            ("G4", 2022, 2, "EEE", "FFF"),
            ("G5", 2022, 2, "GGG", "HHH"),
            ("G6", 2022, 2, "III", "JJJ"),
        ]
    )


def _fourth_down_opener_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G3", "G4", "G5"],
            # G3: AAA (home) is a 3-pt underdog -> home aggressive dog -> positive.
            # G4: EEE (home) is favoured by 7 -> home aggressive favourite -> negative.
            # G5: exact opener pick'em -> interaction forced to 0.
            # G6 is deliberately absent (missing opener line) -> NaN.
            "tue_open_home_spread": [-3.0, 7.0, 0.0],
        }
    )


def test_fourth_down_interaction_nan_before_either_side_has_history() -> None:
    derived = derive_fourth_down_interaction_features(
        _fourth_down_features(), pbp=_fourth_down_pbp(), opener_lines=_fourth_down_opener_lines()
    ).set_index("game_id")
    assert pd.isna(derived.loc["G1", FOURTH_DOWN_INTERACTION_COLUMN])
    assert pd.isna(derived.loc["G2", FOURTH_DOWN_INTERACTION_COLUMN])
    assert pd.isna(derived.loc["G1b", FOURTH_DOWN_INTERACTION_COLUMN])


def test_fourth_down_interaction_sign_convention() -> None:
    derived = derive_fourth_down_interaction_features(
        _fourth_down_features(), pbp=_fourth_down_pbp(), opener_lines=_fourth_down_opener_lines()
    ).set_index("game_id")
    # Aggressive underdog (home) -> positive.
    assert derived.loc["G3", FOURTH_DOWN_INTERACTION_COLUMN] == pytest.approx(0.5)
    # Aggressive favourite (home) -> negative (the FADE case).
    assert derived.loc["G4", FOURTH_DOWN_INTERACTION_COLUMN] == pytest.approx(-1.0)
    # Exact opener pick'em -> forced to 0, not NaN (both rolling rates known).
    assert derived.loc["G5", FOURTH_DOWN_INTERACTION_COLUMN] == pytest.approx(0.0)


def test_fourth_down_interaction_nan_when_opener_spread_is_missing() -> None:
    derived = derive_fourth_down_interaction_features(
        _fourth_down_features(), pbp=_fourth_down_pbp(), opener_lines=_fourth_down_opener_lines()
    ).set_index("game_id")
    # G6 is absent from the opener-lines fixture entirely.
    assert pd.isna(derived.loc["G6", FOURTH_DOWN_INTERACTION_COLUMN])


def test_fourth_down_interaction_leakage() -> None:
    """Mutating G3's own 4th-down decisions must never change G3's own
    (strictly-prior) interaction, but legitimately changes a later game
    rolling up G3's own outcome as history."""

    features = _fourth_down_features()
    opener_lines = _fourth_down_opener_lines()
    baseline = derive_fourth_down_interaction_features(
        features, pbp=_fourth_down_pbp(), opener_lines=opener_lines
    ).set_index("game_id")

    mutated_pbp = _fourth_down_pbp()
    mutated_pbp.loc[
        (mutated_pbp["game_id"] == "G3") & (mutated_pbp["posteam"] == "AAA"), "play_type"
    ] = "punt"
    mutated = derive_fourth_down_interaction_features(
        features, pbp=mutated_pbp, opener_lines=opener_lines
    ).set_index("game_id")

    assert mutated.loc["G3", FOURTH_DOWN_INTERACTION_COLUMN] == pytest.approx(
        baseline.loc["G3", FOURTH_DOWN_INTERACTION_COLUMN]
    )


def test_fourth_down_interaction_attach_is_additive() -> None:
    base = _fourth_down_features()
    features = pd.DataFrame({"game_id": base["game_id"], "some_existing_feature": 1.0}).assign(
        season=base["season"].to_numpy(),
        week=base["week"].to_numpy(),
        home_team=base["home_team"].to_numpy(),
        away_team=base["away_team"].to_numpy(),
    )
    widened = attach_fourth_down_interaction_features(
        features, pbp=_fourth_down_pbp(), opener_lines=_fourth_down_opener_lines()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [FOURTH_DOWN_INTERACTION_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


def test_fourth_down_interaction_attach_refuses_to_overwrite_an_existing_column() -> None:
    features = _fourth_down_features().assign(**{FOURTH_DOWN_INTERACTION_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=FOURTH_DOWN_INTERACTION_COLUMN):
        attach_fourth_down_interaction_features(
            features, pbp=_fourth_down_pbp(), opener_lines=_fourth_down_opener_lines()
        )


# ---------------------------------------------------------------------------
# Registered candidate profiles: production plus exactly the one column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "column"),
    [
        ("weak_stack_opening_drive_epa", OPENING_DRIVE_EPA_COLUMN),
        ("weak_stack_q3_point_diff", Q3_POINT_DIFF_COLUMN),
        ("weak_stack_fourth_down_interaction", FOURTH_DOWN_INTERACTION_COLUMN),
    ],
)
def test_registered_profile_is_production_plus_the_declared_one_column(
    profile: str, column: str
) -> None:
    baseline = set(margin_feature_columns("market_residual", "weak_stack"))
    treatment = set(margin_feature_columns("market_residual", profile))
    assert treatment - baseline == {column}
    assert baseline - treatment == set()
