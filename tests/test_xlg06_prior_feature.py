"""Release-blocking tests for the XLG-06 Stage-4 prior wiring (no network)."""

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.xlg06_prior import blend_prior
from nfl_ats.xlg06_prior_feature import (
    PRIOR_ON_PRODUCTION_FEATURE_COLUMNS,
    attach_rookie_prior_features,
    build_player_week_panel,
    derive_prior_expectations,
)

PARAMS = {"intercept": 0.0, "slope": 1.0}


def _linked() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["AAA", "BBB", "CCC"],
            "position": ["WR", "RB", "TE"],
            "year": [2018, 2018, 2019],
            "rating_num": [0.90, 0.80, 0.95],
            "rookie_season": [2020, 2020, 2021],
            "rating": [0.90, 0.80, 0.95],
            "gsis_id_dup": ["AAA", "BBB", "CCC"],
        }
    ).drop(columns="gsis_id_dup")


def _panel_rows() -> pd.DataFrame:
    rows = []
    # AAA (HOME): two 2020 weeks then a loud 2021 week. BBB (AWAY): one 2020
    # week plus a quiet 2021 week, interleaved after AAA's rows. CCC: never
    # plays (no rows) -> cannot move any team value.
    for season, week, snaps, epa in (
        (2020, 1, 50.0, 5.0),
        (2020, 2, 50.0, 5.0),
        (2021, 5, 60.0, 600.0),
    ):
        rows.append(
            {
                "gsis_id": "AAA",
                "season": season,
                "week": week,
                "team": "HOME",
                "off_snaps": snaps,
                "weekly_epa": epa,
            }
        )
    for season, week, snaps, epa in ((2020, 1, 40.0, 2.0), (2021, 5, 30.0, 1.0)):
        rows.append(
            {
                "gsis_id": "BBB",
                "season": season,
                "week": week,
                "team": "AWAY",
                "off_snaps": snaps,
                "weekly_epa": epa,
            }
        )
    return pd.DataFrame(rows)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2020,
                "week": 3,
                "home_team": "HOME",
                "away_team": "AWAY",
                "spread_line": -3.0,
            },
            {
                "game_id": "G2",
                "season": 2021,
                "week": 6,
                "home_team": "HOME",
                "away_team": "AWAY",
                "spread_line": -3.0,
            },
        ]
    )


def test_wiring_is_additive_and_guarded() -> None:
    games = _games()
    out = derive_prior_expectations(games, _panel_rows(), _linked(), PARAMS)
    assert list(out.columns) == [
        "game_id",
        "home_rookie_prior_skill",
        "away_rookie_prior_skill",
        "diff_rookie_prior_skill",
    ]
    merged = attach_rookie_prior_features(
        games, panel=_panel_rows(), linked=_linked(), params=PARAMS
    )
    for column in games.columns:
        pd.testing.assert_series_equal(merged[column], games[column])
    assert set(PRIOR_ON_PRODUCTION_FEATURE_COLUMNS).issubset(merged.columns)
    dupe = games.copy()
    dupe["home_rookie_prior_skill"] = 0.0
    with pytest.raises(DataContractError, match="already carries"):
        attach_rookie_prior_features(dupe, panel=_panel_rows(), linked=_linked(), params=PARAMS)


def test_g1_uses_only_strictly_prior_weeks() -> None:
    out = derive_prior_expectations(_games(), _panel_rows(), _linked(), PARAMS)
    g1 = out.loc[out["game_id"].eq("G1")].iloc[0]
    # AAA at G1 (2020w3): career 100 snaps / 10 EPA over 2 games.
    expected_home = blend_prior(0.90, 5.0, 100.0, intercept=0.0, slope=1.0, n0=300.0)
    assert g1["home_rookie_prior_skill"] == pytest.approx(expected_home)
    # BBB at G1: career 40 snaps / 2 EPA over 1 game.
    expected_away = blend_prior(0.80, 2.0, 40.0, intercept=0.0, slope=1.0, n0=300.0)
    assert g1["away_rookie_prior_skill"] == pytest.approx(expected_away)
    assert g1["diff_rookie_prior_skill"] == pytest.approx(expected_home - expected_away)


def test_post_cutoff_rows_cannot_move_a_prior() -> None:
    # A row dated after BOTH games (2021w7) is post-cutoff for each and must
    # move nothing; a row between the games would legitimately enter G2.
    panel = _panel_rows()
    future = pd.DataFrame(
        [
            {
                "gsis_id": "AAA",
                "season": 2021,
                "week": 7,
                "team": "HOME",
                "off_snaps": 500.0,
                "weekly_epa": 500.0,
            }
        ]
    )
    before = derive_prior_expectations(_games(), panel, _linked(), PARAMS)
    after = derive_prior_expectations(
        _games(), pd.concat([panel, future], ignore_index=True), _linked(), PARAMS
    )
    pd.testing.assert_frame_equal(before, after)


def test_cross_player_state_never_leaks() -> None:
    # AAA's loud 2021 week (600 EPA) must not move BBB's G2 prior: BBB's
    # career is its own 70 snaps / 3.0 EPA over 2 games.
    out = derive_prior_expectations(_games(), _panel_rows(), _linked(), PARAMS)
    g2 = out.loc[out["game_id"].eq("G2")].iloc[0]
    expected_away = blend_prior(0.80, 1.5, 70.0, intercept=0.0, slope=1.0, n0=300.0)
    assert g2["away_rookie_prior_skill"] == pytest.approx(expected_away)
    expected_home = blend_prior(0.90, 610.0 / 3.0, 160.0, intercept=0.0, slope=1.0, n0=300.0)
    assert g2["home_rookie_prior_skill"] == pytest.approx(expected_home)


def test_inactive_side_reads_nan_not_zero() -> None:
    panel = _panel_rows().loc[_panel_rows()["gsis_id"].eq("AAA")].copy()
    out = derive_prior_expectations(_games(), panel, _linked(), PARAMS)
    assert out["away_rookie_prior_skill"].isna().all()
    assert out["diff_rookie_prior_skill"].isna().all()
    assert out["home_rookie_prior_skill"].notna().all()


def test_panel_builder_rejects_missing_keys() -> None:
    snaps = pd.DataFrame({"gsis_id": ["A"], "season": [2020]})
    stats = pd.DataFrame({"player_id": ["A"]})
    with pytest.raises(DataContractError, match="missing columns"):
        build_player_week_panel(snaps, stats)
