from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.qb_identity_features import (
    QB_REVENGE_COLUMN,
    ROOKIE_QB_DEBUT_FADE_COLUMN,
    _canonical_schedule_team,
    attach_qb_revenge_features,
    attach_rookie_qb_debut_fade_features,
    describe_rookie_qb_debut_population,
    draft_team_by_gsis_id,
    qb_revenge_join_diagnostics,
)
from nfl_ats.qb_identity_features import (
    derive_qb_revenge_features as decision_derive_qb_revenge_features,
)
from nfl_ats.qb_identity_features import (
    derive_rookie_qb_debut_fade_features as decision_derive_rookie_qb_debut_fade_features,
)


def _game(
    game_id: str,
    season: int,
    gameday: str,
    home: str,
    away: str,
    home_qb: str | None,
    away_qb: str | None,
    game_type: str = "REG",
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "gametime": "13:00",
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "home_qb_id": home_qb,
        "away_qb_id": away_qb,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rosters(rows: list[tuple[int, str, float]]) -> pd.DataFrame:

    return pd.DataFrame(rows, columns=["season", "gsis_id", "years_exp"])


def _debut_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("h_prior", 2020, "2020-09-06", "AAA", "ZZZ", "H1", "ZQ"),
            _game("g1", 2020, "2020-09-10", "AAA", "BBB", "H1", "R1"),
            _game("g2", 2020, "2020-09-13", "CCC", "AAA", "R2", "H1"),
            _game("g_vet", 2020, "2020-09-06", "DDD", "EEE", "V1", "ZQ2"),
            _game("g3", 2020, "2020-09-20", "FFF", "BBB", "H1", "R1"),
            _game("g_unresolved", 2020, "2020-09-08", "GGG", "HHH", "H1", "U1"),
            _game("g_both", 2021, "2021-09-12", "III", "JJJ", "R3", "R4"),
            _game("g_post", 2020, "2021-01-10", "AAA", "KKK", "H1", "PP1", game_type="WC"),
        ]
    )


def _debut_rosters() -> pd.DataFrame:
    return _rosters(
        [
            (2020, "H1", 6.0),
            (2020, "ZQ", 3.0),
            (2020, "R1", 0.0),
            (2020, "R2", 0.0),
            (2020, "V1", 5.0),
            (2020, "ZQ2", 4.0),
            (2021, "R3", 0.0),
            (2021, "R4", 0.0),
        ]
    )


def test_debut_rookie_sign_convention_away_is_positive() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g1", ROOKIE_QB_DEBUT_FADE_COLUMN] == 1.0


def test_debut_rookie_sign_convention_home_is_negative() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g2", ROOKIE_QB_DEBUT_FADE_COLUMN] == -1.0


def test_veteran_whose_first_archived_start_is_not_a_debut() -> None:

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_vet", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_second_start_is_never_flagged_as_a_debut() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g3", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_unresolved_years_exp_is_never_flagged_a_debut() -> None:

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_unresolved", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_both_sides_debuting_simultaneously_is_zero() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_both", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_postseason_game_is_never_flagged() -> None:

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_post", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_describe_rookie_qb_debut_population_diagnostic() -> None:
    diagnostic = describe_rookie_qb_debut_population(_debut_schedule(), _debut_rosters())
    assert diagnostic["n_first_archived_reg_starts"] == 9
    assert diagnostic["n_confirmed_rookie_debuts"] == 4
    assert diagnostic["n_confirmed_non_rookie_first_starts"] == 4
    assert diagnostic["n_unresolved_years_exp"] == 1


def test_rookie_debut_leakage_ignores_unrelated_outcome_columns() -> None:

    schedule = _debut_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 20
    schedule["away_score"] = 17
    baseline = derive_rookie_qb_debut_fade_features(schedule, _debut_rosters()).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -14.0
    mutated["home_score"] = 3
    mutated["away_score"] = 41
    after = derive_rookie_qb_debut_fade_features(mutated, _debut_rosters()).set_index("game_id")
    pd.testing.assert_series_equal(
        baseline[ROOKIE_QB_DEBUT_FADE_COLUMN], after[ROOKIE_QB_DEBUT_FADE_COLUMN]
    )


def test_rookie_debut_attach_is_purely_additive() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_rookie_qb_debut_fade_features(
        features, schedule=schedule, rosters=_debut_rosters(), depth_charts=_fixture_depth(schedule)
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [ROOKIE_QB_DEBUT_FADE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_rookie_debut_attach_requires_the_join_key() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_rookie_qb_debut_fade_features(features, schedule=schedule, rosters=_debut_rosters())


def test_rookie_debut_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], ROOKIE_QB_DEBUT_FADE_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=ROOKIE_QB_DEBUT_FADE_COLUMN):
        attach_rookie_qb_debut_fade_features(features, schedule=schedule, rosters=_debut_rosters())


def test_rookie_debut_derive_requires_every_schedule_column() -> None:
    schedule = _debut_schedule().drop(columns=["home_qb_id"])
    with pytest.raises(DataContractError, match="home_qb_id"):
        derive_rookie_qb_debut_fade_features(schedule, _debut_rosters())


def test_franchise_code_normalization_current_and_historical_codes_match() -> None:

    codes = pd.Series(["OAK", "LV", "SD", "LAC", "STL", "SL", "LA", "WAS", "ARI"])
    canonical = _canonical_schedule_team(codes)
    assert list(canonical) == ["LV", "LV", "LAC", "LAC", "LA", "LA", "LA", "WAS", "ARI"]


def test_draft_team_name_to_code_covers_every_relocation_variant() -> None:
    combine = pd.DataFrame(
        {
            "pfr_id": ["p_oak", "p_lv", "p_sd", "p_lac", "p_stl", "p_lar", "p_wr", "p_wf", "p_wc"],
            "draft_team": [
                "Oakland Raiders",
                "Las Vegas Raiders",
                "San Diego Chargers",
                "Los Angeles Chargers",
                "St. Louis Rams",
                "Los Angeles Rams",
                "Washington Redskins",
                "Washington Football Team",
                "Washington Commanders",
            ],
            "draft_year": [2005, 2021, 2005, 2021, 2005, 2021, 2005, 2019, 2022],
        }
    )
    rosters = pd.DataFrame(
        {
            "pfr_id": combine["pfr_id"],
            "gsis_id": [f"g_{pfr}" for pfr in combine["pfr_id"]],
        }
    )
    lookup = draft_team_by_gsis_id(combine, rosters)
    assert lookup["g_p_oak"] == "LV"
    assert lookup["g_p_lv"] == "LV"
    assert lookup["g_p_sd"] == "LAC"
    assert lookup["g_p_lac"] == "LAC"
    assert lookup["g_p_stl"] == "LA"
    assert lookup["g_p_lar"] == "LA"
    assert lookup["g_p_wr"] == "WAS"
    assert lookup["g_p_wf"] == "WAS"
    assert lookup["g_p_wc"] == "WAS"


def test_draft_team_by_gsis_id_rejects_unrecognized_names() -> None:
    combine = pd.DataFrame(
        {"pfr_id": ["p1"], "draft_team": ["Los Angeles Xtreme"], "draft_year": [2001]}
    )
    rosters = pd.DataFrame({"pfr_id": ["p1"], "gsis_id": ["g1"]})
    with pytest.raises(DataContractError, match="Los Angeles Xtreme"):
        draft_team_by_gsis_id(combine, rosters)


def test_draft_team_by_gsis_id_keeps_earliest_draft_year_on_duplicate() -> None:

    combine = pd.DataFrame(
        {
            "pfr_id": ["p1", "p1"],
            "draft_team": ["Oakland Raiders", "Kansas City Chiefs"],
            "draft_year": [2010, 2012],
        }
    )
    rosters = pd.DataFrame({"pfr_id": ["p1"], "gsis_id": ["g1"]})
    lookup = draft_team_by_gsis_id(combine, rosters)
    assert lookup["g1"] == "LV"


def _revenge_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("r1", 2020, "2020-09-10", "SEA", "OAK", "Q1", "Q9"),
            _game("r2", 2020, "2020-09-13", "SD", "DEN", "Q9", "Q2"),
            _game("r3", 2020, "2020-09-20", "LV", "LAC", "Q4", "Q3"),
            _game("r4", 2020, "2020-09-27", "DEN", "SEA", "Q9", "Q9"),
            _game("r5", 2020, "2020-10-04", "KC", "DEN", "QUNK", "Q9"),
        ]
    )


def _revenge_lookup() -> dict[str, str]:
    return {
        "Q1": "LV",
        "Q2": "LAC",
        "Q3": "LV",
        "Q4": "LAC",
        "Q9": "GB",
    }


def test_qb_revenge_sign_convention_home_is_positive() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r1", QB_REVENGE_COLUMN] == 1.0


def test_qb_revenge_sign_convention_away_is_negative() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r2", QB_REVENGE_COLUMN] == -1.0


def test_qb_revenge_both_sides_simultaneously_is_zero() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r3", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_neither_side_is_zero() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r4", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_unjoined_qb_is_treated_as_zero_never_guessed() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r5", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_join_diagnostics_counts() -> None:
    diagnostic = qb_revenge_join_diagnostics(_revenge_schedule(), _revenge_lookup())
    assert diagnostic["n_qb_side_starts"] == 10
    assert diagnostic["n_resolved_draft_team"] == 9
    assert diagnostic["join_rate"] == pytest.approx(0.9)


def test_qb_revenge_leakage_ignores_unrelated_outcome_columns() -> None:

    schedule = _revenge_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 24
    schedule["away_score"] = 10
    lookup = _revenge_lookup()
    baseline = derive_qb_revenge_features(schedule, lookup).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -21.0
    mutated["home_score"] = 6
    mutated["away_score"] = 27
    after = derive_qb_revenge_features(mutated, lookup).set_index("game_id")
    pd.testing.assert_series_equal(baseline[QB_REVENGE_COLUMN], after[QB_REVENGE_COLUMN])


def test_qb_revenge_attach_is_purely_additive() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_qb_revenge_features(
        features,
        schedule=schedule,
        draft_team_lookup=_revenge_lookup(),
        depth_charts=_fixture_depth(schedule),
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [QB_REVENGE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_qb_revenge_attach_requires_the_join_key() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_qb_revenge_features(features, schedule=schedule, draft_team_lookup=_revenge_lookup())


def test_qb_revenge_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], QB_REVENGE_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=QB_REVENGE_COLUMN):
        attach_qb_revenge_features(features, schedule=schedule, draft_team_lookup=_revenge_lookup())


def test_qb_revenge_derive_requires_every_schedule_column() -> None:
    schedule = _revenge_schedule().drop(columns=["home_qb_id"])
    with pytest.raises(DataContractError, match="home_qb_id"):
        derive_qb_revenge_features(schedule, _revenge_lookup())


def _fixture_depth(schedule: pd.DataFrame) -> pd.DataFrame:
    from nfl_ats.players import _schedule_kickoff_utc

    rows = []
    for index, game in schedule.iterrows():
        for side in ("home", "away"):
            rows.append(
                {
                    "team": game[f"{side}_team"],
                    "gsis_id": game.get(f"{side}_qb_id"),
                    "position": "QB",
                    "depth_rank": 1,
                    "depth_observed_at": _schedule_kickoff_utc(schedule).loc[index]
                    - pd.Timedelta(hours=24),
                }
            )
    return pd.DataFrame(rows)


def derive_qb_revenge_features(schedule, lookup):
    return decision_derive_qb_revenge_features(
        schedule, lookup, depth_charts=_fixture_depth(schedule)
    )


def derive_rookie_qb_debut_fade_features(schedule, rosters):
    return decision_derive_rookie_qb_debut_fade_features(
        schedule, rosters, depth_charts=_fixture_depth(schedule)
    )


def test_late_qb_assignment_does_not_change_decision_features():
    schedule = pd.DataFrame([_game("g", 2025, "2025-09-14", "NE", "NYJ", "late", "away")])
    depth = pd.DataFrame(
        [
            {
                "team": "NE",
                "gsis_id": "early",
                "position": "QB",
                "depth_rank": 1,
                "depth_observed_at": "2025-09-13T12:00Z",
            },
            {
                "team": "NE",
                "gsis_id": "late",
                "position": "QB",
                "depth_rank": 1,
                "depth_observed_at": "2025-09-14T17:00Z",
            },
            {
                "team": "NYJ",
                "gsis_id": "away",
                "position": "QB",
                "depth_rank": 1,
                "depth_observed_at": "2025-09-13T12:00Z",
            },
        ]
    )
    expected = decision_derive_qb_revenge_features(schedule, {"early": "NYJ"}, depth_charts=depth)
    assert expected[QB_REVENGE_COLUMN].iloc[0] == 1
    schedule["home_qb_id"] = "unrelated"
    pd.testing.assert_frame_equal(
        expected,
        decision_derive_qb_revenge_features(schedule, {"early": "NYJ"}, depth_charts=depth),
    )
    unknown = decision_derive_qb_revenge_features(
        schedule, {"early": "NYJ"}, depth_charts=depth.drop(columns="depth_observed_at")
    )
    assert unknown[QB_REVENGE_COLUMN].isna().all()


def test_rookie_debut_uses_projection_and_only_prior_completed_starts():
    schedule = pd.DataFrame(
        [
            _game("past", 2025, "2025-09-07", "NE", "NYJ", "veteran", "away"),
            _game("target", 2025, "2025-09-14", "NE", "NYJ", "late", "away"),
        ]
    )
    depth = _fixture_depth(schedule)
    depth.loc[depth.gsis_id.eq("late"), "gsis_id"] = "rookie"
    rosters = _rosters([(2025, "rookie", 0), (2025, "veteran", 5), (2025, "away", 5)])
    original = decision_derive_rookie_qb_debut_fade_features(schedule, rosters, depth_charts=depth)
    assert original.loc[1, ROOKIE_QB_DEBUT_FADE_COLUMN] == -1
    schedule.loc[1, "home_qb_id"] = "rookie"
    pd.testing.assert_frame_equal(
        original,
        decision_derive_rookie_qb_debut_fade_features(schedule, rosters, depth_charts=depth),
    )
    schedule.loc[0, "home_qb_id"] = "rookie"
    assert (
        decision_derive_rookie_qb_debut_fade_features(schedule, rosters, depth_charts=depth).loc[
            1, ROOKIE_QB_DEBUT_FADE_COLUMN
        ]
        == 0
    )


def test_recorded_qb_feature_has_explicit_oracle_prefix():
    from nfl_ats.qb_identity_features import oracle_derive_qb_revenge_features

    result = oracle_derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup())
    assert "oracle_" + QB_REVENGE_COLUMN in result
    assert QB_REVENGE_COLUMN not in result
