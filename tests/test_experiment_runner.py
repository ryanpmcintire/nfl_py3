from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nfl_ats.experiment_runner as experiment_runner_module
from nfl_ats import cli
from nfl_ats.experiment_runner import (
    FLAG_BUILDERS,
    HONEST_REFIT_WIDENING_UPPER_BOUND,
    ExperimentRunnerError,
    ExperimentSpecError,
    FeatureArmConfig,
    FlagBuilder,
    _base_team_game_table,
    _bias_battery_team_game_table,
    _block_bootstrap_subset_gap,
    _build_referee_type_trait_data,
    _flag_backup_qb_start,
    _flag_division_revenge_game,
    _flag_extra_rest_edge,
    _flag_forecast_weather_kn_dome_cold_windy,
    _flag_forecast_weather_kn_precip_high_total,
    _flag_forecast_weather_kn_temp_gap_cold_visitor,
    _flag_forecast_weather_kn_temp_swing_prior_week,
    _flag_forecast_weather_kn_warm_team_cold_late,
    _flag_home_underdog,
    _flag_interim_hc_active,
    _flag_interim_hc_fired_year_one,
    _flag_interim_hc_first_game,
    _flag_interim_hc_home,
    _flag_large_favorite,
    _flag_motivation_mismatch,
    _flag_recent_player_arrest,
    _flag_referee_home_penalty_tilt_bottom_quartile,
    _flag_referee_home_penalty_tilt_top_quartile,
    _flag_referee_penalty_rate_bottom_quartile,
    _flag_referee_penalty_rate_top_quartile,
    _flag_referee_rookie_home_cover,
    _flag_referee_veteran_home_cover,
    _flag_sandwich_spot,
    _flag_short_week,
    _flag_west_coast_early_kickoff,
    _interim_coach_team_game_table,
    _opener_graded_features,
    _RegistryLock,
    classify_subset_bias_result,
    experiment_spec_from_payload,
    experiment_spec_to_payload,
    load_experiment_spec,
    run_experiment,
    run_experiment_cli,
    run_feature_arm_experiment,
    run_subset_bias_experiment,
    scale_subset_effect,
    widening_factor_to_recross_zero,
)
from nfl_ats.weak_signals import WeakSignalError, load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "example_subset_bias",
        "hypothesis": "Some pregame-safe subset covers more than its complement.",
        "experiment_type": "subset_bias",
        "population": {"league": "nfl", "seasons": [2009, 2025], "grade": "close"},
        "construct": {"flag_builder": "home_underdog", "params": {}},
        "endpoints": {"primary": "accuracy", "secondary": []},
        "blocking": {"primary": "week", "secondary": "season"},
        "samples": 2000,
        "seed": 20260818,
        "reliability_check": {"method": "not_applicable", "reason": "situational, not a trait"},
    }
    payload.update(overrides)
    return payload


def _feature_arm_spec_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "example_feature_arm",
        "hypothesis": "Some candidate feature profile beats the baseline.",
        "experiment_type": "feature_arm",
        "population": {"league": "nfl", "seasons": [2018, 2019], "grade": "close"},
        "construct": {
            "baseline": {"feature_profile": "base", "ridge_alpha": 10.0},
            "candidate": {"feature_profile": "base", "ridge_alpha": 50.0},
        },
        "endpoints": {"primary": "accuracy", "secondary": []},
        "blocking": {"primary": "week", "secondary": "season"},
        "samples": 2000,
        "seed": 20260818,
        "reliability_check": {"method": "not_applicable", "reason": "compares two model arms"},
    }
    payload.update(overrides)
    return payload


def test_valid_spec_parses_and_round_trips() -> None:
    spec = experiment_spec_from_payload(_spec_payload())
    assert spec.name == "example_subset_bias"
    assert spec.seasons == (2009, 2025)
    assert spec.samples == 2000
    assert spec.seed == 20260818
    assert spec.block_primary == "week"
    assert spec.block_secondary == "season"
    assert experiment_spec_from_payload(experiment_spec_to_payload(spec)) == spec


def test_spec_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ExperimentSpecError, match="unknown top-level fields"):
        experiment_spec_from_payload(_spec_payload(bogus=1))


def test_spec_requires_seed_with_no_default() -> None:
    payload = _spec_payload()
    del payload["seed"]
    with pytest.raises(ExperimentSpecError, match="missing required field 'seed'"):
        experiment_spec_from_payload(payload)


def test_spec_rejects_non_integer_seed() -> None:
    with pytest.raises(ExperimentSpecError, match="seed must be an integer"):
        experiment_spec_from_payload(_spec_payload(seed=20260818.5))


def test_spec_rejects_unknown_experiment_type() -> None:
    with pytest.raises(ExperimentSpecError, match="Unknown experiment_type"):
        experiment_spec_from_payload(_spec_payload(experiment_type="mystery"))


def test_spec_rejects_unknown_league() -> None:
    payload = _spec_payload()
    payload["population"] = {**payload["population"], "league": "xfl"}  # type: ignore[dict-item]
    with pytest.raises(ExperimentSpecError, match="population\\.league"):
        experiment_spec_from_payload(payload)


def test_spec_rejects_seasons_out_of_order() -> None:
    payload = _spec_payload()
    payload["population"] = {**payload["population"], "seasons": [2020, 2010]}  # type: ignore[dict-item]
    with pytest.raises(ExperimentSpecError, match="out of order"):
        experiment_spec_from_payload(payload)


def test_spec_requires_flag_builder() -> None:
    payload = _spec_payload()
    payload["construct"] = {"params": {}}
    with pytest.raises(ExperimentSpecError, match="flag_builder is required"):
        experiment_spec_from_payload(payload)


def test_spec_endpoints_primary_must_be_accuracy() -> None:
    payload = _spec_payload()
    payload["endpoints"] = {"primary": "brier", "secondary": []}
    with pytest.raises(ExperimentSpecError, match="endpoints\\.primary must be 'accuracy'"):
        experiment_spec_from_payload(payload)


def test_spec_endpoints_secondary_rejects_unknown_metric() -> None:
    payload = _spec_payload()
    payload["endpoints"] = {"primary": "accuracy", "secondary": ["mae"]}
    with pytest.raises(ExperimentSpecError, match="unknown metric"):
        experiment_spec_from_payload(payload)


def test_spec_endpoints_secondary_must_be_empty_for_subset_bias() -> None:
    payload = _spec_payload()
    payload["endpoints"] = {"primary": "accuracy", "secondary": ["brier"]}
    with pytest.raises(ExperimentSpecError, match="must be empty for subset_bias"):
        experiment_spec_from_payload(payload)


def test_spec_blocking_secondary_must_differ_from_primary() -> None:
    payload = _spec_payload()
    payload["blocking"] = {"primary": "week", "secondary": "week"}
    with pytest.raises(ExperimentSpecError, match="must differ from"):
        experiment_spec_from_payload(payload)


def test_spec_reliability_not_applicable_requires_a_reason() -> None:
    payload = _spec_payload()
    payload["reliability_check"] = {"method": "not_applicable", "reason": "   "}
    with pytest.raises(ExperimentSpecError, match="reason is required"):
        experiment_spec_from_payload(payload)


def test_spec_reliability_split_half_does_not_require_a_reason() -> None:
    payload = _spec_payload()
    payload["reliability_check"] = {"method": "split_half"}
    spec = experiment_spec_from_payload(payload)
    assert spec.reliability_method == "split_half"


def test_spec_rejects_too_few_bootstrap_samples() -> None:
    with pytest.raises(ExperimentSpecError, match="samples must be at least 10"):
        experiment_spec_from_payload(_spec_payload(samples=5))


def test_load_experiment_spec_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "spec.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExperimentSpecError, match="not valid JSON"):
        load_experiment_spec(bad)


def test_load_experiment_spec_reads_a_valid_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload()), encoding="utf-8")
    spec = load_experiment_spec(spec_path)
    assert spec.name == "example_subset_bias"


def test_feature_arm_spec_parses_and_round_trips() -> None:
    spec = experiment_spec_from_payload(_feature_arm_spec_payload())
    assert spec.experiment_type == "feature_arm"
    assert spec.flag_builder == ""
    assert spec.construct_params == {}
    assert spec.feature_arm_baseline == FeatureArmConfig(feature_profile="base", ridge_alpha=10.0)
    assert spec.feature_arm_candidate == FeatureArmConfig(feature_profile="base", ridge_alpha=50.0)
    assert experiment_spec_from_payload(experiment_spec_to_payload(spec)) == spec


def test_feature_arm_spec_rejects_subset_bias_construct_fields() -> None:
    payload = _feature_arm_spec_payload(construct={"flag_builder": "home_underdog", "params": {}})
    with pytest.raises(ExperimentSpecError, match="unknown fields for feature_arm"):
        experiment_spec_from_payload(payload)


def test_subset_bias_spec_rejects_feature_arm_construct_fields() -> None:
    payload = _spec_payload(
        construct={
            "baseline": {"feature_profile": "base"},
            "candidate": {"feature_profile": "base"},
        }
    )
    with pytest.raises(ExperimentSpecError, match="unknown fields for subset_bias"):
        experiment_spec_from_payload(payload)


def test_feature_arm_spec_requires_baseline_and_candidate() -> None:
    payload = _feature_arm_spec_payload()
    construct = dict(payload["construct"])  # type: ignore[call-overload]
    del construct["candidate"]
    payload["construct"] = construct
    with pytest.raises(ExperimentSpecError, match="construct\\.candidate is required"):
        experiment_spec_from_payload(payload)


def test_feature_arm_spec_rejects_unknown_feature_profile() -> None:
    payload = _feature_arm_spec_payload()
    payload["construct"]["candidate"] = {"feature_profile": "not_a_real_profile"}  # type: ignore[index]
    with pytest.raises(ExperimentSpecError, match="feature_profile must be one of"):
        experiment_spec_from_payload(payload)


def test_feature_arm_spec_ridge_alpha_defaults_and_validates() -> None:
    payload = _feature_arm_spec_payload()
    payload["construct"]["candidate"] = {"feature_profile": "player"}  # type: ignore[index]
    spec = experiment_spec_from_payload(payload)
    assert spec.feature_arm_candidate is not None
    assert spec.feature_arm_candidate.ridge_alpha == pytest.approx(10.0)

    payload["construct"]["candidate"] = {  # type: ignore[index]
        "feature_profile": "player",
        "ridge_alpha": -1.0,
    }
    with pytest.raises(ExperimentSpecError, match="ridge_alpha must be positive"):
        experiment_spec_from_payload(payload)


def test_feature_arm_spec_endpoints_secondary_may_be_non_empty() -> None:
    payload = _feature_arm_spec_payload(
        endpoints={"primary": "accuracy", "secondary": ["brier", "logloss"]}
    )
    spec = experiment_spec_from_payload(payload)
    assert spec.endpoint_secondary == ("brier", "logloss")


def test_scale_subset_effect_matches_hand_arithmetic() -> None:
    assert scale_subset_effect(0.02, sign=1, fraction_of_slate=0.25) == pytest.approx(0.5)
    assert scale_subset_effect(0.02, sign=-1, fraction_of_slate=0.25) == pytest.approx(-0.5)


def test_scale_subset_effect_rejects_bad_sign_and_fraction() -> None:
    with pytest.raises(ValueError, match="sign must be"):
        scale_subset_effect(0.02, sign=0, fraction_of_slate=0.5)
    with pytest.raises(ValueError, match="fraction_of_slate"):
        scale_subset_effect(0.02, sign=1, fraction_of_slate=1.5)


def test_widening_factor_matches_the_registered_mod06_precedent() -> None:
    factor = widening_factor_to_recross_zero(-0.526, -0.043)
    assert factor == pytest.approx(1.089, abs=0.005)
    assert factor < HONEST_REFIT_WIDENING_UPPER_BOUND


def test_classification_stays_unresolved_inside_the_honest_band() -> None:
    result = classify_subset_bias_result(estimate=-0.526, lower=-1.019, upper=-0.043)
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert result.widening_factor == pytest.approx(1.089, abs=0.005)


def test_classification_refutes_only_past_the_honest_band() -> None:
    result = classify_subset_bias_result(estimate=-1.0, lower=-1.3, upper=-0.2)
    assert result.widening_factor is not None
    assert result.widening_factor > HONEST_REFIT_WIDENING_UPPER_BOUND
    assert result.classification == "refuted_mechanism"
    assert result.closing_ground == "wrong_sign_resolved"


def test_classification_never_closes_on_an_interval_crossing_zero() -> None:
    result = classify_subset_bias_result(estimate=0.4, lower=-0.6, upper=1.4)
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert result.widening_factor is None


def test_classification_never_produces_bounded_by_control() -> None:
    for estimate, lower, upper in (
        (-0.1, -5.0, 5.0),
        (-2.0, -3.0, -1.0),
        (2.0, -1.0, 5.0),
        (-0.01, -0.02, -0.001),
    ):
        result = classify_subset_bias_result(estimate=estimate, lower=lower, upper=upper)
        assert result.classification in ("unresolved_below_power", "refuted_mechanism")
        assert result.closing_ground in (None, "wrong_sign_resolved")


def test_widening_factor_requires_a_negative_upper_and_a_lower_estimate() -> None:
    with pytest.raises(ValueError, match="upper must be strictly below zero"):
        widening_factor_to_recross_zero(-1.0, 0.5)
    with pytest.raises(ValueError, match="estimate must be strictly below upper"):
        widening_factor_to_recross_zero(-0.01, -0.02)


def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 1, 2, 1],
            "home_team": ["NE", "OAK", "NE", "NE"],
            "away_team": ["BUF", "KC", "BUF", "BUF"],
            "home_cover": [1.0, 0.0, np.nan, 1.0],
            "spread_line": [-3.0, 4.0, -3.0, 6.5],
            "game_type": ["REG", "REG", "REG", "POST"],
        }
    )


def test_base_team_game_table_drops_pushes_and_postseason_and_canonicalizes() -> None:
    table = _base_team_game_table(_synthetic_features())
    assert len(table) == 4
    assert set(table["game_id"]) == {"g1", "g2"}
    assert "OAK" not in set(table["team"])
    assert "LV" in set(table["team"])
    home_rows = table.loc[table["is_home"]]
    assert set(home_rows["team"]) == {"NE", "LV"}
    g1_home = table.loc[(table["game_id"] == "g1") & table["is_home"]].iloc[0]
    assert g1_home["team_covered"] == pytest.approx(1.0)
    assert g1_home["team_spread"] == pytest.approx(-3.0)
    g1_away = table.loc[(table["game_id"] == "g1") & ~table["is_home"]].iloc[0]
    assert g1_away["team_covered"] == pytest.approx(0.0)
    assert g1_away["team_spread"] == pytest.approx(3.0)
    assert (table["week_block"] == 202001).all()


def test_base_team_game_table_requires_needed_columns() -> None:
    with pytest.raises(ExperimentRunnerError, match="missing columns"):
        _base_team_game_table(pd.DataFrame({"game_id": ["g1"]}))


def test_flag_home_underdog_matches_hand_computation() -> None:
    features = _synthetic_features()
    construct = _flag_home_underdog(features, (2009, 2025), {}, REPO_ROOT)
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    expected = (table["game_id"] == "g1") & table["is_home"]
    assert (flag == expected).all()
    assert construct.sign == 1
    assert construct.eligible is None
    assert construct.reliability is None


def test_flag_large_favorite_respects_threshold_param() -> None:
    features = _synthetic_features()
    construct = _flag_large_favorite(features, (2009, 2025), {"threshold": 2.0}, REPO_ROOT)
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    expected = table["team_spread"] > 2.0
    assert (flag == expected).all()
    assert int(flag.sum()) == 2
    assert construct.sign == -1


def test_flag_builders_registry_has_the_documented_names() -> None:
    assert set(FLAG_BUILDERS) == {
        "penalty_rate_quartile",
        "home_underdog",
        "large_favorite",
        "recent_player_arrest",
        "drought_severe_grass",
        "division_revenge_game",
        "extra_rest_edge",
        "short_week",
        "west_coast_early_kickoff",
        "sandwich_spot",
        "backup_qb_start",
        "motivation_mismatch",
        "referee_penalty_rate_top_quartile",
        "referee_penalty_rate_bottom_quartile",
        "referee_home_penalty_tilt_top_quartile",
        "referee_home_penalty_tilt_bottom_quartile",
        "referee_veteran_home_cover",
        "referee_rookie_home_cover",
        "referee_high_flag_heavy_underdog",
        "referee_dpi_tilt_pass_heavy_favorite",
        "referee_holding_tilt_run_heavy",
        "referee_flag_rate_high_total_line",
        "forecast_weather_kn_warm_team_cold_late",
        "forecast_weather_kn_temp_gap_cold_visitor",
        "forecast_weather_kn_wind_passing_away_favorite",
        "forecast_weather_kn_precip_high_total",
        "forecast_weather_kn_temp_swing_prior_week",
        "forecast_weather_kn_dome_cold_windy",
        "interim_hc_active",
        "interim_hc_first_game",
        "interim_hc_home",
        "interim_hc_fired_year_one",
    }
    for builder in FLAG_BUILDERS.values():
        assert builder.leagues == ("nfl",)


def test_recent_player_arrest_is_strictly_pregame_and_canonicalizes_teams(
    tmp_path: Path,
) -> None:
    features = pd.DataFrame(
        {
            "game_id": ["g_jax", "g_exact", "g_old"],
            "season": [2024, 2024, 2024],
            "week": [2, 2, 2],
            "gameday": ["2024-09-22", "2024-09-22", "2024-09-22"],
            "home_team": ["JAX", "DEN", "IND"],
            "away_team": ["BUF", "KC", "CHI"],
            "home_cover": [1.0, 0.0, 1.0],
            "spread_line": [-3.0, 2.5, -1.5],
            "game_type": ["REG", "REG", "REG"],
        }
    )
    incidents = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "incident_date": ["2024-09-03", "2024-09-17", "2024-09-02", "2024-09-18"],
            "team": ["JAC", "DEN", "IN", "BUF"],
            "outcome_archive_only": ["a", "b", "c", "d"],
        }
    )
    path = tmp_path / "incidents.parquet"
    incidents.to_parquet(path, index=False)

    construct = _flag_recent_player_arrest(
        features,
        (2024, 2024),
        {"incidents_path": str(path), "window_days": 14},
        tmp_path,
    )
    flagged = construct.table.loc[construct.flag, ["game_id", "team", "incident_age_days"]]

    assert flagged.to_dict("records") == [
        {"game_id": "g_jax", "team": "JAX", "incident_age_days": 14.0}
    ]
    assert construct.sign == -1
    assert construct.reliability is None

    incidents["outcome_archive_only"] = "retrospective mutation"
    incidents.to_parquet(path, index=False)
    mutated = _flag_recent_player_arrest(
        features,
        (2024, 2024),
        {"incidents_path": str(path), "window_days": 14},
        tmp_path,
    )
    pd.testing.assert_series_equal(construct.flag, mutated.flag)


def test_recent_player_arrest_category_filter_is_safe_and_declarative(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "game_id": ["g_one", "g_two"],
            "season": [2024, 2024],
            "week": [3, 3],
            "gameday": ["2024-09-22", "2024-09-22"],
            "home_team": ["JAX", "IND"],
            "away_team": ["BUF", "CHI"],
            "home_cover": [1.0, 0.0],
            "spread_line": [-3.0, 2.5],
            "game_type": ["REG", "REG"],
        }
    )
    incidents = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "incident_date": ["2024-09-03", "2024-09-03", "2024-09-10", "2024-09-18"],
            "team": ["JAC", "BUF", "IN", "CHI"],
            "category": ["Assault, gun", "DUI", "Sexual abuse", "Battery"],
            "outcome_archive_only": ["a", "b", "c", "d"],
        }
    )
    path = tmp_path / "incidents.parquet"
    incidents.to_parquet(path, index=False)
    params = {
        "incidents_path": str(path),
        "window_days": 14,
        "category_contains_any": ["assault", "sexual abuse", "battery"],
    }

    construct = _flag_recent_player_arrest(features, (2024, 2024), params, tmp_path)
    flagged = construct.table.loc[construct.flag, ["game_id", "team", "incident_age_days"]]

    assert sorted(flagged.to_dict("records"), key=lambda row: row["game_id"]) == [
        {"game_id": "g_one", "team": "JAX", "incident_age_days": 14.0},
        {"game_id": "g_two", "team": "IND", "incident_age_days": 7.0},
    ]
    assert "3 of 4 source rows matched before team mapping" in construct.population_note

    incidents["outcome_archive_only"] = "retrospective mutation"
    incidents.to_parquet(path, index=False)
    mutated = _flag_recent_player_arrest(features, (2024, 2024), params, tmp_path)
    pd.testing.assert_series_equal(construct.flag, mutated.flag)


def _interim_game(
    *,
    game_id: str,
    season: int,
    week: int,
    gameday: str,
    home_team: str,
    away_team: str,
    home_coach: str,
    away_coach: str,
    home_cover: float = 1.0,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": "REG",
        "gameday": gameday,
        "home_team": home_team,
        "away_team": away_team,
        "home_coach": home_coach,
        "away_coach": away_coach,
        "home_cover": home_cover,
        "spread_line": -3.0,
    }


def _write_interim_coach_repo(tmp_path: Path) -> tuple[Path, Path]:
    games = [
        _interim_game(
            game_id="aaa2013",
            season=2013,
            week=1,
            gameday="2013-09-08",
            home_team="AAA",
            away_team="OPP",
            home_coach="OLD_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="aaa2014",
            season=2014,
            week=1,
            gameday="2014-09-07",
            home_team="AAA",
            away_team="OPP",
            home_coach="OLD_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="aaa2015w1",
            season=2015,
            week=1,
            gameday="2015-09-13",
            home_team="AAA",
            away_team="OPP",
            home_coach="OLD_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="aaa2015w2",
            season=2015,
            week=2,
            gameday="2015-09-20",
            home_team="AAA",
            away_team="OPP",
            home_coach="OLD_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="aaa2015w3",
            season=2015,
            week=3,
            gameday="2015-09-27",
            home_team="AAA",
            away_team="OPP",
            home_coach="NEW_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="aaa2015w4",
            season=2015,
            week=4,
            gameday="2015-10-04",
            home_team="AAA",
            away_team="OPP",
            home_coach="NEW_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="bbb2013",
            season=2013,
            week=1,
            gameday="2013-09-08",
            home_team="BBB",
            away_team="OPP",
            home_coach="W",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="bbb2014",
            season=2014,
            week=1,
            gameday="2014-09-07",
            home_team="BBB",
            away_team="OPP",
            home_coach="Y",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="bbb2015w1",
            season=2015,
            week=1,
            gameday="2015-09-13",
            home_team="BBB",
            away_team="OPP",
            home_coach="Y",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="bbb2015w3",
            season=2015,
            week=3,
            gameday="2015-09-27",
            home_team="BBB",
            away_team="OPP",
            home_coach="Z",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ccc2014",
            season=2014,
            week=1,
            gameday="2014-09-07",
            home_team="CCC",
            away_team="OPP",
            home_coach="P",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ccc2015w1",
            season=2015,
            week=1,
            gameday="2015-09-13",
            home_team="CCC",
            away_team="OPP",
            home_coach="P",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ccc2015w3",
            season=2015,
            week=3,
            gameday="2015-09-27",
            home_team="CCC",
            away_team="OPP",
            home_coach="Q",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ddd2015w1",
            season=2015,
            week=1,
            gameday="2015-09-13",
            home_team="DDD",
            away_team="OPP",
            home_coach="OLDD",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ddd2015w3",
            season=2015,
            week=3,
            gameday="2015-09-27",
            home_team="DDD",
            away_team="OPP",
            home_coach="OLDD",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="ddd2015w4",
            season=2015,
            week=4,
            gameday="2015-10-04",
            home_team="DDD",
            away_team="OPP",
            home_coach="OLDD",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="eee2015w1",
            season=2015,
            week=1,
            gameday="2015-09-13",
            home_team="EEE",
            away_team="OPP",
            home_coach="SUSP_COACH",
            away_coach="OPP_COACH",
        ),
        _interim_game(
            game_id="eee2015w3",
            season=2015,
            week=3,
            gameday="2015-09-27",
            home_team="EEE",
            away_team="OPP",
            home_coach="EEE_INTERIM",
            away_coach="OPP_COACH",
        ),
    ]

    feature_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_cover",
        "spread_line",
        "game_type",
    ]
    features = pd.DataFrame([{k: g[k] for k in feature_cols} for g in games])
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    schedules_cols = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
    ]
    schedules = pd.DataFrame([{k: g[k] for k in schedules_cols} for g in games])
    raw_dir = tmp_path / "data" / "raw" / "20200101T000000Z"
    raw_dir.mkdir(parents=True)
    schedules.to_parquet(raw_dir / "schedules.parquet")

    parsed = pd.DataFrame(
        [
            {
                "entry_id": 1,
                "interim_coach_name": "NEW_COACH",
                "team_abbr": "AAA",
                "predecessor_coach_name": "OLD_COACH",
                "predecessor_status": "fired",
                "takeover_date_pfr": "Sept. 27, 2015",
                "takeover_date_iso": "2015-09-27",
                "season": 2015,
                "joinable_2009plus": True,
                "date_source": "pfr_primary",
                "notes": "",
            },
            {
                "entry_id": 2,
                "interim_coach_name": "Z",
                "team_abbr": "BBB",
                "predecessor_coach_name": "Y",
                "predecessor_status": "fired",
                "takeover_date_pfr": "Sept. 27, 2015",
                "takeover_date_iso": "2015-09-27",
                "season": 2015,
                "joinable_2009plus": True,
                "date_source": "pfr_primary",
                "notes": "",
            },
            {
                "entry_id": 3,
                "interim_coach_name": "Q",
                "team_abbr": "CCC",
                "predecessor_coach_name": "P",
                "predecessor_status": "fired",
                "takeover_date_pfr": "Sept. 27, 2015",
                "takeover_date_iso": "2015-09-27",
                "season": 2015,
                "joinable_2009plus": True,
                "date_source": "pfr_primary",
                "notes": "",
            },
            {
                "entry_id": 4,
                "interim_coach_name": "NEWD",
                "team_abbr": "DDD",
                "predecessor_coach_name": "OLDD",
                "predecessor_status": "fired",
                "takeover_date_pfr": "Sept. 27, 2015",
                "takeover_date_iso": "2015-09-27",
                "season": 2015,
                "joinable_2009plus": True,
                "date_source": "secondary_schedules_parquet",
                "notes": "fallback join test",
            },
            {
                "entry_id": 5,
                "interim_coach_name": "EEE_INTERIM",
                "team_abbr": "EEE",
                "predecessor_coach_name": "SUSP_COACH",
                "predecessor_status": "suspended",
                "takeover_date_pfr": "Sept. 27, 2015",
                "takeover_date_iso": "2015-09-27",
                "season": 2015,
                "joinable_2009plus": True,
                "date_source": "pfr_primary",
                "notes": "",
            },
            {
                "entry_id": 6,
                "interim_coach_name": "NOT_JOINABLE",
                "team_abbr": "AAA",
                "predecessor_coach_name": "ANCIENT",
                "predecessor_status": "fired",
                "takeover_date_pfr": "Dec. 1, 2005",
                "takeover_date_iso": "2005-12-01",
                "season": 2005,
                "joinable_2009plus": False,
                "date_source": "pfr_primary",
                "notes": "pre-2009, must be excluded entirely",
            },
        ]
    )
    interim_dir = tmp_path / "data" / "raw" / "interim_coaches" / "20200101T000000Z"
    interim_dir.mkdir(parents=True)
    parsed.to_csv(interim_dir / "parsed_table.csv", index=False)

    return features_path, tmp_path


def test_interim_coach_join_matches_by_name_and_falls_back_to_takeover_date(
    tmp_path: Path,
) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    table, trait_data = _interim_coach_team_game_table(features, repo_root)

    assert trait_data.n_entries_total == 6
    assert trait_data.n_entries_joinable == 5

    aaa = table.loc[table["team"] == "AAA"].sort_values("week")
    assert aaa.set_index("week")["under_interim"].to_dict() == {
        1: False,
        2: False,
        3: True,
        4: True,
    }
    w3 = aaa.loc[aaa["week"] == 3].iloc[0]
    w4 = aaa.loc[aaa["week"] == 4].iloc[0]
    assert w3["first_game_under_interim"] and w3["interim_game_number"] == 1
    assert (not w4["first_game_under_interim"]) and w4["interim_game_number"] == 2

    ddd = table.loc[table["team"] == "DDD"].sort_values("week")
    assert ddd.set_index("week")["under_interim"].to_dict() == {1: False, 3: True, 4: True}


def test_interim_coach_fired_year_one_known_and_unknown_cases(tmp_path: Path) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    table, _trait_data = _interim_coach_team_game_table(features, repo_root)

    def _under_interim_row(team: str) -> pd.Series:
        rows = table.loc[(table["team"] == team) & table["under_interim"]]
        assert rows["fired_coach_was_year_one"].nunique() == 1
        assert rows["fired_coach_year_one_known"].nunique() == 1
        return rows.iloc[0]

    aaa = _under_interim_row("AAA")
    assert bool(aaa["fired_coach_year_one_known"])
    assert not aaa["fired_coach_was_year_one"]

    bbb = _under_interim_row("BBB")
    assert bbb["fired_coach_year_one_known"]
    assert bbb["fired_coach_was_year_one"]

    ccc = _under_interim_row("CCC")
    assert not ccc["fired_coach_year_one_known"]
    assert not ccc["fired_coach_was_year_one"]


def test_flag_interim_hc_active_exclude_suspension_param(tmp_path: Path) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)

    included = _flag_interim_hc_active(features, (2009, 2025), {}, repo_root)
    included_table = included.table.reset_index(drop=True)
    included_flag = included.flag.reset_index(drop=True)
    assert "EEE" in set(included_table.loc[included_flag, "team"])
    assert included.sign == 1

    excluded = _flag_interim_hc_active(
        features, (2009, 2025), {"exclude_suspension_cases": True}, repo_root
    )
    excluded_table = excluded.table.reset_index(drop=True)
    excluded_flag = excluded.flag.reset_index(drop=True)
    assert "EEE" not in set(excluded_table.loc[excluded_flag, "team"])
    assert {"AAA", "BBB", "CCC", "DDD"}.issubset(set(excluded_table.loc[excluded_flag, "team"]))


def test_flag_interim_hc_first_game_flags_only_the_first_stint_game(tmp_path: Path) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    construct = _flag_interim_hc_first_game(features, (2009, 2025), {}, repo_root)
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    aaa_flagged_weeks = set(table.loc[flag & (table["team"] == "AAA"), "week"])
    assert aaa_flagged_weeks == {3}
    assert construct.eligible is None
    assert construct.sign == 1


def test_flag_interim_hc_home_restricts_eligible_to_under_interim_population(
    tmp_path: Path,
) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    construct = _flag_interim_hc_home(features, (2009, 2025), {}, repo_root)
    table = construct.table.reset_index(drop=True)
    eligible = construct.eligible.reset_index(drop=True)
    under_interim = table["under_interim"].reset_index(drop=True)
    assert (eligible == under_interim).all()
    assert eligible.sum() > 0


def test_flag_interim_hc_fired_year_one_eligible_requires_known_tenure(tmp_path: Path) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    construct = _flag_interim_hc_fired_year_one(features, (2009, 2025), {}, repo_root)
    table = construct.table.reset_index(drop=True)
    eligible = construct.eligible.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)

    ccc_rows = table["team"] == "CCC"
    assert not eligible.loc[ccc_rows].any()

    bbb_eligible_flagged = table.loc[eligible & (table["team"] == "BBB"), :]
    assert not bbb_eligible_flagged.empty
    assert flag.loc[bbb_eligible_flagged.index].all()

    aaa_eligible_flagged = table.loc[eligible & (table["team"] == "AAA"), :]
    assert not aaa_eligible_flagged.empty
    assert not flag.loc[aaa_eligible_flagged.index].any()
    assert construct.sign == -1


def test_interim_coach_join_raises_loudly_on_an_unmatched_entry(tmp_path: Path) -> None:
    features_path, repo_root = _write_interim_coach_repo(tmp_path)
    features = pd.read_parquet(features_path)
    interim_path = (
        repo_root / "data" / "raw" / "interim_coaches" / "20200101T000000Z" / "parsed_table.csv"
    )
    parsed = pd.read_csv(interim_path)
    parsed.loc[parsed["entry_id"] == 4, "takeover_date_iso"] = "2015-12-31"
    parsed.to_csv(interim_path, index=False)
    with pytest.raises(ExperimentRunnerError, match="matched NEITHER"):
        _interim_coach_team_game_table(features, repo_root)


def _fc_game(
    *,
    game_id: str,
    week: int,
    home_team: str,
    away_team: str,
    home_cover: float,
    spread_line: float,
    total_line: float,
    roof: str,
    temp: float,
    wind: float,
    season: int = 2009,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": "REG",
        "home_team": home_team,
        "away_team": away_team,
        "home_cover": home_cover,
        "spread_line": spread_line,
        "total_line": total_line,
        "roof": roof,
        "temp": temp,
        "wind": wind,
        "gameday": f"{season}-09-{week:02d}",
        "stadium": f"{home_team} Stadium",
    }


def _write_forecast_weather_repo(
    tmp_path: Path, games: list[dict[str, Any]], forecasts: list[dict[str, Any]]
) -> pd.DataFrame:
    feature_cols = [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "home_cover",
        "spread_line",
        "total_line",
        "temp",
        "wind",
        "gameday",
    ]
    features = pd.DataFrame([{k: g[k] for k in feature_cols} for g in games])

    schedules = pd.DataFrame(
        [{"game_id": g["game_id"], "stadium": g["stadium"], "roof": g["roof"]} for g in games]
    )
    raw_dir = tmp_path / "data" / "raw" / "20200101T000000Z"
    raw_dir.mkdir(parents=True)
    schedules.to_parquet(raw_dir / "schedules.parquet")

    forecast_df = pd.DataFrame(forecasts)
    archive_dir = tmp_path / "forecast_archive"
    archive_dir.mkdir(parents=True)
    forecast_df.to_parquet(archive_dir / "forecasts.parquet")

    return features


def _fc_params() -> dict[str, Any]:
    return {"forecast_archive_path": "forecast_archive/forecasts.parquet"}


def _fc_forecast(
    game_id: str, *, temp: float, wind: float, precip: float, status: str = "ok"
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "forecast_temp_f": temp,
        "forecast_wind_mph": wind,
        "forecast_precip_prob_pct": precip,
        "fetch_status": status,
    }


def test_forecast_weather_game_table_and_cells_match_hand_computation(tmp_path: Path) -> None:
    games = [
        _fc_game(
            game_id="g_dome1",
            week=1,
            home_team="DEN",
            away_team="SEA",
            home_cover=1.0,
            spread_line=-3.0,
            total_line=45.0,
            roof="dome",
            temp=70.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_dome2",
            week=2,
            home_team="DEN",
            away_team="KC",
            home_cover=0.0,
            spread_line=-3.0,
            total_line=45.0,
            roof="dome",
            temp=68.0,
            wind=4.0,
        ),
        _fc_game(
            game_id="g_kc_home",
            week=3,
            home_team="KC",
            away_team="DEN",
            home_cover=1.0,
            spread_line=2.0,
            total_line=45.0,
            roof="outdoors",
            temp=80.0,
            wind=6.0,
        ),
        _fc_game(
            game_id="g_test_dome_cold_windy",
            week=10,
            home_team="SEA",
            away_team="DEN",
            home_cover=1.0,
            spread_line=-1.0,
            total_line=45.0,
            roof="outdoors",
            temp=25.0,
            wind=20.0,
        ),
        _fc_game(
            game_id="g_test_dome_cold_windy_control",
            week=11,
            home_team="SEA",
            away_team="KC",
            home_cover=0.0,
            spread_line=1.0,
            total_line=45.0,
            roof="outdoors",
            temp=60.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_temp_gap",
            week=12,
            home_team="SEA",
            away_team="KC",
            home_cover=1.0,
            spread_line=-2.0,
            total_line=45.0,
            roof="outdoors",
            temp=55.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_precip",
            week=13,
            home_team="SEA",
            away_team="DEN",
            home_cover=0.0,
            spread_line=3.0,
            total_line=50.0,
            roof="outdoors",
            temp=50.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_precip_control",
            week=14,
            home_team="SEA",
            away_team="DEN",
            home_cover=1.0,
            spread_line=3.0,
            total_line=40.0,
            roof="outdoors",
            temp=50.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_warm_late",
            week=15,
            home_team="SEA",
            away_team="MIA",
            home_cover=1.0,
            spread_line=-2.0,
            total_line=45.0,
            roof="outdoors",
            temp=30.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_warm_early",
            week=5,
            home_team="SEA",
            away_team="MIA",
            home_cover=0.0,
            spread_line=-2.0,
            total_line=45.0,
            roof="outdoors",
            temp=30.0,
            wind=5.0,
        ),
        _fc_game(
            game_id="g_test_temp_swing",
            week=16,
            home_team="SEA",
            away_team="DEN",
            home_cover=1.0,
            spread_line=-2.0,
            total_line=45.0,
            roof="outdoors",
            temp=60.0,
            wind=5.0,
        ),
    ]
    forecasts = [
        _fc_forecast("g_dome1", temp=70.0, wind=5.0, precip=10.0),
        _fc_forecast("g_dome2", temp=68.0, wind=4.0, precip=10.0),
        _fc_forecast("g_kc_home", temp=80.0, wind=6.0, precip=10.0),
        _fc_forecast("g_test_dome_cold_windy", temp=20.0, wind=15.0, precip=10.0),
        _fc_forecast("g_test_dome_cold_windy_control", temp=20.0, wind=15.0, precip=10.0),
        _fc_forecast("g_test_temp_gap", temp=40.0, wind=5.0, precip=10.0),
        _fc_forecast("g_test_precip", temp=45.0, wind=5.0, precip=75.0),
        _fc_forecast("g_test_precip_control", temp=45.0, wind=5.0, precip=75.0),
        _fc_forecast("g_test_warm_late", temp=30.0, wind=5.0, precip=10.0),
        _fc_forecast("g_test_warm_early", temp=30.0, wind=5.0, precip=10.0),
        _fc_forecast("g_test_temp_swing", temp=90.0, wind=5.0, precip=10.0),
    ]
    features = _write_forecast_weather_repo(tmp_path, games, forecasts)
    params = _fc_params()

    dome = _flag_forecast_weather_kn_dome_cold_windy(features, (2009, 2025), params, tmp_path)
    dtable = dome.table.reset_index(drop=True)
    dflag = dome.flag.reset_index(drop=True)
    flagged = set(dtable.loc[dflag, "game_id"])
    assert "g_test_dome_cold_windy" in flagged
    assert "g_test_dome_cold_windy_control" not in flagged
    assert dome.sign == 1
    row = dtable.loc[dtable["game_id"] == "g_test_dome_cold_windy"].iloc[0]
    assert row["team_covered"] == pytest.approx(1.0)
    assert bool(row["outdoor"]) is True

    gap = _flag_forecast_weather_kn_temp_gap_cold_visitor(features, (2009, 2025), params, tmp_path)
    gtable, gflag = gap.table.reset_index(drop=True), gap.flag.reset_index(drop=True)
    assert "g_test_temp_gap" in set(gtable.loc[gflag, "game_id"])

    precip = _flag_forecast_weather_kn_precip_high_total(features, (2009, 2025), params, tmp_path)
    ptable, pflag = precip.table.reset_index(drop=True), precip.flag.reset_index(drop=True)
    pflagged = set(ptable.loc[pflag, "game_id"])
    assert "g_test_precip" in pflagged
    assert "g_test_precip_control" not in pflagged

    warm = _flag_forecast_weather_kn_warm_team_cold_late(features, (2009, 2025), params, tmp_path)
    wtable, wflag = warm.table.reset_index(drop=True), warm.flag.reset_index(drop=True)
    wflagged = set(wtable.loc[wflag, "game_id"])
    assert "g_test_warm_late" in wflagged
    assert "g_test_warm_early" not in wflagged

    swing = _flag_forecast_weather_kn_temp_swing_prior_week(
        features, (2009, 2025), params, tmp_path
    )
    stable, sflag = swing.table.reset_index(drop=True), swing.flag.reset_index(drop=True)
    assert "g_test_temp_swing" in set(stable.loc[sflag, "game_id"])


def _game(
    *,
    game_id: str,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    result: float,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "game_id": game_id,
        "season": season,
        "week": week,
        "gameday": (pd.Timestamp("2020-09-01") + pd.Timedelta(days=7 * (week - 1)))
        .date()
        .isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "result": result,
        "spread_line": 0.0,
        "game_type": "REG",
        "div_game": 0,
        "neutral_site": 0,
        "weekday": "Sunday",
        "gametime": "13:00",
        "temp": 70.0,
        "home_rest": 7,
        "away_rest": 7,
        "roof": "outdoors",
        "surface": "grass",
        "home_qb_name": "HQB",
        "away_qb_name": "AQB",
    }
    payload.update(overrides)
    return payload


def _write_bias_battery_repo(tmp_path: Path, games: list[dict[str, Any]]) -> Path:
    feature_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "result",
        "spread_line",
        "game_type",
        "div_game",
        "neutral_site",
        "weekday",
        "gametime",
        "temp",
    ]
    schedule_cols = [
        "game_id",
        "home_rest",
        "away_rest",
        "roof",
        "surface",
        "home_qb_name",
        "away_qb_name",
    ]
    features = pd.DataFrame([{k: g[k] for k in feature_cols} for g in games])
    features["result"] = pd.to_numeric(features["result"])
    features["spread_line"] = pd.to_numeric(features["spread_line"])
    features["home_cover"] = np.select(
        [
            features["result"] > features["spread_line"],
            features["result"] < features["spread_line"],
        ],
        [1.0, 0.0],
        default=np.nan,
    )
    features["ats_margin"] = features["result"] - features["spread_line"]
    schedules = pd.DataFrame([{k: g[k] for k in schedule_cols} for g in games])
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    raw_dir = tmp_path / "data" / "raw" / "20200101T000000Z"
    raw_dir.mkdir(parents=True)
    schedules.to_parquet(raw_dir / "schedules.parquet")
    return features_path


def _read_bias_battery_features(features_path: Path) -> pd.DataFrame:
    return pd.read_parquet(features_path)


def test_bias_battery_team_game_table_requires_schedules_snapshot(tmp_path: Path) -> None:
    features_path = _write_bias_battery_repo(
        tmp_path,
        [_game(game_id="g1", season=2020, week=1, home_team="AAA", away_team="BBB", result=3.0)],
    )
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ExperimentRunnerError, match=r"No data/raw/\*/schedules\.parquet snapshot"):
        _bias_battery_team_game_table(_read_bias_battery_features(features_path), empty_root)


def test_flag_division_revenge_game_matches_hand_computation(tmp_path: Path) -> None:
    games = [
        _game(
            game_id="g1",
            season=2020,
            week=1,
            home_team="CCC",
            away_team="AAA",
            result=7.0,
            div_game=1,
        ),
        _game(
            game_id="g2",
            season=2020,
            week=2,
            home_team="AAA",
            away_team="CCC",
            result=3.0,
            div_game=1,
        ),
    ]
    features_path = _write_bias_battery_repo(tmp_path, games)
    construct = _flag_division_revenge_game(
        _read_bias_battery_features(features_path), (2009, 2025), {}, tmp_path
    )
    table, flag = construct.table.reset_index(drop=True), construct.flag.reset_index(drop=True)
    aaa_g2 = flag.loc[(table["game_id"] == "g2") & (table["team"] == "AAA")]
    ccc_g2 = flag.loc[(table["game_id"] == "g2") & (table["team"] == "CCC")]
    aaa_g1 = flag.loc[(table["game_id"] == "g1") & (table["team"] == "AAA")]
    assert bool(aaa_g2.iloc[0]) is True
    assert bool(ccc_g2.iloc[0]) is False
    assert bool(aaa_g1.iloc[0]) is False
    assert construct.sign == 1
    assert construct.eligible is None


def test_flag_extra_rest_edge_and_short_week_match_hand_computation(tmp_path: Path) -> None:
    games = [
        _game(
            game_id="g1",
            season=2020,
            week=1,
            home_team="AAA",
            away_team="BBB",
            result=3.0,
            home_rest=10,
            away_rest=3,
        )
    ]
    features_path = _write_bias_battery_repo(tmp_path, games)
    features = _read_bias_battery_features(features_path)

    rest_construct = _flag_extra_rest_edge(features, (2009, 2025), {}, tmp_path)
    rt = rest_construct.table.reset_index(drop=True)
    rf = rest_construct.flag.reset_index(drop=True)
    assert bool(rf.loc[rt["team"] == "AAA"].iloc[0]) is True
    assert bool(rf.loc[rt["team"] == "BBB"].iloc[0]) is False
    assert rest_construct.sign == 1

    short_construct = _flag_short_week(features, (2009, 2025), {}, tmp_path)
    st = short_construct.table.reset_index(drop=True)
    sf = short_construct.flag.reset_index(drop=True)
    assert bool(sf.loc[st["team"] == "AAA"].iloc[0]) is False
    assert bool(sf.loc[st["team"] == "BBB"].iloc[0]) is True
    assert short_construct.sign == -1


def test_flag_west_coast_early_kickoff_matches_hand_computation(tmp_path: Path) -> None:
    games = [
        _game(
            game_id="g1",
            season=2020,
            week=1,
            home_team="NYJ",
            away_team="SEA",
            result=3.0,
            gametime="09:30",
        ),
        _game(
            game_id="g2",
            season=2020,
            week=2,
            home_team="SEA",
            away_team="NYJ",
            result=3.0,
            gametime="09:30",
        ),
        _game(
            game_id="g3",
            season=2020,
            week=3,
            home_team="NYJ",
            away_team="SEA",
            result=3.0,
            gametime="16:00",
        ),
    ]
    features_path = _write_bias_battery_repo(tmp_path, games)
    construct = _flag_west_coast_early_kickoff(
        _read_bias_battery_features(features_path), (2009, 2025), {}, tmp_path
    )
    table, flag = construct.table.reset_index(drop=True), construct.flag.reset_index(drop=True)
    sea_g1 = flag.loc[(table["game_id"] == "g1") & (table["team"] == "SEA")]
    sea_g2 = flag.loc[(table["game_id"] == "g2") & (table["team"] == "SEA")]
    sea_g3 = flag.loc[(table["game_id"] == "g3") & (table["team"] == "SEA")]
    assert bool(sea_g1.iloc[0]) is True
    assert bool(sea_g2.iloc[0]) is False
    assert bool(sea_g3.iloc[0]) is False
    assert construct.sign == -1


def test_flag_sandwich_spot_matches_hand_computation(tmp_path: Path) -> None:
    games = [
        _game(
            game_id="g1",
            season=2020,
            week=1,
            home_team="AAA",
            away_team="X1",
            result=3.0,
            div_game=1,
        ),
        _game(
            game_id="g2",
            season=2020,
            week=2,
            home_team="AAA",
            away_team="X2",
            result=3.0,
            div_game=0,
        ),
        _game(
            game_id="g3",
            season=2020,
            week=3,
            home_team="AAA",
            away_team="X3",
            result=3.0,
            div_game=1,
        ),
        _game(
            game_id="g4",
            season=2020,
            week=1,
            home_team="BBB",
            away_team="Y1",
            result=3.0,
            div_game=0,
        ),
        _game(
            game_id="g5",
            season=2020,
            week=2,
            home_team="BBB",
            away_team="Y2",
            result=3.0,
            div_game=0,
        ),
        _game(
            game_id="g6",
            season=2020,
            week=3,
            home_team="BBB",
            away_team="Y3",
            result=3.0,
            div_game=0,
        ),
    ]
    features_path = _write_bias_battery_repo(tmp_path, games)
    construct = _flag_sandwich_spot(
        _read_bias_battery_features(features_path), (2009, 2025), {}, tmp_path
    )
    table, flag = construct.table.reset_index(drop=True), construct.flag.reset_index(drop=True)
    aaa_g2 = flag.loc[(table["game_id"] == "g2") & (table["team"] == "AAA")]
    aaa_g1 = flag.loc[(table["game_id"] == "g1") & (table["team"] == "AAA")]
    aaa_g3 = flag.loc[(table["game_id"] == "g3") & (table["team"] == "AAA")]
    bbb_g5 = flag.loc[(table["game_id"] == "g5") & (table["team"] == "BBB")]
    assert bool(aaa_g2.iloc[0]) is True
    assert bool(aaa_g1.iloc[0]) is False
    assert bool(aaa_g3.iloc[0]) is False
    assert bool(bbb_g5.iloc[0]) is False
    assert construct.sign == -1


def test_flag_backup_qb_start_matches_hand_computation(tmp_path: Path) -> None:
    games = [
        _game(
            game_id="g1",
            season=2020,
            week=1,
            home_team="AAA",
            away_team="X1",
            result=3.0,
            home_qb_name="QB1",
        ),
        _game(
            game_id="g2",
            season=2020,
            week=2,
            home_team="AAA",
            away_team="X2",
            result=3.0,
            home_qb_name="QB1",
        ),
        _game(
            game_id="g3",
            season=2020,
            week=3,
            home_team="AAA",
            away_team="X3",
            result=3.0,
            home_qb_name="QB1",
        ),
        _game(
            game_id="g4",
            season=2020,
            week=4,
            home_team="AAA",
            away_team="X4",
            result=3.0,
            home_qb_name="QB2",
        ),
        _game(
            game_id="g5",
            season=2020,
            week=5,
            home_team="AAA",
            away_team="X5",
            result=3.0,
            home_qb_name="QB1",
        ),
    ]
    features_path = _write_bias_battery_repo(tmp_path, games)
    construct = _flag_backup_qb_start(
        _read_bias_battery_features(features_path), (2009, 2025), {}, tmp_path
    )
    table = construct.table.reset_index(drop=True)
    flag = construct.flag.reset_index(drop=True)
    assert construct.eligible is not None
    eligible = construct.eligible.reset_index(drop=True)

    def _row(game_id: str) -> tuple[bool, bool]:
        mask = (table["game_id"] == game_id) & (table["team"] == "AAA")
        return bool(eligible.loc[mask].iloc[0]), bool(flag.loc[mask].iloc[0])

    for game_id in ("g1", "g2", "g3"):
        elig, _ = _row(game_id)
        assert elig is False
    elig4, flagged4 = _row("g4")
    assert elig4 is True
    assert flagged4 is True
    elig5, flagged5 = _row("g5")
    assert elig5 is True
    assert flagged5 is False
    assert construct.sign == 1


def test_flag_motivation_mismatch_matches_hand_computation(tmp_path: Path) -> None:
    games = []
    for week in range(1, 11):
        win = week % 2 == 1
        games.append(
            _game(
                game_id=f"mmh{week}",
                season=2020,
                week=week,
                home_team="MMH",
                away_team=f"P{week}",
                result=10.0 if win else -10.0,
            )
        )
    for week in range(1, 10):
        win = week in (1, 2)
        games.append(
            _game(
                game_id=f"badt{week}",
                season=2020,
                week=week,
                home_team="BADT",
                away_team=f"Q{week}",
                result=10.0 if win else -10.0,
            )
        )
    games.append(
        _game(game_id="target", season=2020, week=11, home_team="MMH", away_team="BADT", result=3.0)
    )
    features_path = _write_bias_battery_repo(tmp_path, games)
    construct = _flag_motivation_mismatch(
        _read_bias_battery_features(features_path), (2009, 2025), {}, tmp_path
    )
    table, flag = construct.table.reset_index(drop=True), construct.flag.reset_index(drop=True)
    mmh_target = flag.loc[(table["game_id"] == "target") & (table["team"] == "MMH")]
    badt_target = flag.loc[(table["game_id"] == "target") & (table["team"] == "BADT")]
    mmh_week5 = flag.loc[(table["game_id"] == "mmh5") & (table["team"] == "MMH")]
    assert bool(mmh_target.iloc[0]) is True
    assert bool(badt_target.iloc[0]) is False
    assert bool(mmh_week5.iloc[0]) is False
    assert construct.sign == 1


def _referee_game(
    *,
    game_id: str,
    old_game_id: str,
    season: int,
    official_name: str,
    penalties_total: float,
    penalties_on_home: float,
    penalties_on_away: float,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "old_game_id": old_game_id,
        "season": season,
        "week": 1,
        "home_team": "HOM",
        "away_team": "AWY",
        "home_cover": 1.0,
        "spread_line": -3.0,
        "game_type": "REG",
        "official_name": official_name,
        "penalties_total": penalties_total,
        "penalties_on_home": penalties_on_home,
        "penalties_on_away": penalties_on_away,
    }


def _referee_quartile_games() -> list[dict[str, Any]]:

    return [
        _referee_game(
            game_id="g_a20",
            old_game_id="OLD_A20",
            season=2020,
            official_name="REF_A",
            penalties_total=10.0,
            penalties_on_home=6.0,
            penalties_on_away=4.0,
        ),
        _referee_game(
            game_id="g_a21",
            old_game_id="OLD_A21",
            season=2021,
            official_name="REF_A",
            penalties_total=99.0,
            penalties_on_home=1.0,
            penalties_on_away=1.0,
        ),
        _referee_game(
            game_id="g_b20",
            old_game_id="OLD_B20",
            season=2020,
            official_name="REF_B",
            penalties_total=13.0,
            penalties_on_home=6.0,
            penalties_on_away=7.0,
        ),
        _referee_game(
            game_id="g_b21",
            old_game_id="OLD_B21",
            season=2021,
            official_name="REF_B",
            penalties_total=50.0,
            penalties_on_home=2.0,
            penalties_on_away=2.0,
        ),
        _referee_game(
            game_id="g_c20",
            old_game_id="OLD_C20",
            season=2020,
            official_name="REF_C",
            penalties_total=16.0,
            penalties_on_home=5.0,
            penalties_on_away=11.0,
        ),
        _referee_game(
            game_id="g_c21",
            old_game_id="OLD_C21",
            season=2021,
            official_name="REF_C",
            penalties_total=50.0,
            penalties_on_home=2.0,
            penalties_on_away=2.0,
        ),
        _referee_game(
            game_id="g_d20",
            old_game_id="OLD_D20",
            season=2020,
            official_name="REF_D",
            penalties_total=19.0,
            penalties_on_home=4.0,
            penalties_on_away=15.0,
        ),
        _referee_game(
            game_id="g_d21",
            old_game_id="OLD_D21",
            season=2021,
            official_name="REF_D",
            penalties_total=99.0,
            penalties_on_home=1.0,
            penalties_on_away=1.0,
        ),
    ]


def _referee_battery_games() -> list[dict[str, Any]]:

    return [
        *_referee_quartile_games(),
        _referee_game(
            game_id="g_rookie21",
            old_game_id="OLD_ROOKIE21",
            season=2021,
            official_name="REF_ROOKIE",
            penalties_total=12.0,
            penalties_on_home=6.0,
            penalties_on_away=6.0,
        ),
        _referee_game(
            game_id="g_vet19",
            old_game_id="OLD_VET19",
            season=2019,
            official_name="REF_VETERAN",
            penalties_total=12.0,
            penalties_on_home=6.0,
            penalties_on_away=6.0,
        ),
        _referee_game(
            game_id="g_vet20",
            old_game_id="OLD_VET20",
            season=2020,
            official_name="REF_VETERAN",
            penalties_total=12.0,
            penalties_on_home=6.0,
            penalties_on_away=6.0,
        ),
        _referee_game(
            game_id="g_vet21",
            old_game_id="OLD_VET21",
            season=2021,
            official_name="REF_VETERAN",
            penalties_total=12.0,
            penalties_on_home=6.0,
            penalties_on_away=6.0,
        ),
    ]


def _write_referee_battery_repo(tmp_path: Path, games: list[dict[str, Any]]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_cover",
        "spread_line",
        "game_type",
    ]
    features = pd.DataFrame([{k: g[k] for k in feature_cols} for g in games])
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    schedules = pd.DataFrame(
        [{"game_id": g["game_id"], "old_game_id": g["old_game_id"]} for g in games]
    )
    raw_dir = tmp_path / "data" / "raw" / "20200101T000000Z"
    raw_dir.mkdir(parents=True)
    schedules.to_parquet(raw_dir / "schedules.parquet")

    officials = pd.DataFrame(
        [
            {
                "game_id": g["old_game_id"],
                "official_name": g["official_name"],
                "position": "Referee",
                "season": g["season"],
                "season_type": "REG",
            }
            for g in games
        ]
    )
    game_penalties = pd.DataFrame(
        [
            {
                "game_id": g["game_id"],
                "penalties_total": g["penalties_total"],
                "penalties_on_home": g["penalties_on_home"],
                "penalties_on_away": g["penalties_on_away"],
            }
            for g in games
        ]
    )
    officials_dir = tmp_path / "data" / "raw" / "officials" / "20200101T000000Z"
    officials_dir.mkdir(parents=True)
    officials.to_parquet(officials_dir / "officials.parquet")
    game_penalties.to_parquet(officials_dir / "game_penalties.parquet")
    return features_path


def _read_referee_battery_features(features_path: Path) -> pd.DataFrame:
    return pd.read_parquet(features_path)


def test_referee_trait_table_requires_officials_snapshot(tmp_path: Path) -> None:
    features_path = _write_referee_battery_repo(tmp_path, _referee_battery_games())
    features = _read_referee_battery_features(features_path)
    empty_root = tmp_path / "empty"
    (empty_root / "data" / "raw" / "20200101T000000Z").mkdir(parents=True)
    pd.read_parquet(
        tmp_path / "data" / "raw" / "20200101T000000Z" / "schedules.parquet"
    ).to_parquet(empty_root / "data" / "raw" / "20200101T000000Z" / "schedules.parquet")
    with pytest.raises(ExperimentRunnerError, match=r"No data/raw/officials/\*/officials\.parquet"):
        _flag_referee_penalty_rate_top_quartile(features, (2009, 2025), {}, empty_root)


def test_flag_referee_penalty_rate_quartiles_use_the_prior_season_lag(tmp_path: Path) -> None:
    features_path = _write_referee_battery_repo(tmp_path, _referee_quartile_games())
    features = _read_referee_battery_features(features_path)

    top = _flag_referee_penalty_rate_top_quartile(features, (2009, 2025), {}, tmp_path)
    table, flag = top.table.reset_index(drop=True), top.flag.reset_index(drop=True)
    home_2021 = table["is_home"] & (table["season"] == 2021)
    flagged_games = set(table.loc[home_2021 & flag, "game_id"])
    assert flagged_games == {"g_d21"}
    assert top.sign == 1
    assert top.reliability_pairs == 4
    assert top.reliability == pytest.approx(0.0)

    bottom = _flag_referee_penalty_rate_bottom_quartile(features, (2009, 2025), {}, tmp_path)
    btable, bflag = bottom.table.reset_index(drop=True), bottom.flag.reset_index(drop=True)
    bhome_2021 = btable["is_home"] & (btable["season"] == 2021)
    assert set(btable.loc[bhome_2021 & bflag, "game_id"]) == {"g_a21"}
    assert bottom.sign == -1


def test_flag_referee_home_penalty_tilt_quartiles_use_the_prior_season_lag(tmp_path: Path) -> None:
    features_path = _write_referee_battery_repo(tmp_path, _referee_quartile_games())
    features = _read_referee_battery_features(features_path)

    top = _flag_referee_home_penalty_tilt_top_quartile(features, (2009, 2025), {}, tmp_path)
    table, flag = top.table.reset_index(drop=True), top.flag.reset_index(drop=True)
    home_2021 = table["is_home"] & (table["season"] == 2021)
    assert set(table.loc[home_2021 & flag, "game_id"]) == {"g_d21"}
    assert top.sign == 1

    bottom = _flag_referee_home_penalty_tilt_bottom_quartile(features, (2009, 2025), {}, tmp_path)
    btable, bflag = bottom.table.reset_index(drop=True), bottom.flag.reset_index(drop=True)
    bhome_2021 = btable["is_home"] & (btable["season"] == 2021)
    assert set(btable.loc[bhome_2021 & bflag, "game_id"]) == {"g_a21"}
    assert bottom.sign == -1


def test_flag_referee_veteran_and_rookie_home_cover_match_hand_computation(tmp_path: Path) -> None:
    features_path = _write_referee_battery_repo(tmp_path, _referee_battery_games())
    features = _read_referee_battery_features(features_path)

    veteran = _flag_referee_veteran_home_cover(
        features, (2009, 2025), {"veteran_threshold": 2}, tmp_path
    )
    vtable, vflag = veteran.table.reset_index(drop=True), veteran.flag.reset_index(drop=True)
    vhome_2021 = vtable["is_home"] & (vtable["season"] == 2021)
    assert set(vtable.loc[vhome_2021 & vflag, "game_id"]) == {"g_vet21"}
    assert veteran.sign == -1
    assert veteran.reliability is None

    rookie = _flag_referee_rookie_home_cover(features, (2009, 2025), {}, tmp_path)
    rtable, rflag = rookie.table.reset_index(drop=True), rookie.flag.reset_index(drop=True)
    rhome_2021 = rtable["is_home"] & (rtable["season"] == 2021)
    assert set(rtable.loc[rhome_2021 & rflag, "game_id"]) == {"g_rookie21"}
    assert rookie.sign == 1
    assert rookie.reliability is None


def test_referee_flags_do_not_use_this_games_own_penalty_count(tmp_path: Path) -> None:

    games = _referee_quartile_games()
    features_path = _write_referee_battery_repo(tmp_path, games)
    features = _read_referee_battery_features(features_path)
    baseline = _flag_referee_penalty_rate_top_quartile(features, (2009, 2025), {}, tmp_path)
    btable = baseline.table.reset_index(drop=True)
    bflag = baseline.flag.reset_index(drop=True)
    baseline_flag = bool(bflag.loc[(btable["game_id"] == "g_d21") & btable["is_home"]].iloc[0])
    assert baseline_flag is True

    mutated_root = tmp_path / "mutated"
    mutated_games = [dict(g) for g in games]
    for g in mutated_games:
        if g["game_id"] == "g_d21":
            g["penalties_total"] = 1.0
            g["penalties_on_home"] = 0.0
            g["penalties_on_away"] = 0.0
    features_path_mutated = _write_referee_battery_repo(mutated_root, mutated_games)
    features_mutated = _read_referee_battery_features(features_path_mutated)
    mutated = _flag_referee_penalty_rate_top_quartile(
        features_mutated, (2009, 2025), {}, mutated_root
    )
    mtable = mutated.table.reset_index(drop=True)
    mflag = mutated.flag.reset_index(drop=True)
    mutated_flag = bool(mflag.loc[(mtable["game_id"] == "g_d21") & mtable["is_home"]].iloc[0])
    assert mutated_flag is True


def _write_game_penalty_types_fixture(
    officials_dir: Path, games: list[dict[str, Any]], penalty_type: str
) -> None:

    game_penalty_types = pd.DataFrame(
        [
            {
                "game_id": g["game_id"],
                "penalty_type": penalty_type,
                "penalties_total": g["penalties_total"],
                "penalties_on_home": g["penalties_on_home"],
                "penalties_on_away": g["penalties_on_away"],
            }
            for g in games
        ]
    )
    game_penalty_types.to_parquet(officials_dir / "game_penalty_types.parquet")


def test_referee_type_trait_uses_the_prior_season_lag(tmp_path: Path) -> None:

    games = _referee_quartile_games()
    _write_referee_battery_repo(tmp_path, games)
    officials_dir = tmp_path / "data" / "raw" / "officials" / "20200101T000000Z"
    _write_game_penalty_types_fixture(officials_dir, games, "Offensive Holding")

    trait = _build_referee_type_trait_data(tmp_path, "Offensive Holding")
    row = trait.game_trait.loc[trait.game_trait["game_id"] == "g_d21"].iloc[0]
    assert int(row["lag_type_quartile"]) == 4
    row_a = trait.game_trait.loc[trait.game_trait["game_id"] == "g_a21"].iloc[0]
    assert int(row_a["lag_type_quartile"]) == 1
    assert trait.reliability_pairs == 4
    assert trait.penalty_type == "Offensive Holding"


def test_referee_type_trait_does_not_use_this_games_own_penalty_type_count(
    tmp_path: Path,
) -> None:

    games = _referee_quartile_games()
    _write_referee_battery_repo(tmp_path, games)
    officials_dir = tmp_path / "data" / "raw" / "officials" / "20200101T000000Z"
    _write_game_penalty_types_fixture(officials_dir, games, "Offensive Holding")
    baseline = _build_referee_type_trait_data(tmp_path, "Offensive Holding")
    baseline_row = baseline.game_trait.loc[baseline.game_trait["game_id"] == "g_d21"].iloc[0]
    assert int(baseline_row["lag_type_quartile"]) == 4

    mutated_root = tmp_path / "mutated"
    mutated_games = [dict(g) for g in games]
    for g in mutated_games:
        if g["game_id"] == "g_d21":
            g["penalties_total"] = 1.0
            g["penalties_on_home"] = 0.0
            g["penalties_on_away"] = 0.0
    _write_referee_battery_repo(mutated_root, mutated_games)
    mutated_officials_dir = mutated_root / "data" / "raw" / "officials" / "20200101T000000Z"
    _write_game_penalty_types_fixture(mutated_officials_dir, mutated_games, "Offensive Holding")
    mutated = _build_referee_type_trait_data(mutated_root, "Offensive Holding")
    mutated_row = mutated.game_trait.loc[mutated.game_trait["game_id"] == "g_d21"].iloc[0]
    assert int(mutated_row["lag_type_quartile"]) == 4


def test_block_bootstrap_subset_gap_recovers_a_known_gap() -> None:
    df = pd.DataFrame(
        {
            "week_block": [1, 1, 2, 2],
            "team_covered": [1.0, 0.0, 1.0, 0.0],
        }
    )
    flag = pd.Series([True, False, True, False])
    draws = _block_bootstrap_subset_gap(
        df, flag=flag, value_col="team_covered", block_col="week_block", samples=200, seed=1
    )
    assert len(draws) == 200
    assert np.allclose(draws, 100.0)


def test_run_subset_bias_experiment_rejects_unknown_builder(tmp_path: Path) -> None:
    spec = experiment_spec_from_payload(
        _spec_payload(construct={"flag_builder": "not_a_real_builder", "params": {}})
    )
    with pytest.raises(ExperimentRunnerError, match="Unknown construct\\.flag_builder"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_run_subset_bias_experiment_opener_grade_rejects_non_nfl_league(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_builder = FlagBuilder(
        name="cfb_test_builder",
        leagues=("cfb",),
        description="test-only",
        build=_flag_home_underdog,
    )
    monkeypatch.setitem(FLAG_BUILDERS, "cfb_test_builder", fake_builder)
    payload = _spec_payload(construct={"flag_builder": "cfb_test_builder", "params": {}})
    payload["population"] = {"league": "cfb", "seasons": [2020, 2025], "grade": "opener"}
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="only wired for league='nfl'"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_opener_graded_features_requires_needed_columns(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunnerError, match="missing columns needed for opener grading"):
        _opener_graded_features(
            pd.DataFrame({"game_id": ["g1"]}), repo_root=tmp_path, market_root=None
        )


def test_opener_graded_features_overwrites_spread_line_and_home_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 1, 1, 2],
            "gameday": ["2020-09-10", "2020-09-13", "2020-09-13", "2020-09-20"],
            "result": [10.0, -3.0, 7.0, np.nan],
            "game_type": ["REG", "REG", "POST", "REG"],
            "home_team": ["NE", "KC", "SF", "DAL"],
            "away_team": ["BUF", "LV", "SEA", "NYG"],
            "spread_line": [-3.0, 2.5, -6.0, 1.0],
            "home_cover": [1.0, 0.0, 1.0, np.nan],
            "ats_margin": [13.0, -5.5, 13.0, np.nan],
        }
    )

    def fake_build_pairing_table(
        root: Path, *, capture_kind: str, labels: tuple[str, ...], schedule: pd.DataFrame
    ) -> pd.DataFrame:
        del root, capture_kind, labels, schedule
        return pd.DataFrame(
            {
                "game_id": ["g1", "g1", "g2"],
                "decision_label": ["tue_open", "sun_late_close", "tue_open"],
                "home_spread": [-2.0, -3.0, 3.0],
            }
        )

    def fake_close_reference_table(pairing: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
        del schedule
        rows = pairing.loc[pairing["decision_label"] == "sun_late_close"]
        return rows[["game_id"]].assign(close_home_spread=-3.0, close_source="sun_late_close")

    monkeypatch.setattr(experiment_runner_module, "build_pairing_table", fake_build_pairing_table)
    monkeypatch.setattr(
        experiment_runner_module, "close_reference_table", fake_close_reference_table
    )

    graded, note = _opener_graded_features(features, repo_root=tmp_path, market_root=tmp_path)

    assert list(graded["game_id"]) == ["g1"]
    row = graded.iloc[0]
    assert row["spread_line"] == pytest.approx(-2.0)
    assert row["ats_margin"] == pytest.approx(10.0 - (-2.0))
    assert row["home_cover"] == pytest.approx(1.0)
    assert "Opener-grade population" in note
    assert "1 REG-season games" in note


def test_run_subset_bias_experiment_opener_grade_matches_close_when_lines_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    features = _deterministic_features()
    features = features.copy()
    features["gameday"] = pd.Timestamp("2020-09-01") + pd.to_timedelta(
        7 * (features["week"] - 1), unit="D"
    )
    features["ats_margin"] = np.where(features["home_cover"] == 1.0, 1.0, -1.0)
    features["result"] = features["spread_line"] + features["ats_margin"]
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    def fake_build_pairing_table(
        root: Path, *, capture_kind: str, labels: tuple[str, ...], schedule: pd.DataFrame
    ) -> pd.DataFrame:
        del root, capture_kind, labels
        rows = []
        for _, r in schedule.iterrows():
            rows.append(
                {
                    "game_id": r["game_id"],
                    "decision_label": "tue_open",
                    "home_spread": r["spread_line"],
                }
            )
            rows.append(
                {
                    "game_id": r["game_id"],
                    "decision_label": "sun_late_close",
                    "home_spread": r["spread_line"],
                }
            )
        return pd.DataFrame(rows)

    def fake_close_reference_table(pairing: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
        del schedule
        rows = pairing.loc[pairing["decision_label"] == "sun_late_close"]
        return (
            rows[["game_id", "home_spread"]]
            .rename(columns={"home_spread": "close_home_spread"})
            .assign(close_source="sun_late_close")
        )

    monkeypatch.setattr(experiment_runner_module, "build_pairing_table", fake_build_pairing_table)
    monkeypatch.setattr(
        experiment_runner_module, "close_reference_table", fake_close_reference_table
    )

    close_spec = experiment_spec_from_payload(_spec_payload(samples=500))
    close_result = run_subset_bias_experiment(
        close_spec, repo_root=tmp_path, features_path=features_path
    )

    opener_payload = _spec_payload(samples=500)
    opener_payload["population"] = {**opener_payload["population"], "grade": "opener"}  # type: ignore[dict-item]
    opener_spec = experiment_spec_from_payload(opener_payload)
    opener_result = run_subset_bias_experiment(
        opener_spec,
        repo_root=tmp_path,
        features_path=features_path,
        market_root=tmp_path / "market",
    )

    assert opener_result.n_flag == close_result.n_flag
    assert opener_result.n_total == close_result.n_total
    assert opener_result.effect == pytest.approx(close_result.effect)
    assert opener_result.primary.estimate == pytest.approx(close_result.primary.estimate)
    assert opener_result.primary.lower == pytest.approx(close_result.primary.lower)
    assert opener_result.primary.upper == pytest.approx(close_result.primary.upper)
    assert "Opener-grade population" in opener_result.population_note


def test_run_subset_bias_experiment_rejects_split_half_on_a_traitless_builder(
    tmp_path: Path,
) -> None:
    payload = _spec_payload(reliability_check={"method": "split_half"})
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="no persistent per-entity trait"):
        run_subset_bias_experiment(spec, repo_root=tmp_path)


def test_run_subset_bias_experiment_rejects_a_missing_feature_table(tmp_path: Path) -> None:
    spec = experiment_spec_from_payload(_spec_payload())
    with pytest.raises(ExperimentRunnerError, match="Feature table not found"):
        run_subset_bias_experiment(
            spec, repo_root=tmp_path, features_path=tmp_path / "absent.parquet"
        )


def test_run_feature_arm_experiment_rejects_opener_grade(tmp_path: Path) -> None:
    payload = _feature_arm_spec_payload()
    payload["population"] = {**payload["population"], "grade": "opener"}  # type: ignore[dict-item]
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="only supports population\\.grade='close'"):
        run_feature_arm_experiment(spec, repo_root=tmp_path)


def test_run_feature_arm_experiment_rejects_non_nfl_league(tmp_path: Path) -> None:
    payload = _feature_arm_spec_payload()
    payload["population"] = {**payload["population"], "league": "cfb"}  # type: ignore[dict-item]
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="only wired for league='nfl'"):
        run_feature_arm_experiment(spec, repo_root=tmp_path)


def test_run_feature_arm_experiment_rejects_split_half(tmp_path: Path) -> None:
    payload = _feature_arm_spec_payload(reliability_check={"method": "split_half"})
    spec = experiment_spec_from_payload(payload)
    with pytest.raises(ExperimentRunnerError, match="no persistent per-entity trait"):
        run_feature_arm_experiment(spec, repo_root=tmp_path)


def test_run_feature_arm_experiment_rejects_a_missing_feature_table(tmp_path: Path) -> None:
    spec = experiment_spec_from_payload(_feature_arm_spec_payload())
    with pytest.raises(ExperimentRunnerError, match="Feature table not found"):
        run_feature_arm_experiment(
            spec, repo_root=tmp_path, features_path=tmp_path / "absent.parquet"
        )


def test_run_experiment_dispatches_feature_arm_through_run_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = experiment_spec_from_payload(_feature_arm_spec_payload())
    monkeypatch.setattr(
        experiment_runner_module,
        "run_feature_arm_experiment",
        lambda spec, *, repo_root, features_path=None: "sentinel",
    )
    assert run_experiment(spec, repo_root=tmp_path) == "sentinel"


def _deterministic_features(n_weeks: int = 10) -> pd.DataFrame:

    rows = []
    game = 0
    for week in range(1, n_weeks + 1):
        for pair in range(4):
            game += 1
            home_dog = pair == 0
            spread_line = -3.0 if home_dog else 3.0
            if home_dog:
                home_cover = 1.0 if week % 5 != 0 else 0.0
            else:
                home_cover = 1.0 if pair % 2 == 0 else 0.0
            rows.append(
                {
                    "game_id": f"g{game}",
                    "season": 2020,
                    "week": week,
                    "home_team": f"T{pair}A",
                    "away_team": f"T{pair}B",
                    "home_cover": home_cover,
                    "spread_line": spread_line,
                    "game_type": "REG",
                }
            )
    return pd.DataFrame(rows)


def test_run_subset_bias_experiment_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    spec = experiment_spec_from_payload(_spec_payload(samples=2000))
    result = run_subset_bias_experiment(spec, repo_root=tmp_path, features_path=features_path)

    home_dog_rows = features.loc[features["spread_line"] < 0]
    expected_subset_cover = float(home_dog_rows["home_cover"].mean())
    assert expected_subset_cover == pytest.approx(0.8)
    assert result.n_total == 80
    assert result.n_flag == 10
    assert result.fraction_of_slate == pytest.approx(result.n_flag / result.n_total)
    assert result.effect == pytest.approx(result.raw_gap_pct * result.fraction_of_slate, abs=1e-9)
    assert result.raw_gap_pct > 0.0
    assert result.sign == 1

    assert 0.0 <= result.primary.probability_positive <= 1.0
    assert result.primary.lower <= result.primary.estimate <= result.primary.upper
    assert result.secondary is not None
    assert result.classification.classification in ("unresolved_below_power", "refuted_mechanism")


def test_run_feature_arm_experiment_end_to_end_on_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    features_path = tmp_path / "features.parquet"
    pd.DataFrame({"game_id": [f"g{i}" for i in range(1, 9)]}).to_parquet(features_path)

    game_ids = [f"g{i}" for i in range(1, 9)]
    seasons = [2020] * 4 + [2021] * 4
    weeks = [1, 1, 2, 2, 1, 1, 2, 2]
    home_cover = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def fake_walk_forward_outcomes(
        features_arg: pd.DataFrame,
        *,
        start_season: int,
        end_season: int | None = None,
        regressor: str = "ridge",
        min_edge: float = 0.02,
        min_train_games: int = 500,
        feature_profile: str = "base",
        methods: tuple[str, ...] = ("market_residual",),
        ridge_alpha: float = 10.0,
    ) -> Any:
        del (
            features_arg,
            start_season,
            end_season,
            regressor,
            min_edge,
            min_train_games,
            feature_profile,
            methods,
        )
        if ridge_alpha == 10.0:
            probability = [0.5] * 8
        else:
            probability = [0.9 if c == 1.0 else 0.1 for c in home_cover]
        predictions = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": seasons,
                "week": weeks,
                "home_cover": home_cover,
                "home_cover_probability": probability,
            }
        )
        return SimpleNamespace(predictions=predictions)

    monkeypatch.setattr(
        experiment_runner_module, "walk_forward_outcomes", fake_walk_forward_outcomes
    )

    payload = _feature_arm_spec_payload(
        endpoints={"primary": "accuracy", "secondary": ["brier", "logloss"]}, samples=500
    )
    spec = experiment_spec_from_payload(payload)
    result = run_feature_arm_experiment(spec, repo_root=tmp_path, features_path=features_path)

    assert result.paired_games == 8
    assert result.accuracy_primary.estimate == pytest.approx(25.0)
    assert result.accuracy_secondary is not None
    assert result.accuracy_secondary.estimate == pytest.approx(25.0)
    assert result.brier_primary is not None
    assert result.brier_primary.estimate == pytest.approx(0.24, abs=1e-9)
    assert result.brier_secondary is not None
    assert result.logloss_primary is not None
    assert result.logloss_primary.estimate > 0.0
    assert result.logloss_secondary is not None
    assert result.classification.classification in ("unresolved_below_power", "refuted_mechanism")


def test_run_feature_arm_experiment_omits_metrics_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path = tmp_path / "features.parquet"
    pd.DataFrame({"game_id": ["g1"]}).to_parquet(features_path)

    def fake_walk_forward_outcomes(features_arg: pd.DataFrame, **kwargs: Any) -> Any:
        del features_arg, kwargs
        predictions = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "season": [2020, 2020],
                "week": [1, 2],
                "home_cover": [1.0, 0.0],
                "home_cover_probability": [0.6, 0.4],
            }
        )
        return SimpleNamespace(predictions=predictions)

    monkeypatch.setattr(
        experiment_runner_module, "walk_forward_outcomes", fake_walk_forward_outcomes
    )
    payload = _feature_arm_spec_payload(
        endpoints={"primary": "accuracy", "secondary": []}, samples=200
    )
    spec = experiment_spec_from_payload(payload)
    result = run_feature_arm_experiment(spec, repo_root=tmp_path, features_path=features_path)

    assert result.brier_primary is None
    assert result.brier_secondary is None
    assert result.logloss_primary is None
    assert result.logloss_secondary is None


def test_run_feature_arm_experiment_reuses_an_identical_deterministic_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path = tmp_path / "features.parquet"
    pd.DataFrame({"game_id": ["g1", "g2"]}).to_parquet(features_path)
    calls = 0

    def fake_walk_forward_outcomes(features_arg: pd.DataFrame, **kwargs: Any) -> Any:
        nonlocal calls
        del features_arg, kwargs
        calls += 1
        return SimpleNamespace(
            predictions=pd.DataFrame(
                {
                    "game_id": ["g1", "g2"],
                    "season": [2020, 2020],
                    "week": [1, 2],
                    "home_cover": [1.0, 0.0],
                    "home_cover_probability": [0.6, 0.4],
                }
            )
        )

    monkeypatch.setattr(
        experiment_runner_module, "walk_forward_outcomes", fake_walk_forward_outcomes
    )
    payload = _feature_arm_spec_payload(
        construct={
            "baseline": {"feature_profile": "base", "ridge_alpha": 10.0},
            "candidate": {"feature_profile": "base", "ridge_alpha": 10.0},
        },
        endpoints={"primary": "accuracy", "secondary": []},
        samples=20,
    )

    result = run_feature_arm_experiment(
        experiment_spec_from_payload(payload), repo_root=tmp_path, features_path=features_path
    )

    assert calls == 1
    assert result.paired_games == 2
    assert result.accuracy_primary.estimate == pytest.approx(0.0)


_PBP_ROOT = REPO_ROOT / "data" / "pbp" / "raw"
_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "game_features.parquet"
_LOCAL_DATA_AVAILABLE = _PBP_ROOT.is_dir() and _FEATURES_PATH.is_file()


@pytest.mark.skipif(
    not _LOCAL_DATA_AVAILABLE, reason="local PBP/feature data not present in this checkout"
)
@pytest.mark.full
def test_penalty_discipline_reproduces_the_recorded_registry_entry() -> None:

    spec = experiment_spec_from_payload(
        {
            "name": "penalty_discipline_reproduction",
            "hypothesis": (
                "Teams with the lowest lagged penalty rate cover more against the "
                "spread than teams with the highest."
            ),
            "experiment_type": "subset_bias",
            "population": {"league": "nfl", "seasons": [2009, 2025], "grade": "close"},
            "construct": {"flag_builder": "penalty_rate_quartile", "params": {}},
            "endpoints": {"primary": "accuracy", "secondary": []},
            "blocking": {"primary": "week", "secondary": "season"},
            "samples": 20000,
            "seed": 20260818,
            "reliability_check": {
                "method": "split_half",
                "reason": "penalty_rate_quartile carries a persistent team-season trait",
            },
        }
    )
    result = run_subset_bias_experiment(spec, repo_root=REPO_ROOT)

    assert result.effect == pytest.approx(0.3288, abs=5e-4)
    assert result.n_flag + result.n_complement == 4085
    assert result.primary.block_count == 277
    assert result.primary.lower == pytest.approx(-1.0389, abs=1e-3)
    assert result.primary.upper == pytest.approx(1.6849, abs=1e-3)
    assert result.primary.probability_positive == pytest.approx(0.6828, abs=1e-3)
    assert result.primary.standard_error == pytest.approx(0.6938, abs=1e-3)
    assert result.reliability == pytest.approx(0.261, abs=2e-3)
    assert result.reliability_pairs == 512
    assert result.classification.classification == "unresolved_below_power"
    assert result.classification.closing_ground is None


_GAME_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "game_features.parquet"
_GAME_FEATURES_AVAILABLE = _GAME_FEATURES_PATH.is_file()


@pytest.mark.skipif(
    not _GAME_FEATURES_AVAILABLE, reason="local game_features.parquet not present in this checkout"
)
@pytest.mark.full
def test_feature_arm_identical_arms_measure_exactly_zero_on_real_data() -> None:
    spec = experiment_spec_from_payload(
        {
            "name": "feature_arm_identity_anchor",
            "hypothesis": (
                "Two identically-configured feature_arm arms must measure zero difference."
            ),
            "experiment_type": "feature_arm",
            "population": {"league": "nfl", "seasons": [2023, 2023], "grade": "close"},
            "construct": {
                "baseline": {"feature_profile": "base", "ridge_alpha": 10.0},
                "candidate": {"feature_profile": "base", "ridge_alpha": 10.0},
            },
            "endpoints": {"primary": "accuracy", "secondary": ["brier", "logloss"]},
            "blocking": {"primary": "week", "secondary": "season"},
            "samples": 500,
            "seed": 20260819,
            "reliability_check": {
                "method": "not_applicable",
                "reason": "identity check, not a trait",
            },
        }
    )
    result = run_feature_arm_experiment(spec, repo_root=REPO_ROOT)

    assert result.paired_games > 0
    for block_result in (
        result.accuracy_primary,
        result.accuracy_secondary,
        result.brier_primary,
        result.brier_secondary,
        result.logloss_primary,
        result.logloss_secondary,
    ):
        assert block_result is not None
        assert block_result.estimate == pytest.approx(0.0, abs=1e-9)
        assert block_result.lower == pytest.approx(0.0, abs=1e-9)
        assert block_result.upper == pytest.approx(0.0, abs=1e-9)
        assert block_result.probability_positive == pytest.approx(0.5)
    assert result.classification.classification == "unresolved_below_power"


def test_registry_lock_is_exclusive_and_cleans_up(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    with (
        _RegistryLock(registry_path, timeout=0.2, poll=0.02),
        pytest.raises(ExperimentRunnerError, match="Could not acquire"),
        _RegistryLock(registry_path, timeout=0.2, poll=0.02),
    ):
        pass
    assert not registry_path.with_suffix(".json.lock").exists()


def test_run_experiment_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload(samples=500)), encoding="utf-8")
    registry_path = tmp_path / "weak_signals.json"
    artifacts_root = tmp_path / "artifacts"

    outcome = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=True,
        features_path=features_path,
        registry_path=registry_path,
        artifacts_root=artifacts_root,
    )
    assert outcome.dry_run is True
    assert outcome.artifact_directory is None
    assert outcome.registry_record is None
    assert outcome.preview["classification"] in ("unresolved_below_power", "refuted_mechanism")
    assert not registry_path.exists()
    assert not artifacts_root.exists()


def test_run_experiment_cli_writes_artifact_and_registry_then_enforces_single_write(
    tmp_path: Path,
) -> None:
    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload(samples=500)), encoding="utf-8")
    registry_path = tmp_path / "weak_signals.json"
    registry_root = tmp_path / "registry"
    artifacts_root = tmp_path / "artifacts"

    first = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=False,
        features_path=features_path,
        registry_path=registry_path,
        registry_root=registry_root,
        artifacts_root=artifacts_root,
        run_id_value="20260818T000000Z",
    )
    assert first.artifact_directory is not None
    artifact_dir = Path(first.artifact_directory)
    assert (artifact_dir / "metadata.json").is_file()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "example_subset_bias"
    assert "provenance" in metadata

    registry = load_registry(registry_path)
    assert "example_subset_bias" in registry.signals

    experiment_rows = sorted((registry_root / "experiments" / "experiment-run").glob("*.json"))
    assert [path.name for path in experiment_rows] == ["20260818T000000Z.json"]

    with pytest.raises(WeakSignalError, match="already recorded"):
        run_experiment_cli(
            spec_path,
            repo_root=tmp_path,
            dry_run=False,
            features_path=features_path,
            registry_path=registry_path,
            registry_root=registry_root,
            artifacts_root=artifacts_root,
            run_id_value="20260818T000001Z",
        )

    second = run_experiment_cli(
        spec_path,
        repo_root=tmp_path,
        dry_run=False,
        replace=True,
        features_path=features_path,
        registry_path=registry_path,
        registry_root=registry_root,
        artifacts_root=artifacts_root,
        run_id_value="20260818T000002Z",
    )
    assert second.registry_record is not None
    assert second.registry_record["recorded"] == "example_subset_bias"

    experiment_rows = sorted((registry_root / "experiments" / "experiment-run").glob("*.json"))
    assert [path.name for path in experiment_rows] == [
        "20260818T000000Z.json",
        "20260818T000001Z.json",
        "20260818T000002Z.json",
    ]


def test_experiment_run_cli_writes_only_under_the_env_isolated_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    registry_root = tmp_path / "registry"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))

    features = _deterministic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_payload(samples=500)), encoding="utf-8")

    exit_code = cli.main(["experiment", "run", str(spec_path), "--features", str(features_path)])
    assert exit_code == 0

    weak_signals_path = registry_root / "weak_signals.json"
    assert weak_signals_path.is_file()
    registry = load_registry(weak_signals_path)
    assert "example_subset_bias" in registry.signals

    experiment_rows = list((registry_root / "experiments" / "experiment-run").glob("*.json"))
    assert len(experiment_rows) == 1
    row = json.loads(experiment_rows[0].read_text(encoding="utf-8"))
    assert row["command"] == "experiment-run"
    assert row["weak_signal_name"] == "example_subset_bias"
