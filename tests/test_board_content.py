from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import board_content
from nfl_ats.board_content import GameRow
from nfl_ats.data import DataContractError
from nfl_ats.public_board import assert_spread_explorer_matches_card
from nfl_ats.spread_explorer import (
    SpreadExplorerGameParams,
    widget_home_cover_probability,
)


def _headline_artifacts(root: Path) -> tuple[dict, Path]:
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "active-model",
        "feature_table_sha256": "same-table",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "probability_method": "gaussian",
        "calibration_method": "none",
        "historical_evaluation": {"accuracy": 0.52, "games": 100, "correct": 52},
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    directory = root / "opener_evaluation" / "20260905T000000Z"
    directory.mkdir(parents=True)
    metadata = {
        "active_model_id": "active-model",
        "active_model_config": {
            "feature_profile": "weak_stack",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "target": "market_residual",
            "probability_method": "gaussian",
            "calibration_method": "none",
        },
        "provenance": {"feature_table": {"sha256": "same-table"}},
        "games": 100,
        "metrics": {"opener_accuracy_probability_rule": 0.57},
        "uncertainty": [
            {
                "metric": "opener_accuracy_probability_rule",
                "block": block,
                "lower": 0.51,
                "upper": 0.62,
            }
            for block in ("week", "season")
        ],
    }
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    from nfl_ats.public_board import PLAYED_UNION_MEMBER_IDS

    composition = root / "overlay_subset_composition" / "20260905T000001Z"
    composition.mkdir(parents=True)
    (composition / "result.json").write_text(
        json.dumps(
            {
                "source_artifact": str(directory / "per_game.parquet"),
                "n_scored_games": 100,
                "subsets": [
                    {"members": sorted(PLAYED_UNION_MEMBER_IDS), "candidate_accuracy": 0.59}
                ],
            }
        ),
        encoding="utf-8",
    )
    from nfl_ats.four_overlay_composition import COMPOSITION_ORDER, POLICY_ID

    served = root / "unserved_tilt_marginals" / "20260905T000002Z"
    served.mkdir(parents=True)
    (served / "result.json").write_text(
        json.dumps(
            {
                "active_model_id": "active-model",
                "n_scored_games": 100,
                "seasons": [2020, 2025],
                "served_card_accuracy": 0.5688622754491018,
                "served_policy": {
                    "policy_id": POLICY_ID,
                    "members": list(COMPOSITION_ORDER),
                },
            }
        ),
        encoding="utf-8",
    )
    return active, path


def test_baseline_artifact_changes_board_and_readme_together(tmp_path: Path) -> None:
    from dataclasses import replace

    from _board_content_fixtures import build_fixture_content

    from nfl_ats.board_terminal import render
    from nfl_ats.public_board import load_baseline_measurement
    from nfl_ats.readme_state import render_active_model_block

    active, path = _headline_artifacts(tmp_path)
    fixture = build_fixture_content()
    for accuracy in (0.57, 0.61):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["metrics"]["opener_accuracy_probability_rule"] = accuracy
        path.write_text(json.dumps(metadata), encoding="utf-8")
        measurement = load_baseline_measurement(tmp_path)
        headline = board_content._build_headline_stats(
            tmp_path,
            active,
            prospective_scoreboard=fixture.headline.prospective_scoreboard,
        )
        assert measurement.accuracy == accuracy
        assert measurement.season_interval == (0.51, 0.62)
        assert measurement.played_accuracy == 0.59
        assert headline.raw_model_ci == (51.0, 62.0)
        assert f"{accuracy:.1%}" in render(replace(fixture, headline=headline))
        assert f"**{accuracy:.2%}**" in render_active_model_block(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_profile", "player"),
        ("regressor", "other"),
        ("ridge_alpha", 2000.0),
        ("target", "margin"),
        ("probability_method", "ecdf"),
        ("calibration_method", "isotonic"),
    ],
)
def test_same_table_different_recipe_never_matches(
    tmp_path: Path, field: str, value: object
) -> None:
    from nfl_ats.public_board import (
        find_matching_opener_evaluation,
        find_matching_overlay_composition,
        load_baseline_measurement,
    )

    active, path = _headline_artifacts(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["active_model_config"][field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert find_matching_opener_evaluation(tmp_path, active) is None
    assert find_matching_overlay_composition(tmp_path, active) is None
    with pytest.raises(ValueError, match="active model"):
        load_baseline_measurement(tmp_path, active)


def test_wrong_model_id_raises_even_when_table_and_recipe_match(tmp_path: Path) -> None:
    from nfl_ats.public_board import load_baseline_measurement

    active, path = _headline_artifacts(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["active_model_id"] = "different-model"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="active model"):
        load_baseline_measurement(tmp_path, active)
    with pytest.raises(board_content.NumberProvenanceError):
        board_content.verify_number_provenance(tmp_path)


def test_legacy_ecdf_evaluation_cannot_supply_gaussian_headline(tmp_path: Path) -> None:
    from nfl_ats.public_board import find_matching_opener_evaluation

    active, path = _headline_artifacts(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["active_model_id"]
    del metadata["active_model_config"]["probability_method"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert find_matching_opener_evaluation(tmp_path, active) is None
    active["probability_method"] = "ecdf"
    assert find_matching_opener_evaluation(tmp_path, active) is not None


def test_interval_cannot_leak_from_a_newer_different_model(tmp_path: Path) -> None:
    from nfl_ats.public_board import load_baseline_measurement

    _active, path = _headline_artifacts(tmp_path)
    newer = path.parent.parent / "20260906T000000Z"
    newer.mkdir()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["active_model_id"] = "other-model"
    metadata["uncertainty"][0]["lower"] = 0.99
    (newer / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    measurement = load_baseline_measurement(tmp_path)
    assert measurement.week_interval == (0.51, 0.62)
    assert measurement.directory == path.parent


def test_composition_cannot_match_only_a_directory_timestamp(tmp_path: Path) -> None:
    from nfl_ats.public_board import find_matching_overlay_composition

    _active, path = _headline_artifacts(tmp_path)
    composition = tmp_path / "overlay_subset_composition" / "20260905T000001Z" / "result.json"
    payload = json.loads(composition.read_text(encoding="utf-8"))
    payload["source_artifact"] = str(
        tmp_path / "other-tree" / "opener_evaluation" / path.parent.name / "per_game.parquet"
    )
    composition.write_text(json.dumps(payload), encoding="utf-8")
    assert find_matching_overlay_composition(tmp_path) is None


def _game(pick_team: str, home: str, away: str) -> GameRow:
    return GameRow(
        game_id="2026_01_TEST",
        gameday=__import__("datetime").date(2026, 9, 10),
        weekday_name="Thursday",
        home=home,
        away=away,
        market_spread=3.0,
        pick_team=pick_team,
        pick_probability=0.55,
        confidence_word="lean",
        is_best=True,
        is_flipped=False,
    )


def _params(game_id: str = "2026_01_TEST") -> SpreadExplorerGameParams:
    return SpreadExplorerGameParams(
        game_id=game_id,
        home_team="SEA",
        away_team="NE",
        center=1.2,
        residual_mean=0.4,
        residual_std=6.5,
        card_line=3.0,
        card_home_cover_probability=widget_home_cover_probability(3.0, 1.2, 0.4, 6.5),
    )


def test_cover_curve_empty_when_no_game() -> None:
    assert board_content._build_cover_curve(pd.DataFrame(), None) == ()


def test_cover_curve_prefers_card_verified_mapping_over_legacy_sweep() -> None:
    game = _game("SEA", home="SEA", away="NE")
    sweep = pd.DataFrame(
        {
            "game_id": ["2026_01_TEST", "2026_01_TEST"],
            "line_offset": [-1.0, 0.0],
            "home_cover_probability": [0.4, 0.5],
        }
    )
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(sweep, game, params)
    assert len(curve) > 2
    zero_point = next(point for point in curve if point.offset == 0.0)
    assert zero_point.probability == pytest.approx(params[game.game_id].card_home_cover_probability)
    assert zero_point.probability != pytest.approx(0.5)


def test_cover_curve_falls_back_to_gaussian_when_sweep_is_empty() -> None:
    game = _game("SEA", home="SEA", away="NE")
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(pd.DataFrame(), game, params)
    assert len(curve) > 2
    zero_point = next(point for point in curve if point.offset == 0.0)
    assert zero_point.probability == pytest.approx(params[game.game_id].card_home_cover_probability)


def test_cover_curve_gaussian_fallback_orients_to_away_pick() -> None:
    game = _game("NE", home="SEA", away="NE")
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(pd.DataFrame(), game, params)
    zero_point = next(point for point in curve if point.offset == 0.0)
    home_probability = params[game.game_id].card_home_cover_probability
    assert zero_point.probability == pytest.approx(1.0 - home_probability)


def test_cover_curve_empty_when_no_sweep_and_no_gaussian_params() -> None:
    game = _game("SEA", home="SEA", away="NE")
    assert board_content._build_cover_curve(pd.DataFrame(), game, None) == ()
    assert board_content._build_cover_curve(pd.DataFrame(), game, {}) == ()


def test_load_spread_explorer_params_skips_non_gaussian_methods() -> None:
    metadata = {"probability_method": "ecdf"}
    result = board_content._load_spread_explorer_params(
        metadata, pd.DataFrame({"game_id": ["x"]}), board_content._default_data_root()
    )
    assert result == {}


def test_load_spread_explorer_params_skips_empty_predictions() -> None:
    metadata = {"probability_method": "gaussian"}
    result = board_content._load_spread_explorer_params(
        metadata, pd.DataFrame(), board_content._default_data_root()
    )
    assert result == {}


def test_assert_spread_explorer_matches_card_guard_fires_on_mismatch() -> None:

    params = {"2026_01_TEST": _params()}
    predictions = pd.DataFrame({"game_id": ["2026_01_TEST"], "home_cover_probability": [0.999]})
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(params, predictions)


def test_assert_spread_explorer_matches_card_guard_passes_on_match() -> None:
    game_params = _params()
    params = {game_params.game_id: game_params}
    predictions = pd.DataFrame(
        {
            "game_id": [game_params.game_id],
            "home_cover_probability": [game_params.card_home_cover_probability],
        }
    )
    assert_spread_explorer_matches_card(params, predictions)


def test_guard_fires_for_any_game_in_a_multi_game_week() -> None:

    good_game_params = _params("2026_01_GOOD")
    bad_game_params = SpreadExplorerGameParams(
        game_id="2026_01_BAD",
        home_team="KC",
        away_team="DEN",
        center=0.5,
        residual_mean=0.1,
        residual_std=7.0,
        card_line=-3.0,
        card_home_cover_probability=widget_home_cover_probability(-3.0, 0.5, 0.1, 7.0),
    )
    params = {good_game_params.game_id: good_game_params, bad_game_params.game_id: bad_game_params}
    predictions = pd.DataFrame(
        {
            "game_id": [good_game_params.game_id, bad_game_params.game_id],
            "home_cover_probability": [
                good_game_params.card_home_cover_probability,
                0.999,
            ],
        }
    )
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(params, predictions)


def test_cover_curve_fallback_offsets_match_sweep_half_width_and_step() -> None:

    offsets = board_content._COVER_CURVE_FALLBACK_OFFSETS
    assert math.isclose(min(offsets), -board_content.SWEEP_HALF_WIDTH)
    assert math.isclose(max(offsets), board_content.SWEEP_HALF_WIDTH)
    assert offsets == tuple(sorted(offsets))


def test_load_source_policy_view_absent_block_is_not_recorded() -> None:

    view = board_content._load_source_policy_view({"season": 2026, "week": 1}, None)
    assert view.recorded is False
    assert view.card_state == board_content.SOURCE_POLICY_NOT_RECORDED
    assert view.card_state_label == "NOT RECORDED"
    assert view.rows == ()
    assert view.evaluated_at is None


def test_load_source_policy_view_reads_full_block() -> None:

    metadata = {
        "source_policy": {
            "state": "degraded",
            "evaluated_at_utc": "2026-09-03T14:00:00+00:00",
            "sources": {
                "odds_opener": {
                    "state": "complete",
                    "reason": "snapshot is 30.0 min old, inside the 180 min budget",
                    "age_minutes": 30.0,
                    "budget_minutes": 180,
                    "fallback": "publish on the newest opener snapshot on disk",
                },
                "injuries_nflverse": {
                    "state": "degraded",
                    "reason": "no snapshot present (budget 120 min)",
                    "age_minutes": None,
                    "budget_minutes": 120,
                    "fallback": "the previous weekly snapshot is reused",
                },
            },
            "unobserved": ["airnow_weather"],
        }
    }
    view = board_content._load_source_policy_view(metadata, None)
    assert view.recorded is True
    assert view.card_state == "degraded"
    assert view.card_state_label == "DEGRADED"
    assert view.evaluated_at == "2026-09-03T14:00:00+00:00"

    by_id = {row.source_id: row for row in view.rows}
    assert by_id["odds_opener"].state == "complete"
    assert by_id["odds_opener"].budget_minutes == 180
    assert by_id["odds_opener"].observed_at == "2026-09-03T13:30:00+00:00"
    assert by_id["odds_opener"].observed_at_text == "as-of 2026-09-03 13:30 UTC"
    assert by_id["injuries_nflverse"].state == "degraded"
    assert by_id["injuries_nflverse"].observed_at is None
    assert by_id["injuries_nflverse"].observed_at_text == "no snapshot"
    assert by_id["airnow_weather"].state == "unobserved"


def test_load_source_policy_view_malformed_state_falls_back_to_not_recorded() -> None:

    metadata = {"source_policy": {"state": "not-a-real-state", "sources": {}}}
    view = board_content._load_source_policy_view(metadata, None)
    assert view.recorded is True
    assert view.card_state == board_content.SOURCE_POLICY_NOT_RECORDED


def test_load_source_policy_view_prefers_the_persisted_file_over_metadata(
    tmp_path: Path,
) -> None:

    (tmp_path / "source_policy.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "evaluated_at_utc": "2026-09-03T14:00:00+00:00",
                "sources": {
                    "odds_opener": {
                        "state": "complete",
                        "reason": "snapshot is 10.0 min old, inside the 180 min budget",
                        "age_minutes": 10.0,
                        "budget_minutes": 180,
                        "fallback": "publish on the newest opener snapshot on disk",
                    }
                },
                "unobserved": [],
            }
        ),
        encoding="utf-8",
    )
    metadata_with_a_different_block = {
        "source_policy": {"state": "blocked", "sources": {}, "unobserved": []}
    }
    view = board_content._load_source_policy_view(metadata_with_a_different_block, tmp_path)
    assert view.recorded is True
    assert view.card_state == "complete"
    assert [row.source_id for row in view.rows] == ["odds_opener"]

    empty_dir = tmp_path / "no_file_here"
    empty_dir.mkdir()
    fallback_view = board_content._load_source_policy_view(
        metadata_with_a_different_block, empty_dir
    )
    assert fallback_view.card_state == "blocked"

    assert board_content._load_source_policy_view({}, empty_dir).recorded is False


def test_load_source_policy_view_without_data_root_stays_not_recorded(tmp_path: Path) -> None:

    empty_dir = tmp_path / "no_file_here"
    empty_dir.mkdir()
    view = board_content._load_source_policy_view({}, empty_dir)
    assert view.recorded is False
    assert view.computed_live is False
    assert view.card_state == board_content.SOURCE_POLICY_NOT_RECORDED


def test_load_source_policy_view_computes_live_report_when_nothing_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.delenv("SPORTRADAR_API_KEY", raising=False)
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    view = board_content._load_source_policy_view(
        {},
        None,
        data_root=data_root,
        artifacts_root=artifacts_root,
        now=datetime(2026, 9, 5, tzinfo=UTC),
    )
    assert view.recorded is False
    assert view.computed_live is True
    assert view.card_state == "degraded"
    assert view.rows
    by_id = {row.source_id: row for row in view.rows}
    assert by_id.pop("player_arrests").state == "unobserved"
    assert by_id.pop("injuries_sportradar").state == "not_configured"
    assert all(row.state == "degraded" for row in by_id.values())


def test_load_source_policy_view_prefers_persisted_over_live(tmp_path: Path) -> None:

    (tmp_path / "source_policy.json").write_text(
        json.dumps({"state": "complete", "sources": {}, "unobserved": []}), encoding="utf-8"
    )
    view = board_content._load_source_policy_view(
        {}, tmp_path, data_root=tmp_path, artifacts_root=tmp_path
    )
    assert view.recorded is True
    assert view.computed_live is False
    assert view.card_state == "complete"


def test_game_row_explanation_text_defaults_to_not_recorded() -> None:
    game = _game("SEA", home="SEA", away="NE")
    assert game.explanation_text == board_content.EXPLANATION_NOT_RECORDED_TEXT


def test_load_pick_explanations_returns_empty_when_forecast_dir_is_none() -> None:
    assert board_content._load_pick_explanations(None) == {}


def test_load_pick_explanations_returns_empty_when_file_absent(tmp_path: Path) -> None:
    assert board_content._load_pick_explanations(tmp_path) == {}


def test_load_pick_explanations_returns_empty_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "explanations.json").write_text("not valid json", encoding="utf-8")
    assert board_content._load_pick_explanations(tmp_path) == {}


def test_load_pick_explanations_skips_a_row_with_a_malformed_nested_field(tmp_path: Path) -> None:

    (tmp_path / "explanations.json").write_text(
        json.dumps(
            {
                "explanations": [
                    {"game_id": "2026_01_BAD", "text": "should be skipped", "market_line": "oops"},
                    {"game_id": "2026_01_GOOD", "text": "a fine explanation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    texts = board_content._load_pick_explanations(tmp_path)
    assert texts == {"2026_01_GOOD": "a fine explanation"}


def test_load_pick_explanations_reads_real_file(tmp_path: Path) -> None:
    (tmp_path / "explanations.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "count": 2,
                "explanations": [
                    {"game_id": "2026_01_SEA_NE", "text": "SEA at NE: the market line is -3."},
                    {"game_id": "2026_01_KC_DEN", "text": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    texts = board_content._load_pick_explanations(tmp_path)
    assert texts == {"2026_01_SEA_NE": "SEA at NE: the market line is -3."}


def test_load_tiebreaker_view_not_published_when_forecast_dir_is_none() -> None:
    view = board_content._load_tiebreaker_view(None, {})
    assert view.recorded is False
    assert view.note == board_content.TIEBREAKER_NOT_PUBLISHED_TEXT


def test_load_tiebreaker_view_not_published_when_file_and_metadata_absent(
    tmp_path: Path,
) -> None:
    view = board_content._load_tiebreaker_view(tmp_path, {})
    assert view.recorded is False
    assert view.note == board_content.TIEBREAKER_NOT_PUBLISHED_TEXT
    assert view.matchup_text == ""
    assert view.market_total_text == "--"


def test_load_tiebreaker_view_returns_empty_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "tiebreaker.json").write_text("not valid json", encoding="utf-8")
    view = board_content._load_tiebreaker_view(tmp_path, {})
    assert view.recorded is False


def test_load_tiebreaker_view_reads_the_persisted_sidecar(tmp_path: Path) -> None:
    (tmp_path / "tiebreaker.json").write_text(
        json.dumps(
            {
                "model_id": "test-model",
                "season": 2026,
                "week": 1,
                "home": "KC",
                "away": "DEN",
                "market_total": 43.0,
                "blended_total": 43.0421,
                "implied_margin": 2.75,
                "guess_home": 22,
                "guess_away": 19,
            }
        ),
        encoding="utf-8",
    )
    view = board_content._load_tiebreaker_view(
        tmp_path, {"active_model_id": "test-model", "season": 2026, "week": 1}
    )
    assert view.recorded is True
    assert view.matchup_text == "DEN at KC"
    assert view.market_total_text == "43"
    assert view.blended_total_text == "43.04"
    assert view.implied_margin_text == "KC by 2.75"
    assert view.guess_score_text == "KC 22 - DEN 19"
    assert view.note == board_content.TIEBREAKER_NUDGE_NOTE


def test_load_tiebreaker_view_falls_back_to_a_metadata_block(tmp_path: Path) -> None:

    metadata = {
        "active_model_id": "test-model",
        "season": 2026,
        "week": 1,
        "tiebreaker": {
            "model_id": "test-model",
            "season": 2026,
            "week": 1,
            "home": "SEA",
            "away": "NE",
            "market_total": 44.5,
            "blended_total": 44.6,
            "implied_margin": -3.0,
        },
    }
    view = board_content._load_tiebreaker_view(tmp_path, metadata)
    assert view.recorded is True
    assert view.matchup_text == "NE at SEA"
    assert view.implied_margin_text == "NE by 3.00"
    assert view.guess_score_text == ""


def test_load_tiebreaker_view_prefers_the_persisted_file_over_metadata(tmp_path: Path) -> None:
    (tmp_path / "tiebreaker.json").write_text(
        json.dumps(
            {
                "model_id": "test-model",
                "season": 2026,
                "week": 1,
                "home": "KC",
                "away": "DEN",
                "market_total": 43.0,
                "blended_total": 43.04,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "active_model_id": "test-model",
        "season": 2026,
        "week": 1,
        "tiebreaker": {
            "model_id": "test-model",
            "season": 2026,
            "week": 1,
            "home": "SEA",
            "away": "NE",
            "market_total": 44.5,
            "blended_total": 44.6,
        },
    }
    view = board_content._load_tiebreaker_view(tmp_path, metadata)
    assert view.matchup_text == "DEN at KC"


def test_load_tiebreaker_view_incomplete_block_falls_back_to_not_published(
    tmp_path: Path,
) -> None:

    (tmp_path / "tiebreaker.json").write_text(
        json.dumps({"home": "KC", "away": "DEN", "market_total": 43.0}), encoding="utf-8"
    )
    view = board_content._load_tiebreaker_view(tmp_path, {})
    assert view.recorded is False
    assert view.note == board_content.TIEBREAKER_NOT_PUBLISHED_TEXT


@pytest.mark.parametrize("field,value", [("model_id", "old"), ("season", 2025), ("week", 2)])
def test_tiebreaker_drops_stale_identity(tmp_path: Path, field: str, value: object) -> None:
    block = {
        "model_id": "current",
        "season": 2026,
        "week": 1,
        "home": "KC",
        "away": "DEN",
        "market_total": 43,
        "blended_total": 44,
    }
    block[field] = value
    (tmp_path / "tiebreaker.json").write_text(json.dumps(block))
    metadata = {"active_model_id": "current", "season": 2026, "week": 1}
    assert not board_content._load_tiebreaker_view(tmp_path, metadata).recorded


def test_tiebreaker_drops_previous_publication_in_same_week(tmp_path: Path) -> None:
    block = {
        "model_id": "current",
        "season": 2026,
        "week": 1,
        "forecast_artifact": "margin_predictions/old",
        "home": "KC",
        "away": "DEN",
        "market_total": 43,
        "blended_total": 44,
    }
    metadata = {"active_model_id": "current", "season": 2026, "week": 1}
    active = {
        "model_id": "current",
        "weekly_forecast": {"season": 2026, "week": 1, "artifact": "margin_predictions/current"},
    }
    (tmp_path / "tiebreaker.json").write_text(json.dumps(block))
    assert not board_content._load_tiebreaker_view(tmp_path, metadata, active=active).recorded
    block["forecast_artifact"] = "margin_predictions/current"
    (tmp_path / "tiebreaker.json").write_text(json.dumps(block))
    assert board_content._load_tiebreaker_view(tmp_path, metadata, active=active).recorded


def test_source_note_and_legend_use_plain_words() -> None:
    from nfl_ats.board_content import SOURCE_POLICY_COMPUTED_LIVE_NOTE, SOURCE_POLICY_LEGEND

    assert SOURCE_POLICY_LEGEND == (
        "complete: fresh enough to use; degraded: we fell back to an older copy; "
        "blocked: we refused to publish; grey: not due yet or not set up."
    )
    assert SOURCE_POLICY_COMPUTED_LIVE_NOTE == (
        "These checks describe the sources available now; they were not saved with the picks."
    )


def test_injury_pick_note_requires_saved_feature_evidence() -> None:
    from nfl_ats.board_content import SourcePolicyRow, SourcePolicyView, injury_pick_note

    audit = {"prediction_safety": {"checks_passed": ["injury_feature_presence"], "warnings": []}}
    for state, expected in (
        ("complete", "Injury reports informed these picks (latest copy from Friday morning)."),
        ("degraded", "Injury data was stale, so these picks used an older copy."),
        (
            "blocked",
            "Injury reports were not available for these picks; "
            "they lean on lineups and recent play.",
        ),
    ):
        source = SourcePolicyView(
            card_state=state,
            evaluated_at="2026-09-05T20:00:00Z",
            recorded=True,
            rows=(
                SourcePolicyRow(
                    "injuries_nflverse_timestamps", state, "2026-09-04T10:00:00Z", 60, ""
                ),
            ),
        )
        assert injury_pick_note(audit, source) == expected
        assert (
            injury_pick_note({}, source)
            == "Whether injury reports informed these picks was not recorded."
        )
    not_yet = {
        "prediction_safety": {
            "checks_passed": ["injury_feature_presence"],
            "warnings": [
                "injury feature block is entirely null/zero across 9 column(s): no injury "
                "report rows exist yet for 2026 week 1 in the newest player snapshot (x)"
            ],
        }
    }
    assert injury_pick_note(not_yet, source) == (
        "No injury reports had been published yet when these picks were made; "
        "they lean on lineups and recent play."
    )
    live = SourcePolicyView("complete", None, (), False, computed_live=True)
    assert (
        injury_pick_note({}, live)
        == "Whether injury reports informed these picks was not recorded."
    )
    empty = {
        "prediction_safety": {
            "checks_passed": ["injury_feature_presence"],
            "warnings": ["injury feature block is entirely null/zero across 4 column(s)"],
        }
    }
    assert injury_pick_note(empty, live).startswith("Injury reports were not available")


def _week1_kickoffs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_SF_LA", "2026_01_ATL_PIT", "2026_01_GB_MIN", "2026_01_DEN_KC"],
            "kickoff": [
                "2026-09-11 00:35:00+00:00",
                "2026-09-13 17:00:00+00:00",
                "2026-09-13 20:25:00+00:00",
                "2026-09-15 00:15:00+00:00",
            ],
        }
    )


def test_pick_lock_label_applies_min_of_kickoff_and_sunday_four_pm() -> None:
    from nfl_ats.board_content import _week_sunday_lock, pick_lock_label

    frame = _week1_kickoffs()
    sunday_lock = _week_sunday_lock(frame)
    assert sunday_lock is not None
    labels = {row.game_id: pick_lock_label(row.kickoff, sunday_lock) for row in frame.itertuples()}
    assert labels["2026_01_SF_LA"] == ("Thu 8:35 PM ET", False)
    assert labels["2026_01_ATL_PIT"] == ("Sun 1:00 PM ET", False)
    assert labels["2026_01_GB_MIN"] == ("Sun 4:00 PM ET", True)
    assert labels["2026_01_DEN_KC"] == ("Sun 4:00 PM ET", True)


def test_pick_lock_label_is_none_when_no_kickoff_instant_is_known() -> None:
    from nfl_ats.board_content import _week_sunday_lock, pick_lock_label

    sunday_lock = _week_sunday_lock(_week1_kickoffs())
    assert pick_lock_label(None, sunday_lock) == (None, False)
    assert pick_lock_label("not a time", sunday_lock) == (None, False)
    assert pick_lock_label("2026-09-13 17:00:00+00:00", None) == (None, False)
    assert _week_sunday_lock(pd.DataFrame({"game_id": ["x"]})) is None
    assert _week_sunday_lock(pd.DataFrame({"game_id": ["x"], "kickoff": [None]})) is None


def test_game_row_lock_text_reads_as_a_sentence_fragment() -> None:
    from _board_content_fixtures import build_fixture_games

    by_id = {game.game_id: game for game in build_fixture_games()}
    assert by_id["2026_01_SF_LA"].lock_text == "Locks Thu 8:35 PM ET"
    assert by_id["2026_01_DEN_KC"].lock_text == "Locks Sun 4:00 PM ET, before kickoff"
    assert by_id["2026_01_ARI_LAC"].lock_text is None


def test_headline_uses_the_served_union_row_and_compares_retired_four(tmp_path: Path) -> None:
    from _board_content_fixtures import build_fixture_content

    from nfl_ats.public_board import PLAYED_UNION_MEMBER_IDS

    active, _ = _headline_artifacts(tmp_path)
    path = next((tmp_path / "overlay_subset_composition").glob("*/result.json"))
    payload = json.loads(path.read_text())
    payload["subsets"] = [
        {
            "members": sorted(PLAYED_UNION_MEMBER_IDS | {"spread_gap_zone_fade_overlay"}),
            "candidate_accuracy": 0.5542248835662009,
        },
        {"members": sorted(PLAYED_UNION_MEMBER_IDS), "candidate_accuracy": 0.552228875582169},
    ]
    path.write_text(json.dumps(payload))
    headline = board_content._build_headline_stats(
        tmp_path,
        active,
        prospective_scoreboard=build_fixture_content().headline.prospective_scoreboard,
    )
    assert headline.played_card_pct == pytest.approx(56.88622754491018)
    assert headline.prior_chain_pct == pytest.approx(55.42248835662009)
    assert "three-member" not in headline.played_card_caption
    assert "actually on the board this week" in headline.played_card_caption
    assert "lacks an explained mechanism" in headline.selection_caveat_text


def test_scoreboard_pairs_new_played_policy_with_retired_union() -> None:
    from nfl_ats.four_overlay_composition import POLICY_ID
    from nfl_ats.retired_four_member_union import INCUMBENT_CHALLENGER_ID

    common = {"game_id": "g", "decision_home_spread": 7.5}
    played = pd.DataFrame([{**common, "decision_policy_id": POLICY_ID, "pick_side": "HOME"}])
    challengers = pd.DataFrame(
        [
            {**common, "challenger_id": INCUMBENT_CHALLENGER_ID, "pick_side": "AWAY"},
            {
                **common,
                "challenger_id": "overlay_production_chain_coach_arrest_incumbent",
                "pick_side": "HOME",
            },
        ]
    )
    result = board_content._build_prospective_scoreboard(
        played, challengers, pd.DataFrame([{"game_id": "g", "result": 10.0}])
    )
    assert "played policy 1-0 vs. prior chain 0-1" in result.headline_text


def test_injury_report_state_covers_every_sentence_injury_pick_note_can_produce() -> None:

    from nfl_ats.board_content import (
        SourcePolicyRow,
        SourcePolicyView,
        injury_pick_note,
        injury_report_state,
    )

    passed = {"prediction_safety": {"checks_passed": ["injury_feature_presence"], "warnings": []}}

    def view(state: str, *, observed: str | None = "2026-09-04T10:00:00Z") -> SourcePolicyView:
        return SourcePolicyView(
            card_state=state,
            evaluated_at="2026-09-05T20:00:00Z",
            recorded=True,
            rows=(SourcePolicyRow("injuries_nflverse_timestamps", state, observed, 60, ""),),
        )

    none_yet = {
        "prediction_safety": {
            "checks_passed": ["injury_feature_presence"],
            "warnings": ["no injury report rows exist yet for 2026 week 1"],
        }
    }
    empty_block = {
        "prediction_safety": {
            "checks_passed": ["injury_feature_presence"],
            "warnings": ["injury feature block is entirely null/zero across 9 column(s)"],
        }
    }
    cases = [
        (injury_pick_note({}, view("complete")), "NOT RECORDED", "not_recorded"),
        (injury_pick_note(none_yet, view("complete")), "NONE PUBLISHED YET", "not_due"),
        (injury_pick_note(empty_block, view("complete")), "NOT AVAILABLE", "blocked"),
        (injury_pick_note(passed, view("blocked")), "NOT AVAILABLE", "blocked"),
        (injury_pick_note(passed, view("degraded")), "OLDER COPY", "degraded"),
        (injury_pick_note(passed, view("complete")), "REPORTS USED", "complete"),
        (injury_pick_note(passed, view("complete", observed=None)), "REPORTS USED", "complete"),
    ]
    for note, expected_label, expected_state in cases:
        assert injury_report_state(note) == (expected_label, expected_state), note
    assert sum(1 for _note, label, _state in cases if label == "NOT RECORDED") == 1


def test_board_content_injury_chip_reads_off_its_own_sentence() -> None:

    from dataclasses import replace

    from _board_content_fixtures import build_fixture_content

    from nfl_ats.board_content import INJURY_NOTE_OLDER_COPY

    content = replace(build_fixture_content(), injury_note=INJURY_NOTE_OLDER_COPY)
    assert content.injury_state_label == "OLDER COPY"
    assert content.injury_state_class == "degraded"


def _rival_ledgers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    games = [
        ("2026_01_BAL_IND", "BAL", "IND", "HOME", -3.0),
        ("2026_01_CHI_CAR", "CHI", "CAR", "AWAY", 2.5),
        ("2026_01_NE_SEA", "NE", "SEA", "AWAY", -6.0),
    ]
    played = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "decision_policy_id": "overlay_union_coach_division_revenge_player_arrests_v2",
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "pick_side": side,
                "decision_home_spread": spread,
            }
            for game_id, away, home, side, spread in games
        ]
    )
    rival_sides = {
        "rain_on_grass_dog_challenger": ["AWAY", "HOME", "AWAY"],
        "hc_year_one_fade_overlay": ["HOME", "AWAY", "AWAY"],
        "division_revenge_tilt_overlay": ["AWAY", "AWAY", "AWAY"],
    }
    rows = [
        {
            "season": 2026,
            "week": 1,
            "challenger_id": challenger_id,
            "game_id": game_id,
            "away_team": away,
            "home_team": home,
            "pick_side": side,
            "decision_home_spread": spread,
        }
        for challenger_id, sides in rival_sides.items()
        for (game_id, away, home, _played_side, spread), side in zip(games, sides, strict=True)
    ]
    rows.append(
        {
            "season": 2026,
            "week": 1,
            "challenger_id": "best_pick_nomination_v2",
            "game_id": "2026_01_NE_SEA",
            "away_team": "NE",
            "home_team": "SEA",
            "pick_side": "AWAY",
            "decision_home_spread": -6.0,
        }
    )
    outcomes = pd.DataFrame([{"game_id": "2026_01_BAL_IND", "result": 10.0}])
    return played, pd.DataFrame(rows), outcomes


def test_build_rival_rules_counts_disagreements_and_names_the_contested_game() -> None:
    played, rivals, outcomes = _rival_ledgers()
    panel = board_content._build_rival_rules(played, rivals, outcomes, season=2026, week=1)

    assert panel.recorded
    assert panel.count_text == "4 recorded beside this week's card"
    assert panel.summary == (
        "Of the 3 that pick a whole card, 2 take a different team somewhere this week. "
        "BAL at IND is the pick they argue with most: 2 of the 3 take the other side. "
        "1 other rule agrees with the card on every game this week."
    )
    assert [
        (row.name, row.differs_text, row.games_text, row.record_text) for row in panel.rows
    ] == [
        (
            "Rain-on-grass underdog tilt",
            "2 of 3",
            "BAL at IND (BAL), CHI at CAR (CAR)",
            "0-1 so far, card 1-0 on those games",
        ),
        (
            "Division-revenge tilt",
            "1 of 3",
            "BAL at IND (BAL)",
            "0-1 so far, card 1-0 on those games",
        ),
    ]
    assert "Year-one coach fade" not in [row.name for row in panel.rows]
    assert panel.single_game_line == (
        "1 more rule names a single game rather than a whole card: NE at SEA (1)."
    )


def test_build_rival_rules_is_dormant_until_the_week_has_rows() -> None:

    from nfl_ats.board_content import RIVAL_RULES_NONE_RECORDED

    played, rivals, outcomes = _rival_ledgers()
    empty = pd.DataFrame()
    for first, second in ((empty, rivals), (played, empty), (empty, empty)):
        panel = board_content._build_rival_rules(first, second, outcomes, season=2026, week=1)
        assert not panel.recorded
        assert panel.summary == RIVAL_RULES_NONE_RECORDED
    assert not board_content._build_rival_rules(
        played, rivals, outcomes, season=2026, week=2
    ).recorded
    assert not board_content._build_rival_rules(
        played, rivals, outcomes, season=None, week=None
    ).recorded


def test_build_rival_rules_pairs_the_two_ledgers_not_the_live_forecast() -> None:

    played, rivals, outcomes = _rival_ledgers()
    unpaired = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "challenger_id": "surface_switch_tilt_overlay",
                "game_id": "2026_01_NOT_ON_THE_CARD",
                "away_team": "SF",
                "home_team": "LA",
                "pick_side": "HOME",
                "decision_home_spread": -1.0,
            }
        ]
    )
    panel = board_content._build_rival_rules(
        played, pd.concat([rivals, unpaired], ignore_index=True), outcomes, season=2026, week=1
    )
    assert "Turf-surface switch" not in [row.name for row in panel.rows]
    assert panel.count_text == "4 recorded beside this week's card"
