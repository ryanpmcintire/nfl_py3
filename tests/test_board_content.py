"""Tests for :mod:`nfl_ats.board_content`'s cover-curve / spread-explorer
fallback (2026-08-31 full-site conversion, item 4: "keep the spread
explorer's exact published-card math and guard").

``_build_cover_curve`` prefers REAL swept ``line_sweep`` rows wherever they
exist; these tests exercise the Gaussian closed-form fallback path used when
they do not, and the build-time guard that must fire before that fallback is
ever trusted.
"""

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
    # Pick is the home team, so the curve's own home-oriented published
    # probability should reproduce card_home_cover_probability at offset 0.
    assert zero_point.probability == pytest.approx(params[game.game_id].card_home_cover_probability)


def test_cover_curve_gaussian_fallback_orients_to_away_pick() -> None:
    game = _game("NE", home="SEA", away="NE")  # pick is the AWAY team
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
    """The REQUIRED build-time guard (preserved verbatim from
    ``public_board._assert_spread_explorer_matches_card`` via the public
    wrapper) must raise when the widget formula disagrees with the
    published card -- proving board_content.py cannot silently trust a
    Gaussian read that would show a different number than the one already
    on the page."""

    params = {"2026_01_TEST": _params()}
    predictions = pd.DataFrame(
        {"game_id": ["2026_01_TEST"], "home_cover_probability": [0.999]}
    )  # deliberately wrong vs. the widget's own computed probability
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
    assert_spread_explorer_matches_card(params, predictions)  # must not raise


def test_guard_fires_for_any_game_in_a_multi_game_week() -> None:
    """2026-08-31 owner redirect: the line-offset adjuster now covers EVERY
    game's deep dive, not only the Best Pick's -- so the guard must catch a
    mismatch on ANY game in the week, not just the first/only one a
    single-game test would exercise."""

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
                0.999,  # deliberately wrong for the SECOND game only
            ],
        }
    )
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(params, predictions)


def test_cover_curve_fallback_offsets_match_sweep_half_width_and_step() -> None:
    """Regression guard: the fallback grid must span the SAME domain as a
    real sweep (``SWEEP_HALF_WIDTH``), never wider or coarser -- otherwise a
    chart built from the fallback would look different from one built from
    real rows for no real reason."""

    offsets = board_content._COVER_CURVE_FALLBACK_OFFSETS
    assert math.isclose(min(offsets), -board_content.SWEEP_HALF_WIDTH)
    assert math.isclose(max(offsets), board_content.SWEEP_HALF_WIDTH)
    assert offsets == tuple(sorted(offsets))


# ---------------------------------------------------------------------------
# ENG-34: the ENG-14 ``source_policy`` block, read from the synchronized
# forecast's own ``metadata.json`` (see ``nfl_ats.publishing``'s
# ``SourcePolicyReport.to_metadata()`` shape).
# ---------------------------------------------------------------------------


def test_load_source_policy_view_absent_block_is_not_recorded() -> None:
    """Every forecast in this repo today has no ``source_policy`` key at all
    (measured 2026-09-04: ``publishing.py`` computes the report but only
    returns it from ``publish_active_predictions``'s result dict) -- this
    must degrade to the explicit not-recorded view, never raise."""

    view = board_content._load_source_policy_view({"season": 2026, "week": 1}, None)
    assert view.recorded is False
    assert view.card_state == board_content.SOURCE_POLICY_NOT_RECORDED
    assert view.card_state_label == "NOT RECORDED"
    assert view.rows == ()
    assert view.evaluated_at is None


def test_load_source_policy_view_reads_full_block() -> None:
    """Shaped exactly as ``SourcePolicyReport.to_metadata()`` writes it."""

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
    # evaluated_at_utc minus this row's own age_minutes (30.0).
    assert by_id["odds_opener"].observed_at == "2026-09-03T13:30:00+00:00"
    assert by_id["odds_opener"].observed_at_text == "as-of 2026-09-03 13:30 UTC"
    assert by_id["injuries_nflverse"].state == "degraded"
    assert by_id["injuries_nflverse"].observed_at is None
    assert by_id["injuries_nflverse"].observed_at_text == "no snapshot"
    assert by_id["airnow_weather"].state == "unobserved"


def test_load_source_policy_view_malformed_state_falls_back_to_not_recorded() -> None:
    """A block IS present (``recorded`` stays ``True``, matching what's
    literally on disk) but its ``state`` is not one of the three real
    values -- never invent or display an unknown card state."""

    metadata = {"source_policy": {"state": "not-a-real-state", "sources": {}}}
    view = board_content._load_source_policy_view(metadata, None)
    assert view.recorded is True
    assert view.card_state == board_content.SOURCE_POLICY_NOT_RECORDED


def test_load_source_policy_view_prefers_the_persisted_file_over_metadata(
    tmp_path: Path,
) -> None:
    """ENG-34 follow-up: ``publishing.py`` now persists the block as
    ``source_policy.json`` beside the forecast artifact (additive; the
    forecast's own ``metadata.json`` is never rewritten). That file must win
    over a ``metadata["source_policy"]`` key when both are present, and be
    read at all when ``metadata`` itself has no such key."""

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
    assert view.card_state == "complete"  # the FILE's state, not metadata's "blocked"
    assert [row.source_id for row in view.rows] == ["odds_opener"]

    # No file on disk -- falls back to metadata's own key.
    empty_dir = tmp_path / "no_file_here"
    empty_dir.mkdir()
    fallback_view = board_content._load_source_policy_view(
        metadata_with_a_different_block, empty_dir
    )
    assert fallback_view.card_state == "blocked"

    # Neither -- the explicit not-recorded view.
    assert board_content._load_source_policy_view({}, empty_dir).recorded is False


# ---------------------------------------------------------------------------
# UI-20(c): the ENG-14 report computed live at build time when nothing was
# persisted (dashboard improvement queue, ROADMAP.md).
# ---------------------------------------------------------------------------


def test_load_source_policy_view_without_data_root_stays_not_recorded(tmp_path: Path) -> None:
    """Every existing caller that omits ``data_root``/``artifacts_root``
    (both keyword-only, both default ``None``) must see EXACTLY the prior
    behaviour -- this is the regression guard for the additive signature
    change."""

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
    """Nothing persisted (no ``source_policy.json``, no metadata key) but a
    ``data_root``/``artifacts_root`` is supplied: a REAL live report, not
    the placeholder -- every source in this fixture's empty tree is
    "absent", which every non-fail-closed source's policy degrades to."""

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
    assert view.rows  # real per-source rows, never empty
    by_id = {row.source_id: row for row in view.rows}
    # player_arrests was never passed a snapshot instant -- unobserved, not
    # falsely blocked (report_for_publication's own fail-open contract).
    assert by_id.pop("player_arrests").state == "unobserved"
    assert by_id.pop("injuries_sportradar").state == "not_configured"
    assert all(row.state == "degraded" for row in by_id.values())


def test_load_source_policy_view_prefers_persisted_over_live(tmp_path: Path) -> None:
    """A real persisted block still wins even when a data_root/artifacts_root
    is also supplied -- the live computation is a fallback, never a
    replacement for the real recorded state."""

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


# ---------------------------------------------------------------------------
# ENG-12 wiring (UI-20(a)): per-pick "Why this pick" explanation text.
# ---------------------------------------------------------------------------


def test_load_pick_explanations_returns_empty_when_forecast_dir_is_none() -> None:
    assert board_content._load_pick_explanations(None) == {}


def test_load_pick_explanations_returns_empty_when_file_absent(tmp_path: Path) -> None:
    assert board_content._load_pick_explanations(tmp_path) == {}


def test_load_pick_explanations_returns_empty_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "explanations.json").write_text("not valid json", encoding="utf-8")
    assert board_content._load_pick_explanations(tmp_path) == {}


def test_load_pick_explanations_skips_a_row_with_a_malformed_nested_field(tmp_path: Path) -> None:
    """A row whose nested component is the WRONG TYPE (a string where a
    mapping is expected) must be skipped, not crash the whole page build."""

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
                    {"game_id": "2026_01_KC_DEN", "text": ""},  # empty text -- excluded
                ],
            }
        ),
        encoding="utf-8",
    )
    texts = board_content._load_pick_explanations(tmp_path)
    assert texts == {"2026_01_SEA_NE": "SEA at NE: the market line is -3."}


# ---------------------------------------------------------------------------
# UI-20(g): _load_tiebreaker_view
# ---------------------------------------------------------------------------


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
    # Never a fabricated matchup/number when nothing was published.
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
    """No sidecar file at all -- a future writer that instead adds a
    ``tiebreaker`` block to ``metadata.json`` must still be read."""

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
    # No guess score supplied -- optional, never fabricated.
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
    """A block missing a required field (here: no ``blended_total``) must
    degrade to not-published rather than render a half-filled guess."""

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
