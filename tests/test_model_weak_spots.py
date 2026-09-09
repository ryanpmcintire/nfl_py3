"""Synthetic opener diagnostics, identity guard, renderer and assistant contracts."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats.board_assistant import answer, build_knowledge_for_model
from nfl_ats.board_site_content import _load_model_page_content, load_model_weak_spots
from nfl_ats.board_terminal import render_model_page
from nfl_ats.model_weak_spots import (
    HOME_FAVOURITE,
    HOME_UNDERDOG,
    UNAVAILABLE,
    WeakSpots,
    build_weak_spots,
)
from nfl_ats.public_board import OpenerEvaluationArtifacts


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tue_open_home_spread": [-7.5, 8, -10, 7, 6.75, 3.25, 10.25, 0],
            "margin_vs_open": [2, -1, 0, 1, -1, 1, 1, 1],
            "pick_home_at_open_probability_rule": [
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            "correct_at_open_probability_rule": [1, 0, 1, 0, 1, 1, 1, 1],
            "home_cover_probability_at_open": [0.6, 0.7, 0.2, 0.4, 0.3, 0.6, 0.6, 0.5],
            "correct_at_open": [0] * 8,
        }
    )


def test_boundaries_pushes_pick_rule_and_confidence() -> None:
    source = frame()
    original = source.copy(deep=True)
    rows = build_weak_spots(source).rows
    assert [r.games for r in rows] == [1, 1, 2, 2, 1]
    row = rows[3]
    assert row.accuracy == 0.5
    assert row.confidence == pytest.approx(0.65)
    assert row.favourite_pick_rate == 0.5
    assert row.favourite_cover_rate == 0
    assert row.favourite_accuracy == 0
    assert row.underdog_accuracy == 1
    assert rows[0].favourite_accuracy is None
    assert rows[0].favourite_cover_rate is None
    pd.testing.assert_frame_equal(source, original)


def test_home_split_excludes_pickem_and_pushes_and_reads_the_home_side() -> None:
    split = build_weak_spots(frame()).home_split
    assert [(r.spread, r.home_side) for r in split] == [
        (label, side)
        for label in ("0-3", "3.5-6.5", "7-7.5", "7.5-10", "10.5+")
        for side in (HOME_FAVOURITE, HOME_UNDERDOG)
    ]
    by_key = {(r.spread, r.home_side): r for r in split}
    assert by_key[("0-3", HOME_FAVOURITE)].games == 0
    assert by_key[("0-3", HOME_UNDERDOG)].games == 0
    assert by_key[("0-3", HOME_UNDERDOG)].home_cover_rate is None
    favourite = by_key[("7.5-10", HOME_FAVOURITE)]
    underdog = by_key[("7.5-10", HOME_UNDERDOG)]
    assert (favourite.games, favourite.home_cover_rate, favourite.accuracy) == (1, 0.0, 0.0)
    assert favourite.home_confidence == pytest.approx(0.7)
    assert (underdog.games, underdog.home_cover_rate, underdog.accuracy) == (1, 1.0, 1.0)
    assert underdog.home_confidence == pytest.approx(0.6)
    assert by_key[("10.5+", HOME_UNDERDOG)].games == 0
    assert by_key[("10.5+", HOME_FAVOURITE)].games == 1
    assert underdog.cells == ("7.5-10", HOME_UNDERDOG, "1", "100.0%", "60.0%", "100.0%")
    assert "home underdog (1 game) the home team covered 100%" in underdog.plain
    assert "no decided games" in by_key[("0-3", HOME_UNDERDOG)].plain
    assert WeakSpots().home_split_text == UNAVAILABLE


def test_home_split_sums_to_the_sided_bucket_counts() -> None:
    spots = build_weak_spots(frame())
    for row in spots.rows:
        sided = sum(r.games for r in spots.home_split if r.spread == row.spread)
        assert sided <= row.games


def test_empty_and_missing_values_do_not_invent_numbers() -> None:
    assert all(r.games == 0 for r in build_weak_spots(frame().iloc[:0]).rows)
    assert all(r.games == 0 for r in build_weak_spots(frame().iloc[:0]).home_split)
    source = frame()
    source.loc[:, "home_cover_probability_at_open"] = float("nan")
    assert all(r.accuracy is None for r in build_weak_spots(source).rows)
    assert all(r.home_cover_rate is None for r in build_weak_spots(source).home_split)


def test_matching_loader_ignores_newer_wrong_model(tmp_path: Path) -> None:
    active = {"model_id": "current", "feature_table_sha256": "table"}
    metadata = {"active_model_id": "current", "provenance": {"feature_table": {"sha256": "table"}}}
    for name, identity in (("20260901T000000Z", "current"), ("20260902T000000Z", "stale")):
        root = tmp_path / "opener_evaluation" / name
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({**metadata, "active_model_id": identity}))
        frame().to_parquet(root / "per_game.parquet")
    assert load_model_weak_spots(tmp_path, active).rows[3].games == 2
    assert load_model_weak_spots(tmp_path, {**active, "model_id": "missing"}) == WeakSpots()


def test_model_panel_and_assistant_share_table_and_missing_sentence(tmp_path: Path) -> None:
    model = _load_model_page_content(
        tmp_path,
        registry_root=tmp_path,
        board=build_fixture_content(),
        opener=OpenerEvaluationArtifacts({}, pd.DataFrame()),
        active={},
        generated_at=datetime(2026, 9, 7, tzinfo=UTC),
    )
    assert model.weak_spots == WeakSpots()
    unavailable = render_model_page(model)
    assert unavailable.split("<script")[0].count(UNAVAILABLE) == 2
    assert 'id="weak-spots-home-h"' in unavailable
    model = replace(model, weak_spots=build_weak_spots(frame()))
    rendered = render_model_page(model)
    assert "Where the model is weak" in rendered
    assert '<td data-label="Spread size">7.5-10</td><td data-label="Games">2</td>' in rendered
    assert "model said about 65%" in rendered
    assert UNAVAILABLE not in rendered
    assert "Home favourite or home underdog" in rendered
    assert (
        '<td data-label="Spread size">7.5-10</td><td data-label="Home team was">Home underdog</td>'
        '<td data-label="Games">1</td><td data-label="Home team covered">100.0%</td>'
        '<td data-label="Model expected home to cover">60.0%</td>'
        '<td data-label="Model right">100.0%</td>'
    ) in rendered
    assert "home underdogs on big spreads have covered more often" in rendered
    knowledge = build_knowledge_for_model(model)
    assert knowledge["weak_spots"][3]["confidence"] == pytest.approx(0.65)
    assert knowledge["weak_spots_home_split"][7]["home_side"] == HOME_UNDERDOG
    assert knowledge["weak_spots_home_split"][7]["home_confidence"] == pytest.approx(0.6)
    for question in ("where is the model weak", "how does the model do on big spreads"):
        response = answer(question, knowledge)
        assert response.topic == "weak_spots"
        assert response.text == model.weak_spots.text
        assert response.anchors == ("model.html#weak-spots-h",)
    for question in (
        "how does the model do with home underdogs",
        "does the model like home dogs on big spreads",
        "how has the model done on home favourites",
    ):
        response = answer(question, knowledge)
        assert response.topic == "weak_spots_home_split"
        assert response.text == model.weak_spots.home_split_text
        assert response.anchors == ("model.html#weak-spots-home-h",)


def test_missing_matching_file_does_not_fall_back(tmp_path: Path, monkeypatch) -> None:
    import nfl_ats.board_site_content as content

    monkeypatch.setattr(content, "find_matching_opener_evaluation", lambda *args: ({}, tmp_path))
    assert content.load_model_weak_spots(tmp_path, {}) == WeakSpots()
    pd.DataFrame({"wrong_schema": [1]}).to_parquet(tmp_path / "per_game.parquet")
    assert content.load_model_weak_spots(tmp_path, {}) == WeakSpots()


def _aligned_frame() -> pd.DataFrame:
    """A per-game table the aligned opener evaluation writes (2026-09-08):
    served columns beside their ``_raw`` twins and the per-game push."""
    source = frame()
    source["home_side_offset_at_open"] = [0.0, 0.8, 1.9, -0.2, 0.4, 0.0, 1.9, 0.0]
    source["pick_home_at_open_probability_rule_raw"] = [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    source["correct_at_open_probability_rule_raw"] = [1, 1, 1, 0, 1, 1, 1, 1]
    return source


def test_home_correction_reads_both_sides_and_this_weeks_push() -> None:
    from nfl_ats.model_weak_spots import build_home_correction

    assert build_home_correction(frame()) is None
    correction = build_home_correction(
        _aligned_frame(),
        {"0-3": 0.03, "3.5-6.5": 0.38, "7": -0.17, "7.5-10": 0.78, "10.5+": 1.92},
        {"0-3": 504, "3.5-6.5": 468, "7": 65, "7.5-10": 160, "10.5+": 113},
    )
    assert correction is not None
    assert [r.spread for r in correction.rows] == ["0-3", "3.5-6.5", "7", "7.5-10", "10.5+"]
    by_bucket = {r.spread: r for r in correction.rows}
    row = by_bucket["7.5-10"]
    assert row.this_week_points == pytest.approx(0.78)
    assert row.learned_from_games == 160
    assert row.archive_games == 2
    assert row.picks_changed == 1
    assert row.accuracy_with == 0.5
    assert row.accuracy_without == 1.0
    assert row.cells == ("7.5-10", "+0.78", "160", "2", "1", "50.0%", "100.0%")
    assert by_bucket["10.5+"].picks_changed == 1
    assert by_bucket["10.5+"].accuracy_with == 1.0
    assert by_bucket["10.5+"].accuracy_without == 1.0
    assert correction.archive_games == 7
    assert correction.picks_changed == 2
    assert correction.accuracy_with == pytest.approx(5 / 7)
    assert correction.accuracy_without == pytest.approx(6 / 7)
    assert correction.this_week_available is True
    no_week = build_home_correction(_aligned_frame())
    assert no_week is not None
    assert no_week.rows[0].this_week_points is None
    assert no_week.rows[0].cells[1] == "--"
    assert no_week.this_week_available is False


def test_model_panel_renders_the_home_side_push(tmp_path: Path) -> None:
    from html import escape

    from nfl_ats.model_weak_spots import (
        HOME_CORRECTION_UNAVAILABLE,
        build_home_correction,
    )

    model = _load_model_page_content(
        tmp_path,
        registry_root=tmp_path,
        board=build_fixture_content(),
        opener=OpenerEvaluationArtifacts({}, pd.DataFrame()),
        active={},
        generated_at=datetime(2026, 9, 8, tzinfo=UTC),
    )
    unavailable = render_model_page(model)
    assert 'id="weak-spots-push-h"' in unavailable
    assert escape(HOME_CORRECTION_UNAVAILABLE) in unavailable.split("<script")[0]
    spots = replace(
        build_weak_spots(_aligned_frame()),
        home_correction=build_home_correction(_aligned_frame(), {"7.5-10": 0.78}, {"7.5-10": 160}),
    )
    rendered = render_model_page(replace(model, weak_spots=spots))
    assert "The home-side push" in rendered
    assert escape(HOME_CORRECTION_UNAVAILABLE) not in rendered
    assert (
        '<td data-label="Spread size">7.5-10</td>'
        '<td data-label="This week&#x27;s push (points)">+0.78</td>'
        '<td data-label="Learned from games">160</td>'
        '<td data-label="Archive games">2</td>'
        '<td data-label="Picks it changed">1</td>'
        '<td data-label="Right with the push">50.0%</td>'
        '<td data-label="Right without it">100.0%</td>'
    ) in rendered
    assert "changed 2 of 7 picks" in rendered
    assert "right 71.4% with it and 85.7% without" in rendered
    assert "the record with that push on" in rendered


def test_loader_attaches_this_weeks_push_from_the_linked_forecast(
    tmp_path: Path, monkeypatch
) -> None:
    from nfl_ats import board_site_content

    evaluation = tmp_path / "opener_evaluation" / "run"
    evaluation.mkdir(parents=True)
    _aligned_frame().to_parquet(evaluation / "per_game.parquet")
    forecast = tmp_path / "margin_predictions" / "week"
    forecast.mkdir(parents=True)
    from nfl_ats.home_side_location import HOME_SIDE_OFFSET_POLICY

    def write_sidecar(policy: str) -> None:
        (forecast / "home_side_offset.json").write_text(
            json.dumps(
                {
                    "served": True,
                    "fit": {
                        "policy": policy,
                        "offsets": {"10.5+": 1.92},
                        "prior_games": {"10.5+": 113},
                    },
                    "games": [],
                }
            )
        )

    write_sidecar(HOME_SIDE_OFFSET_POLICY)
    monkeypatch.setattr(
        board_site_content,
        "find_matching_opener_evaluation",
        lambda _root, _active: ({}, evaluation),
    )
    active = {"weekly_forecast": {"artifact": "margin_predictions/week"}}
    spots = load_model_weak_spots(tmp_path, active)
    assert spots.home_correction is not None
    assert spots.home_correction.this_week_available is True
    by_bucket = {r.spread: r for r in spots.home_correction.rows}
    assert by_bucket["10.5+"].this_week_points == pytest.approx(1.92)
    assert by_bucket["10.5+"].learned_from_games == 113
    assert by_bucket["0-3"].this_week_points is None
    write_sidecar("home_side_offset_by_bucket_v1")
    spots = load_model_weak_spots(tmp_path, active)
    assert spots.home_correction is not None
    assert spots.home_correction.this_week_available is False
    (forecast / "home_side_offset.json").write_text(json.dumps({"served": False}))
    spots = load_model_weak_spots(tmp_path, active)
    assert spots.home_correction is not None
    assert spots.home_correction.this_week_available is False
