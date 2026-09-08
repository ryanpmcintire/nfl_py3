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
    # Bucket 7.5-10 holds two non-push rows: home -7.5 (home is the underdog,
    # home pick = underdog pick, home covered, correct) and home +8 (home is
    # the favourite, home pick = favourite pick, home failed to cover, wrong).
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
    # The level-odds line (spread 0) is the only 0-3 game: neither side counts it.
    assert by_key[("0-3", HOME_FAVOURITE)].games == 0
    assert by_key[("0-3", HOME_UNDERDOG)].games == 0
    assert by_key[("0-3", HOME_UNDERDOG)].home_cover_rate is None
    # 7.5-10: home +8 favourite failed to cover (model said 70% home); home -7.5
    # underdog covered (model said 60% home). The home -10 push never counts.
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
    # Both tables say so in the visible markup (the assistant corpus in the
    # trailing <script> repeats the sentence); no stale numbers anywhere.
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
