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
from nfl_ats.model_weak_spots import UNAVAILABLE, WeakSpots, build_weak_spots
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


def test_empty_and_missing_values_do_not_invent_numbers() -> None:
    assert all(r.games == 0 for r in build_weak_spots(frame().iloc[:0]).rows)
    source = frame()
    source.loc[:, "home_cover_probability_at_open"] = float("nan")
    assert all(r.accuracy is None for r in build_weak_spots(source).rows)


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
    assert UNAVAILABLE in render_model_page(model)
    model = replace(model, weak_spots=build_weak_spots(frame()))
    rendered = render_model_page(model)
    assert "Where the model is weak" in rendered
    assert '<td data-label="Spread size">7.5-10</td><td data-label="Games">2</td>' in rendered
    assert "model said about 65%" in rendered
    knowledge = build_knowledge_for_model(model)
    assert knowledge["weak_spots"][3]["confidence"] == pytest.approx(0.65)
    for question in ("where is the model weak", "how does the model do on big spreads"):
        response = answer(question, knowledge)
        assert response.topic == "weak_spots"
        assert response.text == model.weak_spots.text
        assert response.anchors == ("model.html#weak-spots-h",)


def test_missing_matching_file_does_not_fall_back(tmp_path: Path, monkeypatch) -> None:
    import nfl_ats.board_site_content as content

    monkeypatch.setattr(content, "find_matching_opener_evaluation", lambda *args: ({}, tmp_path))
    assert content.load_model_weak_spots(tmp_path, {}) == WeakSpots()
    pd.DataFrame({"wrong_schema": [1]}).to_parquet(tmp_path / "per_game.parquet")
    assert content.load_model_weak_spots(tmp_path, {}) == WeakSpots()
