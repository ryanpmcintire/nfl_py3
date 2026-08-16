"""Tests for the public GitHub Pages picks board renderer and its artifact loader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    build_public_board_html,
    confidence_tier,
    load_public_board_artifacts,
    pick_side_and_line,
    render_public_board,
    spread_label,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_confidence_tier_bands() -> None:
    assert confidence_tier(0.50) == ("flip", "Coin flip")
    assert confidence_tier(0.5199) == ("flip", "Coin flip")
    assert confidence_tier(0.52) == ("lean", "Lean")
    assert confidence_tier(0.5699) == ("lean", "Lean")
    assert confidence_tier(0.57) == ("strong", "Strong lean")
    assert confidence_tier(0.95) == ("strong", "Strong lean")


def test_spread_label_formatting() -> None:
    assert spread_label(0) == "PK"
    assert spread_label(-7.5) == "-7½"
    assert spread_label(0.5) == "+½"
    assert spread_label(-0.5) == "-½"
    assert spread_label(3) == "+3"
    assert spread_label(-10) == "-10"
    assert spread_label(3.2) == "+3.2"


def test_pick_side_and_line_sign_convention() -> None:
    home_favored = pd.Series(
        {"home_team": "LAC", "away_team": "ARI", "spread_line": -3.5, "home_cover_probability": 0.7}
    )
    team, line, probability = pick_side_and_line(home_favored)
    assert (team, line, probability) == ("LAC", 3.5, 0.7)

    away_pick = pd.Series(
        {"home_team": "LAC", "away_team": "ARI", "spread_line": -3.5, "home_cover_probability": 0.3}
    )
    team, line, probability = pick_side_and_line(away_pick)
    assert (team, line, probability) == ("ARI", -3.5, 0.7)


# ---------------------------------------------------------------------------
# render_public_board: fixture predictions/sweep/explanations
# ---------------------------------------------------------------------------


def _predictions_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "gameday": ["2026-09-13", "2026-09-10"],
            "weekday": ["Sun", "Thu"],
            "gametime": ["13:00:00", "20:15:00"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.38, 0.62],
            "predicted_market_residual": [0.4, -1.1],
            "method": ["market_residual", "market_residual"],
        }
    )


def _sweep_fixture() -> pd.DataFrame:
    rows = []
    for game_id, quoted in (("2026_01_ARI_LAC", 3.5), ("2026_01_SF_LA", -3.5)):
        for offset in (-0.5, 0.0, 0.5):
            rows.append(
                {
                    "method": "market_residual",
                    "game_id": game_id,
                    "quoted_line": quoted,
                    "line_offset": offset,
                    "alternative_line": quoted + offset,
                    "home_cover_probability": 0.5 + offset * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_render_public_board_includes_only_whitelisted_fields() -> None:
    predictions = _predictions_fixture()
    predictions["bookmaker"] = ["DraftKings", "FanDuel"]
    predictions["price"] = [-110, -105]
    sweep = _sweep_fixture()
    explanations = {"2026_01_ARI_LAC": "The model leans LAC by a hair."}

    page = render_public_board(
        predictions,
        sweep,
        explanations,
        season=2026,
        week=1,
        model_id="model-123",
        historical_accuracy=0.5205,
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
    )

    # Whitelisted fields present.
    assert "ARI @ LAC" in page
    assert "SF @ LA" in page
    assert "The model leans LAC by a hair." in page
    assert "Model and market are close on this one." in page  # SF/LA has no explanation
    assert "model-123" in page
    assert "2026-08-16 20:00 UTC" in page
    assert "Sun 09/13" in page
    assert "Thu 09/10" in page

    # Forbidden fields never rendered, even when present on the input frame.
    assert "DraftKings" not in page
    assert "FanDuel" not in page
    assert "-110" not in page
    assert "-105" not in page
    assert "bookmaker" not in page.lower()


def test_render_public_board_disclaimer_present_top_and_bottom() -> None:
    page = render_public_board(_predictions_fixture(), _sweep_fixture())
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
    top_index = page.index(DISCLAIMER_SHORT)
    bottom_index = page.index(DISCLAIMER_FULL)
    assert top_index < bottom_index
    for phrase in ("not betting advice", "not proof of a profitable edge", "21+", "1-800-GAMBLER"):
        assert phrase in page


def test_render_public_board_picks_and_confidence_and_edge() -> None:
    page = render_public_board(_predictions_fixture(), _sweep_fixture())
    # Row 1: home prob 0.38 < 0.5 -> away ARI picked; pick_line = spread_line unchanged (+3.5).
    assert "ARI +3½" in page
    # Row 2: home prob 0.62 >= 0.5 -> home LA picked; pick_line = -spread_line = +3.5.
    assert "LA +3½" in page
    assert page.count("62%") == 2  # both picks land at a 62% picked-side probability
    assert "-0.4" in page  # row 1 edge: home not picked -> edge = -predicted_market_residual
    assert "-1.1" in page  # row 2 edge: home picked -> edge = predicted_market_residual


def test_render_public_board_empty_predictions_still_has_shell() -> None:
    page = render_public_board(pd.DataFrame())
    assert "No games are scheduled" in page
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
    assert "<html" in page


def test_render_public_board_no_sweep_omits_strip_without_error() -> None:
    page = render_public_board(_predictions_fixture(), sweep=None)
    assert "ARI @ LAC" in page


def test_render_public_board_declares_utf8_charset_before_body() -> None:
    """A missing/late charset mojibakes the vulgar-fraction glyphs (e.g. "½" -> "Â½")
    on any static server that omits a charset response header -- verified live against
    the canonical mock. The <meta charset> tag must appear inside <head>, before any
    half-point spread label reaches the byte stream."""

    page = render_public_board(_predictions_fixture(), _sweep_fixture())
    assert '<meta charset="utf-8">' in page
    head_index = page.index("<head>")
    charset_index = page.index('<meta charset="utf-8">')
    half_point_index = page.index("½")
    assert head_index < charset_index < half_point_index


# ---------------------------------------------------------------------------
# load_public_board_artifacts / build_public_board_html: artifact chain
# ---------------------------------------------------------------------------


def _write_board_fixture(root: Path, *, with_decomposition: bool = True) -> None:
    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _predictions_fixture().to_csv(forecast / "recommendations.csv", index=False)
    sweep = pd.concat(
        [_sweep_fixture(), _sweep_fixture().assign(method="market")], ignore_index=True
    )
    sweep.to_parquet(forecast / "line_sweep.parquet", index=False)

    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")

    if with_decomposition:
        decomposition = root / "market_decomposition" / "20260101T000000Z"
        decomposition.mkdir(parents=True)
        pd.DataFrame(
            {
                "game_id": ["2026_01_ARI_LAC", "2026_01_ARI_LAC", "2026_01_SF_LA"],
                "family": ["context", "elo", "context"],
                "explanation": [
                    "The model leans LAC by a hair.",
                    "The model leans LAC by a hair.",
                    "The model and market agree on this one.",
                ],
            }
        ).to_parquet(decomposition / "attribution.parquet", index=False)


def test_load_public_board_artifacts_reads_synchronized_chain(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    artifacts = load_public_board_artifacts(tmp_path)
    assert len(artifacts.predictions) == 2
    # sweep is filtered to the active method only (market rows excluded).
    assert set(artifacts.sweep["method"].unique()) == {"market_residual"}
    assert artifacts.explanations["2026_01_ARI_LAC"] == "The model leans LAC by a hair."
    assert artifacts.metadata["season"] == 2026
    assert artifacts.active["model_id"] == "model-123"


def test_load_public_board_artifacts_without_decomposition_has_no_explanations(
    tmp_path: Path,
) -> None:
    _write_board_fixture(tmp_path, with_decomposition=False)
    artifacts = load_public_board_artifacts(tmp_path)
    assert artifacts.explanations == {}


def test_load_public_board_artifacts_missing_active_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No synchronized active ATS model"):
        load_public_board_artifacts(tmp_path)


def test_load_public_board_artifacts_model_id_mismatch_raises(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    forecast_metadata_path = tmp_path / "margin_predictions" / "forecast" / "metadata.json"
    metadata = json.loads(forecast_metadata_path.read_text(encoding="utf-8"))
    metadata["active_model_id"] = "wrong-model"
    forecast_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="model ID does not match"):
        load_public_board_artifacts(tmp_path)


def test_load_public_board_artifacts_unsynchronized_forecast_raises(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    forecast_metadata_path = tmp_path / "margin_predictions" / "forecast" / "metadata.json"
    metadata = json.loads(forecast_metadata_path.read_text(encoding="utf-8"))
    metadata["synchronization_status"] = "UNLINKED"
    forecast_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="not synchronized"):
        load_public_board_artifacts(tmp_path)


def test_build_public_board_html_end_to_end(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    page = build_public_board_html(tmp_path, generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC))
    assert "model-123" in page
    assert "2026-08-16 20:00 UTC" in page
    assert "The model leans LAC by a hair." in page
    assert "52%" in page  # historical accuracy 0.5205 rounds to 52% in the byline
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
