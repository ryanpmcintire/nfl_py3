"""Tests for the public GitHub Pages site: its three pages and their artifact loaders.

Three layers, cheapest first:

* the sign-convention helpers are pure, so they are unit-tested with no
  artifacts at all;
* each ``render_*_page`` function is driven with fixture frames and asserted on
  the design-system markers it must emit (the shared stylesheet, the ``.ats``
  root, the components' own class hooks) and on the disclaimers;
* :func:`build_public_site` is driven end-to-end against temporary artifact
  trees, where the licensing blocklist is scanned across ALL THREE pages.

The blocklist is the load-bearing test in this file. The public pages may show
only what the tracked public markdown card already shows -- see the
``nfl_ats.public_board`` module docstring -- so book identities and raw
market-feed field names must never reach any generated page.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    FINDINGS_PAGE,
    PICKS_PAGE,
    TRACK_RECORD_PAGE,
    build_public_site,
    load_opener_evaluation_artifacts,
    load_public_board_artifacts,
    pick_side,
    render_findings_page,
    render_picks_page,
    render_track_record_page,
    spread_words,
)

# ---------------------------------------------------------------------------
# The licensing blocklist
# ---------------------------------------------------------------------------

# Book identities from the purchased odds feed. None of these can appear by
# accident, so they are scanned against the raw HTML of every page.
FORBIDDEN_BOOKS = ("DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet", "Bovada")

# Raw market-feed column names that ride along on recommendations.csv and must
# be dropped on render. Also unambiguous, so also scanned against raw HTML.
FORBIDDEN_FIELDS = (
    "home_spread_odds",
    "away_spread_odds",
    "total_line",
    "quoted_line",
    "alternative_line",
)

# Per-book price values. These are scanned against the page's VISIBLE TEXT
# rather than its HTML: the charts draw with clip-path polygons whose percent
# coordinates legitimately contain arbitrary decimals, so a raw-HTML substring
# scan for a number would be checking CSS geometry, not published data.
FORBIDDEN_VALUES = ("-110", "-105", "46.5")

_TAG = re.compile(r"<[^>]+>")
_HEAD_BLOCK = re.compile(r"<(style|script)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def _rendered(text: str) -> str:
    """A content string as the components escape it (``escape()`` quotes too)."""

    return escape(text)


def _visible_text(page: str) -> str:
    """Everything a reader can see: no stylesheet, no script, no markup."""

    return _TAG.sub(" ", _HEAD_BLOCK.sub(" ", page))


def assert_public_safe(page: str) -> None:
    """Every guardrail a generated public page must satisfy."""

    for book in FORBIDDEN_BOOKS:
        assert book not in page
    for field in FORBIDDEN_FIELDS:
        assert field not in page
    text = _visible_text(page)
    for value in FORBIDDEN_VALUES:
        assert value not in text
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
    assert page.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert '<meta name="viewport"' in page
    assert '<div class="ats">' in page
    # The Streamlit-only theme sync must never ship on a static page: it polls a
    # Streamlit-owned element that does not exist here.
    assert "stApp" not in page
    assert "__atsThemeInterval" not in page


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_spread_words_uses_home_oriented_sign_convention() -> None:
    assert spread_words("LAC", "ARI", 3.5) == "LAC -3.5"
    assert spread_words("LAC", "ARI", -3.5) == "ARI -3.5"
    assert spread_words("LAC", "ARI", 10.0) == "LAC -10"
    assert spread_words("LAC", "ARI", 0.0) == "pick 'em"
    assert spread_words("LAC", "ARI", float("nan")) == "pick 'em"


def test_pick_side_takes_the_favoured_side() -> None:
    home_favored = pd.Series(
        {"home_team": "LAC", "away_team": "ARI", "spread_line": -3.5, "home_cover_probability": 0.7}
    )
    assert pick_side(home_favored) == ("LAC", 0.7)

    away_pick = pd.Series(
        {"home_team": "LAC", "away_team": "ARI", "spread_line": -3.5, "home_cover_probability": 0.3}
    )
    team, probability = pick_side(away_pick)
    assert team == "ARI"
    assert probability == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# render_picks_page: fixture predictions/sweep/explanations
# ---------------------------------------------------------------------------


def _predictions_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "gameday": ["2026-09-13", "2026-09-10"],
            "weekday": ["Sunday", "Thursday"],
            "gametime": ["13:00", "20:15"],
            "kickoff": ["2026-09-13 17:00:00+00:00", "2026-09-11 00:15:00+00:00"],
            "game_type": ["REG", "REG"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.38, 0.62],
            "predicted_market_residual": [-2.4, -1.1],
            "fair_spread": [1.1, -4.6],
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


def _leaky_predictions() -> pd.DataFrame:
    """The fixture card plus every raw market-feed field it actually carries."""

    predictions = _predictions_fixture()
    predictions["bookmaker"] = ["DraftKings", "FanDuel"]
    predictions["home_spread_odds"] = [-110, -105]
    predictions["away_spread_odds"] = [-110, -105]
    predictions["total_line"] = [46.5, 46.5]
    return predictions


def test_render_picks_page_includes_only_allowlisted_fields() -> None:
    page = render_picks_page(
        _leaky_predictions(),
        _sweep_fixture(),
        {"2026_01_ARI_LAC": "The model leans ARI by two and a half points."},
        season=2026,
        week=1,
        model_id="model-123",
        historical_accuracy=0.5205,
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
    )

    # Allowlisted fields present.
    assert "ARI at LAC" in page
    assert "SF at LA" in page
    assert "LAC -3.5" in page  # the one consensus market line, in words
    assert "The model leans ARI by two and a half points." in page
    assert "model-123" in page
    assert "2026-08-16 20:00 UTC" in page
    assert "Sunday 13:00 ET" in page
    assert "Thursday 20:15 ET" in page
    assert "52%" in page  # historical accuracy 0.5205 in the footer byline

    # Forbidden fields never rendered, even though every one of them is present
    # on the input frame.
    assert_public_safe(page)
    assert "bookmaker" not in page.lower()


def test_render_picks_page_sorts_by_kickoff_not_confidence() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.index("SF at LA") < page.index("ARI at LAC")


def test_render_picks_page_uses_the_shared_design_system() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    # theme.stylesheet() role tokens, both dark scopes included.
    assert "--series-model: #2a78d6;" in page
    assert "@media (prefers-color-scheme: dark)" in page
    assert '.ats[data-theme="dark"]' in page
    # viz components, by their own class/label hooks.
    assert 'class="ats-sweep"' in page  # sweep_curve
    assert "Chance the pick covers" in page  # probability_meter
    assert "our number" in page  # line_journey
    assert 'class="kicker"' in page and 'class="hero num"' in page
    # The sweep interaction ships as its own script tag on this page only.
    assert "__atsSweepWired" in page
    # Simple top nav linking all three pages.
    for filename in (FINDINGS_PAGE, TRACK_RECORD_PAGE):
        assert f'href="{filename}"' in page


def test_render_picks_page_line_journey_omits_opener_archive_and_predicted_close() -> None:
    """MKT-09 licensing: only our fair line and the card's one consensus line.

    ``viz.line_journey`` renders a "close guess" legend entry whenever a
    predicted close is passed; the public site never passes one.
    """

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "our number" in page
    assert "close guess" not in page


def test_render_picks_page_strong_lean_gate() -> None:
    predictions = _predictions_fixture()
    explanations = {"2026_01_ARI_LAC": "Lineup continuity carries this one."}
    page = render_picks_page(predictions, _sweep_fixture(), explanations)
    # Row 1 residual -2.4 clears the 1.5-point gate; row 2 at -1.1 does not.
    assert "What we think the market is missing" in page
    assert "We make this line 2.4 points different from the market, on the ARI side." in page
    assert "Lineup continuity carries this one." in page
    assert "We land close to the market's number here" in page
    assert "1 strong lean<" in page


def test_render_picks_page_strong_lean_without_explanation_says_so() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "The per-game breakdown behind this lean has not been published" in page


def test_render_picks_page_header_carries_the_flat_confidence_note() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), season=2026, week=1)
    assert "strongest leans have not proven more likely to win than the rest" in page
    assert "no pick gets extra weight" in page


def test_render_picks_page_empty_predictions_still_has_shell() -> None:
    page = render_picks_page(pd.DataFrame())
    assert "No games are scheduled" in page
    assert "No pick card yet" in page
    assert_public_safe(page)


def test_render_picks_page_no_sweep_omits_curve_without_error() -> None:
    page = render_picks_page(_predictions_fixture(), sweep=None)
    assert "ARI at LAC" in page
    assert 'class="ats-sweep"' not in page


def test_render_picks_page_declares_utf8_charset_before_any_non_ascii() -> None:
    """A missing/late charset mojibakes the page's non-ASCII glyphs (the em dash
    and "≈" in the disclaimer, the "·" separators) on any static server that
    omits a charset response header. The <meta charset> tag must appear inside
    <head>, before the first non-ASCII byte."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), season=2026, week=1)
    head_index = page.index("<head>")
    charset_index = page.index('<meta charset="utf-8">')
    non_ascii_index = next(index for index, char in enumerate(page) if ord(char) > 127)
    assert head_index < charset_index < non_ascii_index


# ---------------------------------------------------------------------------
# render_findings_page
# ---------------------------------------------------------------------------


def test_render_findings_page_carries_every_finding_and_group() -> None:
    from nfl_ats.dashboard.findings_content import FINDINGS, GROUPS

    page = render_findings_page(generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC))
    for group in GROUPS:
        assert _rendered(group.title) in page
        assert _rendered(group.legend) in page
    for finding in FINDINGS:
        assert _rendered(finding.question) in page
        assert _rendered(finding.plain_answer) in page
        assert _rendered(finding.source) in page
    assert "Everything the research has settled, in plain English" in page
    assert "How to read any number on this dashboard" in page
    assert_public_safe(page)
    # No charts on this page, so no interaction wiring either.
    assert "__atsSweepWired" not in page


def test_render_findings_page_hero_tiles_render_as_stat_tiles() -> None:
    from nfl_ats.dashboard.findings_content import HERO_TILES

    page = render_findings_page()
    for tile in HERO_TILES:
        assert tile.value in page
        assert _rendered(tile.kicker) in page


# ---------------------------------------------------------------------------
# render_track_record_page
# ---------------------------------------------------------------------------


def _opener_metadata_fixture() -> dict[str, object]:
    return {
        "games": 1537,
        "metrics": {"opener_accuracy": 0.5249, "close_accuracy": 0.5109},
        "uncertainty": [
            {
                "block": "season",
                "metric": "opener_accuracy",
                "lower": 0.5021,
                "upper": 0.5431,
            },
            {"block": "week", "metric": "opener_accuracy", "lower": 0.499, "upper": 0.5508},
        ],
    }


def _season_summary_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2021],
            "games": [227, 239],
            "opener_accuracy": [0.4682, 0.5551],
            "close_accuracy": [0.4688, 0.5443],
        }
    )


def _active_fixture() -> dict[str, object]:
    return {
        "model_id": "model-123",
        "historical_evaluation": {
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"season": {"lower": 0.5019, "upper": 0.5414}},
        },
    }


def test_render_track_record_page_hero_tiles_and_seasons() -> None:
    page = render_track_record_page(
        _opener_metadata_fixture(),
        _season_summary_fixture(),
        _active_fixture(),
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
    )
    assert _rendered("Against the pool's line") in page
    assert "52.5%" in page
    assert "Against the closing line" in page
    assert "51.1%" in page
    assert "+2.5 points vs. a coin flip" in page
    assert "1,537 games" in page
    # The active model's own record and its season-blocked range.
    assert "1,080 correct out of 2,075 games" in page
    assert "could plausibly sit anywhere from 50.2% to 54.1%" in page
    # season_bars plus its table-view twin, including the losing season.
    assert "No season is left off" in page
    assert "46.8%" in page
    assert "1 of the 2 seasons finished above the coin flip." in page
    assert "2020 was the COVID season" in page
    # The honest-reading footer, quoting the artifact's own season range.
    assert "Four things that keep these numbers honest" in page
    assert _rendered("The pool grade's honest range runs from about 50.2% to 54.3%") in page
    assert_public_safe(page)


def test_render_track_record_page_without_artifacts_says_so() -> None:
    page = render_track_record_page()
    assert "The pool grade has not been measured yet" in page
    assert "No active model is linked yet" in page
    assert "Every headline here is the middle of a range" in page
    assert_public_safe(page)


# ---------------------------------------------------------------------------
# Artifact loaders and the end-to-end site build
# ---------------------------------------------------------------------------


def _write_board_fixture(
    root: Path, *, with_decomposition: bool = True, with_opener: bool = True
) -> None:
    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # The real recommendations.csv carries the raw market-feed columns; writing
    # them here is what gives the blocklist scan below something to catch.
    _leaky_predictions().to_csv(forecast / "recommendations.csv", index=False)
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
            "intervals": {"season": {"lower": 0.5019, "upper": 0.5414}},
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
                    "The model leans ARI by a hair.",
                    "The model leans ARI by a hair.",
                    "The model and market agree on this one.",
                ],
            }
        ).to_parquet(decomposition / "attribution.parquet", index=False)

    if with_opener:
        # Two runs: only the newest may be read.
        stale = root / "opener_evaluation" / "20250101T000000Z"
        stale.mkdir(parents=True)
        (stale / "metadata.json").write_text(
            json.dumps({"games": 1, "metrics": {"opener_accuracy": 0.99}}), encoding="utf-8"
        )
        opener = root / "opener_evaluation" / "20260101T000000Z"
        opener.mkdir(parents=True)
        (opener / "metadata.json").write_text(
            json.dumps(_opener_metadata_fixture()), encoding="utf-8"
        )
        _season_summary_fixture().to_csv(opener / "season_summary.csv", index=False)


def test_load_public_board_artifacts_reads_synchronized_chain(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    artifacts = load_public_board_artifacts(tmp_path)
    assert len(artifacts.predictions) == 2
    # sweep is filtered to the active method only (market rows excluded).
    assert set(artifacts.sweep["method"].unique()) == {"market_residual"}
    assert artifacts.explanations["2026_01_ARI_LAC"] == "The model leans ARI by a hair."
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


def test_load_opener_evaluation_artifacts_reads_the_newest_run(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    opener = load_opener_evaluation_artifacts(tmp_path)
    assert opener.metadata["games"] == 1537
    assert list(opener.seasons["season"]) == [2020, 2021]


def test_load_opener_evaluation_artifacts_absent_is_empty_not_an_error(tmp_path: Path) -> None:
    opener = load_opener_evaluation_artifacts(tmp_path)
    assert opener.metadata == {}
    assert opener.seasons.empty


def test_build_public_site_writes_three_pages(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    pages = build_public_site(tmp_path, generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC))
    assert set(pages) == {PICKS_PAGE, FINDINGS_PAGE, TRACK_RECORD_PAGE}

    for name, page in pages.items():
        assert page.rstrip().endswith("</html>"), name
        assert_public_safe(page)
        # Every page links to the other two and marks itself as current.
        assert 'aria-current="page"' in page
        for other in (PICKS_PAGE, FINDINGS_PAGE, TRACK_RECORD_PAGE):
            if other != name:
                assert f'href="{other}"' in page

    assert "The model leans ARI by a hair." in pages[PICKS_PAGE]
    assert "model-123" in pages[PICKS_PAGE]
    assert "2026-08-16 20:00 UTC" in pages[PICKS_PAGE]
    assert "52.5%" in pages[TRACK_RECORD_PAGE]
    assert "No season is left off" in pages[TRACK_RECORD_PAGE]
    assert "Everything the research has settled" in pages[FINDINGS_PAGE]


def test_build_public_site_without_an_opener_grade_still_builds(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path, with_opener=False)
    pages = build_public_site(tmp_path)
    assert "The pool grade has not been measured yet" in pages[TRACK_RECORD_PAGE]
    assert_public_safe(pages[TRACK_RECORD_PAGE])


def test_build_public_site_without_an_active_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No synchronized active ATS model"):
        build_public_site(tmp_path)
