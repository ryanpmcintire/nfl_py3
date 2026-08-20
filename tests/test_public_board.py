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

from nfl_ats import weak_signals
from nfl_ats.data import DataContractError
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.player_arrests_back_side_overlay import ArrestOverlayResult
from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    FINDINGS_PAGE,
    PICKS_PAGE,
    TRACK_RECORD_PAGE,
    build_public_site,
    confidence_word,
    load_opener_evaluation_artifacts,
    load_prospective_challengers,
    load_public_board_artifacts,
    pick_side,
    render_findings_page,
    render_picks_page,
    render_track_record_page,
    spread_words,
)
from nfl_ats.snapshots import write_snapshot
from nfl_ats.spread_explorer import SpreadExplorerGameParams

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


def test_render_picks_page_includes_the_season_ops_timeline() -> None:
    """D5 (owner request, 2026-08-20): the weekly cadence strip -- five
    checkpoints, the Week 1 lock date, and the movement-policy explanation --
    renders on the picks page by default (no ``challengers`` needed)."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "Season ops" in page
    assert "How a week actually happens now" in page
    assert "Locks Tuesday, September 8, 2026" in page
    for day in ("Tue", "Wed", "Thu", "Sat", "Sun AM"):
        assert f"&middot; {day}<" in page
    assert "If the market moves a full point, we follow it" in page
    assert "Sunday-night and Monday-night games lock here too, early" in page
    assert_public_safe(page)


def test_render_picks_page_movement_policy_note_quotes_the_registered_evidence() -> None:
    """The movement-policy note must QUOTE the registered evidence sentence
    from ``model_only_refresh_incumbent`` -- never re-type a number by hand
    -- when that challenger is passed in."""

    challengers = [
        {
            "challenger_id": "model_only_refresh_incumbent",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {"threshold_frozen": "1.0, exactly as measured in this test fixture."},
        }
    ]
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), challengers=challengers)
    assert "1.0, exactly as measured in this test fixture." in page
    assert "Not yet measured on this build" not in page


def test_render_picks_page_movement_policy_note_degrades_without_the_challenger() -> None:
    """Omitting ``challengers`` (every existing caller/test) must degrade to
    the generic pointer, never invent a number or raise."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "Not yet measured on this build" in page


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


def _weak_signal_payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "recorded_at": "2026-08-19",
        "description": "a synthetic open lead for tests",
        "source": "docs/example.md",
        "effect": 0.6,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2020, 2025],
        "probability_positive": 0.82,
        "interval": [-0.2, 1.4],
    }
    body.update(overrides)
    return body


def _weak_signal_registry_fixture() -> weak_signals.Registry:
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": {"synthetic_open_lead": _weak_signal_payload()},
    }
    return weak_signals.registry_from_payload(payload)


def test_render_findings_page_renders_the_watching_section_from_a_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "What we're watching" is generated straight from the registry, with no
    prose written by hand -- prove it renders a synthetic lead end to end.

    ``FINDINGS``/``LEAD_BLURBS`` are emptied here so curation validation has
    nothing to check against the synthetic single-entry registry fixture;
    this test is about the auto-leads section, not curation (see the
    dedicated curation tests below and in ``tests/test_findings_registry.py``).
    """

    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    page = public_board.render_findings_page(weak_signal_registry=_weak_signal_registry_fixture())
    assert "What we&#x27;re watching" in page
    assert "a synthetic open lead for tests" in page
    assert "P+ 0.82" in page
    assert "1 recorded signals" in page  # the fixture registry has exactly one entry
    assert_public_safe(page)


def test_render_findings_page_lists_challengers_when_given_some(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    challengers = [
        {
            "challenger_id": "synthetic_challenger",
            "status": "ACTIVE_PROSPECTIVE",
            "status_reason": "a synthetic challenger for tests",
        }
    ]
    page = public_board.render_findings_page(
        weak_signal_registry=_weak_signal_registry_fixture(), challengers=challengers
    )
    assert "synthetic challenger" in page
    assert "active prospective" in page


# ---------------------------------------------------------------------------
# Challenger board: caveat chips, opener/close divergence, and deactivation
# ---------------------------------------------------------------------------


def test_render_findings_page_challenger_card_shows_caveat_chips_and_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    challengers = [
        {
            "challenger_id": "synthetic_caveat_challenger",
            "status": "ACTIVE_PROSPECTIVE",
            "status_reason": "x",
            "evidence": {
                "classification": "unresolved_below_power",
                "probability_positive": 0.9,
                "tuesday_visibility_caveat": "A long caveat about Tuesday visibility timing.",
                "opener_graded": {"probability_positive": 0.8},
                "close_graded": {"probability_positive": 0.2},
            },
        }
    ]
    page = public_board.render_findings_page(
        weak_signal_registry=_weak_signal_registry_fixture(), challengers=challengers
    )
    assert "tuesday visibility caveat" in page
    assert "opener/close disagree in sign" in page
    # The caveat's own long prose is never inlined as a chip label.
    assert "A long caveat about Tuesday visibility timing." not in page


def test_render_findings_page_challenger_card_greys_a_deactivated_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    challengers = [
        {
            "challenger_id": "synthetic_deactivated_challenger",
            "status": "DEACTIVATED_STRUCTURAL_NO_OP",
            "status_reason": "Original registration rationale, kept for the audit trail.",
            "status_reason_update": (
                "DEACTIVATED because the underlying data source cannot populate this field "
                "before kickoff. Measured across 816 simulated team-weeks: zero clean flips."
            ),
            "evidence": {},
        }
    ]
    page = public_board.render_findings_page(
        weak_signal_registry=_weak_signal_registry_fixture(), challengers=challengers
    )
    assert 'style="opacity:0.6;"' in page
    assert "Why it is not live" in page
    assert "DEACTIVATED because the underlying data source cannot populate" in page
    assert "<summary>Full reason</summary>" in page
    # The (superseded) original registration rationale must never be shown
    # instead of the later correction when both are present.
    assert "Original registration rationale" not in page


def test_render_findings_page_challenger_card_active_entry_is_not_greyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    challengers = [
        {"challenger_id": "synthetic_active", "status": "ACTIVE_PROSPECTIVE", "status_reason": "y"}
    ]
    page = public_board.render_findings_page(
        weak_signal_registry=_weak_signal_registry_fixture(), challengers=challengers
    )
    assert "opacity:0.6" not in page
    assert "Why it is not live" not in page


def test_first_sentence_truncates_long_prose_with_an_ellipsis() -> None:
    from nfl_ats.public_board import _first_sentence

    text = "First sentence here. Second sentence that should not appear in the lead."
    assert _first_sentence(text) == "First sentence here."
    long_no_period = "word " * 100
    lead = _first_sentence(long_no_period, max_len=50)
    assert lead.endswith("...")
    assert len(lead) <= 53


# ---------------------------------------------------------------------------
# Challenger week-preview dispatch: best_pick_nomination_v3 and the honest
# "evaluated at lock time" fallback for heavy-refit / live-fetch challengers
# ---------------------------------------------------------------------------


def test_challenger_week_previews_lock_time_notes_for_heavy_or_live_challengers(
    tmp_path: Path,
) -> None:
    from nfl_ats import public_board
    from nfl_ats.card_view import resolve_overlay

    challengers = [
        {"challenger_id": challenger_id, "status": "ACTIVE_PROSPECTIVE"}
        for challenger_id in (
            "ecdf_mapping_incumbent",
            "era_weighted_half_life_8",
            "forecast_cold_visitor_tilt",
            "model_only_refresh_incumbent",
        )
    ]
    predictions = _predictions_fixture()
    overlay = resolve_overlay(predictions, None)
    previews = public_board._challenger_week_previews(
        challengers, predictions, tmp_path, overlay=overlay, nomination=None
    )
    assert "Evaluated at lock time" in previews["ecdf_mapping_incumbent"]
    assert "Evaluated at lock time" in previews["era_weighted_half_life_8"]
    assert "LIVE weather forecast" in previews["forecast_cold_visitor_tilt"]
    assert "Thursday/Saturday/Sunday refresh pass" in previews["model_only_refresh_incumbent"]


def test_challenger_week_previews_best_pick_v3_degrades_without_market_data(
    tmp_path: Path,
) -> None:
    from nfl_ats import public_board
    from nfl_ats.card_view import resolve_overlay

    challengers = [{"challenger_id": "best_pick_nomination_v3", "status": "ACTIVE_PROSPECTIVE"}]
    predictions = _predictions_fixture()
    overlay = resolve_overlay(predictions, None)
    previews = public_board._challenger_week_previews(
        challengers,
        predictions,
        tmp_path,
        overlay=overlay,
        nomination=None,
        metadata={},
    )
    assert "Could not be computed this week" in previews["best_pick_nomination_v3"]


# ---------------------------------------------------------------------------
# The research funnel strip: N signals -> N live challengers -> 1 active model
# ---------------------------------------------------------------------------


def test_render_findings_page_research_funnel_strip_counts_from_the_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every funnel number must be DERIVED from the fixture inputs in this
    test, never pasted as a literal that happens to match today's code --
    that is the whole point of "computed at build time, never hardcoded"."""

    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": {
            "synthetic_lead_1": _weak_signal_payload(),
            "synthetic_lead_2": _weak_signal_payload(
                description="a second synthetic lead",
                probability_positive=0.22,
                interval=[-1.0, 0.3],
            ),
            "synthetic_lead_3": _weak_signal_payload(
                description="a third synthetic lead",
                probability_positive=0.63,
                interval=[-0.5, 1.5],
            ),
        },
    }
    registry = weak_signals.registry_from_payload(payload)
    challengers = [
        {"challenger_id": "a", "status": "ACTIVE_PROSPECTIVE", "status_reason": "x"},
        {"challenger_id": "b", "status": "ACTIVE_PROSPECTIVE", "status_reason": "y"},
        {"challenger_id": "c", "status": "CLOSED_BEFORE_ACTIVATION", "status_reason": "z"},
    ]

    page = public_board.render_findings_page(
        weak_signal_registry=registry, challengers=challengers, active_model_id="model-123"
    )

    # Every expectation below is DERIVED from the fixture data above, never a
    # literal pasted from today's rendering -- that is the point of the test.
    expected_signals = len(registry.signals)
    expected_active_challengers = sum(
        1 for entry in challengers if entry["status"] == "ACTIVE_PROSPECTIVE"
    )

    assert "The research pipeline" in page
    # Exact kicker-then-value fragments (mirrors ``viz.stat_tile``'s markup),
    # not a bare digit check, so this cannot pass by matching an unrelated
    # number elsewhere on the page.
    assert (
        f'<p class="kicker">Signals recorded</p><div class="hero num">{expected_signals:,}</div>'
        in page
    )
    assert (
        '<p class="kicker">Live 2026 challengers</p>'
        f'<div class="hero num">{expected_active_challengers}</div>' in page
    )
    assert '<p class="kicker">Active model</p><div class="hero num">1</div>' in page
    assert_public_safe(page)


def test_render_findings_page_research_funnel_strip_zero_without_an_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``active_model_id`` (a build with no synchronized active model, or
    a direct caller that omits it) must show 0, not silently drop the tile
    or show a stale "1"."""

    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    page = public_board.render_findings_page(weak_signal_registry=_weak_signal_registry_fixture())
    assert '<p class="kicker">Active model</p><div class="hero num">0</div>' in page
    # No challengers passed either -- the challenger count must also read 0, not omit itself.
    assert '<p class="kicker">Live 2026 challengers</p><div class="hero num">0</div>' in page


# ---------------------------------------------------------------------------
# Era-magnitude row on era_trend_* lead cards
# ---------------------------------------------------------------------------


def _era_magnitude_profile_fixture(root: Path) -> None:
    directory = root / "era_magnitude_profile" / "20260819T204710Z"
    directory.mkdir(parents=True)
    payload = {
        "fixed_eras": [
            {"key": "era_2009_2014", "season_lo": 2009, "season_hi": 2014},
            {"key": "era_2015_2019", "season_lo": 2015, "season_hi": 2019},
            {"key": "era_2020_2025", "season_lo": 2020, "season_hi": 2025},
        ],
        "signals": {
            "synthetic_construct": {
                "era_results": {
                    "era_2009_2014": {"insufficient_data": True},
                    "era_2015_2019": {
                        "insufficient_data": False,
                        "effect": 1.5,
                        "week_blocked": {
                            "lower": 0.2,
                            "upper": 2.8,
                            "probability_positive": 0.97,
                        },
                    },
                    "era_2020_2025": {
                        "insufficient_data": False,
                        "effect": 3.0,
                        "week_blocked": {
                            "lower": 1.0,
                            "upper": 5.0,
                            "probability_positive": 0.99,
                        },
                    },
                }
            },
            # A second, flat-schema variant actually present in the real
            # artifact (production_model_opener_proxy_edge): no nested
            # "week_blocked" object and no "effect" key at all -- the point
            # estimate is "estimate", with lower/upper/probability_positive
            # directly on the era row.
            "synthetic_flat_construct": {
                "era_results": {
                    "era_2015_2019": {
                        "insufficient_data": False,
                        "estimate": 0.89,
                        "lower": -1.84,
                        "upper": 3.64,
                        "probability_positive": 0.734,
                    },
                    "era_2020_2025": {
                        "insufficient_data": False,
                        "estimate": 3.36,
                        "lower": 0.60,
                        "upper": 6.04,
                        "probability_positive": 0.9908,
                    },
                }
            },
        },
    }
    (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_era_magnitude_profile_missing_directory_is_empty(tmp_path: Path) -> None:
    from nfl_ats.public_board import load_era_magnitude_profile

    assert load_era_magnitude_profile(tmp_path) == {}


def test_load_era_magnitude_profile_parses_eras_and_skips_insufficient_data(
    tmp_path: Path,
) -> None:
    from nfl_ats.public_board import load_era_magnitude_profile

    _era_magnitude_profile_fixture(tmp_path)
    profile = load_era_magnitude_profile(tmp_path)
    assert set(profile.keys()) == {"synthetic_construct", "synthetic_flat_construct"}
    rows = profile["synthetic_construct"]
    assert [row.era_label for row in rows] == ["2015-2019", "2020-2025"]
    assert rows[0].effect == 1.5
    assert rows[0].interval == (0.2, 2.8)
    assert rows[0].probability_positive == 0.97


def test_load_era_magnitude_profile_supports_the_flat_estimate_schema(tmp_path: Path) -> None:
    """The real artifact carries a second shape for at least one signal
    (production_model_opener_proxy_edge): no nested ``week_blocked``, the
    point estimate stored as ``estimate`` instead of ``effect``. Both shapes
    must parse, or that signal's card silently loses its era row."""

    from nfl_ats.public_board import load_era_magnitude_profile

    _era_magnitude_profile_fixture(tmp_path)
    profile = load_era_magnitude_profile(tmp_path)
    rows = profile["synthetic_flat_construct"]
    assert [row.era_label for row in rows] == ["2015-2019", "2020-2025"]
    assert rows[0].effect == 0.89
    assert rows[0].interval == (-1.84, 3.64)
    assert rows[0].probability_positive == 0.734


def test_render_findings_page_era_trend_lead_shows_per_era_magnitude(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": {
            "era_trend_synthetic_construct": _weak_signal_payload(
                description="Season-trend magnitude drift of a synthetic construct.",
                probability_positive=0.97,
                interval=[0.5, 4.2],
            ),
        },
    }
    registry = weak_signals.registry_from_payload(payload)
    _era_magnitude_profile_fixture(tmp_path)

    page = public_board.render_findings_page(weak_signal_registry=registry, artifacts_root=tmp_path)
    assert "Same pattern, three eras" in page
    assert "2015-2019" in page
    assert "2020-2025" in page
    # The insufficient-data era must never render as a silent zero.
    assert "2009-2014" not in page
    assert_public_safe(page)


def test_render_findings_page_without_artifacts_root_omits_the_era_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board

    monkeypatch.setattr(public_board, "FINDINGS", ())
    monkeypatch.setattr(public_board, "LEAD_BLURBS", ())
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": {
            "era_trend_synthetic_construct": _weak_signal_payload(
                description="Season-trend magnitude drift of a synthetic construct.",
                probability_positive=0.97,
                interval=[0.5, 4.2],
            ),
        },
    }
    registry = weak_signals.registry_from_payload(payload)
    page = public_board.render_findings_page(weak_signal_registry=registry)
    assert "Same pattern, three eras" not in page
    assert_public_safe(page)


def test_render_findings_page_raises_when_a_cited_registry_entry_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The freshness contract, exercised at the RENDER boundary: a curated
    finding whose registry_fingerprints no longer match the live registry
    must fail the build loudly, not ship a stale claim quietly."""

    from nfl_ats import public_board
    from nfl_ats.dashboard.findings_content import Finding
    from nfl_ats.findings_registry import CurationError

    stale_finding = Finding(
        question="Does the synthetic lead help?",
        verdict="unproven",
        plain_answer="A stale claim about a signal that has since moved.",
        detail="This finding's fingerprint deliberately does not match the live entry.",
        source="docs/example.md",
        registry_keys=("weak_signal:synthetic_open_lead",),
        registry_fingerprints=("a-fingerprint-that-will-never-match",),
        curated_as_of="2020-01-01",
    )
    monkeypatch.setattr(public_board, "FINDINGS", (stale_finding,))

    with pytest.raises(CurationError, match="is stale against"):
        public_board.render_findings_page(weak_signal_registry=_weak_signal_registry_fixture())


def test_render_findings_page_raises_on_a_registry_key_that_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfl_ats import public_board
    from nfl_ats.dashboard.findings_content import Finding
    from nfl_ats.findings_registry import CurationError

    bogus_finding = Finding(
        question="Does a nonexistent signal help?",
        verdict="unproven",
        plain_answer="This cites a key nobody ever recorded.",
        detail="detail",
        source="docs/example.md",
        registry_keys=("weak_signal:this_key_was_never_recorded",),
        registry_fingerprints=("anything",),
        curated_as_of="2026-08-19",
    )
    monkeypatch.setattr(public_board, "FINDINGS", (bogus_finding,))

    with pytest.raises(CurationError, match="does not exist"):
        public_board.render_findings_page(weak_signal_registry=_weak_signal_registry_fixture())


# ---------------------------------------------------------------------------
# render_track_record_page
# ---------------------------------------------------------------------------


def _opener_metadata_fixture() -> dict[str, object]:
    return {
        "games": 1537,
        # Must match the active model's profile (_active_fixture) or the loader
        # will correctly refuse to publish this run's numbers.
        "active_model_config": {"feature_profile": "player"},
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
    assert _rendered("Model baseline at the pool's line") in page
    assert "52.5%" in page
    assert "Promoted arrest policy evaluation" in page
    assert "53.76%" in page
    assert "probability_positive=0.8562" in page
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
    # The rule explainer degrades gracefully with no opener-evaluation artifact
    # at all, rather than crashing or silently omitting the section.
    assert "How the picks are graded" in page
    assert "has not been measured on this archive yet" in page
    assert_public_safe(page)


# ---------------------------------------------------------------------------
# The grading-rule explainer: production rule vs. sign rule, in plain English
# ---------------------------------------------------------------------------


def _opener_metadata_with_both_rules_fixture() -> dict[str, object]:
    """Mirrors the real shape of ``artifacts/opener_evaluation/20260819T174244Z/
    metadata.json`` closely enough to exercise both rule branches: sign-rule
    fields (``opener_accuracy``) AND production probability-rule fields
    (``opener_accuracy_probability_rule``), read from the same run."""

    return {
        "games": 1537,
        "active_model_config": {"feature_profile": "player"},
        "metrics": {
            "opener_accuracy": 0.5283,
            "close_accuracy": 0.5156,
            "opener_accuracy_probability_rule": 0.5336,
            "close_accuracy_probability_rule": 0.5209,
        },
        "uncertainty": [],
    }


def test_render_track_record_page_rule_explainer_names_baseline_and_played_policy() -> None:
    page = render_track_record_page(_opener_metadata_with_both_rules_fixture())
    assert "How the picks are graded" in page
    assert "The model baseline and played policy" in page
    assert "The raw model probability rule -- the baseline beneath today" in page
    assert "The played policy:" in page
    assert "The sign rule -- the original grading protocol:" in page
    # Both numbers come from the SAME artifact reading the tiles below use --
    # no number is invented for this section.
    assert "scores 53.4% at the opener on this archive" in page
    assert "scores 52.8% on the same games" in page
    assert "53.76% versus 53.36%" in page
    assert_public_safe(page)


def test_render_track_record_page_rule_explainer_falls_back_to_sign_rule_only_artifact() -> None:
    """An artifact predating the two-rule evaluator (no ``*_probability_rule``
    keys) must still explain both rules in plain English -- it just cannot
    quote a production-rule number that was never measured."""

    page = render_track_record_page(_opener_metadata_fixture())
    assert "The raw model probability rule -- the baseline beneath today" in page
    assert "The played policy:" in page
    assert "has not been measured on this archive yet" in page
    # The sign-rule grade IS available on this artifact and must still be quoted.
    assert "scores 52.5% on the same games" in page
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
        # Three runs, mirroring the 2026-08-18 incident:
        #   - an older run of the ACTIVE profile (must lose to the newer one),
        #   - the active profile's real run (must win),
        #   - a NEWER run of a DIFFERENT profile (must be skipped entirely).
        # Before the profile filter existed the last of these silently won and
        # published another model's grade under the active model's id.
        stale = root / "opener_evaluation" / "20250101T000000Z"
        stale.mkdir(parents=True)
        (stale / "metadata.json").write_text(
            json.dumps(
                {
                    "games": 1,
                    "active_model_config": {"feature_profile": "player"},
                    "metrics": {"opener_accuracy": 0.99},
                }
            ),
            encoding="utf-8",
        )
        opener = root / "opener_evaluation" / "20260101T000000Z"
        opener.mkdir(parents=True)
        (opener / "metadata.json").write_text(
            json.dumps(_opener_metadata_fixture()), encoding="utf-8"
        )
        _season_summary_fixture().to_csv(opener / "season_summary.csv", index=False)

        other_profile = root / "opener_evaluation" / "20260102T000000Z"
        other_profile.mkdir(parents=True)
        (other_profile / "metadata.json").write_text(
            json.dumps(
                {
                    "games": 1537,
                    "active_model_config": {"feature_profile": "player_value"},
                    "metrics": {"opener_accuracy": 0.4242, "close_accuracy": 0.4242},
                }
            ),
            encoding="utf-8",
        )


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


def test_load_opener_evaluation_artifacts_reads_the_newest_run_of_the_active_profile(
    tmp_path: Path,
) -> None:
    _write_board_fixture(tmp_path)
    opener = load_opener_evaluation_artifacts(tmp_path, active_feature_profile="player")
    # Newest wins WITHIN the active profile: the 2025 run (games=1) loses to the
    # 2026 one, while the even newer `player_value` run is skipped entirely.
    assert opener.metadata["games"] == 1537
    assert list(opener.seasons["season"]) == [2020, 2021]


def test_opener_artifacts_never_publish_a_different_models_grade(tmp_path: Path) -> None:
    """A newer run of another feature profile must not override the active model.

    Regression test for the 2026-08-18 incident: a ``player_value`` research run
    written minutes after the active ``weak_stack`` run took over the published
    track-record tiles, so the page showed 52.4%/51.8% while the active model's
    real figures were 52.83%/51.56% -- still credited to the active model by id.
    """

    _write_board_fixture(tmp_path)

    # The newest directory on disk belongs to the WRONG profile.
    newest = sorted((tmp_path / "opener_evaluation").iterdir(), reverse=True)[0]
    newest_metadata = json.loads((newest / "metadata.json").read_text(encoding="utf-8"))
    assert newest_metadata["active_model_config"]["feature_profile"] == "player_value"

    opener = load_opener_evaluation_artifacts(tmp_path, active_feature_profile="player")
    assert opener.metadata["active_model_config"]["feature_profile"] == "player"
    assert opener.metadata["metrics"]["opener_accuracy"] == 0.5249

    # No run for the active profile at all is empty, never another model's run.
    empty = load_opener_evaluation_artifacts(tmp_path, active_feature_profile="nonexistent")
    assert empty.metadata == {}
    assert empty.seasons.empty


def test_load_opener_evaluation_artifacts_absent_is_empty_not_an_error(tmp_path: Path) -> None:
    opener = load_opener_evaluation_artifacts(tmp_path)
    assert opener.metadata == {}
    assert opener.seasons.empty


def test_build_public_site_writes_three_pages(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    pages = build_public_site(
        tmp_path,
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
        require_fresh_arrest_overlay=False,
    )
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

    # The research funnel strip threads the REAL active model id from
    # ``active_ats_model.json`` end to end (_write_board_fixture writes
    # "model-123") into a count of 1, not a hardcoded literal.
    assert '<p class="kicker">Active model</p><div class="hero num">1</div>' in pages[FINDINGS_PAGE]
    # The rule explainer is threaded onto the track-record page end to end.
    assert "How the picks are graded" in pages[TRACK_RECORD_PAGE]
    assert "model-123" in pages[PICKS_PAGE]
    assert "2026-08-16 20:00 UTC" in pages[PICKS_PAGE]
    assert "52.5%" in pages[TRACK_RECORD_PAGE]
    assert "No season is left off" in pages[TRACK_RECORD_PAGE]
    assert "Everything the research has settled" in pages[FINDINGS_PAGE]


def test_build_public_site_without_an_opener_grade_still_builds(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path, with_opener=False)
    pages = build_public_site(tmp_path, require_fresh_arrest_overlay=False)
    assert "The pool grade has not been measured yet" in pages[TRACK_RECORD_PAGE]
    assert_public_safe(pages[TRACK_RECORD_PAGE])


def test_build_public_site_without_an_active_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No synchronized active ATS model"):
        build_public_site(tmp_path, require_fresh_arrest_overlay=False)


def test_render_picks_page_marks_one_best_pick_from_the_confirmed_signal() -> None:
    """POL-09: exactly one game is badged, and it is the widest sweep run.

    The fixture sweep rises with line_offset, so ARI/LAC (a HOME pick at 0.55)
    holds >= 0.5 from offset 0.0 upward while SF/LA (an AWAY pick) holds from
    0.0 downward -- both width 0.5, so the game_id tie-break decides. What
    matters here is that the page badges exactly one, deterministically.
    """

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.count("BEST PICK OF THE WEEK") == 2  # the banner and one card badge


def test_render_picks_page_without_a_sweep_marks_no_best_pick() -> None:
    page = render_picks_page(_predictions_fixture(), pd.DataFrame())
    assert "BEST PICK OF THE WEEK" not in page


def test_render_picks_page_best_pick_is_regular_season_only() -> None:
    """The pool awards a Best Pick per regular-season week; playoffs get none."""

    predictions = _predictions_fixture()
    predictions["game_type"] = "DIV"
    page = render_picks_page(predictions, _sweep_fixture())
    assert "BEST PICK OF THE WEEK" not in page


# ---------------------------------------------------------------------------
# B1/B2 (2026-08-19): the overlay + Best Pick nomination, via nfl_ats.card_view
# ---------------------------------------------------------------------------


def _overlay_predictions_fixture() -> pd.DataFrame:
    """Two games: one a clean year-1-coach-fade candidate (KEEP hosts YR1,
    the model sides with YR1), one an unrelated control the overlay must not
    touch. Mirrors ``tests/test_publishing.py``'s own overlay fixture."""

    return pd.DataFrame(
        {
            "game_id": ["2026_01_KEEP_YR1", "2026_01_OTHER1_OTHER2"],
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "gameday": ["2026-09-10", "2026-09-10"],
            "weekday": ["Thursday", "Thursday"],
            "gametime": ["20:15", "20:15"],
            "kickoff": ["2026-09-11 00:15:00+00:00", "2026-09-11 00:15:00+00:00"],
            "away_team": ["YR1", "OTHER2"],
            "home_team": ["KEEP", "OTHER1"],
            "spread_line": [-3.5, 2.5],
            # KEEP (home, kept coach) is NOT picked -- YR1 (away, year-1) is.
            "home_cover_probability": [0.35, 0.55],
            "predicted_market_residual": [-2.0, 1.0],
            "fair_spread": [-1.0, 3.0],
            "method": ["market_residual", "market_residual"],
        }
    )


def _write_overlay_schedule_snapshot(data_root: Path) -> None:
    schedules = pd.DataFrame(
        [
            ("2025_01_KEEP_OPP", 2025, "REG", 1, "KEEP", "OPP", "Steady", "OppC"),
            ("2025_01_YR1_OPP2", 2025, "REG", 1, "YR1", "OPP2", "Old1", "OppC2"),
            ("2026_01_KEEP_YR1", 2026, "REG", 1, "KEEP", "YR1", "Steady", "New1"),
            ("2026_01_OTHER1_OTHER2", 2026, "REG", 1, "OTHER1", "OTHER2", "X", "Y"),
        ],
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
        ],
    )
    write_snapshot(
        schedules,
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )


def test_render_picks_page_applies_the_coach_fade_overlay_and_discloses_the_flip(
    tmp_path: Path,
) -> None:
    """B1: the live site used to show BAL (the model's own, un-overlaid pick)
    at IND while the published card had already flipped that pick to IND.
    The public page must render the OVERLAID pick and its disclosure."""

    _write_overlay_schedule_snapshot(tmp_path)
    page = render_picks_page(_overlay_predictions_fixture(), data_root=tmp_path)

    # The overlaid pick (KEEP), not the model's own raw pick (YR1).
    assert "KEEP" in page
    assert "1 pick flipped by the coach-fade overlay" in page
    assert "Coach-fade overlay applied" in page
    assert "Flipped from YR1 (the model" in page
    assert "to KEEP." in page
    assert_public_safe(page)


def test_render_picks_page_without_data_root_leaves_the_overlay_off(tmp_path: Path) -> None:
    """No ``data_root`` (or none with a snapshot) degrades to a no-op overlay,
    exactly like ``publishing.py`` and the dashboard."""

    page = render_picks_page(_overlay_predictions_fixture())
    assert "YR1" in page
    assert "coach-fade overlay" not in page.lower()


def test_render_picks_page_discloses_active_arrest_policy_when_no_pick_flips() -> None:
    predictions = _predictions_fixture()
    false_flags = pd.Series(False, index=predictions.index)
    arrest_overlay = ArrestOverlayResult(
        overlaid_predictions=predictions.copy(),
        flips=(),
        home_flags=false_flags,
        away_flags=false_flags,
    )

    page = render_picks_page(predictions, arrest_overlay=arrest_overlay)

    assert "player-arrest policy active" in page
    assert "0 picks flipped this week" in page
    assert "53.76%" in page
    assert_public_safe(page)


def test_render_picks_page_uses_v2_nomination_end_to_end(tmp_path: Path) -> None:
    """B2: end-to-end with a real walk-forward fit and a real dispersion pool,
    the same fixture shape as ``tests/test_publishing.py``'s v2 test -- the
    public page must show v2's nominee and its disclosure sentence, not v1's
    alphabetical-tie-break framing."""

    from datetime import date, timedelta

    import numpy as np

    from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS

    game_ids = ["2026_01_AAA_BBB", "2026_01_CCC_DDD", "2026_01_EEE_FFF"]
    train_rows = 150
    total = train_rows + len(game_ids)
    start = date(2019, 9, 1)
    index = np.arange(total)
    features = pd.DataFrame(
        {
            "game_id": [f"train_{v:03d}" for v in range(train_rows)] + game_ids,
            "season": np.where(index < train_rows, 2019, 2026),
            "week": np.where(index < train_rows, (index // 15) + 1, 1),
            "gameday": [start + timedelta(days=int(v)) for v in range(train_rows)]
            + [date(2026, 9, 10)] * len(game_ids),
            "away_team": "AWY",
            "home_team": "HME",
        }
    )
    all_features = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)
    for feature_index, column in enumerate(all_features, start=1):
        features[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    features["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    rng = np.random.default_rng(20260819)
    features["ats_margin"] = rng.normal(loc=0.0, scale=8.0, size=total)
    features["home_cover"] = (features["ats_margin"] > 0).astype(float)
    features["result"] = features["spread_line"] + features["ats_margin"]
    features.loc[index >= train_rows, ["home_cover", "ats_margin", "result"]] = np.nan
    features_path = tmp_path / "v2_features.parquet"
    features.to_parquet(features_path)

    predictions = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2026,
            "week": 1,
            "game_type": "REG",
            "gameday": ["2026-09-10"] * len(game_ids),
            "weekday": ["Thursday"] * len(game_ids),
            "gametime": ["20:15"] * len(game_ids),
            "kickoff": ["2026-09-11 00:15:00+00:00"] * len(game_ids),
            "away_team": ["AWY1", "AWY2", "AWY3"],
            "home_team": ["HME1", "HME2", "HME3"],
            "spread_line": [2.5, -2.5, 2.5],
            "home_cover_probability": [0.30, 0.60, 0.45],
            "method": "market_residual",
        }
    )
    metadata = {
        "season": 2026,
        "week": 1,
        "feature_profile": "base",
        "regressor": "ridge",
        "min_train_games": 100,
        "provenance": {"feature_table": {"path": str(features_path)}},
    }

    data_root = tmp_path / "data"
    snapshot_dir = data_root / "market" / "raw" / "20260818T130000Z"
    snapshot_dir.mkdir(parents=True)
    tuesday = pd.Timestamp("2026-08-18T13:00:00Z")
    kickoff = pd.Timestamp("2026-09-10T17:00:00Z")
    book_lines = {
        game_ids[0]: [2.5, 2.5],
        game_ids[1]: [-2.5, -3.0],
        game_ids[2]: [2.5, 4.5],
    }
    quotes_rows = [
        {
            "nflverse_game_id": game_id,
            "provider_event_id": game_id,
            "bookmaker_key": f"book{i}",
            "market": "spreads",
            "outcome_side": "HOME",
            "home_spread_line": line,
            "observed_at_utc": tuesday,
            "commence_time_utc": kickoff,
        }
        for game_id, lines in book_lines.items()
        for i, line in enumerate(lines)
    ]
    pd.DataFrame(quotes_rows).to_parquet(snapshot_dir / "quotes.parquet")

    page = render_picks_page(predictions, metadata=metadata, data_root=data_root)
    assert "nominated by calibrated probability among low-disagreement games" in page
    assert "24 of the 35" not in page  # the stale v1 tie-break framing is gone


# ---------------------------------------------------------------------------
# D1: the week board
# ---------------------------------------------------------------------------


def test_confidence_word_bands() -> None:
    assert confidence_word(0.50) == "slight"
    assert confidence_word(0.529) == "slight"
    assert confidence_word(0.53) == "lean"
    assert confidence_word(0.56) == "lean"
    assert confidence_word(0.561) == "strong"


def test_render_picks_page_week_board_anchors_to_each_card() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert 'class="data week-board"' in page
    assert '<a href="#2026_01_ARI_LAC">ARI at LAC</a>' in page
    assert '<a href="#2026_01_SF_LA">SF at LA</a>' in page
    assert 'id="2026_01_ARI_LAC"' in page
    assert 'id="2026_01_SF_LA"' in page
    # The board comes before the first detail card in document order.
    assert page.index('class="data week-board"') < page.index('id="2026_01_SF_LA"')


def test_render_picks_page_week_board_stars_the_best_pick() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    board = page[page.index('class="data week-board"') : page.index("</table>")]
    assert "best-flag" in board


# ---------------------------------------------------------------------------
# D2: the sweep curve is collapsed behind a details toggle
# ---------------------------------------------------------------------------


def test_render_picks_page_sweep_curve_is_collapsed_by_default() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    marker = "Confidence if the line moves (four points either side)"
    idx = page.index(marker)
    # The summary text sits inside a <details> tag, not a bare <p>, so the
    # chart it wraps starts collapsed.
    assert "<summary>" in page[idx - 40 : idx + len(marker) + 20]


# ---------------------------------------------------------------------------
# B4: an explanation whose own residual disagrees with the live card is
# dropped rather than rendered as a contradiction.
# ---------------------------------------------------------------------------


def test_load_public_board_artifacts_drops_a_stale_explanation(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path, with_decomposition=False)
    forecast = tmp_path / "margin_predictions" / "forecast"
    predictions = pd.read_csv(forecast / "recommendations.csv")
    # The live card's own residual for ARI at LAC is -2.4 (see _predictions_fixture).
    live_residual = float(
        predictions.loc[
            predictions["game_id"].eq("2026_01_ARI_LAC"), "predicted_market_residual"
        ].iloc[0]
    )
    assert abs(live_residual) > 2.0

    decomposition = tmp_path / "market_decomposition" / "20260101T000000Z"
    decomposition.mkdir(parents=True)
    pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "family": ["context", "context"],
            "explanation": [
                "The model essentially agrees with the market on this game (a 0.1-point gap).",
                "The model and market agree on this one.",
            ],
            # Stale for ARI/LAC (0.1 vs the live -2.4); consistent for SF/LA.
            "predicted_residual": [0.1, -1.1],
        }
    ).to_parquet(decomposition / "attribution.parquet", index=False)

    artifacts = load_public_board_artifacts(tmp_path)
    assert "2026_01_ARI_LAC" not in artifacts.explanations
    assert "2026_01_SF_LA" in artifacts.explanations


# ---------------------------------------------------------------------------
# B5: the season-caption tie handling
# ---------------------------------------------------------------------------


def test_render_track_record_page_season_caption_distinguishes_an_exact_tie() -> None:
    seasons = pd.DataFrame(
        {
            "season": [2020, 2021, 2022],
            "games": [227, 239, 255],
            "opener_accuracy": [0.5000, 0.5636, 0.5040],
            "close_accuracy": [0.5089, 0.5527, 0.4879],
        }
    )
    page = render_track_record_page(_opener_metadata_fixture(), seasons, _active_fixture())
    # Never claim a dead-even season finished "above" the coin flip.
    assert "3 of the 3 seasons finished above the coin flip" not in page
    assert "2 of the 3 seasons finished above the coin flip" in page
    assert "landed exactly at it (2020)" in page
    assert "2020 was the COVID season" in page


def test_render_track_record_page_season_caption_unchanged_without_a_tie() -> None:
    """No exact tie in the fixture -- the caption reads exactly as before B5."""

    page = render_track_record_page(
        _opener_metadata_fixture(), _season_summary_fixture(), _active_fixture()
    )
    assert "1 of the 2 seasons finished above the coin flip." in page
    assert "landed exactly at it" not in page


# ---------------------------------------------------------------------------
# D3: challengers + Best Pick sections on the track record page
# ---------------------------------------------------------------------------


def test_load_prospective_challengers_reads_the_registered_list(tmp_path: Path) -> None:
    payload = {
        "challengers": [
            {
                "challenger_id": "mod07_weak_signal_stack",
                "status": "ACTIVE_PROSPECTIVE",
                "evidence": {"registry_verdict": "unresolved", "probability_positive": 0.8745},
            },
            "not_a_dict_entry",
        ]
    }
    path = tmp_path / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    challengers = load_prospective_challengers(tmp_path)
    assert len(challengers) == 1
    assert challengers[0]["challenger_id"] == "mod07_weak_signal_stack"


def test_load_prospective_challengers_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_prospective_challengers(tmp_path) == []


def test_render_track_record_page_lists_challengers_from_the_registered_json() -> None:
    challengers = [
        {
            "challenger_id": "hc_year_one_fade_overlay",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {
                "classification": "unresolved_below_power",
                "probability_positive": 0.932,
            },
        },
        {
            "challenger_id": "player_qb_continuity|ridge_alpha=1|calibration=none",
            "status": "CLOSED_BEFORE_ACTIVATION",
            "evidence": {},
        },
    ]
    page = render_track_record_page(challengers=challengers)
    assert "The live test starts Sep 8, 2026" in page
    assert "hc year one fade overlay" in page
    assert "unresolved below power" in page
    assert "P+ 0.93" in page
    assert "player qb continuity" in page
    assert "CLOSED_BEFORE_ACTIVATION" not in page  # humanized, not the raw enum
    assert "closed before activation" in page


def test_render_track_record_page_without_challengers_omits_the_section() -> None:
    page = render_track_record_page()
    assert "The live test starts Sep 8, 2026" not in page


def test_render_track_record_page_best_pick_section_shows_the_honest_budget_for_v1() -> None:
    """v1 carries no long-form method_note, so the rule is spelled out inline."""

    page = render_track_record_page(
        best_pick_rule="v1", best_pick_team="ARI", best_pick_method_note="2 games tied."
    )
    assert "about +0.9 points" in page
    assert "not +8.7" in page
    assert "This week's nomination: <b>ARI</b>, chosen by " in page
    assert "the incumbent v1 rule (sweep_robustness)" in page
    assert "2 games tied." in page


def test_render_track_record_page_best_pick_section_v2_does_not_repeat_the_rule_name() -> None:
    """v2's own method_note already names the rule in full-sentence form; the
    section must not ALSO say "chosen by the v2 rule (...)" right next to it
    -- that would state the same thing twice in one paragraph."""

    page = render_track_record_page(
        best_pick_rule="v2",
        best_pick_team="MIA",
        best_pick_method_note="nominated by calibrated probability among low-disagreement games.",
    )
    assert "about +0.9 points" in page
    assert "This week's nomination: <b>MIA</b>. nominated by calibrated probability" in page
    assert "chosen by the v2 rule" not in page


def test_render_track_record_page_best_pick_section_without_a_nomination_says_so() -> None:
    page = render_track_record_page()
    assert "No Best Pick is nominated this week" in page


def test_build_public_site_threads_data_root_and_nomination_through_both_pages(
    tmp_path: Path,
) -> None:
    """The picks page and the track-record page's Best Pick section must
    never disagree about which game/rule is nominated -- they are computed
    ONCE in ``build_public_site`` and shared."""

    _write_board_fixture(tmp_path)
    pages = build_public_site(
        tmp_path,
        data_root=tmp_path / "data",
        require_fresh_arrest_overlay=False,
    )
    assert "This week's nomination:" in pages[TRACK_RECORD_PAGE]


# ---------------------------------------------------------------------------
# Spread explorer (owner request, 2026-08-20)
# ---------------------------------------------------------------------------


def _spread_explorer_params_fixture() -> dict[str, SpreadExplorerGameParams]:
    """Hand-built params for the two ``_predictions_fixture()`` games, chosen
    so ``home_cover_probability(card_line) == the fixture's own probability``
    to floating-point precision -- computed with scipy directly, independent
    of the widget's own erf approximation, so a rendering test never
    accidentally depends on the widget formula being correct."""

    from scipy import stats

    params = {}
    for game_id, home, away, line, target_probability in (
        ("2026_01_ARI_LAC", "LAC", "ARI", 3.5, 0.38),
        ("2026_01_SF_LA", "LA", "SF", -3.5, 0.62),
    ):
        mean, std, center = 0.0, 12.0, 0.0
        # Solve for a center that reproduces the target probability exactly
        # at (line, mean, std): threshold = line - center, want
        # 1 - Phi((threshold-mean)/std) == target_probability.
        z = stats.norm.isf(target_probability)
        threshold = z * std + mean
        center = line - threshold
        params[game_id] = SpreadExplorerGameParams(
            game_id=game_id,
            home_team=home,
            away_team=away,
            center=center,
            residual_mean=mean,
            residual_std=std,
            card_line=line,
            card_home_cover_probability=target_probability,
        )
    return params


def test_render_picks_page_renders_the_spread_explorer_widget() -> None:
    page = render_picks_page(
        _predictions_fixture(), _sweep_fixture(), spread_explorer=_spread_explorer_params_fixture()
    )
    assert page.count('class="spread-explorer"') == 2
    assert 'id="ats-se-data"' in page
    assert "New: spread explorer" in page
    assert "as of this build" in page
    # The initial slider value is the card's own line for each game.
    assert 'value="3.5"' in page
    assert 'value="-3.5"' in page
    # The JSON blob carries both games, keyed by game_id.
    match = re.search(r'id="ats-se-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    assert set(payload) == {"2026_01_ARI_LAC", "2026_01_SF_LA"}
    assert payload["2026_01_ARI_LAC"]["home"] == "LAC"
    assert payload["2026_01_ARI_LAC"]["line"] == 3.5


def test_render_picks_page_without_spread_explorer_omits_the_widget() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert 'class="spread-explorer"' not in page
    assert "ats-se-data" not in page
    assert "New: spread explorer" not in page


def test_render_picks_page_spread_explorer_only_renders_for_games_with_params() -> None:
    """A game missing from the map (e.g. it dropped out of the refit
    universe) renders without a widget rather than raising -- the same
    per-game graceful degradation every other optional card feature here
    follows."""

    one_game = {"2026_01_ARI_LAC": _spread_explorer_params_fixture()["2026_01_ARI_LAC"]}
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), spread_explorer=one_game)
    assert page.count('class="spread-explorer"') == 1
    assert 'data-game-id="2026_01_ARI_LAC"' in page
    assert 'data-game-id="2026_01_SF_LA"' not in page


def test_spread_explorer_widget_formula_matches_the_fixtures_own_probability() -> None:
    """The rendered JSON blob's own (center, mean, std), run through the SAME
    erf-based formula the browser widget uses, must reproduce the fixture's
    ``home_cover_probability`` at the card's own line -- proving the
    fixture above is internally consistent AND exercising the exact
    production formula (``widget_home_cover_probability``) end to end."""

    from nfl_ats.spread_explorer import widget_home_cover_probability

    for probability, params in zip(
        (0.38, 0.62), _spread_explorer_params_fixture().values(), strict=True
    ):
        computed = widget_home_cover_probability(
            params.card_line, params.center, params.residual_mean, params.residual_std
        )
        assert computed == pytest.approx(probability, abs=1e-6)


# ---------------------------------------------------------------------------
# Spread explorer -- end to end through build_public_site (a real refit)
# ---------------------------------------------------------------------------

_SE_FEATURE_PROFILE = "base"
_SE_RIDGE_ALPHA = 10.0
_SE_MIN_TRAIN_GAMES = 100
_SE_SEASON = 2020
_SE_WEEK = 4


def _write_gaussian_board_fixture(
    root: Path, data_root: Path, model_frame: pd.DataFrame
) -> pd.DataFrame:
    """A minimal, real (non-hand-typed) board fixture: the forecast card is
    built via an actual ``fit_margin_models_for_week`` refit at
    ``probability_method="gaussian"`` -- what ``compute_spread_explorer_params``
    needs to refit against and reproduce, unlike ``_write_board_fixture``'s
    hand-typed probabilities. Returns the card for assertions.
    """

    target, margin_models = fit_margin_models_for_week(
        model_frame,
        season=_SE_SEASON,
        week=_SE_WEEK,
        regressor="ridge",
        min_train_games=_SE_MIN_TRAIN_GAMES,
        feature_profile=_SE_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=_SE_RIDGE_ALPHA,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    predicted = model.predict(target, probability_method="gaussian")
    card = target.copy()
    card["method"] = "market_residual"
    card["home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
    card["predicted_market_residual"] = predicted["predicted_market_residual"].to_numpy()
    card["fair_spread"] = predicted["predicted_margin"].to_numpy()
    card["game_type"] = "REG"
    card["gameday"] = pd.to_datetime(card["gameday"]).dt.strftime("%Y-%m-%d")
    card["weekday"] = "Sunday"
    card["gametime"] = "13:00"
    card["kickoff"] = pd.to_datetime(card["gameday"]).astype(str)

    feature_path = data_root / "processed" / "spread_explorer_test_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_parquet(feature_path)

    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-gauss",
        "synchronization_status": "SYNCHRONIZED",
        "season": _SE_SEASON,
        "week": _SE_WEEK,
        "probability_method": "gaussian",
        "regressor": "ridge",
        "ridge_alpha": _SE_RIDGE_ALPHA,
        "feature_profile": _SE_FEATURE_PROFILE,
        "min_train_games": _SE_MIN_TRAIN_GAMES,
        "provenance": {"feature_table": {"path": str(feature_path)}},
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    card.to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-gauss",
        "method": "market_residual",
        "feature_profile": _SE_FEATURE_PROFILE,
        "regressor": "ridge",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": _SE_SEASON,
            "week": _SE_WEEK,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    return card


def test_build_public_site_renders_spread_explorer_for_a_gaussian_active_model(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_gaussian_board_fixture(artifacts_root, data_root, model_frame)

    pages = build_public_site(
        artifacts_root,
        data_root=data_root,
        require_fresh_arrest_overlay=False,
    )

    assert pages[PICKS_PAGE].count('class="spread-explorer"') == len(card)
    match = re.search(r'id="ats-se-data">(.*?)</script>', pages[PICKS_PAGE])
    assert match is not None
    payload = json.loads(match.group(1))
    assert set(payload) == set(card["game_id"].astype(str))


def test_build_public_site_without_gaussian_probability_method_omits_the_widget(
    tmp_path: Path,
) -> None:
    """``_write_board_fixture`` never sets ``probability_method`` (defaults to
    ``"ecdf"``) -- an older/rolled-back active model has no closed-form
    mean/sd the widget's formula can read, so the page must still build, just
    without the widget (the same graceful-degradation contract every other
    optional artifact here follows)."""

    _write_board_fixture(tmp_path)
    pages = build_public_site(tmp_path, require_fresh_arrest_overlay=False)
    assert 'class="spread-explorer"' not in pages[PICKS_PAGE]
    assert "ats-se-data" not in pages[PICKS_PAGE]


def test_build_public_site_refuses_a_drifted_gaussian_card(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    """The REQUIRED consistency check: if the published card's own
    ``home_cover_probability`` cannot be reproduced from a refit (e.g. the
    feature table drifted after the card was built), the build must fail
    loudly rather than ship a widget that could disagree with the published
    pick."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_gaussian_board_fixture(artifacts_root, data_root, model_frame)
    card_path = artifacts_root / "margin_predictions" / "forecast" / "recommendations.csv"
    card = pd.read_csv(card_path)
    card.loc[0, "home_cover_probability"] = 0.999999
    card.to_csv(card_path, index=False)

    with pytest.raises(DataContractError, match="do not"):
        build_public_site(
            artifacts_root,
            data_root=data_root,
            require_fresh_arrest_overlay=False,
        )
