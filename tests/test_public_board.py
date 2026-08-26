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
from html import escape, unescape
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import team_explorer, weak_signals
from nfl_ats.dashboard.findings_content import (
    HEADLINE,
    PLAYED_CARD_EXPECTATION_HERO,
)
from nfl_ats.data import DataContractError
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.player_arrests_back_side_overlay import (
    ArrestFlip,
    ArrestOverlayResult,
)
from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    FINDINGS_PAGE,
    LEDGER_PAGE,
    MODELS_PAGE,
    PICKS_PAGE,
    POOL_PAGE,
    TEAM_EXPLORER_PAGE,
    TRACK_RECORD_PAGE,
    _default_weak_signals_registry_path,
    build_public_site,
    confidence_word,
    glossary_abbr,
    load_model_ledger_html,
    load_opener_evaluation_artifacts,
    load_prospective_challengers,
    load_public_board_artifacts,
    load_waterfall_feed,
    pick_side,
    render_findings_page,
    render_models_page,
    render_picks_page,
    render_team_explorer_page,
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
    """A content string as the components escape it (``escape()`` quotes too),
    then ``**spans**`` render as <strong> (2026-08-24 emphasis law)."""

    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(text))


def _visible_text(page: str) -> str:
    """Everything a reader can see: no stylesheet, no script, no markup."""

    return _TAG.sub(" ", _HEAD_BLOCK.sub(" ", page))


def _forbidden_field_pattern(field: str) -> re.Pattern[str]:
    """Word-boundary match for a raw market-feed field name.

    A bare substring check was "unambiguous" only until ledger.html started
    rendering weak-signal registry NAMES, one of which is
    ``penalty_crew_flag_rate_high_total_line`` -- a legitimate English
    identifier that happens to contain the substring ``total_line`` with no
    boundary before it (``high_TOTAL_LINE``, not a leaked JSON key). ``_``
    counts as a word character in ``\\b``, so a boundary exists before a
    genuine leak like ``"total_line": 47.5`` (preceded by a quote) but NOT
    inside a compound identifier like that one (preceded by another ``_``) --
    this keeps the check exactly as strict against the real leak pattern
    while fixing the false positive.
    """

    return re.compile(r"\b" + re.escape(field) + r"\b")


def assert_public_safe(page: str) -> None:
    """Every guardrail a generated public page must satisfy."""

    for book in FORBIDDEN_BOOKS:
        assert book not in page
    for field in FORBIDDEN_FIELDS:
        assert not _forbidden_field_pattern(field).search(page), field
    text = _visible_text(page)
    for value in FORBIDDEN_VALUES:
        assert value not in text
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
    assert page.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert '<meta name="viewport"' in page
    assert '<div class="ats">' in page
    # No host theme-sync script may ever ship on a static page: it would poll
    # an element that does not exist here.
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

    # Consolidation law (2026-08-23): no aggregate accuracy byline in the
    # default view anymore -- the footer names the model and the line caveat
    # only.
    assert "long-run accuracy" not in _index_default_view(page)

    # Forbidden fields never rendered, even though every one of them is present
    # on the input frame.
    assert_public_safe(page)
    assert "bookmaker" not in page.lower()


def test_render_picks_page_sorts_by_kickoff_not_confidence() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.index("SF at LA") < page.index("ARI at LAC")


# ---------------------------------------------------------------------------
# Glossary tooltips (wave-1 UX finding: "P+", "Ledger mini", "Evidence P+",
# "Challenger watch" were unexplained at point of use on the picks page)
# ---------------------------------------------------------------------------


def test_glossary_covers_every_research_vocabulary_term() -> None:
    for term in ("P+", "Ledger mini", "Evidence P+", "Challenger watch"):
        abbr = glossary_abbr(term)
        assert abbr.startswith('<abbr title="')
        assert f">{term}</abbr>" in abbr
        # The tooltip text itself must never be empty.
        title = abbr.split('title="', 1)[1].split('"', 1)[0]
        assert len(title) > 20


def test_glossary_abbr_rejects_unknown_terms() -> None:
    with pytest.raises(KeyError, match="not in the site glossary"):
        glossary_abbr("totally made up term")


def test_render_picks_page_glosses_research_vocabulary_at_point_of_use() -> None:
    challengers = [
        {
            "challenger_id": "model_only_refresh_incumbent",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {"probability_positive": 0.93},
        }
    ]
    page = render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        challengers=challengers,
        model_id="model-123",
    )
    # Every flagged term carries its plain-language expansion where it appears.
    assert '<abbr title="' in page
    assert ">P+</abbr>" in page
    assert ">Ledger mini</abbr>" in page
    assert ">Evidence P+</abbr>" in page
    assert ">Challenger watch</abbr>" in page
    # The ledger row and the challenger-watch line both gloss their bare P+.
    assert '<td class="num"><abbr title="' in page
    assert "Our confidence that a measured effect is real" in page
    assert_public_safe(page)


def test_render_picks_page_glosses_survive_without_challengers() -> None:
    """The default build (no challengers.json yet) still glosses the panel
    titles and the Evidence P+ column header."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert ">Ledger mini</abbr>" in page
    assert ">Evidence P+</abbr>" in page
    assert ">Challenger watch</abbr>" in page
    assert_public_safe(page)


def test_render_picks_page_uses_the_shared_design_system() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    # theme.stylesheet() role tokens, both dark scopes included.
    assert "--series-model: #2a78d6;" in page
    assert "@media (prefers-color-scheme: dark)" in page
    assert '.ats[data-theme="dark"]' in page
    # viz components, by their own class/label hooks.
    assert 'class="ats-cover"' in page  # cover_curve
    assert "our number" in page  # line_journey
    assert 'class="kicker"' in page and 'class="num"' in page
    # The de-firehose deep dive: one collapsed line per game, tools behind a toggle.
    # Row 1: LAC favored by 3.5, model takes the dog ARI (62%) -- fair ARI +1.1.
    # Row 2: SF favored by 3.5, model takes the home dog LA (62%).
    assert (
        'Pick <b>ARI</b> (+3.5) &middot; covers <span class="num">62%</span> '
        '&middot; fair ARI <span class="num">+1.1</span>' in page
    )
    assert 'Pick <b>LA</b> (+3.5) &middot; covers <span class="num">62%</span>' in page
    assert "<summary>Cover odds across hypothetical lines</summary>" in page
    assert "Cover chance" not in page  # the per-game hero meter is gone
    # The cover-curve drag handler ships as its own script tag on this page only.
    assert "__atsCoverWired" in page
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
    # The no-opinion filler caption sentence is gone (2026-08-23 de-firehose):
    # the collapsed pick line already conveys "close to the market".
    assert "We land close to the market" not in page


def test_render_picks_page_strong_lean_count_matches_the_board_buckets() -> None:
    """B3: the At-a-glance lean count is COMPUTED from the frame the board
    renders, using the same confidence_word buckets -- never from a
    different threshold on a different frame."""

    predictions = _predictions_fixture()
    expected = sum(
        1 for _, row in predictions.iterrows() if confidence_word(pick_side(row)[1]) == "strong"
    )
    assert expected > 0  # the fixture must exercise the non-trivial branch
    page = render_picks_page(predictions, _sweep_fixture())
    assert f"{expected} strong lean{'s' if expected != 1 else ''}" in page


def test_render_picks_page_strong_lean_without_explanation_omits_the_block() -> None:
    """B4: a strong lean with no published breakdown omits the kicker and
    the whole block (fail-quiet) rather than promising insight it cannot
    deliver."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "What we think the market is missing" not in page
    assert "The per-game breakdown behind this lean has not been published" not in page


def test_render_picks_page_header_carries_the_flat_confidence_note() -> None:
    """2026-08-23 copy fix: the garbled strongest-leans dek is replaced by one
    plain sentence about what grading actually is."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), season=2026, week=1)
    assert "Picks are graded against Tuesday-frozen lines all season." in page
    assert "strongest" not in page
    assert "ultra-weight" not in page
    assert "no pick gets extra weight" not in page


def test_render_picks_page_empty_predictions_still_has_shell() -> None:
    page = render_picks_page(pd.DataFrame())
    assert "No games are scheduled" in page
    assert "No pick card yet" in page
    assert_public_safe(page)


def test_index_default_view_percentages_are_only_hero_measured_and_cover_chances() -> None:
    """Enumerate EVERY default-visible percentage on the consolidated index:
    each must be either the ≈55% planning hero, the measured chain-history
    line, or a per-game cover chance ('covers NN%'). Nothing else -- that is
    the owner's consolidation law stated positively."""

    chain = 0.541583499667332
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), played_chain_accuracy=chain)
    view = _index_default_view(page)
    cover_chances = {f"{pick_side(row)[1]:.0%}" for _, row in _predictions_fixture().iterrows()}
    # The hero carries its own approx sign; remove it whole so the digit scan
    # cannot re-match the bare number inside it.
    assert PLAYED_CARD_EXPECTATION_HERO in view
    without_hero = view.replace(PLAYED_CARD_EXPECTATION_HERO, " ")
    visible = re.findall(r"\d+(?:\.\d+)?%", without_hero)
    allowed = {f"{chain:.1%}", *cover_chances}
    for percentage in visible:
        assert percentage in allowed, (
            f"unexpected percentage {percentage!r} default-visible on index "
            f"(allowed: {sorted(allowed)})"
        )
    # ...and nothing allowed went missing.
    assert f"{chain:.1%}" in view
    for cover_chance in cover_chances:
        assert re.search(rf"covers\s+{cover_chance} ", view)


def test_render_picks_page_no_sweep_omits_curve_without_error() -> None:
    page = render_picks_page(_predictions_fixture(), sweep=None)
    assert "ARI at LAC" in page
    # No sweep AND no spread-explorer params: no chart at all for either
    # game, the same silent omission the two retired tools followed.
    assert 'class="ats-cover"' not in page


def test_render_picks_page_includes_the_season_ops_timeline() -> None:
    """D5 (owner request, 2026-08-20): the weekly cadence -- five checkpoints,
    the Week 1 lock date, and the movement-policy explanation -- renders on
    the picks page by default (no ``challengers`` needed), compressed to a
    flat strip by the 2026-08-23 redesign."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "Season ops" in page
    assert "the grading line freezes Tuesday" in page
    assert "Week 1, 2026 locks Tuesday, September 8, 2026." in page
    for day in ("Tue", "Wed", "Thu", "Sat", "Sun AM"):
        assert f"<b>{day}</b>" in page
    assert "If the market moves a full point, we follow it." in page
    assert "Sunday- and Monday-night games lock there too" in page
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


def test_render_team_explorer_page_empty_state_without_data() -> None:
    """No local feature table -> a quiet empty state, never an error."""

    page = render_team_explorer_page(pd.DataFrame(), generated_at=datetime(2026, 8, 24, tzinfo=UTC))
    assert "No team-state data yet" in page
    assert_public_safe(page)
    assert "<nav" in page and "Team trends" in page


def test_render_team_explorer_page_renders_trends_from_schema_fixture() -> None:
    """Driven with the schema fixture, the page emits the design-system markers
    and the per-team overview / trend / matchup sections (no real data needed)."""

    state_table = team_explorer.make_schema_fixture()
    page = render_team_explorer_page(state_table, generated_at=datetime(2026, 8, 24, tzinfo=UTC))
    assert_public_safe(page)
    # Overview + per-team trend + matchup comparison all present.
    assert "Team strength, game by game" in page
    assert "Season-by-season trends" in page
    assert "Head-to-head comparison" in page
    # The interactive comparer ships its payload and a script.
    assert 'id="ats-te-data"' in page
    assert 'id="ats-te-a"' in page
    # Latest season from the fixture is rendered.
    assert "Latest season shown: 2025" in page
    # Every team in the fixture appears in the overview.
    for team in ("ARI", "BUF", "KC", "SF"):
        assert f">{team}<" in page or f"{team}</b>" in page


def test_render_team_explorer_page_handles_unknown_metric_gracefully() -> None:
    """An unknown metric request must raise, not silently render garbage."""

    import pytest

    with pytest.raises(ValueError):
        render_team_explorer_page(
            team_explorer.make_schema_fixture(), metrics=["not_a_real_metric"]
        )


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
    assert "__atsCoverWired" not in page


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
    # P+ now carries its own diverging tone (above/below the 0.5 decision
    # midpoint), so the numeral sits inside a span. Asserting the tone as
    # well as the value is a stronger check than the old bare-text one.
    assert '<span class="delta pos">0.82</span>' in page
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


def _surface_preview_schedule() -> pd.DataFrame:
    """ARI hosts two REG games on grass (grass-modal home surface), then
    visits LAC on fieldturf -- the flagged grass-to-turf switch. SF has no
    home games, so its modal surface is unresolved and SF@LA never flags."""

    rows = [
        ("2026_02_ARI_OPPA", 2026, "REG", 2, "ARI", "OPPA", "grass"),
        ("2026_03_ARI_OPPB", 2026, "REG", 3, "ARI", "OPPB", "grass"),
        ("2026_01_ARI_LAC", 2026, "REG", 1, "LAC", "ARI", "fieldturf"),
        ("2026_01_SF_LA", 2026, "REG", 1, "LA", "SF", "fieldturf"),
    ]
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "game_type", "week", "home_team", "away_team", "surface"],
    )


def _surface_preview_predictions(**extra: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "season": [2026, 2026],
            "gameday": ["2026-09-13", "2026-09-10"],
            "weekday": ["Sunday", "Thursday"],
            "gametime": ["13:00", "20:15"],
            "kickoff": ["2026-09-13 17:00:00+00:00", "2026-09-11 00:15:00+00:00"],
            "game_type": ["REG", "REG"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [3.5, -3.5],
            # ARI@LAC picks AWAY (0.38 < 0.5) on a flagged game; SF@LA picks
            # HOME (0.62 >= 0.5), which the asymmetric rule never flips anyway.
            "home_cover_probability": [0.38, 0.62],
            "predicted_market_residual": [-2.4, -1.1],
            "fair_spread": [1.1, -4.6],
            "method": ["market_residual", "market_residual"],
        }
    )
    for column, values in extra.items():
        frame[column] = values
    return frame


def test_challenger_week_previews_surface_switch_tilt_flips_from_schedules(
    tmp_path: Path,
) -> None:
    """Call site (site builder): with a local schedule snapshot present, the
    surface-switch preview renders its flip sentence from the module's OWN
    schedules-derived flag."""

    from nfl_ats import public_board
    from nfl_ats.card_view import resolve_overlay

    write_snapshot(
        _surface_preview_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=tmp_path / "raw",
    )
    challengers = [{"challenger_id": "surface_switch_tilt_overlay", "status": "ACTIVE_PROSPECTIVE"}]
    predictions = _surface_preview_predictions()
    overlay = resolve_overlay(predictions, None)

    previews = public_board._challenger_week_previews(
        challengers,
        predictions,
        tmp_path,
        overlay=overlay,
        nomination=None,
    )

    assert "Would flip 1 pick on this week's card" in previews["surface_switch_tilt_overlay"]
    assert "ARI at LAC (ARI to LAC)" in previews["surface_switch_tilt_overlay"]


def test_challenger_week_previews_surface_switch_tilt_survives_a_preexisting_flag_column(
    tmp_path: Path,
) -> None:
    """The 2026-08-24 rehearsal crash: the feature table now ships
    ``surface_switch_flag`` as a model input, so the card can arrive with a
    same-named column. The preview must degrade to the documented no-op path
    -- deriving its own flag from schedules -- never raise KeyError, and a
    misleading foreign flag value must not change the verdict (absent and
    present render the SAME sentence)."""

    from nfl_ats import public_board
    from nfl_ats.card_view import resolve_overlay

    write_snapshot(
        _surface_preview_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=tmp_path / "raw",
    )
    challengers = [{"challenger_id": "surface_switch_tilt_overlay", "status": "ACTIVE_PROSPECTIVE"}]
    absent = _surface_preview_predictions()
    # Deliberately WRONG values: flags the unflagged game, clears the flagged
    # one -- the overlay's own derivation must win over both.
    present = _surface_preview_predictions(
        surface_switch_flag=[True, False],
    )
    overlay_absent = resolve_overlay(absent, None)
    overlay_present = resolve_overlay(present, None)

    previews_absent = public_board._challenger_week_previews(
        challengers,
        absent,
        tmp_path,
        overlay=overlay_absent,
        nomination=None,
    )
    previews_present = public_board._challenger_week_previews(
        challengers,
        present,
        tmp_path,
        overlay=overlay_present,
        nomination=None,
    )

    assert previews_absent == previews_present
    assert (
        "Would flip 1 pick on this week's card" in previews_present["surface_switch_tilt_overlay"]
    )
    assert "ARI at LAC (ARI to LAC)" in previews_present["surface_switch_tilt_overlay"]


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


def test_render_track_record_page_story_sections_and_appendix() -> None:
    """2026-08-24 re-architecture: the page leads with the six story sections
    (each canonical figure exactly once, from the pinned constants), then the
    appendix tables keep their fixture figures."""
    page = render_track_record_page(
        _opener_metadata_fixture(),
        _season_summary_fixture(),
        _active_fixture(),
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
    )
    # The story, in order, with its section kickers.
    assert "How good is this, honestly?" in page
    for kicker in (
        "THE PROJECT",
        "THE MODEL",
        "MEASURED",
        "PLANNING ESTIMATE",
        "TWO LINES, ONE RECORD",
        "FALSIFIABILITY",
    ):
        assert kicker in page
    assert "beats-the-line model that respects injuries" in page
    assert f"<b>{HEADLINE.opener}</b>" in page
    assert f"<b>{PLAYED_CARD_EXPECTATION_HERO}</b>" in page
    assert f"<b>{HEADLINE.close}</b>" in page
    assert "the ledger is the referee" in page
    # The appendix keeps the graded tables.
    assert "The tables behind every number above" in page
    assert "52.5%" in page
    assert "53.76%" in page
    assert ">P+</abbr> 0.86" in page
    assert "1,537 games" in page
    # The active model's own record and its season-blocked range.
    assert "1,080 correct out of 2,075 games" in page
    assert "Its plausible range runs from 50.2% to 54.1%" in page
    assert "Its plausible range runs from 50.2% to 54.1%" in page
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
    # The story still renders from pinned constants with no artifacts at all.
    assert "How good is this, honestly?" in page
    assert "MEASURED" in page
    assert "the measured chain figure appears here once its evaluation" in page
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


def test_build_public_site_writes_four_pages(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    pages = build_public_site(
        tmp_path,
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
        require_fresh_arrest_overlay=False,
    )
    assert set(pages) == {
        PICKS_PAGE,
        MODELS_PAGE,
        TEAM_EXPLORER_PAGE,
        FINDINGS_PAGE,
        TRACK_RECORD_PAGE,
        POOL_PAGE,
        LEDGER_PAGE,
    }

    for name, page in pages.items():
        assert page.rstrip().endswith("</html>"), name
        assert_public_safe(page)
        # Every page links to the others and marks itself as current.
        assert 'aria-current="page"' in page
        for other in (
            PICKS_PAGE,
            MODELS_PAGE,
            TEAM_EXPLORER_PAGE,
            FINDINGS_PAGE,
            TRACK_RECORD_PAGE,
            POOL_PAGE,
            LEDGER_PAGE,
        ):
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
    # The story renders from constants; the appendix rule explainer says the
    # grade has not been measured rather than inventing one.
    assert "How good is this, honestly?" in pages[TRACK_RECORD_PAGE]
    assert "has not been measured on this archive yet" in pages[TRACK_RECORD_PAGE]
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
    assert page.count("BEST PICK OF THE WEEK") == 1  # the P1 summary callout


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
    assert "flipped from YR1 (the model" in page
    assert "to KEEP.</p>" in page
    # Consolidation law (2026-08-23): the rule's historical cover rate stays
    # inside a collapsed toggle, not default-visible.
    assert "<summary>Rule evidence</summary>" in page
    assert "about 47%" in page
    assert "47%" not in _index_default_view(page)
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
    # Consolidation law (2026-08-23): the policy's archive evaluation no
    # longer rides in the footer -- it lives on track_record.html and,
    # per-game, behind the collapsed Policy-evidence toggle.
    assert "53.76%" not in page
    assert_public_safe(page)


def test_arrest_flip_evidence_percentages_stay_collapsed() -> None:
    """When the arrest policy flips a pick, its historical evaluation
    percentages must render ONLY inside a collapsed toggle -- never
    default-visible next to the picks."""

    predictions = _predictions_fixture()
    home_flags = pd.Series([True, False], index=predictions.index)
    flip = ArrestFlip(
        game_id="2026_01_ARI_LAC",
        matchup="ARI at LAC",
        original_pick_team="ARI",
        flipped_to_team="LAC",
    )
    arrest_overlay = ArrestOverlayResult(
        overlaid_predictions=predictions.copy(),
        flips=(flip,),
        home_flags=home_flags,
        away_flags=~home_flags,
    )

    page = render_picks_page(predictions, _sweep_fixture(), arrest_overlay=arrest_overlay)

    # The flip itself stays disclosed in plain sight; the archive numbers
    # stay one click away.
    assert "Arrest rule applied" in page
    assert "<summary>Policy evidence</summary>" in page
    default_view = _index_default_view(page)
    for banned in ("53.76%", "53.36%", "0.8562"):
        assert banned not in default_view
    assert "53.76%" in page  # still disclosed, collapsed
    assert ">P+</abbr> 0.86" in page


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
# D2: the cover curve is collapsed behind a details toggle
# ---------------------------------------------------------------------------


def test_render_picks_page_cover_curve_is_collapsed_by_default() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    marker = "Cover odds across hypothetical lines"
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
    assert "Year-one coach fade" in page
    assert "hc year one fade overlay" not in page
    assert "unresolved below power" in page
    assert '<span class="delta pos">0.93</span>' in page
    assert "QB-continuity alpha probe" in page
    assert "player qb continuity" not in page
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
    assert "+8.68" not in page
    assert "This week's nomination: <b>ARI</b>, chosen by " in page
    assert "the standard rule (most robust line sweep)" in page
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


def test_render_picks_page_renders_the_cover_curve_with_gaussian_payload() -> None:
    """2026-08-26 merge: the picks page carries ONE chart per game (the cover
    curve), whose shared script payload carries each game's Gaussian read
    (center/mean/std) for the on-chart slider's live drag -- replacing the
    retired standalone "Spread explorer" widget's own script/payload."""

    page = render_picks_page(
        _predictions_fixture(), _sweep_fixture(), spread_explorer=_spread_explorer_params_fixture()
    )
    assert page.count('class="ats-cover"') == 2
    assert 'id="ats-cover-data"' in page
    assert "as of this build" in page
    # The chart's own domain is OFFSETS from the quoted line (see
    # SWEEP_HALF_WIDTH), so the slider's initial value is always 0 -- the
    # card's own line is reproduced trivially at that offset, regardless of
    # whether it is a whole or half-point number.
    assert page.count('value="0"') >= 2
    # The JSON blob carries both games, keyed by game_id, WITH their Gaussian
    # fit for the drag handler's erf formula.
    match = re.search(r'id="ats-cover-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    assert set(payload) == {"2026_01_ARI_LAC", "2026_01_SF_LA"}
    assert payload["2026_01_ARI_LAC"]["home"] == "LAC"
    assert payload["2026_01_ARI_LAC"]["line"] == 3.5
    assert "center" in payload["2026_01_ARI_LAC"]
    assert "center" in payload["2026_01_SF_LA"]


def test_render_picks_page_without_spread_explorer_still_charts_from_real_sweep() -> None:
    """No Gaussian params: the chart still renders (real ``line_sweep`` rows
    are the preferred source regardless), but the shared payload carries no
    Gaussian fields for these games, so the drag handler's JS falls back to
    linear interpolation across the real points instead of the erf formula."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.count('class="ats-cover"') == 2
    match = re.search(r'id="ats-cover-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    assert set(payload) == {"2026_01_ARI_LAC", "2026_01_SF_LA"}
    for game in payload.values():
        assert "center" not in game


def test_render_picks_page_gaussian_payload_only_for_games_with_params() -> None:
    """A game missing from the Gaussian map (e.g. it dropped out of the refit
    universe) still gets a chart from its real sweep row, but its payload
    entry carries no Gaussian fields -- the same per-game graceful
    degradation every other optional card feature here follows, now scoped
    to the FORMULA rather than to the whole chart."""

    one_game = {"2026_01_ARI_LAC": _spread_explorer_params_fixture()["2026_01_ARI_LAC"]}
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), spread_explorer=one_game)
    assert page.count('class="ats-cover"') == 2
    match = re.search(r'id="ats-cover-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    assert "center" in payload["2026_01_ARI_LAC"]
    assert "center" not in payload["2026_01_SF_LA"]


def test_render_picks_page_gaussian_only_game_still_gets_a_chart_without_sweep() -> None:
    """A game with a Gaussian read but NO saved sweep row (an older or
    rolled-back artifact tree) still gets a chart, synthesized from the same
    closed-form formula (:data:`nfl_ats.public_board._COVER_CURVE_FALLBACK_OFFSETS`)
    rather than being left blank -- the graceful-degradation contract runs
    both directions."""

    page = render_picks_page(
        _predictions_fixture(), sweep=None, spread_explorer=_spread_explorer_params_fixture()
    )
    assert page.count('class="ats-cover"') == 2
    match = re.search(r'id="ats-cover-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    for game in payload.values():
        assert "center" in game


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


def test_build_public_site_renders_cover_curve_for_a_gaussian_active_model(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    """``_write_gaussian_board_fixture`` writes no ``line_sweep.parquet``, so
    every game's chart here comes ENTIRELY from the Gaussian fallback
    synthesis (``_COVER_CURVE_FALLBACK_OFFSETS`` -- see ``_game_deep_dive``),
    a real end-to-end exercise of that path through an actual refit."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_gaussian_board_fixture(artifacts_root, data_root, model_frame)

    pages = build_public_site(
        artifacts_root,
        data_root=data_root,
        require_fresh_arrest_overlay=False,
    )

    assert pages[PICKS_PAGE].count('class="ats-cover"') == len(card)
    match = re.search(r'id="ats-cover-data">(.*?)</script>', pages[PICKS_PAGE])
    assert match is not None
    payload = json.loads(match.group(1))
    assert set(payload) == set(card["game_id"].astype(str))
    assert all("center" in game for game in payload.values())


def test_build_public_site_without_gaussian_probability_method_has_no_gaussian_payload(
    tmp_path: Path,
) -> None:
    """``_write_board_fixture`` never sets ``probability_method`` (defaults to
    ``"ecdf"``) -- an older/rolled-back active model has no closed-form
    mean/sd the drag handler's erf formula can read. It DOES write a real
    ``line_sweep.parquet``, though, so the chart still renders from that (the
    merged component's preferred source) -- only the payload's Gaussian
    fields are missing, the same graceful-degradation contract every other
    optional artifact here follows, now scoped to the formula rather than to
    the whole chart."""

    _write_board_fixture(tmp_path)
    pages = build_public_site(tmp_path, require_fresh_arrest_overlay=False)
    page = pages[PICKS_PAGE]
    assert page.count('class="ats-cover"') == 2
    match = re.search(r'id="ats-cover-data">(.*?)</script>', page)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload
    assert all("center" not in game for game in payload.values())


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


# ---------------------------------------------------------------------------
# 2026-08-23 redesign: "Ledger base + Terminal layout" chrome replaces the
# retired Observatory theme pack (its injection, chalk defs and toggle tests
# are gone; the site_theme package itself is orphaned, not deleted).
# ---------------------------------------------------------------------------


def test_light_palette_hex_budget() -> None:
    """The light CSS block may declare at most 10 distinct hex colors."""

    from nfl_ats.public_board import _PAGE_CHROME

    light = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[0]
    hexes = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", light))
    # Still 10. The 2026-08-25 semantic-colour pass added a confidence ramp, a
    # diverging delta pair and a status-pill palette WITHOUT spending new
    # chrome: every one of them is derived with var()/color-mix() from the
    # tokens already here, so meaning-bearing colour costs zero budget.
    assert len(hexes) <= 10
    assert "#2a78d6" in hexes
    assert "#1a7f37" in hexes
    # Re-stepped 2026-08-25. The previous pair (#c0392b / #b35900) measured
    # dE 7.5 in NORMAL vision and 2.7 under deuteranopia against each other --
    # two "distinguishable" status roles that were effectively one colour.
    # These values pass every check of the dataviz validator; do not revert
    # them without re-running it.
    assert {"#9b2418", "#d59200"} <= hexes
    assert "--critical: #9b2418;" in light
    assert "--serious: #d59200;" in light
    # Derived, not literal -- this is what keeps the budget at 10.
    assert "--warning: var(--serious);" in light
    assert "--pos: var(--good);" in light
    assert "--neg: var(--critical);" in light
    # The confidence ramp moved from a one-hue color-mix to the validated
    # status tokens (2026-08-25, owner: every bar looked the same). Still
    # derived, so the budget is still untouched.
    assert "--band-1: var(--critical);" in light


def test_sticky_header_separates_with_a_hairline_not_a_shadow() -> None:
    from nfl_ats.public_board import _PAGE_CHROME

    board_rules = _PAGE_CHROME[_PAGE_CHROME.index(".ats table.week-board th") :]
    rules = board_rules[: board_rules.index("}")]
    assert "box-shadow" not in rules
    assert "border-bottom: 1px solid var(--baseline);" in rules


def test_dark_block_exists_via_media_query() -> None:
    from nfl_ats.public_board import _PAGE_CHROME

    assert "@media (prefers-color-scheme: dark)" in _PAGE_CHROME
    dark = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[1]
    for token in ("#0b0c0e", "#141518", "#f7f8f8", "#b4b8bf", "#7d828b", "#23252b", "#6ea8dc"):
        assert token in dark


def test_dark_mode_seq_ramp_and_semantic_tokens_are_lightened() -> None:
    """B9: the dark sequential ramp shifts lighter for contrast and the
    critical/serious roles split into distinguishable hues."""

    from nfl_ats.dashboard.theme import TOKENS_DARK

    assert TOKENS_DARK["seq-550"] == "#4d94e0"
    assert TOKENS_DARK["critical"] != TOKENS_DARK["serious"]
    from nfl_ats.public_board import _PAGE_CHROME

    dark = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[1]
    # Re-stepped 2026-08-25 and re-validated against the dark surface. The
    # previous steps failed FOUR checks -- three outside the lightness band,
    # --series-model below the chroma floor (reading gray), CVD separation
    # dE 5.4, and a normal-vision floor of 11.5 between critical and serious,
    # i.e. this test's own premise was false. Now worst adjacent normal-vision
    # dE is 15.2. Do not revert without re-running the validator.
    assert "--critical: #c9483c;" in dark
    assert "--serious: #b8891f;" in dark


def test_week_board_numeric_cells_are_tabular() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert '<td data-label="Line" class="num">' in page
    assert "font-variant-numeric: tabular-nums" in page


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    a, b = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_muted_text_tokens_meet_aa_normal_text_contrast() -> None:
    """WCAG 1.4.3 regression pin: --muted drives 11-12px text (.fine,
    .kicker, td::before data-labels), so it must clear the 4.5:1 normal-text
    bar on every background it renders against, light and dark."""

    from nfl_ats.dashboard.theme import TOKENS_DARK, TOKENS_LIGHT
    from nfl_ats.public_board import _PAGE_CHROME

    chrome_light = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[0]
    chrome_dark = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[1]
    muted_light = re.search(r"--muted:\s*(#[0-9a-f]{6})", chrome_light)
    muted_dark = re.search(r"--muted:\s*(#[0-9a-f]{6})", chrome_dark)
    assert muted_light is not None and muted_dark is not None

    # Page chrome surfaces/planes (light + dark).
    assert _contrast_ratio(muted_light.group(1)[1:], "ffffff") >= 4.5
    assert _contrast_ratio(muted_light.group(1)[1:], "fafaf8") >= 4.5
    assert _contrast_ratio(muted_dark.group(1)[1:], "0b0c0e") >= 4.5
    assert _contrast_ratio(muted_dark.group(1)[1:], "141518") >= 4.5

    # The shared theme stylesheet's own muted token (same defect class).
    assert _contrast_ratio(TOKENS_LIGHT["muted"][1:], "fcfcfb") >= 4.5
    assert _contrast_ratio(TOKENS_LIGHT["muted"][1:], "f9f9f7") >= 4.5
    assert _contrast_ratio(TOKENS_DARK["muted"][1:], "1a1a19") >= 4.5
    assert _contrast_ratio(TOKENS_DARK["muted"][1:], "0d0d0d") >= 4.5


def test_page_shell_ships_landmarks_skip_link_and_visible_focus() -> None:
    """WCAG 1.3.1 / 2.4.1 / 2.4.7 regression pins on the shared shell that
    every public page inherits: one <main> landmark, a keyboard-reachable
    skip link targeting it, a <footer> landmark, and a :focus-visible
    outline rule."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert '<a class="skip-link" href="#main-content">Skip to content</a>' in page
    assert page.count('<main id="main-content">') == 1
    assert "<footer" in page
    assert ".ats .skip-link:focus { left: 0; }" in page
    assert ":focus-visible" in page
    assert "outline: 2px solid var(--series-model)" in page


def test_informative_glyph_flags_carry_aria_labels_not_title_only() -> None:
    """WCAG 1.1.1 regression pin: the Best Pick star and the overlay flip
    arrows are informative non-text marks; title alone is not reliably
    announced, so each ships role=img + aria-label alongside the title."""

    from nfl_ats.public_board import _week_board

    board = _week_board(
        _predictions_fixture(),
        flipped_by_game={"2026_01_SF_LA": True},
        best_pick_id="2026_01_SF_LA",
        why_by_game={},
    )
    assert 'class="best-flag" role="img" aria-label="Best Pick of the week"' in board
    assert 'aria-label="Flipped by a production overlay' in board


def test_every_page_opens_at_h1_and_sections_nest_below_it() -> None:
    """WCAG 1.3.1 regression pin: page_header emits the single <h1> and
    section headers are <h2>, so the document outline has a top level."""

    from nfl_ats.dashboard import viz

    header = viz.page_header("Track record", "How often the picks landed", "Two lines.")
    assert '<h1 class="title page-title">How often the picks landed</h1>' in header
    assert "<h2" not in header and "</h2>" not in header

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.count('<h1 class="title page-title">') == 1
    from nfl_ats.public_board import _section_header

    assert '<h2 class="title" style="margin-bottom:6px;">' in _section_header(
        "Game notes", "One block per game", "Anchored from the board above."
    )


def test_index_renders_the_four_panel_terminal_grid() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert 'class="ledger-grid"' in page
    for panel in ("panel-summary", "panel-board", "panel-ledger", "panel-watch"):
        assert f'class="panel {panel}"' in page
    # The board sits inside the grid; deep-dive blocks come after it.
    assert page.index('class="ledger-grid"') < page.index('class="deep-game"')


def test_week_board_rows_expand_into_subrows() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    # The Best-Pick row carries an extra class so it can be emphasised, so
    # count the class rather than one exact attribute string.
    assert page.count('class="board-game') == 2
    assert page.count('<tr class="board-sub"><td colspan="5">') == 2


def test_no_observatory_references_remain_in_generated_pages(tmp_path: Path) -> None:
    _write_board_fixture(tmp_path)
    pages = build_public_site(tmp_path, require_fresh_arrest_overlay=False)
    for name, page in pages.items():
        assert "theme-obs" not in page, name
        assert "site_theme/" not in page, name
        assert "chalk-filter" not in page, name
        assert "theme-toggle-mount" not in page, name


# ---------------------------------------------------------------------------
# Model Ledger section (fail-open)
# ---------------------------------------------------------------------------


def _write_ledger_tree(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True)
    (registry_dir / "weak_signals.json").write_text(
        json.dumps(
            {
                "signals": {
                    "mod08_smooth_cdf_mapping": {
                        "classification": "unresolved_below_power",
                        "effect": 0.684,
                        "probability_positive": 0.8666,
                        "interval": [-0.444, 1.841],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prospective = tmp_path / "prospective"
    prospective.mkdir(parents=True)
    (prospective / "challengers.json").write_text(
        json.dumps(
            {
                "challengers": [
                    {
                        "challenger_id": "smooth_cdf_mapping",
                        "status": "SUPERSEDED_BY_PROMOTION",
                        "evidence": {
                            "registry_source": [
                                "registry/weak_signals.json:mod08_smooth_cdf_mapping"
                            ],
                            "probability_positive": 0.5536,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "active_ats_model.json").write_text(
        json.dumps(
            {
                "model_id": "abc123def4567890",
                "feature_profile": "weak_stack",
                "method": "market_residual",
                "historical_evaluation": {
                    "accuracy": 0.5209638554216868,
                    "games": 2075,
                    "artifact": "margins/20260820T004951Z",
                    "intervals": {
                        "season": {"lower": 0.5078765661351946, "upper": 0.5345932252330292}
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_model_ledger_section_embeds_promoted_badge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ledger_tree(tmp_path)
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    fragment = load_model_ledger_html(tmp_path)
    assert '<span class="badge-glyph">\u2713</span>PROMOTED</span>' in fragment

    models_page = render_models_page(fragment)
    assert "Every arm the card could come from" in models_page
    assert_public_safe(models_page)

    # De-clutter revision: the picks page itself stays clean -- the ledger
    # lives on its own page only.
    picks_page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "Every arm the card could come from" not in picks_page
    assert_public_safe(picks_page)


def test_models_page_failopen_note_without_ledger() -> None:
    page = render_models_page(None)
    assert "Ledger unavailable right now" in page
    assert_public_safe(page)


def test_model_ledger_section_omits_quietly_without_a_challenger_file(
    tmp_path: Path,
) -> None:
    assert load_model_ledger_html(tmp_path) == ""


def test_model_ledger_failopen_warning_box_on_registry_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prospective = tmp_path / "prospective"
    prospective.mkdir(parents=True)
    (prospective / "challengers.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))

    fragment = load_model_ledger_html(tmp_path)
    assert "MODEL LEDGER UNAVAILABLE" in fragment

    models_page = render_models_page(fragment)
    assert "MODEL LEDGER UNAVAILABLE" in models_page
    assert_public_safe(models_page)

    picks_page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "MODEL LEDGER UNAVAILABLE" not in picks_page


# ---------------------------------------------------------------------------
# Per-game "Why this pick" attribution (waterfall feed, fail-open)
# ---------------------------------------------------------------------------


def _feed_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "game_id": "2026_01_ARI_LAC",
        "market_line": 3.5,
        "edge_vs_spread": 0.4105409770192769,
        "key_number_distance": 0.0,
        "steps": [
            {
                "step_id": "market",
                "label": "Market-implied expectation",
                "delta_points": 0.0,
                "cumulative_points": 0.0,
            },
            {
                "step_id": "family:defense",
                "label": "defense contribution",
                "delta_points": -0.3421560120417648,
                "cumulative_points": -0.3421560120417648,
            },
        ],
        "flip_events": [{"overlay": "coach_fade", "would_flip_alone": True}],
        "rationale_sentences": [
            "documented early-season line biases moves this 0.5 points against SEA"
        ],
    }
    entry.update(overrides)
    return entry


def test_load_waterfall_feed_reads_latest_pointer(tmp_path: Path) -> None:
    run = tmp_path / "waterfall_feed" / "r1"
    run.mkdir(parents=True)
    (run / "feed.json").write_text(
        json.dumps({"schema_version": 1, "games": [_feed_entry()]}), encoding="utf-8"
    )
    (tmp_path / "waterfall_feed" / "latest.json").write_text(
        json.dumps({"latest": "r1"}), encoding="utf-8"
    )
    feed = load_waterfall_feed(tmp_path)
    assert set(feed) == {"2026_01_ARI_LAC"}


def test_load_waterfall_feed_missing_pointer_is_empty(tmp_path: Path) -> None:
    assert load_waterfall_feed(tmp_path) == {}


def test_load_waterfall_feed_dangling_run_or_bad_games_are_empty(tmp_path: Path) -> None:
    feed_dir = tmp_path / "waterfall_feed"
    feed_dir.mkdir()
    (feed_dir / "latest.json").write_text(json.dumps({"latest": "missing"}), encoding="utf-8")
    assert load_waterfall_feed(tmp_path) == {}
    (feed_dir / "latest.json").write_text(json.dumps({"games": "not a list"}), encoding="utf-8")
    assert load_waterfall_feed(tmp_path) == {}


def test_week_board_details_panel_carries_feed_numbers() -> None:
    page = render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        waterfall_feed={"2026_01_ARI_LAC": _feed_entry()},
    )
    assert '<details class="why-pick"><summary>Why this pick</summary>' in page
    assert "documented early-season line biases moves this 0.5 points against SEA" in page
    assert "Market-implied expectation" in page
    assert "+0.41" in page or "0.41 pts" in page  # edge-vs-market, from edge_vs_spread
    assert "model-vs-market edge 0.41 pts" in page
    assert "0.00 pts from the nearest key number" in page
    assert "coach_fade: flips this pick on its own" in page
    # The other game has no feed entry: quiet note, never an exception.
    assert page.count("Attribution not published.") == 1


def test_week_board_empty_steps_render_quiet_note() -> None:
    page = render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        waterfall_feed={"2026_01_ARI_LAC": {"steps": []}},
    )
    assert page.count("Attribution not published.") == 2
    assert '<details class="why-pick">' not in page


# ---------------------------------------------------------------------------
# Margin-interval text row (theme-neutral info parity)
# ---------------------------------------------------------------------------


def test_game_card_margin_interval_row_renders_card_quantiles() -> None:
    predictions = _predictions_fixture()
    predictions["margin_lower_50"] = [-5.6, -2.0]
    predictions["margin_upper_50"] = [10.3, 6.0]
    predictions["margin_lower_80"] = [-13.9, -8.0]
    predictions["margin_upper_80"] = [18.8, 12.0]
    page = render_picks_page(predictions, _sweep_fixture())
    assert "Projected margin intervals:" not in page
    assert "cover margin: 50% CI [-5.6, +10.3] &middot; 80% CI [-13.9, +18.8]" in page
    assert "cover margin: 50% CI [-2.0, +6.0]" in page


def test_game_card_without_margin_quantiles_renders_no_interval_row() -> None:
    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "Projected margin intervals" not in page
    assert "cover margin:" not in page


# ---------------------------------------------------------------------------
# 2026-08-23 cold-read QA fixes
# ---------------------------------------------------------------------------


def test_week_board_carries_the_best_pick_and_flip_legend() -> None:
    """B7: the legend under the board explains the star/flip glyphs and the
    strength ordering."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "best pick" in page
    assert "flipped by an overlay rule" in page
    assert "slight &lt; lean &lt; strong, by model-vs-market gap" in page


def test_sweep_table_formats_are_one_decimal_and_zero_never_signed() -> None:
    """B13: the sweep table-view twin uses one decimal everywhere and never
    renders zero as '+0'."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert "<td>0.0</td>" in page
    assert "<td>+0.0</td>" not in page
    assert "<td>-0.0</td>" not in page
    assert "<td>-0.5</td>" in page
    assert "<td>+0.5</td>" in page
    assert ">45.0%</span></td>" in page


def test_why_pick_step_labels_render_sentence_case() -> None:
    """B14: feed step labels are sentence-cased at render time."""

    page = render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        waterfall_feed={"2026_01_ARI_LAC": _feed_entry()},
    )
    assert "Defense contribution" in page
    assert "defense contribution" not in page


def test_disclaimer_appears_once_short_top_full_footer_only() -> None:
    """B11: one short bold line at the top; the full paragraph lives only in
    the footer -- no duplicated bold footer line."""

    for render in (
        lambda: render_picks_page(_predictions_fixture(), _sweep_fixture()),
        lambda: render_findings_page(),
        lambda: render_track_record_page(),
        lambda: render_models_page(None),
    ):
        page = render()
        assert page.count(DISCLAIMER_SHORT) == 1
        assert page.count(DISCLAIMER_FULL) == 1


def test_disclaimers_carry_no_percentage_figures() -> None:
    """2026-08-23 consolidation law (owner directive): legal chrome never
    competes with the page's own numbers -- the disclaimers must not contain
    any percentage figure at all."""

    assert "%" not in DISCLAIMER_SHORT
    assert "%" not in DISCLAIMER_FULL


def test_deep_dive_overlay_notes_are_plain_english_without_doc_refs() -> None:
    """B16: the production-policy note reads as plain English (no 'four-member
    production policy applied: triggered by ...' jargon) and no dangling
    docs/*.md sentence fragments ship on the deep-dive cards."""

    from nfl_ats.public_board import _game_deep_dive

    row = pd.Series(
        {
            "game_id": "2026_01_AAA_BBB",
            "home_team": "BBB",
            "away_team": "AAA",
            "spread_line": -3.5,
            "predicted_market_residual": -2.0,
            "fair_spread": -1.5,
            "home_cover_probability": 0.6,
        }
    )
    block, _chart_payload = _game_deep_dive(
        row, pd.DataFrame(), "", production_members=("coach_fade",)
    )
    assert (
        "One of four production rules applied: this game flipped by the year-one-coach fade."
        in block
    )
    assert "docs/" not in block
    assert "docs/overlay_subset_composition.md." not in block


# ---------------------------------------------------------------------------
# B1/B2: findings.html must not leak internal audit prose, and the dek must
# not claim "No jargon".
# ---------------------------------------------------------------------------


def test_findings_page_does_not_leak_internal_audit_prose() -> None:
    if not Path(_default_weak_signals_registry_path()).is_file():
        pytest.skip("live weak-signals registry absent")

    page = render_findings_page()
    assert "NOT deleted per AGENTS" not in page
    assert "LABEL-CORRECTED reconciliation" not in page
    assert "scratchpad/" not in page
    assert "details in the research registry" in page
    assert_public_safe(page)


def test_findings_page_dek_does_not_claim_no_jargon() -> None:
    """B2: the dek states what the page actually does; the false 'No jargon'
    claim is gone."""

    page = render_findings_page()
    assert "No jargon" not in page
    assert "Every finding states its evidence and how confident we are" in page


# ---------------------------------------------------------------------------
# 2026-08-23 de-firehose revision (owner's rendered-page review): ONE crowned
# number on the picks page; every other percentage collapsed or subordinate.
# ---------------------------------------------------------------------------


_INNERMOST_DETAILS = re.compile(
    r"<details\b[^>]*>(?:(?!<details\b).)*?</details>", re.DOTALL | re.IGNORECASE
)


def _html_without_collapsed_content(page: str) -> str:
    """The page as a reader sees it by DEFAULT: style/script blocks and every
    (possibly nested) ``<details>`` subtree removed."""

    text = _HEAD_BLOCK.sub(" ", page)
    previous = None
    while previous != text:  # peel innermost details first, repeat until stable
        previous = text
        text = _INNERMOST_DETAILS.sub(" ", text)
    return text


def _index_default_view(page: str) -> str:
    """The default view as plain reader-visible text: collapsed subtrees and
    scripts/styles removed, tags stripped, entities resolved -- safe to scan
    for banned numerals without CSS geometry or entity spellings masking a
    match."""

    return unescape(_TAG.sub(" ", _html_without_collapsed_content(page)))


#: The consolidation law's banned numerals: every accuracy figure that used
#: to leak into the index default view (panel fine print, ladder prose,
#: footer byline). None may appear outside the ONE collapsed ladder.
_BANNED_DEFAULT_VIEW_NUMERALS = ("53.4", "52.1", "53.76", "53.36", "55.4", "0.00", "0.49")

#: Measured on the fixture page (2026-08-23): 6 '%' default-visible -- two
#: disclaimers (fixed legal chrome), the ≈55% hero, the 54.2% measured line,
#: and one per-game cover chance per fixture game. Pinned at actual+5.
_VISIBLE_PCT_BUDGET = 11


def _consolidated_picks_page() -> str:
    """The picks page in its real consolidated shape: chain history present."""

    return render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        spread_explorer=_spread_explorer_params_fixture(),
        played_chain_accuracy=0.541583499667332,
    )


def test_index_default_view_carries_exactly_two_accuracy_stats() -> None:
    """THE OWNER'S CONSOLIDATION LAW, pinned: the index page's DEFAULT view
    carries exactly TWO accuracy statistics -- the ≈55% planning hero and
    the 54.2% measured chain history -- plus the per-game cover chances.
    Every other accuracy percentage lives inside ONE collapsed ``<details>``;
    none of the old firehose numerals may appear default-visible."""

    page = _consolidated_picks_page()
    default_view = _index_default_view(page)
    for banned in _BANNED_DEFAULT_VIEW_NUMERALS:
        assert banned not in default_view, f"{banned!r} leaked into the default view"
    # The two allowed stats are present and labeled as what they are.
    assert f">{PLAYED_CARD_EXPECTATION_HERO}</div>" in page
    assert "Planning estimate" in default_view
    assert "Measured chain history:" in default_view
    assert "54.2%" in default_view
    # The picks themselves stay: per-game cover chances are not hidden.
    assert 'covers <span class="num">62%</span>' in page
    # ...and everything else lives on the story page, linked from Panel 1.
    assert "What this number means" in page
    assert "<summary>Where these numbers come from" not in page


def test_index_visible_percentage_budget_stays_tight() -> None:
    """Default-visible '%' occurrences on index.html: measured at 6 on this
    fixture (two disclaimers, hero, measured line, one cover chance per
    game); pinned at actual+5 so a future regression cannot quietly rebuild
    the percentage firehose outside collapsed blocks."""

    visible = _index_default_view(_consolidated_picks_page())
    count = visible.count("%")
    assert count <= _VISIBLE_PCT_BUDGET, f"visible-percentage budget blown: {count} '%'"
    assert count >= 4, f"default view lost its required stats: only {count} '%'"


# ---------------------------------------------------------------------------
# Canonical-figure HOME guard (owner law, 2026-08-23): each canonical stat
# renders as a figure ONLY on its home page -- every repeat references it
# verbally. Source half: tests/test_number_variables.py. Rendered half: here.
# ---------------------------------------------------------------------------


def _track_record_home_page() -> str:
    """The track-record page in its full fixture shape (both grading rules,
    season table, active model) -- the home of the opener/close grades and
    the arrest evaluation figures."""

    return render_track_record_page(
        _opener_metadata_with_both_rules_fixture(),
        _season_summary_fixture(),
        _active_fixture(),
    )


def _models_home_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The models page with a REAL populated ledger (its production shape)."""

    _write_ledger_tree(tmp_path)
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    return render_models_page(load_model_ledger_html(tmp_path))


def test_arrest_evaluation_figures_render_only_on_track_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """53.76 / 53.36 are the arrest evaluation's figures; their ONE home is
    track_record.html's default view. Findings now names the evaluation
    verbally ("the arrest evaluation (track-record page)") and the picks and
    models pages carry no copy of it default-visible."""

    views = {
        "index": _index_default_view(_consolidated_picks_page()),
        "findings": _index_default_view(render_findings_page()),
        "track_record": _index_default_view(_track_record_home_page()),
        "models": _index_default_view(_models_home_page(monkeypatch, tmp_path)),
    }
    for token in ("53.76", "53.36"):
        assert token in views["track_record"], f"{token} vanished from its home page (track_record)"
        for page in ("index", "findings", "models"):
            assert token not in views[page], (
                f"canonical figure {token} rendered on {page}; repeats must "
                "reference the arrest evaluation verbally"
            )


def test_close_grade_figure_renders_only_on_track_record_and_ledger_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The active model's close grade (52.1) is homed on track_record.html.
    The models page is the designated SECOND home for PER-ROW ledger track
    records only -- so there the figure may appear inside the ledger table
    and nowhere else. Findings and index carry no copy default-visible (the
    findings close tile was removed with the 2026-08-23 home law; the study
    backtests that happen to round to the same digits live behind 'How we
    know' toggles)."""

    assert "52.1" in _index_default_view(_track_record_home_page())
    for page, view in (
        ("index", _index_default_view(_consolidated_picks_page())),
        ("findings", _index_default_view(render_findings_page())),
    ):
        assert "52.1" not in view, (
            f"canonical close grade re-rendered on {page}; repeats must reference it verbally"
        )
    models_page = _models_home_page(monkeypatch, tmp_path)
    table_start = models_page.index("<table")
    table_end = models_page.index("</table>") + len("</table>")
    outside_table = _index_default_view(models_page[:table_start] + models_page[table_end:])
    assert "52.1" not in outside_table


def test_index_has_exactly_one_24px_number_the_crowned_stat() -> None:
    """Exactly ONE inline 24px font size on the whole picks page -- Panel 1's
    crowned stat. Page titles carry their 24px via the shared ``page-title``
    class instead of inline styles, so this stays true.

    2026-08-23 revision: the hero is the played card's HONEST EXPECTATION --
    the pinned planning constant, never a measured figure -- and the measured
    chain history is the secondary line beneath it."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert page.count("font-size:24px") == 1
    summary_index = page.index('class="panel panel-summary"')
    crowned_index = page.index("font-size:24px")
    board_index = page.index('class="panel panel-board"')
    assert summary_index < crowned_index < board_index
    # The crowned stat is the PLANNING expectation, labeled as one.
    assert "PLAYED CARD \u2014 HONEST EXPECTATION VS TUESDAY LINES" in page
    assert f">{PLAYED_CARD_EXPECTATION_HERO}</div>" in page
    assert "Planning estimate for the played card." in page
    assert '<a href="track_record.html">What this number means &#8594;</a>' in page
    # The page header title itself is class-sized now, visually unchanged;
    # it is an <h1> since the a11y pass (one h1 per page, WCAG 1.3.1).
    assert 'class="title page-title">This week&#x27;s picks</h1>' in page


def test_crowned_stat_keeps_the_constant_hero_and_shows_the_measured_chain_history() -> None:
    """With the played-chain artifact reachable, the hero STAYS the pinned
    expectation constant (it must never be recomputed from an artifact) and
    the chain accuracy appears as the strong secondary line. Consolidation
    law: NOTHING else renders in Panel 1 -- the old fine print (sequential
    chain, raw baseline, selection caveat) moved into the collapsed ladder."""

    page = render_picks_page(
        _predictions_fixture(),
        _sweep_fixture(),
        played_chain_accuracy=0.541583499667332,
    )
    assert f">{PLAYED_CARD_EXPECTATION_HERO}</div>" in page
    assert '<strong>Measured chain history: <span class="num">54.2%</span></strong>' in page
    # The fine-print lines are gone from the block entirely, and the page
    # links to the story page instead of carrying the ladder itself.
    assert "Sequential chain:" not in page
    assert "already discounted here" not in page  # old caveat wording
    assert "What this number means" in page
    # The raw baseline appears ONLY as the no-artifact fallback label.
    default_view = _index_default_view(page)
    assert "raw model before policy overlays" not in default_view.lower()
    assert "Raw model before policy overlays" not in page


def test_crowned_stat_falls_back_to_the_exact_raw_chain_baseline_label() -> None:
    """Without a chain artifact the MEASURED line degrades to the raw-model
    opener baseline labeled EXACTLY "Raw chain baseline" -- never a silently
    mislabeled chain figure -- and Panel 1 still carries nothing else."""

    page = render_picks_page(_predictions_fixture(), _sweep_fixture())
    assert (
        "<strong>Raw chain baseline: "
        f'<span class="num">{HEADLINE.opener}</span></strong></p>' in page
    )
    default_view = _index_default_view(page)
    assert "Raw chain baseline:" in default_view
    assert "Measured chain history:" not in page
    assert "Sequential chain:" not in page


def test_index_links_to_the_story_page_instead_of_carrying_the_ladder() -> None:
    """2026-08-24 re-architecture: the picks page carries NO number ladder at
    all -- Panel 1 links to the story page, and the selection caveat lives
    there (collapsed), never on index."""

    for page in (
        render_picks_page(_predictions_fixture(), _sweep_fixture()),
        render_picks_page(
            _predictions_fixture(),
            _sweep_fixture(),
            played_chain_accuracy=0.541583499667332,
        ),
    ):
        assert "0.00 pts (P+ 0.49)" not in page
        assert "<summary>Where these numbers come from" not in page
        assert '<a href="track_record.html">What this number means' in page


def test_story_page_carries_the_selection_caveat_collapsed() -> None:
    """The union's selection-inflation numbers and its out-of-sample re-check
    render ONLY inside the story page's collapsed ``<details>`` under the
    planning section -- present on the page, never default-visible."""

    page = render_track_record_page(played_chain_accuracy=0.541583499667332)
    assert "0.00 pts (P+ 0.49)" in page  # inside the collapsed details
    assert "The selection discount, in numbers" in page
    assert "0.00 pts (P+ 0.49)" not in _index_default_view(page)


def test_story_page_ladder_is_one_collapsed_details_with_the_exact_rungs() -> None:
    """2026-08-24 re-architecture: the ladder lives on the STORY page, one
    collapsed ``<details>`` under the planning section; the rung set is
    exactly :func:`ladder_rungs`'s output; and the old promoted-arrest
    paragraph stays out of it (the appendix owns that comparison)."""

    from nfl_ats.dashboard.findings_content import ladder_rungs as pinned_rungs
    from nfl_ats.public_board import _story_sections

    section = _story_sections(0.541583499667332)
    details_start = section.index('<details class="ceiling-ladder"')
    summary_close = section.index("</summary>", details_start)
    assert "The selection discount, in numbers" in section[details_start:summary_close]
    rungs = pinned_rungs(0.541583499667332)
    for rung in rungs:
        assert f"<p>{rung}</p>" in section
    # The details block contains exactly the pinned rungs -- no extras.
    details_end = section.index("</details>", details_start)
    block = section[summary_close:details_end]
    assert block.count("<p>") == len(rungs)
    assert "Promoted player-arrest policy evaluation" not in block
    assert "53.76%" not in block

    without_chain = _story_sections(None)
    for rung in pinned_rungs(None):
        assert f"<p>{rung}</p>" in without_chain


def test_ledger_mini_column_header_reads_evidence_p_plus() -> None:
    challengers = [
        {
            "challenger_id": "hc_year_one_fade_overlay",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {"probability_positive": 0.932},
        }
    ]
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), challengers=challengers)
    # The header keeps its label and now carries the plain-language tooltip.
    assert ">Evidence P+</abbr></th>" in page
    assert "<th>Best P+</th>" not in page


def test_challenger_watch_renders_human_names_in_plain_ink() -> None:
    """Raw registry ids never reach the watch panel; names are plain ink (no
    colored/green accent links -- accent discipline)."""

    challengers = [
        {"challenger_id": "movement_rule_composed_v1", "status": "ACTIVE_PROSPECTIVE"},
        {
            "challenger_id": "nflcom_friday_refresh_out2_starters_v1",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {"probability_positive": 0.61},
        },
        {"challenger_id": "surface_switch_tilt_overlay", "status": "ACTIVE_PROSPECTIVE"},
    ]
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), challengers=challengers)
    watch = page[page.index("Challenger watch") : page.index("Game notes")]
    assert "Follow line moves \u22651pt" in watch
    assert "Fade 2+ Out designations" in watch
    assert "Turf-surface switch" in watch
    for raw_id in ("movement_rule_composed_v1", "nflcom_friday_refresh_out2_starters_v1"):
        assert raw_id not in watch
    assert "<a href=" not in watch  # plain ink, not links/accent color


def test_challenger_watch_shows_top_six_and_collapses_the_rest() -> None:
    probabilities = [0.90, 0.55, 0.80, 0.60, 0.70, 0.65, 0.75, 0.50]
    challengers = [
        {
            "challenger_id": f"synthetic_{index}",
            "status": "ACTIVE_PROSPECTIVE",
            "evidence": {"probability_positive": probability},
        }
        for index, probability in enumerate(probabilities)
    ]
    page = render_picks_page(_predictions_fixture(), _sweep_fixture(), challengers=challengers)
    assert "<summary>show all</summary>" in page
    before_toggle, after_toggle = page.split("<summary>show all</summary>", 1)
    watch_before = before_toggle[before_toggle.index("Challenger watch") :]
    assert watch_before.count("<li>") == 6
    details_body = after_toggle.split("</details>", 1)[0]
    assert details_body.count("<li>") == 2
    # Strongest evidence leads: P+ 0.90 first, P+ 0.50 last among the visible.
    # The numeral now sits inside its own diverging-tone span (above/below the
    # 0.5 decision midpoint), so ordering is checked on the toned value.
    assert watch_before.index('<span class="delta pos">0.90</span>') < watch_before.index(
        '<span class="delta pos">0.70</span>'
    )


_REGISTERED_CHALLENGER_IDS: tuple[str, ...] = (
    "mod07_weak_signal_stack",
    "player_qb_continuity|ridge_alpha=1|calibration=none",
    "hc_year_one_fade_overlay",
    "best_pick_nomination_v2",
    "best_pick_nomination_v3",
    "best_pick_big_spread_eligibility",
    "injury_value_lost_tilt_overlay",
    "division_revenge_tilt_overlay",
    "backup_qb_fade_overlay",
    "surface_switch_tilt_overlay",
    "spread_gap_zone_fade_overlay",
    "smooth_cdf_mapping",
    "ecdf_mapping_incumbent",
    "era_weighted_half_life_8",
    "forecast_cold_visitor_tilt",
    "model_only_refresh_incumbent",
    "interim_hc_first_game_tilt_overlay",
    "forecast_weather_kn_warm_team_cold_late_tilt",
    "forecast_weather_kn_precip_high_total_tilt",
    "injury_signal_refresh_tilt",
    "player_arrests_recent_14d_back_side_overlay",
    "player_arrests_recent_14d_no_overlay_incumbent",
    "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
    "overlay_production_chain_coach_arrest_incumbent",
    "movement_rule_composed_v1",
    "nflcom_friday_refresh_out2_starters_v1",
)


def test_challenger_display_name_map_covers_every_registered_challenger() -> None:
    """Every challenger id registered in artifacts/prospective/challengers.json
    has a reader-facing display name; unknown future ids fall back to the
    humanized id rather than raising."""

    from nfl_ats.public_board import _CHALLENGER_DISPLAY_NAMES

    for challenger_id in _REGISTERED_CHALLENGER_IDS:
        assert challenger_id in _CHALLENGER_DISPLAY_NAMES, challenger_id
        assert _CHALLENGER_DISPLAY_NAMES[challenger_id].strip()
    from nfl_ats.public_board import _challenger_display_name

    assert _challenger_display_name("brand_new_future_challenger") == (
        "brand new future challenger"
    )


def test_load_played_chain_accuracy_reads_the_newest_run(tmp_path: Path) -> None:
    older = tmp_path / "overlay_subset_composition" / "20260101T000000Z"
    newer = tmp_path / "overlay_subset_composition" / "20260201T000000Z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "result.json").write_text(
        json.dumps({"production_chain_reference": {"coach_then_arrest_sequential": 0.40}}),
        encoding="utf-8",
    )
    (newer / "result.json").write_text(
        json.dumps(
            {
                "production_chain_reference": {
                    "coach_then_arrest_sequential": {"candidate_accuracy": 0.541583499667332}
                }
            }
        ),
        encoding="utf-8",
    )
    from nfl_ats.public_board import load_played_chain_accuracy

    assert load_played_chain_accuracy(tmp_path) == pytest.approx(0.541583499667332)

    # Fail-open: absent directory or unusable payload -> None, never a raise.
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    assert load_played_chain_accuracy(empty) is None
    broken = tmp_path / "overlay_subset_composition" / "20260301T000000Z"
    broken.mkdir(parents=True)
    (broken / "result.json").write_text("{not json", encoding="utf-8")


def test_build_public_site_threads_the_played_chain_figure_into_the_summary(
    tmp_path: Path,
) -> None:
    """End-to-end: the loader's figure reaches the picks page's crowned stat
    through ``build_public_site``, not via any hand-typed literal."""

    _write_board_fixture(tmp_path)
    pages = build_public_site(
        tmp_path,
        generated_at=datetime(2026, 8, 16, 20, 0, tzinfo=UTC),
        require_fresh_arrest_overlay=False,
    )
    picks = pages[PICKS_PAGE]
    assert "PLAYED CARD \u2014 HONEST EXPECTATION VS TUESDAY LINES" in picks
    # The fixture writes no overlay_subset_composition run, so the measured
    # line must degrade to the labeled raw-chain baseline, never invent a
    # chain figure.
    assert "Raw chain baseline:" in picks
    # Consolidation law end-to-end. This fixture deliberately has NO
    # overlay_subset_composition run, so Panel 1 shows its one DEGRADED stat
    # ("Raw chain baseline: 53.4%", the pinned fallback label); every other
    # banned numeral stays out of the default view regardless.
    default_view = _index_default_view(picks)
    for banned in _BANNED_DEFAULT_VIEW_NUMERALS:
        if banned == "53.4":
            continue
        assert banned not in default_view
    assert "Raw chain baseline:" in default_view
    assert "53.4%" in default_view  # the one allowed degraded stat
    assert "raw model before policy overlays" not in default_view.lower()
    assert "Raw model before policy overlays" not in picks
    assert_public_safe(picks)


# ---------------------------------------------------------------------------
# 2026-08-25 semantic-colour pass. Colour was almost entirely decorative: the
# status tokens were defined but `var(--good)` was used zero times, `--warning`
# was referenced twice and never defined at all, and the model ledger emitted
# 28 badge-* elements that no stylesheet ever styled. These pin the contract
# that pass established, not the specific hues (those are pinned by the
# palette-budget tests above, with the validator output recorded there).
# ---------------------------------------------------------------------------


def test_every_css_variable_used_is_actually_defined() -> None:
    """`--warning` was used twice and defined nowhere, so the MODEL LEDGER
    UNAVAILABLE card rendered colourless. Nothing may reference a token that
    does not exist."""

    from nfl_ats.public_board import _PAGE_CHROME

    used = set(re.findall(r"var\((--[a-z0-9-]+)", _PAGE_CHROME))
    defined = set(re.findall(r"^\s*(--[a-z0-9-]+):", _PAGE_CHROME, re.M))
    assert not used - defined


def test_signed_values_keep_their_sign_so_colour_is_never_the_only_channel() -> None:
    """Diverging encoding is legal only because the sign is always rendered."""

    from nfl_ats.public_board import _signed, delta_html

    assert "+" in _signed(1.5) and "pos" in _signed(1.5)
    assert "-" in _signed(-1.5) and "neg" in _signed(-1.5)
    assert "zero" in _signed(0.0)
    assert "+1.25" in delta_html(1.25) and "delta pos" in delta_html(1.25)
    assert "-1.25" in delta_html(-1.25) and "delta neg" in delta_html(-1.25)
    # An exact zero picks NO side -- that is what makes the midpoint neutral.
    assert "delta zero" in delta_html(0.0)
    assert "delta zero" in delta_html(None)


def test_probability_positive_diverges_around_the_decision_midpoint() -> None:
    """0.5, never 0.95: predeclared thresholds govern what the docs may claim,
    never which card is played (AGENTS.md)."""

    from nfl_ats.public_board import p_plus_html

    assert "delta pos" in p_plus_html(0.51, "0.51")
    assert "delta neg" in p_plus_html(0.49, "0.49")
    assert "delta zero" in p_plus_html(0.5, "0.50")
    # A value between 0.5 and 0.95 must still read as favouring the candidate.
    assert "delta pos" in p_plus_html(0.62, "0.62")


def test_season_accuracy_diverges_around_the_coin_flip() -> None:
    from nfl_ats.public_board import accuracy_vs_coin_flip_html

    assert "delta pos" in accuracy_vs_coin_flip_html(0.534)
    assert "delta neg" in accuracy_vs_coin_flip_html(0.488)
    assert "delta zero" in accuracy_vs_coin_flip_html(0.5)
    assert "delta zero" in accuracy_vs_coin_flip_html(None)


def test_confidence_meter_colours_the_three_bands_it_already_encoded() -> None:
    """One hue, three discrete steps keyed to the SAME bands the shape carries
    -- never a continuous gradient over the underlying probability."""

    from nfl_ats.public_board import confidence_meter

    assert "band-1" in confidence_meter("slight")
    assert "band-2" in confidence_meter("lean")
    assert "band-3" in confidence_meter("strong")
    # Still aria-hidden: the word beside it remains the accessible label.
    assert 'aria-hidden="true"' in confidence_meter("strong")


def test_status_pills_always_ship_their_label() -> None:
    """Status hue is reserved for state and never the only channel."""

    from nfl_ats.public_board import pill_html

    assert ">promoted<" in pill_html("good", "promoted")
    assert "is-good" in pill_html("good", "promoted")
    assert "is-idle" in pill_html("nonsense-tone", "tracked")


def test_ledger_badge_classes_are_styled_rather_than_merely_emitted() -> None:
    """model_ledger's own docstring claimed these "reuse the design-system
    classes"; they were emitted 28 times and styled zero times."""

    from nfl_ats.public_board import _PAGE_CHROME

    for selector in (".ats .badge-promoted", ".ats .badge-challenger", ".ats .badge-muted"):
        assert selector in _PAGE_CHROME


def test_forced_colors_keeps_status_distinguishable_without_hue() -> None:
    """Print and forced-colors strip the palette; shape must survive."""

    from nfl_ats.public_board import _PAGE_CHROME

    block = _PAGE_CHROME[_PAGE_CHROME.index("@media (forced-colors: active)") :]
    assert "border-radius: 0" in block


def test_metric_colour_means_good_not_positive() -> None:
    """Owner-caught defect, 2026-08-25: "Defense EPA/play allowed" is a metric
    where a NEGATIVE number is a GOOD defence, and the first semantic-colour
    pass tinted it red because it tinted by SIGN. The colour contradicted the
    help text beside it. Sign and merit are different axes."""

    from nfl_ats.public_board import _signed

    # Higher-is-better: sign and merit agree.
    assert "delta pos" in _signed(0.12, good_direction=1)
    assert "delta neg" in _signed(-0.12, good_direction=1)
    # Lower-is-better: a NEGATIVE value is GOOD and must read green.
    assert "delta pos" in _signed(-0.12, good_direction=-1)
    assert "delta neg" in _signed(0.12, good_direction=-1)
    # The sign character never changes -- only the hue does.
    assert "-0.12" in _signed(-0.12, good_direction=-1)
    assert "+0.12" in _signed(0.12, good_direction=-1)
    # Unknown direction must be NEUTRAL, never a guess: a wrong colour is worse
    # than no colour because it contradicts the text.
    assert "delta zero" in _signed(0.12, good_direction=0)


def test_every_trend_metric_declares_a_direction() -> None:
    """A new metric must not silently inherit a polarity that is backwards."""

    from nfl_ats.team_explorer import (
        METRIC_GOOD_DIRECTION,
        METRIC_LABELS,
        metric_good_direction,
    )

    assert set(METRIC_GOOD_DIRECTION) == set(METRIC_LABELS)
    assert all(value in (-1, 1) for value in METRIC_GOOD_DIRECTION.values())
    # The tempting shortcut "offence positive, defence negative" is wrong twice.
    assert metric_good_direction("off_turnover_rate") == -1
    assert metric_good_direction("off_sack_rate") == -1
    assert metric_good_direction("def_takeaway_rate") == 1
    assert metric_good_direction("def_sack_rate") == 1
    assert metric_good_direction("def_epa_per_play") == -1
    assert metric_good_direction("off_epa_per_play") == 1


def test_confidence_bands_are_an_ordinal_ramp_not_one_hue() -> None:
    """Owner, 2026-08-25: every bar was the same colour, so a strong pick was
    no easier to spot than a weak one. Weak/middling/strong must be three
    DISTINCT hues, each validated as mutually distinguishable."""

    from nfl_ats.public_board import _PAGE_CHROME

    light = _PAGE_CHROME.split("@media (prefers-color-scheme: dark)", 1)[0]
    assert "--band-1: var(--critical);" in light
    assert "--band-2: var(--serious);" in light
    assert "--band-3: var(--good);" in light
    # Derived, so the ordinal ramp still costs zero chrome budget.
    assert "--band-1: #" not in light


def test_the_pick_is_emphasised_above_the_fields_beside_it() -> None:
    from nfl_ats.public_board import _PAGE_CHROME

    assert ".ats .pick-team" in _PAGE_CHROME
    assert "tr.is-best-pick" in _PAGE_CHROME
    # The type treatment (size/weight/letter-spacing/colour) carries the
    # emphasis on its own -- a left accent rule shipped alongside it read
    # back as a stray blue vertical line beside every pick (owner,
    # 2026-08-26) and was removed. Regression guard: no border-left on the
    # Pick column, in either the base or best-pick variant.
    assert 'td[data-label="Pick"]' not in _PAGE_CHROME
    assert "border-left" not in _PAGE_CHROME
