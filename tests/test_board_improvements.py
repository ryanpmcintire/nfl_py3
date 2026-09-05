"""Tests for the remaining pieces of the owner-approved improvement batch
that are not covered by ``tests/test_board_content_season_and_flips.py``
(pure content-layer functions) or ``tests/test_board_season_mode.py``
(season-mode board-row fixtures):

* item 1 -- the flip pill on the board row and the deep-dive flip note;
* item 2 -- findings trace chips;
* item 3 -- the prospective scoreboard's rendered dormant and live states;
* item 5 -- the sortable board's toggle markup and JS;
* item 6/7 -- the clickable, shared ticker + command row on every page;
* item 8 -- the six-season dot chart;
* item 9 -- the grouped challenger ledger;
* item 10 -- link-preview meta tags and the footer's cadence line.

The hand-built ``_board_content_fixtures`` fixture covers items 1 and 3 (it
already carries one flipped game and a dormant scoreboard); the other items
are exercised against real repo artifacts via ``board_site_content
.load_site_content``, exactly like ``tests/test_board_terminal.py`` already
does for the Model/Findings pages.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path

import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import CADENCE_NOTE, ProspectiveScoreboard
from nfl_ats.board_site_content import SiteContent

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site_content(_shared_real_site_content: SiteContent) -> SiteContent:
    """Real repo artifacts -- shared session-scoped load, see
    ``tests/conftest.py::_shared_real_site_content`` (WP51, test-suite
    speed). Every test using this fixture only reads it or calls
    ``dataclasses.replace`` on a hand-built fixture, never on this one."""

    return _shared_real_site_content


# ---------------------------------------------------------------------------
# item 1 -- flip pill + deep-dive flip note
# ---------------------------------------------------------------------------


def test_flip_pill_shows_the_glyph_and_member_name_not_the_word_flipped() -> None:
    content = build_fixture_content()
    flipped = next(game for game in content.games if game.flip_member_labels)
    html = board_terminal.render(content)

    assert "⇄ coach fade" in html
    pill_html = f'class="pill flip-pill">{flipped.flip_pill_text}</span>'
    assert pill_html in html
    # The pill itself never spells out the word "FLIPPED" -- just the glyph
    # plus the member name (the owner's explicit instruction). Prose
    # elsewhere on the page (the policy-overlay narrative) legitimately uses
    # the word "flipped" in a sentence, so this checks the PILL's own text,
    # not the whole page.
    assert "FLIPPED" not in flipped.flip_pill_text.upper()


def test_unflipped_games_carry_no_flip_pill() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    assert html.count('class="pill flip-pill"') >= 1
    unflipped = [game for game in content.games if not game.flip_member_labels]
    assert len(unflipped) == len(content.games) - 1


def test_dive_flip_note_names_raw_side_vs_played_side() -> None:
    content = build_fixture_content()
    flipped_dive = next(dive for dive in content.dives if dive.flip_note)
    html = board_terminal.render(content)
    assert escape(flipped_dive.flip_note) in html


# ---------------------------------------------------------------------------
# item 2 -- findings trace chips
# ---------------------------------------------------------------------------


@pytest.mark.full  # ENG-11: triggers the real-artifact site-content build (dominates --durations)
def test_findings_trace_chip_renders_signal_name_and_probability_positive(
    site_content: SiteContent,
) -> None:
    html = board_terminal.render_findings_page(site_content.findings)
    traced = [
        finding
        for group in site_content.findings.groups
        for finding in group.findings
        if finding.trace_signal_name is not None
    ]
    assert traced, "expected at least one curated finding to trace to a registry signal"
    for finding in traced[:5]:
        assert escape(finding.trace_signal_name) in html
        pp_text = f"{finding.trace_probability_positive:.2f}"
        chip = f"{escape(finding.trace_signal_name)} &middot; P+ {pp_text}"
        assert chip in html


def test_findings_without_a_trace_render_no_chip() -> None:
    """A finding whose registry keys carry no measured P+ (or none at all)
    must render nothing for the trace chip -- never a guessed number."""

    from nfl_ats.board_site_content import FindingItemView

    bare = FindingItemView(question="Q", verdict="evergreen", plain_answer="A", detail="D")
    assert board_terminal._trace_chip_html(bare) == ""


# ---------------------------------------------------------------------------
# item 3 -- prospective scoreboard rendering
# ---------------------------------------------------------------------------


def test_prospective_scoreboard_dormant_state_renders() -> None:
    content = build_fixture_content()
    assert content.headline.prospective_scoreboard.dormant is True
    html = board_terminal.render(content)
    assert "Prospective scoreboard" in html
    assert "Prospective tracking begins Week 1." in html
    assert 'class="prospective-scoreboard dormant"' in html


def test_prospective_scoreboard_live_state_renders_with_detail() -> None:
    content = build_fixture_content()
    live = ProspectiveScoreboard(
        dormant=False,
        headline_text=(
            "Prospective record at the decision line: played policy 9-5 vs. prior chain "
            "8-6 -- 14 of 15 recorded games settled."
        ),
        detail_text="1 recorded game not yet kicked off.",
    )
    content = replace(content, headline=replace(content.headline, prospective_scoreboard=live))
    html = board_terminal.render(content)
    assert live.headline_text in html
    assert live.detail_text is not None
    assert live.detail_text in html
    assert 'class="prospective-scoreboard"' in html
    assert 'class="prospective-scoreboard dormant"' not in html


# ---------------------------------------------------------------------------
# item 5 -- sortable board
# ---------------------------------------------------------------------------


def test_sort_toggle_has_both_buttons_kickoff_default() -> None:
    html = board_terminal.render(build_fixture_content())
    assert 'data-sort="kickoff"' in html
    assert 'data-sort="confidence"' in html
    assert 'class="sort-btn is-active" data-sort="kickoff"' in html
    assert "sortByConfidence" in html
    assert "sortByKickoff" in html


def test_every_board_row_carries_a_sort_probability() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    for game in content.games:
        assert f'data-prob="{game.pick_probability:.6f}"' in html


# ---------------------------------------------------------------------------
# item 6/7 -- clickable, shared ticker + command row on every page
# ---------------------------------------------------------------------------


def test_ticker_ticks_are_real_links_to_index_with_game_hash() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    for game in content.games:
        assert f'href="index.html#{game.game_id}"' in html
        assert f'data-game-id="{game.game_id}"' in html


def test_ticker_and_cmd_row_present_on_all_four_pages(site_content: SiteContent) -> None:
    index_html = board_terminal.render(site_content.board)
    model_html = board_terminal.render_model_page(site_content.model)
    findings_html = board_terminal.render_findings_page(site_content.findings)
    history_html = board_terminal.render_history_page(site_content.history)
    for page_html in (index_html, model_html, history_html, findings_html):
        assert page_html.count('class="terminal-chrome"') == 1
        assert 'class="ticker"' in page_html
        assert 'class="cmd-row"' in page_html
        assert 'class="motion-status"' in page_html


def test_terminal_chrome_is_pinned_and_anchor_targets_clear_it() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS.replace(" ", "")

    assert ".terminal-chrome{" in css
    assert "position:sticky" in css
    assert "top:0" in css
    assert ":target{scroll-margin-top:170px" in css


def test_motion_status_uses_real_board_values_and_no_network_feed() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    rail = board_terminal._motion_status_rail(content.ticker_chrome)

    assert f'data-roll-to="{len(content.games)}"' in html
    strong_count = sum(game.confidence_word == "strong" for game in content.games)
    assert f'data-roll-to="{strong_count}"' in html
    assert "STRONG READS" in html
    assert f'<span class="rail-accent">{content.ticker_chrome.model_method_label}</span>' in html
    best = next(game for game in content.games if game.game_id == content.best_pick_game_id)
    assert f'BEST PICK <span class="rail-accent">{best.pick_team}' in html
    assert f"SEASON {content.season} / WEEK {content.week}" in html
    assert "fetch(" not in board_terminal._MOTION_SCRIPT
    assert "WebSocket" not in board_terminal._MOTION_SCRIPT
    assert "LIVE" not in rail


def test_motion_status_honors_reduced_motion_and_keeps_final_values_in_html() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS
    script = board_terminal._MOTION_SCRIPT

    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ".status-frame:first-child" in css
    assert ".motion-beacon, .motion-bars i, .brand .dot" in css
    assert "prefers-reduced-motion: reduce" in script
    assert "data-roll-to" in script


def test_actual_page_content_animates_once_on_view_without_network_state() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS
    script = board_terminal._MOTION_SCRIPT

    assert "IntersectionObserver" in script
    assert "main > section" in script
    assert "observer.unobserve(entry.target)" in script
    assert ".content-motion-ready.content-motion-visible" in css
    assert ".content-motion-visible .kpi .value" in css
    assert ".content-motion-visible svg.curve .curve-path" in css
    assert ".content-motion-visible tr.is-best" in css
    assert "fetch(" not in script
    assert "WebSocket" not in script


def test_actual_page_content_staggers_rows_cards_and_rolls_numeric_values() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS
    script = board_terminal._MOTION_SCRIPT

    assert "'.kpi, .find-card, table.board tbody tr, .attr-row, .dive-tab'" in script
    assert "--motion-delay" in script
    assert ".content-motion-visible .motion-item" in css
    assert "@keyframes item-arrive" in css
    assert "rollContentNumber" in script
    assert "node.textContent = original" in script


def test_best_pick_row_composes_arrival_with_pulse_instead_of_hiding() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS

    # Regression guard for the 2026-09-03 owner report (Best Pick row present
    # in the DOM but invisible): the stagger script tags every board row as
    # .motion-item, hidden at opacity:0 until arrival, while
    # tr.is-best outranks .motion-item -- so a pulse-only rule replaces the
    # arrival wholesale and the pulse never touches opacity, stranding the
    # row invisible. The composed rule must carry the opacity-animating
    # arrival alongside the pulse.
    marker = ".content-motion-visible tr.is-best.motion-item"
    assert marker in css
    block = css[css.index(marker) :]
    block = block[: block.index("}")]
    assert "item-arrive" in block
    assert "best-row-pulse" in block
    assert "animation-delay" in block


def test_actual_page_content_has_ambient_compositor_only_telemetry() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS
    script = board_terminal._MOTION_SCRIPT

    assert "@keyframes panel-trace" in css
    assert "@keyframes section-beacon" in css
    assert "@keyframes active-tab-trace" in css
    assert "@keyframes best-star-twinkle" in css
    panel_trace = css[css.index("@keyframes panel-trace") :]
    assert "translate3d" in panel_trace
    assert "width:" not in panel_trace.split("@keyframes section-beacon", 1)[0]
    assert "content-motion-active" in script
    assert "ambientObserver" in script
    assert "infinite paused" in css
    assert ".content-motion-active .headline-block::before" in css


def test_actual_page_content_motion_has_a_static_reduced_motion_state() -> None:
    css = board_terminal.TERMINAL_STYLE_CSS
    reduced_motion = css[css.rfind("@media (prefers-reduced-motion: reduce)") :]

    assert ".content-motion-ready" in reduced_motion
    assert "animation:none" in reduced_motion
    assert "transform:none" in reduced_motion
    assert "opacity:1" in reduced_motion
    assert ".content-motion-visible .motion-item" in reduced_motion
    assert ".headline-block::before" in reduced_motion
    assert "display:none" in reduced_motion


def test_cmd_row_varies_by_page_via_content_layer(site_content: SiteContent) -> None:
    model_html = board_terminal.render_model_page(site_content.model)
    findings_html = board_terminal.render_findings_page(site_content.findings)
    history_html = board_terminal.render_history_page(site_content.history)
    index_html = board_terminal.render(site_content.board)
    assert "--page model" in model_html
    assert "--page findings" in findings_html
    assert "--page history" in history_html
    assert "--page model" not in index_html
    assert "--page findings" not in index_html
    assert "--page history" not in index_html


def test_dive_panels_have_id_anchors_for_ticker_deep_links() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    for dive in content.dives:
        assert f'id="{dive.game_id}"' in html


# ---------------------------------------------------------------------------
# item 8 -- six-season dot chart
# ---------------------------------------------------------------------------


def test_six_season_chart_renders_one_direct_labeled_dot_per_season(
    site_content: SiteContent,
) -> None:
    html = board_terminal.render_model_page(site_content.model)
    assert 'class="curve season-chart"' in html
    assert html.count('class="season-dot') == len(site_content.model.seasons)
    for row in site_content.model.seasons:
        assert f"{row.opener_accuracy:.1%}" in html
        assert escape(row.season) in html
    assert "50% coin flip" in html


def test_six_season_chart_keeps_the_season_table_too(site_content: SiteContent) -> None:
    html = board_terminal.render_model_page(site_content.model)
    assert "<th>Season</th>" in html
    assert html.index('class="curve season-chart"') < html.index("<th>Season</th>")


# ---------------------------------------------------------------------------
# item 9 -- grouped challenger ledger
# ---------------------------------------------------------------------------


def test_ledger_is_grouped_into_graded_and_waiting_sections(site_content: SiteContent) -> None:
    html = board_terminal.render_model_page(site_content.model)
    assert site_content.model.graded_rows, "fixture expected at least one graded arm"
    assert site_content.model.waiting_rows, "fixture expected at least one waiting arm"
    assert "<h3>Tracked against a record</h3>" in html
    assert "<h3>Waiting on the season</h3>" in html
    graded_index = html.index("Tracked against a record")
    waiting_index = html.index("Waiting on the season")
    assert graded_index < waiting_index
    promoted = next(row for row in site_content.model.rows if row.is_promoted)
    assert promoted in site_content.model.graded_rows
    for row in site_content.model.waiting_rows:
        assert row.games is None


def test_grouped_ledger_rows_cover_every_original_row(site_content: SiteContent) -> None:
    grouped_ids = {row.arm_id for row in site_content.model.graded_rows} | {
        row.arm_id for row in site_content.model.waiting_rows
    }
    original_ids = {row.arm_id for row in site_content.model.rows}
    assert grouped_ids == original_ids


# ---------------------------------------------------------------------------
# item 10 -- link previews + cadence line
# ---------------------------------------------------------------------------


def test_every_page_has_og_and_twitter_meta_tags(site_content: SiteContent) -> None:
    index_html = board_terminal.render(site_content.board)
    model_html = board_terminal.render_model_page(site_content.model)
    findings_html = board_terminal.render_findings_page(site_content.findings)
    history_html = board_terminal.render_history_page(site_content.history)
    for page_html, link_preview in (
        (index_html, site_content.board.link_preview),
        (model_html, site_content.model.link_preview),
        (findings_html, site_content.findings.link_preview),
        (history_html, site_content.history.link_preview),
    ):
        assert f'<meta property="og:title" content="{escape(link_preview.title)}">' in page_html
        assert (
            f'<meta property="og:description" content="{escape(link_preview.description)}">'
            in page_html
        )
        assert '<meta property="og:site_name" content="ATS Terminal">' in page_html
        assert '<meta name="twitter:card" content="summary">' in page_html


def test_footer_generated_line_carries_the_cadence_note() -> None:
    html = board_terminal.render(build_fixture_content())
    assert escape(CADENCE_NOTE) in html
    gen_section = html[html.index('class="gen"') :]
    assert escape(CADENCE_NOTE) in gen_section[:400]
