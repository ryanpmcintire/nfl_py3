"""Tests for the ATS Terminal renderer (:mod:`nfl_ats.board_terminal`).

``board_terminal.render`` is a pure function over
:class:`nfl_ats.board_content.BoardContent`, so most of these tests render
from the shared hand-built fixture in ``tests/_board_content_fixtures.py`` --
no artifact tree needed. ``render_model_page``/``render_findings_page`` have
no hand-built fixture (like the picks page's own ``_board_content_fixtures``
predates the six-extra-page draft this replaced), so they get integration-
level coverage against real repo artifacts via
``board_site_content.load_site_content``, loaded once per test module.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from html import escape
from pathlib import Path

import pytest
from _board_content_fixtures import (
    build_fixture_content,
    build_fixture_content_with_degraded_states,
)

from nfl_ats import board_terminal
from nfl_ats.board_content import (
    BANNED_BOILERPLATE,
    SourcePolicyRow,
    SourcePolicyView,
)
from nfl_ats.board_site_content import (
    ChallengerAssessment,
    HistoryPageContent,
    HistoryPickRow,
    SiteContent,
)
from nfl_ats.lineup_view import TeamLineup

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NO_DESK_TOKENS = (
    "skin-toggle",
    "Cover Desk",
    "desk/",
    "terminal/",
    "ats-board-skin",
)
_NO_OBSERVATORY_TOKENS = ("theme-obs", "site_theme/", "chalk-filter", "theme-toggle-mount")
_OLD_SITE_PAGE_HREFS = (
    "team_explorer.html",
    "pool.html",
    "ledger.html",
    "models.html",
)


@pytest.fixture(scope="module")
def site_content(_shared_real_site_content: SiteContent) -> SiteContent:
    """Real repo artifacts -- The Model and Findings pages have no
    hand-built fixture (unlike the This Week page's
    ``_board_content_fixtures``). Loaded once for the whole test session via
    ``tests/conftest.py::_shared_real_site_content`` (WP51, test-suite
    speed), shared with ``tests/test_board_improvements.py`` and (through
    ``tests/test_board_site.py``'s ``site`` fixture) ``build_site``."""

    return _shared_real_site_content


def _assert_nav_lists_every_page(html: str) -> None:
    for filename, _label, _title in board_terminal.SITE_PAGES:
        assert f'href="{filename}"' in html, f"nav missing link to {filename}"


def _assert_no_desk_references(html: str) -> None:
    for token in _NO_DESK_TOKENS:
        assert token not in html, f"found dropped-Desk-skin token {token!r}"


def _assert_no_observatory_references(html: str) -> None:
    for token in _NO_OBSERVATORY_TOKENS:
        assert token not in html


def _assert_no_cut_page_links(html: str) -> None:
    for href in _OLD_SITE_PAGE_HREFS:
        assert f'href="{href}"' not in html, f"nav links to a page cut from the site: {href}"


def _mockup_style() -> str:
    mockup_path = (
        Path(__file__).resolve().parents[1] / "src" / "nfl_ats" / "board_terminal_style.css"
    )
    return mockup_path.read_text(encoding="utf-8")


def test_terminal_style_css_constant_matches_asset_file() -> None:
    assert _mockup_style() == board_terminal.TERMINAL_STYLE_CSS


def test_terminal_stylesheet_verbatim_prefix_is_byte_identical_to_the_mockup() -> None:
    """The CSS the page ships is verbatim mockup CSS plus clearly delimited
    appended blocks (degraded states, the game selector/adjuster, evidence
    density, extended pages) -- never a re-expression of the design."""

    css = board_terminal.TERMINAL_STYLE_CSS
    marker = "/* degraded states -- appended to verbatim mockup sheet */"
    assert marker in css
    verbatim_prefix = css[: css.index(marker)].rstrip()
    # The verbatim prefix must not itself contain any "appended" marker --
    # i.e. everything before the first appended-block comment is untouched
    # mockup CSS.
    assert "appended to verbatim mockup sheet" not in verbatim_prefix


def test_terminal_page_has_viewport_meta_and_doctype() -> None:
    page = board_terminal.render(build_fixture_content())
    assert page.startswith("<!doctype html>")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert '<meta charset="utf-8">' in page
    assert "<title>ATS Terminal</title>" in page


def test_terminal_board_table_is_wrapped_in_its_overflow_container() -> None:
    page = build_fixture_content()
    html = board_terminal.render(page)
    match = re.search(r'<div class="board-scroll">(.*?)</div></section>', html, re.S)
    assert match is not None
    assert '<table class="board">' in match.group(1)


def test_terminal_renders_all_sixteen_games() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    assert len(content.games) == 16
    for game in content.games:
        assert game.away in html
        assert game.home in html
        assert game.pick_team in html
        assert game.probability_text in html
    # 16 game rows (15 plain + 1 best-pick) plus 4 day-group header rows.
    assert html.count('<tr class="game') == 16
    assert html.count('<tr class="grp">') == 4


def test_terminal_best_pick_flag_renders_once() -> None:
    html = board_terminal.render(build_fixture_content())
    assert html.count("best-flag") >= 1
    assert "&#9733;" in html


def test_terminal_no_desk_references() -> None:
    html = board_terminal.render(build_fixture_content())
    _assert_no_desk_references(html)


def test_terminal_no_cut_page_links() -> None:
    html = board_terminal.render(build_fixture_content())
    _assert_no_cut_page_links(html)


def test_terminal_stylesheet_class_set_is_subset_of_mockup_plus_allowlist() -> None:
    """Every class used in the generated BODY is either a mockup class
    (used in its body OR merely defined in its CSS, including the appended
    blocks this conversion added) or one of the small, explicit additive
    classes for the degraded states."""

    mockup_html = _mockup_style()
    generated = board_terminal.render(build_fixture_content())
    body_match = re.search(r"</head>\s*<body>(.*)</body>", generated, re.S)
    assert body_match is not None
    generated_classes = {
        c for value in re.findall(r'class="([^"]*)"', body_match.group(1)) for c in value.split()
    }
    mockup_defined = set(re.findall(r"\.([A-Za-z][\w-]*)", mockup_html))
    allowlist = {
        "attr-empty",
        "dash",
        "note",
        "chart-empty",
        "is-active",
        # UI-20(a), 2026-09-05: the "Why this pick" collapsed row under each
        # pick row on the board table -- an additive state row, not a mockup
        # redesign (see board_terminal._why_this_pick_row_html).
        "explain",
    }
    extra = generated_classes - mockup_defined - allowlist
    assert extra == set(), f"classes not in the mockup or allowlist: {extra}"


def test_terminal_no_observatory_references() -> None:
    html = board_terminal.render(build_fixture_content())
    _assert_no_observatory_references(html)


def test_terminal_no_illustrative_tag_survives() -> None:
    """The mockup's 'Illustrative breakdown' sample-tag must never appear on
    a page built from real content."""

    html = board_terminal.render(build_fixture_content())
    assert "Illustrative breakdown" not in html
    assert "constructed for this mockup" not in html


def test_terminal_nav_has_exactly_four_pages() -> None:
    html = board_terminal.render(build_fixture_content())
    _assert_nav_lists_every_page(html)
    assert len(board_terminal.SITE_PAGES) == 4
    labels = {label for _filename, label, _title in board_terminal.SITE_PAGES}
    assert labels == {"This week", "The model", "History", "What we've learned"}


def _history_fixture(*, picks: tuple[HistoryPickRow, ...] = ()) -> HistoryPageContent:
    board = build_fixture_content()
    return HistoryPageContent(
        generated_at_text="2026-09-02 12:00:00 UTC",
        picks=picks,
        primary_available=bool(picks),
        primary_error=None,
        challenger_assessments=(),
        ticker_chrome=board.ticker_chrome,
        link_preview=board.link_preview,
    )


def test_history_empty_state_is_truthful() -> None:
    html = board_terminal.render_history_page(_history_fixture())
    assert "No recorded model picks yet" in html
    assert "primary paper-decision ledger currently has 0 rows" in html


def test_history_renders_confidence_outcomes_best_pick_and_hides_pending_score() -> None:
    rows = (
        HistoryPickRow(
            "settled",
            2026,
            1,
            "AWY",
            "HME",
            "HOME",
            -3.5,
            0.61,
            "model-1",
            True,
            "settled",
            True,
            "24 at 17",
        ),
        HistoryPickRow(
            "push",
            2026,
            1,
            "AW2",
            "HM2",
            "AWAY",
            2.5,
            0.52,
            "model-1",
            False,
            "push",
            None,
            "20 at 20",
        ),
        HistoryPickRow(
            "pending",
            2026,
            2,
            "AW3",
            "HM3",
            "AWAY",
            1.5,
            0.58,
            "model-1",
            False,
            "pending",
            None,
            "SECRET FUTURE SCORE",
        ),
    )
    html = board_terminal.render_history_page(_history_fixture(picks=rows))
    assert "61.0%" in html
    assert "Correct" in html
    assert "Push" in html
    assert "Pending" in html
    assert "Best pick" in html
    assert "24 at 17" in html
    assert "SECRET FUTURE SCORE" not in html


def test_history_renders_settled_challenger_assessment_without_play_decision_threshold() -> None:
    content = _history_fixture()
    content = HistoryPageContent(
        **{
            **content.__dict__,
            "challenger_assessments": (
                ChallengerAssessment(
                    "challenger-x",
                    "Challenger X",
                    2,
                    1,
                    1,
                    0,
                    1,
                    0.5,
                    2.5,
                    0.87,
                    0.01,
                    0.04,
                    "Settled prospectively at the frozen decision/opener line; "
                    "paired with active model.",
                ),
            ),
        }
    )
    html = board_terminal.render_history_page(content)
    assert "Challenger X" in html
    assert "+2.50 pts" in html
    assert "probability_positive 0.87" in html
    assert "frozen decision/opener line" in html
    assert "promotion threshold" in html


def test_terminal_headline_main_foot_text_stays_mockup_scale() -> None:
    """Regression guard for the 2026-08 coordinator finding: a long foot
    caption inside ``.headline-main`` (``flex:0 0 auto``, no max-width in
    the verbatim mockup CSS) balloons the box and crushes the ``.caveat``
    sibling into a single-word rail. The played-card foot text used here
    must stay short (mockup scale), never the long prose caption."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    assert content.headline.played_card_foot_text in html
    # The layout hazard is VISIBLE text: strip non-rendered <script> blocks
    # first (the UI-16 assistant corpus embeds the long caption as data in
    # its application/json blob, which cannot balloon any flex box).
    visible = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert content.headline.played_card_caption not in visible
    assert len(content.headline.played_card_foot_text) < 80


def test_terminal_headline_main_has_a_defensive_max_width_rule() -> None:
    """Defense in depth alongside the short-caption fix: even if a future
    caption grows long again, the stylesheet itself must not let
    ``.headline-main`` balloon and crush ``.caveat``."""

    rule_bodies = re.findall(r"\.headline-main\{([^}]*)\}", board_terminal.TERMINAL_STYLE_CSS)
    assert len(rule_bodies) >= 2, "expected the mockup rule plus an appended override"
    override = rule_bodies[-1]  # CSS cascade: the LAST rule with equal specificity wins
    assert "max-width" in override
    assert "flex:01auto" in override.replace(" ", "")


# ---------------------------------------------------------------------------
# Mobile-width overflow fix (2026-08-31 390px-iframe audit): a policy overlay
# id, challenger-ledger registry keys, a findings trace chip, a watching-
# lead's channel name/artifact path, and a signal-registry name are all long
# unbreakable mono identifiers that escaped every overflow-clipping ancestor
# at narrow widths. See ``tests/test_board_site.py`` for the real-content,
# real-page HTML scan; these test the stylesheet contract directly.
# ---------------------------------------------------------------------------


def _selectors_with_declaration(css: str, declaration_pattern: str) -> set[str]:
    """Every selector (normalized to single-spaced, comma-split tokens) that
    appears in a rule whose body matches ``declaration_pattern``. Strips
    ``/* ... */`` comments first -- this file's comments routinely mention a
    class name (e.g. ``.policy-note``) in prose right before the real rule,
    and without stripping them that prose glues onto the first selector in
    the following comma list. Works across ``@media`` blocks too: the regex
    only ever matches an innermost, non-nested ``selector{body}`` pair, so an
    enclosing ``@media (...){`` never itself completes a match and is simply
    skipped over."""

    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    selectors: set[str] = set()
    for selector_group, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if re.search(declaration_pattern, body):
            for raw in selector_group.split(","):
                selectors.add(re.sub(r"\s+", " ", raw).strip())
    return selectors


def test_mobile_overflow_fix_css_covers_every_long_identifier_class() -> None:
    """Every class that can hold a long unbreakable mono identifier must
    carry ``overflow-wrap:anywhere`` -- not the legacy ``break-word`` the
    ``.ledger-fixed td`` rule already used, which does not shrink an
    element's min-content size and so never lets a flex/grid item actually
    shrink to fit (that's the whole reason ``anywhere`` was introduced
    alongside it)."""

    covered = _selectors_with_declaration(
        board_terminal.TERMINAL_STYLE_CSS, r"overflow-wrap\s*:\s*anywhere"
    )
    required = {
        ".policy-note",  # index.html + model.html: policy overlay id / model id
        ".evidence-pill",  # model.html: challenger-ledger registry keys
        ".trace-chip",  # findings.html: registry signal name trace chip
        ".attr-row .chan",  # findings.html watching-leads + index.html dive attribution
        ".attr-row .chan-sub",  # findings.html watching-leads artifact paths
        ".mono-id",  # findings.html signal-registry name, model.html ledger arm fallback
    }
    missing = required - covered
    assert not missing, f"no overflow-wrap:anywhere rule for: {missing}"


def test_mobile_overflow_fix_attr_row_first_column_can_shrink() -> None:
    """``.attr-row``'s first grid column is track ``1fr``, which -- like a
    flex item -- defaults to ``min-width:auto`` and will not shrink below
    its own content's min-content width. ``overflow-wrap`` on ``.chan``
    alone cannot help unless this parent explicitly opts out of that
    floor."""

    bodies = re.findall(
        r"\.attr-row\s*>\s*div:first-child\{([^}]*)\}", board_terminal.TERMINAL_STYLE_CSS
    )
    assert bodies, ".attr-row > div:first-child rule not found"
    assert "min-width:0" in bodies[-1].replace(" ", "")


def test_mobile_overflow_fix_board_table_cells_wrap_onto_multiple_lines() -> None:
    """The mobile board-collapse (``@media (max-width:680px)``) turns every
    ``table.board td`` into a flex container, ``flex-wrap:nowrap`` by
    default. A cell with more than one child -- several ``.evidence-pill``
    chips, or a name plus its ``.game-sub`` caption -- needs
    ``flex-wrap:wrap`` or those children are forced onto one un-shrinking
    line regardless of any ``overflow-wrap`` set on them."""

    css = board_terminal.TERMINAL_STYLE_CSS
    index = css.rfind("@media (max-width:680px)")
    assert index != -1, "no max-width:680px mobile block found"
    tail = css[index:]
    assert re.search(r"table\.board td\{[^}]*flex-wrap\s*:\s*wrap", tail), (
        "the mobile block's table.board td rule never gained flex-wrap:wrap"
    )


def test_terminal_attribution_labels_are_plain_english() -> None:
    """Regression guard: no raw jargon family ids (``player_qb``,
    ``weekly_context``) or a wall of near-zero rows -- curated, capped,
    plain-English labels only."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    for jargon in ("_contribution", "player_qb", "weekly_context", "Player_qb"):
        assert jargon not in html
    best_dive = next(dive for dive in content.dives if dive.is_best)
    assert best_dive.attribution.net_label is not None
    assert best_dive.attribution.net_label in html
    assert len(best_dive.attribution.rows) <= 6


# ---------------------------------------------------------------------------
# Game selector + line-offset adjuster (2026-08-31 owner redirect: the
# standalone "spread explorer" page folded into This Week's deep dive).
# ---------------------------------------------------------------------------


@pytest.mark.full  # ENG-11: renders from real on-disk artifacts (dominates --durations)
def test_this_week_page_renders_from_real_artifacts_with_guard_proven_adjusters(
    site_content: SiteContent,
) -> None:
    """End-to-end proof (not just the hand-built fixture) that
    ``load_board_content`` builds a real, guard-proven adjuster for every
    game the active model's probability method supports, and the page
    renders every one of them without raising."""

    html = board_terminal.render(site_content.board)
    assert html.startswith("<!doctype html>")
    # UI-20 layout A (2026-09-05): the board's own rows are the selector now
    # (no more separate ``.dive-tab`` strip) -- one row-link per game, one
    # inspector panel per game.
    assert html.count('class="row-link"') == len(site_content.board.games)
    assert html.count('<div class="dive-panel"') == len(site_content.board.games)
    dives_with_adjuster = [dive for dive in site_content.board.dives if dive.adjuster is not None]
    for dive in dives_with_adjuster:
        assert f'data-center="{dive.adjuster.center:.6f}"' in html  # type: ignore[union-attr]


def test_board_rows_select_every_game_and_default_to_the_best_pick() -> None:
    """UI-20 layout A (2026-09-05, owner: "layout A is definitely the
    best. lets go with that."): the old standalone ``.dive-selector`` tab
    strip is gone -- the board's own rows (``.row-link`` anchors) are the
    one and only selector into the inspector's panels."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    assert html.count('class="row-link"') == len(content.games)
    assert html.count('<div class="dive-panel"') == len(content.games)
    # Exactly one panel is visible by default -- the Best Pick's.
    assert html.count(" hidden>") == len(content.games) - 1


def test_default_inspector_panel_is_the_best_pick_and_only_one_is_visible() -> None:
    """New for layout A: exactly one inspector panel is visible without
    JavaScript or a URL hash, and it is the Best Pick's -- every other
    game's panel carries ``hidden``."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    best_id = content.best_pick_game_id
    assert best_id is not None
    best_panel_open = f'<div class="dive-panel" id="{best_id}" data-game-id="{best_id}">'
    assert best_panel_open in html
    assert f'<div class="dive-panel" id="{best_id}" data-game-id="{best_id}" hidden>' not in html
    assert html.count(" hidden>") == len(content.games) - 1


def test_every_game_has_an_inspector_panel() -> None:
    """New for layout A: every game on the board gets a panel in the
    inspector, addressable by its own game id -- not only the Best Pick."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    for game in content.games:
        assert f'<div class="dive-panel" id="{game.game_id}"' in html
        assert f'data-game-id="{game.game_id}"' in html


def test_board_row_anchors_resolve_to_a_real_inspector_panel_id() -> None:
    """New for layout A: with JavaScript disabled, each board row is a
    plain ``href="#<game_id>"`` link -- every one of those ids must name a
    real ``.dive-panel`` so the link actually resolves to something."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    hrefs = re.findall(r'class="row-link" href="#([^"]+)"', html)
    assert len(hrefs) == len(content.games)
    panel_ids = set(re.findall(r'<div class="dive-panel" id="([^"]+)"', html))
    for game_id in hrefs:
        assert game_id in panel_ids


def test_week_grid_holds_the_board_and_inspector_columns_in_order() -> None:
    """New for layout A: the board and inspector both render inside one
    ``.week-grid``, board first (left column), inspector second (right
    column) -- matching the approved "board + inspector" mockup."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    grid_index = html.index('<div class="week-grid">')
    board_index = html.index('aria-labelledby="board-h"')
    inspector_index = html.index('aria-labelledby="dive-h"')
    assert grid_index < board_index < inspector_index
    assert 'class="board-col"' in html
    assert 'class="inspector-col"' in html


def test_week_grid_css_is_two_columns_on_desktop_and_stacks_below_1100px() -> None:
    """New for layout A: desktop (>=1100px) is a real two-column CSS grid;
    below 1100px it stacks to one column (board first, by document order)."""

    css = board_terminal.TERMINAL_STYLE_CSS
    assert "main.week-page{ width:100%; max-width:1320px; margin:0 auto; }" in css
    assert "grid-template-columns:minmax(700px, 1.3fr) minmax(0, 1fr)" in css
    assert ".board-col table.board{ table-layout:fixed; }" in css
    assert "@media (max-width:1100px){" in css
    stacked = css[css.index("@media (max-width:1100px){") :]
    assert ".week-grid{ grid-template-columns:1fr; }" in stacked


def test_dive_panels_render_every_games_attribution_and_chart_without_raising() -> None:
    """The 15 non-Best-Pick games in the default fixture exercise the
    degraded "attribution not published" / "cover curve not published"
    paths; the renderer must handle all of it without raising."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    assert "Attribution not published." in html
    assert "chart-empty" in html
    assert "adjuster-empty" in html


def test_dive_degraded_states_render_without_raising_for_the_best_pick_too() -> None:
    content = build_fixture_content_with_degraded_states()
    html = board_terminal.render(content)
    assert html.count("Attribution not published.") == len(content.games)


def test_adjuster_widget_carries_guard_proven_params_for_the_best_pick() -> None:
    content = build_fixture_content()
    best_dive = next(dive for dive in content.dives if dive.is_best)
    assert best_dive.adjuster is not None
    html = board_terminal.render(content)
    assert f'data-center="{best_dive.adjuster.center:.6f}"' in html
    assert f'data-mean="{best_dive.adjuster.residual_mean:.6f}"' in html
    assert f'data-std="{best_dive.adjuster.residual_std:.6f}"' in html
    assert "ats-adjuster" in html
    assert "adjuster-slider" in html


def test_inspector_panel_star_marks_the_best_pick_only() -> None:
    """UI-20 layout A: the old standalone selector's star check moves to
    the inspector's own panels -- across all 16 (mostly hidden) panels,
    only the Best Pick's carries the star in its header."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    panels_match = re.search(r'<div class="dive-panels">(.*?)</section>', html, re.S)
    assert panels_match is not None
    assert panels_match.group(1).count("&#9733;") == 1


def test_cover_curve_offset_zero_note_renders_when_present() -> None:
    content = build_fixture_content()
    best_dive = next(dive for dive in content.dives if dive.is_best)
    assert best_dive.cover_curve_offset_zero_note is not None
    html = board_terminal.render(content)
    assert escape(best_dive.cover_curve_offset_zero_note) in html


def test_cover_curve_marker_is_oriented_to_the_pick_side() -> None:
    """Regression guard: the curve's own marker/aria-label must name the
    PICK side, never the home team when the pick is the away team (2026-08
    coordinator finding: marker read "LV +3.5" while the headline read
    "MIA +3.5" for the same game)."""

    content = build_fixture_content()
    best_pick = next(game for game in content.games if game.is_best)
    html = board_terminal.render(content)
    assert f"{best_pick.pick_team} {best_pick.pick_spread_text}" in html
    dive_section = html[html.index('id="dive-h"') :]
    assert best_pick.pick_team in dive_section


def test_terminal_index_title_stays_unqualified() -> None:
    """Regression guard: only ``index.html`` keeps the bare ``ATS Terminal``
    title; every other page must be qualified with its own page label."""

    html = board_terminal.render(build_fixture_content())
    assert "<title>ATS Terminal</title>" in html


# ---------------------------------------------------------------------------
# The Model page (the aggregate model record; row-level outcomes live in History).
# ---------------------------------------------------------------------------


def test_model_page_renders_real_ledger_and_season_facts(site_content: SiteContent) -> None:
    html = board_terminal.render_model_page(site_content.model)
    assert html.startswith("<!doctype html>")
    assert "ATS Terminal — The model" in html
    _assert_nav_lists_every_page(html)
    _assert_no_desk_references(html)
    _assert_no_cut_page_links(html)
    _assert_no_observatory_references(html)
    assert site_content.model.ledger_available
    row = site_content.model.rows[0]
    assert escape(row.display_name) in html
    assert escape(row.summary_sentence) in html
    season = site_content.model.seasons[0]
    assert escape(season.season) in html
    for rung in site_content.model.ladder_rungs:
        assert escape(rung) in html
    if site_content.model.families:
        assert escape(site_content.model.families[0].label) in html


def test_model_page_headline_matches_this_week_headline(site_content: SiteContent) -> None:
    """The one deliberate cross-page dedup exception: The Model page's
    headline strip must be the SAME object This Week renders, not a
    recomputed copy."""

    assert site_content.model.headline is site_content.board.headline


def test_model_page_promoted_row_never_prints_the_bare_no_cited_evidence_phrase(
    site_content: SiteContent,
) -> None:
    """Regression guard for the 2026-08-31 browser-QA finding: the promoted
    row's evidence cell must carry a real provenance line, never the
    ``.micro``-forced-uppercase ``NO CITED EVIDENCE`` that read like an
    indictment."""

    html = board_terminal.render_model_page(site_content.model)
    assert "NO CITED EVIDENCE" not in html
    promoted = next(row for row in site_content.model.rows if row.is_promoted)
    assert not promoted.evidence  # the promoted row never cites outside registry evidence
    assert "PROMOTED" in html  # the status badge itself still renders


def test_model_page_ledger_evidence_collapses_past_the_inline_limit(
    site_content: SiteContent,
) -> None:
    """Regression guard for the row-height bug: a challenger with many
    evidence entries must collapse the overflow behind a ``<details>``
    toggle rather than rendering every chip inline unbounded."""

    html = board_terminal.render_model_page(site_content.model)
    many_evidence_rows = [row for row in site_content.model.rows if len(row.evidence) > 3]
    if many_evidence_rows:
        assert "evidence-more" in html
        assert f"+{len(many_evidence_rows[0].evidence) - 3} more" in html or any(
            f"+{len(row.evidence) - 3} more" in html for row in many_evidence_rows
        )


def test_model_page_ledger_table_uses_a_fixed_layout(site_content: SiteContent) -> None:
    """Regression guard for the row-height bug's other half: auto layout
    let one column's long content squeeze another down to a sliver."""

    html = board_terminal.render_model_page(site_content.model)
    assert 'table class="board ledger-fixed"' in html
    assert "<colgroup>" in html


def test_model_ledger_interval_never_renders_an_impossible_percentage(
    site_content: SiteContent,
) -> None:
    """Regression guard for the 2026-08-31 browser-QA unit bug: several
    challenger rows carry accuracy-POINTS effect intervals (e.g.
    ``surface_switch_tilt_overlay``'s ``[0.29, 2.038]``), which a percent
    formatter rendered as absurd percentages -- ``[29.0%, 203.8%]``,
    ``[79.0%, 3167.0%]``. The smoking gun: no rendered ledger interval may
    ever show a percentage above 100%, and no accuracy-points-typed interval
    may render a ``%`` sign at all -- it must render as signed points."""

    html = board_terminal.render_model_page(site_content.model)
    for match in re.findall(r"(-?\d+(?:\.\d+)?)%", html):
        assert float(match) <= 100.0, f"ledger rendered an impossible {match}% interval bound"

    points_rows = [
        row
        for row in site_content.model.rows
        if row.interval_low is not None
        and row.interval_high is not None
        and row.interval_unit == "accuracy_points"
    ]
    assert points_rows, "fixture expected at least one accuracy-points ledger row"
    for row in points_rows:
        cell_text = board_terminal._ledger_interval_text(row)
        assert "%" not in cell_text, (
            f"{row.arm_id} accuracy-points interval rendered a % sign: {cell_text!r}"
        )
        assert "pts" in cell_text

    rate_rows = [
        row
        for row in site_content.model.rows
        if row.interval_low is not None
        and row.interval_high is not None
        and row.interval_unit == "accuracy_rate"
    ]
    assert rate_rows, "fixture expected the promoted row's accuracy-rate interval"
    for row in rate_rows:
        cell_text = board_terminal._ledger_interval_text(row)
        assert "%" in cell_text, f"{row.arm_id} accuracy-rate interval lost its % formatting"


def test_model_ledger_every_live_challenger_arm_has_a_human_display_name() -> None:
    """Regression guard for the 2026-08-31 browser-QA name-gap bug:
    ``pbp08_protection_mismatch_tilt_overlay`` rendered its raw id as its
    display name, because it was missing from ``CHALLENGER_DISPLAY_NAMES``
    -- every OTHER arm has a curated human name (that mapping's own
    docstring: "Human names for every arm id"). This reads the LIVE
    ``artifacts/prospective/challengers.json`` (never a hand-typed id list,
    the exact staleness that let the gap regress silently), so a future
    challenger added to the registry without a curated name fails this test
    immediately rather than rendering its raw id on the public site."""

    from nfl_ats.dashboard.findings_content import CHALLENGER_DISPLAY_NAMES

    challengers_path = _REPO_ROOT / "artifacts" / "prospective" / "challengers.json"
    payload = json.loads(challengers_path.read_text(encoding="utf-8"))
    for entry in payload["challengers"]:
        challenger_id = str(entry["challenger_id"])
        assert challenger_id in CHALLENGER_DISPLAY_NAMES, (
            f"{challenger_id!r} has no curated display name -- it will render its raw id "
            "on the public Model Ledger"
        )
        assert CHALLENGER_DISPLAY_NAMES[challenger_id].strip()
        assert CHALLENGER_DISPLAY_NAMES[challenger_id] != challenger_id


# ---------------------------------------------------------------------------
# What We've Learned (Findings, plus the compact signal-registry summary
# that replaced the standalone Signal Ledger page).
# ---------------------------------------------------------------------------


def test_findings_page_renders_real_findings(site_content: SiteContent) -> None:
    html = board_terminal.render_findings_page(site_content.findings)
    assert "ATS Terminal — What we&#x27;ve learned" in html or "What we've learned" in html
    _assert_nav_lists_every_page(html)
    _assert_no_desk_references(html)
    _assert_no_cut_page_links(html)
    _assert_no_observatory_references(html)
    group = site_content.findings.groups[0]
    finding = group.findings[0]
    assert escape(finding.plain_answer) in html
    assert escape(finding.question) in html


def test_findings_page_real_content_carries_no_banned_boilerplate(
    site_content: SiteContent,
) -> None:
    """Every other banned-boilerplate check in this suite (e.g.
    ``tests/test_board_content_coverage.py``,
    ``tests/test_played_card_expectation.py``) runs against a HAND-BUILT
    fixture, so it can never catch a banned phrase that only exists in the
    real registry text -- "Research this week" (UI-20(b)) and "What we're
    watching" both print registry ``plain_summary``/``description`` prose
    almost verbatim (``board_terminal._recent_activity_category_html`` /
    ``_watching_lead_html``). This is the ONE test in the suite that scans
    the real, registry-fed findings page for :data:`BANNED_BOILERPLATE`,
    guarding against a phrase like "not a promotion or wagering claim"
    (found live on ``docs/findings.html`` 2026-09-05, sourced from
    ``registry/weak_signals.json``'s ``per13_durability_on_production_opener_ats``
    entry, reworded to plain English rather than banned outright -- see that
    constant's own docstring for why a bare "wagering" ban would
    false-positive elsewhere) surviving a future registry write."""

    html = board_terminal.render_findings_page(site_content.findings)
    for phrase in BANNED_BOILERPLATE:
        assert phrase not in html.lower(), phrase


def test_findings_page_has_no_standalone_challenger_cards_field() -> None:
    """Dedup regression guard: the tracked-challenger cards this page used
    to render are superseded by The Model page's own (richer) ledger rows
    -- the content object must not carry that field at all."""

    from nfl_ats.board_site_content import FindingsPageContent

    assert "challengers" not in FindingsPageContent.__dataclass_fields__


def test_findings_page_signal_registry_summary_renders(site_content: SiteContent) -> None:
    html = board_terminal.render_findings_page(site_content.findings)
    summary = site_content.findings.ledger_summary
    assert str(summary.total_signals) in html
    for status, count in summary.counts_by_status.items():
        assert str(count) in html, f"missing count for status {status!r}"
    for row in summary.notable:
        # Rendered as words, not the raw registry id (owner mandate,
        # 2026-09-05: "this is for humans not the opus autist") --
        # ``board_terminal.humanize_identifier`` is the one place that
        # translation happens.
        assert escape(row.name.replace("_", " ")) in html


def test_findings_page_ledger_summary_is_not_the_full_registry_table() -> None:
    """The compact secondary section shows the registry's highest-
    confidence signals, capped -- never every recorded signal (that would
    just be the old standalone Signal Ledger page again)."""

    from nfl_ats.board_site_content import _NOTABLE_SIGNAL_LIMIT

    assert _NOTABLE_SIGNAL_LIMIT <= 10


def test_ledger_rows_appear_on_model_page_not_on_findings_page(site_content: SiteContent) -> None:
    """Dedup guard: a model-ledger arm's display name is distinctive
    (e.g. "Played card — model + fix-up rules") and must not leak onto
    Findings -- the ledger lives on exactly one page now."""

    model_html = board_terminal.render_model_page(site_content.model)
    findings_html = board_terminal.render_findings_page(site_content.findings)
    for row in site_content.model.rows:
        assert escape(row.display_name) in model_html
        assert escape(row.display_name) not in findings_html


def test_cut_legacy_page_renderers_no_longer_exist() -> None:
    """The owner cut these pages entirely from the build and nav (2026-08-31
    redirect) -- the renderer functions themselves must be gone, not just
    unwired, so nothing can accidentally call them back into the site."""

    assert not hasattr(board_terminal, "render_team_explorer_page")
    assert not hasattr(board_terminal, "render_pool_workbench_page")
    assert not hasattr(board_terminal, "render_models_page")
    assert not hasattr(board_terminal, "render_signal_ledger_page")


# ---------------------------------------------------------------------------
# ENG-34: the SOURCES panel -- ENG-14's source_policy card state, surfaced
# on the This Week page beside the board.
# ---------------------------------------------------------------------------


def _content_with_source_policy(view: SourcePolicyView):
    return replace(build_fixture_content(), source_policy=view)


def test_sources_panel_neutral_states_have_plain_words_and_grey_style() -> None:
    view = SourcePolicyView(
        card_state="complete",
        evaluated_at="2026-09-08T15:30:00+00:00",
        recorded=True,
        rows=(
            SourcePolicyRow("inactives", "not_due", None, 5590, "", "2026-09-13T15:30:00+00:00"),
            SourcePolicyRow("injuries_sportradar", "not_configured", None, 6450, ""),
        ),
    )
    html = board_terminal.render(_content_with_source_policy(view))
    assert (
        "not due yet; posted about 90 minutes before each kickoff, first on Sunday morning" in html
    )
    assert "not set up; the league injury report stands in" in html
    assert "grey: not due yet or not set up" in html
    assert ".src-state.not_due,.src-state.not_configured{ color:var(--text-faint); }" in html


def test_sources_panel_legacy_neutral_row_has_no_invented_day() -> None:
    row = SourcePolicyRow("inactives", "not_due", None, 5590, "")
    assert row.neutral_note == "not due yet; posted about 90 minutes before each kickoff"


def test_sources_panel_absent_block_renders_not_recorded_without_crashing() -> None:
    """The shared fixture never sets ``source_policy`` -- ``BoardContent``'s
    own default is the explicit not-recorded view, matching what a real
    forecast whose metadata predates ENG-14 persistence also degrades to
    (see ``nfl_ats.board_content._default_source_policy_view``)."""

    html = board_terminal.render(build_fixture_content())
    assert 'class="sources-panel policy-note"' in html
    assert "NOT RECORDED" in html
    assert "No source-freshness block is recorded for this forecast" in html
    assert (
        "complete: fresh enough to use; degraded: we fell back to an older copy; "
        "blocked: we refused to publish" in html
    )
    assert 'class="src-row"' not in html


def test_sources_panel_renders_one_line_per_source_with_worst_wins_state() -> None:
    view = SourcePolicyView(
        card_state="degraded",
        evaluated_at="2026-09-03T14:00:00+00:00",
        rows=(
            SourcePolicyRow(
                source_id="odds_opener",
                state="complete",
                observed_at="2026-09-02T09:00:00+00:00",
                budget_minutes=180,
                reason="snapshot is 30.0 min old, inside the 180 min budget",
            ),
            SourcePolicyRow(
                source_id="injuries_nflverse",
                state="degraded",
                observed_at=None,
                budget_minutes=120,
                reason="no snapshot present (budget 120 min)",
            ),
        ),
        recorded=True,
    )
    html = board_terminal.render(_content_with_source_policy(view))
    assert 'class="sources-panel policy-note"' in html
    # Header line carries the worst-wins card state, not either row's own.
    assert '<span class="src-state degraded">DEGRADED</span></b>' in html
    # Rendered as words, not the raw source id (owner mandate, 2026-09-05).
    assert "odds opener" in html
    assert 'class="src-state complete"' in html
    assert "injuries nflverse" in html
    assert 'class="src-state degraded"' in html
    assert "no snapshot" in html  # observed_at_text for the unobserved row
    assert (
        "complete: fresh enough to use; degraded: we fell back to an older copy; "
        "blocked: we refused to publish" in html
    )


def test_sources_panel_blocked_state_renders_with_state_class() -> None:
    view = SourcePolicyView(
        card_state="blocked",
        evaluated_at="2026-09-03T14:00:00+00:00",
        rows=(
            SourcePolicyRow(
                source_id="player_arrests",
                state="blocked",
                observed_at=None,
                budget_minutes=90,
                reason="no snapshot present (budget 90 min)",
            ),
        ),
        recorded=True,
    )
    html = board_terminal.render(_content_with_source_policy(view))
    assert "BLOCKED" in html
    assert 'class="src-state blocked"' in html


def test_sources_panel_follows_table_tiebreaker_and_policy() -> None:
    """Picks come first; all supporting panels remain below them."""

    html = board_terminal.render(build_fixture_content())
    header_index = html.index('id="board-h"')
    sources_index = html.index('class="sources-panel policy-note"')
    policy_index = html.index('<div class="policy-note"><b>Policy overlay</b>')
    table_index = html.index('<table class="board">')
    tiebreaker_index = html.index("Tiebreaker guess</summary>")
    assert header_index < table_index < tiebreaker_index < policy_index < sources_index


def test_sources_panel_is_static_markup_with_no_script_required() -> None:
    """The panel must render with JS disabled, like the rest of the board:
    it is plain HTML built once in Python, never populated by
    ``<script>``-run DOM code."""

    view = SourcePolicyView(
        card_state="complete",
        evaluated_at="2026-09-03T14:00:00+00:00",
        rows=(
            SourcePolicyRow(
                source_id="odds_opener",
                state="complete",
                observed_at="2026-09-02T09:00:00+00:00",
                budget_minutes=180,
                reason="snapshot is 30.0 min old, inside the 180 min budget",
            ),
        ),
        recorded=True,
    )
    html = board_terminal.render(_content_with_source_policy(view))
    without_scripts = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
    assert 'class="sources-panel policy-note"' in without_scripts
    assert "odds opener" in without_scripts
    assert (
        "complete: fresh enough to use; degraded: we fell back to an older copy; "
        "blocked: we refused to publish" in without_scripts
    )


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ("2026-09-05T15:00:00+00:00", "updated 3 hours ago"),
        ("2026-09-05T17:30:00+00:00", "updated less than an hour ago"),
        ("2026-09-04T09:00:00+00:00", "updated Friday morning"),
        ("2026-09-05T19:00:00+00:00", "updated after this card was checked"),
        ("invalid", "update time unavailable"),
        (None, "no snapshot"),
    ],
)
def test_source_age_uses_card_evaluation_time(observed: str | None, expected: str) -> None:
    assert board_terminal._relative_update(observed, "2026-09-05T18:00:00+00:00") == expected


def test_index_reader_text_has_no_utc_timestamp() -> None:
    content = build_fixture_content()
    content = replace(
        content, headline=replace(content.headline, synced_at_text="2026-09-05 14:14 UTC")
    )
    html = board_terminal.render(content)
    reader_html = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.S)
    reader_text = re.sub(r"<[^>]+>", "", reader_html)
    assert "UTC" not in reader_text
    assert "How to read this number" in reader_text
    assert "selection caveat" not in reader_text


def test_lineup_caption_and_missing_injury_report_are_plain_words() -> None:
    lineup = TeamLineup(
        team="NE",
        players=(),
        as_of="2026-09-04T09:00:00+00:00",
        source="NFLVERSE DEPTH CHARTS",
        injury_status=(
            "no players listed on this week's injury report (or the report is not yet "
            "published); per-player probabilities use the play-probability model's own "
            "recent-history and roster-status features"
        ),
    )
    html = board_terminal._lineup_team_html(lineup)
    assert "depth chart from Friday morning" in html
    assert (
        escape(
            "No one from this team is on this week's injury report yet, so these chances "
            "come from each player's recent playing time and roster status."
        )
        in html
    )
    assert "UTC" not in html
    assert "injury feed:" not in html
