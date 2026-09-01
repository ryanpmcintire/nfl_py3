"""Content-coverage guarantee for the This Week page: every fact
:mod:`nfl_ats.board_content` produces on :class:`~nfl_ats.board_content
.BoardContent` actually reaches the rendered page.

Replaces the old cross-skin parity suite (``tests/test_board_skins.py``),
retired when the Cover Desk skin was dropped (2026-08-31 owner redirect --
"Let's drop the Desk theme altogether and just focus on the Terminal
theme."). The guarantee this suite protects is unchanged: a future number
change (a new experiment, a refreshed interval, updated findings) that
reaches ``board_content.py`` must be visible on the page ``board_terminal.py``
renders from it -- a change that lands in the content model but never
reaches the page is exactly the bug this suite exists to catch.

The companion dedup guarantee added by the 2026-08-31 redirect (each fact
lives on exactly ONE page, except the This Week headline strip, which is
deliberately also shown on The Model page) is covered in
``tests/test_board_terminal.py`` alongside the pages it concerns.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import BoardContent, HeadlineStats


def _render() -> tuple[str, BoardContent]:
    content = build_fixture_content()
    return board_terminal.render(content), content


def test_every_game_fact_appears_on_the_page() -> None:
    html, content = _render()
    for game in content.games:
        assert game.home in html, f"missing home team {game.home}"
        assert game.away in html, f"missing away team {game.away}"
        assert game.pick_team in html, f"missing pick team for {game.game_id}"
        assert game.pick_spread_text in html, (
            f"missing pick spread {game.pick_spread_text} for {game.game_id}"
        )
        assert game.probability_text in html, (
            f"missing probability {game.probability_text} for {game.game_id}"
        )


def test_headline_stat_values_and_intervals_appear_on_the_page() -> None:
    html, content = _render()
    headline = content.headline
    facts = [
        headline.played_card_value_text,
        headline.prior_chain_value_text,
        headline.raw_model_value_text,
        headline.close_grade_value_text,
        f"{headline.raw_model_ci[0]:.2f}%",
        f"{headline.raw_model_ci[1]:.2f}%",
    ]
    for fact in facts:
        assert fact in html, f"missing headline fact {fact!r}"


def test_model_id_appears_on_the_page() -> None:
    html, content = _render()
    assert content.headline.model_id is not None
    assert content.headline.model_id in html


def test_every_dives_attribution_net_label_appears_when_available() -> None:
    html, content = _render()
    for dive in content.dives:
        if not dive.attribution.available:
            continue
        assert dive.attribution.net_label is not None
        assert dive.attribution.net_label in html


def test_cover_curve_offset_zero_note_appears_when_present() -> None:
    html, content = _render()
    for dive in content.dives:
        if dive.cover_curve_offset_zero_note is not None:
            assert escape(dive.cover_curve_offset_zero_note) in html


def test_selection_caveat_sentence_appears_on_the_page() -> None:
    html, content = _render()
    assert escape(content.headline.selection_caveat_text) in html


def test_disclaimer_appears_on_the_page() -> None:
    html, content = _render()
    assert escape(content.disclaimer.short) in html
    assert escape(content.disclaimer.full) in html


def test_findings_text_appears_on_the_page() -> None:
    html, content = _render()
    for finding in content.findings:
        assert escape(finding.text) in html, f"missing finding {finding.tag}"


def test_best_pick_note_appears_on_the_page() -> None:
    html, content = _render()
    assert escape(content.best_pick_note) in html


def test_policy_composition_or_narrative_appears_on_the_page() -> None:
    html, content = _render()
    policy = content.policy
    expected = policy.rich_narrative if policy.rich_narrative else policy.composition_text
    assert escape(expected) in html


def test_a_content_only_number_change_reaches_the_page() -> None:
    """The guarantee this whole suite exists for: change ONE field on the
    content model and the rendered page changes with it."""

    content = build_fixture_content()
    changed_headline = replace(content.headline, raw_model_value_text="61.9%")
    changed_content = replace(content, headline=changed_headline)
    assert isinstance(changed_headline, HeadlineStats)

    html = board_terminal.render(changed_content)
    assert "61.9%" in html


def test_every_dive_matchup_label_appears_on_the_page() -> None:
    html, content = _render()
    for dive in content.dives:
        assert dive.pick_team in html
        assert dive.home in html


def test_flip_pill_text_appears_for_every_flipped_game() -> None:
    """Owner-approved improvement batch, item 1: a flip pill's exact
    content-layer text (glyph + member name(s)) must reach the page for
    EVERY flipped game, not just the one the fixture happens to exercise."""

    html, content = _render()
    for game in content.games:
        if game.flip_member_labels:
            assert game.flip_pill_text in html


def test_flip_note_appears_for_every_flipped_dive() -> None:
    html, content = _render()
    for dive in content.dives:
        if dive.flip_note:
            assert escape(dive.flip_note) in html


def test_prospective_scoreboard_text_appears_on_the_page() -> None:
    html, content = _render()
    scoreboard = content.headline.prospective_scoreboard
    assert escape(scoreboard.headline_text) in html
    if scoreboard.detail_text is not None:
        assert escape(scoreboard.detail_text) in html


def test_ticker_chrome_games_appear_via_the_ticker() -> None:
    html, content = _render()
    for game in content.ticker_chrome.games:
        assert game.ticker_text in html


def test_link_preview_title_and_description_appear_on_the_page() -> None:
    html, content = _render()
    assert escape(content.link_preview.title) in html
    assert escape(content.link_preview.description) in html


def test_a_content_only_flip_member_label_change_reaches_the_page() -> None:
    """The same content-coverage guarantee as
    ``test_a_content_only_number_change_reaches_the_page``, exercised for
    the flip pill: change ONE game's flip labels and the rendered page
    changes with it."""

    content = build_fixture_content()
    flipped_index = next(
        index for index, game in enumerate(content.games) if game.flip_member_labels
    )
    changed_game = replace(content.games[flipped_index], flip_member_labels=("division revenge",))
    changed_games = list(content.games)
    changed_games[flipped_index] = changed_game
    changed_content = replace(content, games=tuple(changed_games))

    html = board_terminal.render(changed_content)
    assert "⇄ division revenge" in html
