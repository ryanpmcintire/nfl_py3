"""Fixture tests for the 2026-09-05 dashboard-improvement wave (ROADMAP.md
UI-20, items (a)/(b)/(c)):

(a) per-pick "Why this pick" explanations (``board_content.GameRow
    .explanation_text`` / ``board_terminal._why_this_pick_html``) -- UI-20
    layout A (2026-09-05) relocated these from a collapsed row under every
    board pick into that game's own inspector panel; see the section (a)
    tests below for the current contract;
(b) the "Research this week" section on the findings page
    (``findings_registry.recent_registry_activity`` /
    ``board_terminal._recent_activity_section_html``);
(c) the SOURCES panel's live-computed fallback when nothing was persisted
    for this forecast (``board_content._load_source_policy_view`` /
    ``board_terminal._source_policy_panel_html``).

These are pure-renderer tests over hand-built content objects (the same
discipline ``tests/_board_content_fixtures.py`` already uses) -- no real
artifact tree, so they are immune to the concurrent ``data/processed``
rewrite this session's other lanes are doing.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import (
    EXPLANATION_NOT_RECORDED_TEXT,
    SOURCE_POLICY_COMPUTED_LIVE_NOTE,
    SourcePolicyRow,
    SourcePolicyView,
    injury_pick_note,
)
from nfl_ats.board_site_content import (
    FindingsPageContent,
    HonestyRuleView,
    RecentActivityCategoryView,
    RecentActivityEntryView,
    RecentActivityView,
    SignalLedgerSummary,
)

# ---------------------------------------------------------------------------
# (a) "Why this pick"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["complete", "degraded", "blocked"])
def test_injury_state_is_visible_between_board_and_tiebreaker(state: str) -> None:
    sources = SourcePolicyView(
        state,
        "2026-09-05T20:00:00Z",
        (SourcePolicyRow("injuries_nflverse_timestamps", state, "2026-09-04T10:00:00Z", 60, ""),),
        True,
    )
    note = injury_pick_note(
        {"prediction_safety": {"checks_passed": ["injury_feature_presence"], "warnings": []}},
        sources,
    )
    content = replace(build_fixture_content(), source_policy=sources, injury_note=note)
    html = board_terminal.render(content)
    assert html.count(note) == 1
    assert html.index("</tbody></table>") < html.index(note) < html.index("Tiebreaker guess")


def test_lineup_legend_and_labels_match_playing_time_targets() -> None:
    from nfl_ats.lineup_view import team_lineup

    lineup = team_lineup(
        {
            "team": "LV",
            "players": [
                {
                    "name": "Fixture player",
                    "position": "WR",
                    "play_probability": 0.9,
                    "start_probability": 0.7,
                    "probability_reason": "Recent playing time",
                }
            ],
        }
    )
    content = build_fixture_content()
    dive = replace(content.dives[0], home_lineup=lineup, away_lineup=None)
    html = board_terminal._lineups_html(dive)
    assert "plays 90%" in html
    assert "starts 70%" in html
    assert 'title="Fills a starting slot by playing time"' in html
    assert (
        "plays = takes at least one snap; starts = fills a starting slot by playing time." in html
    )
    assert "Colour shows availability risk: green is low, amber is medium, red is high." in html


def _panel_chunks(html: str) -> dict[str, str]:
    """Split the This Week page's HTML into one chunk per inspector panel,
    keyed by game id -- each chunk runs from just after that panel's own
    opening ``id="..."`` through (but not including) the next panel's, so
    a hidden panel's own content can be checked in isolation without a
    full HTML parser."""

    marker = '<div class="dive-panel" id="'
    parts = html.split(marker)
    chunks: dict[str, str] = {}
    for part in parts[1:]:
        game_id, _, rest = part.partition('"')
        chunks[game_id] = rest
    return chunks


def test_why_this_pick_renders_once_per_game_hidden_unless_selected() -> None:
    """UI-20 layout A (2026-09-05, owner: "layout A is definitely the
    best. lets go with that."): the collapsed ``<details>`` this used to
    be, printed under EVERY board row, is retired -- the same explanation
    text now renders once per game, inside that game's own inspector panel
    (``board_terminal._why_this_pick_html``). Only the board's
    pre-selected game (the Best Pick) is visible without JavaScript or a
    URL hash; every other game's copy sits inside a ``hidden`` panel, the
    same "adds nothing to the default view" guarantee the old collapsed
    row gave, expressed through panel visibility instead of ``<details>``."""

    content = build_fixture_content()
    html = board_terminal.render(content)
    assert html.count("<b>Why this pick</b> &mdash;") == len(content.games)
    chunks = _panel_chunks(html)
    default_game_id = content.best_pick_game_id
    assert default_game_id is not None
    for game in content.games:
        chunk = chunks[game.game_id]
        assert "Why this pick</b> &mdash;" in chunk
        is_hidden = chunk.split(">", 1)[0].strip().endswith("hidden")
        assert is_hidden == (game.game_id != default_game_id)


def test_why_this_pick_shows_the_not_recorded_fallback_when_absent() -> None:
    """The shared fixture never sets ``explanation_text`` -- ``GameRow``'s
    own default (:data:`EXPLANATION_NOT_RECORDED_TEXT`) must render, never
    an empty disclosure."""

    content = build_fixture_content()
    assert all(game.explanation_text == EXPLANATION_NOT_RECORDED_TEXT for game in content.games)
    html = board_terminal.render(content)
    assert html.count(EXPLANATION_NOT_RECORDED_TEXT) == len(content.games)


def test_why_this_pick_renders_a_real_explanation_when_present() -> None:
    content = build_fixture_content()
    first, *rest = content.games
    explained = replace(
        first,
        explanation_text=(
            f"{first.away} at {first.home}: the market line used for this pick is +3. "
            "The model's own probability for this pick to cover is 61.2%; this is a "
            "single-game estimate, not the project's historical accuracy."
        ),
    )
    new_games = (explained, *rest)
    content = replace(content, games=new_games)
    html = board_terminal.render(content)
    assert "61.2%" in html
    assert explained.explanation_text.split(".")[0] in html


def test_why_this_pick_percentages_stay_hidden_unless_the_games_panel_is_selected() -> None:
    """A real explanation's percentages must not inflate the page's
    default-visible percentage count -- the same de-firehose discipline the
    rest of this board already follows (evidence chips, spread adjuster).
    UI-20 layout A (2026-09-05): the mechanism moved from a collapsed
    ``<details>`` to panel ``hidden``, but the guarantee is identical -- a
    game that isn't the board's current selection contributes nothing
    visible."""

    content = build_fixture_content()
    first, *rest = content.games
    assert first.game_id != content.best_pick_game_id  # not the pre-selected game
    explained = replace(
        first, explanation_text="The model's own probability for this pick is 61.2%."
    )
    content = replace(content, games=(explained, *rest))
    html = board_terminal.render(content)
    assert "61.2%" in html
    chunk = _panel_chunks(html)[first.game_id]
    assert "61.2%" in chunk
    assert chunk.split(">", 1)[0].strip().endswith("hidden")


def test_row_link_class_is_in_the_stylesheet_allowlist() -> None:
    """Regression guard for the mockup-class-set test (successor to the
    retired ``explain`` row class, UI-20 layout A 2026-09-05: the board's
    own rows, via ``.row-link``, replaced the old per-row ``<details>``
    disclosure as the reader's entry point): the additive
    ``row-link``/``is-selected``/``week-grid`` classes this restructure
    introduced must actually be reachable from ``board_terminal_style.css``,
    not just silently allowlisted."""

    html = board_terminal.render(build_fixture_content())
    assert 'class="row-link"' in html
    css = board_terminal.TERMINAL_STYLE_CSS
    assert ".row-link" in css
    assert ".is-selected" in css
    assert ".week-grid" in css


# ---------------------------------------------------------------------------
# (c) SOURCES panel -- computed-at-build-time fallback
# ---------------------------------------------------------------------------


def test_sources_panel_shows_live_note_when_computed_live() -> None:
    view = SourcePolicyView(
        card_state="degraded",
        evaluated_at="2026-09-05T14:00:00+00:00",
        rows=(
            SourcePolicyRow(
                source_id="odds_opener",
                state="degraded",
                observed_at=None,
                budget_minutes=180,
                reason="no snapshot present (budget 180 min)",
            ),
        ),
        recorded=False,
        computed_live=True,
    )
    html = board_terminal.render(replace(build_fixture_content(), source_policy=view))
    assert SOURCE_POLICY_COMPUTED_LIVE_NOTE in html
    assert '<span class="src-state degraded">DEGRADED</span></b>' in html
    assert "odds opener" in html


def test_sources_panel_omits_live_note_when_really_recorded() -> None:
    view = SourcePolicyView(
        card_state="complete",
        evaluated_at="2026-09-05T14:00:00+00:00",
        rows=(
            SourcePolicyRow(
                source_id="odds_opener",
                state="complete",
                observed_at="2026-09-05T13:30:00+00:00",
                budget_minutes=180,
                reason="snapshot is 30.0 min old, inside the 180 min budget",
            ),
        ),
        recorded=True,
        computed_live=False,
    )
    html = board_terminal.render(replace(build_fixture_content(), source_policy=view))
    assert SOURCE_POLICY_COMPUTED_LIVE_NOTE not in html


def test_sources_panel_not_recorded_placeholder_omits_live_note() -> None:
    html = board_terminal.render(build_fixture_content())  # default source_policy: not_recorded
    assert "No source-freshness block is recorded for this forecast" in html
    assert SOURCE_POLICY_COMPUTED_LIVE_NOTE not in html


# ---------------------------------------------------------------------------
# (b) "Research this week"
# ---------------------------------------------------------------------------


def _findings_fixture(recent_activity: RecentActivityView) -> FindingsPageContent:
    board = build_fixture_content()
    return FindingsPageContent(
        generated_at_text=board.generated_at_text,
        hero_tiles=(),
        groups=(),
        watching_leads=(),
        recent_activity=recent_activity,
        honesty_rules=(HonestyRuleView("Rule", "Body"),),
        ledger_summary=SignalLedgerSummary(
            total_signals=0, counts_by_status={}, counts_by_category={}, notable=()
        ),
        ticker_chrome=board.ticker_chrome,
        link_preview=board.link_preview,
    )


def test_recent_activity_section_renders_empty_window_correctly() -> None:
    activity = RecentActivityView(window_days=7, screened_count=0, resolved_count=0, categories=())
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "Research this week" in html
    assert "No new screens recorded this week." in html
    assert "0 screened" in html
    assert "0 resolved" in html


def test_recent_activity_section_renders_grouped_entries() -> None:
    activity = RecentActivityView(
        window_days=7,
        screened_count=2,
        resolved_count=1,
        categories=(
            RecentActivityCategoryView(
                category="onfield",
                entries=(
                    RecentActivityEntryView(
                        plain_summary="A plain-English summary of a fresh screen.",
                        effect_text="+0.40 accuracy points",
                        direction_sentence="Leans FOR the pattern described -- 70% confidence "
                        "in that direction (not yet resolved; see the interval).",
                        closed_label=None,
                    ),
                    RecentActivityEntryView(
                        plain_summary="A refuted mechanism, closed this week.",
                        effect_text="-1.50 accuracy points",
                        direction_sentence="Leans AGAINST the pattern described -- read this as "
                        "a lead for the OTHER side, 99% confidence in that direction (not yet "
                        "resolved; see the interval).",
                        closed_label="Resolved the other way",
                    ),
                ),
            ),
        ),
    )
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "2 screened" in html
    assert "1 resolved" in html
    assert "onfield (2)" in html
    assert "A plain-English summary of a fresh screen." in html
    assert "+0.40 accuracy points" in html
    assert "Leans FOR the pattern described" in html
    assert "Resolved the other way" in html
    assert "failed" not in html.lower()
    # The whole group is one collapsible unit (de-firehose discipline).
    assert '<details class="table-view">' in html


def test_recent_activity_section_never_says_contains_zero() -> None:
    activity = RecentActivityView(
        window_days=7,
        screened_count=1,
        resolved_count=0,
        categories=(
            RecentActivityCategoryView(
                category="market",
                entries=(
                    RecentActivityEntryView(
                        plain_summary="A small measured effect.",
                        effect_text="+0.10 accuracy points",
                        direction_sentence="Leans FOR the pattern described -- 55% confidence "
                        "in that direction (not yet resolved; see the interval).",
                        closed_label=None,
                    ),
                ),
            ),
        ),
    )
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "contains zero" not in html.lower()
