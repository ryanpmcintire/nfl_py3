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
    RecentActivityEntryView,
    RecentActivityView,
    SignalLedgerSummary,
)


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

    marker = '<div class="dive-panel" id="'
    parts = html.split(marker)
    chunks: dict[str, str] = {}
    for part in parts[1:]:
        game_id, _, rest = part.partition('"')
        chunks[game_id] = rest
    return chunks


def test_why_this_pick_renders_once_per_game_hidden_unless_selected() -> None:

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

    content = build_fixture_content()
    first, *rest = content.games
    assert first.game_id != content.best_pick_game_id
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

    html = board_terminal.render(build_fixture_content())
    assert 'class="row-link"' in html
    css = board_terminal.TERMINAL_STYLE_CSS
    assert ".row-link" in css
    assert ".is-selected" in css
    assert ".week-grid" in css


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
    html = board_terminal.render(build_fixture_content())
    assert "No source-freshness block is recorded for this forecast" in html
    assert SOURCE_POLICY_COMPUTED_LIVE_NOTE not in html


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
    activity = RecentActivityView(
        window_days=7, screened_count=0, resolved_count=0, still_open_count=0, entries=()
    )
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "Research this week" in html
    assert "No new screens recorded this week." in html
    assert "0 signals looked at" in html
    assert "0 resolved either way" in html
    assert "0 still open" in html


def test_recent_activity_section_renders_up_to_eight_entries() -> None:
    activity = RecentActivityView(
        window_days=7,
        screened_count=2,
        resolved_count=1,
        still_open_count=1,
        entries=(
            RecentActivityEntryView(
                plain_summary="A plain-English summary of a fresh screen.",
                effect_text="+0.40 accuracy points",
                chance_it_helps_text="chance it helps: 70%",
                closed_label=None,
            ),
            RecentActivityEntryView(
                plain_summary="A refuted mechanism, closed this week.",
                effect_text="-1.50 accuracy points",
                chance_it_helps_text="chance it helps: 1%",
                closed_label="Resolved the other way",
            ),
        ),
    )
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "2 signals looked at" in html
    assert "1 resolved either way" in html
    assert "1 still open" in html
    assert "A plain-English summary of a fresh screen." in html
    assert "+0.40 accuracy points" in html
    assert "chance it helps: 70%" in html
    assert "Resolved the other way" in html
    assert "failed" not in html.lower()


def test_recent_activity_section_never_says_contains_zero() -> None:
    activity = RecentActivityView(
        window_days=7,
        screened_count=1,
        resolved_count=0,
        still_open_count=1,
        entries=(
            RecentActivityEntryView(
                plain_summary="A small measured effect.",
                effect_text="+0.10 accuracy points",
                chance_it_helps_text="chance it helps: 55%",
                closed_label=None,
            ),
        ),
    )
    html = board_terminal.render_findings_page(_findings_fixture(activity))
    assert "contains zero" not in html.lower()
