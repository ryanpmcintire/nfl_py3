"""Fixture tests for the 2026-09-05 dashboard-improvement wave (ROADMAP.md
UI-20, items (a)/(b)/(c)):

(a) per-pick "Why this pick" explanations, collapsed under each This Week
    pick row (``board_content.GameRow.explanation_text`` /
    ``board_terminal._why_this_pick_row_html``);
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

import re
from dataclasses import replace

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import (
    EXPLANATION_NOT_RECORDED_TEXT,
    SOURCE_POLICY_COMPUTED_LIVE_NOTE,
    SourcePolicyRow,
    SourcePolicyView,
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


def test_why_this_pick_renders_under_every_game_row_collapsed_by_default() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    assert html.count('<tr class="explain"') == len(content.games)
    assert html.count(">Why this pick</summary>") == len(content.games)
    # Collapsed by default -- no <details open>.
    for match in re.finditer(r'<tr class="explain"[^>]*>.*?</tr>', html, flags=re.S):
        assert "<details open" not in match.group(0)


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
            "single-game estimate, not the project's historical accuracy. This is a "
            "descriptive research summary, not a wagering recommendation."
        ),
    )
    new_games = (explained, *rest)
    content = replace(content, games=new_games)
    html = board_terminal.render(content)
    assert "61.2%" in html
    assert explained.explanation_text.split(".")[0] in html


def test_why_this_pick_percentages_stay_collapsed_inside_details() -> None:
    """A real explanation's percentages must not inflate the page's
    default-visible percentage count -- the same de-firehose discipline the
    rest of this board already follows (evidence chips, spread adjuster)."""

    content = build_fixture_content()
    first, *rest = content.games
    explained = replace(
        first, explanation_text="The model's own probability for this pick is 61.2%."
    )
    content = replace(content, games=(explained, *rest))
    html = board_terminal.render(content)
    assert "61.2%" in html
    stripped = re.sub(r"<details.*?</details>", "", html, flags=re.S)
    assert "61.2%" not in stripped


def test_why_this_pick_row_is_in_the_stylesheet_allowlist() -> None:
    """Regression guard for the mockup-class-set test: the additive
    ``explain`` row class must actually be reachable from
    ``board_terminal_style.css`` review, not just silently allowlisted."""

    html = board_terminal.render(build_fixture_content())
    assert 'class="explain"' in html


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
    assert "odds_opener" in html


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
