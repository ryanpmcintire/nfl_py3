"""Fixture tests for the 2026-09-05 dashboard-improvement wave 2 (ROADMAP.md
UI-20, items (g)/(h)):

(g) the pool's tiebreaker guess on This Week, shown beside the market total
    it was built from, the blended total, and the implied margin
    (``board_content.TiebreakerView`` / ``board_terminal._tiebreaker_panel_html``);
(h) per-season and per-week opener-vs-close grading, side by side, on the
    History page (``board_site_content.SeasonGradeRow`` /
    ``board_site_content.HistoryWeekGrade`` /
    ``board_terminal._history_grading_section_html``).

These are pure-renderer tests over hand-built content objects (the same
discipline ``tests/test_public_board_wave1.py`` already uses) -- no real
artifact tree, so they are immune to a concurrent ``data/processed``
rewrite in the shared tree. Loader-level unit tests for
``board_content._load_tiebreaker_view`` live in ``tests/test_board_content.py``;
``board_site_content._season_grade_rows``/``_history_week_grades`` live in
``tests/test_board_site_content.py``.
"""

from __future__ import annotations

import re
from dataclasses import replace

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import (
    TIEBREAKER_NOT_PUBLISHED_TEXT,
    TIEBREAKER_NUDGE_NOTE,
    TiebreakerView,
)
from nfl_ats.board_site_content import (
    HISTORY_GRADE_CAPTION,
    HISTORY_WEEK_NOT_SETTLED_NOTE,
    NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE,
    NO_OPENER_LINE_ARCHIVED_SEASON_NOTE,
    HistoryPageContent,
    HistoryWeekGrade,
    SeasonGradeRow,
)

# ---------------------------------------------------------------------------
# (g) Tiebreaker guess
# ---------------------------------------------------------------------------


def test_tiebreaker_panel_shows_not_published_by_default() -> None:
    """The shared fixture never sets ``tiebreaker`` -- ``BoardContent``'s
    own default (not-published) must render, never an empty panel."""

    content = build_fixture_content()
    assert content.tiebreaker.recorded is False
    html = board_terminal.render(content)
    assert TIEBREAKER_NOT_PUBLISHED_TEXT in html
    assert "Tiebreaker guess</summary>" in html


def test_tiebreaker_panel_renders_a_real_guess() -> None:
    view = TiebreakerView(
        recorded=True,
        home_team="KC",
        away_team="DEN",
        market_total=43.0,
        blended_total=43.04,
        implied_margin=2.75,
        guess_home=22,
        guess_away=19,
        note=TIEBREAKER_NUDGE_NOTE,
    )
    content = replace(build_fixture_content(), tiebreaker=view)
    html = board_terminal.render(content)
    assert "DEN at KC" in html
    assert "market total 43" in html
    assert "blended total 43.04" in html
    assert "implied margin KC by 2.75" in html
    assert "guess KC 22 - DEN 19" in html
    # ``TIEBREAKER_NUDGE_NOTE`` contains an apostrophe that ``escape()``
    # turns into ``&#x27;`` on render -- check the escape-safe part.
    assert "market-anchored nudge, not a forecast" in html
    assert "MAE 10.42 vs 10.55" in html


def test_tiebreaker_panel_numbers_stay_collapsed_inside_details() -> None:
    """A real guess's numbers must not inflate This Week's default-visible
    content -- the same de-firehose discipline "Why this pick" already
    follows."""

    view = TiebreakerView(
        recorded=True,
        home_team="KC",
        away_team="DEN",
        market_total=43.0,
        blended_total=43.04,
        implied_margin=2.75,
        guess_home=22,
        guess_away=19,
        note=TIEBREAKER_NUDGE_NOTE,
    )
    content = replace(build_fixture_content(), tiebreaker=view)
    html = board_terminal.render(content)
    assert "43.04" in html
    stripped = re.sub(r"<details.*?</details>", "", html, flags=re.S)
    assert "43.04" not in stripped


def test_tiebreaker_panel_never_says_contains_zero_or_failed() -> None:
    for view in (
        build_fixture_content().tiebreaker,
        TiebreakerView(
            recorded=True,
            home_team="KC",
            away_team="DEN",
            market_total=43.0,
            blended_total=43.04,
            implied_margin=2.75,
            guess_home=22,
            guess_away=19,
            note=TIEBREAKER_NUDGE_NOTE,
        ),
    ):
        html = board_terminal.render(replace(build_fixture_content(), tiebreaker=view))
        assert "contains zero" not in html.lower()
        assert "failed" not in html.lower()


def test_tiebreaker_panel_omits_guess_score_when_not_supplied() -> None:
    view = TiebreakerView(
        recorded=True,
        home_team="SEA",
        away_team="NE",
        market_total=44.5,
        blended_total=44.6,
        implied_margin=-3.0,
        guess_home=None,
        guess_away=None,
        note=TIEBREAKER_NUDGE_NOTE,
    )
    content = replace(build_fixture_content(), tiebreaker=view)
    html = board_terminal.render(content)
    assert "implied margin NE by 3.00" in html
    assert ", guess" not in html


# ---------------------------------------------------------------------------
# (h) History: opener vs close, side by side
# ---------------------------------------------------------------------------


def _history_fixture(
    *,
    season_grades: tuple[SeasonGradeRow, ...] = (),
    week_grades: tuple[HistoryWeekGrade, ...] = (),
    grade_caption: str = "",
) -> HistoryPageContent:
    board = build_fixture_content()
    return HistoryPageContent(
        generated_at_text="2026-09-05 12:00:00 UTC",
        picks=(),
        primary_available=False,
        primary_error=None,
        challenger_assessments=(),
        ticker_chrome=board.ticker_chrome,
        link_preview=board.link_preview,
        season_grades=season_grades,
        week_grades=week_grades,
        grade_caption=grade_caption,
    )


def test_history_grading_section_absent_when_nothing_to_show() -> None:
    html = board_terminal.render_history_page(_history_fixture())
    assert "Opener vs close, side by side" not in html


def test_history_grading_section_renders_season_rows_and_caption() -> None:
    rows = (
        SeasonGradeRow(season_label="2024", games=272, opener_accuracy=0.545, close_accuracy=0.553),
        SeasonGradeRow(season_label="2025", games=272, opener_accuracy=0.532, close_accuracy=0.509),
    )
    html = board_terminal.render_history_page(
        _history_fixture(season_grades=rows, grade_caption=HISTORY_GRADE_CAPTION)
    )
    assert "Opener vs close, side by side" in html
    assert "54.5%" in html and "55.3%" in html
    assert "-0.8%" in html
    # ``HISTORY_GRADE_CAPTION`` contains apostrophes that ``escape()`` turns
    # into ``&#x27;`` on render -- check the escape-safe part.
    assert "close is the market at its sharpest" in html
    assert "settles picks at the OPENER" in html


def test_history_grading_section_season_gap_row_never_blank() -> None:
    rows = (
        SeasonGradeRow(
            season_label="Before the opener archive",
            games=538,
            opener_accuracy=None,
            close_accuracy=None,
            note=NO_OPENER_LINE_ARCHIVED_SEASON_NOTE,
        ),
        SeasonGradeRow(season_label="2025", games=272, opener_accuracy=0.53, close_accuracy=0.51),
    )
    html = board_terminal.render_history_page(_history_fixture(season_grades=rows))
    assert NO_OPENER_LINE_ARCHIVED_SEASON_NOTE in html
    assert 'colspan="3"' in html


def test_history_grading_section_renders_week_rows() -> None:
    rows = (
        HistoryWeekGrade(
            season=2026,
            week=1,
            picks=2,
            opener_settled=2,
            opener_wins=2,
            opener_accuracy=1.0,
            close_settled=1,
            close_wins=1,
            close_accuracy=1.0,
        ),
    )
    html = board_terminal.render_history_page(_history_fixture(week_grades=rows))
    assert "2026 / W1" in html
    assert "2-0 (100.0%)" in html
    assert "1-0 (100.0%)" in html


def test_history_grading_section_week_missing_close_never_blank() -> None:
    rows = (
        HistoryWeekGrade(
            season=2026,
            week=2,
            picks=1,
            opener_settled=1,
            opener_wins=0,
            opener_accuracy=0.0,
            close_settled=0,
            close_wins=0,
            close_accuracy=None,
            note=NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE,
        ),
    )
    html = board_terminal.render_history_page(_history_fixture(week_grades=rows))
    assert NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE in html


def test_history_grading_section_unplayed_week_says_not_yet_settled() -> None:
    rows = (
        HistoryWeekGrade(
            season=2026,
            week=3,
            picks=1,
            opener_settled=0,
            opener_wins=0,
            opener_accuracy=None,
            close_settled=0,
            close_wins=0,
            close_accuracy=None,
            note=HISTORY_WEEK_NOT_SETTLED_NOTE,
        ),
    )
    html = board_terminal.render_history_page(_history_fixture(week_grades=rows))
    assert HISTORY_WEEK_NOT_SETTLED_NOTE in html
    assert NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE not in html


def test_history_grading_section_never_says_contains_zero_or_failed() -> None:
    rows = (
        SeasonGradeRow(
            season_label="Before the opener archive",
            games=538,
            opener_accuracy=None,
            close_accuracy=None,
            note=NO_OPENER_LINE_ARCHIVED_SEASON_NOTE,
        ),
    )
    html = board_terminal.render_history_page(
        _history_fixture(season_grades=rows, grade_caption=HISTORY_GRADE_CAPTION)
    )
    assert "contains zero" not in html.lower()
    assert "failed" not in html.lower()
