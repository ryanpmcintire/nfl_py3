"""Fixture tests for season mode (owner-approved improvement batch, item 4):
board rows for FINAL games (final score, cover result, a quiet row tint,
meters replaced by the outcome), the hero's running record strip, and every
required state -- all-upcoming (must render exactly as today), mixed week,
fully-graded week, pushes, and a Best Pick win/loss.

Every fixture here is built by ``dataclasses.replace`` off the shared
``_board_content_fixtures.build_fixture_content()`` 16-game board, so these
tests exercise the SAME renderer real weekly content flows through -- no
new content path exists just for tests.
"""

from __future__ import annotations

from dataclasses import replace

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import BoardContent, GameRow, SeasonRecordStrip


def _finalize(game: GameRow, cover_result: str, home_score: int, away_score: int) -> GameRow:
    return replace(
        game,
        final=True,
        cover_result=cover_result,
        final_score_text=f"{game.away} {away_score} at {game.home} {home_score}",
    )


def _with_games(content: BoardContent, updated: dict[str, GameRow]) -> BoardContent:
    games = tuple(updated.get(game.game_id, game) for game in content.games)
    return replace(content, games=games)


# ---------------------------------------------------------------------------
# all-upcoming -- today's rendering, unchanged
# ---------------------------------------------------------------------------


def test_all_upcoming_board_has_no_final_markup_or_hero_strip() -> None:
    content = build_fixture_content()
    assert content.season_record is None
    assert all(not game.final for game in content.games)

    html = board_terminal.render(content)
    # Checked as class-attribute USAGE (the trailing/leading quote), never a
    # loose substring -- every one of these names is also a CSS selector in
    # the page's own <style> block, which must not trip a false positive.
    assert 'class="season-record-strip"' not in html
    assert 'outcome-win"' not in html
    assert 'outcome-loss"' not in html
    assert 'outcome-push"' not in html
    assert 'final-win"' not in html
    assert 'final-loss"' not in html
    assert 'final-push"' not in html
    # Every game still shows its confidence meter, exactly like today.
    assert html.count('class="meter"') == len(content.games)


# ---------------------------------------------------------------------------
# mixed week -- some final, some upcoming
# ---------------------------------------------------------------------------


def test_mixed_week_renders_final_and_upcoming_rows_cleanly() -> None:
    content = build_fixture_content()
    win_game = content.games[0]
    loss_game = content.games[1]
    updated = {
        win_game.game_id: _finalize(win_game, "win", home_score=24, away_score=13),
        loss_game.game_id: _finalize(loss_game, "loss", home_score=20, away_score=21),
    }
    mixed = _with_games(content, updated)
    mixed = replace(
        mixed,
        season_record=SeasonRecordStrip(
            week_record_text="This week: 1-1 so far",
            season_record_text="Season to date: 1-1",
            best_pick_record_text=None,
        ),
    )

    html = board_terminal.render(mixed)
    assert 'class="season-record-strip"' in html
    assert "This week: 1-1 so far" in html
    assert 'final-win"' in html
    assert 'final-loss"' in html
    assert "outcome outcome-win" in html
    assert "outcome outcome-loss" in html
    # The 14 still-upcoming games keep their meter.
    assert html.count('class="meter"') == len(content.games) - 2
    assert "Covered" in html
    assert "No cover" in html


# ---------------------------------------------------------------------------
# fully-graded week
# ---------------------------------------------------------------------------


def test_fully_graded_week_replaces_every_meter_with_an_outcome() -> None:
    content = build_fixture_content()
    updated = {
        game.game_id: _finalize(
            game, "win" if index % 2 == 0 else "loss", home_score=24, away_score=13
        )
        for index, game in enumerate(content.games)
    }
    graded = _with_games(content, updated)
    graded = replace(
        graded,
        season_record=SeasonRecordStrip(
            week_record_text="This week: 8-8 so far",
            season_record_text="Season to date: 8-8",
            best_pick_record_text="Best Pick: 1-0",
        ),
    )

    html = board_terminal.render(graded)
    assert html.count('class="meter"') == 0
    # One "outcome" wrapper div per game -- ``class="outcome outcome-<result>"``,
    # distinct from its own nested ``outcome-score``/``outcome-word`` spans.
    assert html.count('class="outcome outcome-') == len(content.games)
    assert "Best Pick: 1-0" in html


# ---------------------------------------------------------------------------
# pushes
# ---------------------------------------------------------------------------


def test_push_renders_its_own_tint_and_word_not_win_or_loss() -> None:
    content = build_fixture_content()
    push_game = content.games[2]
    updated = {push_game.game_id: _finalize(push_game, "push", home_score=23, away_score=20)}
    pushed = _with_games(content, updated)
    pushed = replace(
        pushed,
        season_record=SeasonRecordStrip(
            week_record_text="This week: 0-0-1 so far",
            season_record_text="Season to date: 0-0-1",
            best_pick_record_text=None,
        ),
    )

    html = board_terminal.render(pushed)
    assert 'final-push"' in html
    assert 'outcome-push"' in html
    assert "Push" in html
    assert 'final-win"' not in html
    assert 'final-loss"' not in html


# ---------------------------------------------------------------------------
# Best Pick win and loss
# ---------------------------------------------------------------------------


def test_best_pick_win_renders_in_the_hero_strip() -> None:
    content = build_fixture_content()
    best = next(game for game in content.games if game.is_best)
    updated = {best.game_id: _finalize(best, "win", home_score=24, away_score=13)}
    won = _with_games(content, updated)
    won = replace(
        won,
        season_record=SeasonRecordStrip(
            week_record_text="This week: 1-0 so far",
            season_record_text="Season to date: 1-0",
            best_pick_record_text="Best Pick: 1-0",
        ),
    )
    html = board_terminal.render(won)
    assert "Best Pick: 1-0" in html
    assert 'final-win"' in html


def test_best_pick_loss_renders_in_the_hero_strip() -> None:
    content = build_fixture_content()
    best = next(game for game in content.games if game.is_best)
    updated = {best.game_id: _finalize(best, "loss", home_score=13, away_score=24)}
    lost = _with_games(content, updated)
    lost = replace(
        lost,
        season_record=SeasonRecordStrip(
            week_record_text="This week: 0-1 so far",
            season_record_text="Season to date: 0-1",
            best_pick_record_text="Best Pick: 0-1",
        ),
    )
    html = board_terminal.render(lost)
    assert "Best Pick: 0-1" in html
    assert 'final-loss"' in html
