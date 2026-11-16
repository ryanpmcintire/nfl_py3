# served-terms thin-move check

## Goal
Judge a thin line move by how many current books quote the game, not only by fitted leader books with a Tuesday baseline.

## State
scripts/check_served_terms.py `_thin_move_games` now sets `books` to the larger of leader books with a baseline and distinct leadership-weight books quoting the game by the check time. The raw baseline count is kept as `leader_books_with_baseline`. The alert still fires at `books < ANCHOR_MINIMUM_BOOKS` (existing constant). No new constants.
Week 10 replay (as-of 2026-11-15T13:00Z): MIN at GB books 1 -> 6, no problem. ruff format and check clean.

## Tried
Replays for Weeks 6-9 at several as-of times: `thin_move_games` empty, so no real one-book game exists in stored data to show still flagging.

## Next
Served move itself still uses the median of leader books with a baseline (pick_probability.market_move_toward_home, available when leader_books > 0); that is unchanged and is a separate question.

## Open
- Whether a one-book served move should be damped is not decided here.
