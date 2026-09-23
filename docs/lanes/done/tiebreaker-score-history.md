# Tiebreaker score and season history

## Goal
Keep the displayed score consistent with the projected winner, distinguish the spread handicap from the projected margin, and include both completed weeks in season totals accuracy.

## State
Complete. The published Week 3 score is CHI 23–PHI 22, with a projected Chicago margin of 0.9 points. Chicago +3.5 is the spread handicap. The comparison through Week 2 includes two pregame guesses: mean absolute total error 7.0 points versus the market's 8.0. ATS picks and probabilities are unchanged.

## Tried
Preserved the projected winner when selecting integer scores. Replaced the most-likely-score claim with a rounding explanation. Recovered the eligible Week 1 and Week 2 published guesses and added immutable publication history; grading excludes post-deadline guesses and compares the same games for both totals. Regenerated with `publish-predictions --with-board` and reviewed the rendered desktop and phone panels, with no horizontal overflow. Verification: 4,529 tests passed, 9 skipped; Ruff format, Ruff check, and mypy passed.

## Next
None for this bounded fix. Continue other dashboard backlog work in a fresh thread.

## Open
No unresolved issues for this fix. Published source history remains under ignored `artifacts/published/tiebreakers/`; generation and grading logic is in `src/nfl_ats/tiebreaker_history.py`.
