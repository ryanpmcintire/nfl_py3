# Week 3 dashboard completeness

## Goal
Bring the current week's dashboard up to date, fill available game details, and keep Week 1 and Week 2 in the shared card layout.

## State
Completed. Measured: the regenerated Week 3 page has lineups for all 32 teams, explanations for all 16 games, and the final-game tiebreaker. Missing current book lines explicitly say "No quote" and explain that the card uses pool lines. Picks and cover chances are unchanged.

## Tried
Rebuilt current-week lineups, published the active forecast and all four site pages, and reviewed the rendered desktop/mobile diff. Measured: all 48 current/archive games retained their picks; only Week 3 lineup and explanation payloads changed. Week switching, game-room controls, receipts, keyboard navigation, and mobile overflow checks passed for Weeks 1–3. Format, Ruff, and mypy passed. Full suite: 4,528 passed, 9 skipped, one style-contract failure; fixed that class and reran all 88 board tests successfully.

## Next
Resume source recovery in the existing odds-source lanes; retain the shared current/archive layout.

## Open
Current public book quotes remain unavailable; Week 3 injury reports are not published yet. Publication reports degraded odds-refresh and transaction freshness. Why the prior active forecast lacked publication artifacts remains unconfirmed; this task repaired those artifacts without replacing recorded decisions.
