# UI-36

## Goal
History record by favourite and underdog and by home and road in plain words (ROADMAP UI-36).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: A pool player asks whether the card does better on favourites or dogs, at home or on the road. Display the played-card record in four cells with the coin-flip range beside it so a small lopsided split is not read as a trend. Fitted term: None: display only, from the played-card archive; no edge claim.

## Tried
Novelty check only. Nearest: UI-28, UI-29, UI-35. Differs: Splits the played card by side type; the existing strips split by week or leader.

## Next
First unit for one agent: Add a four-cell table to the History view (side we took: favourite or underdog, home or road) with games, covers and the coin-flip range, and review the rendered page at phone width.

## Open
Rules: fix in place in the current design system; reader text in plain pool words with no ids, hashes, model names or research jargon; every displayed percentage comes from an artifact keyed to the active model; no edge claim; publish-board and rendered-page diff reviewed by the orchestrator. Look count and family declared before any sign is seen. Data: Played-card archive and scorecard (local).

## Update 2026-11-03 18:15
Unit 1 done: board_site_content.py (_side_type_rows, _coin_flip_range, COIN_FLIP_RANGE_LEVEL, SeasonSoFar.side_type_rows) and board_terminal.py (_side_type_html fold after calibration). Rendered rows: fav 31-31 of 62, dog 34-25 of 59, home 32-33 of 65, road 33-23 of 56, each with coin range. Phone-width review and publish left to orchestrator.
