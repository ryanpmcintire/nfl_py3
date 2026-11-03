# UI-37

## Goal
Rest and travel facts on each row: days of rest, short week, second road game in a row, time zones crossed (ROADMAP UI-37).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: A pool player checks rest and travel before trusting a side. Show plain facts only, with no effect claimed, as UI-27 did for the forecast. Fitted term: None: display only; no probability change.

## Tried
Novelty check only. Nearest: UI-27, UI-32, MOD-29. Differs: Shows rest and travel as facts on the row; MOD-29 studies rest as a term.

## Next
First unit for one agent: Add a compact chip to each unplayed row from the schedule (days of rest for each team, second road game, time zones crossed) and review at phone width.

## Open
Rules: fix in place in the current design system; reader text in plain pool words with no ids, hashes, model names or research jargon; every displayed percentage comes from an artifact keyed to the active model; no edge claim; publish-board and rendered-page diff reviewed by the orchestrator. Look count and family declared before any sign is seen. Data: nflverse schedules and venue table (local).
