# UI-33

## Goal
Pick depends on a status: one plain line when a starter's unresolved designation could change the side (ROADMAP UI-33).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: PER-10's injury scenario mixture already gives the cover chance if a questionable starter plays or sits. Fixes the card and board in place with a line naming the player and the side if he sits. Display only; the served side remains the one fitted probability.

## Tried
Novelty check only. Nearest: PER-10 (mixture, done), UI-30 (changed since Tuesday), UI-22; grep 'if he sits' and 'scenario' in UI rows found none.

## Next
First unit for one agent: Add the line for games where the sit scenario flips the side or moves the cover chance past a stated amount from the scenario artifact; check at 1440 and 390. Data: PER-10 scenario output (local).

## Open
Reader text carries no ids or jargon; scenario weights must be keyed to the active model. Zero crossing closes nothing; one fitted probability selects the side.

## Update 2026-11-03 16:58 (agent)
Part 1 STOPPED: no served artifact stores per-scenario sides or sit probabilities. The PER-10 kernel (injury_scenarios.py) was deleted in b7ed31d and the producer was never built (done/injury-scenario-producer.md). UI-33 needs the producer first; next unit = build and store it keyed to the active model.
Part 2 DONE: attribution Net = sum of model adjustments from the market-line start, oriented to the pick. Before: "Net toward DEN -2.5 / -1.27 pts". After: "Our adjustments moved the forecast away from DEN versus the market line, by / 1.27 pts" (abs value; direction in words). Files: board_content.py ~2948, board_terminal.py ~1354. Tests 309 passed. Week 9 render not re-quoted (build_site not run).
