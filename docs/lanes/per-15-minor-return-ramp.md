# PER-15

## Goal
Return from a short absence: snap fraction and efficiency in the first game back after one or two missed starts, non-quarterback starters (ROADMAP PER-15).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: A starter returning from a one or two game absence plays a reduced share and fewer snaps at full speed, which the Friday designation does not describe. Fitted term: expected snap-fraction loss times position value in the existing availability input, replacing a flat active flag.

## Tried
Novelty check only. Nearest: PER-12 (quarterback return, three or more missed), PER-11 (acquired players), LEAD-106 (language read of status), MOD-41 (workload); grep 'return from' found only quarterbacks.

## Next
First unit for one agent: Measure first-game-back snap fraction versus the player's trailing mean by position group for 1, 2 and 3+ missed games; cluster bootstrap by team-season; pass to the margin ridge only if it adds out of season. Data: Snap counts 2012-2025 and weekly injury status (free).

## Open
Overlaps LEAD-106 if its read already carries a ramp; check after its rescore. Zero crossing closes nothing; one fitted probability selects the side.
