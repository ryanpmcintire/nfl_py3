# UI-32

## Goal
Key-number chip: on each row, say plainly when the line sits on or beside 3 or 7 and the chance the game lands exactly on the number (ROADMAP UI-32).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: The discrete margin read already gives push and cover chances; readers cannot see that a 3 is worth more than a 2.5. Fixes the existing row in place using the served discrete read output. Display only; no new fitted term.

## Tried
Novelty check only. Nearest: UI-31 (half-point chip), MOD-18 and LEAD-84 (push probability served), UI-22; grep 'key number' in UI rows found none.

## Next
First unit for one agent: Add a short chip when the line is within a point of 3, 7, 10 or 14 showing landing chance and which side it helps; render at 1440 and 390 and check it against the served cover chance. Data: Served discrete margin distribution (local).

## Open
Must read the served model's artifact, never a recomputation. Zero crossing closes nothing; one fitted probability selects the side.
