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
- 2026-11-03 17:20 ET (root): REVERTED. The note read spread-explorer params, which load_board_content leaves empty when the active model uses calibrated mass (board_content.py ~4107: spread_explorer_params = {} if calibrated_mass), so it rendered on no game (measured: all 15 Week 9 notes empty; publish 'unchanged'). Next unit: find where the calibrated-mass lattice's per-game margin mass is stored for the active model (or computed in the served path) and read P(margin = key) from it; if it is not stored, the producer must store it keyed to the active model first.
- 2026-11-03 17:40 ET (unit 2): DONE in code, not published. Read path: the served calibrated sweep (served_sweep = calibrated_discrete_sweep(...) in load_board_content, board_content.py ~4005) carries push_probability at every alternative_line, so P(margin = key) is the push chance at line +/-key. New _key_number_note (board_content.py ~3670) appends one sentence to explanation_text (also for locked games) for lines within KEY_NUMBER_NOTE_REACH of KEY_LINE_ATOMS; side rule pick_line*(|line|-key)>0. Measured: 8 of 15 Week 9 games get a note (all -2.5/+3.5/-6.5), the other 7 sit away from 3 and 7. 309 tests pass. Next: root regenerates and publishes, reviews rendered page at 1440 and 390; consider 10 and 14 only if KEY_LINE_ATOMS is widened by its owner.
