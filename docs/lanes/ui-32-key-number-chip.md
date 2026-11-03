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

## Unit 1 (2026-11-03 ~17:30 local)
Decision: rows and inspector had no key-number cue (UI-31 covers hooks only). Added one plain sentence to the inspector's Why this pick text (not the row): board_content.py `_key_number_note` (GameRow.key_number_note) reads the served discrete reader via `spread_explorer_three_way_probability` at the whole line beside/on the card line (push mass = chance of landing exactly there); key numbers are `KEY_LINE_ATOMS` (3, 7) from key_line_pick_read.py; board_terminal.py `_inspector_section` appends it. Only 3 and 7 are declared as a named constant, so 10/14/17 are not cued. JS layout only moves the Why node, never rewrites text. Week 9 has no whole-number 3 or 7 lines; half-point lines beside 3 (2.5, 3.5) and 7 (6.5) carry the note.
Next: confirm rendered text and tests; orchestrator regenerates and publishes.
- 2026-11-03 17:18 ET (root review): side wording was wrong for +3.5 and -3.5 picks (a 3-point margin covers a +3.5 underdog and fails a -3.5 favourite); fixed to 'on your side' iff pick_line x (|line| - key) > 0; checked on all four cases.
