# UI-19 season-record status

## Goal

Resolve whether UI-19 meets its acceptance criterion without rerunning a forecast or publication.

## State

- Complete.
- **Read:** `ROADMAP.md` requires a settled-week History answer to cite the live season strip first and
  the archive second, with the behavior pinned by fixture checks.
- **Read:** `board_assistant.py:1062-1076` builds the live record first and then labels the played-card
  value as the archive. `board_assistant.py:1124-1139` uses that answer for History.
- **Read:** `board_terminal.py:2272-2318` renders the live card record, finished-game count, Best Pick,
  and challenger assessments in `This season so far`.
- **Measured:** The locked read-only probe exited 0. The active manifest selects the 2026 Week 3
  forecast, settlement has 33 finished games, and the served ledger has 48 rows across Weeks 1-3.
  Saved `docs/history.html` renders 24-9 over 33 finished games and a 1-1 Best Pick record.
- **Read:** `week1-covers-settled.md` records the first fully settled week at 9-7 across reader surfaces.

## Tried

- A scoped lane search found no dedicated UI-19 lane. The only lane reference was the unverified triage
  note in `backlog-triage-2026-09-22.md`.
- The active manifest identifies the current forecast. Fixed owner paths resolve the settlement results
  and served decision ledger used by the live season block.

## Next

- The ROADMAP row now records completion and links this evidence.

## Open

- None for UI-19. Future settlements refresh the displayed record but do not block acceptance.
