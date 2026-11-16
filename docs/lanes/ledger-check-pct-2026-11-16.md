# card-ledger-check percentage comparison (2026-11-16)

## Goal
`nfl-ats card-ledger-check` also compares the displayed cover chance (one decimal) per game across board, card file, pick-revision ledger.

## State
- src/nfl_ats/card_ledger_check.py (not pinned) now reports field displayed_cover_chance; anchor is the board value, other surfaces listed against it.
- Measured: run reports CLE 53.9/52.0, MIN 53.8/50.9, NE 53.5/52.6, DAL 51.4/50.1 (8 rows: card file and ledger each). Exit 1. pytest -k "ledger or card": 181 passed. ruff clean.

## Tried
- Ledger chance = new_home_cover_probability flipped to pick side; ledger only has flip-time rows so it mirrors the card defect.

## Next
- Root applies tests/scratch/board_pct pick_refresh patch; rerun the check, expect zero disagreements.

## Open
- Consumers: publish-predictions and refresh result add a warning only; scripts/refresh_lineup_forecast.py runs the check with check=False. capture_scheduler.py does not reference it.
