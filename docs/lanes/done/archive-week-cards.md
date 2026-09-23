# Past-week card layout

## Goal
Make past weeks look and work like the current week's card.

## State
Complete. Weeks 1 and 2 share the current board, inspector, tabs, game room,
score scenarios, and receipts. Generated site pages are refreshed.

## Tried
Replaced the separate archive renderer with shared card rendering and scoped
interactions to the selected week. Measured: all 32 archived picks and probabilities
and all 16 current games are unchanged. Desktop and mobile interaction checks pass.

## Next
No follow-up required for this task.

## Open
Historical details without a saved source remain unavailable.

## Verification
- `uv run --no-sync pytest tests/test_board_terminal.py -q --basetemp .tmp/archive-review/pytest-terminal-final`: 88 passed.
- Mypy: passed for 238 source files.
- Ruff lint and format checks: passed.
- `node --check` for the three edited JavaScript files: passed.
- `uv run --no-sync nfl-ats publish-board`: succeeded.
- Headless Edge reviewed desktop/mobile rendering and exercised both archived weeks,
  selection restoration, keyboard navigation, game rooms, receipts, and score tools.

Commands used the locked repository environment through `.\.tools\uv.exe`.
