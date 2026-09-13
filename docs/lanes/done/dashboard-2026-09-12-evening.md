# dashboard-2026-09-12-evening

## Goal

This session's binding dashboard improvement (AGENTS.md, dashboard every
session): one reader-facing change, `publish-board`, push.

## State

Survey done (opencode lane B, `big-pickle`, 2026-09-12 20:00 ET): three
candidates in `tests/scratch/lanes/board_dashboard_proposals.md`. Chosen:
candidate 2, a visible market-move phrase under the Books-now number
("half a point against NE"), because candidate 1 (a rendered kickoff
countdown) would go stale on a static page (AGENTS.md: no number on the
site may go stale). Built by opencode lane E (`mimo-v2.5-free`), verified
(measured 20:13 ET: 108 board tests pass) and published 20:30 ET with
`nfl-ats publish-board`: `GameRow.market_move_label`
(`src/nfl_ats/board_content.py`), the `market-move` span in
`board_terminal.py`, two CSS rules. Rendered diff: every Books-now cell
gains one sub-line ("half a point against NO" in red, "2.5 points toward
PIT", "unchanged"). Committed and pushed with the 2026-09-12 evening lanes.

## Tried

- big-pickle wrote its report under a different name and tried to write
  to `C:\Repos\...` (wrong drive); the report was still recoverable from
  `tests/scratch/lanes/board_dashboard_proposals.md`.

## Next

(none; finished)

## Open

(none)
