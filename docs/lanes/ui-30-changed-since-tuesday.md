# UI-30

## Goal
Changed-since-Tuesday strip: games whose side or number moved since the Tuesday card, in plain words (ROADMAP UI-30).

## State
Both declared pieces are built; the number-moved piece is published to `docs/` locally, not committed or pushed (2026-11-14).

- Side and Best Pick changes (unit 1, 2026-11-03): `board_terminal.py` `_changed_strip_html`, data from `content.week_changes` (`board_content.py` `_build_week_changes`; archived weeks via `board_archive.py` `_archived_changes`).
- Number moved, side unchanged (unit 2, 2026-11-14), display only, no fitted term:
  - `board_move_reason.py`: `MoveReason.toward_pick_points` (signed, positive toward our pick), `move_size_words`, `_toward_pick_points`; `_move_text` now calls both, wording unchanged.
  - `board_content.py`: `GameRow.market_move_toward_pick`, set in `_with_move_reasons`; new `NumberMoveRow` (its `sentence` property is the reader line); `WeekChangesPanel.number_moves`, filled in `_build_week_changes`, largest move first.
  - `board_terminal.py` `_changed_strip_html`: appends a lead line and one line per moved game; says "No pick has changed." when only numbers moved.
- Definition of "moved" is the existing one behind the per-row "The line moved ..." sentence in the Books now cell (`board_move_reason.py` `build_move_reasons`): the row's current market number (`GameRow.market_now`, the Books now value) differs from the first market consensus captured at or after Tuesday noon Eastern by at least `quote_step`, the smallest gap between quoted lines in the captured quotes. No new constant or threshold; the strip and the row cannot disagree.
- A game gets a number line only if it is on the Tuesday card, its side is unchanged, the picks table is not behind the ledger for it, it is unplayed, and the row carries a move. Side-changed games keep their existing line only.
- Rendered for the live week (7 moved games; CAR at NO and MIN at GB changed side, so 5 lines):
  - Same pick, new number: the market has moved on 5 games since Tuesday, biggest move first.
  - SEA at LV: still SEA -3.5 in the pool. The market has moved 3 points toward SEA since Tuesday; books now have SEA -6.5.
  - MIA at IND: still IND -2.5 in the pool. The market has moved 1.5 points toward IND since Tuesday; books now have IND -4.5 to -3.5.
  - LAC at BAL: still LAC +5.5 in the pool. The market has moved 1 point toward LAC since Tuesday; books now have LAC +4.5.
  - KC at ATL: still ATL +3.5 in the pool. The market has moved half a point toward ATL since Tuesday; books now have ATL +3.
  - BUF at NYJ: still NYJ +7.5 in the pool. The market has moved half a point toward NYJ since Tuesday; books now have NYJ +7.5 to +8.5.
- Hand checks against the row's own cells: SEA at LV (Books now SEA -6.5, hover "pool gives SEA -3.5, 3 points better", row sentence "moved 3 points toward SEA") and KC at ATL (Books now ATL +3, "pool gives ATL +3.5, half a point better", "moved half a point toward ATL"). Both agree.
- Verified 2026-11-14: ruff check and format clean on the three files; mypy shows no error on touched lines; `tests/test_public_board.py` 134 passed; `publish-board` exit 0.

## Tried
- Novelty check (2026-11-03): nearest UI-22 (why the line moved), UI-26.
- Rejected a fixed half-point constant: the row already defines a move through the measured `quote_step`, so that definition is reused.
- No line states an effect on cover chance; no served number for that exists on the row.

## Next
1. Orchestrator: review the diff, look at the strip at 1440 and 390 widths (never rendered at 390, CSS read only), commit and push.
2. No "away from" line has rendered yet (all five live moves are toward the pick); read the first one that appears.
3. Optional: when a week has many moved games the strip grows by one line each; decide a collapse only if a real week shows it crowding the picks table.

## Open
- The Tuesday market number is the first captured consensus after Tuesday noon Eastern, which is not always the pool's own number; the line states the pool number and the books' number separately for that reason.
- A game whose Best Pick status changed and whose number also moved gets two lines (one each). None in the live week.
