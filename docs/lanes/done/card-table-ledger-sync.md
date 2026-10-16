# card-table-ledger-sync

## Goal
Card main table, refresh block and board render the same latest picks and cover chances.

## State
Fixed in src/nfl_ats/pick_refresh.py: append_refresh_to_card now rewrites main-table rows (pick, line, cover chance) via _sync_card_table_rows from _served_side_rows(changed_only=False). Verified on a scratch copy of the card: CLE +4.5 57.3%, CAR +1.5 58.4%, ATL 51.1%, LAC 58.7%, DAL 53.9%.

## Tried
Root cause: append_refresh_to_card only replaced the LATE_WEEK_REFRESH block; the table is written only by publish-predictions.

## Next
Best Pick source (decided): Tuesday paper-ledger nominee (HOU at JAX) is held until the Sunday-pass renomination (served_best_pick, best_pick_after); a starred game past its deadline stays (locked_best_pick). Cause of split: calibrated branch of card_view.resolve_nomination ignored the held/renominated id and re-picked the fresh served-probability leader at render time (card ATL at Friday publish, board LAC). Fixed in card_view.py: tuesday_best_pick, apply_held_best_pick, held_game_id wired in resolve_card_view. Verified: publishing._publication_context and load_board_content both star 2026_06_HOU_JAX. ATL 61.9% -> 51.1%: refresh shows movement_delta 2.5, consensus_delta 2.5, late_week_net_move 0.0, 3 books, no flips (inferred: the movement term, not a side change).

## Open
Orchestrator: run publish-predictions and publish-board to move the card and site stars to HOU; not run here. ATL 61.9 vs 51.1 term split not isolated.
