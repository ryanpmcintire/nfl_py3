# ENG-62 Monday capture writes the coming week

## Goal
odds_private_mon (Mon 12:00 ET) stores the coming week's Odds Gap leader quotes (bovada, williamhill_us, mybookieag), not only the Monday-night game. Also confirm the thin-move check counts current consensus books.

## State (2026-11-21 09:57 EST)
- Claim 1 true (measured). The Monday Odds Gap snapshots 20261102T155146Z, 20261109T173453Z, 20261116T172832Z each hold 1 game, the ending week's Monday-night game. The old week pick took only the week of the nearest kickoff.
- Fix: scripts/capture_odds_gap_private.py:15 and :40-59. The capture keeps the nearest upcoming week plus the first week with a game whose pick deadline is still open. The deadline comes from pick_refresh.pick_deadline and sunday_pick_lock (min of kickoff and Sun 16:00 ET); no new constants. Tue-Sun 16:00 ET is unchanged, and after the Sunday lock the next week is added. ruff check and format are clean.
- In-memory re-parse of stored payloads, nothing written (measured): Mon 11-16 goes from 1 game to 14 (wk10 1, wk11 13; williamhill_us 14, mybookieag 10, bovada 9). Mon 11-09 goes from 1 to 15 (wk9 1, wk10 14). Sun 11-15 15:20 ET stays at 5 and Fri 11-20 stays at 12, so it is unchanged before the lock. Script: scratchpad reparse_inmem.py, not in the repo.
- Claim 2 already fixed by 0f50ada2 (2026-11-16, read): books = max(leader books with a baseline, distinct LEADERSHIP_WEIGHTS books quoting the game by now). The check_served_terms.py run exited 0 with problems []. thin_move_games had one row: CIN at WAS, books 5, leader_books_with_baseline 2, move 0.5. That is no alarm.

## Tried
- `capture_scheduler.py --run-job odds_private_mon --dry` exit 0, OK. **--dry does not stop the capture** (read, capture_scheduler.py:3497-3509: dry_command only strips recording flags). It wrote live snapshots 20261121T145632Z-odds-gap-private (12 wk11 games) and 20261121T145636Z (bovada, 12 wk11 games) at Sat 09:56 ET. These are forward live quotes, not a back-fill, and they are the same kind as odds_private_sat 10:00, which now skips under the 30-minute age guard (inferred). Nothing was deleted.

## Next
1. Root: the first live proof is odds_private_mon on Mon 11-23 12:00 ET. Expect about 14 games (wk11 MNF + wk12) in the *-odds-gap-private snapshot.
2. Root, scheduler owner: give the odds_private_* jobs a real dry mode. Proposed diff: add `--dry` to capture_private_sunday_odds.py and capture_odds_gap_private.py so they parse and print without write_market_snapshot. In capture_scheduler.py dry_command (:3501-3508), add `capture_private_sunday_odds.py` to the name set, with `--dry` appended for it instead of `--dry-run`.

## Open
- Mixed-week snapshot week tag: request week = features.iloc[0] week (the earlier week), the same as Bovada's existing mixed snapshots. The served move term ignores the tag (pick_probability load_decision_quotes has no schedule). CLV build_pairing_table with a schedule (clv.py:490-507, 685-702) drops the coming-week rows from a Monday snapshot, which is no loss against today (inferred).
- The move term reads quotes observed at or after the game week's Monday (sharp_book_movement_features.py observed_at_utc.ge(_monday)), so the Monday quote becomes a pre-Wednesday baseline for a book with no Tuesday quote (inferred).

ROADMAP ENG-62 update: 2026-11-21: the Monday capture now keeps the coming week once the ending week's pool deadline has passed (capture_odds_gap_private.py:40-59). A re-parse of the stored 11-16 Monday payload goes from 1 game to 14; Tue-Sun captures are unchanged. Live proof is Mon 11-23 12:00. The thin-move check was already fixed 11-16 (0f50ada2). Open: the scheduler --dry on odds_private_* jobs still captures live. Lane docs/lanes/eng62-monday-capture-2026-11-21.md.
