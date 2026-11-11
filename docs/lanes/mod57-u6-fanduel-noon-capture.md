# MOD-57 u6 - FanDuel noon capture
Stamped 2026-11-11 09:02 EST.

## Goal
Get FanDuel and DraftKings NFL spreads into raw captures near Tuesday noon ET to settle which book the pool copies.

## State
- Step 1 worked: Odds Gap payload already carried fanduel and draftkings in spread_data.all_books; scripts/capture_odds_gap_private.py filtered by LEADERS. Added CAPTURED_BOOKS = LEADERS + fanduel, draftkings and looped over it; book_scope string renamed. No other source needed; source_policies.json untouched.
- Live run: 14 games, 138 quotes, book_games bovada 14, draftkings 14, fanduel 14, mybookieag 13, williamhill_us 14. Snapshot data/market/raw/20261111T140135Z-odds-gap-private (source scan 13:01Z, 1h old).
- Loader (load_decision_quotes live): Wk10 FD vs DK vs median, no game off by over 1.5 (max 1.0, WAS_NYG FD 1.5 vs median 2.5). FD differs from DK on BUF_NYJ, SF_DAL, WAS_NYG.
- check_served_terms before and after: identical summary (14 games, 9 pool differs, no problems). sharp_book_movement_features filters to LEADERSHIP_WEIGHTS so leader input unchanged.
- ruff check and format clean. Scratch: tests/scratch/mod57/u6_compare.py.

## Tried
Public FanDuel endpoint not needed.

## Next
Orchestrator adds jobs (scheduler not edited): name capture_odds_gap_fanduel_noon, Tuesday 11:57 ET and 12:02 ET, argv: .tools/uv.exe run --project . --no-sync python scripts/capture_odds_gap_private.py. Caveat: the feed's source scan can lag up to 70 min (accepted limit), so check source_scan_at_utc per capture.

## Orchestrator decision (2026-11-11 09:04 ET, time from the capture output)
- The capture edit is REVERTED (git diff on scripts/capture_odds_gap_private.py is empty). Reason (read, src/nfl_ats/pool_hook.py:113-117): the served pool-line term anchors on the median across every book's last Tuesday quote, so FanDuel and DraftKings rows in the quote table would change that anchor's book set from next Tuesday without any test. One snapshot (20261111T140135Z) holds the five-book rows; a leaders-only capture at 14:03Z is now the newest, so the 30-minute current-line window no longer reads it.
- Not needed anyway (measured): every one of the 46 Odds Gap snapshots since 2026-09-20 saved the full response.json with all books. Unit 7 (lane mod57-u7-pool-source-from-payloads.md) parses those for Weeks 3-10. No scheduler job added.

## Open
Which book the pool copies: unit 7.
