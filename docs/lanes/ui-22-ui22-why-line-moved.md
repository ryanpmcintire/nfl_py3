# UI-22 Why the line moved

## Goal
One plain sentence per unplayed game whose line moved since the pool line, saying what captured news preceded the move, or "No captured news explains this move." ROADMAP row UI-22.

## State
Code written, not committed, not published. New src/nfl_ats/board_move_reason.py (build_move_reasons: window = Tuesday noon ET; move onset = first market capture whose consensus is at least half a point past the pool line in the final direction; news = beat posts with out/limited classes and injury-report revisions at or before onset; only players whose mean 2026 snap share is above 0.5). GameRow gained move_reason and move_reason_evidence; _with_move_reasons in board_content.py; _market_now_html in board_terminal.py appends the sentence in an existing game-sub span under Books now; board_site.py writes move_reasons.json (sentence plus evidence rows) beside the pages. Built to tests/scratch/ui-audit-2026-11-03/site6; render in final6 (index_390.png NOT yet looked at).
Week 9 result (measured): 9 games moved half a point (CIN-ATL, DEN-CAR, DAL-IND, NYG-PHI, DET-MIA, ARI-SEA, LV-SF, TB-CHI, BUF-MIN); all say "No captured news explains this move" because the only market capture after the pool line (12:01 ET Tuesday) is 12:10 ET and the newest beat post is 06:59 ET. Evidence rows in site6/move_reasons.json.

## Tried
Positive path (news found) was not exercised: a replay on week 8 with fabricated pool lines found no regular-player news for the test teams. Tests: a run during the broken-file moment passed exit 0 but the rerun after the fix was blocked by the call cap.

## Next
1. Run `.venv/Scripts/python.exe -m pytest tests -k "board or dashboard or findings" -q -p no:cacheprovider --ignore=tests/scratch` and fix any failure (test_terminal_stylesheet_class_set... uses only the existing game-sub class).
2. Look at final6/index_390.png; check the sentence wraps in the Books now cell.
3. Exercise the news path once with a real week-8 or later case; confirm the sentence wording.
4. Orchestrator publishes (publish-board) and pushes.

## Open
A reason implies cause; sentence says "Earlier captured news ... reported", not proven. Pool-line capture time is Tuesday noon, so early-week moves will show no-news until later captures exist. The snap-share greater than 0.5 definition of a regular is a majority rule, not fitted.

## Root 2026-11-03 14:18
- Not shipped as built: all 9 Week 9 "moves" were the pool's own half-point hook (pool line 12:01 v whole-number market 12:10), not market moves. Fix running: a move is market consensus v market consensus since the Tuesday market capture; the pool line is never one side; no sentence below the market's own quote granularity.

## Fix 2026-11-03 (market to market)
board_move_reason.py now takes (home, away, pick_team, market_now); Tuesday line = first market capture at/after Tuesday noon ET (mean across books); step = smallest gap between distinct posted book lines (0.5 measured); no sentence below one step. Week 9: no move (all deltas 0 or 0.25 over a 2 s window). Built site7, final7 index_390 looked: no sentence. Tests 338 passed. Remaining: existing "Books now" label still compares market to pool line (half a point toward); outside UI-22. Positive news path still unexercised.
