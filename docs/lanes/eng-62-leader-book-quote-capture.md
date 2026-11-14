# ENG-62 leader-book quote capture

## Goal
bovada, williamhill_us (Caesars), mybookieag spread quotes before Wed 00:00 ET and after, every week, on every game, so the served move term sees the three books it was fitted on, at the quote age it was fitted on.

## State
Path (read): capture_private_sunday_odds.py -> capture_odds_gap_private.py (LEADERS :21, week pick :36-42) -> data/market/raw/*-odds-gap-private -> pick_probability.py:748-772 -> sharp_book_movement_features.py (:115 quote time is the feed scan stamp, :139 baseline is each book's last quote before Wed plus a later one). Offset term: pool_hook.py:85-118, anchor is the median across books of each book's last Tuesday quote (:113-117), dated by observation time (:96).
Earlier weeks (measured 2026-11-08): Wk2 and Wk5 zero books (no pre-Wed capture); Wk9 Caesars absent from the feed Mon-Tue.

Week 10 (measured 2026-11-14 15:04 EST; artifacts/eng62/week10_leader_depth.json): served term 11 games at 3 books, 2 at 2, 1 at 1, fixed after Wednesday. MIN_GB rests on Caesars alone, move -1.5 against a mid-swing Tuesday baseline (3.0 Mon, 5.5 Tue 12:11, 4.5 Tue 14:26, 3.0 now); LAC_BAL 2 books -0.5; LA_ARI 2 books -0.25. Tuesday holes were bovada (13 of 14) and mybookieag (11 of 14). odds_private_mon wrote only the Week 9 Monday game (:36-42 takes the week of the nearest kickoff). Caesars has no other source on disk.

Baseline and anchor age (measured, artifact generated 2026-11-14 15:14 EST; artifacts/eng62/baseline_age.json, script tests/scratch/eng62/baseline_age.py, output baseline_age.out beside it):
- Move term, fit (spread_quotes.parquet hourly rows, 2023-25, 799 games, 2390 game-books): baseline is the Tue 23:55 ET snapshot in 2310 of 2390. Hours before Wed 00:00 ET: median 0.07, p90 0.07, max 12.07; season medians 0.072, 0.073, 0.073; spread within a game median 0, p90 0.2.
- Move term, live Wk3-10 (286 game-books; replication equals served leader_books on every game): median 9.54 h, range 5.41 to 9.98. By week 7.97, 7.99, none (Wk5), 5.98, 9.98, 9.88, 9.54, 9.98.
- What that gap costs, on the fit's own history: a baseline at the 14:25 slot instead of 23:55 changes the game's leader-median move in 23.0 pct of games (21.7 pct by a half point or more, mean abs 0.14 pt).
- Offset term, fit (hook_rows, 661 whole-line games, 2020-25): anchor is the Tue 08:55 ET opener snapshot, 3.07 h before Tuesday noon in every season (range -3.08 to -3.07); the fitted pool line is that anchor plus or minus a half (read pool_hook.py:78), so anchor and pool line share one moment. MOD-57 regrade anchor: +0.93 h (2023-25), -3.08 h (2020-22).
- Offset term, live Wk3-10: 103 games on a Tuesday quote, median +2.46 h after noon, range 1.27 to 6.59 (Wk3 5.04, Wk4 4.62, Wk6 6.59, Wk7-10 2.43 to 2.46); Wk5's 15 games fell back to a snapshot 30.0 h after noon. Replication equals tuesday_market_lines on Wk3 and Wk10.
- Anchor drift on 2023-25 history (12-book median against the noon median): changed in 12.1 pct of games by 14:25 and 25.2 pct by 23:30 (half point or more: 9.9 and 20.8 pct; whole noon lines: 10.7 and 25.3 pct).
- A Tue 23:30 capture: move baseline 9.54 -> about 1.07 h (0.5 h plus the feed's median 34 min scan lag; 0.5 to 1.6), 8.5 h toward the fit. Offset anchor +2.46 -> +11.5 h, 9.0 h further from the fit and beyond anything live so far.
- Different reference times without a pinned edit: no (inferred from the read lines). Both terms take the last Tuesday quote from the same store. The one split is the time basis (scan stamp vs observation); a just-after-midnight capture would use it, but the stamp lags 0.7 to 68 min and is missing on 30 pct of leader rows, so it is not dependable.
- Recommendation: do not add the capture alone; it trades a move mismatch on 23.0 pct of games for 13.1 more points of anchor drift. Land it with the anchor fix (Next 2), which removes both. Inferred, rough: at the read coefficients (move 0.19 to 0.25 logit per point, offset 0.805) the capture alone still nets about 0.03 logit per game gained against 0.01 to 0.02 lost, so it is the fallback, not a harm.

## Tried
No live probe, no capture run, no backfill, no scheduler edit. Bovada public endpoint returned an empty object on all 13 Week 10 runs; bovada comes from the Odds Gap feed only. Capture scripts write to a fixed root (capture_odds_gap_private.py:19), so a shadow capture the served code cannot see needs a script change.

## Next
1. Orchestrator, before the Week 10 card: check MIN at GB's served side with the -1.5 one-book move in mind; no code path changes it without a fitted term.
2. Next pinned bundle (re-enroll the cohort in the same session): pool_hook.py:110-117, stop the anchor at the first snapshot after Tuesday noon (the value :102-109 already compute; live that is the 12:10 capture, +0.2 h) so later Tuesday captures cannot move it. sharp_book_movement_features.py needs no edit; :139 already takes the last pre-Wed quote. In the same change add the Job in artifacts/eng62/late_tuesday_job.txt after capture_scheduler.py:1610 (own entry, grace 25, not the 90-minute generator) and restart the daemon. Result next week: move baseline about 1 h before Wed for every book quoting then, anchor at posting time.
3. If the bundle cannot land before Tue 11-17 23:30 ET, the orchestrator chooses knowingly: no change (both terms as this week) or the capture alone (move fixed, anchor at +11.5 h).
4. Then capture_odds_gap_private.py:36-42: keep every upcoming game whose week has begun, so the Monday job writes the coming week; check readers of the manifest week tag (clv.py:214-215, 323-324).

## Open
How many leader books quote at Tue 23:30 (never captured). Which leader rows lack a scan stamp. This window finds 7 fit games at 2 leader books (bovada), MOD-56 says 3 on all 799. Why bovada and mybookieag drop games on Tuesday afternoon. MOD-56 lane says Wk9 had 18 games; the schedule has 15.
