# MOD-57 U2 - deadline staleness (stamped 2026-11-11 08:36 -0500)

## Goal
Is the live market input staler at each game's pick deadline than the training feature, and which remaining 2026 kickoff slots have no leader-book capture plus refresh near the deadline. Measure and propose only; nothing operational changed.

## State
- Training (read): src/nfl_ats/pick_probability_fit.py:63,154-169 loads sunday_market_probability/20260920_fixed/market_move.parquet, built in scripts/lead69_unit1.py:184-205 by sharp_book_movement_features(include_sunday=True) (src/nfl_ats/sharp_book_movement_features.py:43-169) with cutoff = pick_deadline = min(kickoff, Sunday 16:00 ET) (src/nfl_ats/pick_refresh.py:96-98); last quote per book strictly before the deadline, observed and snapshot time both; move = median over bovada, williamhill_us, mybookieag of (last line minus first line on or after Wednesday).
- Live (read): pick_probability.py:740-788 calls the same function with cutoff = feed time (now), include_sunday=True; pool_relative_move_forward_log.py:102-116 reads last leader quote at or before cutoff.
- Measured (tests/scratch/mod57/u2_staleness.py, csv u2_games.csv), 136 finished Week 1-9 games, last leader quote behind the final feed vs deadline: median 23.5 h, p90 30.0 h. By slot (n, median, p90 hours): Thu 9, 26.3, 146.3; Sun early 75, 22.0, 28.7; Sun late 33, 25.0, 30.0; Sun night 9, 25.0, 30.6; Mon 9, 25.0, 30.6; other (Wed) 1, 482. Feed itself ran 0.75 h (Sun early) or 3.75 h (Sun late/night/Mon, deadline 16:00) before deadline.
- Cause (measured, Week 3-9 logs): the Sunday feed fires about 12:13 ET, before odds_private_sun lands its quote (observed about 12:22-12:26); the served side reads Saturday 10:00 quote.
- 40 of 136 games (29%) have leader-median home line differing by 0.5 or more between feed-time and deadline-time (Sun early 23/75, Sun late 11/33, Mon 4/9, Sun night 1/9, Thu 0/9, Wed 1/1). Sign: positive home line = home favoured, matches the pool file.
- Decisive record (measured, served coefficients move 0.2204 plus availability term, only the move term swapped, no refit): side changes in 1 of 40: 2026_08_ATL_TB, feed HOME 0.526 to deadline AWAY 0.498; home covered (13.5 vs pool line); feed side 1-0, deadline side 0-1. All 40 differing games: feed side 21-19, deadline side 20-20, card 18-22. 1 game is not a finding.
- Slots W10-18 (read, u2_slots.py, schedule snapshot 20261110T140026Z; leader capture = odds_private_*/odds_tape_sun, refresh within 2 h before deadline; paid odds_*.ps1 jobs not counted: odds_thu_tnf FAIL since 2026-09-17, odds_espn_thu MISSED, Bovada returns dict on 2026-11-08 so only Odds Gap books captured):
  - Wed 20:00 (wk12): capture odds_private_wed 18:00, refresh_wed 18:15 - covered.
  - Thu 13:00 (wk12) refresh_thu_inactives_early 11:55, capture NONE. Thu 16:30 (wk12) refresh_thu 15:00 / inactives_late 15:25, capture NONE. Thu 20:15/20:20 (wk10-17) refresh_thu_inactives_primetime 19:15, capture NONE.
  - Fri 13:00 (wk16) odds_private_fri 12:30, refresh NONE. Fri 15:00 (wk12) refresh_last_call_fri 13:30, capture NONE. Fri 16:30, Fri 20:15 (wk16) NONE, NONE.
  - Sat 17:00 (wk15) refresh_sat_inactives_early 15:50, capture NONE. Sat 20:20 (wk15) refresh_sat_inactives_late 19:15, capture NONE.
  - Sun 09:30 (wk10) refresh_last_call_sun_0830, capture NONE.
  - Sun 13:00 (78 games wk10-18): odds_private_sun 12:10 and tape 12:12, refresh_last_call_sun_1245, inactives_early 11:55; but capture lands after the 12:13 feed.
  - Sun 16:05, 16:25, 20:20 and Mon 20:15 (deadline Sun 16:00; 40 games over wk10-17): refresh_sun_inactives_late 15:00 only; capture NONE (tape ends 13:00).
- Alerts (read): scripts/capture_scheduler.py:2935-2984 notify_after_job: any refresh_ or lineups_ job with changed_picks sends send_notification (ntfy+email, urgent), logs PICK-CHANGE and NOTIFY-OK id, then republishes the site. New refresh jobs named refresh_* inherit it.

## Tried
- Reused L.build(week, backfill=True) per-game inputs and L._leader_median/_coefficients; no hand quote parsing.

## Next (proposal, nothing applied; add to capture_scheduler.py job table)
Clone odds_private_wed (PRIVATE_SUNDAY_ODDS_CAPTURE, grace 90) for captures; clone refresh_last_call_* for refreshes (grace 40).
1. odds_private_thu_1140 Thu 11:40 (before refresh_thu_inactives_early 11:55).
2. odds_private_thu_1450 Thu 14:50 (before refresh_thu 15:00 and inactives_late 15:25).
3. odds_private_thu_1905 Thu 19:05 (before refresh_thu_inactives_primetime 19:15).
4. odds_private_fri_1315 Fri 13:15 (before refresh_last_call_fri_1330); refresh_last_call_fri_1230 not needed.
5. odds_private_fri_1500 Fri 15:00 + refresh_last_call_fri_1515 Fri 15:15 (16:30 deadline); odds_private_fri_1845 Fri 18:45 + refresh_last_call_fri_1900 Fri 19:00 (20:15 deadline).
6. odds_private_sat_1530 Sat 15:30 (before refresh_sat_inactives_early 15:50); odds_private_sat_1900 Sat 19:00 (before inactives_late 19:15).
7. odds_private_sun_0815 Sun 08:15 (before refresh_last_call_sun_0830).
8. Sunday 16:00 deadline: odds_private_sun_1430 Sun 14:30, odds_private_sun_1515 Sun 15:15, refresh_last_call_sun_1530 Sun 15:30 (grace 25). Keeps refresh_sun_inactives_late 15:00 reading the 14:30 quote.
9. Sunday 13:00: odds_private_sun runs 12:10 but lands 12:22-12:26; the 12:13 feed precedes it. Move odds_private_sun to 11:45 or make the last-call 12:45 pass the feed of record; confirm with the owner of the scheduler which feed the board reads.

## Orchestrator check and fix (2026-11-11 08:43 ET, time from the scheduler health output)
- Correction (read src/nfl_ats/pick_refresh.py:1150-1160; measured data/scheduler_log.txt 2026-11-08): every refresh-picks pass recomputes the move term at its own run time, so the staleness table above describes the forward-log feed, not the recorded decision. On 11-08 odds_private_sun landed 12:26 and refresh_last_call_sun_1245 ran 12:46-12:48, so Sunday early games were decided on quotes about 35 minutes before kickoff. The 23.5 h median does not apply to them.
- Gaps that are real (read, job table): Thursday primetime pass 19:15 read Wednesday 18:00 quotes; Sunday 16:00-lock games read the 12:26 capture at the 15:00 pass and had no later pass; Thanksgiving, Friday 15:00, Saturday and Sunday 09:30 slots had a refresh with no capture.
- Applied in scripts/capture_scheduler.py: DEADLINE_ODDS_CAPTURE (no age guard) jobs odds_private_thu_1140, _thu_1450, _thu_1900, _fri_1315, _sat_1535, _sat_1900, _sun_0815, _sun_1445, _sun_1520, and refresh_last_call_sun_1535. Sunday 11:40 not added: with the 30-minute guard it could suppress the 12:10 capture that the 12:45 pass depends on.
- Verified (measured): ruff clean; tests/test_capture_scheduler.py and test_scheduled_lock.py 70 passed; `--run-job odds_private_thu_1900 --dry` MANUAL-DRY-RUN OK, 14 games, 82 quotes, three leader books; daemon restarted once, one process tree (pid 23376), health code current, schedule current.
- Closed 2026-11-11 08:57 ET (time from the health output): remaining slots measured from game_features.parquet (tests/scratch/mod57/root_slots.py): Wed 11-25 20:00, Fri 12-25 13:00, 16:30, 20:15. Added odds_private_wed_1900, odds_private_fri_1500, odds_private_fri_1845 and refresh_last_call_fri_1240, _fri_1515, _fri_1900. Scheduler tests 70 passed; `--run-job refresh_last_call_fri_1900 --dry` MANUAL-DRY-RUN OK; daemon restarted once (pid 9364), health code current, schedule current. Every remaining 2026 kickoff slot now has a capture and a refresh inside the two hours before its deadline.

## Open
- Not measured: historical quote density in the training artifact (how far the last quote is from the deadline in 2020-25); the 1-in-40 flip count is small and unresolved_below_power.
- Odds Gap age guard (70 min) and Bovada array error cap what an extra capture can add; unverified that extra captures return fresh quotes.
- Whether refresh_last_call_sun_1245 produces a waterfall feed counted by the forward log (Week 9 feeds show 12:15, not 12:48).
