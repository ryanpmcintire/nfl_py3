# week9-sunday-refresh-2026-11-08

## Goal
Owner asked (Sun 06:30 ET) for fresh lineups, spreads and injuries, a board update, and word of any pick change. Done = refit + recorded refresh, board published and pushed, owner told.

## State
- DONE 07:25 ET: refit (weekly-run, lineups) on the 06:00 nflverse injuries, 06:16 player snapshot, 06:33 injury news and 07:22 ESPN DraftKings lines. `refresh-picks --record-decisions --publish-card --note sunday_morning_owner_request`: 0 of 15 sides changed; Best Pick re-nominated DEN -2.5 (DEN at CAR, 56.7%) -> NO -2.5 (CLE at NO, 64.7%), ledger row recorded. Site published and pushed as 7ce08069.
- ESPN fix (owner objected 07:20 to the permanent ban): capture_espn_pickcenter.py no longer reads a sticky block file; a 403/429/5xx stops that run and writes data/market/espn_pickcenter_last_refusal.json. Old marker data/market/espn_pickcenter_blocked.json (2026-09-20) is no longer read. capture_scheduler.py odds_espn_* jobs enabled. Retry 07:22 captured 14/14 games (data/market/raw/20261108T122248Z).
- Bovada returned `{}` at 06:33 and 06:34; odds_private_sun 12:10 still scheduled.

## Tried
- lineups_sun_am through `--run-job` failed twice; refresh_lineup_forecast.py direct failed once. Root cause in the full log: GitHub read timeout (30 s) on nflverse depth_charts_2026.parquet (build_week_lineups.py:400). Ran the chain's steps one by one instead (refresh_lineup_forecast.py:49-90); all OK.

## Next
1. Daemon (pid 5428, started 11-04) runs stale code until restarted, so ESPN jobs stay off in it. The restart was denied by the permission classifier; owner runs `! scripts\stop_capture_scheduler.cmd` then `! scripts\start_capture_scheduler.cmd` (check `--is-running` between).
2. Scheduler carries the day: refresh_last_call_sun_0830, lineups_sun_am 09:30, refresh_sun 10:00 (card, re-nomination), inactives 11:35/14:40 + refreshes 11:55/15:00, odds_private_sun 12:10. Any flip notifies via ntfy.
3. Single nflverse timeout aborts refresh_lineup_forecast.py; candidate fix is a retry inside the depth-chart fetch (ENG row, not done).

4. Board text defect (owner asked 10:15 why NO -2.5 is the Best Pick): the CLE at NO explanation (explanations.json, artifact 2026-week-09-20261108T144043Z) says "Situational adjustment fired: bye-week rest edge", but flag_bye is not a counted term (pick_probability/20261108T144530Z coefficients.json counted_flag_columns = division, protection) and that overlay pointed toward CLE. The served 64.7% is mostly the 2.0-point leader move toward NO (coef 0.220/pt) plus the pool hook (0.805); raw model 52.6%. Fix: explanation text names only counted terms and the move/hook contributions.

## Open
- Status shows many Sunday 11-01 and Tuesday 11-03 jobs as MISSED/NEVER RUN although the daemon was up for parts of that span; not investigated.

- Checked 2026-11-11 15:19 ET: item 4 (card explanation naming a flag the served probability does not count) was fixed the same day in commit 314eadd6: render_pick_text now replaces the overlay sentence with the served-terms sentence whenever a pick probability model is present (read, src/nfl_ats/card_explanation.py:771-794), and no file published in the last two days contains the old phrase (measured). Nothing owed.
