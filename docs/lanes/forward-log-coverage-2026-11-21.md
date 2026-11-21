# forward-log-coverage-2026-11-21

## Goal
Prove every 2026 forward log (scored after Week 18) is alive for Weeks 9-11. Flag silent deaths, give causes and repair steps. No research verdicts, no scheduler edits.

## State (2026-11-21 11:16 ET, measured from data/scheduler_log.txt and artifacts/)
Alive. Last OK this week, no FAIL streak, a file for every week in its window: mod24_forward_log, armb_forward_log, mod25_forward_q (W9/10/11 files 2/2/1 each), lead99 deadline (14/6/2), lead101 thu/sun, lead105 (W10-18 window, W10 3), lead106 thu (sun view due 11-21 18:00), lead108_forward, lead112, nfelo capture/pregame (W11 4), pooled_stack (W11 6), exchange_move (W11 1), pm_gap deadline (W11 1), prop_market, card_terms (W11 78 rows), pool_move_log (rows 105/85/38), paired_wind (SKIPs are "no games in window"). Score jobs saying "no admissible forward rows yet" on 11-17 are expected (first rows W11).

| Flag | Evidence | Cause | Recoverable |
|---|---|---|---|
| **Daemon runs a stale schedule** | daemon PIDs 38536/44148 started 11-16 18:07 with 420 jobs; file now has 436. Log shows `card_terms_sat_1040`, file has `sat_1035`. | Four commits 11-21 (25078994, ac57a84a, 1956c2e8, e2268b38) not loaded. Never run: team_hfa_forward_log (tue_1400, sun_1215), lead108_mixture_forward (thu_1830, sat_1900, sun_0940), mkt24_holiday_forward_log (tue_2100, wed_2030); odds tape 12:25 end, card slot table. | Nothing lost yet (windows start W12). All of W12 lost if not restarted. |
| **draftcap_forward_log FAIL** 11-19 19:15 (also 11-01) | W11 files 0. Measured refit now (0.028659, -0.231327) vs frozen (0.027339, -0.230242), same with the enrolled snap file 20261018T131559Z and 4 later ones; combine and picks unchanged since Aug. | Guard at scripts/draftcap_forward_log.py:111-114 is correct. Inferred: 62d554ea (11-17 18:05, QB repair + pregame weather in features.py, players.py, pregame_weather_features.py) rewrote 2013-25 rows of game_features_weak_stack.parquet. Last OK 11-15 was before it. | IND at HOU lost. The v1 table is gone, so a recompute now is a back-fill. Sun 11-22 job (11:15) will fail again. |
| **independent_validation** 4 FAIL 11-18..11-20 | "source changed" | LF rewrite of pinned pick_probability.py; fixed 11-21 (docs/lanes/done/validation-v10-crlf-2026-11-21.md). Now `status`: source_unchanged true, 13/122 captured. | 4 captures lost, no back-fill. |
| **Tuesday chain blocked every week** | 11-03, 11-10, 11-17 MISSED for forward_terms, mod25_wfj, mod27, mod54, forward_ledger, lead99 tuesday | weekly_lock MISSED, prerequisite splash_board_tue. Manual runs 11-03 15:50, 11-10 20:56, 11-17 17:55 (before Thursday kickoffs), so W9-11 files exist. | Covered. |
| **pm_gap_pregame_tue OK, wrote nothing** | 11-17 12:30 "no game with a price, book line and published card before its cutoff" | Card published only at 17:53 (manual weekly_lock); this job was never re-run. | W11 Tuesday slot lost (live prices). |
| card_fidelity 9 FAIL in a row (a check, not a log) | e.g. 11-21 10:38 "TB at DET: recorder side HOME != recomputed AWAY" | Check ran before the recorder after a refresh (1956c2e8 ordering fix not loaded). card_terms rows are fine: the recorder runs after the check. | n/a |
| pool_move_log_sun_1255 MISSED 11-15 | window +4m | Odds tape held the queue (fix in 25078994, not loaded) | W10 last-call rows lost. |

## Tried
Inventory: scratch scripts importing `capture_scheduler.SCHEDULE` and parsing the log; output scans of artifacts/*; a draftcap refit that writes nothing (scratchpad dc_diag*.py). No repo files changed except this lane.

## Next (root)
1. Restart the scheduler before Sun 11-22 11:30 ET (the odds tape starts then): kill PIDs 38536 and 44148, start once, check the process list (restart race), and confirm `scheduler started (... 436 enabled jobs)`.
2. DONE 2026-11-21 11:20 ET (measured): draftcap re-enrolled as v2. `artifacts/draftcap_forward/frozen_v2.json` (enrolled 16:19:56Z, seasons 2013-2025, coefficients 0.028659 / -0.231327, digest 633f2257..., declaration of reason, window W11 not-yet-kicked-off through W18, v1 kept, no silent pooling); v1 `frozen.json` untouched. `scripts/draftcap_forward_log.py` reads v2 with the allclose guard unchanged and stamps `frozen_version`. Ran the job command once: `runs/2026-week-11-20261121T162000Z.json`, 12 rows (W11 Sun/Mon), IND_HOU refused past_deadline; a second `--dry` reproduced v2. So the Sun 11-22 11:15 run should pass (the "will fail again" in the table above is superseded). Remaining for root: note the protocol change in docs/lanes/mod23-unit10b-draft-capital-v2.md; ROADMAP MOD-23 line is in docs/lanes/draftcap-forward-log.md; commit the script change.
3. Tue 11-24: do the Splash board read before 12:20 so weekly_lock runs on time. If weekly_lock goes manual, also re-run `prediction_market_gap.py pregame --slot tue` and the six Tuesday dependents.
4. Confirm `independent_validation_sat_1210` logs OK today.

## Open
- Add a weekly_lock prerequisite to pm_gap_pregame_tue (scheduler edit, root).
- Confirm the cause of the draftcap drift by diffing 2013-25 rows against an E: mirror copy of game_features_weak_stack.parquet from 11-15 or earlier.
