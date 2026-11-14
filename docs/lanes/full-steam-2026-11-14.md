# full-steam-2026-11-14

## Goal
Owner 2026-11-14 (Saturday): maximum parallel work. Done = the simulator's final run is graded and SIM-08/SIM-30 closed, every open ROADMAP row carries its lane's verdict, a dashboard improvement is live, and each unit is committed and pushed.

## State (session start 14:28 ET; all pushed, head 0e452e6a at 18:22 ET, unless marked)
- Simulator final run: session background job tests/scratch/sim_finish_queue/forecaster_full.ps1 (10 workers, resumes by fold). Folds 2020-24 saved; fold 2025 started 18:10, about 1 h 55 min. End marker artifacts/sim08_amend6_full.done. If no `sim08_market_anchored.py` process is alive and the marker is missing, rerun the launcher. Grading is rehearsed and three grading-script bugs are fixed; the runbook blocks "When the done marker exists" and "Still owed for SIM-08 after the grade" are in sim-finish-2026-11-11.md (then tierb_terminal.ps1, then control_amend6.ps1, one at a time).
- Card (Week 10): healthy. Scheduler health OK; daemon restarted three times for new jobs and the alert change (pid 15840 at 16:11). Dry runs OK on today's code: refresh_last_call_sun_1535 and refresh_sun with the Best Pick re-nomination (candidate SEA at LV, 62.4%). A failed served-terms or pool-standing check now pushes a notification (delivery proven 16:10); MIN at GB will alert again at 09:15 Sunday (its line move rests on one book).
- Dashboard, four changes live (verified by fetching the deployed page): time-zone and away-streak facts; number-moved lines in the Changed-since-Tuesday strip; dated injury news beside a move, only for the team the line moved away from; the pool panel states both first-place chances (6 in 100 at the stated win rate, under 1 in 100 at the realised one).
- ROADMAP: 134 of 223 done (87 at session start); rows carry today's results (applier and inputs in tests/scratch/reconcile_2026_11_14/). Registry: about 700 cells recorded; base-tag table rebuilt by scripts/eng57_base_tags.py (124 unknown left).
- Forward logs: matched ledger, monitor and stacker through Week 9 with the Tuesday job forward_ledger_tue; new frozen terms from Week 11: lead99_first_view, mod25_wfj_served, sky01snap_idjoin, xlg09_cfb_opener_served; MOD-39 from Week 10; LEAD-105 tiebreak-total read with three jobs; LEAD-101 scorer in the weekly pass; XLG-12 college ledger repaired.
- Findings that deflated a lead (each recorded unresolved, nothing served): the language read's 13-3 was an undeclared row (declared row 23-20); the student model's 7-1 was the raw model against a cold card; the college-centred prior's +1.13 was the penalty; fumble-recovery luck passed college and failed its one NFL confirmation; the college opener-error term adds nothing on the served base; the SIM-16 wrong sign was a name-join artifact (withdrawn); the pre-2020 opening line is a proxy that fades its own error.
- Instruments (memory add-one-noise-floor): a no-information term loses 0.17 to 0.27 accuracy points on the served harness; detectable effect about 1.5 points on 1,503 NFL games, 0.7 to 1.1 in college; no metric fixes it.
- Chain audits: flips of Weeks 7-10 attributed (14 of 18 from the line-move term; settled flips final side 6-9, Tuesday side 9-6); Weeks 7-8 served moves ran on one book (parser, fixed 10-31; never back-fill the live quote store, memory); quote-age mismatch measured; NE at DET neutral-site orientation correct; rescheduled-games inputs final except rest_diff; quarterback inputs describe a starter who is out (SEA) and two debut starters with no inputs.
- Label audit (ENG-54): 100 of 104 cover computations correct; four opener-transfer loaders had inverted feature signs (fixed, 21 cells re-recorded, refutation stands); 46 scripts read the raw probability under a served label; the roster crosswalk matched 65% of snap rows (id join 99.98%).

## Decisions (root)
- Nothing served changes for Week 10. Language read not served. Tuesday reference timing: keep live. Counted-flag gate: keep as served (a freeze plan was withdrawn the same afternoon). XLG-11, T in college, MKT-25 proxy era: unresolved, no closure.
- No more one-term add-on units on the NFL opener harness; candidates in both leagues are screened in college first, then one declared NFL confirmation.
- Tuesday 11-17 before the weekly lock: apply the quarterback input repair (tests/scratch/qb_input_repair/repair_ab.patch, constant to the measured -0.123; lane wk10-qb-input-check-2026-11-14.md). Queued for the next fenced change, not Tuesday: rest_diff (features.py:680-694), ledger move column (pick_refresh.py:629-645), ENG-51 unit 2b.

## Running
XLG-13 transfer-table update with today's pairs (lane xlg-13). Otherwise only the simulator.

## Next
1. Simulator marker appears: follow the two runbook blocks in sim-finish-2026-11-11.md; record; close SIM-08 and SIM-30; fill the PENDING line in docs/sim_engine_summary.md; move sim lanes to done/; ROADMAP rows SIM-08, SIM-30, SIM-31.
2. End of session: quality gates once (ruff format --check, ruff check, mypy src, pytest), final commit and push.
3. Sunday 11-15: served-terms alert 09:15 (expected), star re-nomination 10:00, lead105 deadline read 15:05, scheduler lines in mod57-sixty-percent.md Next.
4. Tuesday 11-17: quarterback repair, then the pool and board read in Chrome (leader's Weeks 9-10 picks for UI-29), weekly lock, pool-standing refresh (five commands, lane pool-standing-refresh.md), pool reruns at Week 10 (POOL-05, 08, 09, 10), XLG-12 freeze line at 12:00, forward_ledger_tue 13:15, the Week 10 audit of which picks depended on the counted flag set.
5. A handoff check fails whenever an agent records cells after the refresh: rerun `nfl-ats handoff`, commit, push.

## Open
- None.
