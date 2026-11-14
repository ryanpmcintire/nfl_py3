# full-steam-2026-11-14

## Goal
Owner 2026-11-14 (Saturday): maximum parallel work. Done = the simulator's final run is graded and SIM-08/SIM-30 closed, every open ROADMAP row carries its lane's verdict, one dashboard improvement is live, and each unit below is committed and pushed.

## State (times from the scheduler log or commits)
- Simulator final run: killed 2026-11-11 20:09 by a planned Windows Update restart, relaunched about 14:35 ET as a session background job: tests/scratch/sim_finish_queue/forecaster_full.ps1 (10 workers, resumes by fold, about 5 hours for folds 2023-25). End marker artifacts/sim08_amend6_full.done. If no `sim08_market_anchored.py` process is alive and the marker is missing, rerun the launcher.
- Pushed 3cb52b0a: dashboard game details show hours ahead of or behind home and away-from-home streaks (lane ui20-2026-11-14.md); pool-standing alarm (scripts/check_pool_standing.py, job pool_standing_check_tue Tue 21:30).
- Pushed f944c1ad: ROADMAP reconcile of 100 stale rows (87 to 122 of 223 done; lanes roadmap-reconcile-2026-11-14-g1..g4.md); scheduler `--health` OVERALL OK after 116 old missed windows were classified and acknowledged (lane scheduler-missed-window-triage-2026-11-14.md).
- Pushed 99ba4b56 (durable tests 1645 passed, measured 14:58):
  - ENG-61: card-ledger-check compares the Best Pick star on the board and in the card file with the ledger's star; both call sites only warn, so it cannot block a publish (read, cli_commands/publishing.py:1100, :1633). Live: all three show PIT at CIN.
  - Offset-anchor contract in scripts/check_served_terms.py: fails on a single-book anchor or a cross-book range above 2.0 (the largest seen in Weeks 1-9); measured 150 games: 1 single-book, 18 two-book, 73 three-book anchors; the fit population's median was 13 books (reported, no rule). Live Week 10: no problems.
  - Forward ledger through Week 9 (lanes eng-49, eng-48, eng-50; artifacts/forward_monitor/per_term.json): ledger equals the card 70-66 in every week. Against the card's probability: language read LEAD-99 43 games +0.063 log loss [+0.019, +0.110], P+ .998, decisive 13-3; mod25_wfj 15 games +0.099, P+ .997, 7-1; draft capital 29 games P+ .90, 7-2; the rest .22 to .94. A constant 0.5 beat the card by 0.059 in Week 9 (card 5-10). 91 looks, 18 cells recorded unresolved. Job forward_ledger_tue Tue 13:15.
  - LEAD-105 tiebreak-total read declared, frozen and logged for Week 10 (LAC at BAL: read 38.0, market 38.5, served guess 37); jobs Mon 19:10, Sun 19:10 game-day and Sun 15:05; scored after 2027-01-10.
  - PER-16 unit 4 (coach-keyed Questionable play rate on the served base): decisive 17-26, -0.60 pts [-1.46, +0.27], P+ .086; 2026 forward log loss below zero but the intercept control is too, so unresolved, not a wrong sign; 30 cells recorded.
  - Scheduler daemon restarted 14:59 for the new jobs (pid 2096, single tree, code and schedule current).
- XLG-12 college shadow ledger: two silent no-ops fixed (stale cached capture; score never refreshed results), Week 10 scored. Root ruling: the 216 Week 11 rows frozen from a capture minutes before kickoff are not opener rows and must not be scored as such; a fix agent is removing or tagging them and adding the guard. XLG-12 files are uncommitted until it returns.
- Running (six agents): LEAD-99 forward decision read (lane lead99-forward-decision-read-2026-11-14.md); XLG-12 opener-basis fix; LEAD-106 Week 9 wording-read scoring; ENG-62 leader-book depth for Week 10; MOD-47 short-week side term; MOD-39 playoff-leverage unit repair and forward enrolment before Sunday kickoffs.

## Tried
- A reconcile agent's "blocked" for MOD-30 (unlicensed public ratings repository) was not accepted as an owner wait; it stays not started.
- `--run-job` was not used for lead105_..._deadline_sun_1505: its argv is the same command the build unit ran for the Week 10 record, and a second run is a paid duplicate read.

## Next
1. LEAD-99 decision read returns: root verifies its numbers against artifacts/forward_monitor/per_term.json, then decides between forward-only, a frozen stacked challenger from Week 11, or a served fitted term (a served change touches pinned files and re-enrols the validation cohort).
2. XLG-12 fix returns: check ledger row counts and the manifest, commit the XLG-12 files together.
3. Refill queue, in order: POOL-08 and POOL-09 Week 9 reruns; UI-30 number-moved lines; MOD-52 record its 17 cells; LEAD-108 once LEAD-106 says its inputs exist; PER-16 role-weighted term.
4. Simulator run ends: sim-finish-amendment6.md Next 4 (evaluate, grade stage 2), scripts/sim30_margin_mass.py, record, close SIM-08 and SIM-30, fill the PENDING line in docs/sim_engine_summary.md, lanes to done/.
5. Sunday 11-15: scheduler lines from mod57-sixty-percent.md Next; star re-nomination at 10:00; first lead105 deadline read at 15:05.
6. Every push needs `nfl-ats handoff` first with HANDOFF.md and README.md staged.

## Open
- None.
