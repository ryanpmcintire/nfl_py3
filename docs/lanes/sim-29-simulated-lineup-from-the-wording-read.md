# SIM-29

## Goal
Simulated lineup availability from the injury wording read: draw each listed starter's play status from his read probability, apply measured replacement costs in the engine, and test the simulated lineup surprise S against the opener residual R (ROADMAP SIM-29). The SIM-31 finish line waits on this forward test.

## State (2026-11-21 16:52 GMT)
- Backtest units 1-2 done; forward test declared below before any Week 12 data exists; forward logger built (scripts/sim29_forward.py `log --week auto`, SKIP exit 0 outside Weeks 12-18, measured 16:51 GMT: `{"skip": true, "week": 11}`, exit 0). Nothing is scheduled yet: the scheduler diff below is for the orchestrator.
- Defect found and fixed (measured): units 1-2 set open_line = -card_line, but card_line equals clv_ledger decision_home_spread in 88 of 88 Weeks 4-9 games and that column is home-favoured positive (correlation with schedule spread_line +0.94 to +1.00 in every 2026 week), so R(open) was margin + line (sd 17.09 against 11.72 fixed). Fixed in scripts/sim29_lineup_sim.py games_2026 (open_line = card_line). Rescored the unit-2 games without re-simulating (artifacts/sim29/opener_sign_fixed_rescore.json), decisive first: S and R agree in sign in 23 of 43 decisive games (S < 0: home covered 8 of 25; S > 0: 6 of 18; S = 0: 6 of 15). Looks (4, family sim29 replacement cost): slope R(open) on S 0.137 [-1.37, 1.93] P+ .583 (58); starter-out R - S -2.34 [-6.37, 1.92] P+ .144 (36), mean S -0.74 v R -3.08; out-against-read R - S -0.85 [-5.09, 3.76] P+ .329 (32); close slope unchanged -0.044 [-1.28, 1.26] P+ .456 (94). All unresolved_below_power. The registry still holds the wrong-sign opener cells: the orchestrator re-records those three with `--replace`.

## Forward test declaration (2026-11-21 16:52 GMT, before Week 12)
- Frozen construction: scripts/sim29_lineup_sim.py blob 5071893a and scripts/sim29_forward.py blob 2ee1f203 (commit as is; every log stores working and HEAD blobs; a log whose blobs differ is not scored); engine blobs sim10_drive f6145854, sim08_market_anchored 0d8aa5ed, mod25_generator bb08f3d9, sim04_loso ae3e6b3c; default engine (SIM10_TERMINAL unset), 2025 transmission fit, anchor = schedule spread_line from the latest schedule capture before the run. Costs QB backup -0.0604, OL starter -0.0380 EPA/play, 5 OL slots. Starters = top QB and top-5 OL by 2026 offense snaps before the week (latest data/players/forward_snaps file). Reads = LEAD-106 forward files recorded before the run, rows captured before kickoff, latest per player. Unit-2 known-absent rule (last weekly_rosters capture before Tuesday 12:00 ET: RES/CUT/RET/EXE/TRD/TRT/TRC or other team read 0); unread non-absent starters read 1.0. 200 draws, 200 simulated games per distinct lineup, seeds crc32 of game_id.
- Population: Weeks 12-18 games with a Tuesday card line (first clv_ledger decision recorded on the opener Tuesday, ET) and a LEAD-106 read captured before kickoff, logged before kickoff. Excluded by construction, with status per game: games kicking off before the week's read (Week 12 Wednesday GB-LA, Thanksgiving 13:00 and 16:30, Friday DEN-PIT; Week 18 Saturday games before the Saturday read).
- Logged per game before kickoff: starters with read, roster state and known-absent flag; lineup configs, frequencies and simulated mean margin per config; sim_base (all starters play), sim_read, simulated shift sim_read - sim_base; Tuesday line; anchor line; served side and home probability (latest pick_revisions row before the run, else the card forecast home_cover_probability) with its source.
- Scored after Week 18 only: realised lineup from snap counts; S = simulated margin of the realised lineup (logged seed, frozen blobs) - sim_read; R = margin - Tuesday line. Decisive record first (S and R both nonzero; pool-line pushes counted and excluded). Looks (5, family sim29 forward): unit-2 looks 1-4 plus the served side's pool-line record split by whether the simulated shift favours it (permutation null). Game-cluster bootstrap 2,000, P+ reported, each recorded with weak-signals record.
- Detectable size (inferred, scaled from the unit-2 bootstrap SEs): about 95-105 logged games; slope SE about 0.64 forward-only (0.51 pooled with the 58 backtest games), so 80% power at 5% needs a slope near 1.8 (1.4 pooled) against a mechanism slope of 1; starter-out R - S needs about 4.5 points. Below power is expected and closes nothing.
- Declared limitation, the Monday-Tuesday roster gap (read): player_snapshot jobs run Wed-Sun 06:15 and 09:15 ET plus Wed-Sat pm (capture_scheduler.py:2437-2447); pfr_transactions Wed and Sat 07:00 ET (:2835-2880); raw captures jump from 20261115T141548Z (Sun) to 20261118T111607Z (Wed). No roster capture exists between Sunday 09:15 ET and the Tuesday noon opener, so Monday-Tuesday moves are missed (unit 2: 58 starter-games out with roster present and no read). The proposed Tuesday 09:15 ET snapshot closes it from the week it starts; each log records its roster snapshot.

## Smoke 2026-11-21 (Week 11, marked, never scored)
`python scripts/sim29_forward.py log --week 11 --smoke --workers 6`, 16:51-16:53 GMT, exit 0, 136 s with engine start (measured) -> artifacts/sim29/forward/smoke/2026_week11_20261121T165136Z_smoke.json, log artifacts/sim29/forward_smoke.log. 13 games: 1 already kicked off (Thursday), 12 no_read_before_kickoff (Saturday read not yet recorded), simulated anyway in smoke mode; reads 0 per starter, so 1 lineup per game; known-absent starters in 5 games (simulated shift -1.29 to +3.50); inputs: schedule 20261121T140107Z, roster 20261115T141548Z, snaps through Week 10, engine blobs equal HEAD, all started processes exited. Defect seen and fixed after the smoke: the card-forecast probability took the last NaN row of predictions.csv (7 of 12 games None); now filtered to the decision's method (verified: MIA-BUF 0.552, NO-CHI 0.620). Expected runtime for a 12-14 game week with reads about 3-5 min at 6 workers (inferred from 136 s here at 1 lineup per game and 2.6 lineups per game in unit 2).

## Proposed scheduler diff (orchestrator applies; capture_scheduler.py not edited)
```
+    *(
+        Job(
+            f"sim29_forward_{day}_{at.replace(':', '')}",
+            day,
+            at,
+            grace,
+            [str(UV), "run", "--no-sync", "python",
+             str(REPO / "scripts" / "sim29_forward.py"), "log", "--week", "auto", "--workers", "6"],
+            True,
+            "SIM-29 forward log: simulated lineup draws from the LEAD-106 reads, written "
+            "before kickoff; skips outside Weeks 12-18.",
+            added_on="2026-11-21",
+        )
+        for day, at, grace in (("thu", "19:00", 60), ("sat", "20:30", 600))
+    ),
+    _player_snapshot_job("tue", "09:15", suffix="_0915"),
```
Thursday 19:00 ET follows the 17:30 Thursday read (written about 17:31) and ends before the 20:15 kickoff; Saturday 20:30 ET follows the 18:00 Saturday read and its grace ends Sunday 06:30 ET, clear of the 09:40-12:00 window and 12:00-12:55.

## Tried
- 2026-11-11 unit 1: read beats the availability ladder on expected missing count per team-week (MAE gain .038 [.012, .067] P+ .9985; artifacts/sim29/lineup_expectation.json).
- 2026-11-21 units 1-2: simulated replacement-cost check, Weeks 2-10 (artifacts/sim29/lineup_sim*.json); known-absence fix moved 6 of 42 starter-out games out of the bucket; opener looks superseded by the sign fix above.

## Next
1. Orchestrator: commit both sim29 scripts at the blobs above; apply the scheduler diff before Tuesday 2026-11-24 09:15 ET; re-record the three opener cells with the sign-fixed numbers (`--replace`).
2. After Week 18: build the scorer per the declaration. Positions beyond QB/OL need a measured replacement cost before they can be drawn.

## Open
Rules: measured mechanism only, never a tuned constant; aggregates are checks; flag-off hashes equal HEAD before any run; unresolved results reported as measured. The simulated shift enters the calibrated probability only as a challenger term fitted out of season.

ROADMAP SIM-29 update: 2026-11-21 opener-sign defect fixed (R was margin plus the Tuesday line); sign-fixed slope 0.14 [-1.37, 1.93] P+ .58 (58 games, decisive agreement 23 of 43), all looks unresolved below power; forward test declared for Weeks 12-18 and logged by scripts/sim29_forward.py, scored after Week 18.
