# SIM-16 Pregame attrition expectation

## Goal
Pregame attrition expectation: measured in-game exit cost times the chance a thin position group loses a starter (ROADMAP SIM-16).

## State
Added 2026-11-03, unmeasured. SIM-12 measured an offensive-line exit at -0.038 EPA per play. A team with a thin backup line has a larger expected loss from exits that are invisible on Tuesday; compute that expectation from depth and hazard.

## Tried
Nothing yet. Novelty: SIM-12 (the hazard and the effect), MOD-33 (last-game exits), PER-10 (inactives). Grep found no pregame expected-attrition term.

## Next
First unit: Compute per-team expected EPA loss from exits (hazard times backup-starter gap) for 2018-2025 and test it against the opener margin residual with a permutation null.

## Open
Data: SIM-12 hazard table, depth charts and snap counts. Zero crossing closes nothing; record unresolved results with a closing ground.

## Unit 1 (measured 2026-11-03)
Script scripts/sim16_attrition.py (out tests/scratch/sim16), ruff clean. Family sim16_pregame_attrition: 3 term looks (hazard x gap, flat hazard x gap, hazard only) + 3x200 within-week permutation placebos. Hazard per OL starter slot (25220 slots 2016-25, exit 4.96%) logit on prior-game exit, Questionable (same-week report, may postdate opener), years_exp (age proxy; no age field), prior snaps/game; fit leave-one-season-out; z-coefs +0.085, +0.143, +0.200, -0.081 (se ~0.025). Gap = 1 - best bench OL prior snaps/game over starters' (mean 0.57). Mean cost 0.0053 EPA/play x 0.038 scale.
Opener grade, 1368 games, intercept-only control acc 52.8% LL 0.6921. Term (home-away): acc 51.8%, LL 0.6935 (gap ins/oos 0.002), coef +0.033 (se 0.054, P>0 0.73), folds +.07 +.09 +.02 -.01 +.02 +.01; decisive 112 games term 49 v control 63; LL gain -0.0015 (-0.0026,-0.0004), P>0 0.004; placebo p>=obs 0.95. Flat-hazard 43-57; hazard-only 52-55.
Availability corr: OL unavailability 0.00, OL continuity 0.11, offense continuity 0.06. Positive coef is the wrong sign for a cost. Recorded 3 cells unresolved_below_power.
Next: none unless a mechanism for sign; backup quality by EPA not snaps.
Recorded in ROADMAP 2026-11-11

## Unit 2 (measured 2026-11-14 17:50, date): name join replaced by a player-id join
Unit 1 joined snap history to starters by player NAME. scripts/sim16_attrition.py now takes `--join {id,name}`, default id (gsis id from `nfl_ats.players.attach_snap_player_ids`; 47 of 310,476 snap rows get none). `--join name` reproduces Unit 1 exactly. Outputs tests/scratch/sim16_rerecord/{name,id,audit_copy}/, compare.py, batch.json. Ruff clean.
- Name join mislabelled 957 of 25,220 starter slots (731 of 15,450 in 2020-25): default snap load although the id join finds prior snaps. Backup gap differs on 805 of 5,044 team-weeks (598 of 3,090); mean gap 0.566 v 0.663; team-weeks at gap 0: 929 v 350.
- Same 1,368 games, name -> id (log-loss gain, interval, P+): full -0.00146 [-0.00264, -0.00037] 0.004 -> -0.00089 [-0.00228, +0.00030] 0.074; flat gap -0.0016 [-0.0029, -0.00046] 0.004 -> -0.00085 [-0.00227, +0.00054] 0.102; hazard only -0.00056 [-0.00163, +0.00051] 0.15 -> -0.00061 [-0.00156, +0.00031] 0.106.
- Full arm by id: coef -0.041 (se 0.054, P>0 0.224; folds -.02 +.01 -.03 -.07 -.06 -.07), the expected sign for a cost (Unit 1 had +0.033); acc 52.6% v control 52.8%; decisive 144 games term 71 v control 73; placebo p>=obs 0.785.
- The ENG-54 audit copy (tests/scratch/eng54/sim16_idjoin.py) reproduces at 1,283 games: -0.00093 [-0.00264, +0.00090] P+ 0.15. It also drops 1,762 starter slots with no prior snaps (85 games), so it is more than a join-key change; the recorded numbers are the join-key-only ones.
- Recorded with `record --batch --replace`: 3 cells `unresolved_below_power`, no closing ground, because no interval is wholly on one side of zero. They were already unresolved in the registry; the wrong-sign wording lived in ROADMAP and the summary. `pool --league nfl --effect-units accuracy_points` loads (exit 0).
- Wrong-sign reading withdrawn (name-join artifact). Unit 1 gave the sign as its reason for no next step, so the no-hook decision loses that ground; the row is open below power. Inferred: the engine site this term would feed, the SIM-12 line-exit hook, is not adopted (read, docs/sim_engine_summary.md:25), so no engine run was available whatever the sign.
- Corrected ROADMAP sentence, replaces the 2026-11-11 verdict sentence (not applied; the orchestrator edits ROADMAP.md): `**Verdict corrected 2026-11-14 (measured, lane sim-16):** wrong-sign reading withdrawn, a player-name-join artifact; by player id on the same 1,368 opener-graded games the term is -0.041 (standard error .054, expected sign), log-loss gain -0.00089 [-0.00228, +0.00030], probability positive 0.074; three cells unresolved below power; no hook built, and the sign no longer argues against one.`
- Corrected docs/sim_engine_summary.md line 29 (not applied): `| SIM-16 | Pregame attrition | Opener-graded term -0.041 (SE .054), expected sign, unresolved (P+ .074); 52.6% v 52.8% control | Not run | Measured, unresolved below power; no hook built |`
Next: apply the two sentences; backup quality by EPA instead of snaps; split-half reliability of the team-week cost so the cell can be adjudicated.
Open: whether the ROADMAP status mark stays as done now that the row is open below power.
