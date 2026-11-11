# SIM-31 - finish the simulator

## Goal
Owner 2026-11-11: get the simulator finished once and for all, in parallel with MOD-57. Finish line, all four:
1. The default engine passes 14 of 14 realism gates on the six-seed mean, real schedules.
2. Every open simulator row (SIM-06 to SIM-30) has a recorded verdict under its declared rule: adopted, not adopted with the measured gates, or measured with no hook warranted. No row left not started.
3. The forecaster test is done: the final engine's cover and push probabilities graded at the opener against the served discrete read on the 2020-25 games (SIM-08 bar, SIM-30), result recorded, and the simulator either wired as a forward-logged challenger or recorded as not serving.
4. Lanes closed to done/, ROADMAP rows updated, one summary of what the engine is and what each mechanism did.

## State (read 2026-11-11 from sim11-strength-transmission.md, engine-patch-queue-2026-11-03.md, sim-27 lane, artifacts/bw_v2_cmp.log)
- Engine: scripts/sim10_drive.py, default `--joint 1`, era-aligned latent, Q4 field-goal state shifts. Six-seed mean 14/14 (adopted 2026-11-02); single seeds 13 to 14.
- Mechanisms run through the engine on three seeds and NOT adopted, each losing gates under the Tier B rule (count not lower, none lost, a gate moves toward real beyond 2x default SD): no-huddle (13/14/13 to 6/6/7, +3.8 pts a game), urgency, fatigue, stakes (toward real on noise and late R2, away on Q4 slope), offensive-line exit and its redesign, truncation tilt (+0.86 pts a game), learned head with three flags (12 v 13 at seed 11).
- SIM-27 refit bandwidths (bw_v2), measured 2026-11-03, verdict never recorded: 13/14/13 to 13/12/12, pts a game -0.39 away from real beyond 2 SD, mass at 14 toward real. Not adopted under the rule.
- Forecaster (SIM-08, read ROADMAP row): market-anchored run 1,537 games: raw simulator worse than served (-0.0136 log loss) and than the market histogram; blend +0.0015 (P+ .86) over a served-only recalibration; cover logit inverted in all folds; mass at 3 short (.130 v .151); Amendment 4 full run (12 to 30 h) not run.
- No simulator process running; 24 logical cores, 38 GB free (measured 10:30 ET).

## Units
- S1 inventory and plan: DONE, docs/sim_finish_inventory.md.
- S2 heavy queue, first item: record the SIM-27 verdict, then the joint truncation-plus-bandwidth arm the SIM-27 lane declared (the two fixes move scoring in opposite directions), seeds 11-13. Running. One heavy job at a time, always.

- Launched 2026-11-11 about 10:45 ET beside S2: L3 SIM-30 key-number mass and grade from the existing market-anchored margins (scripts/sim30_margin_mass.py); L4 SIM-12 quarterback exit hazard and effect (scripts/sim12_qb_exit.py); patch prep for SIM-14 coach aggression and SIM-18 lead protection (patches under tests/scratch/sim_finish/, not applied); bookkeeping of ROADMAP rows 06, 09-13, 16, 23, 24, 26, 27 from the inventory. Not yet launched: L1 SIM-28 timeouts, L2 SIM-25 stratum counts, patch prep for SIM-15, 17, 19, 20, 26.
- Engine file rule while any simulation runs: nobody edits scripts/sim10_drive.py, mod25_generator.py or sim10_situational.py; patches are prepared as files and applied by the orchestrator between runs, flag-off identity hash first.

## Tried
- See State; nine mechanisms measured on real plays, hooked behind flags, none adopted.

## Next (S1 done 2026-11-11 10:40 ET; detail in docs/sim_finish_inventory.md; counts: adopted 2, not adopted 6, no hook warranted 3, hook not built 6, hook built not run 1, not started 6, forecaster 1)
(a) Heavy queue, one job at a time, Tier B = 3 seeds 11-13, workers 11, ~15 min per arm; rule: gate count not lower, none lost, a moved gate toward real beyond 2x dflt SD, else not adopted:
1. S2 (running): dflt rerun s11-13, then trunc_bw (SIM10_TRUNCTILT=1, `--h artifacts/sim10/bandwidths_week_v2.json`); `tests/scratch/tierb_cmp.py trunc_bw 11,12,13`. Decides SIM-21, 23, 27.
2. SIM10_TERMINAL identity (tests/scratch/engine_queue/run_identity.ps1) then Tier B of SIM10_TERMINAL=1 (A4 needs SIM-11 14 gates to pass): ~20 min.
3. SIM-08 A4 smoke, 1 core ~7 min (spec-a lane line 83).
4. Hook arms, each needs a root engine edit with flag-off hash = HEAD first: SIM-15 wind, SIM-20 OC halftime, SIM-14 coach latent, SIM-18 lead 1-8; SIM-22 logging and SIM-25 restricted pool as small runs. ~15 min each.
5. SIM-08 A4 full run, 12-30 h, `$env:SIM10_TERMINAL='1'; .tools\uv.exe run --no-sync python scripts\sim08_market_anchored.py full --run amend4 --arms primary --workers 4 --draws 372 --amend4 1`; then `scripts\sim08_evaluate.py --run amend4 --arms primary`. Blockers cleared by 2-3; no SE floor or missing function found in lanes.
6. SIM-30 grade from the A4 margins (light); SIM-29 small run after.
(b) Light units now, one agent each: L1 SIM-28 timeout tabulation; L2 SIM-25 stratum counts; L3 SIM-30 key-number mass by spread band from artifacts/sim08_market_anchored/main/primary_*.npz; L4 SIM-12 QB exit hazard (OL done); L5 ROADMAP rows 12,13,14,15,18,23,24,26,27 corrected to lane facts; L6 recording SIM-13, 16, 24 verdicts.
(c) Finished, record verdict: SIM-10, 11 adopted; SIM-09 not adopted (four arms); SIM-12 not adopted (ol_exit, ol_exit2); SIM-13 no hook (unresolved_below_power); SIM-16 wrong sign, no hook; SIM-24 start field 13% of lead gap, no hook; SIM-06 head not adopted (12 v 13).

## Open
- None.
