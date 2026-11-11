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
- S1 inventory and plan: fills the table below from all 27 simulator lanes. Running.
- S2 heavy queue, first item: record the SIM-27 verdict, then the joint truncation-plus-bandwidth arm the SIM-27 lane declared (the two fixes move scoring in opposite directions), seeds 11-13. Running. One heavy job at a time, always.

## Tried
- See State; nine mechanisms measured on real plays, hooked behind flags, none adopted.

## Next
- From S1's table: the ordered heavy queue and the data-only units that can run beside it.

## Open
- None.
