# SIM-09 Tier A runs 2026-11-03

## Goal
Run Tier A e5 (old engine, SIM_FAST=1, 11 workers) for u4e, seed 11, compare to crzhk e5, apply the declared adoption rule (lane sim09-urgency-stakes.md). Free the heavy slot by 11:50 ET.

## State
- Started 10:49 ET: sim09_u4e.py e5 --variant u4e --seed 11 --workers 11, log artifacts/sim09/u4e/e5_u4e_11.log, result artifacts/mod25e3/e5_u4e_s11/e5.json (expected).
- Comparator exists: artifacts/mod25e3/e5_crzhk_s11/e5.json.
- Free RAM at start about 30 GB.

## Tried
Returned before completion: I cannot wait for a notification in this harness. Run left RUNNING deliberately (background task bbv7ihnrn, started 10:48 ET, due about 11:25-11:35). Orchestrator: when e5.json lands, compare vs crzhk s11, or kill the python.exe tree by PID if the 11:50 deadline nears.

## Next
Compare e5 gates u4e vs crzhk s11; rule needs seeds 11-13 (this is 1 of 3); then seeds 12, 13.

## Open
Seed SDs for the rule: I2 table (margin .08, nonstr 2.5, r2 .005, strength .5, noise 3.0, xq 1.8, Q4 slope .002, drives .04, pts .12).

## Result (measured, seed 11, finished 10:53 ET, 265 s, 11 workers)
u4e v crzhk: margin SD 14.92 v 15.08; nonstr 210.1 v 206.8; r2 w10-18 .069 v .107; strength RE 35.8 v 47.9 (real 52.3); noise 187.6 v 179.8; xq cov 3.5 v 2.4; Q4 slope -.0395 v -.0408; drives/g 22.88 v 22.84; pts/g 41.8 v 42.5; home edge 1.63 v 1.82.
Rule at 1 of 3 seeds: strength moves away from real by ~24x seed SD, r2 down ~8x SD, noise and pts away beyond 2x SD: condition 3 fails on this seed; no gate pass/fail count computed. Not adoptable on seed 11; seeds 12-13 not run. Heavy slot free (no process left).

## Update 10:55 ET (second agent)
crzhk s12, s13 exist; RAM 36 GB free. Started u4e seed 12 at 10:54 ET (background task b3enweq25, log artifacts/sim09/u4e/e5_u4e_12.log, result artifacts/mod25e3/e5_u4e_s12/e5.json). Order remaining: u4e 13, crzf 11-13 (e5_crzf_s11-13 already exist, check before rerun), crzs 11-13 (exist too), crzkl 11-13. No run starts after 11:42 ET. Kill by PID at 11:50.

## Update 11:10 ET
u4e s12 (294 s) and s13 done. u4e v crzhk, s12/s13: strength RE 37.9/37.1 v 47.0/47.9 (real 52.3, away ~18-22x SD .5); r2 w10-18 .080/.065 v .098/.106; pts 41.9/41.8 v 42.6/42.4; noise 186.8/186.1 v 184.7/179.4 (s13 +2.2x SD away from? real; not computed); margin SD 15.01/14.94 v 15.22/15.09.
u4e VERDICT (3 seeds, rule condition 3): FAIL, strength RE, r2, pts move away from real beyond 2x SD on all three seeds; not adopted (no rule-correction justification declared for u4e; conditions 1-2 gate pass counts not computed). Unresolved -> weak-signals record is orchestrator's.
11:10 ET started crzf s11 (task bpoms6o4z, log artifacts/sim09/crzf/e5_crzf_11.log). Next crzf 12, 13, crzs, crzkl.
