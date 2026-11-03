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

## Update 11:38 ET (adjudication agent, no run started)
Still alive: bash 29152 -> python 29672/33056 running sim09_stakes_gen.py e5 --variant crzs --seed 13 (log artifacts/sim09/crzs/e5_crzs_13.log; e5_crzs_s13/e5.json on disk is dated 2026-10-02, stale until it rewrites). e5.json exists for crzs s11-13, crzf s11-13, u4e s11-13; none for crzkl (no e5_crzkl_* dir). Adjudication of crzf/crzs/crzkl NOT done; u4e verdict above (fail condition 3) not yet recorded via weak-signals record (family sim09_tier_a). Next: after 29152 exits, compare crzf and crzs vs e5_crzhk_s11-13 with the I2 SD table, record u4e/crzf/crzs unresolved_below_power, list crzkl s11-13 as remaining: SIM_FAST=1 F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim09_u4f.py e5 --variant crzkl --seed S --workers 11 (after the Week 9 lock).

## Update 11:38 ET (final for this slot)
Gaps vs crzhk in seed SDs (I2 table), seeds 11/12/13, all measured from artifacts/mod25e3/e5_<v>_s<seed>/e5.json:
crzf: strength RE -12.4/-2.2/-7.5; r2 w10-18 -4.8/-0.4/-1.9; Q4 slope -3.4/-7.0/-6.8; margin SD -1.4/-3.1/-0.7; xq -0.2/-1.6/-2.0; drives +1.6/+0.1/+1.3; pts +0.1/+0.5/+1.3. Verdict: condition 3 fails (strength RE away from real 52.3 on all seeds, Q4 slope shifts). Not adopted; conditions 1-2 pass counts not computed; no rule-correction justification declared.
crzs: strength RE -1.8/+5.1/-1.6; r2 -0.6/+3.8/-1.1; Q4 slope -5.8/+0.7/+0.6; margin SD +1.0/-2.2/+0.7; nonstr +0.8/-2.9/+1.2; noise +1.2/-2.7/+1.1; xq +1.1/0.0/+1.2; drives +2.1/-0.7/+0.8; pts -0.7/-0.9/+1.1. Seed 12 is the outlier (toward real on strength/r2, away on nonstr/noise/margin); seeds 11 and 13 within ~2x SD except Q4 slope s11 -5.8 and drives s11 +2.1. Not adoptable by the rule (no gate moves toward real beyond 2x SD on all seeds); needs real values per gate to judge direction of Q4 slope and drives; pass counts not computed.
u4e: see above, fails.
crzkl s11-13 NOT run (11:38 start would end past safe margin; one seed cannot apply the rule). Heavy slot free. Next: crzkl 11-13 in a fresh slot (sim09_u4f.py --variant crzkl), then gate pass counts for conditions 1-2 and weak-signals record for all four (orchestrator).

## VERDICTS 11:45 ET (measured; means of seeds 11-13 v crzhk; delta in x seed SD; real: strength 52.3, r2 .150, q4 -.073)
| variant | seeds | margin SD | nonstr | r2 | strength | noise | xq | q4 | drives | pts | verdict |
| u4e | 11-13 | -2.1 | +0.7 | -6.4 | -21.4 | +1.8 | 0 | +.2 | -.1 | -5.6 | FAIL cond 3 (strength, r2, pts away from real); not adopted |
| crzf | 11-13 | -1.7 | -0.8 | -2.4 | -7.4 | -0.1 | -1.3 | -5.7 (toward real) | +1.0 | +0.6 | FAIL cond 3 (strength, r2 away); not adopted |
| crzs | 11-13 | -0.2 | -0.3 | +0.7 | +0.5 | -0.1 | +0.8 | -1.5 | +0.7 | -0.2 | no gate moves beyond 2xSD: no demonstrated gain, not adopted (no rule-correction claim) |
| crzkl | none | | | | | | | | | | INCOMPLETE: no e5 dir for any seed |
Gate pass counts not computed (conditions 1-2 not evaluated). Recorded unresolved_below_power, family sim09_tier_a_e5: sim09_tierA_{u4e,crzf,crzs}_vs_crzhk_strength_re (effect = strength RE delta, units slope_difference as nearest). Heavy slot free (29152 exited).
## Remaining runs (after Week 9 lock, one at a time)
crzkl seeds 11,12,13: SIM_FAST=1 F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim09_u4f.py e5 --variant crzkl --seed S --workers 11

## 2026-11-03 gate pass counts, conditions 1-2 (read-only, no sim; script in session scratchpad g.py)
Rule (sim09-urgency-stakes.md line 110) defines no Tier A pass criterion and no stored real interval exists for e5 metrics (inferred). Proxy used, stated: gate passes if |sim - real| <= 2x comparator seed SD (the rule's own cond-3 scale). Real held 2018-25 (measured; e5_real.json real_eval, real_metrics.json, pts from mod25_generator.real_games): margin SD 14.313, nonstr 177.5, r2 w10-18 .1504, strength 52.29, noise 152.36, xq -12.05, Q4 slope -.0728, drives 21.34, pts 45.84.
Result: 0 of 9 gates pass for crzhk, u4e, crzf, crzs on every seed and on the 3-seed mean. Count vs crzhk 0 vs 0 (no fall), no crzhk-passing gate lost: conditions 1-2 hold trivially, so no verdict changes (cond 3 still decides; all not adopted).
Q4/drives mean-of-3 (crzhk -.0402 / 22.818): u4e -.0398 away / 22.813 toward (both ~0); crzf -.0516 toward / 22.859 away; crzs -.0431 toward / 22.847 away. crzhk sits far from real on all nine (margin 15.13, nonstr 207.9, r2 .104, strength 47.6, noise 181.3, xq +4.1, pts 42.5), so a real-interval pass definition could change counts; not computed.
