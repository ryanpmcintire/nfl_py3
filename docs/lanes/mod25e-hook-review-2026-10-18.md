# Hook review 2026-10-18 (read-only)

## Goal
Correctness review of default-off MOD-25e hooks and crH install order.

## State
Reviewed gstop2, toh, v1ad, safeg, safegp, stbk, fd5, rgt, rg, ldr, ldr2, tiltl, servedidx, crH. tdr and ldr2 are NOT installed by crH (LDR2/TDR env ignored, label unchanged).

## Findings (ranked)
1. toh.py:171 dv.task_rng("toh", seed) -> int("toh") ValueError on first eligible late play. Breaks smoke_g2sth.
2. gstop2.py:10,34 global counter _STATE["n"] never reset per task: RNG depends on worker task order, not reproducible; paired boots noisy. Affects smoke_g2s, smoke_g2sth.
3. crH install: LDR (pick wrapper) and RG (Geo.fix repick weights) both correct late-lead class mix; double application in every LDR+RGR run (smoke_ldrg/g2s/g2sth/rq/sg/tk, run_ldrg, run_fe).
4. rg.py/rgt.py ratio applies only inside Geo.fix (violating draws), not all draws it was fit on (inferred).
5. safeg.py:44,53 loss.npy built from build_transition_frame(pbp) with no team_ratings; row alignment with engine arrays unverified (smoke_sg).
6. E162i LDR anomaly: arrays are not stale (neg runs before ldr); suspect downstream f2 tostep/cpick re-picks (sim09_f2.py:385) bypass LDR. Unresolved.
7. tiltl.install() and v1ad use "all" fit only (no fold); fine for post-2017 scoring, but TOHS/season fold never selectable from crH.
8. GSTOP and GSTOP2 not mutually exclusive in crH.

## Next
Fix 1-2 (use dv.task_rng with int salt) before running smoke_g2sth/g2s.

## Orchestrator actions 2026-10-18
- Fixed #1: mod25e_toh.py task_rng salt "toh" (int() crash) -> SALT 9901. Fixed #2: mod25e_gstop2.py process-global counter RNG -> per-task dv.task_rng(SALT 9903, cfg seed). Both compile; unused salts checked against all mod25e SALT constants.
- #4 false alarm: GEO=2 forces every run/pass through Geo.fix (E176d measured 97% exact-distance served), so RG/RGT apply to all Geo-served run/pass draws.
- #3 judged unlikely (LDR reweights play type in the pick; RG reweights outcome class in Geo.fix); keep watching late-lead cells in the LDR+RG 3-seed.
- #5 open: verify SAFEG loss.npy row alignment with engine arrays before running smoke_sg.sh.
- #6 forwarded to E162j (f2 tostep/cpick re-pick after LDR).

## #5 SAFEG alignment
Aligned (measured). build_transition_frame with and without team_ratings both give 346552 rows, identical game_id, play_id, down_i, fp_raw, yards_gained, points_def, next_yardline in order; loss.npy length 346552. Ratings merge is a left merge on game_id and runs before the sort, so no rows are dropped. No edit to mod25e_safeg.py; check not re-run.

## Smoke scoring 2026-10-18 (measured, seed 31, 1 world 4 seasons, burn 2, 544 games)
- GSTOP2 smoke (tests/scratch/e163/tgt_g2s_smoke.txt, tests/scratch/e179/cells.py) v base ldrg v real: final-drive reach .282 v .329 v .423 (gap -.141 [-.246,-.059] v -.094 [-.195,+.009], moved away); P(score) .269 v .329 v .340; clock expiry|reach .045 v .042 v .022; Q4 last play <=5s .268 v .250 v .214; FG kicked <=5s (gsr<=60) .295 v .394 v .404 (n 78/66/319, wide). Checks: SD 14.57 v 15.15, mass3 .129 v .154, pts/g 44.35 v 44.16.
- Verdict GSTOP2: no target moved toward real; reach and FG<=5s moved away; no 3-seed script written. Recorded weak-signals mod25e-gstop2-smoke-s31-final-drive-reach (unresolved_below_power; probability_positive 0.5 placeholder, no interval from a 1-seed smoke).
- g2sth (GSTOP2+TOH): dir ...g2sthldrg_s31 has no e5.json; unscored, still pending. Run score: tgt.py base,g2sth then cells.py with label g2sthldrg.

## GSTOP2 mechanism (measured, tests/scratch/e179/fd.py, seed 31, burn 2, game bootstrap 300 reps, 90%)
- Read: mod25e_gstop2.py wraps Geo.fix; after the repick it replaces only clock_elapsed with a random near-set row sharing the repicked row's stop flag and the drawn row's class. Geo.pol runs it before the base policy (mod25e_geo.py:447-471), so pace multipliers and hurry-up edits still apply. Touches every run/pass draw under GEO=2.
- Final drives (tie/trail 1-3), real v base v GSTOP2, n 324/73/78: plays 4.53/4.38/4.74; s/play 12.0/14.0/14.2; start clock 56/67/64 s; reach .423/.329/.282; clock-end share .417/.575/.615; turnover .077/.014/.000; downs .034/.041/.077; score .340/.329/.269; yards per run/pass 6.9/7.1/6.8; offense timeouts used .58/.14/.14.
- GSTOP2 minus base (unpaired): reach -4.7pp [-16.9,+7.3] P(>0) .26; plays +.32 [-.75,+1.3]; s/play +.12 [-2.3,+2.6]; start -4 s [-31,+15]. Nothing shifts beyond noise (SE of reach diff ~7pp).
- Carrier: none found. Ruled out: pace/policy override (fix runs before base pol), plays per drive, s/play, start clock. The reach drop is within one-world sampling noise (different RNG streams give different games).
- Real shortfalls present in BOTH sims, so not GSTOP2: offense timeout use (.14 v .58), sim s/play 2 s slower, turnovers ~0 v .077, clock-end share too high. These are the candidate mechanisms for the reach gap.
- Recorded mod25e-gstop2-smoke-s31-final-drive-mechanism (unresolved_below_power, rate_pp).

## Next (added)
- Default-off spec (named mechanism, no constant): final-drive timeout use by the trailing offense, taken from the real stop rate conditional on state (clock, margin, timeouts left), added to the stop flag; measure first with a 3-seed base-only read of timeout use in Q4 trailing drives. Do not run more GSTOP2 seeds for reach.

## TOH smoke scoring 2026-10-18 (measured, seed 31, 1 world, 544 games; scorers tests/scratch/e173/score_th.py, final3.py)
- Final-drive reach real .423 v base .329 v g2s .282 v g2sth .250 (g2sth-base -8.6pp [-20.4,+3.5] P>0 .12). Offense TOs used per drive real .58 v .137 v .141 v .176 (g2sth-base +.040 [-.077,+.152] P>0 .73). Score .340/.329/.269/.147 (g2sth-base -19pp [-29,-8]). Clock-end share .417/.575/.615/.676.
- Timeout-call cells (offense >=1 TO, trailing/tied, drop in oto): real <=30s .53, 60-300s .069; g2sth .049 and .036 (g2sth-base 60-300 -.7pp [-1.3,-.1]). TOH does not raise calls; engine rate flat. COUNT of TOH eligible/called not logged in output; hook suspected not effective (check COUNT, oto write-back).
- P(final 3|tied last 5): real .581 (my def), base .571, g2s .623, g2sth .543 (n 46). Aggregates g2sth: SD 15.65 (base 15.15), pts/g 44.23, mass3 .112 (base .154), noise var 177.8 (base 166.7).
- Verdict: fails bar; no 3-seed script. Recorded mod25e-toh-smoke-s31-final-drive-reach (unresolved_below_power). Next: diagnose why TOH leaves calls at ~.05 (COUNT, whether oto/lo_otu sees donor off_to_used).
