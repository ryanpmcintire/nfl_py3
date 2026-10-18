# mod25e-e160-reach-x2

## Goal
Decompose the X2 reach gap on tie/trail 1-3 final Q4 drives (X2 .303 v real .423, B .308) into start spot, plays, seconds per play, timeouts, yards; name the carrier; deferred hook if a mechanism is found. Offline only, no sim run, no edits to existing scripts/.

## State (measured 2026-10-18; seeds 11-13 burn 2, X2 label ...rtgw13gx2, game bootstrap 400 reps 90%; ~170 cells looked at in 6 scripts)
Files: tests/scratch/e160/{frames_x2.pkl (built with tests/scratch/e113/build.py), an1..an7.py, an1/an2/an3/an6.txt}. Cohort n: real 324 drives, X2 4946.
- Reach .423 v .303 (-.120 [-.170,-.075]). Start spot is not it: yl0 73.9 v 74.7 (+.8 [-.4,+2.2]); hs0 55.7 v 67.0 (sim starts with MORE time, +11.3 s [+5.6,+16.9]); plays 4.53 v 4.40 (-.13 [-.45,+.20]). Gap sits inside hs0 bins: (60,120] .847 v .488 (-.360 [-.434,-.279]), (0,60] .214 v .111, >120 1.000 v .824. Clock end without reach .407 v .601 (+.193 [+.149,+.238]).
- Pre-range per play (S-R): yds per run/pass play -0.74 [-1.24,-0.23] (comp -1.66 [-2.40,-0.93]); gain>=10 rate .361 v .296 (-.065 [-.095,-.039]); seconds per rp play +1.68 [+0.69,+2.62]; inc share equal (.220 v .209), comp share equal.
- Seconds: incompletion (yards 0 pass) pre-range 4.97 v 7.56 (+2.59 [+2.17,+3.03]); by time left: (60,120] 5.92 R, 8.85 B, 11.01 X2 (X2-R +5.09 [+4.18,+5.93]; X2-B +2.16 [+1.12,+3.14]); (30,60] 5.51/7.41/8.62. Share of inc plays with dt>=15 s at (60,120]: R .019, B .096, X2 .181 (q90 8.9/14.4/33 s). Completion and sack dt match real within intervals; comp dt<=12 (stop proxy) share .542 v .535.
- Timeouts: output columns show used per drive .23 v 1.09 real (offense .14 v .58) and per-play offense call rate trail1-3 gsr<=60 .049 v .164: ARTIFACT of logging under GEO=2, not engine behaviour. lo_otu/lo_dtu/oto/dto are rebuilt offline in mod25e_f2pr.py:85-110 from the logged idx; GEO=2 logs the repicked row j. Label scan (an4/an5): oto-drop rate in gsr<=120 is .20/.11 on every label up to B and .075/.07 from the first label with g2 on (lo_otu>0 .144 -> .057 at the g2 step, an5.py). Engine consumption uses drawn["off_to_used"] (sim04_engine.py:1036-1042), preserved by Geo.fix. So timeouts do not explain reach; the E129 timeout and pstop frame fields under GEO are void.
- Carrier: scripts/mod25e_geo.py:380-386 (Geo.fix keeps ENGINE_KEEP clock_elapsed/off_to_used/def_to_used from the pre-repick row while yards/flip/points/next_* come from repicked row j; pool weights at :365 are team-kernel only, no clock-stop or timeout class). Always-on at GEO=2 via mod25e_geo.py:438,449. Result: yards==0 passes carry completion-length elapsed (inc dt tail), which the sim sees as clock death before range. Non-GEO B already has .096 tail (0-yard completions are part of the sim inc definition; real inc here = complete_pass==0).
- Not isolated: how much of the .120 reach gap this tail explains (B has the same gap .115 with a smaller tail, so it is at most a partial carrier). Yards -0.74/play and first-down rate -.065 are separate, unexplained here; sim hs0 +11 s start is unexplained.
- Out-of-bounds: pbp has no OOB flag or description column (56 cols checked); proxy = completion dt<=12 s, matches real share. Not a carrier.

## Hook (default off, new file, not installed)
scripts/mod25e_gstop.py: GSTOP=1 wraps Geo.fix weights to rows with the same f2h stop class and timeout class as the pre-repick row (fallback to unmasked weights if none; COUNT gives calls and fallbacks). Stub-checked only (weights [1,0,0,0]); not run in the sim.
Deferred install patch for scripts/mod25e_crH.py:
 1. after line 182 (`ge.install_geo()`) add, inside same block level: `import mod25e_gstop as gst` then `if gst.enabled():` `    gst.install_gstop()`
 2. after the `GEOX == "2"` label block (line ~342) add `if os.environ.get("GSTOP") == "1": label = label + "gs"`
Smoke bar (1 world, 4 seasons): inc share dt>=15 at (60,120] <= B .096 and moving to real .019; COUNT fallback/calls < 5%; yards mean per rp play, margin SD, mass3 not worse than X2 by more than the E154 3-seed intervals.
3-seed bar: reach above X2 .303 with 90% game-bootstrap interval, direction toward .423; P(final 3 | tied) and finals/game toward real; margin SD 15.4, noise, pts not away from real v X2 (.15/5.0 intervals as in E130 verdict).

## E160b (measured 2026-10-18; scripts tests/scratch/e160/b1..b10.py; ~60 cells looked at)
- Yards -0.74 is NOT an engine/Geo/SEL carrier. (a) Frame artifact: tests/scratch/e113/build.py:52 applies np.where(flip==1, 0, ...) before the TD branch and the engine logs TD plays flip=1 (581 of 594 po>=6 rows), so sim TD plays have yds 0 and real keep yards_gained. All-state yl>40 rp yards 5.58 sim v 5.97 real; with TD first 5.968 (pool, flip->0, n 97970, also gives 5.573). Logged yards equal pool yards at idx (diff -0.007; shift nonzero on .8% of rows), so no team/yard shift bias. (b) With TD yds fixed, state-level hurry-up yards are at or above real: Q4 0-60 s, sd -3..0: 6.67 v 6.32 (10+ .275 v .293); 60-120 s: 6.51 v 6.21 (10+ .267 v .249); pass share .897/.863 v .914/.884. The engine conditions on score/time (SCORE_CLIP 21 does not bind at 1-3). Geo repick fires only on violation draws (geo.py:331), near the goal, not at yl>40.
- Cohort per-play yards after fix: S 7.00 v R 7.71 (-0.68 [-1.29,-0.17], P(>0) .013); 10+ rate -.064 [-.093,-.037]. Sim state-level is not short, so the cohort gap is outcome selection (final drives that stall pre-range have lower yards; reach .303 v .423): inferred consequence of low reach, not a cause. Unresolved below this.
- Leaders (side finding): Q4 gsr<=120 lead 1-3: yds 5.34 v 3.50, 4-8: 5.08 v 4.08, 9+: 5.14 v 3.43; pass share .222/.120/.049 v .135/.101/.049; kneel-ended leader drives .171 v .055. Real leaders run/kneel; sim leaders gain more (not examined further).
- hs0 +11.3 s: the opponent drive before the cohort drive starts 21.5 s earlier (200.5 v 179.0) and lasts 9.7 s longer (124.3 v 114.6); gap last play to our start equal (9.3 v 8.7). Per-possession cycle (own start to next start, Q4 hs0 60-420): 113.1 v 105.6 s (+7.5), tie/trail 110.6 v 101.7, leaders 4+ 123.5 v 111.9; every end type longer (punt +4.7, fg +10.6, td +4.4, turn +6.0, downs +8.5); s/play 19.78 v 18.15 (+1.63, equals the +1.68 pre-range seconds per rp play). Not a leader kneel/run-out artifact alone; inferred same clock_elapsed carrier family (geo.py:250, :381), so GSTOP=1 is the test; no new hook.
- Pass bar addition for the GSTOP smoke: Q4 cycle s/play toward 18.15 and hs0 toward 55.7 (X2 19.78 / 67.0).

## Next
1. Orchestrator: install patch after current 3-seed runs end; smoke, then 3-seed.
2. Yards: closed as frame artifact + selection (E160b). Fix build.py TD ordering in any rebuilt frame (td=1 -> yds=yl). hs0/cycle: judged by GSTOP smoke above.
3. Rebuild frames for any timeout/pstop analysis from engine state, not from logged idx under GEO.

## Open
Share of reach gap due to the inc clock tail unmeasured below the hook run. Real cohort n 324.
