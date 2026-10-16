# mod25e-e129-reach

## Goal
Why does N (B + CDW=2, URG2, CKY2, RTD; ...kekuw2tpq2rty2u2) reach range less than B (...kekudwtpq2ugky) on tie/trail 1-3 final Q4 drives: P(reach yl<=40) real .423, B .308, N .253. Also why fewer finals per game (.141 real, .129 B, .119 N). Diagnosis only, no sim run, no script edits.

## State (measured 2026-10-09, seeds 11-13 burn 2, game bootstrap 400 reps 90%; tests/scratch/e129/an1..an10.py + .txt, frames_n.pkl; B frames tests/scratch/e122/frames_ky.pkl; ~260 cells looked at)
- Reach gap N-B -.055 [-.067,-.041]. Pre-range clock deaths +.044 [.028,.059] (.586 -> .630); clock|reach .052 -> .032; punt +.010. Not start states: hs0 60.97 v 60.61, yl0 74.6 v 74.6, plays to death +.24.
- Productivity N-B on the cohort: yds per rp play -.74 [-.92,-.59] (6.32 -> 5.58, real 6.80); first-down rate hs(15,45] -.051 [-.063,-.037] (.374 -> .323, real .392); yds/sec hs(15,45] -.140 [-.169,-.110] (.519 -> .379, real .608). Inc share matches real in cohort (.333 v .328); run yards, run dt unchanged.
- Cause (all Q4 passes, class 3 = completion or sack, N-B): at hs>120 identical (positives 11.58 v 11.58); at hs<=120 positives -0.5 to -1.3 yd every bin; at hs(15,30] sim completion yds exactly -12 is 13.9-15.5% (B 0.0%, pmf 5.4%) and mean class-3 yds 4.9-5.2 vs 9.1 at hs<=15 (not redrawn).
- Source: scripts/mod25e_urg2.py:536 (yards redraw runs on every non-inc k=3 pass because yards_apply), :544 (w = y_cnt[key] + a*Pb, key = hs bin only, a 512), :549-553 (draw ignores down, distance, yardline except hard cut at ymax). b_rows :295-299 clips yards to [-12,60] (YLO line 27) and pools hs>0, but the apply window is hs>LO=15 (line 18). Real class-3 yards<=-12: hs(0,5] 76%, (5,10] 14%, (10,15] 2.8%, (15,30] 1.0-2.7%, (30,120] .7-.9%; the hs<=30 cell holds 6.5% clipped to -12 -> point mass applied only at hs(15,30]. Gap between pmf 5.4% and sim 14% not explained.
- LOSO gain (+.0117/completion, 9/9) was against Pb (hs>120 pooled), not the engine's state-conditional draw; B positives at hs<=120 by yl (20-40, 40-60, 60+) 11.08/12.28/12.66 v real 10.95/12.15/12.83, so the engine draw was already right and N is -1.0 to -1.3 below in each. N pmf is unconditional so it flattens down/distance/field structure.
- Completion dt hs(15,45] +.52 s [+.18,+.89] N-B (12.89 v 12.37, real 11.69): URG2 class 3 elapsed cell pools sacks, completions, sideline stops; out-of-bounds NOT tested (no flag in frames).
- Fewer finals: tie -.0035 [-.0062,-.0009], trail1-3 -.0070 [-.0100,-.0042], lead 1-8 +.0067, lead 9+ +.0058 per game (N-B); total -.0105; hs0<=60 carries -.0058. Inferred: weaker trailing-team late drives leave the leader with the last possession; also oto0 -.041 [-.081,-.008].

## Tried
Frames for N built (build.py), funnel by stage, by clock bin, by yardline, by down, pmf reconstruction from fit_b.json.

## Next
1. Yards redraw: fit on hs>LO only (cell edges inside (15,120]), no clip lump (extend YLO tail or drop end-of-clock class), and replace pooled pmf with a LOSO-fitted tilt of the engine's own conditional draw (quantile map or ratio P_late/P_base per hs bin); score against the engine draw, not Pb. Or set yards_apply False and keep inc-share redraw only (inc fix is the part that matches real).
2. URG2 elapsed class 3 split sack (yds<0) from completion; test out_of_bounds stop as a covariate (field in pbp).
3. Re-run tgt 3-seed; reach target .42, finals per game.

## Open
Sim -12 mass (14%) vs pmf (5.4%): cause unresolved (reading draw_b did not explain). Real cohort n 324 finals, interval wide. Record unresolved with weak-signals record only if a line is closed.

## E130 (URG2=2) in flight 2026-10-16
Script tests/scratch/e114/run_e130.sh (arg `smoke` = 1 world 4 seasons seed 31, log artifacts/mod25e3/e130_smoke_s31.log; then `run_e130.sh full <label>` runs seeds 11-13 -> artifacts/mod25e3/e130_s{11,12,13}.log, e100 boot e130_boot.log, tgt.py -> tests/scratch/e114/tgt_3seed_u3.txt; tgt.py already has reach/score/finals rows). Env = E127 with URG2=2. Expected label ...kekuw2tpq2rty2u3 (read it from the smoke dir name). Smoke launched as bash task b5cbwoef4; python worker PIDs seen 8952/25336 (parent bash 28696), unverified as mine, not killed. Next: when smoke finishes, move the smoke dir to artifacts/mod25e3/smoke/, run full, score vs B/N/real, decide adopt (aggregates not away from real, reach toward .423).

Update 2026-10-16: smoke dir moved to artifacts/mod25e3/smoke/. Full launched as background task bvjvtebeb (out artifacts/mod25e3/e130_full.out); scorer: tests/scratch/e114/tgt_3seed_u3.txt written by run script.
Run is detached bash MSYS PID 40989 (winpid 27124), child python MSYS 40990; wait task bg3vwss34 (tail --pid).
