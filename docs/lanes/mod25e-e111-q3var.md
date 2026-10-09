# mod25e E111 why Q3 final-margin variance is too high

## Goal
Localise the sim excess in Var(margin half->end Q3) (real 50.7 v sim 55.8) and trace it to engine code. Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku seeds 11-13 (39168 games) v real 2009-17 REG (2304). No sim run, no code change.

## State (2026-10-09, measured; tests/scratch/e111/{load,a,b,c,d,e,f,g,h,i}.py and .txt/.csv; 90% game bootstrap, sim-real)
- Var(Q3 diff) real 50.76 v sim 55.79, +5.04 [+2.52,+7.70] P>0 1.00. Q1 diff-var +2.3 [-0.5,+5.2]; Q2 +0.5 [-2.9,+4.0]; Q4 +0.8 [-2.4,+3.6].
- Level, not spread: points/team/quarter Q3 4.74 v 5.14 (+0.40 [+0.28,+0.49]); Q1 4.43 v 4.61 (+0.19 [+0.08,+0.29]); Q2 -0.07 [-0.19,+0.05]; Q4 -0.15 [-0.29,-0.01]. Var(diff)/total pts: Q3 5.35 real v 5.43 sim, so +8.4% points scales 50.8 to 55.0 of the 55.8 (inferred).
- Channel Q3: drives/team 2.841 v 2.950 (+0.108 [+0.083,+0.138], +3.9%); TD/drive .219 v .235 (+.016 [+.009,+.022]); FG made/drive -.013 [-.019,-.007]; pts/drive +.07 [+.03,+.11]; start field position, punt rate flat. Q1: drives +0.119 [.094,.146], TD/drive +.001. Extra drives are the 5th/6th of the quarter (P(5th) .773 v .821). TD excess is in offense leading by 1-8 (.204 v .232, [+.017,+.041]); one of 5 states, a look.
- Broad, not leader-specific: Q3 leader pts +0.51 [.35,.70], trailer +0.28 [.12,.44]; cov(half margin, Q3 change) +2.8 [+0.03,+5.9], mostly lead 14+ (+2.1 [-0.5,+4.4]); leader/trailer receiver both over (+1.6, +1.2). Tied at half (7.7% of games): Var 33.1 v 51.4 (+18.3 [+12.5,+23.8]), 1.45 of the 5.04; 177 real games.
- Pace: rows (incl. no-plays) per quarter Q1 33.95 v 34.75, Q3 34.26 v 35.09 (+2.4%); Q2 40.69 v 39.76, Q4 39.50 v 39.38 (Q1+Q2 sum 74.64 v 74.51). Mean elapsed per row <=60 s equal (Q1 26.31 v 26.39), run/pass equal. Row count shifts from Q2 to Q1 and onward to Q3.
- Clock rows in sim: P(el>60) per row 0.0002 (Q1); real 60<el<=200 rows/game Q1 .107 v sim .009, Q3 .097 v .009; Q2 .091 v .074, Q4 .103 v .113 (CLK restores them). Sim first snap of Q2 18.2 s into the quarter v real 1.3 s, Q4 17.9 v 0.9: sim quarters carry the overrun, real truncate it. Sim Q1/Q3 sum el 917 s v real ~898.
- Mechanism (read): Q1 and Q3 draw elapsed from the cleaned base pool, never from the fitted real draw. scripts/mod25e_clk.py:205 and scripts/mod25e_cdw.py:183 apply only to qtr 2 and 4; scripts/mod25d_variance.py:531 EL_MAX=60 and :534-563 clean_clock replace pool rows with el>60 (legit 60-200 s delays) by class/zone/score cell means; Q1 and Q3 share phase 0 and time_raw (scripts/sim04_engine.py:195-203, :212-213, pool mask :255). Not found: no_play rows are in both (13.4 v 15.3 s mean el, -1.8 s, minor).
- Decomposition of the +0.8 rows/quarter is incomplete: tail removal ~0.2 rows, no-play el ~0.14, first-snap offset ~0.1, remainder ~0.35 unexplained (inferred).

## Tried
Quarter tables; halftime-lead and receiver buckets; offense-state and drive-ordinal splits; gap between drives (flat ~10.4 s, equal); elapsed by class and quarter; real raw el glitch rows (out-of-order play_ids, el up to 3000 s).

## Next
Hook (specify, not wired) CLQ: extend the mod25e_clk.py draw to qtr 1 and 3. Inputs: class x hs bin x score sign x timeouts (as CLK), plus quarter-end truncation el=min(el, time left in quarter) with the real P(snap at quarter break). Target: real el per play 2009-17 REG after dropping out-of-order glitch rows (el above remaining or negative), keeping legit 60-200 s. Design: LOSO by season, held-out log-lik per play of el bins v current cleaned-pool draw, smoothing a chosen LOSO; smoke then 3 seeds, read rows/quarter (Q1 33.95, Q3 34.26), drives/team (Q3 2.841), Q3 var 50.8 as outputs, not targets.

## Open
- Does Q3 TD/drive excess (.016) persist once pace is fixed, or is it the shared phase-0 pool (Q1 TD rate equal, Q3 higher)? Not split.
- Tied-at-half Q3 variance real 33 v sim 51 (177 games) unexplained; permutation null not run.

## CLQ (2026-10-09, built not wired; tests/scratch/e111c/{fit.log,pace.py,smoke.py})
- scripts/mod25e_clq.py (env CLQ=1), fit artifacts/mod25e3/clq/fit.json (`mod25e_clq.py fit`, 3 min, 4536 looks). Q1/Q3 real rows classes inc/term/run/pass (131461 rows; kneel/spike <200 rows left on base pool); dropped 130 rows el>hs, 0 negative, 19 hs<=0.
- LOSO 2009-17 held-out ll per play: gain v untruncated cleaned-pool +0.157 (9/9 folds, 0.150-0.167); v cleaned-pool with el=min(el,hs) +0.0142 (class run .0192, pass .0156, term .0125 9/9, inc .0006 8/9). Truncation alone +0.142; class-only real draw +0.0136, so cells add +0.0006. Chosen cells: hs bins none, sign none (term: sign), timeouts used on the play split, smoothing 128 (run/pass) to 8192; hs bins and Q3 split not selected.
- Draw P(el>60) per run play 0.0031 = real 0.107 rows/game (smoke, no sim).
- Predicted (inferred): covered plays 83.9% of rows; real minus cleaned mean el on covered ~+0.17 s/play -> +4.8 s per quarter -> Q1/Q3 rows about -0.18 (34.57/34.91 v real 33.95/34.26): CLQ fixes under a quarter of the +0.8. Truncation moves the sim overrun out of Q2/Q4 (covered plays are 92-93% of last plays; residual overrun ~18.2 x 0.07 ~ 1.4 s v real 1.3, inferred).
- Wire: mod25e_crH.py after the cdw block (before install_log_sync, line 155) add `import mod25e_clq as cq` / `if cq.enabled(): cq.install_clq()`; label after the "dw" line add `if os.environ.get("CLQ") == "1": label = label + "ql"`.
- Next: orchestrator 3-seed run with CLQ=1; read rows Q1/Q2/Q3/Q4, Q2 first-snap offset, Q3 drives and points. Remaining pace gap (~0.6 rows Q1/Q3) needs another mechanism.
