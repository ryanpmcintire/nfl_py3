# mod25e E113 final Q4 drives of tied/trailing teams (diagnosis)

## Goal
Why final Q4 drives of tied/trailing offenses score about half as often in sim as real (E109: tie .322/.121, trailing 1-8 .202/.103). Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku s11-13 v real REG 2009-17; analysis only, no engine edits.

## State (2026-10-09, measured; tests/scratch/e113/{build,an1..an13}.py and .txt, frames_ku.pkl, frames_dw.pkl)
- Gap is trailing 1-3 and tie only (the drives where a FG ties or wins). Need-FG finals (tie + trail 1-3, n 324 real / 5122 sim) P(score) .340 v .135 (gap .204 [.161,.245], 400 game-bootstrap reps). Trail 4-8 (needs a TD) matches: .071 v .068.
- Funnel (an12): P(reach yl<=40) .423 v .302 (gap .120 [.074,.166]); P(score|reach) .766 v .403 (.364 [.306,.433]); P(clock expiry|reach) .022 v .474 (-.452 [-.478,-.424]); P(score|reach, no clock death) .784 v .766 (+.018 [-.036,+.091]). The kick decision is right; the sim loses the kick to the clock.
- Removing in-range clock deaths closes .54 [.44,.69] of the score gap; the rest (~.46) is reach.
- In range (yl<=40, tie/trail 1-3), P(run/pass snap ends the game): real 1/449 plays at gsr<=60 (.000-.024 per bin), sim .003-.29 (run 10-30 s .17, pass 0-10 s .29, 10-20 s .10; an10, an11). Out of range last 20 s tie/t1-3: real .12-.32 v sim .29-.41. t9+ matches (.26 v .31 in range).
- Kneel: 283 of 432 sim need-FG in-range kneels are the drive's last play (clock gone); real 0 of 31. Real kneel at 9-35 s is followed by FG at 3-6 s (kneel dt = gsr - ~4); sim kneel el is the full ~36-40 s capped at gsr (an6). Kneel-final in range = 5.6% of sim need-FG finals; clock death by run/pass/spike another 8.8%.
- Pace (reach stage, an7-an9): start-time matched, reach (90,150] s .95 v .65, (60,90] .73 v .43; plays 8.7 v 7.2; sec/play 13.8 v 16.2. Contested (sd -8..0) mean dt per play (60,120] 14.45 v 16.73 (-2.28 [-2.73,-1.82]), within-class not mix; sim dt is flat across tie/t1-3/t4-8/t9+ (15.5-15.8) while real is 13.4-13.9 contested v 15.3 hopeless. Incompletion share real .42-.48 v sim .31-.35 in trail/tie at <=120 s, sim flat .30-.35 in every state.
- Not the carrier: start time (sim hs0 equal or later), start spot (+-3 yd), timeouts at start, TO use per play, yards/play (6.6 v 6.2, +.3 [0,.6]), 4th-down FG choice (.93 v .84 in range, small), CDW (dw label) leaves final-drive funnel unchanged (reach .297, clock|reach same).
- Engine: yardline appears in no clock draw. CKU/CKS P(end) ladder mod25e_cks.py:32-55 (band merges trail 1-3 and 4-8, features hs, qtr, band, timeouts, no yl), applied scripts/mod25e_ckc.py:209-240 (pol receives yardline, unused); cku.p_end :152. Kneel el scripts/mod25e_clk.py:393-401 kn_draw and :428 (cells hs bin x pstop x dto only, no sd/yl). Class draw mod25e_endgame.py:212-213 HGB, rerow :223/:235 nearest on [dist, yl, sd/std, hs/std] with no timeouts.
- Real FG attempts in this state: yl 95th pct 41, max 50; FG snap time gsr quantiles 10/50/90 = 2/4/10 s (an13).

## Tried
Start/field position, yards, timeouts, FG choice, CDW: none explain. dw drive stats unchanged.

## Next
Spec (not wired): hook CKY (needs CKU). (1) P(end) ladder gets a kick-preserve level: FG-useful flag sd in [-3,0] x in-range flag (yl<=kick range, range = real FG attempt yl 95th pct, measured), shrinkage as CKS; fit LOSO 2009-17 on kc.rows() plus yl, target play ends half, score binary ll v CKU ladder, per-season stability, look count. (2) Kneel and non-end elapsed in that cell: draw rem (time left) from real pool by hs band as CDR (cdr.py:51-85), LOSO ll rem v el. Check: kill rate by class x gsr bin v an10 (real ~0 in range), then smoke s31 funnel (an12 targets: score|reach .77, clock|reach .02). (3) Pace: urgency-graded elapsed (tie/trail 1-8 v trail 9+) in CDW cells, LOSO ll on el for hs 30-180.

## Open
Does adding the in-range term lower P(end) enough for reach too, or is pace separate? Real n in cell is 449 plays (+31 kneels): LOSO power is limited; pool across Q2 two-minute drill (FG before half) to widen.

## CKY (2026-10-09, scripts/mod25e_cky.py, fit artifacts/mod25e3/cky/fit.json, log tests/scratch/e113k/fit.log, stub tests/scratch/e113k/stub.py)
- Default-off hook (CKY=1), not wired. Cell: qtr 4 (A) , sd -3..0, yl <= Y, hs <= W. Y = FG-attempt yl percentile (Q2+Q4, sd -3..0, hs<=120), pct chosen by nested LOSO: 90 in all 9 folds -> Y 36 (95th would be 41).
- A (P(end), hs<=45, ladder = CKU p as parent, (k+a p)/(n+a)): nested LOSO 329 held-out in-range Q4 plays, 1 expiry; ll gain v CKU on 170 run/pass rows +6.46 nats, 8/9 folds; v pooled-band parent +61.1, 9/9. Looks: 540 A specs x 5 a.
- B (rem/el draw after non-end): W chosen 45 by summed positive per-class gain (W 45/60/90/120 = 49.4/9.5/34.2/24.6 nats). W=45 gains: run +30.2 (5/9 folds, rem), kneel +16.3 (6/8, rem), pass +2.4 (4/9), spike +0.5 (4/9), inc -0.8 (not applied). Per-class gain sign flips across W (run W60 -16.7): B is weak/unstable. Baseline = el-mode on all-yl rows (proxy for current, not the true CLK/CKU pmf).
- Stub (no sim): in range hs 20, yl 30, P(end) .02 run/.02 pass/.015 kneel/.011 spike; kneel median rem 4 s; out of range and hs>45 pass-through unchanged.
- Next: orchestrator wires and smoke-tests (s31 funnel: score|reach .77, clock|reach .02).

## URG result (2026-10-09, measured; scripts/mod25e_urg.py fit|install_urg, artifacts/mod25e3/urg/fit.json, log tests/scratch/e113u/fit.log; default off URG=1, NOT wired, simulator not run)
- Why the gradient is lost (read/measured): base env (no dw) draws elapsed for hs>45 in CLK install_ck (clk.py:290-330; CKC covers hs<=45 only). Cells hs bin x sign band (<=-9, -8..-1, 0, 1..8, 9+: merges trail 1-3 with 4-8) x timeouts, shrunk to the hs-bin marginal with a = run 128, pass 512, inc 8192; median cell weight n/(n+a) in Q4 (45,180] is run .44, pass .33, inc .22. CDW (dw) uses sign3 (trail/tie/lead) only, qs, used flag, a 128: no deficit band at all.
- Model: parent = class x timeout-used(3) x 15 s hs bin, Dirichlet to class x used marginal (a0 grid 8..2048, nested), times exponential tilt exp(theta_g e) per group g (band sign3/trail5/full9 x offense-timeouts>0 x hs>120 x ridge 1|1e4; 24 schemes). LOSO 2009-17, scheme and a0 chosen in an inner LOSO on the 8 training seasons, 62-bin ll, window (45,180]. 192 looks (24 schemes x 4 classes x Q4,Q2).
- Nested ll gain per play, URG v CDW (v CLK-with-no-used-flag): Q4 run +.053 [.041,.065] 9/9 (+.300); Q4 pass +.041 [.032,.050] 9/9 (+.096); Q4 term +.058 8/9 (+.044); Q4 inc +.042 9/9 v CDW but -.011 v CLK 1/9 (not applied); Q2 run +.096, pass +.057, term +.069 (9/9), Q2 inc not applied. All P(positive) ~1.
- Tilt alone over the same parent (urgency term): Q4 run +.0435 [.038,.050] 9/9; Q4 term +.014 P .99 7/9; Q4 inc +.012 9/9; Q4 pass +.004 [-.001,+.008] P .94 7/9; Q2 run -.0005 P .25, Q2 pass -.0001 P .43, Q2 inc +.005, Q2 term +.007. Urgency is a Q4 run/term effect; Q2 gains are from the used-flag/parent, not urgency. Fixed-v-nested in-sample gap .0006-.0085.
- Mean-seconds bias, Q4 contested (60,120] sd -8..0, actual / CLK / CDW / URG: run 18.8/23.7/21.1/18.65; pass 18.8/19.8/18.7/18.3; inc 5.3/5.6/7.2/5.4; term 8.6/9.9/10.6/8.4. Run by deficit (actual / CDW / URG): trail 17+ 26.0/25.9/25.8; 9-16 17.2/26.1/17.1; 4-8 20.3/24.1/20.3; 1-3 20.9/22.1/20.9; tie 18.1/19.9/18.1. Real run clock is fastest at trail 9-16 and slowest at trail 17+ (hopeless); CDW is flat.
- Residual gaps: Q4 lead 9+ pass 25.4 actual v 22.8 URG (n 116); Q4 term lead1-8 14.4 v 10.9 (n 91).
- Applied classes (apply flag: nested gain v CLK >0, P>.5, wins>=5): Q4 term/run/pass, Q2 term/run/pass; inc excluded (CLK stays). Scheme Q4 run full9|to0|h1|l1 a0 128; Q4 pass full9|to0|h1 a0 128.
- Wiring (orchestrator): in scripts/mod25e_crH.py after the CLQ block, before install_log_sync(dv): `import mod25e_urg as ug` / `if ug.enabled(): ug.install_urg()`; label `if os.environ.get("URG") == "1": label = label + "ug"` after the CLQ "ql" line. Env: base + CDW=1 TPO=1 CLQ=1 URG=1 (URG overrides its cells last; TPO range hs 300-900 does not overlap).
- Next: smoke s31 with URG; check sec/play by state (real contested 13.4-13.9 v 15.3 hopeless), reach (90,150] .95 real v .65 sim, plays 8.7 v 7.2 (an7). If reach stays short the remainder is class mix (incompletion share .42-.48 real v .31-.35 sim, inc not covered here) and the P(end) ladder (lane Next 1).

## Smoke s31 1x4 (measured 2026-10-09; tests/scratch/e114/tgt_uk.txt) dw -> +TPO+CLQ1 -> +URG+CKY
Final drive tie/trail 1-3: P(score|reach) .432 -> .448 -> .690 (real .766); P(clock expiry|reach) .364 -> .483 -> .119 (.022); P(score) .147 -> .110 -> .231 (.340); reach .324 -> .228 -> .313 (.423). Trailer 9+ plays/game 11.20 -> 11.27 -> 11.57 (11.61). CKY carries the funnel. 3-seed run with CDW TPO CLQ=2 URG CKY launched (tests/scratch/e114/run_3seed.sh; logs artifacts/mod25e3/e114_s1{1,2,3}.log, e114_boot.log, tests/scratch/e114/tgt_3seed.txt).
