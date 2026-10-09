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
