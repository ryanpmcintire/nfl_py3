# mod25e-e99-regwalkoff

## Goal
Games tied entering the last 5 min of Q4 end at |m|=3 in real v sim (base crHpqokgndecsmfwtjo2as2ypw2). Name the late-regulation mechanism; fit a fix (LOSO, no constants). OT analogue fixed by WFG (docs/lanes/mod25e-e97-ot.md). Parent: docs/lanes/mod25e-generator-fidelity.md.

## State
2026-10-05 Next 1-2 done (analysis only, no engine edits). scripts/mod25e_e99.py -> artifacts/mod25e3/e99/e99.txt (151 looks, one family, bootstrap 300 games, seeds 11-13, real REG 2011-17).
Measured: tied at gsr 300: real 96 games |m|=3 .594 [.495,.685] v sim 1939 games .476 [.455,.498], P(s>r)=.01; |m|=6-7 real .198 v sim .352 (P(s>r)=1.00).
Tied-offense drives starting gsr<=300: P(FG attempt) real .337 [.295,.382] v sim .234; P(FG|reach yl<=40) .582 v .453; P(TD|reach) .159 v .270; P(FGmiss|reach) .159 v .093.
Last 30 s, yl<=40, tied/trail 1-2: FG on down 2 real .458 v sim .278; down 3 real .638 v sim .290 (P(s>r)=0.00 both); down 4 FG real 1.00 v sim .891, go .084 v 0.
FG kicks with gsr<=5 real .355 v sim .187; kick on down 4 real .593 v sim .717; prev play kneel/spike real .154 v sim .060.
Leading 1-2, gsr<=120: sim TD .017 v real 0 (n 80), minor.
Caveat: sim drive "other" (clock expiry) under-labelled (.06 v .176); classification of regulation-end drives is approximate.

2026-10-05 CDR (clock-drain via offense timeout) measured/fit/installed:
- Measured (scripts/mod25e_cdr_m.py, Q4 gsr<=120, sd -2..0, yl<=50, real 2009-17 v sim ypw2 s11-13): offense timeout use given it has TOs real .67/.60/.53/.32/.17 v sim .29/.40/.37/.33/.26 for gsr (0,10]/(10,20]/(20,30]/(30,45]/(45,60]. On timeout plays real el 17.3 (rem<=5 .59) v sim 12.6 (.23) at (20,30], (30,45] rem<=5 .29 v .07: real lets clock run then calls TO with seconds left; sim draws el independent of off_to_used. Spike share gsr<=10 real .50 v sim .28 (not addressed, KN owns it).
- Fit (scripts/mod25e_cdr.py fit -> artifacts/mod25e3/cdr/fit.json): on offense-TO code 0/1 plays hs<=hz, draw remaining clock rem (not el) from season-pooled real pool by hs band; LOSO ll/row rem -2.537 v el -3.188 (gain +.65, n 1213, 300 looks, scope Q4 all sd, hz 120, edges 10/20/30/45/60, smooth .5). hz>120 comparison void (tail bin).
- Installed: scripts/mod25e_cdr.py install_cdr, env CDR=1, hooked mod25e_crH.py run_with_flags end, label suffix cd.
- Smoke (seed 31, 1 world, 4 seasons, workers 1; moved to artifacts/mod25e3/smoke/): P(snap gsr<=5 | tied/down1-2, yl<=40, downs 2-3, Q4 gsr<=300) off .035 (n 259) v on .056 (n 268); late kick share off .643 (n 28) v on .688 (n 32). Direction right, small n, far from real .39.

## Engine path (read)
- mod25e_endgame.py:279 decide(): only down in (1,2,3), non-OT, inwin (:205; KNW widens to Q4 gsr<=600); class drawn from HGB fit :210-213 on down<=3 rows of all Q4 gsr<=600 (no score-state gate), FG is class 2 (CLS :40). Class then replaced by nearest-row rerow (:223).
- 4th down: mod25e_fourth.py:391-446 decide (FD4), model on yl,dist,time_feat,sd,qtr,timeouts; no gsr<=30 sharpness (go .084 at gsr<30).
- Clock/kneel: mod25e_clk.py install_ck :166, install_kn :403 (kneel/spike classes only).
- Wrappers installed mod25e_crH.py:84-92,98-102,176-192.

## Tried
2026-10-05 Open answered (scripts/mod25e_e99h.py -> artifacts/mod25e3/e99/e99h.txt, base ypw2 seeds 11-13 sim states v real, Q4 gsr<=300, tied/trail 1-2, yl<=40, downs 1-3; HGB refit as mod25e_endgame.py:210-213, in-sample):
- HGB is NOT under-fit: mean HGB FG prob v observed FG share within gsr sub-bands, real / sim: down2 gsr<=30 .471/.448 / .28/.271; down3 .504/.557 / .294/.284; sub-bands (0,5] sim .915 pred v .872 obs, (5,15] .376 v .378, (15,30] .024 v .017. Conditional on state the sim FG share equals the HGB and real.
- rerow cannot change class: mod25e_endgame.py:224 selects Lc==CODE_OF[c]; tofix (:253) only changes timeout class. Only a ~4pt gap in (0,5] (.915 -> .872) is unexplained downstream (feature mismatch pstop/oto at runtime or KN :403, unchecked).
- decide gate :279 and FD4 are not the cause of down 2-3. The cause is state composition: share of down 2-3 states at gsr<=5 real 58/148 (.39) v sim 281/1820 (.15); reweighting sim sub-band FG rates to the real mix gives .449 v real .493 (.277 raw), about 80% of the gap.
- Transition: P(next snap gsr<=5 | cur gsr band, in state) real v sim: (5,15] .60/.36, (15,30] .38/.13, (30,60] .12/.03. Per-play el draw is fine: pol pool emulation (:306-339) median el 8.0 = sim 8.0, real in-state 7.0 (mean 14.5/15.7/14.1). Real in-state stop share .218 v sim .119 (definitions not identical).
- No RWF hazard built: the FG hazard conditional on state already matches; a new FG hazard would be a compensating constant.

## Next
1. CDR on 3 seeds (owner-gated) then residual: spike share, rem draw only moves ~1/4 of gap; (old text) Find why real offenses reach gsr<=5 at down 2-3 more often: compare per-play el and next-snap gsr by (code, stop, offense timeout) in Q4 gsr 5-60 in state; suspect stop-class (incompletion/OOB) frequency and timeout use in the pol class draw (mod25e_endgame.py:306-339 matches drawn stop/cls) or CLK/KN (mod25e_clk.py); fit the missing mechanism LOSO.
2. Check the 4pt (0,5] downstream gap by logging runtime features v dec_feats.
3. Re-run tied-entry |m|=3 and E99 sections B/C/D with the fix on seeds 11-13.

## Open
Which upstream mechanism puts real snaps at gsr<=5 (stop-class share, timeout use, deliberate clock run)? FD4 go share at gsr<=30 (.084 v 0) is a separate small-n item (real n tiny).
