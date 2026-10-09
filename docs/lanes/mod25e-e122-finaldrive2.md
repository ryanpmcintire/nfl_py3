# mod25e E122 remaining final-drive gaps (diagnosis only, no sim run, no script edits)

## Goal
Tie/trail 1-3 final Q4 drive on base ...kekudwtpq2ugky s11-13: P(reach yl<=40) .308 v .423, P(clock|reach) .170 v .022, P(score) .220 v .340. Find remaining in-range clock deaths and why fewer drives reach range; specify LOSO hooks.

## State (measured 2026-10-09; tests/scratch/e122/{c,a0..a9}.py, frames_ky.pkl; 400 game-bootstrap reps, 90% intervals; ~400 table cells looked at)
- Baseline reproduced: reach gap .115 [.071,.162]; clk|reach -.148 [-.173,-.121]; score .119 [.073,.159]. Pre-range clock death is the reach gap: sim .586 of finals v real .407 (end clk|nonreach .847 v .706, -.141 [-.191,-.088]).
- In-range sim deaths 264 (of 1557 reach; real 3 of 137): last-play snap in CKY cell (yl<=36, hs<=45) 117; yl 37-40 hs<=45 88; yl<=36 hs 45-60 24; other 4; last snap yl>40 31. Hazard per play yl 37-40 hs<=10/10-20/20-45: .385/.245/.147 (real 0 of 26 plays in yl37-40 hs<=45); yl<=36 hs<=45 in cell .052 (real 1 of 192).
- In-cell deaths are mostly B-draw edge mass, read scripts/mod25e_cky.py:372-382 and 413-421: install builds plan for every class and ignores fit B_final apply (apply False for inc, line 299 only writes it), and the el draw index lim=round(hs) maps to new=hs (death) with mass of pooled survivors at el==hs. P(draw hits full time) measured from fit pools: inc hs3 .80, hs5 .44, hs8 .04; pass hs3 .47, hs5 .53, hs8 .06, hs10 .03; spike hs3 .04; run/kneel rem mode ~.01. Expected edge deaths on sim in-cell plays 84 of 132 observed.
- Reach, hurry-up productivity, Q4 tie/trail 1-3 all drives (a7,a8,a9): yards/sec hs(15,45] .807 v .663 (+.144 [.061,.233]), (45,90] .542 v .448 (+.094 [.044,.146]), (90,180] equal. Components: dt all run/pass (15,30] 7.4 v 9.1, (45,60] 10.7 v 13.0 (-2.2 [-3.2,-1.3]), (60,90] 13.1 v 14.6; (30,45] and (90,180] match; sim jumps at the CKC/URG hs 45 join. Incomplete dt 5.4 real v 7.3-8.5 sim at hs 45-300: CDW inc cells in force because URG excludes inc (mod25e_urg.py apply flags), CLK inc matched (5.6 v 5.3). Incomplete share of passes real .40-.47 v sim .31-.33 at hs<=120, P(+)~1 each bin; sim flat .31-.33 at every hs while real is .33 above 120. Completion yds (15,45] 11.8 v 10.2 (+1.5 [.5,2.7]). Endgame redraw (mod25e_endgame.py:279-298, window hs<=120 via sim09_u4g.WARN_AT) keeps the base row whenever the HGB class equals the base class, so pass outcome under urgency is not redrawn.
- Not carriers: start time (hs0 55.7 real v 61.0 sim), start spot (73.9 v 74.6), starting timeouts (1.22 v 1.25), timeout use per play (.094 v .091), plays to reach (7.46 v 7.48), 20+ yd play share, dt dependence on yardline (sim slow at 45-90 for yl 40-100 equally). Caveat: real inc includes interceptions (~.03/pass); real turnovers on finals .077 v .025 (risk gap, not a reach carrier).

## Tried
Whole funnel reproduced on new base; cell, class, hs, yl, timeouts, pace, class mix, yardage cuts above.

## Next (specs, not wired)
1. CKY: B draw excludes the death bin (el<=lim-1, rem>=1) since A holds P(end); honor apply (inc pass-through). Extend Y to the FG range (real yl 95th pct 41, nested LOSO had chosen 90th=36 on A alone; choose by funnel deaths) and A window HP 45 to 60 (sim hazard yl<=36 hs 45-60 .035 v real 0 of 35). LOSO ll on survivor rows, per-class; smoke s31 funnel check clk|reach .02.
2. Pace: URG-style hs bins fitted to tie/trail 1-3 Q4 at (15,90], continuous across hs 45 join; inc class back to CLK cells (nested: URG inc v CLK -.011) instead of CDW.
3. Endgame hs<=120 pass redraw even when the class matches: kNN row from late real pool (dist, yl, sd, hs) for inc share and completion yards; LOSO ll of inc and yards v base row.
4. Re-run tgt.py/an12.py after 1-3; record unresolved with weak-signals record if any line is closed.

## Open
Real n in cell is small (1 death in 192 plays); power limited. Turnover-risk gap (.133 [.066,.190] at hs0<=15) is untested as a reach carrier.

## CKY2 (2026-10-09, scripts/mod25e_cky2.py, fit artifacts/mod25e3/cky2/fit.json, log tests/scratch/e122k/fit.log, stub.py, inst.py; default off CKY2=1, replaces CKY=1, simulator not run)
- Fix: P(end) is its own draw (u1 < pe -> new=hs); non-end remaining time drawn from rem pmf or censored Kaplan-Meier el pmf truncated to el < hs (no edge bin); per-class apply honoured (non-applied class passes through whole). Selection pop is common to all (Y,HP): Q4 sd -3..0 hs<=90, out-of-cell rows scored by a fixed baseline (el, all-yl, pool).
- Nested LOSO choice: Y 40 (6 of 9 folds), 42 (1), 43 (2); HP 45 in 9/9 (HP 60/75/90 never won). Real FG attempt yl percentile: Y36 .909, 40 .957, 45 .990 (90th=36, 95th=40, 99th=45). A: ladder, edges 20, flat parent, five classes, a 2 (cku parent lost). Final: Y 40, HP 45. Applied: run (rem, a 2), pass (rem, hs edge 30, a 32), kneel (rem, in-cell q4, a 32); inc and spike not applied (gain -1.2 P+ .27, -10.8 P+ .09).
- Gain v no-cell/current proxy (nested, 9 folds): run +22.4 nats P+ .94 6/9; pass +6.9 P+ .80 5/9; kneel +41.4 P+ 1.0 8/9.
- v CKY=1 on its cell (yl<=36, hs<=45, 328 held-out plays, 0 deaths): total -16.8 nats P+ .08 3/9 (inc -1.2, run -3.3, pass +0.6, kneel -1.6, spike -11.2). Applied classes alone -4.3 (tie); loss is the unapplied spike/inc. CKY=1 per-row likelihood is lenient to its death-bin mass (uncens rows only), so this is a likelihood tie, not a sim equivalence; the sim fix is the death edge.
- Stub: draw P(new>=hs) 0 at hs 3/5/8/10/20/40 for run/pass/kneel (CKY=1 .47-.80 at hs 3); sim pol harness (inst.py) hs 8 pass yl 30: 9 of 4000 end (pe .0028). LOSO pe by bin: yl<=40 hs<=45 .0014-.0026 (real 0 of 336); yl 40-45 hs<=20 .03-.04 (real 1 of 24).
- Looks: 21600 (40 cfg x (60 A x 5 a + 5 cls x 24 B x 4 a)).
- Wiring (orchestrator, not applied): scripts/mod25e_crH.py after the RTD block, before install_log_sync(dv): `import mod25e_cky2 as cy2` / `if cy2.enabled(): cy2.install_cky2()`; label `if os.environ.get("CKY2") == "1": label = label + "k2"`; run with CKY unset.
- Next: smoke s31 with CKY2=1 and tgt.py funnel (clk|reach target .02, in-range deaths vs real 3/137); then specs 2-4 above.
