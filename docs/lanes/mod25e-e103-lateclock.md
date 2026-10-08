# mod25e E103 late clock

## Goal
Name the mechanism behind the late-clock gaps on base crHpqokgndecsmfwtjo2as2ypw2xfdcdtb: Q4 last play starts <=5s sim .311 v real .128; Q2 .405 v .263; FG kick gsr<=5 .224 v .355; P(final 3 | tied entering last 5 min) .497 v .625. Fit a hook (LOSO) if a mechanism is named.

## State (2026-10-08, measured; no smoke run, no hook served)
- Rescore (clock2 on tb logs; artifacts/mod25e3/clock2/clock2_xfdcdtb_e103.txt, clock-ended drives): Q4 last play <=5s real .128 v sim .309; Q2 .263 v .401. e100.txt: P(final 3 | tied entering last 5 min) real .625 v sim .4965 (xfdcd .4819). FG kick gsr<=5 not rerun (E101 .355 v .224).
- Scratch analyses must set CLK_LABEL=crHpqokgndecsmfwtjo2as2ypw2xfdcdtb (default is an old base; first dec/dec2/dec3 runs were discarded). Scripts and outputs: tests/scratch/e103/ (dec.out, dec2.out, dec3.out, dec8.txt).
- All last plays of Q4 (not just clock-ended drives) start <=5s: real .213 v sim .355; Q2 .428 v .491. Kneel is not the cause (last-play kneel share Q4 .592 v .589; kneel clock fit matches its rows).
- Decomposition (dec8.txt, Q4, offense trailing/tied): P(play is the half's last play) run hs(10,20] real .41 v sim .09, (20,30] .38 v .05, (0,10] .75 v .51; completion (10,20] .27 v .11, (0,10] .76 v .47. Elapsed of non-final plays matches (run 7.8 v 7.7 s); offense timeout after a non-final run real .62-.73 v sim .24-.34. Path split for run hs(10,20]: real ends .41 / timeout ~.37 / neither ~.22; sim ends .09 / timeout ~.27 / neither ~.64.
- Candidate mechanism (read, mod25e_clk.py:70-72, 166-232): CLK pool clips real el to hs, so a play that ended the half enters the pool as an exact short elapsed (about 41% of run rows at hs 10-20) and is replayed at a larger sim hs as "neither"; right censoring at half end is ignored for run/pass/inc/spike (kneels use a censored hazard).

## Tried
- Per-second hazard (first mod25e_ckc.py): lost LOSO on run/pass; replaced.
- E103b censored-pool draw (scripts/mod25e_ckc.py rewritten: discrete Kaplan-Meier pmf per CLK cell, el>=hs rows right-censored, mixed to hs-bin KM pmf with the CLK smoothing grid; draw by cdf; applies only to classes with positive LOSO gain; CKC=1, label suffix kc). LOSO ll/row v existing pool, hs<=45, 9 seasons (artifacts/mod25e3/ckc/fit.json, 90 looks): term +0.096 (9/9 seasons, cens .52, n 2276); pass +0.004 (6/9, se .005); inc -0.006 (5/9); spike -0.049 (7/9); run -0.113 (0/9, n 1166, cens .18). Applied: term, pass (pass gain below its se). Run refuted as KM-redistributed: pool keeping censored rows as exact short draws predicts held-out half-end plays better, so right-censoring is not the run-gap mechanism in this form (not closed: no resolved-wrong-sign bound recorded; needs weak-signals record by orchestrator).
- Smoke seed 31, default 8 worlds x 8 seasons by mistake (130 files, moved to artifacts/mod25e3/smoke/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkc_s31). Base smoke is 1 world x 4 seasons (2 after burn), so base numbers are noisy. Scores (CLK_LABEL set; tests/scratch/e103/sc31.py sc31b.py dec9.py; artifacts/mod25e3/clock2/clock2_*kc*_s31.txt): Q4 last-play<=5s real .128 / base .302 / kc .274 (kc 4-season subset .295); Q2 .263 / .424 / .402 (.469); P(ends half | Q4 run hs10-20, trailing/tied) real .41 / base .11 / kc .09 (unmoved); FG-snap gsr<=5 real .095 / base .012 (n 164) / kc .039; P(final3 | tied entering last 5 min) real .625 / base .654 (26 games) / kc .489; kc margin SD 15.55 (real 14.63), noise 180.8, mass3 .129 (real .141); base 1x4 sample 16.40 / 206.8 / .132, not comparable.

## Next
Gaps unmoved because the hook does not touch run. Half-end deficit on run/pass at hs 10-30 remains: test the other channel (offense timeout use after non-final plays, real .62-.73 v sim .24-.34, CDR-adjacent) and the play-selection channel (clock-aware call mix). Do not serve CKC; leave off. Next smoke should pass --worlds 1 --seasons 4 to match base smoke size.

## Open
Term/pass KM gain is small and its target gaps did not move; serving needs a 3-seed run. Orchestrator to record the run result via weak-signals record.
