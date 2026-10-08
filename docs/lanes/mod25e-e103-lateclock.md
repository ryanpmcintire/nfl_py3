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
- scripts/mod25e_ckc.py (CKC=1, suffix kc; wired in mod25e_crH.py after install_ck, default off): per-second censored hazard per CLK cell, scope hs<=45. LOSO ll v existing pool (artifacts/mod25e3/ckc/fit.json, hs<=45): inc -0.030, term +0.078 (cens .52), run -0.118 (n 1166, cens .18), pass -0.005, spike -0.175. Hazard form LOSES to the pool on run/pass, so not smoked or served. Hook as written is refuted; the mechanism (censoring) is not.

## Next
Censored-pool draw instead of per-second hazard: keep CLK cells, treat el==hs rows as censored (ends the half if hs_row >= hs_sim, else redistribute to the right: resample from rows with el > hs_row, Kaplan-Meier style); LOSO ll on hs<=45 v pool; if it wins, smoke seed 31 into artifacts/mod25e3/smoke/ and score the four gaps + SD, noise, mass3. Delete or replace mod25e_ckc.py hazard code.

## Open
Offense timeout use probability (real ~.6 v sim ~.3 after trailing runs) is a second, CDR-adjacent channel; FG kick gsr<=5 and tied-entry final-3 may follow once the half-end share is right.
