# mod25e E103 late clock

## Goal
Name the mechanism behind the late-clock gaps on base crHpqokgndecsmfwtjo2as2ypw2xfdcdtb: Q4 last play starts <=5s sim .311 v real .128; Q2 .405 v .263; FG kick gsr<=5 .224 v .355; P(final 3 | tied entering last 5 min) .497 v .625. Fit a hook (LOSO) if a mechanism is named.

## State (2026-10-08, measured)
- Rescore (clock2 on tb logs, scripts/mod25e_clock2.py, CLK_LABEL=tb label; artifacts/mod25e3/clock2/clock2_xfdcdtb_e103.txt): Q4 last play <=5s real .128 v sim .309; Q2 .263 v .401. e100.txt: P(final 3 | tied entering last 5 min) real .625 v sim .4965 (base .4819). FG gsr<=5 not rerun (E101: .355 v .224).
- Scratch analyses MUST set CLK_LABEL=crHpqokgndecsmfwtjo2as2ypw2xfdcdtb (default label is an old base; my first dec/dec2/dec3 runs used it and were discarded). Scripts tests/scratch/e103/dec*.py.
- Decomposition (dec8.txt): non-kneel plays near half end end the half far less often in sim. Q4 trailing run: P(play is the last of the half) hs(10,20] real .41 v sim .09, (20,30] .38 v .05; completion (10,20] .27 v .11, (0,10] .76 v .47; run (0,10] .75 v .51. Elapsed of non-final plays matches (run 7.8 v 7.7 s). Kneel final share matches (Q4 .592 v .589).
- Mechanism (read): CLK clock pool (scripts/mod25e_clk.py:70-72) clips real el to hs, then fold_ll/pool draw (:51-110,:166-232) treats a half-ending play (el==hs) as an exact elapsed time of that many seconds; a play censored at hs=12 is replayed as a 12 s elapsed at hs=18, so the half does not end. Kneels already use a censored hazard (KN); run/pass/inc/spike classes do not.

## Tried
Kneel clock (hazard at hs 20-45, dto 0) checked: model matches its fit rows; not the cause.

## Next
CKC hook scripts/mod25e_ckc.py: censored discrete hazard per class and CLK cell, LOSO smoothing, wired after CLK before KN, label suffix kc; smoke seed 31 into artifacts/mod25e3/smoke/.

## Open
FG kick gsr<=5 and tied-entry final-3 gaps may follow once runs/completions end halves correctly; rescore after the smoke.
