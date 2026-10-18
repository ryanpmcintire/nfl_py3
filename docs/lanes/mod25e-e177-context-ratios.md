# mod25e-e177-context-ratios

## Goal
One general variant replacing RG+RGT: class-ratio reweight of Geo-served run and pass outcomes over context cells (score band 7 x period 4 x down 3 x play type), nested LOSO, K=200 pools. Offline; no existing scripts/ edited; no simulator run.

## State (2026-10-18)
- Declared before running: cells score {trail17+, 9-16, 1-8, tie, lead 1-8, 9-16, 17+} x period {Q1-Q2, Q3, Q4>300, Q4<=300} x down {1-2,3,4} x {run,pass}; 12 outcome classes as rg. Menu (24 combos): merge {full, score5, down2, score5+down2} x lam {1,5,25} x shrink target {1, play-type x down-group parent ratio}. Challenger E177b (declared before running): Q4<=300 split into <=60/60-120/120-180/180-300 (6 periods); family 48 combos. Comparator RG+RGT refit LOSO (lam 1, lead/depth {1-8,9+} x time 4, phase 3). E162g not in comparator.
- Written: tests/scratch/e177/ctx_pool.py (all run/pass queries, down 1-4, fp 1-99, 9 held-out fold pools; S float32; BelowNormal), tests/scratch/e177/ctx_an.py (nested LOSO, held-out LL v cur v RG+RGT with game bootstrap, in-sample gap, ratio stability, effect cells, E161 12 buckets via gk, runs A then B; untested).
- E177 A (4 periods, lam 25): held-out dLL +0.00642, gap 0.0025; 7/20 E161 IN->OUT; run yds late leaders not matched. FAILS.
- E177 B (7 periods, lam 5, S5, parent): dLL +0.00830 [0.00788,0.00876] 9/9, gap 0.0044; minus RG+RGT +0.00367 (RGT terr +0.00596, RG terr -0.00243 P>0 0.064); split-half corr 0.898; 5/20 E161 IN->OUT. FAILS.
- E177 C (declared: B with down 4 and fp<=10 queries served unchanged; ctx_anC.py, ctx_an_C.log; 24 combos, chose S5 lam 5 parent 9/9): dLL +0.00734 [0.00691,0.00775] 9/9, in-sample +0.01108, gap 0.0037; minus RG+RGT all +0.00270 [0.00240,0.00302], RG terr -0.01013 [-0.01373,-0.00671], RGT terr -0.00075 [-0.00258,+0.00098] P>0 0.255, outside +0.00331; sd log ratio median 0.037, split-half 0.867. Effects: neutral 3rd&7+ run FD .238 v real .290 OUT; 9+/17+ leader 3rd FD IN (all splits); late-leader run yds IN, FD <=120 OUT (.175/.175/.210 v real .130/.138/.163) while ref IN. 4th-down buckets IN (unchanged). E161: 3 of 20 IN->OUT (TD yl21-40, TD yl41-99, pp 11-14 reach) v ref 1. C FAILS bar; mod25e_ctx.py NOT written.
- Recorded weak-signals e177-ctx-ratios-C-no-d4-no-goalzone (unresolved_below_power).

- E177 D declared before running (tests/scratch/e177/ctx_anD.py = C with fitted TD class-10 ratio frozen to 1 in every cell, scoring renormalises outside class 10; Q4<=300 lead-run RG territory served by RG's own lead/time cells refit LOSO, RGT kept as CTX; same 24 combos, same bucket bar).
- E177 D RAN (ctx_anD.py, ctx_an_D.log): chosen S5 lam 5 parent 9/9. Held-out dLL +0.00733 [0.00689,0.00776] P>0 1.000 9/9, in-sample +0.01072, gap 0.0034; RG+RGT +0.00463; D minus RG+RGT +0.00269 [0.00242,0.00299] 9/9 folds, RG territory 0 (RG cells kept), RGT territory -0.00164 [-0.00344,+0.00013] P>0 0.062, outside both +0.00309 [0.00281,0.00338]. sd log ratio median 0.032, split-half 0.887. E161: 0 of 20 IN->OUT (ref 1: 4th d6+ conv). Late-leader run rows now IN (RG cells). PASSES bar. Recorded e177-ctx-ratios-D-td-frozen-rg-kept (unresolved_below_power).

- INSTALLED OFFLINE (2026-10-18): scripts/mod25e_ctx.py (CTX=1; raises if RGT=1; skips down 4, q<=10, and Q4<=300 lead run so RG's wrapper serves it), scripts/mod25e_ctx_fit.py (full-data fit, wrote artifacts/mod25e3/ctx/fit.json, 210x12, TD col 1, provenance 9/9 S5 lam 5 parent). Replay tests/scratch/e177/ctx_replay.py: fold-2013 fit through the installed wrapper on 32788 real query states matched ctx_D_W.npz weights exactly (max diff 0.0; 29382 served, 2599 d4/goal skips and 807 RG-territory rows passed through at 1.0). crH.py NOT edited. 3-seed script tests/scratch/e177/run_cx_3seed.sh (base ...gx2faldrg, cand +cx, RGR=1 CTX=1), NOT launched.
- crH PATCH (apply in scripts/mod25e_crH.py): after the rgt block (~line 234) add
    import mod25e_ctx as cxm

    if cxm.enabled():
        cxm.install_ctx()
  (ctx must install after rg so its Geo.fix wraps RG's; RGT unset). Near line 440 after the RGT label line add
    if os.environ.get("CTX") == "1":
        label = label + "cx"

## Tried
A, B, C failed the E161 bucket bar (TD class ratio and RG territory). D passes.

## Next
1. Orchestrator install (crH.py not edited here): new scripts/mod25e_ctx.py modelled on mod25e_rgt.py (enabled() reads CTX=="1"; FIT artifacts/mod25e3/ctx/fit.json with ratio[cell][12], cell=((code*5+sb5)*7+per7)*3+d3, sb5 from score bands {<=-17,-16..-9 merged to 0; -8..-1:1; tie:2; 1..8:3; 9+:4}, per7 {Q1-2,Q3,Q4>300,Q4<=60,60-120,120-180,180-300}, d3 {1-2,3,4}; ratio fit on all 9 seasons, lam 5 toward parent play-type x down-group ratio, class 10 (TD) column =1). Same Geo.fix weight_fn wrapper as rgt; skip when down==4, yardline-to-goal<=10, or phase 3 & code 0 & score_diff>=1 (RG's cells serve it; require RGR=1 with CTX). Replace RGT in the crH install block (after rgt, line ~234) and add `if os.environ.get("CTX")=="1": label+="cx"` near line 440. Fit script: port ctx_anD.py fit() on full data; needs a ctx_fit writer (not written).
2. Run only after orchestrator approves CPU: e5 with RGR=1 CTX=1 (RGT off) vs RGR=1 RGT=1 paired.

## Open
Neutral definition inferred (|sd|<=8, Q1-Q3). OT in last period. RGT territory CTX slightly below RGT (P>0 0.062): D keeps CTX there; alternative is RGT kept, undeclared.
