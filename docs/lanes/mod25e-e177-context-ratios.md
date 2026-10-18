# mod25e-e177-context-ratios

## Goal
One general variant replacing RG+RGT: class-ratio reweight of Geo-served run and pass outcomes over context cells (score band 7 x period 4 x down 3 x play type), nested LOSO, K=200 pools. Offline; no existing scripts/ edited; no simulator run.

## State (2026-10-18)
- Declared before running: cells score {trail17+, 9-16, 1-8, tie, lead 1-8, 9-16, 17+} x period {Q1-Q2, Q3, Q4>300, Q4<=300} x down {1-2,3,4} x {run,pass}; 12 outcome classes as rg. Menu (24 combos): merge {full, score5, down2, score5+down2} x lam {1,5,25} x shrink target {1, play-type x down-group parent ratio}. Challenger E177b (declared before running): Q4<=300 split into <=60/60-120/120-180/180-300 (6 periods); family 48 combos. Comparator RG+RGT refit LOSO (lam 1, lead/depth {1-8,9+} x time 4, phase 3). E162g not in comparator.
- Written: tests/scratch/e177/ctx_pool.py (all run/pass queries, down 1-4, fp 1-99, 9 held-out fold pools; S float32; BelowNormal), tests/scratch/e177/ctx_an.py (nested LOSO, held-out LL v cur v RG+RGT with game bootstrap, in-sample gap, ratio stability, effect cells, E161 12 buckets via gk, runs A then B; untested).
- IN FLIGHT: ctx_pool.py PID 37512 started in background, writes tests/scratch/e177/ctx_pool.npz and ctx_pool.log (est. ~17 min). No results yet.

## Tried
Nothing evaluated yet.

## Next
1. When ctx_pool.log prints "saved", run `OMP_NUM_THREADS=1 ... .venv/Scripts/python.exe tests/scratch/e177/ctx_an.py > ctx_an.log` (background, BelowNormal); fix bugs (unrun).
2. Judge pass bar: CTX held-out LL >= RG+RGT in RG/RGT territory and overall, 9/9 folds, small gap, stable ratios, no E161 IN->OUT, neutral 3rd&7+ run conv toward 26.3, 17+ leader 3rd conv toward 40.0.
3. If pass: write scripts/mod25e_ctx.py (CTX=1, replaces RG and RGT, must not combine with RGR/RGT; pol hook sets cell for all plays, Geo.fix wrapper as in mod25e_rgt.py) plus fit.json, deferred crH patch text, smoke/3-seed bar (cells above, Var(M)/noise, E174 cross-quarter covariance, guards).
4. Record the look family with `nfl-ats weak-signals record` (unresolved_below_power unless a closing ground).

## Open
Neutral definition in ctx_an.py is inferred (|sd|<=8, Q1-Q3). OT folded into period 4. Kill PID 37512 by PID if abandoned.
