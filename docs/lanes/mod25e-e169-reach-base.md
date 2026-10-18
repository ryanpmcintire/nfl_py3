# mod25e-e169-reach-base

## Goal
Rebuild the final-drive reach decomposition (tie/trail 1-3, Q4 last drive) on the provisional base XFA+LDR+RG (label ...tgw13gx2faldrg, seeds 11-13, burn 2, v real 2009-17); name the carrier; fit the trailer analogue of RG if justified. Offline; no existing scripts/ edited.

## State (measured 2026-10-18; tests/scratch/e169/{a1,b1,p1,p1_an,rgt_fit}.py + logs; game bootstrap 400, 90%; ~150 cells looked at)
- Frames: frames_base.pkl (bd2.py = build.py + dist, dropback-run scramble proxy; TD yards fixed). Cohort n real 324, sim 5484.
- Cohort pre-range plays gsr<=300 (S-R): pass share -.018 [-.042,+.006]; ypa -0.65 [-1.30,-0.13]; pass 10+ -.060 [-.094,-.031]; first down per play -.047 [-.072,-.022]; sack proxy (pass yds<0) +.007; comp dt<=12 equal (-.001); inc seconds +2.25 [+1.88,+2.68] (60-120: +4.53 [+3.71,+5.53]); run seconds +2.8. Scramble real .023 v sim 0 (no logged flag). Start-spot bands: ypa gap -1.02 at yl 40-60, -0.33 at 60-80.
- Prevent (all drives, yl>40): Q4 gsr<=180 trail 1-3 pass gain>0 minus ref (Q3/Q4 >600 s, |sd|<=3): real -.060, sim -.011 (S-R +.049); ypa real -0.69, sim +0.04 (S-R +.72 [intervals not bootstrapped]); tied and lead states same direction. Real defenses give up completions late; sim defense does not drop.
- Pool source (rgt_pool.npz, 19,336 queries, 16,291 pass): Geo pool is team/dist/score/time kNN with no game-state class, so late trailer passes mix rows by dist-score-time only. Carrier: scripts/mod25e_geo.py:364-365 (dd over f3 = dist, score, time; weight_fn team kernel only); RG fixes only run leaders (mod25e_rg.py:47-49).
- RGT declared before run (12 combos = depth {1-8,9+ | 1-3,4-8,9+} x time {4 bins | 2 bins} x lam {1,5,25}; classes as RG; run and pass cells separate; phase 3, trail 1+): nested LOSO chose 9/9 depth{1-8,9+} x time{<=60,60-120,120-180,180-300} lam 1. Held-out LL/query +0.0483 [0.0440,0.0522] P>0 1.000 (9/9 folds +.038..+.058), in-sample +.0578, gap .0095; run +.151, pass +.0297. Pass ratios stable across halves (turn 1.4-1.5, TD .82, 1-2fd .3); run turn 2.4 both halves; one sparse cell (run, 9+, <=60 s, n 255) turn ratio 14.25 unstable.
- Effect on targets (trail 1-3 <=120): per-play TD real .0355 cur .0386 adj .0338; FD .293/.323/.316; reach/play .102/.108/.104. Pool reach is NOT short; adj lowers it and raises turnovers, so RGT will not raise cohort reach (inferred). It is a fidelity fix for yards/turnover classes.
- Verdict on reach carrier: not the Geo pool outcome (pool reach/play .108 v real .102). Remaining: inc/run clock seconds (+2.2..+4.5 s per play, GSTOP-addressed), pass 10+ and FD gap (selection plus pool pass-yds 6.9 v 5.9 at 60-120), prevent absent in sim (completion drop, ypa).

## Hook (default off, not installed)
scripts/mod25e_rgt.py (RGT=1, suffix rq; rt is taken by RTD); fit artifacts/mod25e3/rgt/fit.json from tests/scratch/e169/rgt_fit.py (written, cell_of checked 9/4/15, 16 cells). Patch for mod25e_crH.py after the rg block (line ~202): `import mod25e_rgt as rgt` / `if rgt.enabled():` / `    rgt.install_rgt()`; label near line 391: `if os.environ.get("RGT") == "1": label = label + "rq"`.

## Next
1. Orchestrator: smoke RGT after current runs; bars: trail-late pass yds/FD/turn toward real, reach not required to move, guards (mass3, strength, noise).
2. Prevent mechanism (defense completion drop when leading late) is untested as a fitted term; E162 style class ratio by defense-lead state is not it (state is the same class ratio); next unit: pass-share/air-yards or defender-lead effect in engine draw.
3. Record RGT via weak-signals (unresolved_below_power, family mod25e-e169).

## Open
Scramble share, sacks are not logged in sim. ypa prevent DiD has no bootstrap. 12 E161 buckets not run for RGT.
