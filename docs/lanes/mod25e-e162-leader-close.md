# mod25e-e162-leader-close

## Goal
Find why sim leaders gain more yards / pass more than real (Q4 close and blowouts); default-off hooks fitted LOSO; deferred crH install patches. Offline; no existing scripts/ edited (except mod25e_ldr.py).

## State (measured 2026-10-18; tests/scratch/e162/)
- Q4<=120 leaders sim v real: pass share .182/.103/.064 v .124/.082/.058 (lead 1-3/4-8/9-16); run yds 3.72/3.85/3.71 v 2.64/2.94/2.65; FD .265/.256/.265 v .153/.160/.169. Pass share carrier = engine draw; LDR fixes it.
- FAILED pool-feature variants (none installed): E162c phase/time scale, E162d def_to, E162e K. Unused names GEOPH GEOTO GEOK.
- E162f PASS: regime class-ratio reweight of Geo run outcome, phase 3 (Q4<=300) leads 1+, 12 classes, cells lead{1-8,9+} x time{<=60,60-120,120-180,180-300}, lam 1 (9/9 folds). Held-out LL/query +0.0561 [0.0492,0.0633], in-sample +.0668, gap .0107. Late-leader run yds <=120: real 2.82/3.26/2.95 v cur 4.26/4.39/4.35 v adj 2.71/2.79/2.79. 12 E161 buckets: 0 IN->OUT. Files: scripts/mod25e_rg.py (RGR=1, suffix rg, fit artifacts/mod25e3/rg/fit.json by rg_fit.py), rg_pool.py, rg_an.py.
- E162g (declared before run: run+pass reweight, leads 9-16/17-23/24+ x Q3/Q4, run in phase 3 excluded as RG territory so zero overlap; 12 combos = {lead 1 bin or 3 bins} x {Q3|Q4, or Q3|Q4>300|Q4<=300} x lam{1,5,25}; nested LOSO; pools K=200; 25,741 queries run 13,738 pass 12,003). Chosen 9/9: lead 9+, Q3|Q4 split, lam 1. Held-out LL/query +0.00575 [0.00476,0.00672] P>0 1.000, 9/9 folds (+.0017..+.0089), in-sample +.0076, gap .0018; run +.0057, pass +.0058. Ratio stable (run 10nf .63 [.56,.67], <=0 fd .10, TO .65; pass 1-2 fd .33, TO .75). Effect on target is small: Geo pools do NOT over-gain blowout leaders. 24+ Q4 run yds real 3.91 v cur 4.05 v adj 3.94; pass real 5.62 v cur 5.73 v adj 5.90 (away); Q3 24+ run real 4.54 v 4.04 v 4.14, pass 6.73 v 5.80 v 6.03 (Geo UNDER real). FD gradient 24+ minus 9-16: real -.005 [-.017,.006], cur -.010, adj -.013. 24+ TD/play real .040 cur .037 adj .036; reach/play .118/.114/.115. Verdict: statistically real, practically small, not on the blowout-excess carrier; NOT installed, mod25e_rgb.py not written. Recorded via weak-signals (unresolved_below_power, family mod25e-e162-regime-ratio; sample-games field approximate). E161 12-bucket check not run for E162g (gk covers all 25,765 queries; do it only if installing). Files: rgb_pool.py/.npz/.log, rgb_an.py/.log.
- Implication: the blowout-leader excess (E155/E156b) sits outside Geo outcome pools: play mix, drive-level decisions, or non-Geo sim plays (kneels, punts/FG, clock).

## Hook
- LDR=1 (scripts/mod25e_ldr.py): LOSO LL/play -1.1333 v -1.3650 (9/9); patch after `ge.install_geo()` block (mod25e_crH.py:182): `import mod25e_ldr as ld` / `if ld.enabled():` / `    ld.install_ldr()`; label `if os.environ.get("LDR") == "1": label = label + "ld"`.
- RG: after Geo, LDR, KN: `import mod25e_rg as rg` / `if rg.enabled():` / `    rg.install_rg()`; label `if os.environ.get("RGR") == "1": label = label + "rg"`.

## E163 smoke LDR+RG (s31, 1 world x 4 seasons; tests/scratch/e163/)
- Lead 1-3 Q4<=120: pass .191 [.124] FAIL, run yds 3.43 [2.64] FAIL, FD .243 [.153] FAIL, kneel diff .034 FAIL; leads 4-8/9-16 near real. Single world, SE pass ~.03: noisy miss. Guards moved (mass3 .099 to .154, strength 94 to 58) unresolved. 3-seed script not written.

## E163 3-seed LDR+RG (seeds 11-13; tests/scratch/e163/a1m.py, a1m.out; game-bootstrap 90%)
- Lead 1-3 Q4<=120: pass .189 v real .124 (+.065 [.039,.091]) bar .15 FAIL, outside interval; run yds 2.84 v 2.64 (+.20 [-.37,.63]) bar 3.2 PASS; FD .180 v .153 (+.028 [-.005,.059]) bar .20 PASS; kneel .450 v .482 (-.032 [-.079,.007]) bar .03 marginal FAIL, inside interval.
- Leads 4-8/9-16 <=120 near real except 4-8 pass +.025 [.009,.044]. 120-300 run yds under real (-.30..-.49, outside interval), pass/FD/kneel on real.
- v XFA (s12-13, all vs real): XFA run yds +1.0..+1.4 and FD +.10 over real at 4-8/9-16; LDR+RG cuts them (-.84..-1.45, -.08..-.10). Pass share residual at 1-3 remains: XFA .171, LDRG -> ldrg minus XFA +.020 [.007,.035] (LDRG s12-13 above XFA).
- Looks 24 cells x 2 comparisons; no verdict recorded (no closing ground; pass residual unresolved).

## E162h diagnosis (measured 2026-10-18; tests/scratch/e162h/an.py; 3-seed LDR+RG vs real, lead 1-3 Q4<=120, 403 real/6588 sim plays, game-bootstrap 90%)
- LDR cell = lead bin {1-3,4-8,9-16,17+} x 7 time bins (mod25e_ldr.py:14,36-39; fit.json scheme seven, lam 1) x 9 classes; NO down or distance in the cell, ratios pooled across downs. Not pooled with 4-8; no other hook reorders play type (LDR wraps pick_index_nn_conditioned; phase 3 only).
- Excess is 1st/2nd down: pass share 1st sim .128 v real .029 (+.099 [.069,.127]), 2nd .161 v .049 (+.112 [.082,.143]); 3rd .247 v .248 (-.001 [-.055,.057]); 3rd with def timeouts>0 sim .264 v .488 (-.224 [-.335,-.092]). Dist large +.086 [.057,.117]; short 3rd-down short under (-.12). Defense timeouts 0 v >0 both +.07/+.06 (not the carrier). Real 1st/2nd leaders pass ~4%; down-agnostic ratio cannot cut those without cutting 3rd.
- E162h fit (27 combos: time {two,four,seven} x down {none,12|34,1|2|34,12|3|4} x lam {1,5,25}, nested LOSO): chosen 9/9 seven|12,3,4|lam 1. Held-out LL/query -1.1081 v LDR -1.1333 (+.0253, folds +.0187..+.0349); lead 1-3 <=120 (n 940) +.054. Offline pass there: LDR2 ~.09 v LDR ~.08 v engine .25 v real .05. Files: scripts/mod25e_ldr2.py (LDR2=1, install_ldr2 replaces LDR; label l2 not yet in crH), artifacts/mod25e3/ldr2/fit.json, tests/scratch/e162h/.
- Offline LDR on real states: 1st .055 v real .015, 2nd .084 v .027, 3rd .145 v .208. 10-try cap (ldr.py:196) forces 13% unadjusted in <60 s cells (accept .23), 5% overall; moves pass only +.013: not the cause.
- Sim gap exceeds offline terms: sim XFA 1st-down pass .100, LDRG .128 (LDR did not cut); 60-90 s .135 to .183. Cause in sim unresolved (hook order: lpo/adj/bls/desp/risk wrap before LDR); needs LDR COUNT forced/calls from a real run.

## Next
1. (done above) Diagnose lead 1-3 residual (LDR cell 1-3 <=120) or run 3-seed to resolve noise.
2. Locate blowout excess outside Geo: sim v real play mix and drive outcomes for 24+ leads by play type.
3. Record E162c/d/e/f via weak-signals (unresolved_below_power).

## Open
Pass yards for close leaders unchecked. Team kernel absent from all E162 pools. Fold-s ratio pools exclude {s,u}; inner selection includes held-out inner season (mild).
