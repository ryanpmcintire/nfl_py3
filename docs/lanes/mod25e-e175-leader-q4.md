# mod25e E175 leader Q4 300-900 conversion excess

## Goal
Decompose sim v real 17+ leader (offence leading) Q4 gsr 300-900 conversion excess; name carrier; fit one behavioural mechanism LOSO if one fits. Offline, base XFA+LDR+RG s11-13 v real 2009-17.

## State (measured 2026-10-18; tests/scratch/e175/*.py/.txt; game bootstrap, 90%)
- Caveat: sim play-log `yards` is not served yards; position uses logged yl differences.
- Excess pts/drive beyond start-yl value sim-real: 1-8 -0.09 [-0.22,+0.02]; 9-16 -0.05 [-0.20,+0.10]; 17+ +0.26 [+0.12,+0.41] P.99. TD share 17+ +3.6pp [1.3,5.7]; RZ TD/trip +5.4pp; FG range not carrier.
- 3rd conv 17+ play level sim-real +5.0pp [+2.3,+7.5] (drive level +7.0); within-distance effectiveness, not mix or play calling. RECORDED weak-signals id mod25e-e175-leader17-3rd-conv-gap (unresolved_below_power, no closing ground).
- E175b: not strength (+0.5pp [-0.2,+1.2]), personnel (backup QB 12.4%, wrong sign), or real pool share. Real-only LOSO leader dummy -1.1pp, held-out dLL P+ 0.25.
- E175c: GEO=2 force repick of all run/pass plays is the largest step (B->D1), outside red zone (mod25e_geo.py:438,450,471; fix :361-400).
- E175d offline Geo.candidates replay (1165 real 17+ queries): leaders uniform 40.8, kernel only 42.6, current tilt 43.2, TILTL 43.7, real 40.1. Yard shift identically 0 (bias 0, gain 0). Tilt carries only +0.6 of +3.9 [0.55,0.64]; tilt-off does NOT remove the excess; TILTL adds +0.5. Carrier: rating kernel x candidate state (only 8% of weight from 17+ states; 17+ candidates convert 39.5, non-leader strong-offence candidates 43.1).
- E175e (336 looks, LOSO): score-first re-rank FAILS, no module. Exact-distance window holds ~100-230 rows so re-rank cannot bring score-matched rows; all folds chose lam=1 K=400; 17+ chosen 43.4 v real 40.1.

## E175f declared design (2026-10-18, before run)
Offline replay tests/scratch/e175/f1.py. Drop exact-distance filter (dexact=None), same k nearest by f3, conv defined against the query distance. Log-weight = 5 fitted terms: candidate distance gap class (1, 2-3, 4+ vs exact) x3, same lead class, same lead class when 17+. Fit LOSO by season on held-out Bernoulli log-lik of real 3rd-down outcomes over leader queries plus a 5000 all-state sample. Looks: 1 family, 5 params, 9 folds, 3 eval sets. Guards: all-state conv gap and yards CDF vs real.

E175f RESULT (measured, tests/scratch/e175/f1.txt; 6052 queries, 1361 leader; game bootstrap 90%): leader sim-real conv gap current +3.02 [0.78,5.08] P+ .98 -> enlarged fitted LOSO +2.11 [0.19,4.08] P+ .96 (unweighted +2.19). Guard FAILS: all-state gap -0.76 -> -1.25, nonleader -1.86 -> -2.23 (pool widening lowers conv everywhere; leader-minus-nonleader excess 4.88 -> 4.34). Yards CDF max diff improves 0.025 -> 0.0085. Fitted theta (full) dist gap 1 / 2-3 / 4+: -0.32/-0.12/-0.27, same lead +0.11, same 17+ +0.46. Recorded mod25e-e175f-enlarged-pool-leader-conv. No module spec.

## Tried
Leader-conservatism / trailing-defense mechanism (real -1pp, no LL gain); tilt off; yard shift off; score-first re-rank. None installed.

## Next
1. Enlarged pool tested (E175f): trims 0.9pp of the leader gap but breaks the all-state guard; not installed. Remaining open: lead-class-built kernel; sim-side (not pool) cause of 17+ conversion.
2. Compare against E177 class-ratio reweight (live competitor). Do not edit scripts/mod25e_crH.py (other owner).

## Open
Whether the sim's 17+ TD-share excess is strength selection (kernel) or a pool conditioning artifact; a kernel built on lead-class rows rather than rating alone is the unmeasured mechanism.
