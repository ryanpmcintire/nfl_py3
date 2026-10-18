# mod25e-e162-leader-close

## Goal
Find why sim leaders gain more yards / pass more than real (Q4 close and blowouts); default-off hooks fitted LOSO. Offline; no existing scripts/ edited except mod25e_ldr.py and (LDR3 install only) mod25e_crH.py.

## State (measured 2026-10-18; tests/scratch/e162/)
- Q4<=120 leaders sim v real: pass share .182/.103/.064 v .124/.082/.058 (lead 1-3/4-8/9-16); run yds 3.72/3.85/3.71 v 2.64/2.94/2.65; FD .265/.256/.265 v .153/.160/.169. Pass share carrier = engine draw.
- E162c/d/e (phase/time scale, def_to, K pool-feature variants) FAILED the 20-bucket check or did nothing; none installed; not recorded in weak_signals (no admissible effect unit measured, only bucket flips).
- E162f PASS, installed default-off as RG (RGR=1, scripts/mod25e_rg.py, fit artifacts/mod25e3/rg/fit.json): held-out LL/query +0.0561 [0.0492,0.0633], P>0 ~1, 9/9 folds, gap .0107. Recorded (unresolved_below_power, family mod25e-e162-regime-ratio).
- E162g (run+pass reweight, leads 9+): +0.00575 [0.00476,0.00672], 9/9, practically small, NOT installed; recorded.
- Implication: blowout-leader excess is outside Geo outcome pools (play mix, drive decisions, non-Geo plays).
- LDR=1 (mod25e_ldr.py): LOSO LL/play -1.1333 v -1.3650 (9/9), label ld. E163 3-seed LDR+RG at lead 1-3 Q4<=120: pass .189 v real .124 (+.065 [.039,.091]) FAIL; run yds, FD pass; kneel marginal. E162h: excess is 1st/2nd-down pass (sim .13/.16 v real .03/.05); LDR2 (down cell) held-out +.0253 LL, not in crH.
- E162j: LDR arrays fine; ns["DECIDE"] chain re-picks rows after the pick (play type changes 11% of plays, 29% at lead 1-3 Q4<=120) diluting LDR.

## LDR3 (this round)
- scripts/mod25e_ldr3.py: wraps ns["DECIDE"] outermost (LDR3=1, install_ldr3), ratio rejection on DECIDE output class, cell = lead x time x down. `counts` command draws pick+DECIDE 300x per real leader state (phase 3, lead 1+, 2009-2017) from the built sim tables and tallies by row season x class into artifacts/mod25e3/ldr3/decide_counts.npz; `fit` does nested LOSO (36 combos) with season-exclusion exact via the season tally, writes artifacts/mod25e3/ldr3/fit.json. Env: tests/scratch/e162/ldr3_env.sh counts|fit.
- LDR3 refit DONE (14,122 leader queries x 300 draws, 44 min; artifacts/mod25e3/ldr3/fit.json, chosen 9/9 two|d12_3_4|lam 25): held-out LL/query adj -1.6395 v raw DECIDE -1.6404 (+0.0009, folds +.0013 +.0020 +.0009 +.0033 +.0016 -.0015 +.0013 0 -.0004; lead1-3<=120 n940 +.0031/query) v old LDR ratios applied to DECIDE -1.6603 (+.0208). Offline DECIDE pass share at lead 1-3<=120 is already ~.07 (adj ~.065) v real ~.05: the sim excess (.19) is not in DECIDE-at-real-states; look at sim state path. Counts harness uses fake caller frame (possessions=10, task_key (0,0,0)).
- crH patch installed default-off: mod25e_crH.py install after install_log_sync(dv) (:250-256) and label "l3" before "rg" (:444-447); raises if LDR=1 too.
- 3-seed script written, NOT launched: tests/scratch/e162/run_ldr3_3seed.sh (cand ...gx2fal3rg v base ...gx2faldrg).

## Next
1. Orchestrator queues run_ldr3_3seed.sh (needs artifacts/mod25e3/ldr3/fit.json from `fit`); compare lead 1-3 pass/run yds/FD/kneel to a1m.out.
2. If adopted: LDR3 supersedes LDR for the base.

## Open
Which DECIDE layer does the re-picking at leaders is not isolated. DECIDE chain params are not refit per held-out season (counts leak none from the pool rows; chain-internal fits are in-sample). Pass yards for close leaders unchecked.
