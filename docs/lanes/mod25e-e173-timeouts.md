# MOD-25e E173 offense timeouts
## Goal
Does the engine under-call offense timeouts late (final-drive reach .30 v .423)? Offline only.
## State
- Mechanism (read): no timeout decision exists. sim04_engine.py:985-1014 draws one real row by kNN (K_STATE=200, down x phase pool, features dist/fp/score/time/300/off_to/def_to, :228-241) and consumes its off_to_used/def_to_used (:1036-1042). Rate = pool mean over 200 neighbours, marginal over play outcome.
- GEO=2 (mod25e_geo.py:250,380-386) keeps off_to_used/def_to_used/clock from pre-repick row i, outcome from row j: marginal rate unchanged, joint with outcome decoupled (GSTOP covers stop class).
- Fit (tests/scratch/e173/fit.py, tab.csv): 2009-17 trailing/tied, in-bounds non-scoring run/completion, Q2 last 2:00 and Q4 last 5:00, offense has >=1 TO, n 15025, real rate .160. LOSO logistic (log secs, scaled score, tied, TOs, q4, run, def TO) LL -.3358 (+yd/fd -.3351) v engine-implied kNN (leave-season-out pool, class-conditional) -.4112 v const -.4398; fit beats engine in all 9 seasons (+.04 to +.09).
- Cell rates real v engine: <=30s .37-.60 v .17-.23; 30-60s .36-.44 v .16-.21; 60-120s .09-.13 v .14-.20 (engine over); 120-300s .016-.029 v .045-.089 (engine over). Engine smooths the steep time gradient (time scale 300, K 200).
## Tried
Looks: 1 logistic family x 2 variants, 1 kNN; no tuning.
## Next (fix spec, not built)
New default-off module scripts/mod25e_toh.py, env TOH=1 (unused per git grep). Wrap Geo.fix/draw: reweight the K pool by w=p_fit(y_row|state,class_row)/p_pool(y_row), LOSO logistic above (named features, stored coefficients); keeps clock_elapsed jointly with the timeout. Deferred crH patch. Offline replay on 2009-17 states: reweighted cell rates vs real table. Smoke pass bar: final-drive reach tie/trail 1-3 toward .423, P(final 3|tied last 5) toward .625, hs0 not worse, guards.
## Open
Whether timeouts alone move reach (E103: not at hs 10-20); effect of removing over-calls at 60-300s.

## E173b (built, offline-verified; not installed in crH)
- scripts/mod25e_toh.py (TOH=1; TOHS=<season> picks per-season LOSO fit, default all-season; fit.json at artifacts/mod25e3/toh/fit.json written by `python mod25e_toh.py`).
- Mechanism: outermost policy wrapper (after Geo and GSTOP2 inside base pol). For in-bounds run/completion (stop==0, no score, no flip) trailing/tied late (phase 1/3, time<=300, off_to>=1): y~Bernoulli(p_fit); donor = random of K_STATE nearest same-phase, same-play-code rows with off_to_used>0 == y; replace clock_elapsed, off_to_used, def_to_used from that one row. So timeout and clock come from one row class; Geo outcome row j and its stop class kept (donors share stop==0); GSTOP2 clock is overwritten only in eligible states, untouched elsewhere.
- Fit: base logistic + 3 time indicators (<=30,<=60,<=120 s of half), 2 looks total (v1 without indicators: cell rates .594/.300/.133/.033 v real .530/.402/.121/.019, outside intervals; LL -.3358). v2 LOSO LL -.3261.
- Replay 2009-17 (measured, seed 0, n 15025): overall real .1601 v toh .1606. Cells real [Wilson] v toh: <=30 .530[.503,.557] v .503; 30-60 .402[.382,.422] v .409; 60-120 .121[.113,.130] v .126; 120-300 .019[.016,.023] v .019. Non-late rows: rule never fires (190122 in-bounds rows out of scope).
- Caveat (measured): donor clock for called timeouts at 120-300 s is 12.5 v real 22.5 s (kNN spans short-time neighbours); y=0 clocks 23.3 v 25.1.
- crH install patch: in mod25e_crH.py right after the `gs2.install_gstop2()` block insert
  `import mod25e_toh as toh` / `if toh.enabled():` / `    toh.install_toh()` (must follow geo and gstop2; ldr/rg later hooks do not touch pol).
- Smoke pass bar (TOH=1 GEO=2 GSTOP2=1 vs same without TOH, same seeds, no sim run yet): final-drive reach tie/trail 1-3 up from .30 toward .423; P(final 3|tied last 5) toward .625; hs0 not worse; Q4 drives/game not below baseline by more than interval; guards (score/margin dist, plays/game, TO-used per game) unchanged within interval. Report COUNT (eligible, called ~ .16).

## Worker-init crash fix (2026-10-18, measured)
- Cause: install_toh read phase, sc_raw, off_to_raw, time_raw from dv._G["tables"]["arrays"], which holds only the fields at sim04_engine.py:795-816. Fix: mod25c_noise.c_init stashes those columns plus def_to_raw in _G["toh_cols"] when TOH=1 (after _G.update); mod25e_toh.install_toh merges them into the arrays dict. Donors and fit unchanged.
- In-process worker init (H_ss path, smoke env, 5m41s) completes: toh_cols 346552 rows each. Guard: mod25e_crH.guarded_init wraps H_budget/H_ss; in a pool worker it prints the traceback, kills the parent, exits 1; in the main process it re-raises.
- smoke_g2sth.sh re-queued at the end of tests/scratch/e178/queue.txt.

## TOH no-effect trace (2026-10-18, read; no run)
- TOH draws the timeout call (t~Bernoulli(p_fit), mod25e_toh.py:162) and takes clock_elapsed, off_to_used, def_to_used from one donor row (:169-172). Engine consumes exactly those keys: sim04_engine.py:1017 policy(...) then :1036-1042 home_to/away_to = max(0, to - drawn["off_to_used"|"def_to_used"]); clock_elapsed at :1024. No later wrapper rewrites them (grep: only mod25e_rtd.py:191 sets off_to_used=0, RTD not in the smoke label; install_log_sync crH.py:285-298 only syncs log cols 6-10). Install order crH.py:233-237 correct. So a drawn call DOES reach engine state; the write path is not the defect by reading.
- COUNT/_STATE (toh.py:15-16, :144-168) are per-process module dicts, never printed or persisted; pool workers lose them. Eligible/called for the smoke is unknown.
- Final-drive score .329 -> .147 (n~70 drives, one world): clock_elapsed replaced on every eligible play (~84% with y=0 donors, whose clock is 23.3 v real 25.1 s; called donors 12.5 v 22.5 s at 120-300 s) so clocks shorten without extra timeouts; clock-end share .575 -> .676 is NOT explained by that. Unresolved; treat as one-world noise until COUNT exists.
## Next (fix spec)
- No engine hook change is proposed: the sink is already drawn["off_to_used"]/["def_to_used"] consumed at sim04_engine.py:1036-1042.
- Add COUNT persistence: in install_toh pol, also tally by (ph, half-secs cell, t) eligible/called/no_donor and the donor off_to_used>0 rate, dump with atexit/ per-task to artifacts/mod25e3/toh/count_<pid>.json (workers), summed by a scorer. Run one single-seed smoke TOH=1 GEO=2 GSTOP2=1 and check called/eligible ~ .16 overall and cells .50/.41/.13/.02 BEFORE any scoring; if eligible~0, debug gate (stop[int(i)], late_mask sign/units, locals_of); if called ok but oto cells flat, scorer-denominator mismatch (scorer cells include out-of-bounds plays; TOH fires only on in-bounds run/completion).

## TOH count dump (done, unrun)
scripts/mod25e_toh.py now wraps dv.d_play_season in install_toh and writes BUD_OUT/toh_count_<pid>.json (total + cells keyed p<phase>|le30/le60/le120/le300/gt300|run/pass; eligible/called/no_donor). Stub-verified only. tests/scratch/e173/smoke_th2.sh queued in e178/queue.txt after smoke_dl.sh. Its output dir label equals the earlier g2sth smoke and overwrites it; the prior dir was renamed to the same name + `_run1` to keep the scored run.
