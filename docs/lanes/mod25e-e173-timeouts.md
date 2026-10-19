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

## TOH rescore (2026-10-18, measured; tests/scratch/e173/score_th2.py)
- Mismatch (read): real .530/.402/.121/.019 = tab.csv, fit.py:30-33,40 universe (run/completion, in-bounds, no score/flip, sd<=0, off_to>=1). score_th.py:39 sim universe = code<=1 (adds incompletions, out-of-bounds) with no stop filter; the sim parquet has no this-play stop column (pstop of the next row is contaminated by the timeout itself), so exact matching is only possible via the TOH counters.
- Logged sim call rate (oto drop on next play), broad / run-or-yards>0: base .046/.048, g2s .041/.044, th run1 = run2 .039/.043 (cells ~.05 flat, 120-300 .031-.043). Real .16 overall.
- Counters (3 workers, 4 seasons incl. 2 scored): called 944 of 7010 eligible; ~470 expected calls in the 2 scored seasons v ~183 logged. Calls drawn by TOH do not appear in logged oto: TOH logged rate = g2s. So the earlier "inert" verdict stands on the logged output; the scorer-universe explanation does not rescue it (the counters pass the cells, the engine state does not show them). Unresolved: where drawn off_to_used is lost between toh pol and the logged oto.
- run1 v run2: play parquets byte-identical (md5 play_0_2), so final-drive numbers are identical; the record number is correct, no invalidation.
- No run_th_3seed.sh written (bar not met on logged output).

## TOH overwrite trace (2026-10-18, read; probe in flight)
- Read: no wrapper after TOH rewrites off_to_used. RG (rg.py:53-95) and LDR (ldr.py:185-205) only wrap Geo.fix/pick and return base drawn; ctx/ctxscr/rgt return base; RTD sets off_to_used=0 only on return-TD plays (rtd.py:191); cky2 (cky2.py:486-526) copies dict and changes clock_elapsed only; log_sync (crH.py:285-298) touches log cols only. Install order crH.py:233-241 correct. No overwriting line found by reading.
- Probe: tests/scratch/e173/probe/probe.py (smoke env, 60 games, monitors every pol layer, prints off_to_used per layer for TOH draws); log tests/scratch/e173/probe/run.log, pid in tests/scratch/e173/probe/out/r.pkl.pid. Read run.log "lost by TOP" and per-layer lines.
## Next
- Read run.log; if off_to_used survives to TOP, loss is downstream of policy (engine/DECIDE/oto logging at mod25d_variance.py:755-767 tolog), not wrappers.

## TOH probe result (measured 2026-10-19 00:00 ET; tests/scratch/e173/probe/run.log, probe.py; 60 games, workers 1; orchestrator took over the probe PID 36348, it exited 0)
init 339 s; pol chain 29 layers, TOH at depth 3 (innermost first); COUNT calls 8,864, eligible 375, called 52, no_donor 0; TOH outputs with off_to_used>0: 212 of 8,864; "lost by TOP": 27 (13% of those). Per-layer survival dicts all (1.0, 0.0) as printed. Interpretation open: 27 lost of 212 does not match the ~60% shortfall (470 expected v ~183 logged in the smoke); 212 includes donor rows on no-call plays (donor off_to_used>0) — next: split survival by called v not-called and compare to logged oto drops for the same plays.

## Probe audit (2026-10-19, read; no run)
- Probe bug (probe.py:113-115 v :71): td comes from enumerate(names), names outermost-first, so TOH=3; depth_of numbers innermost-first (chain[0]=top -> TOP=28), so TOH is depth 25 and depth 3 is sim09_u4g install_tw. "212 of 8,864 off_to>0" and "27 lost by TOP" measure the base draw after layer 3, not TOH outputs. Printed (off_to,def_to) dicts show the real structure: 0.0 starting at 25 (TOH replaces base timeout with y=0 donor, intended) or at 17 (fgd zeroes it).
- Logged oto = pre-play count: install_chain2 decide sets st["to"] from DECIDE args (mod25d_variance.py:745-753), pol appends it (:764); engine decrements after policy (sim04_engine.py:1036-1042). score_th2.py:30-31 reads a call as the oto drop on the NEXT same-offense play; it misses last play of a drive (flip), Q2/Q4 end plays, game end. Calls need off_to>=1 so zero-timeout case is not a loss.
- Accounting (probe, one 60-game run): calls 8,864 -> TOH eligible 375 -> called 52 (.139) -> survive to engine: UNMEASURED (probe looked at wrong layer) -> logged: UNMEASURED.
## Next
- Rerun probe with td = depth of the toh qualname (25) and split called v not-called; count called with off_to_used>0 at TOP and next-play oto drop; fix only the scorer if survival is 52/52, else the wrapper that zeroes it.

## Probe rerun in flight (2026-10-19)
- probe.py fixed (td = len(chain)-1-toh_d[0] = 25; tally wrapper flags eligible/called per top call; CLASS lines per class at layers 24/25/28). Running: PID 17856, log tests/scratch/e173/probe/run2.log (ends with "exit N"). Read CLASS lines there. TOH write path (read): mod25e_toh.py:164-199 touches only eligible plays (:169-170,:176 early returns give base drawn); on eligible plays writes clock_elapsed, off_to_used, def_to_used from one donor (:196-198), including t=0 donors (off_to_used=0), overwriting a base call.

## Cross-reference (2026-10-18)
f2pr extra_fields rebuilds oto/dto from pool otu/dtu with a per-half reset and ignores post-DECIDE rewrites of off_to_used/def_to_used, so TOH's added calls cannot appear in logged oto (plausible cause of the inert verdict). Engine-truth observer scripts/mod25e_tolog.py (TOLOG=1) built; plan and crH patch text in docs/lanes/mod25e-e171-keynumbers.md "TOLOG observer built".

## Probe result, correct layer (2026-10-19, measured; run2.log, td=25, 60 games, exit 0, no process left)
off_to_used>0 at layer 24 (base) / 25 (TOH) / 28 (top): eligible-called n52 17/52/52; eligible-not-called n323 26/0/0; not-eligible n8489 169/169/169. Base eligible rate 43/375=.115; TOH 52/375=.139. Removed 26 base calls (t=0 donor), added 52 (17 of them already base calls): net +26 calls (+9 pp of eligible). Not-eligible untouched. Survival layer 25 to 28 is 52/52: the write path and all wrappers are clean.
## Next (fix spec)
- Hypothesis confirmed in part: TOH removes base calls on eligible no-call plays, but net is positive, and drawn calls survive to the engine. The ~60% logged shortfall is therefore downstream: scorer (score_th2.py next-play oto-drop misses drive-end, Q2/Q4 end, game-end plays) or the logged oto, not wrappers. Next: count engine-side home_to/away_to decrements per game directly and compare with 52-per-60-games.
- Fix (not built): on t=0 keep base off_to_used/def_to_used (take only clock_elapsed from the donor); on t=1 set off_to_used from the donor, keep def_to_used from base. No constants.

## Orchestrator decision on the probe fix spec (2026-10-19 00:40 ET)
Probe (td=25, 60 games): TOH on eligible plays calls 52/375 (.139), erases 26 base calls, adds 35 new, all survive to the top layer; non-eligible plays untouched. The spec "on t=0 keep base off_to_used" is REJECTED: TOH's fitted P(call | state) is the whole timeout decision on eligible plays (per-cell called/eligible matches real: le30 pass .508 v .53), so keeping base calls on t=0 would double-count. TOH behaves as designed (scripts/mod25e_toh.py:164-199). The remaining defect is measurement: the play log rebuilds oto/dto from the pool row (mod25e_f2pr.py:84-125), so TOH calls never appear. TOLOG (scripts/mod25e_tolog.py, installed default-off, smoke tests/scratch/e171/smoke_tolog.sh first in queue) logs the engine's real timeouts; rescore TOH from to_*.parquet, not from the play log.

## TOH rescore on engine timeouts (2026-10-19, measured; tests/scratch/e173/score_toh_engine.py; smoke ...fag2sthldrgtg_s31, 1 world, scored seasons 2-3, 544 games, 90% game bootstrap, one world is wide)
- Logging error: f2pr oto equals engine oto_pre on only 52.6% of plays (dto 52.7%); engine offence timeouts 4,063 (3.73/game offence, 7.08 both) v f2pr next-play drops 3,431; only 4.2% of engine calls coincide with an f2pr drop, 3,252 f2pr drops have no engine call. The play-log oto/dto is close to unrelated to the engine's.
- Offence call rate (oto_applied>0; tie/trail, no score/flip, oto_pre>=1, last 300 s of Q2/Q4), strict run/comp n 4,604: overall .094 [.088,.100] v real .160; cells 0-30 .419 [.379,.460] v .531; 30-60 .294 [.262,.328] v .402; 60-120 .087 [.076,.100] v .121; 120-300 .016 [.013,.021] v .019. Broad code<=1 .095; all plays .086 (0-30 .298). Sim is below real in every cell (probability positive of sim-minus-real about 0 in three cells; 120-300 within noise). Inferred: TOH fires on eligible plays at its fitted rate but the eligible set covers only part of the strict universe.
- Final drive (Q4 last drive of regulation, tie/trail 1-3): engine timeouts per drive .584 [.429,.728] v real .580 [.509,.651] (f2pr reconstruction says .143). Reach .312 [.221,.390] v real .423; score .234 [.156,.312] v .340; n=77 drives. No matched FD5-base smoke exists (every other s31 dir is FD4 family: reach .329/.282/.250 for ldrg/g2sldrg/g2sthldrg, timeouts .14-.18 by the broken reconstruction).
- Verdict: timeouts per final drive now match real; reach and score still below real and not above the FD4 base. Bar (reach toward .423) not met on this smoke; no run_th_3seed.sh written. Recorded rate_pp unresolved_below_power.
## Next
- A matched FD5-base smoke with TOLOG=1 only (seed 31) would give the same-code reach/score baseline; then the TOH 3-seed (cand label ...gx2fag2sthldrgtg, base ...gx2faldrg; verified against mod25e_crH.py:476-519) if reach moves.
