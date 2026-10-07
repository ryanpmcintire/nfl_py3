# mod25e-e98-regkick

## Goal
Regulation residual of the key-number gap (E96, docs/lanes/mod25e-e96-mass3.md):
sim FG share of TD+FG .370 v real .395; no point after TD .091 v .061; TD/g +.12,
FG/g -.27. Name the mechanism. Parent: docs/lanes/mod25e-generator-fidelity.md.

## State
2026-10-05 E98 analysis done (scripts/mod25e_e98.py -> artifacts/mod25e3/e98/e98.txt, 75 looks, game bootstrap 300).
PAT [measured]: sim try draw is scripts/mod25_mechanisms.py:322-324 (kick-good rate from season>=2015 rows only, 0.9412)
applied to every year; real gate 2011-17 kick good .9713 (2011-14 .9941, 2015-17 .9394). kick good per TD real .9114 v sim .8773
(s-r -.034 [-.041,-.028]); no point after .0609 v .0909 (+.030 [+.024,+.036]). Per-year real kick rate implies td6 .0578 v real .0609.
OTY only touches OT rules; the try draw has no season input. 2pt good/TD .0277 v .0318 (+.004 [+.001,+.008]).
FG [measured]: FG made/g all drives real 3.262 v sim 3.036 (-.226 [-.306,-.144]); make rate | attempt equal (.844 v .840);
4th-down FG share at yl<=40 equal (.715 v .710; era pin refuted as cause). Gap = attempts: reach<=40 -.264/g, P(att|reach) -.016.
Cells: FG made/g -.173 (deepest yl 1-10), -.130 (11-20), +.013 (21-30), +.053 (31-40). Deep reach/g 1-10 -.427 (P(att|reach) equal).
P(TD|reach) too high at 21-40 (.266 v .200, .171 v .130): long scores from distance. FG make by kick distance flat in sim:
<=19 .933 v .996, 20-29 .896 v .967, 40-49 .810 v .763, 50+ .734 v .633 (all P=1.00 one side).
Mechanism [inferred]: outcome drawn from kNN neighbours with fp scaled 2.5 and state cache key rounded to ROUND_FP=5
(scripts/sim04_engine.py:37,46,242-252; K_NEIGHBORS=40 :31; class pick scripts/mod25_mechanisms.py:390-411): a neighbour's
made/miss and TD outcome transfers across +-5 yd, blurring FG make by distance and letting TDs score from outside.

## Tried
Real v sim drive cells by band/down/distance/state; LOSO forms: kick made by rule regime (ll .1107 v pooled .1245 v served
.1342; pre-2015 seasons .0516 v .0870); FG make logistic in kick distance (LOSO ll .3855 v constant .4430, 9/9 seasons, coef
-.098..-.102/yd stable).

## Implemented (2026-10-05, not yet scored on 3 seeds)
scripts/mod25e_paty.py (PATY=1, label x): kick-good by regime of the OTY season year; fit artifacts/mod25e3/paty/fit.json (TRAIN 2009-17 pool rows):
pre-2015 .9913 (n 6660), 2015+ .9412 (n 3283). 2pt [measured]: share of tries .0449 v .0684 (z~5, keyed: served 2pt-go grid scaled by regime share / pooled share),
2pt good .492 v .452 (n 313/241, z~.9, NOT keyed, pooled served rate kept). Wrapper swaps the `pat` closure cell of the mechanisms pol per call.
scripts/mod25e_fgd.py (FGD=1, label fd): logistic in kick distance (fp+17) fit on TRAIN pool FG rows -> artifacts/mod25e3/fgd/fit.json
(coef -.1010, icpt 5.656, n 8928); redraw make/miss, on change swap to a real pool FG row of that outcome at nearest yardline. Applies in OT too (installed after WFG).
Hooks: mod25e_crH.py PATY after ot.install, FGD last. Env add: PATY=1 FGD=1 (label base+xfd).

## Next
1. Engine edit (separate task): try draw keyed to season year (OTY year) with regime-fitted kick rate; FG make drawn from
   logistic in the state's own kick distance; test TD-from-distance by conditioning gain on state yl (deep reach -.43/g is
   upstream, check yardline-relative draw). Report gate cells and E96 gap after.
2. Record weak signals not needed (mechanisms, not signals).

## Open
Whether ROUND_FP rounding also explains deep-reach shortfall (yl<=10 reach -.427/g): untested.

## In flight (2026-10-06 evening)
The 10-05 PATY smoke crashed in every worker (install_paty read `pat` from dv._G["pol"], which is
an OT wrapper) and looped for 26 h; killed by PID. Fixed: mod25e_paty.pat_cell walks the wrapper
closures to the mechanisms pol that holds `pat` (a stale subagent also added find_cell; one helper
should stay). Smoke re-run: base env + YLM=1 YLM_CLASSES=punt WFG=1 PATY=1 FGD=1, seed 31, 1 world,
4 seasons, workers 1, log artifacts/mod25e3/paty/smoke_on.log, output artifacts/mod25e3/e5_<base>xfd_s31.
Next: compare try kick-good by regime and FG make by kick-distance band, off (smoke/e5_..._s31) v on;
if right, one 3-seed e5 with PATY=1 FGD=1 CDR=1 (--workers 3, one at a time) and score with mod25e_era.py.
Comparator note: the on run carries YLM punt + WFG, so the clean off is smoke/e5_crHpqokgndecsmfwtjo2as2ypw2_s31,
not smoke/e5_..._as2_s31 (that one is the 10-05 flag-off run without YLM/WFG, 820 s). Drop the duplicate find_cell.

## Smoke result 2026-10-06 (measured, seed 31, 1 world, 4 seasons 2009-12; off = smoke/e5_...ypw2_s31, on = e5_...ypw2xfd_s31)
PATY works: TD po7 .8814 -> .9367, po6 (no point) .0876 -> .0381, po8 .0310 -> .0252 (all pre-2015 years).
FGD has almost no effect: FG make by kick distance (yl+17) off/on 0-19 .948/.965, 20-29 .891/.888, 30-39 .856/.860,
40-49 .806/.800, 50+ .738/.702; fit implies ~.96 at 25 yd and ~.53 at 55 yd. The engine passes every play through
_G["pol"] (sim04_engine.py:1012), FGD is the outermost wrapper, STATS counters never printed. Suspects: play log
writes po from the drawn row idx rather than the swapped dict; FGs drawn by another path; the hook's yardline arg
unit. Comparison script: session scratchpad cmp_paty.py (reads play_*_*.parquet columns g, yl, qtr, code, po).

## FGD cause and fix 2026-10-06 (subagent)
Cause [read]: play log row is appended by the innermost pol (scripts/mod25c_noise.py:374-415, log.append at ~403 with drawn po/pdf/flip/clock) and every hook wraps it and edits drawn after base returns; scripts/mod25e_budget.py:26-27 builds play_*.parquet from that log, so any post-base edit (FGD swap) was never logged. Yardline arg is fp (yardline_100 equivalent), matches fit. STATS never printed, so the swap did run.
Fix [read]: scripts/mod25e_fgd.py rewrites dv._G["log"][-1] fields 6-10 (code, po, pdf, flip, clock) after a swap. Other post-base wrappers (WFG, YLM, OKK etc.) likely have the same logging blind spot: check which edit outcomes after base.
Probe [measured]: FGD=1 mod25e_crH.py sim --off f3 (f3 off: sim09_f3 pol uses sys._getframe(1) and KeyErrors under a direct-calling outer wrapper with f3 on and no other hooks), 1 world 1 season seed 31, 1022 FGs. Make by kick distance (yl+17), sim vs fit mean: 0-19 .973/.978, 20-29 .973/.958, 30-39 .912/.895, 40-49 .748/.753, 50+ .538/.573. Overall .833.
Caveat [inferred]: the game engine used the swapped outcome all along; only the log lacked it. So the earlier smoke FG-by-distance comparison understated FGD; rerun smoke on and compare, and any score/FG-per-game result already scored was real.
2026-10-06 late: FGD log patch extended to r[11]=j (row idx of swapped play). 3-seed e5 launched, base + YLM punt + WFG +
PATY + FGD + CDR (label suffix ypw2xfdcd), seeds 11-13 sequential --workers 3, logs artifacts/mod25e3/xfdcd/run_s*.log.
Next: score with scripts/mod25e_era.py against base ypw2 (w2_era/era.txt) and rerun E96 mass-3 cells; stale-log audit
of other post-base wrappers is recorded in mod25e-generator-fidelity.md.
Next after the 3-seed run (do not edit sim code while it runs; seeds load code per process): one outermost
log-sync wrapper installed last in mod25e_crH.run_with_flags that rewrites dv._G["log"][-1] fields 6-10 from the
final returned dict (covers KICK+KGZ, WFG, CLK, KN/KNW, TDC, CDR per the stale-log audit in
mod25e-generator-fidelity.md); idx stays per-hook (FGD sets it). Then recheck play-log results: E97 OT FG share,
E95 end-spot, YLM punt slope, CDR snap timing. Game-final checks (era.txt, e96 mass-3) are unaffected.
2026-10-06 21:10: first 3-seed attempt died (AssertionError mod25d_variance.py:1060, log idx must equal play idx; my
r[11]=j edit, reverted). Log-sync installed (mod25e_crH.install_log_sync, last in run_with_flags, under f3 which stays
outermost); CDR already patched t[10] (audit row wrong). Probe s41 1 season: FG make 20-29 .945, 50+ .622; po6 .039.
3-seed relaunched (same env/label); then ERA_VARIANTS=<label> scripts/mod25e_era.py and the E96 mass-3 cells.
2026-10-06 3-seed result (measured, artifacts/mod25e3/xfdcd_era/era.txt; real pool 2009-17, base ypw2 -> ypw2xfdcd):
mass3 .122 -> .125 (.141); late r2 .128 -> .142 (.146); SD 15.59 -> 15.75 (14.63); noise 184.7 -> 185.9 (165);
strength 58.45 -> 62.55 (57.96); pts 45.99 -> 46.20 (45.21); xq cov +5.1 -> +8.1 (-6.4). 3 seeds, no interval yet.
PATY and FGD each match their own real behaviour (play level), so they stay as mechanisms; aggregates are checks.
Open: strength +4.1 and SD +.16 may be seed noise; needs a seed bootstrap before any KEPT verdict on the combined base.
Next: seed-bootstrap the deltas, E96 mass-3 cells by state, CDR snap gsr<=5 share v real .39 from the synced logs.
