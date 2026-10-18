# E172c return-TD excess

## Goal
Find why the sim's defensive return-TD share of run/pass turnover rows is 4.36-4.52% v real 3.40%.

## State (measured, tests/scratch/e172c, e172)
- Excess is a uniform within-state level lift after the draw, not field mix (real rate at sim mix 3.44 [3.23,3.65]), not defense gradient (per-SD slope matches, -0.86 v -0.83 %/SD), not kernel bandwidth, not Geo selection (E172d/e/h).
- Ladder by version (s11): B 3.71 (3 seeds 3.77), g2 4.36, X2 4.44, base 4.52. GEO=2 is the biggest step (+0.66); RTD=1 is not a carrier (mod25e_rtd.py:139-196 opening kickoffs only). Pool 3.52, K-set 3.9-4.05.
- E172j paired logs B v g2 s11: NOT pairable. Only 41 of 8,142 games share a state and differ in idx at the same position, and 2 same-state same-idx rows exist; RNG streams diverge from play 1. The planned play-by-play match is infeasible.
- E172j unpaired by yardline band (B v g2, s11; turnover per 1000 rows / return-TD per 1000 rows): yl 20-40 77.6/1.76 v 67.2/1.85; 60-80 32.2/3.02 v 29.3/3.37; 80-90 29.8/3.22 v 26.8/3.85; 90+ 34.5/4.34 v 40.5/6.87. g2 lowers turnovers per row 5-13% at yl 0-90 and raises them 17% at yl 90+, while return-TD per row rises 13-58% at yl 60+. Share 3.705 +-0.11 (B, 109,863 turnovers) v 4.361 +-0.13 (g2, 101,889).
- Mechanism under test: GEO=2 repicks every run/pass play (mod25e_geo.py:438 always, fix force=True) from a fp-window candidate set where every free row (flip or points_def, not TD) passes unconditionally (mod25e_geo.py:344) while non-free rows need yards+shift < q; free rows include all returned turnovers.

## Tried
Gap-conditioned share, bin decomposition, Geo candidate replay (K-set 3.2-3.6 per turnover), symmetric window, weight-stage replay, state-mix standardization, defense gradient, paired B v g2 play logs (infeasible).

## Next
1. Rerun tests/scratch/e172/p4.py in background (first run killed at cap, no output; takes over 2 min): candidate-set v window v K-set return-TD and flip rate per row by q band decides whether line 344 free-row admission creates the per-row lift.
2. If confirmed: default-off module that restricts free-row admission to the fp-window rate (no unconditional pass), fitted on real window rates, same-state paired by hand against g2 only after owner lifts sim freeze.
3. SAFEG first; crH patch deferred (another agent owns scripts/mod25e_crH.py). TILTL does not fix this.

- p4 cheaper copy tests/scratch/e172/p4b.py (P4N env = query count, seed 11, default 1000). Run with F:\Repos\nfl_py3\.venv\Scripts\python.exe -u (uv is not on bash PATH). stdout is block-buffered and p4b.txt stayed empty after ~90 CPU s (killed, 0 left); the 1000-query run still did not finish. Next: rerun with python -u and P4N=300 as one background call (no trailing &, which detaches and orphans a second run), read per-q-band win/cand/kset rtd and flip.
- Read: GEOC=1 exists (mod25e_geo.py:318,336) and gates free rows on fp>=q; the unconditional free pass at :344 is the default path. Name FREEGEO unused (git grep). Install point candidate mod25e_crH.py:198-201 beside ge.install_geo(). Line 344 NOT confirmed or refuted.

## Open
Why returned turnovers per row fall 5-13% at yl 0-90 under g2 while return TDs rise at yl 60+.

## E172d p4b result (measured 2026-10-18 18:25 ET, orchestrator; P4N=300, seed 11; output tests/scratch/e172/p4b.txt via task log)
- Candidate set == fp window (mean 725.9 rows both; identical rtd/flip 3.73): free-row admission at mod25e_geo.py:344 removes nothing here -> NOT the carrier.
- rtd per flip: window 3.73, unweighted K-set (nearest 200) 4.12, weighted K-set (team kernel + ipw + tilt) 5.23; real share ~3.40. Carrier = served weighting (geo ~:458-466) plus nearest-200 selection.
- By q band (win/kset rtd%): 40-60 .158/.221, 80-100 .454/.528; 60-80 kset below window.
- Next: decompose 5.23 into team kernel, ipw, tilt (each off in turn, same 300 queries); name which weight upweights returned turnovers and why (e.g. ipw on rare rows).

## E172e decomposition (measured 2026-10-18, tests/scratch/e172/p5.py p7.py p8.py p9.py; 300 queries seed 11, single thread)
- Weights, rtd per flip % on nearest-200 K-set: none 4.12; team kernel alone 5.23; ipw alone 4.01; tilt alone 4.14; team+ipw 5.21; team+tilt 5.26; all 5.23. Team kernel is the whole +1.11; ipw and tilt are null.
- Team kernel parts (mod25e_geo.py:458-460, TEAM_KERNEL_LAMBDA=1e6 so home/away match is a hard split): home 3.85, off 4.22, def 4.23, off+def 4.48, home+off 4.44, all 5.23. Control with shuffled off_row/def_row, same h and lambda: 3.63. Rating information, not ESS 33 or self-row (self excluded: 5.77), carries the lift. Returned turnovers cluster in rating-and-side neighbourhoods; the kernel moves mass to them.
- 3.73 -> 4.12 step (nearest-200 by f3 columns [0,2,3] = distance, score, time): distance 3.71, score 3.93, time 4.05, score+time 4.15, all 4.12; random 200 in window 3.53. Time-left and score-diff terms pull returned turnovers in (mod25e_geo.py:422 dd over f3).
- Window flip 6.44%, real all-row flip 6.75% rtd/flip 3.52; served weighting leaves flip share 6.39% and non-flip yards unchanged (mean 5.31 v 5.27 window, sd 8.64 v 8.62).
- Recorded: weak-signals e172d_teamkernel_returntd_lift unresolved_below_power. Not a closing ground; no interval.

## Next (replaces Next 1-3 above)
Default-off module spec CLSDRAW (git grep clean), file scripts/mod25e_clsdraw.py with install_clsdraw() called beside ge.install_geo() at mod25e_crH.py:198-201 and wrapping Geo.fix at mod25e_geo.py:419-425 (weight_fn use). Mechanism: the team kernel, ipw and tilt condition a row's yards on team strength; outcome class (returned TD, other turnover, no turnover) is drawn from the fp-window class shares of the candidate set P["ids"][pos], then the row is picked kernel-weighted within that class (fall back to window class members when K-set holds none). No constants added.
Per-query class pinning of weights (rescale K-set weights to window class mass) is WRONG: p8 gave 2.61 because many K-sets hold no returned row. Use the two-stage draw.
Offline replay (p9, same 300 queries): rtd/flip 3.731 (served 5.234, K-set 4.123), flip share 6.437% v real 6.746%, non-flip yards mean 5.252 sd 8.677 P(>=10) 0.2043 v real 5.176 / 8.534 / 0.2034 and window 5.227 / 8.705 / 0.2048. Guard: assert per-run flip share and non-flip yards mean/sd stay within the real interval before enabling.
Open: 3.73 v real 3.40-3.52 remains (fp-window q mix); time and score terms in the f3 near-200 set add +0.39 and are a second carrier, SAFEG-style review before serving. Sim freeze: paired g2 check only after owner lifts it.

## CLSDRAW unit (2026-10-18, agent stopped before building; nothing measured)
- Concern to test first: unweighted window-share class draw erases real rating dependence of returned-turnover rate. Needs (1) real P(rtd|turnover), P(turnover|play) by off/def rating quantile, game-bootstrap 90%, v kernel-implied rates; (2) scripts/mod25e_clsdraw.py (CLSDRAW=1, git grep clean) with LOSO multinomial logit on class (rtd/other turnover/none) and stage-2 kernel-weighted pick, wrap Geo.fix mod25e_geo.py:419-425; (3) crH patch note + tests/scratch/e172/smoke_cd.sh from tests/scratch/e163/smoke_g2s.sh (add CLSDRAW=1).
- Blocker: dr.build() (mod25e_draw.py:21, dv._d_init) takes >2 min on a loaded machine (about 20 other python procs). tests/scratch/e172/cd_keys.py lists arrays keys (need game id/season for bootstrap and LOSO; real_load in mod25d_variance.py:23 gives g, season via meta, sorted by game_id,play_id, check arrays align). Its output never arrived; killed.
- Next: run cd_keys.py once as one background call when CPU is free (py -u, thread caps), then write cd_real.py (part 1) before building the module.

## CLSDRAW part 1 result (measured 2026-10-18, tests/scratch/e172/cd_keys.py cd_real.py cd_real.txt; draw_cache.npz holds 20 arrays, no game id)
- Arrays align row-for-row with real_load(2009-17) (346,552 rows); game id and season come from the real_load join in cd_real.py.
- Real rtd per turnover by rating quintile (low to high): offence 5.36 4.02 3.31 2.74 2.32, top-bottom -3.04pp [-3.79,-2.27], prob_pos 0.000; defence 4.97 3.78 3.29 3.10 2.49, -2.48pp [-3.30,-1.79], prob_pos 0.000. Turnover per play: offence +0.50pp [0.31,0.71] prob_pos 1.000; defence +0.10 [-0.10,0.30] prob_pos 0.80. Overall 6.75% and 3.52%; seasons 2.55-4.15.
- Real data HAS strong rating dependence (about 3pp across quintiles, comparable to 2.7x, versus the kernel's overall level lift +1.11 pp). Concern confirmed: an unweighted window-share class draw (p9 two-stage, 3.73) erases it; the kernel lift is a level shift, but its direction toward similar-rated neighbours is real. Kernel by-quintile rates not computed (p8/p9 print overall only). Recorded e172_cd_real_rtd_rating_gradient unresolved_below_power.
## Next (parts 2-3, replaces CLSDRAW unit Next)
2. Add by-quintile kernel-implied rtd per turnover (300 queries, p9 harness, group by query off_row/def_row quintile) to see whether served 5.23 vs window 3.73 matches real gradient shape.
3. scripts/mod25e_clsdraw.py (CLSDRAW=1): LOSO multinomial logit class (rtd/other turnover/none) on off_row, def_row, plus window state; stage-2 kernel-weighted pick; wrap Geo.fix mod25e_geo.py:419-425; use draw_cache.npz for fits. Smoke from e163/smoke_g2s.sh after sim freeze lifts.

## CLSDRAW class model, declared before fitting (2026-10-18)
- Stage 1: multinomial logit, 3 classes (returned TD, other turnover, none) on real run/pass rows 2009-17. Features: 10 field-position decade bands of the rounded yardline (FP_BAND_YD=10, structural), down one-hots, pass indicator, scaled distance, score and time features (columns 0,2,3 of the draw features), off_row, def_row. No penalty; no tuned constant. LOSO by season; held-out log likelihood v (a) window class shares and (b) kernel-weighted K-set class shares, both excluding the query season and shrunk to the pooled rate with a pseudo-count chosen to favour the baseline.
- Stage 2: pick the row kernel-weighted within the drawn class from the K-set, falling back to window class members, then to renormalised classes present in the window.
- Pass bar: gradient by off/def quintile reproduced inside real game-bootstrap 90% bands, overall rtd/turnover in the real interval, flip share and non-flip yards mean/sd/P(>=10) in real intervals. Fitted artifact: artifacts/mod25e3/clsdraw/.
