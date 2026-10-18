# mod25e-e91-startspot

## Goal
Name the mechanism behind the persistence gap's resolved carrier: the opponent
(B) of an early over-performer (A) starts later drives at better field position
in real games (E88 slope -.133 yd/pt) than in the sim (-.056). Parent lane
docs/lanes/mod25e-generator-fidelity.md; E88/E89 detail in
docs/lanes/done/mod25e-generator-fidelity-log-2026-10-04.md.

## State
2026-10-05 E91 run (scripts/mod25e_e91.py -> artifacts/mod25e3/e91/e91.txt, 40 looks, 300 game boots, base jo2as2 s11-13). Verdict unresolved_below_power on the slope carrier; named attenuation defect measured.

## Tried
E91 part 1 (typed B H2 drives, s = e + t): gap s-r +.047 [+.011,+.076]; e piece +.037 [-.023,+.092] P .89; t piece +.010 [-.042,+.068] P .60. By type t carries to +.057 (P .95), fg +.069 (P .96), fgmiss -.123 (P .01), td -.001; e carries td +.052, punt +.025. No single piece, mixed sign.
Part 2a: slope of B start on A end-spot, real vs sim: punt .69 vs .24, to .85 vs .65, fgmiss .57 vs .16 (P(sim>real) 0.00). Part 2b: drawn pool row yl (pool.parquet[idx], 97% rows consistent) on sim yl on flip plays: punt slope .37, to .77; mean gap -.7..-.9 yd, mean |gap| 12 yd.
Mediated b_se*b_ex overshoots total (punt +.109 gap vs +.036 total), so the attenuation is real but its share of the x gap is not established.

## Tried (E93)
scripts/mod25e_e93.py -> artifacts/mod25e3/e93/e93.txt (shift_keep; e93_shift.txt = cond censoring). Offline replay of 528008 sim flip plays, 13 looks, 300 boots: B start = sim start + (drawn row yl - sim yl) (relative transfer, composes with STF z), touchback when >=100, missed FG by rule (offset 8 yd measured on drawn rows) or same shift form. b_se restored: punt .665-.712 (real .692), to .806 (.848), fgmiss .595-.602 (.569; rule form overshoots .72). Level not clean: punt start -1.2..-2.2 yd vs real, touchbacks punt .067 vs .097, fgmiss tb .19-.23 vs .08 (sim miss spots 3.3 yd closer than real, e.g. 76.1 vs 72.8, a separate gap). E88 x slope: real -.1326, base -.0556, replay -.0661 [-.0733,-.0590] to -.0706; replay-base -.011..-.015 (P(replay<base) .99-1.0), covers ~15-20% of the gap; replay-real +.062..+.067 P 1.00. Verdict: start coupling is a real defect but NOT the x carrier; no wrapper written (level checks broken, x unmoved).

## Next
DONE. 1. Decompose B's start yardline after a possession change into A's
   previous-drive end spot (yardline of the punt/turnover/downs play) plus the
   transfer (net punt, return, turnover return). Which piece carries the slope on
   A's H1 residual, real vs sim (base crHpqokgndecsmfwtjo2as2 s11-13)?
DONE. 2. Engine coupling test: on a flip, sim04_engine.py:1116 sets yardline to the
   drawn neighbour row's absolute next_yardline; the neighbour's own yardline
   differs from the sim's (kernel blur), and yard_shift/ADJ never touch flip
   starts. Measure real vs sim slope of start yl on the end-spot yardline and the
   sim's drawn-row-minus-sim yardline gap on flip plays.

## Next (E92, superseded by E93)
Fix form tested offline in E93, not installed: on flip plays set start = (100 - sim yl) + the drawn row's relative transfer (next_yardline minus (100 - row yl)), blended with the absolute form by a weight fitted LOSO on real b_se; no constant.

## Open
Answered: restoring b_se moves x only ~.011-.015 of .077. Remaining carrier unnamed (b_ex, e on x, is the other leg of the mediation; sim A end spot vs H1 residual). Optional: fix start coupling anyway only if level checks (punt -1 yd, touchback censoring) get a measured rule.

## Tried (E94, 2026-10-05, 54 fit looks + 13 replay looks, 300 game boots)
scripts/mod25e_e94.py -> artifacts/mod25e3/e94/e94.txt, fit.json. Real pool flip rows (scores removed via next sd = -sd): next start slope on yl punt -.732, to_run -.88, to_pass -.78, downs ~-.94..-.98, fgmiss -.86; net punt rises 27 (yl 21-40) to 43 (yl 81-99). Window (min rows, LOSO CRPS, 9 seasons, beats pooled-all 9/9 every class; nested = chosen): punt 640, fgmiss 10, to_run 40, to_pass 320, downs_run 20, downs_pass 20. Punt window is flat (CRPS 5.457-5.469 over m 10..1280) so punt is only weakly window-sensitive; all-yardline pool is far worse (8.39).
Replay s11-13 (redraw within window, STF z carried): punt slope .681 [.678,.684] vs real .692 (diff -.010 [-.025,+.004]); punt start 73.93 vs 74.19 (-.26 [-.54,-.03]); punt tb .102 vs .097 (+.004 [-.001,+.010]); end spot matches. PUNT PASSES. Turnover slope .783 vs .848 (-.065), start -1.57 yd, tb .068 vs .055; missed FG slope .640 vs .569 (+.07 [-.01,+.14]), start +1.81, tb .152 vs .083, miss spot unchanged (76.0 vs 72.8, separate gap). E88 x slope replay -.0637 [-.0696,-.0565] vs base -.0556, real -.1326; replay-base -.008 [-.016,+.002] P(replay<base) .94. Closes: start coupling not the x carrier (E93 and E94 agree). Not closed on weak-signal grounds.
Wrapper scripts/mod25e_ylm.py (install_ylm, YLM=1, reads e94/fit.json, uses engine arrays of TRAIN seasons) written, NOT run: no smoke, no flag-off check (x unmoved, to/fgmiss levels off). crH hooks if wanted: in run_with_flags insert before the STF block `import mod25e_ylm as ym` / `if ym.enabled(): ym.install_ylm()`; in cmd_e5 label chain add `if os.environ.get("YLM") == "1": label = label + "y"` after the STF label.
