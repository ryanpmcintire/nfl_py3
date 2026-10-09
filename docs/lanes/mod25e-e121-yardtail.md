# mod25e E121 yardage tail and own-territory TD, real v pre-GEO v GEO

## Goal
After GEO=2, does the own-territory TD excess (E118) persist, and through which link (long-gain tail, drive survival, turnovers/punts)? Read-only. Scratch tests/scratch/e121/ (build, an, rep, net, dn, yb, clk, clk2, pool; rep_all.txt, net.txt). Sim dirs: smoke s31 pre-GEO ...kekudwtpqlugky, GEO ...kekudwtpq2ugkyg2 (4 seasons, 1088 games each); real 2009-17 REG (2304 games).

## State (2026-10-09, measured; game bootstrap 90%, 200 reps; ~260 stats x 2 contrasts + ~100 table cells, 6 families: gain, drive, series, turnover, dispersion, pace)
- Measurement trap: sim log `yards` on drive-ending plays (TD, turnover, downs) is next-drive start minus yl (garbage); engine yards = yardline - next live-play yardline (sim04_engine.py:644, includes penalty enforcement). Raw real yards_gained v sim log looked like a pass tail +40% in own territory; on the engine definition (real net = yl - next yl, continuing plays) it vanishes. Gain stats use net (real) / log y (sim) on continuing plays, TD gain = yl, turnover plays excluded.
- Tail: pass P(gain>=20) all zones real .0938 pre .0978 GEO .0891 (GEO-real -.0047 [-.0070,-.0026]); run .0316/.0310/.0270; P40 pass .0172/.0182/.0176. Explosive plays per drive pre +.003 [-.004,+.011], GEO -.029 [-.037,-.022]. No excess tail.
- Own-territory TD: P(TD|start own<25) real .1567 pre .1692 (+.0126 [+.0055,+.0187]) GEO .1559 (-.0008 [-.0071,+.0050], P+ .45); own 25-50 .2111 / +.0104 / GEO -.0052 [-.0146,+.0048]. Link was P(TD|reach opp 50) pre +.016 [+.006,+.025] -> GEO -.006 [-.014,+.004]; reach rates equal (P reach50 -.003). Series success A/B/C within .003 of real.
- Remaining: TD/team-game real 2.29 pre 2.46 GEO 2.20 (GEO-real -.088 [-.146,-.026]) = fewer drives (22.65 v 23.14, -.49 [-.69,-.32]) and plays (136.1 v 140.1, -4.0 [-4.6,-3.5]) at TD/drive -.0035 [-.008,+.002]. GEO v pre: plays -3.75 [-4.35,-3.24], drives -.77. Per-play elapsed pre->GEO: run 33.6->34.6 s (real 35.3), pass non-inc 28.7->29.3 (30.7), inc 7.1->7.6 (6.6), kickoff row 12.4->14.2. Punt-ended drives +.014 [+.007,+.020]; turnover+downs endings -.012 [-.016,-.008] (sim logs fewer 4th-down fails).
- Dispersion (within team-season): points/team-game var real 59.7 pre 67.7 (+8.0 [+3.4,+11.8]) GEO 60.3 (+.6 [-3.1,+4.2]); TD var/mean .713 / .725 / .760 (GEO +.046 [-.001,+.086], P+ .94); explosive var/mean .894 / .886 / .912.

## Tried
Gain quantiles by zone and play type, drive and series links, turnover/sack by zone, pool size at exact yl (median 740 rows per down x type, k=200: 5% of plays <=200).

## Next
Spec (not written): GEO re-pick (mod25e_geo.py:249 REPICK_KEYS, :319-322) overwrites clock_elapsed, off_to_used, def_to_used from a pool row chosen on (dist, score, time) only, pool keyed down x play type with no phase, k=200 of ~740. Keep the engine's phase-conditioned clock_elapsed/timeouts from the original pick (sim04_engine.py:505-526) and take only points, yards, next_*, flip from row j; or add phase to the pool key with k chosen LOSO (grid k 25..400 x phase on/off = 10 looks x 9 folds, score held-out log-lik of {TD, next-yl bin, elapsed bin}). Check: plays/game, run/pass el, kickoff-row el, no targets.

## Open
Real drive count uses my kickoff/possession split (23.1 v engine 22.7, E118). 4th-down-fail share needs a flip column in the sim frame. unresolved_below_power for TD var/mean (P+ .94); no closure. Look count above is per family, not corrected.
