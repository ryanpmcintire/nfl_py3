# MOD-25e E126: why L loses team-strength expression (strength_re_var 56.1 -> 42.3, real 58.0)

## Goal
Name the cause of the B -> L strength loss (B=...kekudwtpq2ugky, L=...kekuw2tpq2g2rty2u2; GEO=2 added) and specify the fix. Read-only diagnosis; no sim run, no scripts/ edit.

## State
- measured (tests/scratch/e126/reg.py; sim ratings rebuilt from the seed via gen.gen_world_latents/make_schedule/season_ratings; net = off_r + def_r; seeds 11-13, worlds x seasons 2-7; real = BLUP rows, in-sample): drive points per unit net: real 6.18, B 4.40, L 3.89 (-12%). Pass yards per unit (own/mid zone): real 7.7/8.7, B 7.2/6.4, L 5.3/5.3. Run yards: real 2.9/2.4, B 2.6/2.5, L 2.3/1.8. TD per unit (pass/run): real .142/.090, B .110/.072, L .090/.054. Slope loss is in every state, largest own/mid.
- measured (ess.py): the team KERNEL is not the cause. ESS 29.9 (GEO band 0) v 30.9 (engine); rating transfer T 0.708 v 0.704; band widening (2,5,13) changes nothing; pool N median 611 (6% of states N<200).
- measured (ess2.py, 3000 real states, strong v weak matchup, gap 0.21): engine weights = kernel x home x IPW x EPA-tilt x TILT x dkern (mod25d_variance.py:955-957,963-967). Geo weight_fn (mod25e_geo.py:~371-376; mod25e_gz.py:104-109 same) = kernel x home x IPW x PASS_W only: no TILT, no dkern.
  dY/dnet, dTD/dnet: pass engine 6.89/.081, Geo current 5.83/.049, Geo+TILT 7.19/.084, Geo+TILT+dk 6.01/.095. Run engine 1.40/.053, current 1.13/.036, +TILT 1.21/.059, +TILT+dk 0.93/.064. ESS engine 13 v Geo 30 (flatter = less team signal).

## Tried
Kernel/pool-size hypothesis (refuted). Weight-term audit (found). Other new hooks (rtd, urg, cky2, cdw) do not read off_row/TILT_T; not tested for outcome overrides.

## Next
1. In Geo.fix weight_fn (and gz.py), add the engine's TILT factor exp(TILT_T[near] @ (TILT_AO*(off_rt-TILT_LO)+TILT_AD*(def_rt-TILT_LD))); dkern optional (f3 already includes distance; adding it overshot TD slope in the stub). Call the same function the engine uses so no copy drifts.
2. Re-run E5 for L with fix, seeds 11-13; check strength_re_var, r2, TD/drive vs real; GEO=2 still has TD-low bias to check separately.
3. If strength still low, test cdw=2/urg2/cky2/rtd individually (outcome overrides).

## Open
Stub uses real BLUP ratings and 3000 queries without SEs; sim run is the confirmation. Real slopes are in-sample BLUP (inflated).

## E126f (TILT in Geo/GZ weight_fns, flag GZT=1)
- read: added tilt_active/tilt_factor to scripts/mod25e_geo.py (before install_geo), used in Geo weight_fn; scripts/mod25e_gz.py pol_inner multiplies tilt_factor only when GZT=1 and TILT_T in ns (unset = unchanged); scripts/mod25e_crH.py label +"gt" after f2 when GZT=1. tilt_factor is the engine formula (mod25d_variance.py:956) on ns TILT_T/AO/AD/LO/LD. dkern not added: engine applies it to its own K nearest rows; Geo/GZ pools are already matched on distance (f3), and the stub shows dk overshoots TD (pass .095 v engine .081).
- measured (tests/scratch/e126f/ess3.py, 3000 states, dY/dnet, dTD/dnet; code1=pass, code0=run): pass engine 6.89/.081, GZ pool (geo0) 5.83/.049 -> +TILT 7.19/.084, Geo(band5) 5.79/.033 -> 7.90/.054. Run engine 1.40/.053, geo0 1.13/.036 -> 1.21/.059, geo5 2.18/.023 -> 2.76/.037. GZ+TILT matches engine on pass; Geo band5 pass TD still low (.054 v .081) and run yards high (pool differs; separate).
- Next: run E5 L with GZT=1 (GZ=1 and GEO=2 variants), seeds 11-13, compare strength_re_var/r2/TD-per-drive.
