# E159 X2 strength excess decomposition

## Goal
Name what carries X2 strength_re_var 61.8 v real 58.0 (B 56.1, 3 seeds). Offline only; no edits to existing scripts/; scratch in tests/scratch/e159/ (lk.py, lk2.py, lk3.py, lk4.py, logs, lk.pkl).

## State (measured unless marked)
- E159-E159c summary (about 100 looks, 4 labels, unresolved_below_power, no refuted mechanism): X2-B drive-point gain is offense only (+0.52 [+.45,+.60]); in-sample team-season def slope was biased up about 40 by same-game ratings; per-channel sim v real-held-out slopes show no channel carrying a missing half; source-row def transfer slope 0.464 v off 0.768 traces to one scalar kernel h on both axes (sim04_engine.py:524, mod25d_variance.py:934, sim04_engine.py:35; same pattern mod25e_geo.py:456-460, mod25e_gz.py:103-105, mod25e_draw.py:32-35, mod25e_rulestr.py:108-110). Hook scripts/mod25e_axish.py default off (def diff times std(eo)/std(ef) about 1.85); deferred crH install and pass bar: in mod25e_crH.py init wrap inspect.getsource with mod25e_axish.patched_source(dv, inspect.getsource) under a new cfg letter, then mod25e_axish.install(dv, eo, ef) with eo, ef = dv.blup_rows(pbp, trans, dv.TRAIN); bar: def transfer 0.70-0.85 with off 0.768 +/-0.02, held-out def slope not above -62.7, strength_re_var 58.0 +/-4, drive off slope >= 5.18, GZ pass off slope in [.35,.58], E154 guards unchanged, the four other kernel sites patched first.

## E159d-f (recorded 2026-10-18, 6 cells via weak-signals record batch, unresolved_below_power)
- Like-for-like held-out (lk.py, lk5.py, lk6.py; 144 / 48 seasons, 90% season-block bootstrap): strength_re_var real 55.8, B 56.2, X2 65.0 (+9.3 [3.0,15.6] P .988). Rating-linked, not residual. Ablation seed 11: GEO=2 -16.4, RTD -1.8, GZT3 cols 1+3 +13.2 [9.3,17.0] (sack +10.6, conv +6.7); D1 54.3 v B 59.2.
- Tilt provenance (read): tilt_coef.json (mod25d_variance.py:491-494 via cfg tilt_file; crH cfg = dv.DV["crzhk"] has tilt=1, tilt_file=artifacts/mod25d/tilt_coef.json, measured by H.flags("")). In-sample 2009-17, not LOSO; LO/LD league means over 2009-19.

## E159g: LOSO tilt refit (scripts/mod25e_tiltl.py, TILTL=1, suffix tl; not installed)
- Fit: per held-out season s in 2009-17, AO/AD for conv (col 1) and sack (col 3) maximise Bernoulli LL of real outcome under kernel-weighted candidates (draw.stage_weights team*ipw*dk*epa times tilt; K=sim.K_STATE nearest state rows excluding season s, drawn rate among rows with D_k), 3000 queries/season/col; LO/LD from training seasons only. Smoke (60/season) ran: AO/AD noisy, held-out LL fold ~ current ~ zero.
- Full run: `mod25e_tiltl.py fit --per-season 3000` log tests/scratch/e159/tiltl_full.log, writes artifacts/mod25e3/tiltl/fit.json (folds, transfer slopes); then `mod25e_tiltl.py report`.
- Install patch (deferred, in crH after init): `import mod25e_tiltl as tl; if tl.enabled(): tl.install(season)`; synthetic seasons use the "all" fit. Pass bar (declared): smoke/3-seed held-out strength_re_var toward 55.8 (lk5/lk6), offense SD toward .0597, guards SD/noise/pts/Q4 slope/mass3 not worse than current.

## Next
Read fit.json report; write E159g numbers here; record E159g via weak-signals record; then smoke with TILTL=1 once sim runs are idle.

## Open
Does LOSO conditional tilt keep the +13 strength spread; ratings include same-season BLUP (caveat).

## E159g result (measured, fit.json; 9 folds, 54000 queries, K=200)
- AO/AD per fold mean (sd): conv AO .63 (.11), AD 2.06 (.13) [current .51/.27]; sack AO -.35 (.20, all negative), AD -.24 (.37, 2 of 9 positive) [current -1.99/-.79]. Stable signs except sack AD.
- LL/play: in-sample fold -.54316 v zero -.54354; held-out fold -.54345 v current -.54373 = zero. Gap in-sample v held-out .0003; fold minus current +1.7 nats/fold (6/9 folds positive); recorded e159g_loso_tilt_vs_current_heldout_ll.
- Transfer (held-out, rate SD = |slope| x rating SD; fold/current/zero/real): conv off .0360/.0365/.0279/.0364; conv def .0281/.0164/.0139/.0286; sack off .0090/.0145/.0076/.0080; sack def .0035/.0041/.0025/.0031. Current sack gain over-transfers offense 1.8x real; LOSO matches real; LOSO conv def AD 2.06 v .27 fixes the missing def transfer (fits E159c def slope gap).
- Inferred: LOSO refit shrinks sack-carried offense SD (+.011 at current) so strength_re_var would fall below D1 54.3 toward g2rt 41-48 plus conv gain; gx2/other steps not tested. Smoke with TILTL=1 pending (sim idle). Processes reaped.

## E165 smoke TILTL (measured 2026-10-18; seed 31, 1 world, 4 seasons, 544 games; scripts tests/scratch/e165/)
- Hook fired (inferred, not logged): label suffix tl; strength_re_var 94.4 -> 64.7 (real 58.0; lk.py held-out 94.7 -> 64.8 v real 55.8), offense SD .0728 -> .0641 (real .0597), def SD .0396 -> .0498.
- Guards XFA -> TL (real): margin SD 16.13 -> 15.58 (14.63) better; noise 164.5 -> 178.1 (165) worse; Q4 slope .0015 -> .0228 (-.060) worse; mass3 .099 -> .151 (.141) better; xq cov 23.8 -> 17.6 (-6.4) better; pts/game 43.97 -> 44.49.
- Leader 24+ Q3+Q4 (an3): pts/drive real 1.555, XFA 1.587, TL 1.948 (+.39 [.09,.69] P.98); success sim-real .040 -> .054; run ypp +.47 -> +.51; TD/play -.0025 -> +.0072. Leader excess grew; tilt over-transfer not the carrier.
- Verdict FAIL v declared bar (noise, Q4 slope not held; one world, no SE). Single-world; unresolved_below_power, not recorded. run_tl_3seed.sh NOT written.
