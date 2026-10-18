# mod25e-e172-defscores

## Goal
Safeties and defensive/return TDs per game, sim (label ...ldrg, s11-13, 39168 games) v real 2009-17 (2304 games); name the over-producing path; specify the fix.

## State (measured, tests/scratch/e172, e172f; games bootstrap 90%)
- Safeties/game real .0586 v sim .0962 (+.0376 [+.029,+.046]); snaps yl>=90 equal (+.004 [-.004,+.011]); yl<90 sim .0397 v real .0056. Punt safeties equal overall. Def TD/game real .395 v sim .458 (+.063 [+.037,+.091]); return-TD share +1.6pp unexplained, not a coordinate defect.
- Root defects: Geo.violates (mod25e_geo.py:309-314) exempts points_def>0; safety source rows carry transition yards +31.4 (free kick), real pbp loss mean 3.98; Geo pol routes only run/pass (446-450), so punt safeties are never checked. Far-field (yl<80) sim safeties 957 of 4994: run 599, pass 153, punt 205; none reach goal from sim spot; src-v-sim yl gap median 24.7; real far-field safety-drive last rows 12 of 153.
## E172b SAFEG (scripts/mod25e_safeg.py, SAFEG=1, suffix sg; installed in crH after v1ad)
- Rule: drawn scrimmage safety violates iff pbp_loss[i] < 100-q; repick safety with enough loss, else non-safety. Offline (290204 real queries): yl80-89 real .35 drawn .58 safeg .06; yl90-94 7.9/5.3/4.7; yl95-99 18.7/12.8/10.5; non-safety draws changed 0. Replay on real states cannot show the sim yl<80 excess (draw path differs).
## E172g SAFEGP (scripts/mod25e_safegp.py, SAFEGP=1, built 2026-10-18, NOT installed)
- Rule (declared before run, ONE): a drawn punt row with points_def==2, points_off<6 is served as a safety iff fp_raw[idx] <= q (sim snap at least as deep as the source snap). Else repick from non-safety punt rows (code 2, pdf 0, po<6) as the K_STATE nearest by fp_raw to q, uniform; keep engine clock/timeout fields; yards = new row yards + (drawn - source) rating shift. Wraps dv._G["pol"].
- Offline: `mod25e_safegp.py check` -> artifacts/mod25e3/safegp/check.json, log tests/scratch/e172f/safegp_check.txt. Arms: own (nn draw at own state) and far (draw from another punt state, emulating the sim gap).
- Offline result (measured, 22007 real 4th-down punt states, per 1000 punts, 90%): far arm yl1-49 drawn .55 served 0; yl50-79 real .43 drawn .64 served .14 [0,.29]; yl80-89 real 1.22 drawn 1.22 served 1.22; yl90-94 not cut. Own-state arm yl50-79 .50 to .36. Non-safety punts changed 0 of 44014 draws; 12 of 26 safety draws repicked, 0 unresolved. Looks: 5 bins x 2 arms.
- crH patch (deferred, not applied): after the SAFEG block (so after v1ad, stb/stbk, geo; before gstop): `import mod25e_safegp as sgp` / `if sgp.enabled(): sgp.install_safegp()`; label `if os.environ.get("SAFEGP") == "1": label = label + "sp"` after the sg label. Installed last = outermost, sees the raw engine draw before gstop/ldr hooks.
- Joint smoke bar (SAFEG=1 SAFEGP=1): safeties/game toward .0586 (target within +-.02); yl<80 safeties toward real .0052/game (<=.01); yl>=90 not cut below real; def TD/game .458 +-.01; pts/g unchanged within noise; guards pass.
## Next
Owner/orchestrator applies the crH patch when no sim run is live, runs joint smoke. Return-TD share still open. Kick/punt return TDs not in sim play log.
