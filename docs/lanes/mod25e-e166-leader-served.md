# mod25e-e166-leader-served

## Goal
Locate why served 24+ leader Q3/Q4 plays diverge from pool expectation (E166), then E166b: name the "absorbed flip/score source row" path and the sim-chg-over-yards +0.4-0.5 gap.

## State (E166b, measured 2026-10-18, diag seed 97, 1 world, 1 season truncated to 120 games, 14913 run/pass plays, XFA env)
- Reproduced: in the log, 0.93% of non-flip run/pass plays have a log-idx source row that is flip/score (src yards mean -47 to -71, yl mean 4.4, downs 1-3), and chg minus inferred yards (YG[logidx]+ysh) = +0.452.
- The sim is correct: chg minus TRUE served drawn yards = -0.004 (27 clip cases of 13707); true served yards for absorbed plays mean +0.83 (3 of 127 have |y|>=15). The ball moves by served yards; next drive start and points per drive are unaffected. Real >20 gap rate analogue: |chg-served|>20 is 0.2% (matches real .26%).
- Cause (read): mod25e_gz.py:127-128 repicks goal-zone plays from row j (flags and yards = YG[j]+shift) but sets new["idx"]=i0 and st["shift"]; GZ PL.append (mod25e_gz.py:~145-148) logs (v[0], v[1], sh) with v[1] the incoming (Geo) row, dropping j. Inferred yards YG[v1]+sh therefore differ from served by YG[j]-YG[v1] (mean +0.45); when the incoming Geo row was a TD/flip goal-zone row and GZ rewrote it (~29%), the log shows a flip/score source with non-flip served play. Geo/FLIPA/v1ad repicks copy flags and yards from the same row (geo.py:381-389), engine non-flip branch is yardline - yards (sim04_engine.py:1098-1121) - none are defective.
- Conclusion: E166 "absorbed rows" and "g_chg exceeds served yards" are a PLAYLOG attribution artifact, not a sim defect; no sim fix needed. E166 clean-yards leader result stands.

## Tried
Wrapper-chain logging (broke frame-depth layers; abandoned), PLAYLOG-class swap recording engine frame locals (works). Files: tests/scratch/e166/diag.py, diag.sh, a6-a9.py, diag.pkl (17 MB, deletable), diag_base.pkl may not exist.

## Next
1. Create scripts/mod25e_servedidx.py (NEW, default off SERVEDIDX=1; no comments): ORIG=gz.install_gz; src=inspect.getsource(ORIG); assert one occurrence each; replace `            st["shift"] = shift\n` by itself plus `            st["served"] = int(j)\n`; replace `            super().append(v if sh is None else (v[0], v[1], sh))` by `sv = st.pop("served", None)` then `super().append(v if sh is None else (v[0], v[1] if sv is None else sv, sh))`; exec into a copy of gz.__dict__ and call install_gz. Deferred crH install: at mod25e_crH.py:71 call servedidx.install_gz_served() instead of gz.install_gz() when servedidx.enabled(); label suffix "sx". Not installed.
2. Verify once: rerun tests/scratch/e166/diag.sh with SERVEDIDX=1 (diag.py needs the 3-line hook: import mod25e_gz/mod25e_servedidx; if enabled, gz.install_gz=install_gz_served, before sys.argv) then a8.py/a9.py. Pass bar: absorbed share 0.93% -> ~0 (legit residual only), chg minus inferred +0.45 -> 0, served play records byte-identical to base (seed 97), so pts/drive and pts/g are unchanged by construction.
3. Cleanup: delete F:/Repos/nfl_py3/artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekudwtpq2ugkyg2rtgw13gx2fa_s97 and tests/scratch/e166/diag.pkl.
4. Re-run the E166 leader decomposition with PLAYLOG served idx (rebuild S.pkl via a1.py after the install) to confirm no 24+ bookkeeping excess.

## Open
Module not yet written (tool cap hit). No python process left running.
