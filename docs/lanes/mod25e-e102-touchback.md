# mod25e E102 touchback event

## Goal
STF shifts punt/kickoff receiving starts off the rule spot (punt touchbacks .001 sim v real ~.094). Fix: draw touchback as its own event with a fitted probability, place it at the rule spot, apply the STF shift only to non-touchback returns. Flag STB=1 (label suffix tb), default off.

## State
- scripts/mod25e_stb.py: `cmd_fit` (real 2009-17 pool rows from sim04_engine.build_transition_frame; TB := next_yardline == 80 for punts, in {75,80} for kickoffs; LOSO over 9 seasons by held-out log loss) writes artifacts/mod25e3/stb/fit.json and an.txt; `install_stb` is the hook.
- Fit (measured): punt TB logit on punt spot fp, fp^2 beats constant by +854 LOSO ll (9/9 seasons, fp alone +738); kickoff constant .504 (spots 75 .341 / 80 .659). Kicking-team season-rate proxy (own game excluded, season-demeaned): punt -0.01 ll, kickoff +154 ll (a real team effect, but no measured map to the sim STF latent, so not served).
- Wiring: scripts/mod25e_crH.py installs STB after YLM and before STF (so STF is outer); scripts/mod25e_stf.py skips rows with `stb_tb`. install_log_sync stays outermost. Label suffix tb after cd.
- Sim has no calendar season, so kickoff spot is drawn from the pooled 2009-17 TB-spot mix, not a per-season rule.
- Smoke: see Tried.

## Tried
Smoke launched 12:04 (python chain PIDs 42892,10752,1992,37416,16332; log artifacts/mod25e3/stb/smoke_on.log; out artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtb_s31). Not yet scored. Score with: `.tools/uv.exe run --no-sync python <scratchpad>/cmp.py e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtb_s31` (compares real v off smoke v on; copy cmp.py from the session scratchpad if absent), then move output to smoke/ and kill leftover PIDs.

## Next
Orchestrator 3-seed run if smoke passes: base env + STB=1, `scripts/mod25e_crH.py e5 --workers 3 --seed {11,12,13}`.

## Open
Team-level kickoff TB effect (+154 ll) needs a measured map to the STF coverage latent before it can enter.
