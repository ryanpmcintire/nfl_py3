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
Smoke done (872 s), moved to artifacts/mod25e3/smoke/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtb_s31 (measured, seed 31, 4 seasons). Touchback share real v off v on: punt .0935 / .0000 / .0962; kickoff .504 / .003 / .506. Punt start mean 74.35 / 74.33 / 74.36; kickoff 76.05 / 75.78 / 76.18. Non-TB punt start 73.77 / 74.33 / 73.76.

## 3-seed result 2026-10-08 (measured; artifacts/mod25e3/e102_boot/e100.txt, xfdcdtb_era/; season bootstrap within seed, 1000 reps)
ypw2xfdcd -> ypw2xfdcdtb (real): margin SD 15.75 -> 15.50 [-.44,-.08] (14.63); noise 185.9 -> 183.7 [-5.9,+1.5] (165);
strength 62.55 -> 56.67 [-10.3,-1.8] (57.96); xq cov +8.1 -> +1.6 [-10.8,-1.9] (-6.4); Q4 slope -.044 -> -.057 (-.060);
mass3 .1247 -> .1253 (.141); pts 46.12 -> 46.10 (45.21). Away from real: late r2 .142 -> .132 [-.019,-.002], 3 seeds only
(.146); P(final 3 | tied after reg) .573 -> .546 [-.049,-.002] (.682). KEPT: the hook matches its own real touchback
rates and five of the eight aggregate checks move toward real; the strength rise from E100 is resolved by this fix.
New base label crHpqokgndecsmfwtjo2as2ypw2xfdcdtb (base env + YLM=1 YLM_CLASSES=punt WFG=1 PATY=1 FGD=1 CDR=1 STB=1).

## Next
Done. Open items move to mod25e-generator-fidelity.md Next 0.
