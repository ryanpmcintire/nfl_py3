# SIM-10 drive-level resampler (replaces the hook stack as the generator engine)

## Goal
Pass the MOD-25e generator gates against real 2009-17 games with one engine whose unit is a real
possession segment (first snap to next first snap: drive, score, PAT, kick, return), drawn by game
state. Owner 2026-10-24: keep working until it passes; no checkpoint stops.

## State (2026-10-24, scripts/sim10_drive.py, artifacts/sim10/)
Pipeline: build -> latfit -> real -> hgbsel -> outfit -> patfit -> fittilt -> sim.
Best config: `sim --models 1 --pat 1 --mufree 1,4 --dr 1 --h bandwidths_week.json` (+ testing
`--qstrata 1`). State dims: yl, sd, log secs-in-half, timeouts, season, week; per-phase bandwidths
fitted by held-out likelihood of the tilted draw. Draw = kernel x clock-feasibility mask, then
tilt to posterior strength (latent / s) and doubly-robust targets for TD/FG/def score (strength-
tilted drawn mean + HGB(sim state) - HGB(drawn)); elapsed target = HGB(sim state). End-of-half
phases tilt to outcomes only. PAT at exact post-TD score. OT state-only.
Latents (latent_fit.json): drift phi by weighted LS on 15 within-season lags (off .9917, def 1);
variances and off/def correlation moment-matched so the generator reproduces real LOGO
covariances (off .00622, def .00245, cross -.00075).
Outcome models: HGB (LOSO-selected), monotonic in strength for TD (+) and def score (-); stacking
and per-phase intercepts rejected (worse held-out when cross-fitted); week feature rejected
(held-out worse 6/8).

## Results (gate runs, 8x8 worlds; real 90% interval in brackets)
wk config, seeds 11/12/13: margin SD 14.73/14.80/14.69 ok; noise 168 ok; mass3 .141 ok; pts ok;
OT .061-.063 ok; ties .0031-.0036 (edge); home 2.08/2.12/2.41 [2.17,2.71]; Q4 slope -.080/-.080/
-.072 [-.080,-.039]; xq -21.5/-18.4/-17.9 [-17.7,6.6]; strength 48.7/50.9/46.9 [51.4,64.9];
r2 w10-18 .128/.134/.117 [.131,.161]. Seed 12 is inside or delta-includes-zero on every gate.
Remaining gap is one mechanism: strength transmission (game EPA responds ~.95 to latent; Q3
under-responds .88 because its time-blind kernel mixes Q4 drives). Catch-up excess follows
(persistence term ~.75 x strength). Testing quarter strata (wkq) for that.

## Tried (condensed; details in commit messages and artifacts)
PAT model (key numbers), OT rule-ended drives, clock-feasibility mask, doubly-robust targets,
latent persistence moment-matching (+5 strength), week dimension (+r2 .116 -> .126 avg),
DR restricted to scoring outcomes (DR biased elapsed -3 s), no strength tilt at half ends (ESS).
Rejected: mean-matching tilt on yl/sd, stacking layers, phase-restricted DR, posterior-free mu,
strength tilt in all phases, week as HGB feature.
Diagnostics in scratchpad: diagpar.py (parallel draw logs; env DR, MUFREE), phan.py, epaph.py,
q4an.py, qslope.py, homean.py. Under 1,000 games strength slopes are noise.

## Next
Read wkq_s11-13; if strength/r2/xq move inside, refit fittilt under quarter strata, rerun 3 seeds,
then commit script + lane + ROADMAP and update the MOD-25e lanes.

## Open
OT strength-neutral (pool too small). Real strength 58 sits high in its own interval.
