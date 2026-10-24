# SIM-10 drive-level resampler (replaces the hook stack as the generator engine)

## Goal
Pass the generator gates against real 2009-17 games with one engine whose unit is a real
possession segment (first snap to next first snap: drive, score, PAT, kick, return), drawn by game
state. Owner 2026-10-24: finish it, lock the best config, report.

## State (LOCKED 2026-10-24 night, scripts/sim10_drive.py, artifacts/sim10/)
Pipeline: build -> latfit -> real -> hgbsel -> outfit -> patfit -> fittilt -> sim. All artifacts
exist. Locked config is the CLI default: `python scripts/sim10_drive.py sim --seed S --workers 11`
= `--models 1 --pat 1 --mufree 1,4 --dr 1 --qstrata 1 --eoh outcome
--h artifacts/sim10/bandwidths_week.json` (~6-10 min per 8x8 seed). Verified: the bare default on
seed 11 reproduces wkq_s11 exactly (max abs gate diff 0.0; artifacts/sim10/sim10_s11/).
Engine: kernel on yl, sd, log secs-in-half, timeouts, season, week x clock-feasibility mask,
quarter strata; tilt to posterior strength and doubly-robust TD/FG/def-score targets + HGB
elapsed; regimes 1,4 (last 2 min of each half) tilt outcomes only; PAT at exact post-TD score;
OT state-only. `--eoh mu|muscore` keeps the rejected end-of-half variants runnable.

## Results (8x8 worlds, seeds 11/12/13; pass = 3-seed mean inside real 90% interval; 14 gates)
| config | mean pass | seed passes | strength (real 58.0 [51.4,64.9]) | r2 w10-18 (.146 [.131,.161]) |
|---|---|---|---|---|
| wkq (locked: wk + quarter strata) | 12/14 | 12,12,10 | 47.2 FAIL | .117 FAIL |
| wk (prior default) | 11/14 | 7,8,11 | 48.8 FAIL | .126 FAIL |
| muscore (end-of-half strength + td/fg/ds) | 10/14 | 9,11,13 | 49.2 FAIL | .115 FAIL |
| muonly (end-of-half strength only) | 10/14 | 9,12,9 | 50.9 FAIL | .129 FAIL |
wkq passes margin SD, cross-quarter catch-up, Q4 slope, mass 3/7/10/14, points, home edge, OT,
ties, noise variance. Logs: artifacts/sim10/<label>_s<seed>.log, gates in <label>_s<seed>/.

## Tried and rejected (measured)
End-of-half strength-only (mass 3 and 10 fail) and strength + scoring targets (seed spread wide,
catch-up and home fail on mean); mean-matching tilt on yl/sd; stacked layers and per-phase
intercepts; phase-restricted DR; DR with held-out residuals; strength tilt in all phases with
outcome targets; pregame-only strength; week as HGB feature.

## Next
None for SIM-10. Any future work is a new row: strength transmission.

## Open
Strength transmits weak in every config (sim strength variance 47-51 v real 58; game EPA
responds ~.95 to latent); strength variance and late R2 fail in all four configs; end-of-half matching did not fix it on the
3-seed mean. OT strength-neutral (371-segment pool).
