# SIM-10 drive-level resampler (replaces the hook stack as the generator engine)

## Goal
Pass the generator gates against real 2009-17 games with one engine whose unit is a real
possession segment (first snap to next first snap: drive, score, PAT, kick, return), drawn by game
state. Owner 2026-10-24: keep working until it passes; no checkpoint stops; owner frustrated at
7.5 h without a finish - bounded plan below, then lock and report.

## State (2026-10-24 evening, scripts/sim10_drive.py, artifacts/sim10/)
Pipeline: build -> latfit -> real -> hgbsel -> outfit -> patfit -> fittilt -> sim. All artifacts
exist; nothing needs rebuilding. Best committed config ("wk"):
`sim --models 1 --pat 1 --mufree 1,4 --dr 1 --h artifacts/sim10/bandwidths_week.json --seed S
--workers 11` (~10 min per seed; run in background, never two sims at once).
Engine: kernel on yl, sd, log secs-in-half, timeouts, season, week (per-phase bandwidths fitted by
held-out likelihood of the tilted draw) x clock-feasibility mask; tilt to posterior strength
(latent / s, s=.969) and doubly-robust TD/FG/def-score targets + HGB elapsed; end-of-half phases
(regimes 1,4 = last 2 min of each half) tilt outcomes only; PAT at exact post-TD score; OT
state-only. Latents moment-matched to real LOGO covariances (latent_fit.json).
Env options (experimental, off by default): SIM10_MU_ONLY=1,4 (end-of-half tilt strength only),
SIM10_MU=pregame, SIM10_LATENT=<path>. CLI option --qstrata 1 (quarter strata; rejected).

## Results (8x8 worlds; real 90% interval)
wk, seeds 11/12/13 avg: passes margin SD, noise, mass 3/7/10/14, pts, OT, ties, home 2.20, Q4
slope -.077. Fails: strength 48.8 [51.4,64.9], r2 w10-18 .126 [.131,.161], xq -19.3 [-17.7,6.6].
All three = one mechanism: team strength transmits ~5% weak. Measured cause: game EPA responds .955
to latent (should be 1); end-of-half plays (~8%) are not strength-matched (respond ~.5).
SIM10_MU_ONLY=1,4 seed 12: strength 52.5 PASS, xq -14.2 PASS, home/Q4 pass, r2 .126 (delta CI
includes 0), but mass3 .132 FAILS (end-of-half PAT/FG decisions lose their outcome targets).
Seed 13 was running at handoff: artifacts/sim10/muonly_s13.log / gates.json.

## Tried and rejected (measured)
Mean-matching tilt on yl/sd; stacking layers and per-phase intercepts (worse held-out when
cross-fitted); phase-restricted DR; DR with held-out residuals (pts +1.2, xq -23.6); strength tilt
in all phases with outcome targets; pregame-only strength; week as HGB feature; quarter strata
(better xq/home, worse strength/r2). Diagnostics live in the session scratchpad (gone after clear):
rebuild from descriptions in this lane if needed.

## Next (bounded; then lock, commit, push, report table to owner)
1. Read muonly_s13. 2. Implement end-of-half tilt = strength AND scoring targets (td, fg, ds) but
NOT elapsed (keeps PAT/FG decision structure, frees the clock to the kernel); run seeds 12, 13.
3. Pick the config with the most gates passing over 3 seeds (wk is the fallback), set it as the
CLI default, commit + push, report the pass/fail table. No further experiment rounds.

## Open
OT strength-neutral (371-segment pool too small to tilt). Real strength 58 sits high in its own
interval; sim is inside delta-CI on most seeds.
