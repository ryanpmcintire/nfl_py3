# SIM-11 strength transmission (closes SIM-10's two failing gates)

## Goal
Get strength variance (SIM-10 47.2 v real 58.0 [51.4, 64.9]) and weeks 10-18 R2 (.117 v .146
[.131, .161]) inside the real interval on the 3-seed mean without losing the other 12 gates.
Owner 2026-10-24: get them close to reality. ROADMAP SIM-11.

## State
LOCKED as the CLI default 2026-10-24: `python scripts/sim10_drive.py sim --seed S --workers 11`
= `--droof 1 --drmu 0 --mulabel 1 --strength artifacts/sim11/strength_terms.json` on top of the
SIM-10 lock. Bare default seed 11 reproduces s11lab_s11 exactly (max gate diff 0.0, sim11_s11).
3-seed mean (artifacts/sim10/s11lab_s1{1,2,3}/, logs artifacts/sim11/): 13/14. Strength 52.1/
50.9/54.4 = 52.4 PASS; R2 .118/.125/.117 = .120 FAIL; tie .0033, xq -14.95, q4 -.076, mass3
.1383 (all pass, near edges). Old SIM-10 config: `--strength "" --droof 0 --drmu 1 --mulabel 0`.
Mechanisms: --mulabel tilts drawn drives' strength label to cx*theta + cs*(sd - mean)
(posterior_table) with outcome targets at full theta; --strength per-phase terms
(sim11_strength.py strfit; LOSO td loss .45914 raw v .45921 oos); --droof/--drmu 0 honest,
strength-free DR residuals. Points per unit strength .94 -> 1.00 of real.

## Tried (measured; probes 11x2 seed 21, artifacts/sim11/draws_<tag>.parquet)
- Rejected: phase-scaled strength target (unstable); outcome-only tilt (team quality lost,
  seed 11 strength 46.3, R2 .107, breaks mass3/q4/xq); drive-EPA target (redundant, 5x slower);
  plays-per-drive target (per-play EPA .894 v .942). Code removed.
- Special teams + no-play penalties: ~3% of real persistent margin strength (not modelled).

## Next
Unit 3, team-wide form latent. Weeks 10-18 R2 by predictor, real v sim: flat season-to-date .153
v .145; last 3 games .087 v .068; EWMA (gate) .140 v .118. Real off x def cross-cov lags 1-4 v
9-15: 2009-17 -.96 v -.47, 2018-25 -1.09 v -.69, 2009-25 diff -.45 [-1.02,+.13] P .91; sim
flat. Single-profile fit cannot pin the decay (free fit degenerates; fixed .5-.8 gives net form
var ~1e-3, P+ .67-.75); fit off, def, cross profiles jointly on 2009-25 before building.
Generator: mod25_generator.gen_world_latents + sim10_drive.cmd_latfit (latent_fit.json).

## Open
Persistent plays-per-game: real LOGO cov 3.56 v sim 0.8; not a gate.
