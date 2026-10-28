# SIM-11 strength transmission (closes SIM-10's two failing gates)

## Goal
Get strength variance (SIM-10 47.2 v real 58.0 [51.4, 64.9]) and weeks 10-18 R2 (.117 v .146
[.131, .161]) inside the real interval on the 3-seed mean without losing the other 12 gates.
Owner 2026-10-24: get them close to reality. ROADMAP SIM-11.

## State
LOCKED default (commit 0df0a23d): `python scripts/sim10_drive.py sim --seed S --workers 11`
= `--droof 1 --drmu 0 --mulabel 1 --strength artifacts/sim11/strength_terms.json`; bare seed 11
reproduces s11lab_s11 exactly. Random schedules (SIM-10 protocol), seeds 11-13: 13/14, strength
52.4 PASS, R2 .120 FAIL. Old config: `--strength "" --droof 0 --drmu 1 --mulabel 0`.
Flag `--realsched 1` (default 0): each sim season plays the matching real 2009-17 schedule
(16 games, 17 weeks), teams randomly mapped to slots: the gates' own measurement design.
3-seed means on real schedules (artifacts/sim10/rs_s1*, rs10_s1*): SIM-11 11/14 (fails strength
49.8, R2 .116, q4 -.0806 v floor -.0797); SIM-10 config 9/14 (strength 46.6, R2 .116, xq -20.6,
q4 -.084, mass14 .0510). Seed-level game-path noise ~1.5-2 strength points.

## Tried (measured; probes 11x2 seed 21; probe per-phase noise ~+-.08)
- Kept: label-matched strength target (tail effective sample 13-31 -> 75-232; TD target miss
  5-8 pp -> .3), per-phase strength terms, OOF and state-only DR residuals.
- Rejected: phase-scaled target; outcome-only tilt; drive-EPA and plays-per-drive targets;
  label tilt in end-of-half regimes too (end2h points 2.83 v real 1.80, EPA all .917 v .942);
  adjacent-season labels (label var +4.5% off, +17% def; not worth a rebuild); recalibrating
  latent variance to simulated EPA (would hide the .94 transmission; compensating knob).
- Form (weeks-scale team strength): opponent-adjusted offence lags 1-4 minus 9-15 +.97e-3
  [+.10,+1.98] P+ .965 (2009-25); exact-likelihood LOSO, net adjusted EPA: 2009-17 best decay .8
  (held-out +1.22 nats, form var 3.3e-3, season 8.9e-3); 2009-25 decay .8 +.27 nats, var 1.9e-3.
  Real schedules make consecutive opponents dissimilar (opp strength cov lags 1-4 -.84 v -.37,
  P .00), which hides form in raw EPA. Estimated R2 gain from form alone ~+6% (inferred).
- Diagnosis (measured, artifacts/sim11/draws_lab21.parquet): achieved drive mu on sim true strength slope .68 in every non-end-of-half phase (sd tgt .103 v a_mu .071); drive EPA slope already matches real (lab21 all .664 v real .658), so the .94 is the label-matched target's cx attenuation: real mu (posterior mean) has sd ~.07 < sim true sd .103, so deconvolved targets leave support (old mulabel 0 = TD miss, tail ESS 13-31). Resampling cannot repair it without new support or a joint outcome/mu target; no legit fix found.
- Rejected 2026-10-27: additive form latent (`--form 1`, SIM10_FORM; ratio 3.3e-3/8.9e-3, decay .8, seed 11 realsched): margin_sd 15.61 (fail), noise_re_var 191 (fail), q4 -.044, mass_3/7 fail, strength 53.0 pass, R2 .111 (no gain). Variance added on top inflates noise. Flag left default 0 (mod25_generator.py gen_world_latents form_ratio, sim10_drive.py FORM_*).

- 2026-10-27 unit: candidate (a) skipped as the same family as the rejected outcome-only tilt (a residual tilt still reweights drives by outcome, whose response to strength is the .664 already matched). (b) variance-preserving split added as `--form 2` (SIM10_FORM=2; mod25_generator.py `form_preserve`: persistent scaled by sqrt(1-ratio), form added); results below.

## Next
1. Adopt `--realsched 1` as the protocol for every new comparison (like-for-like with the
   real gates); random schedules read strength 2.6 higher for SIM-11 (0.6 for SIM-10).
2. Remaining persistent-strength deficit (real schedules): per-play EPA LOGO .88, margin LOGO
   .88 of real. Engine per-play EPA transmission .94 is the main piece (within-outcome drives
   come from teams ~2/3 as strong: shrunk labels); then excess catch-up (q4 -.076 v -.060),
   special teams (~3%), form (R2).

## Open
Persistent plays-per-game: real LOGO cov 3.56 v sim ~1.0; not a gate.

2026-10-27 result: `--form 2` seed 11 realsched (artifacts/sim10/rsfp_s11) vs default (rs_s11): strength 40.5 v 48.6 [51.4,64.9], R2 .092 v .117 [.131,.161], noise_re 182.7 v 170.3 FAIL, q4 -.083 v -.073, margin_sd 14.94, xq -17.5, mass_7 .079 fail. Worse on every gate; no 3-seed run. Variance moved to form is not seen as persistent strength through the .68 transmission. Not default. Next: (c) excess catch-up, or a new-support transmission fix.
