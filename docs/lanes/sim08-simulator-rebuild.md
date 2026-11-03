# SIM-08 simulator rebuild

## Goal
Repair transition and terminal-state behavior, then diagnose late-game bias
without fitting weights or changing served probabilities from these diagnostics.

## State
**Measured:** September 28 repaired quarter boundaries, zero elapsed transitions,
timeouts, and historical overtime termination/conversion rules. The final fixed
replay has 10,002 games, zero caps, but only 2/5 key-number bands pass.
Log-loss improvement versus the training histogram is -0.000656
[-0.013844, +0.014832], probability_positive 0.370370. Versus the preceding
engine, improvement is -0.001595 [-0.002528, -0.000795], probability_positive 0
in 27 conditional season draws; Monte Carlo and prior selection are omitted.
All four effects remain registered unresolved; no serving change.
[Complete 147-look result, including superseded arm](../simulator_overtime_repair_20260928.md).

## Tried
- **Measured:** conversion repair reconciles 768/768 final margins. Full repair
  history: `done/sim08-transition-repair-history.md`.
- **Measured:** 12 neighbor looks and four chronological comparisons diagnose
  conditional scoring and clock behavior. Chronological one-score elapsed delta
  is -2.11 seconds [-2.88, -1.43], probability_positive 0 in 2,000 draws;
  scoring frequency is +0.17 points [-0.43, +0.77], probability_positive 0.706.
- **Measured:** three validation source-age slopes and six training rolling-era
  slopes bring that diagnostic family to 25 looks. Excluding 2009, training slopes
  are +0.830 [0.282, 1.391], +0.741 [0.118, 1.315], and +0.802 [0.334, 1.286]
  seconds/year, probability_positive 0.999, 0.989, and 0.9995. Cross-role game
  identity limits those bootstrap intervals; these are not causal rollout effects.
- **Measured:** 144 instrumented rollouts preserve all same-seed outputs, with
  zero caps. Exact matching yields 2 selected scoring transitions; the separately
  declared timeout-omission sensitivity yields 9 in 9 rollout clusters from
  6 scheduled games. Both fail declared 30-transition/10-cluster support gates.
  Neither computes an effect. Two rollout endpoint looks are recorded.
- **Measured:** the expanded 96-game/eight-seed sample yields 67 matched scoring
  transitions in 66 rollouts from 40 scheduled games. Its fixed-selection paired
  bootstrap estimates selected-minus-actual elapsed discrepancy -0.751 seconds
  [-2.728, +1.173], probability_positive 0.2175 (2,000 draws; 55 actual games).
  This fourth rollout endpoint look is recorded as `unresolved_below_power`.
- Full methods, artifacts, uncertainty and limitations are retained in
  `done/sim08-late-game-diagnostic-history.md`.

## Next
Historical rule defects are repaired; broader predictive accuracy is unresolved.
Preserve `artifacts/sim08_overtime_repair/20260928T171712Z/` with all three arms,
source/input hashes, 30,006 simulations and actual-game losses. Any new mechanism
or 2025+ rules extension needs a declaration before another endpoint is scored.
Do not patch key-number masses, fit on these diagnostics, or promote this engine.

## Declaration (2026-11-02)
Unit 3 market-anchored sim is predeclared in `docs/sim08_market_anchored_declaration.md`
(34 looks, primary L4 blend v served-only recalibration, 2020-2025 1,537 games, LOSO 6
folds, planted positive control). Next: commit it, then build the anchored arm (no code
comments), 20-game smoke, planted control, full run one job, <=4 workers.

## Open
Finite-draw probability_positive of 0 or 1 is not impossibility or certainty.
The deficient margin-3 mass and rollout calibration remain unresolved. Support
failure closes no signal; these diagnostics authorize no weight or serving change.

## Unit 3 build (2026-11-02, subagent)
Built `scripts/sim08_market_anchored.py` (imports sim10_drive engine unmodified; `engine()` rebuilds locked
defaults, `play()` calls `Engine.game` with ratings set from latents x_diff, x_env). Anchoring solve:
`transmission_fit` (OLS of simulated margin/total on design latents, 2x2 inverse in `solve_latents`).
Smoke: `python scripts/sim08_market_anchored.py smoke --design-n 1500 --draws 150 --x-diff-range 0.8 --x-env-range 0.5`
-> tests/scratch/sim08_anchor/smoke.json. Positive control (as declared, planted on the LOSO
recalibrated served logit, 500 reps): `... control --deltas 0.0005 0.0010 0.0020 0.0040 0.0080` ->
control.json; detects all (pp>=0.90 in 100% of reps) because the planted gain is near-deterministic per game
(caveat; a noisy plant would need an amendment). Engine 0.44 s/game-draw at 1 worker.
Open before full run: per-fold kernel pool needs LOSO upstream (blocked per 9a0a951d); opener totals
missing for 11 of 1537 games (spread-only fallback in `anchored_draws`).
Smoke result (measured, fold 2025, 20 games, 150 draws, design 1500, x_diff +-0.8, x_env +-0.5): ANCHOR FAILED.
Margin slope 17.2 (SE 0.9) fine but total response to x_env is non-monotone (linear R2 .074, quadratic .181;
probe: total 70/47/59 at x_env -1/0/+1), so the 2x2 linear inverse put x_env at -0.3..-1.6 and sim totals 65-79
v opener 36-55 (signed err +23.8). Margin mean err -2.7 (z rms 4.3). Cause: design range far outside engine
support (s=0.969, real xq ~ +-0.2). Next: re-measure with design ranges from the sim-world xq distribution, fit
total on x_env with |x_diff| term or monotone local lookup (a mechanism measured, not a knob), re-smoke.
Cost: 0.26 s/game-draw design, 0.58 s/draw anchored (contended) -> 1000 draws x 1537 games = 100-250 h single
worker; infeasible at declared 1000 draws, needs a declared amendment (draw count).

## Unit 3 re-anchor (2026-11-02, subagent 2)
Support from engine latents (support.json) x_diff +-0.5, x_env +-0.21; script now solves bounded joint quadratic
(response/solve_latents), smoke takes --design-csv. Smoke 9,600 design/300 draws: margin err -0.41 z -0.97 pass;
total +0.95 z +3.10 FAIL (persists across design size). Amendment 1 appended to the declaration (N=372, noisy plant).
Next: smoke with design pool drawn from held-out fold contexts (code change to design_run pool), then the noisy
control code, then the full run (one job, 4 workers). Per-fold LOSO kernel pool still open.

## Unit 3 re-anchor 2 (2026-11-02, subagent 3)
Code: engine() now sets SIM10_JOINT=1 (new default; was unset in old smokes = pre-SIM-11 engine), `support` command,
`smoke --heldout-pool 1 --save-design`, per-game sim_home_cover/loss stored, `control --noisy 1` (Amendment 1 item 5:
kappa bisection on LOSO blend, sd = sqrt(train_sd^2 + (2/sqrt(draws))^2); --sd-from-smoke or --train-sd).
Support on joint engine (tests/scratch/sim08_anchor/support_joint.json, 92,160 games): x_diff 0.1/1/5/25/50/75/95/99/99.9% =
-0.492/-0.380/-0.267/-0.109/0.001/0.109/0.267/0.376/0.502; x_env -0.205/-0.153/-0.107/-0.041/0.003/0.047/0.111/0.155/0.205;
sd 0.162/0.066, corr 0.007. Unchanged vs old (latents are engine-independent); box stays +-0.5/+-0.21.
Smoke5 (joint, held-out-fold pool, 9600 design, 300 draws, 20 games) launched PID 33748 (python child; parent 33164),
out smoke5.json/log, design saved design_joint_heldout.csv. Next: read z, then noisy control on smoke population.
Result (measured): smoke5 and identical-seed rerun smoke6: margin err -0.645 (z -1.56), total err +0.145 (z +0.38), pass.
Total bias cleared by held-out-fold context pool (confounded with joint-engine switch, inferred). 0.116 s/draw, 36 GB free.
Transmission: margin 1.93+37.5 xd+3.9 xe (R2 .403); total 45.7+2.5 xd+47.1 xe (R2 .162, quad .172). Amendment 2 appended.
Noisy control coded and executed (smoke6 population, 20 reps, not a look): deltas .001/.008 kappa .017/.045, realized .00044/.0071.
Next: per-fold LOSO kernel pool decision, then full run (one job, 4 workers, N=372, heldout-pool per fold); control verdict
only after full-run train sd. No processes left running.
