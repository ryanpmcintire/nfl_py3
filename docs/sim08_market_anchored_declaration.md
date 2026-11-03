# SIM-08 unit 3: market-anchored simulator, predeclaration

Filed before any anchored simulation is scored. No existing declaration covers
this unit (searched `docs/` for SIM-08 and declaration files; the SIM-04 LOSO
re-grade of 2026-09-26 in `docs/sim04_unit_log.md` is the baseline this unit
extends, not a declaration). Lane: `docs/lanes/sim08-simulator-rebuild.md`.
ROADMAP SIM-08 unit (3) of four; unit (4) is the re-grade this file declares.
Scope: research and paper decisions only. Scoring a mechanism before this file
is committed is a protocol breach.

## 1. Hypothesis and named mechanism

The 2026-09-26 re-grade graded an engine that did not see the market line, so
its losses measured engine defects and the priced team-quality ceiling. The
hypothesis: if the simulator takes its pregame location from the opener
(spread and total) and contributes only shape and interaction, its discrete
margin distribution conditional on the line prices cover, push and miss better
than the served read, or adds a fitted term to it.

Mechanism, three named parts, no free scale:

1. Strength anchor. The persistent team-strength latent difference is set so
   the engine's expected home margin equals the opener-implied margin. The map
   from latent difference to expected margin is the engine's own measured
   margin-on-state transmission (SIM-11 lane: slope 26.7 simulated v 29.8 real;
   re-measured per training fold from simulated worlds), inverted per game.
   The 7% transmission deficit is reported, never multiplied back in.
2. Environment anchor. A pace and scoring environment latent is set so the
   engine's expected total equals the opener total, through the engine's own
   measured response of mean total to that latent, measured per training fold.
   SIM-04 measured total slope on opening total 0.343 against real 0.867; the
   anchored arm is judged on whether anchoring removes that gap by construction
   and whether the shape after anchoring is correct.
3. Shape. Drive sequence, clock, catch-up, overtime and key-number mass come
   from the locked SIM-10/SIM-11 engine unchanged. No mass at 3, 7, 10, 14 or
   17 is patched. The known defect (mass at 3 .103 v .152) is reported as
   measured.

Outputs are checks, not knobs: anchored mean margin and total matching the
opener inside Monte Carlo error checks the inversion; it is not a fit target.

## 2. Population and seasons

- Graded: the 1,537 opener-graded regular-season games, 2020-2025, the same set
  as the 2026-09-26 re-grade (`artifacts/sim04_loso/20260926T203158Z/`), so
  deltas are comparable. Playoffs excluded. Pushes retained (three-way).
- Engine kernel pool, strength terms, transmission and environment response
  curves, and blend weights use seasons strictly other than the held-out
  season. Held-out season s enters no fitted object of its own fold
  (leave-one-season-out, six folds).
- Draws: 1,000 per game per arm (ROADMAP unit 4 floor); seeds recorded.

## 3. Chronology

At the prediction timestamp (opener post) the unit may use: opener spread and
total; prior-games-only strength and rosters as the feature builder enforces;
the locked engine. It may not use the closing line, any line move, the final
score, inactives posted after the opener, or any same-season future game for a
parameter. The runtime leakage guard in the feature builder stays on. Grading
is at the opener; a close-graded number never vetoes.

## 4. Baselines (identical games, folds, draws)

- Market: training-season discrete margin histogram conditional on the opener
  line (no model).
- Model-only: the base model's calibrated cover probability before situational
  terms.
- Served: the served calibrated four-term probability, mapped to
  cover/push/miss by the same discrete mapping the card uses.
- Served-only one-term LOSO recalibration: the bar the 2026-09-26 blend failed
  to clear (-0.0004 [-0.0011, +0.0003], probability_positive 0.14).

## 5. Primary metric and grading

Primary: three-way (cover, push, miss) log loss at the opener, computed from
the discrete margin distribution conditional on the line (key numbers 3, 7, 10,
14, 17 carry their own mass); positive delta means candidate better; 95%
interval by block bootstrap over the 107 week blocks; `probability_positive`
reported. The primary contrast is L4: the fitted-term blend (served logit plus
anchored-sim logit) against served-only LOSO recalibration, because one
calibrated probability selects every pick and the simulator enters only as a
fitted term. Brier and a decile reliability table for every arm. No Gaussian or
pooled-residual mapping is served; a smooth read, if built, is a challenger
that must beat the discrete read at the opener. Fold coefficients (served
logit, sim logit, intercept) are reported per fold with range; the
decisive-game record (games where blend and baseline pick different sides) is
reported before the headline.

## 6. Looks, counted in advance (34)

Registry family `sim08_market_anchored` unless stated. Every look is recorded
with `nfl-ats weak-signals record` as `unresolved_below_power` before any
write-up calls it settled.

| id | look | n |
|---|---|---|
| L1 | anchored sim (spread+total) raw v served, 3-way log loss | 1 |
| L2 | L1 v model-only | 1 |
| L3 | L1 v market histogram | 1 |
| L4 | blend v served-only recalibration (PRIMARY) | 1 |
| L5 | blend v served raw | 1 |
| A1 | secondary arm, spread-only anchor: L1-L5 | 5 |
| S | per-season deltas 2020-2025 for L1 and L4 | 12 |
| P | push-line log loss (about 34 pushes) | 1 |
| B | opener-line bands (<=3, 3.5-6.5, 7-9.5, 10+) for L4 | 4 |
| K | key-number mass 3,7,10,14,17 v real, conditional on line (family `sim08_market_anchored_shape`) | 5 |
| T | total slope on opening total v real 0.867 | 1 |
| O | tied-late-game overtime rate v real | 1 |

Total 34 (5 + 5 + 12 + 1 + 4 + 5 + 1 + 1). Coefficient stability, reliability,
Brier and decisive-game record are descriptive and reported with the primary.
Any extra cell or arm adds to the count and is declared as an amendment below,
never edited in. No cell is presented as the finding unless it is L4.

## 7. Leave-one-season-out fitting plan

Every parameter is chosen out of season and the held-out season scored;
in-sample and out-of-sample results and their gap are reported together.

- Latent-to-margin transmission and latent-to-total response: measured by
  simulating training seasons' worlds in each fold; held-out season excluded.
  Measured mechanisms cited to the fold artifact, never tuned.
- Blend: weights on logit(p_served), logit(p_sim) and an intercept, fitted by
  maximum likelihood on the five training seasons per fold, scored on the
  held-out season; pooled over folds only for the reported delta.
- No threshold, cut point, bandwidth or draw count is selected on graded
  games. Kernel bandwidths and strength terms stay locked at SIM-11 defaults
  (commit 0df0a23d). Any engine change needs its own declaration.

## 8. Positive control and detectable effect

Before any L4 verdict, the evaluator is run on the 1,537 games with a planted
candidate: the served logit plus a planted term whose weight on the realised
outcome is set analytically to a target expected log-loss gain delta (planted,
so detection is not circular). 500 bootstrap replicates, same block bootstrap,
same folds. Detection means probability_positive >= 0.90 in at least 80% of
replicates. Targets: delta 0.0010 (scale of the blend-v-recalibration contrast;
prior interval half-width about 0.0007, standard error about 0.0004, inferred
detectable about 0.0009) and delta 0.0080 (scale of contrasts against raw
served and market; standard error about 0.0033, inferred detectable about
0.007). The control reports the smallest delta detected; that is the power
statement attached to every verdict.

Closure only on (a) a resolved wrong sign, whole interval on the wrong side of
zero, or zero split-half reliability, or (b) the positive control having shown
the evaluator detects an effect at least as small as the hypothesis implies.
Otherwise `unresolved_below_power`. An interval containing zero never closes
or declines this work.

## 9. CPU plan

One simulation job at a time, at most 4 workers, math threads pinned to 1
(OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS = 1). Check free RAM
before launch; the weekly pipeline and capture scheduler have priority and are
never stopped. A 20-game smoke run measures seconds per game before the full
six-fold run, and the run size is stated from that measurement. Launch as a
background tool call and wait for the completion notification; no polling or
sleep loops; list and kill only PIDs started, never by image name. No code
comments or docstrings in any new `.py`.

## 10. What a result implies

A positive L4 with probability_positive well above 0.5 authorises considering
a fitted term for the served probability through the existing promotion path;
it does not serve by itself. A result between 0.5 and 0.9 is
`unresolved_below_power` and keeps the work open. The simulator alone flips no
pick. The shape gate (K, T, O) is reported whatever L4 shows.

## Amendments

### Amendment 1 (before any look), 2026-11-02

No graded outcome, blend, or L-level metric has been computed. Smoke games were
compared to opener lines and opener totals only (market quantities, not results).

1. Design range. The first smoke used x_diff +-0.8, x_env +-0.5, outside the
   engine's own support. The range is now the 0.1% and 99.9% quantiles of the
   home-minus-away and mean-of-home-and-away expected-play-quality latents
   (xq) that the locked engine generates: gen_world_latents plus season_ratings
   from the locked latent fit, xq = (off_team + def_opp)/s with s = 0.969, over
   40 worlds x 9 seasons on the engine's real schedules (92,160 games; the
   engine's real schedules cover 2009-2017 only). Measured quantiles
   (tests/scratch/sim08_anchor/support.json): x_diff 0.1%/1%/5%/25%/50%/75%/95%/99%/99.9%
   = -0.498/-0.378/-0.267/-0.109/0.000/0.109/0.266/0.378/0.499 (sd 0.162);
   x_env = -0.208/-0.157/-0.110/-0.046/-0.002/0.043/0.105/0.150/0.198 (sd 0.066);
   corr -0.0005. Design box: x_diff in [-0.5, 0.5], x_env in [-0.21, 0.21],
   uniform. Targets beyond the box are solved at the box edge (bounded solve).
2. Transmission on that range (9,600 design games, 4 pooled seeds' draws;
   2,400-game fit shown): margin = 2.21 + 38.2 x_diff + 4.3 x_env (SE 0.96, 2.3),
   R2 0.400 linear, 0.401 quadratic: linear and monotone. Total is monotone
   increasing in x_env (sextile means 42.3, 40.8, 42.8, 47.6, 51.1, 58.1) with
   slope 47.4 (SE 2.4); the earlier non-monotone response was an artifact of
   the out-of-support range. Total is U-shaped in x_diff (sextile means 51.3,
   44.5, 46.3, 45.0, 45.5, 50.1; linear slope -0.02): blowout games score more.
   Handling: a measured full quadratic in (x_diff, x_env) for both margin and
   total (R2 total 0.141 linear to 0.198 quadratic), solved jointly by bounded
   least squares; no hand constant.
3. Re-smoke (fold 2025, same 20 games, design csv above). 150 draws, 2,400 design:
   margin error -1.21 (z -2.92), total +0.79 (z +2.45). 300 draws, 9,600 design:
   margin error -0.41 (SE 0.43, z -0.97, pass), total +0.95 (SE 0.31, z +3.10,
   FAIL), per-game rms z 2.65 and 2.14 (above 1: fit uncertainty and context
   mismatch add to MC). Larger design removed the margin bias; the total bias
   persisted, so it is not design noise. Unresolved candidate cause: the design
   pool draws (season, week) from the five training seasons while smoke games
   carry the held-out season's own contexts. The full run therefore draws each
   fold's design pool from the held-out fold's (season, week) contexts (schedule
   facts, no outcomes), and the smoke is repeated on that pool before the full
   run; a total bias above 2 z there blocks the run.
4. Draw count. Measured 0.117-0.136 s per anchored draw at one worker
   (smokes 2-4; 0.58 s in the contended first smoke is not used). Budget
   4 workers x 6 h = 86,400 worker-s. Design: 9,600 games x 0.15 s x 6 folds
   = 8,640 s. Remaining 77,760 s / (1,537 games x 0.136 s = 209 s per
   draw-per-game) = 372 draws per game. Declared N = 372 (not 1,000: 1,000 needs
   14.5 h at 4 workers). MC error at N = 372: cover probability SE
   sqrt(0.25/372) = 0.026, logit SE 0.104; margin-mean SE 17.2/sqrt(372) = 0.89
   points. Effect on the primary metric: MC noise in the sim logit adds about
   w x 0.5 x 0.104 / sqrt(1537) = 0.0013 w to the SE of the pooled delta, versus
   the bootstrap SE of about 0.0004, where w is the fitted blend weight on the
   sim logit. Noise in the regressor attenuates w toward zero and cannot
   create a positive delta; the verdict is reported with that attenuation, and
   a null at this N is `unresolved_below_power`.
5. Positive control. The deterministic plant (weight on the realised outcome)
   overstated power: every delta was detected in 100% of replicates. Replace it
   with a noisy plant: candidate logit c_i = recalibrated logit_i +
   kappa (2y_i - 1) + e_i, e_i drawn independently per game and replicate with
   sd equal to the measured sd of (sim logit - served logit) over the training
   folds of the full run plus the MC logit SE of 0.104 in quadrature; kappa is
   solved analytically (bisection) so the LOSO-fitted blend's expected
   log-loss gain equals the target delta. The candidate passes through the same
   LOSO blend fit and block bootstrap, 500 fresh-noise replicates per delta,
   deltas 0.0010 and 0.0080. The control runs after the full run's training-fold
   sd is known; a verdict is not issued before it.

### Amendment 2 (before any look), 2026-11-02

No graded outcome, blend, or L-level metric has been computed. Smoke games were
compared to opener lines and opener totals only.

1. Engine. The anchored arm runs the SIM-11 default engine (joint-label target,
   SIM10_JOINT=1, commit 2d38e381). The scripted engine setup had left the flag
   unset in smokes 1-4; it is now set in `engine()`.
2. Support on that engine (tests/scratch/sim08_anchor/support_joint.json, 40 worlds
   x 9 seasons, 92,160 games, same method as Amendment 1). x_diff 0.1%/1%/5%/25%/
   50%/75%/95%/99%/99.9% = -0.492/-0.380/-0.267/-0.109/0.001/0.109/0.267/0.376/
   0.502 (sd 0.162); x_env = -0.205/-0.153/-0.107/-0.041/0.003/0.047/0.111/0.155/
   0.205 (sd 0.066); corr 0.007. The latents are generated independently of the
   rollout engine, so the support is unchanged and the box stays +-0.5, +-0.21.
3. Transmission (9,600 design games, linear fit, SE in brackets): margin = 1.93 +
   37.5 (0.47) x_diff + 3.9 (1.1) x_env, R2 0.403; total = 45.7 + 2.5 (0.47) x_diff +
   47.1 (1.1) x_env, R2 0.162 linear, 0.172 quadratic. Margin and total are
   linear-monotone as before; the anchor still uses the full quadratic.
4. Design contexts. Each fold's design pool draws (season, week) from the held-out
   fold's own pregame contexts (schedule facts only; implemented as
   `--heldout-pool 1`). This makes Amendment 1 item 3's rule operative.
5. Smoke on that pool (fold 2025, same 20 games, 300 draws, 9,600 design):
   margin error -0.65 (z -1.56), total error +0.14 (z +0.38). Both within Monte
   Carlo error; the total bias (+0.95, z +3.10) is gone. The engine change and the
   pool change were made together, so their separate contributions are not
   separated (inferred cause: the training-season context pool).
6. Noisy planted control (item 5 of Amendment 1) is coded (`control --noisy 1`):
   kappa bisection on the LOSO two-column blend (recalibrated served logit,
   candidate), noise sd = sqrt(train_sd^2 + (2/sqrt(draws))^2). The execution run on
   the 20-game smoke population only shows it runs; it is not a verdict.
