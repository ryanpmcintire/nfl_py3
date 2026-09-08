# MOD-18 Lane U: deriving how much of the model's disagreement to trust

Lane R (`docs/residual_slope.md`) rescaled the model's raw residual by a
walk-forward per-bucket slope and lost through the played card. Lane U keeps
the mechanism and replaces the one number lane R admitted it had borrowed: the
shrinkage weight.

## Predeclaration (frozen before computation, 2026-09-08)

Everything in this section was written and saved BEFORE any candidate number
was computed. The replay stage records this file's SHA-256 at that moment
(`artifacts/research/laneU/reproduction.json`, field
`predeclaration_sha256`), so the measured section appended below cannot have
influenced it.

### What motivates the arms

Read: `docs/residual_slope.md`, section "Why it lost, and what to fit
differently". Lane R's slope was shrunk toward the incumbent weight 1.0 with
`beta_b = (n_b * beta_hat_b + 100) / (n_b + 100)`, the `PRIOR_WEIGHT_GAMES =
100` count prior the served S3 offset uses (read:
`src/nfl_ats/home_side_location.py:34`). That constant was derived for an
estimator of a MEAN and carries no information about how precisely a SLOPE is
estimated, so it mis-weights in both directions at once:

- Read (`docs/residual_slope.md`, same section, from
  `artifacts/research/laneR/slopes.parquet`): in `7.5-10` the walk-forward
  slope wandered from -1.89 (2020) to about 0 (2022-2024) to +0.91 at the
  Week 1 2026 fit, and lane R's served `beta` there sat between 0.41 and 0.67
  for most of the archive. Lane R paid **-3.608 accuracy points [-7.104,
  -0.500]** through the card in that bucket (read: `docs/residual_slope.md`,
  "By bucket").
- Read (same document, "Positive control"): in `0-3` the slope estimate is
  noise around zero, +0.0498 [-0.4919, +0.5911], and 504 prior games still
  dragged `beta` to 0.207 -- the largest bucket moved a long way off the
  incumbent on no evidence at all.

A count prior cannot tell those two cases apart, because it only sees `n`. The
slope's own standard error can: it is small exactly when the departure from
1.0 is well measured, and large when the estimate is noise. That is the whole
of the change. The owner's rule that an unjustified number gating an
irreversible decision is a defect (read: `AGENTS.md`, "Required
verification" preamble and the memory entry `underived-constants-are-wrong`)
applies to the 100 here exactly as it applied to `PRIOR_WEIGHT_GAMES`.

This remains a MECHANISM, not a threshold flip (read: `AGENTS.md`, "No
unexplained threshold flips on the played card"): the same estimator runs in
every bucket, is fitted only on strictly prior completed games, and reduces to
the incumbent wherever the data do not resolve a departure from it. R2b is the
exception and is disclosed as such below.

### R2 (the arm): empirical-Bayes shrinkage from the slope's own precision

For a target week (season `S`, week `W`) and lane-J spread bucket `b`:

1. Prior rows `P = prior_rows_before(archive_stream, S, W)` -- the SAME
   exclusions the served S3 offset and lane R use (read:
   `src/nfl_ats/home_side_location.py:239`): strictly earlier weeks, seasons
   at or after `S - 5`, rows with a recorded result and a valid raw point
   only.
2. `x_i` = the model's RAW out-of-time residual at that archived Tuesday
   opener; `y_i` = actual home margin minus that opener line. Identical to
   lane R (imported from `scripts/residual_slope_opener_eval.py`,
   `slope_inputs`).
3. `beta_hat_b` = ordinary least-squares slope of `y` on `x` WITH an
   intercept, identical to lane R (imported `ols_slope`).
4. NEW -- the slope's own standard error. Within-week correlation is ZERO by
   mandate (read: `AGENTS.md` / memory `within-week-correlation-is-zero`;
   never estimated, never padded), so the classical homoskedastic
   least-squares standard error is the declared estimator, with no clustering
   term:

   ```
   Sxx   = sum (x - xbar)^2
   RSS   = sum (y - a_hat - beta_hat * x)^2
   se_b^2 = (RSS / (n_b - 2)) / Sxx
   ```

   A bucket is ESTIMABLE when `n_b >= 4`, `Sxx > 0` and `se_b^2` is finite and
   strictly positive. A bucket that is not estimable serves the incumbent
   (`beta*_b := 1.0`) and contributes nothing to `tau^2`.
5. `tau^2`, the between-bucket variance of the true slopes AROUND THE
   SHRINKAGE TARGET 1.0, by the DerSimonian-Laird moment identity with the
   centre fixed at 1.0 rather than estimated. Declared choice, stated with its
   reason: standard DerSimonian-Laird centres `Q` on the precision-weighted
   mean and therefore estimates dispersion about that mean, which is not the
   prior variance a shrinkage toward the INCUMBENT needs; fixing the centre at
   the shrinkage target is the same moment estimator with `K` degrees of
   freedom instead of `K - 1`. Over the `K` estimable buckets, with
   `w_b = 1 / se_b^2`:

   ```
   Q      = sum_b w_b * (beta_hat_b - 1)^2
   tau^2  = max(0, (Q - K) / sum_b w_b)
   ```

   because under `beta_hat_b ~ N(1 + u_b, se_b^2)` with `u_b ~ N(0, tau^2)`,
   `E[Q] = K + tau^2 * sum_b w_b`.
6. The served weight:

   ```
   beta*_b = 1 + (beta_hat_b - 1) * tau^2 / (tau^2 + se_b^2)
   ```

   A noisy bucket (`se_b^2` large relative to `tau^2`) stays near 1; only a
   precisely-estimated departure moves. `tau^2 = 0` -- the buckets do not
   resolve any departure from the incumbent -- gives `beta*_b = 1` in EVERY
   bucket, which is EXACTLY S3, not an approximation of it. Pinned by
   `tests/test_residual_slope_shrunk.py`.
7. Served point for a game in bucket `b`:

   ```
   point_R2 = line + beta*_b * residual_raw + offset_b
   ```

   with `offset_b` the UNCHANGED S3 served offset (`fit_home_side_offsets`,
   zero outside `7` / `7.5-10` / `10.5+`). The offset is NOT refitted under the
   new weight, so any measured difference is attributable to the slope alone.

The centre shift handed to `MarginModel.predict` is lane R's, unchanged and
imported rather than re-implemented (read: `docs/residual_slope.md`, "The
centre shift", and `src/nfl_ats/margin.py:995-1000`):

```
center_offset_R2 = (beta*_b - 1) * residual_raw + offset_b
```

A row whose line has no resolvable bucket keeps `beta = 1`, i.e. the
incumbent.

### R2b (the sibling): the rescale served only at 10.5+

Lane R's ORIGINAL 100-game shrinkage,
`beta_b = (n_b * beta_hat_b + 100) / (n_b + 100)`, applied ONLY in the `10.5+`
bucket. Every other bucket serves `beta = 1`, i.e. exactly S3. This is lane R's
predeclared follow-up 2 (read: `docs/residual_slope.md`, "Two concrete,
predeclarable follow-ups", item 2), carried over verbatim so that R2 and R2b
differ in exactly one thing each from lane R's R1: R2 changes the shrinkage and
keeps every bucket, R2b keeps the shrinkage and changes the buckets.

**Disclosed, in full, before the number is seen:** the `10.5+` restriction is
BUCKET SELECTION AFTER SEEING THE SPLIT, on the same mined archive that
produced the split. Lane R measured `10.5+` at +2.290 [-3.200, +8.276] P+ 0.745
standalone and +7.407 [-5.660, +21.569] P+ 0.834 on its road picks; those are
the numbers that chose this bucket. R2b's reading is therefore a correlated
post-hoc sensitivity, not independent confirmation, and it may never be quoted
as though the bucket had been named in advance.

### Positive control (predeclared)

Unchanged from lane R, so the instrument is the same one: the walk-forward
`beta_hat` for the `3.5-6.5` bucket must stay clearly positive, reported with
a whole-week block bootstrap interval on the prior rows used for the Week 1
2026 fit (imported `slope_intervals`). If it does not, the instrument cannot
see slopes at all and no arm's reading is interpretable; that would be stated
plainly rather than treated as evidence about the mechanism. The lane also
reports `se_b`, `w_b`, `tau^2` and the shrinkage fraction
`tau^2 / (tau^2 + se_b^2)` per bucket, so the derived weight is auditable
against the borrowed one.

### How it is graded

Identical to lane R in every respect except the two arms:

1. Replay S3 EXACTLY first on the active-matched opener archive. Stop before
   any candidate scoring if any served probability differs by more than 1e-9.
2. Paired accuracy at the OPENER (read: `AGENTS.md`, "Grade the decision at
   the OPENER"), non-push games only.
3. Whole-week block bootstrap, 20,000 draws, seed 20260817, within-week
   correlation ZERO -- never estimated, never padded.
4. Standalone (model alone) AND through the played three-member card, via
   `overlay-composition --per-game-artifact` on research `per_game` artifacts
   carrying `research_arm` and `active_model_id` `research_laneU_R2` /
   `research_laneU_R2b`, never the active identity.
5. Reported per season, per bucket, by pick side (home picks / road picks) and
   per bucket x pick side. The pick-side grouping uses the INCUMBENT's
   standalone opener pick, so the cells do not move with the arm.
6. Brier and log loss beside accuracy.
7. The fitted Week 1 2026 weights per bucket with intervals under each arm,
   and the Week 1 2026 sides that would change, read from the linked
   forecast's `home_side_offset.json` sidecar and `predictions.csv`. No
   forecast is regenerated and no pick is changed by this lane.

Family `mod18_home_side_location_v1`; cell names
`mod18_home_side_location_v1_r2_*`; classification `unresolved_below_power`
unless an admissible closing ground applies. The lane never runs
`nfl-ats weak-signals record`: the commands are WRITTEN to
`artifacts/research/laneU/record_commands.ps1` for the coordinator to run
serially.

### Disclosure

This is a POST-HOC refit of a POST-HOC mechanism, on the same mined 1,537-game
archive that S2, S3 and lane R were all scored on. The defect it repairs was
diagnosed on that archive, the `10.5+` bucket R2b restricts to was chosen on
that archive, and the arms are strongly correlated with lane R's R1 and R1b by
construction. Walk-forward fitting removes look-ahead inside each game; it does
not undo archive-level selection. The reading here is a correlated sensitivity
on a reused window, carrying a stated discount, not independent confirmation.

### Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Never state that anything "needs more games"; decide
on expected value (P+ above 0.5 favours playing), and state what the numbers
imply for the DECISION (play R2 or R2b instead of S3?) before what is wrong
with them.

### Decision rule, declared before the numbers

`probability_positive` above 0.5 through the played card favours PLAYING the
arm instead of S3; the pool submits 285 forced cards either way, so declining
a candidate that is more likely than not better is taking the other side of
that bet. R2 and R2b are graded side by side and neither is selected on its
own result; where they disagree, the arm whose bucket set was declared in
advance (R2) is the one the disclosure supports.

## Measured results (appended after computation, 2026-09-08)

The digest stamped at replay time,
`4cecf05b8b6ca43dea025932e3d7b60aca29b5da0fdf917acf71ec2bc9dfa5e6`, is the
SHA-256 of this file with the predeclaration section ONLY; appending this
section changes the file's digest, which is why the frozen value is recorded in
`artifacts/research/laneU/reproduction.json` (`predeclaration_sha256`) rather
than re-derived here.

### The decision first

**Keep S3. Do not play R2 or R2b on the Week 1 card.** Measured through the
played three-member card, 1,503 non-push opener games across 107 weeks:

- **R2: -0.732 accuracy points, 95% [-1.941, +0.464], `probability_positive`
  0.104** (55.156% against S3's 55.888%, 73 composed flips)
- **R2b: +0.000 [-0.404, +0.403], P+ 0.434** (55.888%, 10 composed flips) --
  a coin flip that touches ten picks in six seasons

Playing R2 is taking the 10/90 side of that bet. R2b is barely a bet at all
through the card: it moves ten picks and nets exactly zero.

**The one reading that does favour an arm** is R2b on the model alone, before
the overlays: **+0.200 [-0.327, +0.731], P+ 0.736** (54.757% against 54.558%,
13 flips). The card is the grade the predeclaration named as the decision
grade, and the card says 0.434, so S3 stays; but the standalone read is the
better-than-even side and it is recorded as such, not discarded.

### The finding: lane R's diagnosis is NOT what was wrong

Lane R concluded that its arm lost because the 100-game count prior was "far
too weak for a SLOPE". This lane derived the weight from the slope's own
precision instead and **the card result did not move**: R1 scored -0.732
[-1.932, +0.404] P+ 0.099, R2 scores -0.732 [-1.941, +0.464] P+ 0.104. That is
not a near miss on the same estimator -- the served weights genuinely changed
(season means of the served `beta`, `artifacts/research/laneU/slopes.parquet`
against `artifacts/research/laneR/slopes.parquet`):

| bucket | R1 beta 2020 | R1 beta 2025 | R2 beta 2020 | R2 beta 2025 | R2 shrinkage fraction 2020 | R2 shrinkage fraction 2025 |
|---|---:|---:|---:|---:|---:|---:|
| 0-3 | 0.852 | 0.404 | **0.718** | 0.396 | 0.232 | 0.851 |
| 3.5-6.5 | 0.860 | 0.639 | **0.826** | 0.624 | 0.311 | 0.869 |
| 7 | 0.959 | 0.788 | 0.927 | 0.754 | 0.039 | 0.462 |
| 7.5-10 | 0.668 | 0.650 | **0.219** | 0.636 | 0.196 | 0.664 |
| 10.5+ | 1.040 | -0.059 | 0.992 | -0.121 | 0.024 | 0.572 |

The shrinkage fraction does exactly what it was designed to do -- it serves
only 2-31% of the estimated departure from the incumbent on 2020's thin
windows and 46-87% of it by 2025, where the same slopes are measured five
times as precisely -- and the forced pick is no better for it. Note the one
place the derived weight is MORE aggressive than the borrowed one: `7.5-10` in
2020, mean served `beta` 0.219 against R1's 0.668, because that window's raw
slope was about -1.9 and even a fifth of a departure that large is a long way
from 1. **Inferred:** the defect lane R located was real but not decisive; the
arm's loss is a bucket-set problem, not a weight-calibration problem.

Where the loss sits is unchanged and now measured under two independently
derived weightings:

| cell | n | R1 card (lane R) | R2 card (lane U) |
|---|---:|---|---|
| 7.5-10 | 194 | -3.608 [-7.104, -0.500] P+ 0.007 | **-4.124 [-8.287, -0.495] P+ 0.011** |
| 7.5-10, picked the road team | 112 | -5.357 [-10.680, -0.862] P+ 0.005 | **-8.036 [-15.741, -1.639] P+ 0.005** |
| 10.5+ | 131 | +0.000 [-4.688, +4.878] P+ 0.447 | +0.763 [-3.731, +5.385] P+ 0.574 |
| 10.5+, picked the road team | 54 | +1.852 [-9.091, +13.208] P+ 0.568 | +1.852 [-9.091, +13.208] P+ 0.568 |

`7.5-10` is the bucket whose slope is closest to the incumbent on the full
prior window (+0.9133 [+0.0200, +1.8041], P(slope > 0) 0.977) -- the residual
there is worth very nearly its face value -- and every weighting that
walk-forwards through the thin early seasons discounts it anyway and pays for
it. That is the mechanism's boundary, stated as a boundary rather than as a
threshold: rescale where the slope is resolved away from 1, leave it alone
where it is resolved near 1.

### On the seven cells whose interval excludes zero

Three cells sit entirely below zero (`7.5-10` card, widening factor needed to
re-cross zero 1.136x; `7.5-10` road picks card 1.256x and standalone 1.206x),
two more marginally so (R2b's 2020 Brier and log loss, 1.045x and 1.040x); two
cells sit entirely above zero (R2b's 2022 Brier and log loss). All 192 cells
are nevertheless recorded `unresolved_below_power` and **no closing ground is
claimed**, for a reason that is a multiplicity argument rather than the banned
"contains zero" one: seven of 192 correlated cells excluding zero is *fewer*
than the ~10 chance alone would produce, and the two arms scored here share
their archive, their games and most of their picks with lane R's, so the
apparent replication of the `7.5-10` cell is not independent. The closure is
mechanically admissible (whole interval below zero, widening above the 1.099x
honest-refit bound `nfl_ats.experiment_runner` uses) and remains available to a
human adjudication; this lane does not take it on its own authority, and lane
R's identical cells are already in the registry as unresolved.

### Positive control (predeclared): PASSES, unchanged from lane R

Walk-forward slopes on the 1,310 prior archive rows used for the Week 1 2026
fit, whole weeks resampled, 20,000 draws, seed 20260817
(`artifacts/research/laneU/week1.json`, `positive_control`). The `3.5-6.5`
control reads **+0.5653 [-0.1309, +1.2193], P(slope > 0) 0.944**, so the
instrument can see a slope where one exists and the negative overall reading is
a reading of the ARM. `10.5+` reads -1.1745 [-2.2255, -0.1138] on prior-only
rows, reproducing lane L's headline out of time.

### The Week 1 2026 weights, with their intervals

`tau^2 = 0.6359` at the Week 1 2026 fit. `beta_hat` and its interval are the
positive-control bootstrap above; `se` is the classical least-squares standard
error the shrinkage is derived from.

| bucket | prior games | beta_hat | 95% | se | shrinkage fraction | R2 beta | R2b beta | R1 beta (lane R) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0-3 | 504 | +0.0498 | [-0.4919, +0.5911] | 0.2984 | 0.877 | **+0.166** | 1.000 | +0.207 |
| 3.5-6.5 | 468 | +0.5653 | [-0.1309, +1.2193] | 0.2935 | 0.881 | **+0.617** | 1.000 | +0.642 |
| 7 | 65 | +0.4413 | [-1.3763, +2.2933] | 0.8634 | 0.460 | **+0.743** | 1.000 | +0.780 |
| 7.5-10 | 160 | +0.9133 | [+0.0200, +1.8041] | 0.5168 | 0.704 | **+0.939** | 1.000 | +0.947 |
| 10.5+ | 113 | -1.1745 | [-2.2255, -0.1138] | 0.6405 | 0.608 | **-0.322** | **-0.154** | -0.154 |

With five to six seasons of prior rows the derived weight and the borrowed one
have nearly converged (0.166 against 0.207, 0.617 against 0.642, 0.939 against
0.947); they differ most in the early archive, which is where the arm's history
was written.

### Decision table (S3 versus R2 versus R2b)

1,503 non-push opener games, 107 weeks, week-blocked bootstrap 20,000 draws,
seed 20260817, within-week correlation zero.

| read | S3 | R2 | R2 delta, 95%, P+ | R2b | R2b delta, 95%, P+ |
|---|---:|---:|---|---:|---|
| model alone, opener | 54.558% | 53.693% | -0.865 [-2.236, +0.472] P+ 0.096 (95 flips) | **54.757%** | **+0.200 [-0.327, +0.731] P+ 0.736** (13 flips) |
| THROUGH the played card | **55.888%** | 55.156% | **-0.732 [-1.941, +0.464] P+ 0.104** (73) | **55.888%** | **+0.000 [-0.404, +0.403] P+ 0.434** (10) |
| Brier (improvement) | 0.251584 | 0.251103 | +0.000481 [-0.000886, +0.001839] P+ 0.756 | 0.251204 | +0.000380 [-0.000212, +0.001023] P+ 0.892 |
| log loss (improvement) | 0.696624 | 0.695585 | +0.001039 [-0.001774, +0.003832] P+ 0.767 | 0.695848 | +0.000776 [-0.000438, +0.002094] P+ 0.891 |
| Week 1 2026 sides changed | reference | 3 of 16 | ARI at LAC, NE at SEA, NYJ at TEN | 1 of 16 | ARI at LAC |

Both arms make the stated chances better while R2 makes the forced pick worse,
the same pattern lane R measured: shrinking the residual pulls the probability
toward 50%, which is right for a model stating about 56% and right about 54%,
and near the threshold it also flips picks.

### By pick side (where lane L located the hole)

Cells cut by the INCUMBENT's standalone opener pick, so the grouping does not
move with the arm.

| cell | n | R2 standalone | R2 card | R2b standalone | R2b card |
|---|---:|---|---|---|---|
| picked the home team | 651 | -0.307 [-1.813, +1.173] P+ 0.312 | -0.154 [-1.601, +1.311] P+ 0.376 | -0.154 [-0.496, +0.000] P+ 0.000 | -0.154 [-0.496, +0.000] P+ 0.000 |
| picked the road team | 852 | -1.291 [-3.515, +0.832] P+ 0.110 | -1.174 [-3.055, +0.613] P+ 0.090 | **+0.469 [-0.364, +1.351] P+ 0.827** | +0.117 [-0.590, +0.813] P+ 0.566 |
| 10.5+, road picks | 54 | +7.407 [-5.660, +21.569] P+ 0.834 | +1.852 [-9.091, +13.208] P+ 0.568 | +7.407 [-5.660, +21.569] P+ 0.834 | +1.852 [-9.091, +13.208] P+ 0.568 |
| 10.5+, home picks | 77 | +0.000 P+ 0.000 | +0.000 P+ 0.000 | -1.299 [-4.110, +0.000] P+ 0.000 | -1.299 [-4.110, +0.000] P+ 0.000 |
| 7.5-10, road picks | 112 | -8.929 [-17.476, -1.527] P+ 0.007 | -8.036 [-15.741, -1.639] P+ 0.005 | +0.000 | +0.000 |
| 7.5-10, home picks | 82 | +2.439 [-2.353, +7.692] P+ 0.774 | +1.220 [-2.632, +5.682] P+ 0.602 | +0.000 | +0.000 |

R2b's whole standalone gain is its road picks (+0.469 P+ 0.827), and its whole
standalone loss is one home pick at 10.5+ (-0.154 P+ 0.000, a single game). The
mechanism lane L described -- the model fading a big home favourite on its own
ratings -- is where every point of it lives.

### By bucket

| bucket | n | S3 right (card) | R2 standalone | R2 card | R2b standalone | R2b card |
|---|---:|---:|---|---|---|---|
| 0-3 | 552 | 57.43% | -1.268 [-3.172, +0.691] P+ 0.082 | -0.362 [-2.026, +1.301] P+ 0.289 | +0.000 | +0.000 |
| 3.5-6.5 | 552 | 57.79% | -0.906 [-2.749, +0.914] P+ 0.138 | -0.362 [-2.120, +1.357] P+ 0.298 | +0.000 | +0.000 |
| 7 | 74 | 50.00% | +4.054 [+0.000, +9.091] P+ 0.952 | +0.000 P+ 0.000 | +0.000 | +0.000 |
| 7.5-10 | 194 | 51.55% | -4.124 [-8.947, +0.488] P+ 0.027 | **-4.124 [-8.287, -0.495] P+ 0.011** | +0.000 | +0.000 |
| 10.5+ | 131 | 51.15% | **+3.053 [-2.326, +8.871] P+ 0.834** | +0.763 [-3.731, +5.385] P+ 0.574 | +2.290 [-3.200, +8.276] P+ 0.745 | +0.000 [-4.688, +4.878] P+ 0.447 |

R2's `10.5+` cell is the best card reading either lane has produced for the
mechanism (+0.763, P+ 0.574) -- the derived weight is more aggressive there in
the late seasons (-0.32 against R2b's -0.15) and it is paid for.

### By season

| season | n | R2 standalone | R2 card | R2b standalone | R2b card |
|---|---:|---|---|---|---|
| 2020 | 220 | -0.455 [-3.017, +2.315] P+ 0.295 | -0.455 [-3.017, +2.315] P+ 0.295 | +0.000 | +0.000 |
| 2021 | 236 | +1.695 [+0.000, +3.448] P+ 0.942 | +1.271 [-0.426, +2.991] P+ 0.884 | -0.424 [-1.293, +0.000] P+ 0.000 | -0.424 [-1.293, +0.000] P+ 0.000 |
| 2022 | 248 | -2.016 [-4.783, +0.784] P+ 0.054 | -1.210 [-4.065, +1.594] P+ 0.168 | +0.403 [-0.844, +1.709] P+ 0.615 | +0.403 [-0.844, +1.709] P+ 0.615 |
| 2023 | 266 | -1.128 [-4.981, +2.602] P+ 0.250 | -1.504 [-4.444, +1.154] P+ 0.120 | +1.504 [-0.375, +3.637] P+ 0.915 | +0.376 [-0.769, +1.544] P+ 0.615 |
| 2024 | 266 | -1.128 [-4.943, +2.632] P+ 0.251 | -0.752 [-4.511, +2.682] P+ 0.313 | +0.376 [+0.000, +1.136] P+ 0.644 | +0.376 [+0.000, +1.136] P+ 0.644 |
| 2025 | 267 | -1.873 [-5.703, +1.887] P+ 0.153 | -1.498 [-4.461, +1.533] P+ 0.140 | -0.749 [-1.859, +0.000] P+ 0.000 | -0.749 [-1.859, +0.000] P+ 0.000 |

R2b is positive in three of six seasons and negative in two, on one to four
touched picks a season. Nothing in that pattern is a season effect; it is the
same handful of big-line games arriving in different years.

### Week 1 2026 (read only, no forecast regenerated, no pick changed)

The linked card `margin_predictions/2026-week-01-20260908T124514Z` replays from
its own sidecar to 9.99e-16.

| game | line | raw residual | S3 home cover | R2 (beta) | R2b (beta) |
|---|---:|---:|---:|---:|---:|
| ARI at LAC | 10.5 | -5.807 | 0.3903 | **0.6287** (-0.322) | **0.5992** (-0.154) |
| NE at SEA | 3.5 | -0.445 | 0.4976 | 0.5030 (+0.617) | 0.4976 (1.000) |
| NYJ at TEN | 1.5 | -0.867 | 0.4843 | 0.5071 (+0.166) | 0.4843 (1.000) |

R2 would change three of sixteen sides, R2b exactly one: ARI at LAC, where the
ridge disagrees with a 10.5 line by 5.8 points and the arms discount that
disagreement to nothing or reverse it. Neither is served; S3's card stands.

### Replay

Measured (`artifacts/research/laneU/reproduction.json`): archive
`artifacts/opener_evaluation/20260908T115957Z` (the active model
`a4c757efd2525da6`'s matched evaluation, 1,537 games), feature digest
`457aafb7...` equal to the active manifest. S3's served probabilities replay to
a maximum gap of **3.77e-15**, the served offsets to 6.66e-16 and the raw
residual to 1.40e-13 -- all far inside the 1e-9 stop rule, so candidate scoring
ran.

### What to fit next (inferred, predeclarable, not measured here)

1. The hybrid neither arm is: the empirical-Bayes weight served ONLY in
   `10.5+`. R2's `10.5+` card cell is the best of the four bucket readings
   (+0.763 P+ 0.574) and R2b shows that restricting the bucket set removes the
   `7.5-10` bleed entirely. That combination has never been graded as an arm and
   must be predeclared as one, with the same post-hoc bucket disclosure R2b
   carries; it may not be quoted from this table.
2. A resolved-slope gate rather than a bucket list: serve the rescale only
   where the walk-forward slope's interval excludes 1.0, which is the
   mechanism's own boundary and needs no bucket named in advance. On the Week 1
   2026 window that gate would fire in `10.5+` alone.

### Provenance and scope

Measured this session by
`.\.tools\uv.exe run --no-sync python scripts\residual_slope_shrunk_opener_eval.py
--stage replay|score|week1|record`. Artifacts (all provenance-stamped) under
`artifacts/research/laneU/`: `replay.parquet`, `slopes.parquet`,
`scored.parquet`, `cells.json`, `week1.parquet`, `week1.json`,
`reproduction.json`, `record_commands.ps1`, the research
`opener_evaluation/R2` and `opener_evaluation/R2b` per-game artifacts (model
identities `research_laneU_R2` / `research_laneU_R2b`, never the active
identity) with their `overlay_subset_composition` runs, and `commands.json`
with the exact successful argv. The Week 1 2026 card was READ only.

192 `weak-signals record` commands (96 cells x 2 arms) are written to
`artifacts/research/laneU/record_commands.ps1` for the coordinator to run
serially -- this lane never touches the shared registry, and
`NFL_ATS_REGISTRY_DIR` pointed at `artifacts/research/laneU/registry` for every
stage. Every row is `unresolved_below_power`, carries the disclosure above in
`--notes` and a pool-player `--plain-summary`, and all 192 were verified to
parse against the real CLI parser (`nfl_ats.cli.build_parser().parse_args`,
parse only, no handler, nothing written) with 192 unique names and no collision
against the 3,774 signals already in `registry/weak_signals.json`.
