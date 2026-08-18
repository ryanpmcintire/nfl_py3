# Does the calibration step distort small effects? (defect D1)

Owner: this document, `src/nfl_ats/calibration_distortion.py`,
`tests/test_calibration_distortion.py`, and
`scripts/calibration_distortion_screen.py`.

`docs/revisit_list.md` names five defects and says one of them gates the whole
list: **D1 — the calibration step distorts small effects.** If D1 is real at
the 1-point scale, every terminal negative in this project is suspect and the
Tier 1 re-run list is understated. If it is not, Tier 1 shrinks to the D4
degenerate-bootstrap case plus the two entries that recorded no continuous
evidence at all. This document runs that experiment.

**Why a new document rather than an extension of `docs/purged_cv.md`:** the
thing under test is not the purged-CV scheme. It is
`fit_margin_model`/`fit_cfb_residual_model`'s out-of-time residual sample,
which the production NFL walk-forward evaluator uses on every scored week and
every published card. Purged CV is only the instrument this document measures
it with. Filing the result inside the purged-CV document would attach a
pipeline-wide finding to one supplementary evaluator. `docs/purged_cv.md` §3
and §5 now carry a pointer here instead.

Scope: every empirical number below is CFB (rotation-registry rule 8 — "CFB
and non-reserved seasons stay free", `docs/rotation_registry.md`). No NFL
rotation window was assigned or spent, no `registry/*.json` was written, no
model, feature profile, or pick moved.

## 1. What the calibration step actually does to a pick

`MarginModel.predict` (`src/nfl_ats/margin.py:466-521`) builds
`center + residuals` for each game and reads
`home_cover_probability = _smoothed_probability(center + residuals, line)`,
the continuity-corrected `(count(... > line) + 0.5) / (n + 1)`. The forced
pick is `home_cover_probability >= 0.5`. Substituting
`predicted_market_residual = center - line`, that inequality is exactly

```
count(residuals > -predicted_market_residual) >= n / 2
```

which is a plain **threshold on the predicted residual** at
`-residuals_sorted[n - ceil(n/2)]` — one near-median order statistic of the
residual sample. So the production pick and the sign-only ablation differ by
exactly **one number per fitted model: a location offset.**

That is not a re-derivation on paper.
`calibration_distortion.full_pipeline_home_pick` implements the inequality and
`tests/test_calibration_distortion.py::test_full_pipeline_pick_matches_margin_model_predict`
pins it against a real `MarginModel.predict` call; a second test pins
`implied_pick_threshold` by sweeping a 2,001-point grid of predicted residuals
at seven sample sizes. *(measured: `pytest tests/test_calibration_distortion.py`,
20 passed.)*

Two consequences worth stating before any number:

- The location is **not free noise added on top of a good pick**. On real data
  it is a crude recency correction that the deployed ridge — refit on the
  whole expanding history with an unweighted intercept — does not carry, and
  it is measurably worth **+2.12 accuracy points** on the NFL backtest
  (`docs/residual_location.md` §1, `probability_positive` 0.990). *(read, not
  re-measured here; the CFB analogue IS measured below in §6.)*
- A construction that destroys the real target also destroys the drift the
  location exists to correct. In that world the location can only be noise.
  This is the crux of §5.

## 2. What the recorded D1 evidence is — and what it is not

`docs/purged_cv.md`'s positive control plants a `{-1,+1}` column with a known
population forced-pick accuracy, **overwriting** `ats_margin` with
`beta * signal + Normal(0, 15.4)`, and scores it through purged CV at
`n_blocks=40`. Recorded: planted 51.3% came back **49.33%** (wrong sign) with
sign-only at 50.67%; planted 53.0% came back 51.14% with sign-only at 53.01%.

Two properties of that evidence decide how much it can carry:

1. **It is one replicate.** `scripts/purged_validate.py:150-181` calls
   `inject_synthetic_signal(df, target_accuracy=..., seed=42)` once per
   magnitude. There is no replication, so the readout has no measured
   dispersion attached to it. *(read: `scripts/purged_validate.py:153`.)*
2. **It is a LEVEL, not a paired delta.** It asks "what accuracy does the arm
   carrying the plant score", against a population constant. Almost every
   verdict in `registry/weak_signals.json` and
   `registry/rotation_registry.json` is instead a paired delta between two
   arms on the same games, same folds, both passing through their own
   calibration step.

Both were reproduced before anything new was built. Planting in the raw
parquet row order the recorded script uses, at seed 42, `n_blocks=40`:

| Target | Population truth | Full pipeline | Sign-only | Recorded |
|---|---|---|---|---|
| 51.3% | 51.672% | **49.328%** | 50.672% | 49.33% / 50.67% |

*(measured this session, scratch script over `fit_cfb_residual_model` on the
same 40 purged folds; matches the recorded figures to the reported digits.)*
The recorded control is reproducible. The question is what it means.

## 3. Design

Everything is scored on the same 12,500-game CFB table
(`data/processed/cfb_game_features.parquet`) with the frozen
`fit_cfb_residual_model` recipe (ridge alpha 10, `distribution_fraction` 0.20,
`min_train_games` 500) on the same 40 purged, embargoed folds
(`purge_weeks=12`, `embargo_weeks=7`, both derived, not typed).

Within a fold, **both estimators read the identical fitted model** — the full
pipeline applies the residual location, the sign-only ablation does not. That
pairing is what makes the difference between them an estimate of the
calibration step alone, with the fit's own noise cancelled.

Four stages (`scripts/calibration_distortion_screen.py`):

| Stage | Construction | What it isolates |
|---|---|---|
| `overwrite` | the recorded plant, replicated 8x per magnitude across {50.5, 51.0, 51.3, 52.0, 53.0}% | is the recorded readout stable? level vs paired delta |
| `additive` | `plant_additive_effect`: `ats_margin + gamma * carrier`, real target preserved | both arms carry real signal, as real verdicts do |
| `real` | 17 real mean-model contrasts, no plant (14 ordinary, 3 deliberately extreme so the regression has range) | do the two estimators agree on real effects? |
| `mechanism` | sweeps `distribution_fraction`, design width, `n_blocks` | is any distortion found fixable? |

The `additive` stage runs the plant through an orthogonal `{-1,+1}` carrier
**and** through five REAL feature columns spanning R² 0.026 to 0.868 against
the rest of the standardised design (`rest_diff`, `week_cos`,
`conference_game`, `week_sin`, `home_team_games`; measured on the full table).
The team-state block is excluded on purpose: every one of its 24 columns sits
above **R² 0.997** against the others (`diff_x` is `home_x - away_x` by
construction), so dropping one from a baseline arm removes no information and
the contrast is degenerate.

Reader-only families (`residual_location`, `ecdf_smoothing`) are **absent from
the `real` stage on purpose.** They hold the mean model fixed and change only
how a fixed residual sample is read, so the sign-only arm is bit-identical on
both sides and the cross-check carries exactly zero information about them.
That is a real limitation, recorded in §8, not an oversight.

Intervals in the planted stages come from the spread **across replicates**
(independent plant draws on the same corpus, the replication unit that matters
for "would a fresh plant of this size read the same way"), reported as a mean,
a 95% Student-t interval, and `probability_positive`. Intervals in the `real`
stage come from the project's own `experiments.paired_feature_comparisons`,
week-blocked, `samples=20000`.

## 4. The recorded control, replicated: the estimator gap is under 0.35 points, not 2

Same construction, same folds, same magnitudes — 8 independent plant draws per
magnitude instead of one. Truth is each replicate's own **realized**
population accuracy (the plant's realized edge varies with sd 0.45 points
across draws, measured over 200 seeds), so every ratio below is against the
edge that was actually planted, not the nominal target.

`artifacts/calibration_distortion/20260818T154856Z/overwrite_summary.csv`,
full 35-column design, points above 50%:

| Planted | Truth | Level, full pipeline | Level, sign-only | Paired delta, full | Paired delta, sign-only |
|---|---|---|---|---|---|
| 50.5% | +0.62 | −0.40 [−1.04, +0.25] | −0.39 [−0.96, +0.18] | +0.15 [−0.11, +0.40] | +0.16 [−0.08, +0.40] |
| 51.0% | +1.16 | +0.68 [+0.12, +1.24] | +0.75 [−0.04, +1.54] | +0.62 [+0.01, +1.24] | +0.64 [−0.00, +1.28] |
| 51.3% | +1.80 | +0.96 [+0.51, +1.41] | +0.95 [+0.42, +1.48] | +1.07 [+0.53, +1.62] | +0.85 [+0.06, +1.65] |
| 52.0% | +2.14 | +1.20 [+0.56, +1.84] | +1.42 [+0.93, +1.90] | +1.42 [+0.92, +1.93] | +1.58 [+1.06, +2.11] |
| 53.0% | +3.06 | +2.65 [+2.27, +3.03] | +2.66 [+2.32, +3.00] | +2.53 [+2.04, +3.02] | +2.32 [+1.58, +3.05] |

**The paired within-replicate gap — the statistic that isolates the
calibration step, because both estimators read the identical fitted model:**

| Planted | Level gap (full − sign) | `P+` | Delta gap (full − sign) | `P+` |
|---|---|---|---|---|
| 50.5% | −0.00 [−0.32, +0.32] | 0.494 | −0.02 [−0.40, +0.37] | 0.463 |
| 51.0% | −0.07 [−0.43, +0.28] | 0.320 | −0.01 [−0.44, +0.41] | 0.470 |
| 51.3% | +0.01 [−0.25, +0.27] | 0.542 | +0.22 [−0.37, +0.81] | 0.795 |
| 52.0% | −0.22 [−0.58, +0.15] | 0.105 | −0.16 [−0.55, +0.23] | 0.179 |
| 53.0% | −0.01 [−0.10, +0.07] | 0.358 | +0.21 [−0.11, +0.54] | 0.917 |
| **Pooled, 40 replicates** | **−0.059 [−0.169, +0.052]** | **0.146** | **+0.049 [−0.115, +0.213]** | **0.724** |

**This is the number that decides D1.** Across 40 planted replicates spanning
0.5 to 3.5 true accuracy points, dropping the calibration step moves the
measured LEVEL by **−0.059 points, 95% [−0.169, +0.052]** and the measured
PAIRED DELTA by **+0.049 points, 95% [−0.115, +0.213]**. The recorded D1
evidence is a 2.0-point swing (planted +1.67, read −0.67). A 2-point
calibration effect is excluded by these intervals by more than an order of
magnitude, on the construction that produced the recorded number.

### Why the recorded reading looked so different

Seed 42 is a tail draw of a noisy readout, not a property of the pipeline.
At planted 51.3% the single-replicate LEVEL has a replicate-to-replicate sd of
**0.54–0.66 points**, and the single-replicate level GAP has sd **0.31 points**
across the eight draws here and **0.42 points** across the 21-seed sweep in
§4a. Seed 42's gap of **−1.34 points** is the extreme of everything measured.
*(measured: `overwrite_raw.csv` and `<scratchpad>/seed_spread_raw.csv`.)*

A one-draw readout with that dispersion cannot distinguish a 2-point
distortion from zero, which is the methodological lesson independent of the
verdict: **the positive control needs replication, and a single-seed positive
control should not be cited as a measurement.**

### There IS attenuation, and it is not the calibration step's doing

Regressing each replicate's recovered effect on its own realized planted
truth, through the origin, across all 40 replicates:

| Framing | Recovery slope | 95% |
|---|---|---|
| Level, full pipeline | **0.696** | [0.594, 0.797] |
| Level, sign-only | 0.732 | [0.636, 0.828] |
| Paired delta, full pipeline | 0.720 | [0.647, 0.793] |
| Paired delta, sign-only | 0.693 | [0.600, 0.786] |

All four intervals overlap. **In this construction a planted POPULATION edge
comes back at roughly 70% of its size — under every framing and under both
estimators alike.** That is ridge estimation loss (a small coefficient
estimated from noisy data alongside 34 nuisance covariates), not a calibration
artifact: an ablation that removes the calibration step entirely recovers the
same 70%. Note the reference: this 70% is against the plant's own *population*
accuracy, which no fitted model can reach. Against the *achievable* delta —
what a perfect estimator of the planted coefficient would produce on these
games — recovery is 96%, measured in §5.

The decision-relevant property of attenuation is its DIRECTION. Shrinking a
measured effect toward zero cannot flip its sign; only a bias can, and the
measured bias is the 0.05-point gap above. So attenuation makes recorded
effects **understatements of their own magnitude** — a recorded −0.67 is
probably a true −0.9 or so, and a recorded +0.5 a true +0.7 — and it never
manufactures a negative out of a positive.

### 4a. Why seed 42 read the way it did, and why it read that way twice

The recorded control reports two magnitudes at the same seed. They are not two
pieces of evidence. `inject_synthetic_signal` draws `signal = rng.choice(...)`
and then `noise = rng.normal(...)` from a generator seeded only by `seed`
(`src/nfl_ats/purged_cv.py:651-655`), so `target_accuracy=0.513` and
`target_accuracy=0.53` **at seed 42 share the identical signal vector and the
identical noise vector**; only `beta` differs. One unlucky draw was counted
twice.

Re-running the recorded construction exactly (raw parquet row order,
`n_blocks=40`) across seed 42 plus 20 further seeds:

| Target | Level gap (full − sign), 21 seeds | Delta gap, 21 seeds | seed 42's level gap | z vs the other 20 |
|---|---|---|---|---|
| 51.3% | −0.162 [−0.355, +0.031], `P+` 0.047 | −0.141 [−0.333, +0.051], `P+` 0.070 | **−1.344** | **−3.71** |
| 53.0% | −0.247 [−0.469, −0.026], `P+` 0.015 | −0.322 [−0.557, −0.088], `P+` 0.005 | **−1.872** | **−5.31** |

*(measured; `<scratchpad>/seed_spread_raw.csv`.)* Seed 42 is the most extreme
draw at both magnitudes — as it must be, being the same draw. Dropping it
changes the level gaps to −0.103 and −0.166 and the delta gaps to −0.113 and
−0.273, so the conclusion does not rest on excluding it.

**So the calibration step is not free in this construction — it costs about
0.1 to 0.3 accuracy points, and at the 53% magnitude that cost resolves.** It
is simply an order of magnitude smaller than the 2.0-point swing D1 was
written from.

Note the two estimates of the same quantity do not agree on sign: §4's pooled
delta gap over 5 magnitudes is **+0.049 [−0.115, +0.213]** and §4a's over 2
magnitudes and 21 seeds is **−0.14 to −0.32**. They differ in which magnitudes
they pool and in which games received the plant, and their difference is the
size of their own standard errors. The claim these numbers jointly support is
a BOUND, not a point estimate: **whatever the calibration step does to a
measured effect, its magnitude is under about 0.35 accuracy points.** That
bound is what the decision needs; the sign of a 0.1-point quantity is not.

## 5. The bridge: plants that leave the real target in place

`plant_additive_effect` adds `gamma * carrier` to the real `ats_margin`
instead of replacing it, so both arms of the comparison still carry the real
market signal and the calibration sample still sees real target drift — the
situation every recorded verdict is actually in. The reference is a
**within-frame oracle**: the baseline arm's own predicted residual plus the
known planted term `gamma * carrier`, thresholded in that estimator's own
frame. That is the delta a perfect estimator would produce, so the ratio
against it isolates estimation and calibration loss.

`artifacts/calibration_distortion/20260818T154856Z/additive_summary.csv`:

| Carrier | Recovery vs the within-frame oracle, full pipeline | Recovery, sign-only | Gap (full − sign) |
|---|---|---|---|
| Orthogonal `{-1,+1}` (40 draws) | **0.964 [0.893, 1.036]** | **0.963 [0.921, 1.005]** | **−0.182 [−0.290, −0.074]**, `P+` 0.001 |
| Five REAL, collinear columns (25 runs) | 0.757 [0.574, 0.940] | 0.824 [0.635, 1.014] | −0.109 [−0.225, +0.008], `P+` 0.033 |

Per magnitude, orthogonal carrier (points):

| Planted | Oracle delta | Full pipeline | Sign-only | Gap |
|---|---|---|---|---|
| 50.5% | +0.18 / +0.03 | −0.01 | +0.07 | −0.07 [−0.31, +0.16] |
| 51.0% | +0.15 / +0.25 | +0.00 | +0.18 | −0.17 [−0.56, +0.22] |
| 51.3% | +0.61 / +1.00 | +0.63 | +0.99 | −0.36 [−0.66, −0.05] |
| 52.0% | +1.13 / +1.17 | +0.98 | +1.12 | −0.15 [−0.43, +0.14] |
| 53.0% | +1.95 / +2.20 | +1.94 | +2.11 | −0.17 [−0.33, +0.00] |

Per real carrier, pooled over magnitudes (R² against the rest of the design in
brackets):

| Carrier | Full pipeline | Sign-only | Gap |
|---|---|---|---|
| `rest_diff` (0.026) | +0.797 | +1.171 | −0.374 [−0.478, −0.271] |
| `week_cos` (0.230) | +0.411 | +0.387 | +0.024 [−0.452, +0.500] |
| `conference_game` (0.431) | +0.251 | +0.381 | −0.130 [−0.582, +0.323] |
| `week_sin` (0.755) | −0.010 | +0.010 | −0.019 [−0.224, +0.185] |
| `home_team_games` (0.868) | +0.040 | +0.085 | −0.045 [−0.283, +0.193] |

**Three readings, decision-first:**

1. **The full pipeline recovers 96% of the achievable planted delta** on an
   orthogonal additive plant, and the sign-only ablation recovers 96%. There
   is nothing to choose between them at the magnitudes that matter.
2. **The distortion does not depend on how the effect was planted.** It is a
   small negative for an orthogonal carrier (−0.18 pooled) and a smaller one
   for real, collinear carriers (−0.11 pooled), and it stays between −0.07 and
   −0.36 at every magnitude from 0.5 to 3.0 points. The orthogonal-vs-collinear
   axis the revisit list flagged as the likely explanation is **not** where the
   answer lives — inversion is absent for orthogonal plants too, once they are
   replicated.
3. **No sign inversion appears anywhere that matters.** Across 65 additive
   runs, the only condition mean where the two estimators disagree in sign is
   `week_sin` at −0.010 vs +0.010 — a disagreement between two readings of
   zero. The largest condition gap of any size is 0.37 points.

The one condition where the gap resolves — `rest_diff`, −0.374 — is also the
condition with the largest recovered effect (+0.80 to +1.17), so it reads as a
small proportional cost, not a fixed bias that could swamp a small effect.

## 6. Real effects, no planting

### 6a. What the location is worth on real CFB data

Before asking whether the two estimators agree on real contrasts, the same
base arm scored both ways, 12,283 classifiable games, week-blocked
`paired_feature_comparisons` at `samples=20000`:

| | Forced-pick accuracy |
|---|---|
| Full pipeline (production reader) | **51.160%** |
| Sign-only ablation | **51.299%** |
| Difference (full − sign) | **−0.138 points, 95% [−0.962, +0.685], `P+` 0.363** |
| Games where the two rules disagree | **21.0%** |

*(measured; reproduced by `--stage real`, which reports the same figure as
`location_value` in `summary.json`.)*

Two things follow. First, on CFB the residual location is worth **nothing
resolvable in either direction** — which is also what `docs/purged_cv.md`
already reported as a 0.1-point gap and treated as a caveat rather than a
result. Second, and more usefully: the two estimators disagree on **21% of
games** and still land 0.14 points apart. A rule that changes one pick in five
and moves accuracy by a seventh of a point is not a mechanism that can invert
a verdict.

That is a CFB reading, and it does **not** contradict
`docs/residual_location.md`'s NFL +2.12 points: different league, different
feature profile, different evaluator (NFL walk-forward vs CFB purged CV). The
honest summary across both is that the location's value is league- and
profile-specific and modest-to-nil on CFB — not that it is a defect.

**Cross-check that does not resolve, stated rather than buried**: the nearest
recorded CFB entry is `residual_location_shrink_100_cfb` — removing the
residual sample's MEAN location — at **−0.35 points, `P+` 0.105** on the CFB
walk-forward evaluator. This document's nearest equivalent — removing the
location entirely, on CFB purged CV — is **+0.138 in the opposite direction,
`P+` 0.637 for sign-only over full**. The two disagree in point-estimate sign
and agree that neither resolves. They also differ in three ways at once
(mean vs near-median location, walk-forward vs purged CV, 8,933 vs 12,283
games), so this is not a replication failure of either; it is two unresolved
readings of a quantity near zero. Nothing here re-classifies that entry.

### 6b. Do the two estimators agree on real contrasts?

Seventeen real mean-model contrasts against the frozen base design, every arm
on the identical 40 folds and the identical games, week-blocked
`paired_feature_comparisons` at `samples=20000`
(`artifacts/calibration_distortion/20260818T160920Z/real_contrasts.csv`,
accuracy points):

| Contrast | Full pipeline | `P+` | Sign-only | `P+` | Gap (sign − full) | Arm's own flip rate |
|---|---|---|---|---|---|---|
| `alpha_1e6` | −1.905 [−2.880, −0.913] | 0.000 | −0.847 [−1.986, +0.320] | 0.076 | **+1.058** | **88.9%** |
| `market_columns_only` | −0.586 [−1.609, +0.465] | 0.134 | −0.138 [−1.250, +0.982] | 0.408 | +0.448 | 30.9% |
| `drop_offense_state` | −0.497 [−1.301, +0.312] | 0.114 | +0.033 [−0.787, +0.874] | 0.534 | +0.530 | 22.4% |
| `drop_defense_state` | −0.448 [−1.149, +0.265] | 0.107 | −0.008 [−0.777, +0.770] | 0.489 | +0.440 | 24.6% |
| `thin_no_team_state` | −0.301 [−1.178, +0.579] | 0.251 | +0.252 [−0.717, +1.234] | 0.696 | +0.553 | 27.4% |
| `drop_epa_metrics` | −0.195 | 0.270 | −0.212 | 0.242 | −0.017 | 22.5% |
| `diff_only_state` | −0.187 | 0.287 | +0.269 | 0.788 | +0.456 | 20.6% |
| `plus_noise_column` | −0.081 | 0.361 | +0.114 | 0.687 | +0.195 | 20.6% |
| `alpha_100` | −0.065 | 0.176 | +0.000 | 0.477 | +0.065 | 21.4% |
| `alpha_0p1` | −0.033 | 0.270 | −0.024 | 0.262 | +0.009 | 20.8% |
| `alpha_1` | −0.024 | 0.299 | −0.049 | 0.077 | −0.025 | 21.1% |
| `drop_diff_block` | −0.008 | 0.442 | −0.073 | 0.127 | −0.065 | 21.3% |
| `alpha_1000` | +0.090 | 0.672 | −0.016 | 0.463 | −0.106 | 23.1% |
| `drop_spread_line` | +0.155 [−0.024, +0.336] | 0.953 | +0.016 [−0.186, +0.219] | 0.548 | −0.139 | 21.1% |
| `drop_context` | +0.171 [−0.503, +0.857] | 0.686 | −0.562 [−1.297, +0.165] | 0.064 | **−0.733** | 21.5% |
| `drop_experience` | +0.285 [−0.048, +0.623] | 0.950 | −0.244 [−0.572, +0.082] | 0.067 | −0.529 | 20.2% |
| `drop_total_line` | +0.285 [−0.362, +0.924] | 0.801 | −0.114 [−0.695, +0.455] | 0.347 | −0.399 | 16.8% |

Regressing sign-only on full-pipeline across all 17: slope **0.259 ± 0.120**,
`r` 0.487, **9 of 17 sign disagreements**, mean |gap| 0.339 points. Split by
effect size: **0 disagreements among the 2 contrasts above 0.5 points, 9 among
the 15 below it.**

**Read carefully, because this table looks like the D1 smoking gun and is
not.** Fifteen of the seventeen contrasts have a full-pipeline point estimate
inside **±0.5 accuracy points** — below this evaluator's own resolution — and
every one of their intervals crosses zero under both estimators. Nine sign
disagreements among fifteen readings of approximately zero is what two noisy
measurements of a null must produce; it is not evidence that one of them is
distorted. The planted data says exactly the same thing quantitatively: among
the 65 additive runs, restricting to runs where either estimate exceeds 0.5
points leaves **1 sign disagreement in 33**, while the runs below 0.5 points
disagree **18 times in 32**. The disagreement rate is a function of effect
size, not of the estimator.

Where there IS an effect to measure, the two estimators track each other:

| Population | n | Slope of sign-only on full | `r` | Sign disagreements |
|---|---|---|---|---|
| Additive planted runs (true deltas 0–2.2 pts) | 65 | **0.940 ± 0.054** | 0.910 | 19 / 65 (1 / 33 above 0.5 pts) |
| Overwrite planted runs | 56 | **0.916 ± 0.062** | 0.894 | 9 / 56 |
| Real contrasts (15 of 17 are null) | 17 | 0.259 ± 0.120 | 0.487 | 9 / 17 |

**A declared limitation, not a hidden one:** this project could not supply
enough real CFB contrasts with resolvable effects to make the real-contrast
regression informative. That is itself a known result — `docs/scaling_and_transfer.md`
and the "team quality is already priced" finding say most feature-contract
changes on this benchmark are worth less than the instrument's resolution. The
only way to manufacture a large real contrast was to cripple the model
(`alpha_1e6`), and that produces a degenerate arm (§7). The planted
populations, which do span 0–2.2 points of real effect, are what carry the
regression evidence here.

## 7. Mechanism: what actually drives the gap, and is it fixable

The delta gap is an **identity**, not a correlation:
`delta_full − delta_sign = arm_gap(candidate) − arm_gap(baseline)`, where
`arm_gap(x) = level_full(x) − level_sign(x)` is that arm's own location
effect. So the whole question is what makes ONE arm's location effect large.

Two things do, and the real-contrast table shows both:

- **The arm's own disagreement rate.** `alpha_1e6` shrinks every coefficient
  to nothing, so its predicted residual is nearly constant, the location
  threshold decides almost every pick, and the two rules disagree on **88.9%**
  of games — versus 17–31% for every non-degenerate arm. Its gap is 1.06
  points, three times any other arm's. **When a candidate's predicted residual
  has almost no spread, the estimator choice matters a lot; when it has normal
  spread, it does not.**
- **The size of the location relative to that spread.** Across all 80
  mechanism runs, `corr(gap, |threshold| / sd(predicted residual))` is
  **−0.218** (n = 80) — the right sign, weak.

Sweeps, 4 replicates per condition (`mechanism_summary.csv`; intervals are
wide at that replicate count and are reported as such):

| Knob | Value | Gap | mean \|threshold\| | mean sd(pr) |
|---|---|---|---|---|
| `distribution_fraction` (additive) | 0.10 | −0.510 [−1.075, +0.055] | **0.923** | 1.305 |
| | 0.20 (production) | −0.306 [−0.925, +0.313] | 0.554 | 1.340 |
| | 0.35 | −0.306 [−0.612, +0.000] | 0.436 | 1.328 |
| | 0.45 | −0.234 [−0.997, +0.529] | 0.443 | 1.333 |
| design width (overwrite) | full, 35 cols | −0.306 [−0.763, +0.151] | 0.468 | 1.008 |
| | **thin, 11 cols** | **−0.822** [−1.819, +0.175] | 0.380 | **0.674** |
| `n_blocks` (both) | 10 / 20 / 40 / 80 | no ordered pattern | 0.17–0.66 | 0.96–1.37 |

**Is it fixable? Yes, in principle, and the lever is the calibration sample
size.** The location's magnitude falls from 0.92 to 0.44 points as
`distribution_fraction` goes from 0.10 to 0.45, exactly as a sampling-noise
term should (the location is a near-median of an `n`-draw sample), and the gap
falls with it. That is a lead, not a recommendation: changing
`distribution_fraction` changes the production model, and on real CFB data the
location is worth −0.138 [−0.962, +0.685] (§6a) so there is no established
benefit to protect and no established harm to fix. It would need its own
screen.

**One recorded diagnosis is contradicted.** `docs/purged_cv.md` attributes the
instability to "the ridge fit absorbing per-fold noise across ~33
real-but-irrelevant covariates". Removing 24 of those covariates makes the gap
**larger**, not smaller (−0.82 thin vs −0.31 full in the overwrite
construction; +0.21 vs +0.07 in the additive one), because a thinner design
has a smaller predicted-residual spread (sd 0.67 vs 1.01) and the same
location therefore moves a larger share of picks. The covariate count is not
the driver; the ratio of the location to the predicted-residual spread is.

## 8. Verdict

**D1 is not real at the magnitude it was written from.** The recorded evidence
— a planted +1.3 coming back −0.7, a planted +3.0 coming back +1.14 — is one
plant draw, reported twice, from a readout whose replicate-to-replicate
standard deviation is 0.4–0.7 accuracy points. Replicated properly, the
calibration step's effect on a measured quantity is bounded by roughly **0.35
accuracy points** in every construction tested, and the full pipeline recovers
**0.964 [0.893, 1.036]** of the achievable paired delta where the sign-only
ablation recovers **0.963 [0.921, 1.005]**.

Stated as the four questions the revisit list asked:

| Question | Answer |
|---|---|
| Does the calibration step **invert** a small effect? | **No.** No inversion at any magnitude from 0.5 to 3.0 points, on orthogonal or collinear plants, in 121 planted runs. |
| Does it **attenuate** one? | **By at most ~0.2–0.35 points**, and by the same amount whether the effect is 0.5 or 3.0 points, so it is closer to a small fixed cost than to a proportional shrinkage. |
| Is it worse for the **paired delta** the verdicts actually use? | **No.** The delta gap is an identity — the candidate arm's own location effect minus the baseline's — so much of the offset cancels. But the level framing is not badly distorted either once replicated (level gaps −0.06 to −0.25), so the level-vs-delta distinction is not what rescued the result: **replication is.** |
| Do the two estimators disagree on **real** contrasts? | **On 9 of 17, yes — and it means nothing**, because 15 of those 17 contrasts are null at this evaluator's resolution and two noisy readings of zero must disagree about half the time. The planted runs confirm the rate is a function of effect size: 18 disagreements in 32 below 0.5 points, **1 in 33 above it.** |

### What this implies for the Tier 1 re-run list

**Tier 1 shrinks.** The branch `docs/revisit_list.md` reserved for "D1 is an
artifact" is the branch we are in: Tier 1 collapses to the D4
degenerate-bootstrap case (`player_qb_continuity_matched_alpha`) plus the two
entries that recorded no continuous evidence at all (`pbp_drive_bundle`,
`player_qb_continuity`). D1 does not, on this evidence, put every terminal
negative in the project under suspicion.

Applied to the specific rows:

- `cfb_role_continuity` — closed on **−0.67 points**. A 0.35-point bound on the
  calibration step cannot carry −0.67 across zero, and attenuation shrinks
  toward zero rather than away from it, so it cannot have manufactured this
  negative from a positive. **D1 is not a reason to re-run it.** Whether it
  should be re-run for other reasons (D2's understated intervals, the missing
  `probability_positive`) is a separate question this document does not answer.
- `pbp_drive_bundle` (−0.08 points) and `player_qb_continuity` (+0.00 points)
  — both point estimates are smaller than the calibration step's own effect,
  so D1 is formally irrelevant to them: what makes these two suspect is that
  they record **no `probability_positive` at all**, a bare pass/fail. That is
  the continuous-evidence rule, not D1.
- `residual_location_recency_hl200_cfb` / `..._hl400_cfb` — **outside what this
  document can test.** Both arms of a reader-only contrast produce an identical
  sign-only arm, so the cross-check is structurally unavailable. Their
  terminal classification rests on D2's interval question, not D1's.

### The one case where the estimator choice genuinely matters

A candidate whose predicted residual has almost no spread — `alpha_1e6`, where
the two rules disagree on **88.9%** of picks and the gap reaches **1.06
points** — is the exception, and it is diagnosable from the arm itself, not
from hindsight. **Rule of thumb this document supports: report a candidate
both ways whenever its own full-vs-sign disagreement rate departs from the
17–31% band every non-degenerate arm here sits in, or whenever its estimate is
within ~0.35 points of a decision boundary.**

### One methodological change that should stick

The positive control in `scripts/purged_validate.py` should be **replicated**
before it is cited again. A single-seed positive control at this evaluator's
resolution cannot separate a 2-point distortion from zero, and reporting two
magnitudes at the same seed presents one draw as two independent readings.
Replicate count, not construction, is what made the recorded reading
misleading.

**Update, 2026-08-18 — replicated.**

Rerunning positive_control()'s own construction — purged_cv_backtest/_accuracy_and_ci, n_blocks=40 — across 20 independent seeds per magnitude instead of one shared seed (scripts/purged_control_replication.py, which does not modify purged_validate.py): at 51.3%, recorded seed 42 recovers -0.67 points (wrong sign); the fresh 20-seed distribution recovers +0.375 points, 95% [+0.057, +0.694], P+ 0.988, with 15/20 seeds landing on the planted sign — seed 42 ranks 3rd-lowest of 20 (z -1.69). At 53.0%, recorded seed 42 recovers +1.14 points against a fresh mean of +2.245 points, 95% [+1.964, +2.526], P+ > 0.999, 20/20 on-sign — seed 42 is the single lowest of the 20 draws (z -2.10). This confirms the section 4/4a finding through the recorded script's own exact code path rather than calibration_distortion_screen.py's separate construction: D1's originating 2-point sign-flip was a single unlucky seed, not a property of the pipeline.

## 9. What this does NOT license

- **It does not say the calibration step is free.** Its measured cost is
  0.1–0.35 accuracy points and in several conditions that cost resolves
  (`P+` 0.001 for the pooled orthogonal additive gap). Any single recorded
  delta within about 0.35 points of a decision boundary is genuinely
  estimator-dependent and should be reported both ways.
- **It does not clear reader-only families.** `residual_location` and
  `ecdf_smoothing` change only how a fixed residual sample is read, so the
  sign-only ablation is identical on both sides of those contrasts and carries
  zero information about them. Nothing here bears on their verdicts.
- **It does not measure NFL.** Every number is CFB purged CV. The location's
  value differs sharply between the two leagues (CFB −0.14 here vs NFL +2.12
  in `docs/residual_location.md`), so the *size* of the calibration step's
  contribution is not transferable even though the *mechanism* — one location
  offset per fitted model — is identical code in both.
- **It does not address D2, D3, D4 or D5.** Interval width, seed jitter,
  degenerate bootstraps and the decision frame are untouched here.
- **It does not re-classify any registry entry.** No `registry/*.json` was read
  for anything but context, and none was written.
- **It is one instrument.** Purged CV with `n_blocks=40` on 12,500 CFB games.
  The production NFL evaluator is a walk-forward with a growing training
  window and a much smaller corpus; the mechanism is the same code, the
  numbers are not automatically the same.

## 10. Reproducing

```
./.tools/uv.exe run --no-sync python scripts/calibration_distortion_screen.py --stage all
./.tools/uv.exe run --no-sync python scripts/calibration_distortion_screen.py \
    --resummarize artifacts/calibration_distortion/<stamp>
```

Archived runs behind this document:
`artifacts/calibration_distortion/20260818T154856Z/` (overwrite, additive,
mechanism) and `.../20260818T160920Z/` (real contrasts, run after three
deliberately extreme contrasts were added). The 21-seed reproduction of the
recorded control in §2/§4a is a scratch script; every other number is
`--resummarize`-reproducible from the archived raw rows.

Reads `data/processed/cfb_game_features.parquet` only. Writes raw rows and
summaries to `artifacts/calibration_distortion/<stamp>/` (gitignored). No
artifact, registry, or tracked-doc write. `--resummarize` re-derives every
table in this document from the archived raw rows without refitting anything.
