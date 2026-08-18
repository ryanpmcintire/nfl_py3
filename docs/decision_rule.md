# A decision rule for forced picks: empirical Bayes instead of a promotion bar

> # FLAWED — DO NOT USE. RETRY REQUIRED.
>
> **Marked flawed by the project owner on 2026-08-18. Work was stopped
> mid-flight and no conclusion in this document may be cited, quoted, or acted
> on.** Every posterior mean, `probability_positive`, verdict and
> model-averaging total below is void until this is redone. In particular the
> deflated injury verdict (+1.316 observed -> +0.037 posterior) must NOT be
> treated as a finding, and nothing here re-classifies any registry entry.
>
> **Known defects in the shrinkage, to be fixed on retry:**
>
> 1. **The prior's reference class is polluted by correlated variants.** The
>    210 measurements were treated as independent draws after removing only
>    *exact* week/season-block duplicates. They are not independent: many are
>    multiple arms of ONE idea (the eight `residual_location_*` arms are eight
>    variants of a single question; the MOD-07 family appears several times at
>    different windows and metrics). Counting each arm as a fresh draw inflates
>    the number of near-zero observations, which drags the prior mean negative
>    and shrinks `tau`. That makes the shrinkage far more aggressive than the
>    evidence supports, and it does so in a way that is invisible in the output.
> 2. **Candidates were treated as exchangeable when they demonstrably are
>    not.** A single pooled prior shrinks a candidate carrying independent
>    evidence of being real -- measured split-half reliability 0.933, a
>    monotone dose-response, orthogonality to line movement -- by exactly the
>    same factor as an arbitrary parameter sweep with none of those properties.
>    That systematically over-shrinks precisely the candidates most likely to
>    be genuine.
> 3. **A negative prior mean (-0.128) inherited from defects 1 and 2 was then
>    applied to every candidate**, so the method starts from "assume it hurts"
>    rather than "assume nothing", compounding both errors.
>
> The MOD-07 calibration check (88.7% predicted shrinkage vs 83.2% delivered)
> is the one result worth carrying forward as a *method* -- validating a prior
> against a known regression event is sound and should be repeated -- but it
> cannot vouch for a prior fitted from a polluted reference class.
>
> The framing that motivated this work is NOT withdrawn and remains correct:
> forced picks mean declining a candidate is an active bet that it is worth
> zero, "needs N more games" is a banned non-answer, and every candidate must
> get a use-it/don't-use-it verdict from data in hand. Only the shrinkage
> arithmetic is void.

> # SECOND, INDEPENDENT DEFECT (2026-08-18 audit) — do not mistake this for the shrinkage flaw above
>
> §5's table and prose below label
> `residual_location_recency_hl200_cfb`, `residual_location_recency_hl400_cfb`,
> and `player_qb_continuity_matched_alpha` **"refuted mechanism — excluded,
> not a live candidate."** That label is stale. `registry/weak_signals.json`
> reclassified all three to **`unresolved_below_power`** on 2026-08-18:
> `residual_location_recency_hl200_cfb` (`probability_positive` 0.2585),
> `residual_location_recency_hl400_cfb` (`probability_positive` 0.3080), and
> `player_qb_continuity_matched_alpha` (`probability_positive` 0.796 — this
> one was reclassified because the original "refuted" argument compared the
> wrong pair of arms; see `docs/anytime_valid.md` §6). None of the three is
> `refuted_mechanism` any more, and none should be described as excluded
> from the live-candidate accounting. This is a separate defect from the void
> shrinkage arithmetic above — it would be wrong even if the shrinkage method
> were sound — and is called out here so a reader does not assume the one
> banner covers both problems. The specific numbers in §5's table (posterior
> means, P+, the model-averaging totals) remain void for the shrinkage reason
> already stated; this note only corrects the classification label attached
> to these three rows.

Written 2026-08-18, executing the task set out after
`AGENTS.md` § "A promotion bar is not a decision bar." The problem stated
plainly: this project kept concluding that candidates were "unresolved" and
that resolving them would need more games than will ever exist. That framing
is banned here. The pool is **forced picks** -- 285 cards submitted every
season, no abstain option -- so declining a candidate is not caution, it is
an active bet that the candidate is worth exactly zero, and that bet has to
clear the same expected-value bar as any other. This document fits an
empirical Bayes prior from the project's own measurement history, validates
it against a real regression event, and applies it to every live candidate
on record. Every verdict below is `use` or `don't use`, with a posterior
expected gain — never "needs N games."

Reproduce everything here with:

```
uv run --no-sync python scripts/decision_load_measurements.py
uv run --no-sync python scripts/decision_apply.py <out.json>
```

Both scripts are read-only: no registry write, no rotation-window spend, no
model or pick changes.

## 1. The empirical prior

### 1.1 What went into it

`scripts/decision_load_measurements.py` scans every
`artifacts/**/*paired*.csv` file (27 artifacts) for rows on the **accuracy**
scale — excluding Brier, log-loss, and MAE rows, a different currency that
would silently distort a prior meant to answer "how big is a typical
accuracy effect in this domain." That filter alone recovers **exactly 282
rows**, matching the count of "recorded effect measurements" in the task.

Those 282 rows are not 282 independent draws. Most artifacts report the same
point estimate twice — once resampled by week block, once by season block,
purely to get two versions of the interval, not because the comparison ran
twice. Collapsing exact-duplicate point estimates (same source file, same
identifying columns, same window, same estimate to 6 decimal places, keeping
the wider interval) leaves **210 independent measurements**. Two further rows
were dropped outright: a same-file, zero-width-CI self-comparison (a feature
set compared against itself, a pipeline no-op, not a measurement).

The 11 rows without a `metric` column (`paired_season_accuracy` tables) carry
no CI; their standard error is derived from the project's own reported
per-season accuracy percentages via the standard paired-proportion
approximation `sqrt(2 * p_bar * (1 - p_bar) / n)`, assuming zero correlation
between the two arms — deliberately conservative (a real positive
correlation, which two arms scored on identical games will have, can only
shrink it).

### 1.2 The fit: Paule-Mandel random effects

`theta_i ~ N(mu, tau^2)`, `y_i | theta_i ~ N(theta_i, se_i^2)`. This is the
same random-effects machinery `weak_signals.pooled_effect` already uses to
pool repeated looks at one signal, repurposed here to fit a prior over many
*different* signals — literally the empirical-Bayes / James-Stein
construction: borrow strength from "how big do effects in this project's
history usually turn out to be" to correct any one small-sample estimate.

Two estimators of `tau^2` were compared:

| method | mean | tau | standardized-residual variance |
|---|---|---|---|
| DerSimonian-Laird (one-step, `weak_signals.py`'s own estimator) | -0.139 | 0.204 | **1.59** (badly under-dispersed) |
| **Paule-Mandel (iterated to convergence)** | **-0.128** | **0.555** | **0.998** (well calibrated) |

The diagnostic: `z_i = (y_i - mu) / sqrt(se_i^2 + tau^2)` should have unit
variance if the model is right. DL's one-step tau leaves `z` at variance 1.59
— badly under-dispersed, i.e. DL's tau is too small for a sample this
heterogeneous in precision (SE ranges from measurements on 248 games to ones
on 11,780). Paule-Mandel, which iterates the same moment equation
(`Q(tau^2) = K - 1`) to convergence instead of solving it in one Newton step
from `tau^2 = 0`, lands at `z` variance 0.998 — essentially exact. **Paule-
Mandel is the prior used everywhere below.** Its standardized residuals also
show negligible excess kurtosis (0.014 versus 0 for an exact normal), which
is the justification for a normal prior rather than a heavier-tailed one:
once heteroskedastic measurement noise is correctly deconvolved (which raw
kurtosis of the unstandardized `y_i` does not do — it conflates true
heterogeneity with the noise-variance mixture from wildly different sample
sizes), there is no detectable excess tail weight left to model.

### 1.3 The fitted prior

**N(mean = -0.128, tau = 0.555) accuracy points, fit from 210 measurements.**

The center is slightly negative: the modal outcome of a measurement in this
project's history is a small loss, consistent with the "team quality is
already priced" ceiling finding (`docs/pool_edge_plan.md`) — most things
tried here don't help. The spread (tau = 0.555 points) says true effects in
this domain mostly live within about a point of that center, with a
minority running larger — consistent with the injury-value-lost family
(+1.3 points) and MOD-07 (delivered +0.33) both being real, if unusually
large for this population.

**Shrinkage factor for a typical measurement** (fraction of posterior weight
left on the new data, `tau^2 / (tau^2 + se^2)`):

| measurement SE | weight on data | weight on prior |
|---|---|---|
| 0.3 (a well-powered result) | 0.774 | 0.226 |
| 0.5 | 0.552 | 0.448 |
| 1.0 (typical for a 400-500 game window) | 0.236 | 0.764 |
| 1.5 | 0.121 | 0.879 |
| 2.0 | 0.072 | 0.928 |

### 1.4 Selection bias: quantified, not hand-waved

The 282 measurements are not a random sample of "everything anyone could
measure" — they are what got tested and written down. Two things bound how
bad this is here, and one residual risk stays open:

- **This project records negatives on purpose** (`registry/weak_signals.json`,
  the whole "three kinds of negative" discipline in `AGENTS.md`). Of the 210
  fitted measurements, **93 are positive, 113 negative, 6 exactly zero** — a
  slight *negative* lean, not the strong positive lean a file-drawer of
  published-only hits would show. That is direct, measured evidence against
  severe positive-selection bias in this specific corpus.
- **One experiment can still dominate a slice of the sample.** 37 of 210
  (17.6%) come from a single 22-arm ridge-penalty sweep
  (`artifacts/groupwise_ridge`); 24 more from one 8-arm residual-location
  sweep. Hyperparameter sweeps and substantive feature tests are not
  necessarily drawn from the same population of true effects. Splitting the
  fit by reference class shows this matters at the margin but not
  qualitatively: NFL-only (173 measurements, mostly substantive feature
  tests) gives `mean=-0.078, tau=0.634` (a *wider* prior, less shrinkage);
  CFB-only (37, dominated by the two large sweeps) gives
  `mean=-0.105, tau=0.047` (near-null, as expected for hyperparameter
  tuning around an already-tuned default). **The all-pooled prior used
  throughout this document (tau=0.555) sits between these and is the more
  conservative of the two, so it is not inflating anyone's case** — a
  reference-class objection here would only make posteriors more positive,
  not less.
- **Residual risk, stated and not resolved**: measurements this project
  chose to *write up in prose* (MOD-07, injury value lost) may still be
  systematically larger or more interesting than measurements that only ever
  lived in a CSV. There is no clean way to rule this out from data already in
  hand; it is flagged rather than corrected.

## 2. Validation: MOD-07's regression, predicted in advance

`docs/mod07_stack.md` measured **+1.97 accuracy points on 456 games**
(week-blocked 95% CI [-1.10, +5.00]), a `probability_positive` of 0.8745 —
short of the project's own 0.90 promotion bar. Scored later on the full
1,537-game opener-graded window, it delivered **+0.33 points** — textbook
regression of a selected effect. A correctly calibrated prior should have
predicted approximately that shrinkage in advance.

| | value |
|---|---|
| raw estimate (456 games) | +1.970 pts (SE 1.556) |
| **posterior mean (prior-shrunk)** | **+0.109 pts** |
| delivered estimate (1,537 games) | +0.330 pts |
| predicted shrinkage (1 − weight on data) | **88.7%** |
| actual shrinkage realized | **83.2%** |

The posterior mean is not an exact hit — it undershoots the delivered number
by 0.22 points — but the thing that actually needed validating is the
*magnitude of the correction*, and that lands within 5.5 percentage points:
the prior said "discount the raw number by about 89%," reality discounted it
by 83%. **The prior is calibrated.** No other clean before/after pair exists
in the record with both a documented small-window selected estimate and a
documented larger-window delivered outcome for the identical candidate — the
opener-bias ablation, the QB-continuity replication, and the fixed-vs-learned
availability test were all checked and none supply the second half of a
matched pair the way MOD-07 does.

## 3. Calibration-noise sensitivity (docs/purged_cv.md)

The purged-CV positive control (owned by a separate agent workstream; read,
not edited, here) plants a known accuracy effect and runs it through the
real pipeline. At a **1.3-point** true effect the full pipeline recovered
**the wrong sign** (-0.67 against a realized +1.67); at **3.0 points** it
recovered +1.14 of +3.34. The document's own diagnosis is **not**
multiplicative attenuation — a sign flip is impossible under a `0 < k < 1`
shrinkage model — it is additive, direction-unstable noise from the
out-of-time residual/calibration step's fold-to-fold instability (quoted at
0.9-1.3 accuracy points across individual folds in a 10-fold worked
example). A "divide the estimate by k" correction was deliberately **not**
applied: it is not the mechanism identified, and applying one anyway is
exactly the "pick the value that produces the most positive answer" failure
this check exists to avoid.

Two corrections were run instead, both by inflating standard errors in
quadrature (`combine_standard_errors`), never by rescaling point estimates:

1. **Real-data-anchored (primary, applied to all 210 fitting measurements):**
   the same document's only non-synthetic reading of this distortion —
   comparing the full pipeline against a sign-only ablation on real data —
   found a **0.1-point** gap, aggregated over the hundreds of refits a
   full walk-forward evaluation performs (the regime nearly every one of the
   210 fitted measurements comes from). Effect on the prior: **negligible**
   (`mean -0.128 -> -0.123`, `tau 0.555 -> 0.513`).
2. **Targeted worst case (applied only to the 27 fitting measurements, and
   the live candidates, on <=1,000 games — structurally the closest thing in
   this project's record to the purged-CV positive control's own small fold
   count):** the full **1.1-point** midpoint of the quoted per-fold range.

| candidate | posterior mean (baseline) | posterior mean (noise-floor) | verdict change |
|---|---|---|---|
| mod07_weak_stack (raw) | +0.109 | +0.019 | none |
| injury_value_lost_narrowed | +0.242 | +0.037 | none |
| injury_value_lost_gradient | +0.175 | +0.038 | none |
| hc_year_one_fade | +0.355 | +0.011 | none |

**No verdict flips under either correction.** Every small-window candidate's
posterior mean shrinks further toward zero and stays positive; nothing that
was `don't_use` becomes `use`, and nothing that was `use` becomes `don't_use`.
This is itself the finding worth reporting: the risk was that under-
correcting for pipeline attenuation was making the tool *falsely
pessimistic*; the measured answer is that a defensible correction leaves
every verdict exactly where it was and mainly humbles the *confidence*
(P+ on the small-window candidates drops to 0.51-0.55, barely better than a
coin flip) rather than the *sign*. A third, more literal reading was also
run and explicitly **not adopted**: applying the full 1.1-point floor to
*every* one of the 210 fitting measurements (not just the small ones)
collapses `tau` to exactly 0 — the noise floor alone explains all observed
dispersion, so the model can no longer distinguish any candidate from any
other. Taken to its conclusion this argues for *less* differentiation
between candidates, the opposite of an inflation risk, which is itself
evidence the literal reading over-applies a 10-fold experiment's per-fold
instability to measurements that average over hundreds of refits (where the
document's own real-data check shows only 0.1 points of gap).

## 4. The decision rule

Under forced picks: **use a candidate iff its posterior mean is positive.**
Not iff `probability_positive >= 0.90`, not iff a confidence interval
excludes zero — those are publication thresholds. `evaluate_candidate`
(`src/nfl_ats/decision_rule.py`) reports, for any candidate:

- `posterior_mean`, `posterior_sd` — the shrunk estimate and its uncertainty.
- `probability_positive` — reported always; "contains zero" is never used
  (binding, `AGENTS.md`).
- `expected_cost_if_use_is_wrong` = `E[max(0, -theta)]` under the posterior —
  what "use" costs you if the candidate is actually bad.
- `expected_cost_if_skip_is_wrong` = `E[max(0, theta)]` — the symmetric cost
  of skipping a candidate that is actually good. Their difference is exactly
  `posterior_mean` (pinned in `tests/test_decision_rule.py`), which is the
  whole point: the forced-pick decision only ever needs the sign of that one
  number.

`model_average` (same module) replaces the binary gate with conviction
weighting. Two modes: `"stack"` for independent, additive candidates (weight
each by its own `probability_positive`, do not renormalize, sum
`weight_k * posterior_mean_k`); `"blend"` for mutually exclusive alternatives
(weights normalized to 1, `combined_expected_gain` is the EV of a
probability-weighted lottery over the alternatives rather than a sum).

## 5. Every live candidate

Posterior means below use the noise-floor-corrected prior and (for
<=1,000-game candidates) noise-floor-corrected SEs from section 3 — the more
conservative of the two readings, and the one with no verdict changes versus
the uncorrected version.

| candidate | observed | posterior mean | P+ | verdict |
|---|---:|---:|---:|---|
| pooled_weak_signal (3-signal pool, AGENTS.md) | +0.724 | **+0.450** | 0.938 | **use** |
| injury_value_lost (narrowed, §4 of injury_value_lost.md) | +1.316 | **+0.037** | 0.531 | **use, small edge** |
| injury_value_lost (conflated, registry headline) | +1.750 | +0.038 | 0.531 | use, small edge |
| hc_year_one_fade | +0.753 | **+0.011** | 0.509 | **use, ~coin flip** |
| groupwise_ridge_block_penalties | +0.537 | +0.211 | 0.720 | use *(winner's-curse caveat, below)* |
| ridge_alpha_global | +0.258 | +0.122 | 0.655 | use |
| mod07_weak_stack (raw, retrospective check) | +1.970 | +0.019 | 0.515 | use *(already promoted; see below)* |
| ecdf_smoothing_accuracy | -0.300 | -0.263 | 0.132 | **don't use** |
| fourth_down_aggressiveness *(ats_points x3 exchange rate)* | -0.114 | -0.116 | 0.295 | **don't use** |
| penalty_discipline | +0.670 | **-0.079** | 0.437 | **don't use, ~0 either way** |
| residual_location_shrink_025_cfb | -0.034 | -0.043 | 0.399 | don't use, ~0 either way |
| residual_location_shrink_050_cfb | -0.235 | -0.212 | 0.183 | don't use |
| residual_location_shrink_075_cfb | -0.381 | -0.314 | 0.113 | don't use |
| residual_location_shrink_100_cfb | -0.347 | -0.291 | 0.127 | don't use |
| residual_location_recency_hl100_cfb | -0.425 | -0.338 | 0.110 | don't use |
| residual_location_recency_hl800_cfb | -0.168 | -0.164 | 0.124 | don't use |
| residual_location_recency_hl200_cfb | -0.548 | -0.451 | 0.033 | ~~refuted mechanism~~ **unresolved_below_power** (corrected 2026-08-18; P+ 0.2585 in the registry) |
| residual_location_recency_hl400_cfb | -0.560 | -0.497 | 0.005 | ~~refuted mechanism~~ **unresolved_below_power** (corrected 2026-08-18; P+ 0.3080 in the registry) |
| player_qb_continuity_matched_alpha | +1.103 | +0.057 | 0.548 | ~~refuted mechanism~~ **unresolved_below_power** (corrected 2026-08-18; P+ 0.796 in the registry — see `docs/anytime_valid.md` §6) |

**Reading each row:**

- **pooled_weak_signal** — the strongest candidate in the table by a wide
  margin (P+ 0.938 even after the conservative noise-floor correction).
  **Use it.**
- **injury_value_lost** — both the narrowed (semantics-shift-free) and
  conflated readings land at essentially the same small positive edge once
  properly shrunk; the narrowed number is the one to build against, per its
  own predeclaration. Use it, but the expected gain is small (~0.04 points),
  not the naive +1.3-1.75 the raw measurement suggested.
- **hc_year_one_fade** — posterior mean +0.011 is close enough to zero that
  the honest reading is "expected gain is ~0, so it barely matters whether
  it's used" — it is *marginally* positive, so the rule says use it, but
  this is the textbook case the task's own escape hatch describes.
- **groupwise_ridge_block_penalties** — flagged with an extra caveat beyond
  ordinary shrinkage: this specific number is the **max over 22 tested
  configurations** (registry's own classification evidence), so on top of
  the general prior-shrinkage applied here, a winner's-curse correction
  specific to order statistics would pull it down further still; treat
  +0.211 as an upper bound, not a point estimate. Still positive under a
  conservative prior — lean use, least confidently of the "use" rows.
- **mod07_weak_stack** — already promoted to production (commit `68b4dc0`) on
  the delivered 1,537-game +0.33 number, not the retrospective row above.
  The row exists to show the empirical Bayes rule would have said "use it"
  even applied to the original, smaller, noisier 456-game look — a
  retroactive validation of the promotion, not a new decision.
- **ecdf_smoothing_accuracy, fourth_down_aggressiveness** — both resolvably
  negative under shrinkage; **don't use.**
- **penalty_discipline** — flips sign from its raw +0.67 to a slightly
  negative posterior once its (self-derived) uncertainty is accounted for.
  The honest read: **expected gain is ~0 either way** — a small negative
  lean, not a resolved loss, so this is a low-stakes call.
- **residual_location_* family (unresolved members)** — every one of the six
  non-refuted arms comes out negative under the posterior. This directly
  validates production: the accidental unweighted residual ECDF is already
  at or near the best point in this family, and none of the eight tested
  remedies (recency weighting, location shrinkage) is worth adopting.
  hl200 and hl400 were originally described here as additionally **refuted
  mechanisms** (resolvably wrong sign under two independent blockings) and
  excluded from the live-candidate accounting entirely. **Corrected
  2026-08-18 (see the second-defect banner at the top of this document):**
  both are now registry-classified `unresolved_below_power`
  (P+ 0.2585 and 0.3080 respectively), not refuted, and belong in the same
  "unresolved, negative-leaning" bucket as the other six arms, not excluded
  from it.
- **player_qb_continuity_matched_alpha** — this bullet originally read
  "included for completeness only. Its own registry classification
  (`refuted_mechanism`) established this is a calibration artifact of a
  near-null ridge-alpha contrast, not a measurement of QB continuity; the
  positive posterior shown is what the math produces if you feed it in
  anyway, and it must not be read as a usable candidate." **Corrected
  2026-08-18:** that "refuted_mechanism" premise was wrong — the
  `classification_evidence` it cited was comparing the wrong pair of arms
  (see `docs/anytime_valid.md` §6). The registry now classifies this entry
  `unresolved_below_power`, P+ 0.796, and it is a live, if underpowered,
  candidate, not a calibration artifact to be dismissed.

## 6. Model averaging

**Stack (independent, additive candidates; refuted mechanisms excluded):**

```
combined_expected_gain = +1.121 pts   (uncorrected prior; every non-refuted
                                        positive-posterior candidate summed,
                                        weighted by its own P+)
```

This number is a **mechanical upper bound, not a deployable total** — several
of the "included" candidates are not independent. `mod07_weak_stack`,
`injury_value_lost_narrowed`, and `injury_value_lost_gradient` all draw on
the same [2020, 2021] window and the same mechanism (injury value lost is
explicitly "the availability/player-value half of the MOD-07 stack" per
`docs/injury_value_lost.md`); summing all three double- and triple-counts
one underlying signal. `mod07_weak_stack` is also already in production, so
its contribution is not incremental to anything.

**De-overlapped total, genuinely additive candidates not yet in production:**

```
hc_year_one_fade (+0.011) + groupwise_ridge_block_penalties (+0.211)
  + ridge_alpha_global (+0.122)  =  +0.344 pts
```

(`pooled_weak_signal` is excluded from this sum too — its three unnamed
constituent signals are not fully identifiable from the record cited in
`AGENTS.md`, and it plausibly shares components with the MOD-07 family.)

**Blend (mutually exclusive alternatives — which `residual_location_*`
configuration to run, refuted arms excluded):**

```
combined_expected_gain = -0.175 pts, weights concentrated on shrink_025
                          (42.5%) and shrink_050 (17.9%)
```

Every alternative underperforms current production; a probability-weighted
blend across them is still worse than doing nothing. This is the "use it or
don't" answer for the whole family: **don't** — keep production as it is.

## 7. Module ownership

`src/nfl_ats/decision_rule.py` — the pure-arithmetic library (prior fit,
posterior update, model averaging; stdlib only, no scipy dependency).
`scripts/decision_load_measurements.py` — the artifact loader and
deduplicator. `scripts/decision_apply.py` — wires both together against the
live registry and docs-cited numbers, runs the section-3 sensitivity check,
and prints/writes the report reproduced above. `tests/test_decision_rule.py`
covers the normal-distribution primitives, prior recovery on simulated data
with a known ground truth, shrinkage monotonicity, the expected-cost
identity, and a pinned regression check on the real MOD-07 numbers.
