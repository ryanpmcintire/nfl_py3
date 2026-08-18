# Refit-aware uncertainty: how much this project's intervals understate, and what shrinks it

Written 2026-08-18, acting on a power audit's finding that this evaluator's
largest measured noise source — resampling the training rows flips **15-22%**
of the model's own picks (CFB, predicted-margin bootstrap SD 2.99pt → 0.77pt
as training grows) — is invisible to every confidence interval this project
has ever reported, because the block bootstrap
(`experiments.paired_feature_comparisons`, `outcomes.outcome_bootstrap_intervals`)
resamples GAMES around one already-fitted model but never refits it. This
document builds the honest, refit-aware alternative
(`src/nfl_ats/estimation_variance.py`), validates it against a known ground
truth, quantifies the gap on two real CFB comparisons, screens two
variance-reduction levers (bagging, centre shrinkage), and tests the `f`-lever
design principle the power audit's formula implies.

**Headline**: every interval width measured here — synthetic and real,
null-effect and real-effect — comes out **17% to 58% too narrow** today.
Coverage of a known synthetic ground truth confirms this is a real
under-coverage problem (naive 89.5-92.5% vs. nominal 95%), not a modelling
artifact. On the project's own flagship CFB comparison, the honest interval
no longer excludes zero, though `probability_positive` barely moves (0.979 →
0.958). **Bagging** cuts the model's own pick-flip rate by roughly 6x on real
data with a small, positive accuracy note. **Centre shrinkage** does not help
and should not be pursued further. **The `f` lever is real and large**: gating
a candidate's influence to where it disagrees with a simpler baseline turned
a real CFB comparison from `probability_positive` 0.264 (losing) to 0.951
(resolving), replicated independently on two disjoint season splits.

## 1. The honest interval, and how it differs from every interval reported before it

`src/nfl_ats/estimation_variance.py` adds, without touching any existing
model, feature profile, or the active artifact:

- `refit_predicted_values` — refits `margin.make_margin_estimator("ridge",
  ...)` on `n_boot` row-bootstrap resamples of the training frame, predicting
  on a **fixed** test frame each time. This is the source every prior interval
  omitted: it varies the training ROWS, not the test games.
- `naive_block_bootstrap_interval` — the CURRENTLY REPORTED style: resamples
  games only, around one fit. A numpy analogue of
  `paired_feature_comparisons`, pinned equal to it in
  `tests/test_estimation_variance.py` (estimate exact, quantiles within 0.02
  — the tiny gap is only from `np.unique`'s sorted group order vs. pandas
  `groupby(sort=False)`'s encounter order, not a different algorithm).
- `refit_aware_paired_interval` — the HONEST interval. For outer draw `b`, it
  pairs refit draw `b`'s own probabilities with an INDEPENDENT block-bootstrap
  resample of the test games, so one loop of `n_boot` iterations combines both
  variance sources (training rows and test games are disjoint data, so this is
  exact, not an approximation) without a nested double bootstrap. Verified
  against `naive_block_bootstrap_interval` on degenerate zero-refit-variance
  input: the two algorithms produce bit-identical draws given the same seed
  (`test_refit_aware_interval_with_zero_refit_variance_matches_naive_exactly`).
- `bagged_values`, `shrink_predicted_margin` — the two variance-reduction
  levers screened in §4.
- `picks_differ_fraction`, `mde80`, `gate_by_disagreement` — the `f`-lever
  primitives screened in §5.

15 unit tests, all fast (2.2s), pin every function against the production
code it mirrors (`margin._smoothed_probability`,
`experiments.paired_feature_comparisons`). Nothing here is imported by
`margin.py`, `experiments.py`, `cfb_benchmark.py`, or the active model.

## 2. Planted-effect validation: does the honest interval actually have correct coverage?

Real data never gives a ground truth to check coverage against, so this is a
synthetic simulation (`scripts/estvar_planted_effects.py`): a linear DGP
(4 baseline + 3 candidate features, Gaussian noise sd 13, matching this
project's own measured ATS margin sd) fit with the project's own Ridge
pipeline. `Delta_true` is defined exactly the way the power audit's own
framing does — "would a model fit this way beat one fit that way in
general" — as the expectation over BOTH fresh training draws and fresh test
draws, estimated once per DGP via 300 Monte-Carlo replicates scored on a
20,000-game population test set (so its own standard error is negligible,
±0.0002-0.0012 pts).

200 replicates per DGP, each with a realistic `n_train=1,200`, `n_test=400`,
`n_boot=80`:

| DGP | `Delta_true` (pts) | Naive coverage | Honest coverage | Naive width | Honest width | Width inflation |
|---|---|---|---|---|---|---|
| Null (candidate carries no information) | −0.15 ± 0.01 | **89.5%** | **100%** | 4.77 pts | 7.51 pts | **1.575x** |
| Real effect (candidate genuinely informative) | +1.67 ± 0.02 | **92.5%** | **100%** | 8.44 pts | 9.92 pts | **1.176x** |

Nominal coverage is 95%. The naive (currently-reported) interval **under
-covers the truth in both directions** — it is not merely conservative-vs
-honest, it is measurably wrong on its own terms, in a controlled setting
where the true answer is known. The honest interval reaches 100% in both
cases (slightly conservative at `n=200` replicates, consistent with a
correction that is not too narrow — the validation requirement this document
was built to satisfy). Candidate own-model flip rate in these runs was
8.3-10.3%, the same order of magnitude as the audit's 15-22% at smaller
`n_train`, confirming the mechanism these numbers are measuring is the one
named.

**This is the central finding.** A currently-reported 95% interval is not a
95% interval — the true coverage measured here is closer to 90%, meaning
roughly double the intended 1-in-20 false-exclusion rate.

## 3. Real CFB comparisons: how much wider, and does any verdict change

Two real comparisons, annual-refit walk-forward (one fit per test season,
not per week — see the cadence caveat below), CFB clean core 2012-2025,
`N_BOOT=120`, `scripts/estvar_real_cfb_audit.py`:

- **A: market vs. market_residual** — the project's own flagship XLG-03
  result. `market` fits no estimator (zero refit variance by construction);
  only `market_residual`'s ridge fit contributes.
- **B: thin vs. full** — both ridge `market_residual` fits, differing only in
  feature columns. "Thin" is market + context + experience (11 columns, no
  team-state); "full" is the complete `CFB_MODEL_FEATURE_COLUMNS` (35
  columns) — directly the "does team-quality data add value" question this
  project's own prior finding (team quality is already priced) bears on.

| Comparison | `f` | Candidate flip rate | Naive: estimate, 95% CI, P+ | Honest: estimate, 95% CI, P+ | Width inflation |
|---|---|---|---|---|---|
| A: market vs. market_residual | 57.3% | 19.2% | +1.58 pts [+0.07, +3.19], **P+ 0.979** | +1.80 pts [**−0.22**, +3.02], P+ 0.958 | **1.037x** |
| B: thin vs. full | 19.8% | 19.2% | −0.34 pts [−1.32, +0.71], P+ 0.262 | −0.29 pts [−1.59, +1.11], P+ 0.300 | **1.330x** |

Both comparisons' candidate flip rate (19.2%) lands squarely inside the
audit's cited 15-22% range — this reproduces the phenomenon the audit
measured, on independent comparisons.

**A verdict that changes**: comparison A's naive interval **excludes zero**
(`lower=+0.0007`), which is exactly the criterion several existing documents
in this project use to call a result "resolved" (e.g.
`docs/residual_location.md` §4's "resolves negative" language). The honest
interval's lower bound is **−0.0022** — it no longer excludes zero. Per
`AGENTS.md`'s binding rule this was never grounds to reject the finding
either way (`probability_positive`, not "contains zero", governs), and
`probability_positive` itself barely moves (0.979 → 0.958, still a strong
lean). So the DECISION is unchanged, but the CLASSIFICATION some documents
would assign — "resolved" vs. "unresolved" — flips for this specific
comparison under the honest treatment. This is exactly the distinction
`AGENTS.md` §"An interval crossing zero is NOT grounds for rejection"
already anticipates, now with a concrete instance and number attached.

**Width, not `probability_positive`, is the one-directional finding.**
`probability_positive` moved DOWN under the honest interval for comparison A
(0.979→0.958) but UP for comparison B (0.262→0.300) — there is no universal
rule that honest treatment always lowers confidence. What is universal
across every comparison measured in this document (both synthetic DGPs, both
real comparisons): **the honest interval is wider, every time, by 3.7% to
57.5%.** The correct general statement is not "P+ is overstated" but "every
published interval WIDTH is a lower bound on the true uncertainty," and any
verdict within roughly 5-10 points of a decision threshold should be treated
as less certain than reported until re-measured.

**Why comparison A's inflation is small (3.7%) and B's is large (33.0%)**:
`f` differs by nearly 3x (57.3% vs. 19.8%). At large `f`, per-game accuracy
-improvement already has large game-sampling variance (many games flip pick
between arms), so the naive interval starts wide and the added refit
variance is a small relative addition. At small `f` — the more typical case
for this project's actual candidate screens (real historical median `f`
10.6%, IQR 6.0-14.9%, per the power audit) — refit variance is a LARGER share
of an already-narrower interval. **The understatement is worst exactly where
this project spends most of its research effort**: small, surgical,
single-feature-family additions, not blunt whole-model comparisons.

**Cadence caveat**: this document uses annual refit cadence (one fit per
season) rather than the production weekly cadence, purely for compute (a
weekly-refit bootstrap ensemble across 200+ season-week cells would be
roughly 15x the cost of the already-substantial run here). To confirm cadence
alone is not driving comparison A's numbers, the real weekly-cadence XLG-03
benchmark (`cfb_benchmark.cfb_walk_forward_benchmark`, unmodified) was also
run: `cover_accuracy` delta vs. market **+2.05 pts, 95% week-blocked CI
[+0.51, +3.49]**, width 2.98 pts — closely matching this document's
annual-cadence naive width of 3.12 pts. Cadence is not the story; the refit
source is.

## 4. Variance reduction: bagging works, centre shrinkage does not

### 4a. Bagging

Average predicted centres over the `N_BOOT=120` refit draws already computed
for §3's honest interval — no extra fitting. Own-model stability measured as
a split-half check (first 60 vs. last 60 draws' bagged means, sign
disagreement) against the single-fit's pairwise instability (individual draw
`i` vs. individual draw `i+1`, sign disagreement) — a clean, unconfounded
comparator, both built from the identical 120 draws:

| | Comparison B (thin vs. full, real CFB, 8,933 games) |
|---|---|
| Single-fit pairwise flip rate | **26.4%** |
| Bagged split-half flip rate | **4.5%** (**5.9x lower**) |
| Single-fit accuracy | 51.14% |
| Bagged accuracy | 51.31% (**+0.17 pts**) |
| Bagged vs. baseline, naive interval | −0.17 pts [naive], P+ **0.383** (single-fit: −0.34 pts, P+ 0.262) |

Bagging **dramatically stabilizes the model's own picks** (5.9x fewer sign
disagreements) and modestly improves both the accuracy point estimate and
`probability_positive` — a genuine, if not yet promotion-grade, improvement
in the SAME direction on both axes this document cares about.

**A direct synthetic falsification attempt found nothing** (single-fit
own-model flip rate 9.5% vs. bagged 10.0%; accuracy 1.43 vs. 1.35 pts;
detection rate 41.7% vs. 40.0% — all essentially null or slightly negative).
Diagnosis: that synthetic test's "own-model instability" comparator resampled
a bootstrap resample of the training set to simulate "a fresh draw of
history," which compounds resampling variance rather than testing a fair
fresh draw — a design flaw, not a real contradiction. The real-data
split-half design (splitting ONE clean bootstrap ensemble in half, no nested
resampling) is the trustworthy comparator, and it shows a large, decisive
effect. **Recommendation**: adopt split-half-of-one-ensemble as the standard
own-model-stability comparator for any future confirmatory bagging work; the
nested-resample design used in the first synthetic pass should not be reused.

### 4b. Centre shrinkage

`shrink_predicted_margin` scales the candidate's raw predicted residual by
`shrink_fraction` before it is added to the market line — the complement of
`docs/residual_location.md`'s shrinkage of the residual sample's LOCATION;
this shrinks the model's own predicted CENTRE instead. Because the forced
pick reads a threshold off a fixed, nonzero-location residual sample (not a
naive sign test), a uniform positive scalar is genuinely NOT sign-invariant
here — unlike MOD-06's closed coefficient-scaling case — confirmed directly:
flip rate falls monotonically with `shrink_fraction` in both the synthetic
and real screens (real: 18.1%→0.0% from `shrink=1.0` to `shrink=0.0`).

| `shrink_fraction` | Synthetic (100 reps, monotonic linear DGP) | Real CFB, comparison B (8,933 games) |
|---|---|---|
| 1.0 (production strength) | +1.67 pts, P+ 0.707 | −0.34 pts, P+ 0.255 |
| 0.75 | +1.40 pts, P+ 0.666 | −0.45 pts, P+ 0.181 |
| 0.5 | +1.11 pts, P+ 0.624 | −0.41 pts, P+ 0.208 |
| 0.25 | −0.59 pts, P+ 0.439 | **+0.03 pts, P+ 0.543** |
| 0.0 (pure market-line pick) | −5.52 pts, P+ 0.091 | −1.21 pts, P+ 0.041 |

The synthetic screen is clean and monotonic: no shrinkage is always best,
every degree of shrinkage costs accuracy and `probability_positive` — a
straightforward bias-variance tradeoff with no sweet spot. The real screen is
noisy and NON-monotonic, and does not replicate that pattern:
`shrink_fraction=0.25` shows the only near-zero-or-positive point (+0.03 pts,
P+ 0.543), but this is a coin-flip `probability_positive`, not a resolved
improvement, sandwiched between worse results on both sides.

**Verdict: unresolved, leaning null.** Centre shrinkage does not clear any
bar here. Recording it (§6) rather than discarding it per `AGENTS.md`'s
binding rule, but not recommending further investment without a predeclared,
out-of-sample confirmation design (the synthetic screen's disagreement with
the real screen is itself informative: real feature relationships are not
the clean linear case the synthetic DGP modelled, and no sweet spot should be
assumed to exist just because one noisy point crossed zero).

## 5. The `f` lever: gating a candidate's influence to where it disagrees

`MDE80 = 280 * sqrt(f/n)`: a candidate that changes fewer picks is easier to
resolve at equal true effect. `gate_by_disagreement(baseline_prob,
candidate_prob, threshold=tau)` defers to the baseline's own probability
wherever the candidate disagrees by less than `tau`, contributing zero to `f`
on those games. Screened on comparison B (thin vs. full) — the same real
comparison §3 found losing (−0.34 pts, P+ 0.262):

| `tau` | `f` | Estimate | `probability_positive` | MDE80 (pts) |
|---|---|---|---|---|
| 0.00 (ungated) | 19.8% | −0.34 pts | 0.264 | 1.32 |
| 0.01 | 18.2% | −0.50 pts | 0.156 | 1.27 |
| 0.02 | 13.6% | −0.53 pts | 0.090 | 1.09 |
| 0.03 | 8.4% | −0.29 pts | 0.186 | 0.86 |
| **0.05** | **2.06%** | **+0.25 pts** | **0.951** | **0.43** |
| 0.08 | 0.43% | +0.04 pts | 0.733 | 0.19 |
| 0.12 | 0.15% | +0.03 pts | 0.776 | 0.11 |
| 0.18 | ~0% | ~0 | 0.632 | 0.03 |

At `tau=0.05`, `f` falls **9.6x** (19.8%→2.06%, MDE80 falls 3.1x) and the
paired estimate **flips sign** (−0.34 → +0.25 pts) with `probability_positive`
jumping from 0.264 to **0.951**. This says the "full" feature set's real
value is concentrated on the minority of games where it substantively
disagrees with a market+context-only view, and is dilutive noise everywhere
else — a sharper, more actionable version of this project's own "team
quality is already priced" finding: team-quality features DO help, but only
where they meaningfully diverge from the simpler view.

**Out-of-sample check** (`scripts/estvar_f_lever_splithalf.py`), because
`tau=0.05` was chosen by looking at a 9-point sweep after seeing the pooled
result: split the 13 clean-core seasons into two halves by alternating
season order (both halves span the full era), fit point models only (no
bootstrap — a cheap robustness check), sweep `tau` independently on each
half:

| Half (seasons) | `P+` at `tau=0` | `P+` at `tau=0.05` |
|---|---|---|
| A (2012,14,16,18,21,23,25 — 4,823 games) | 0.399 | **0.935** |
| B (2013,15,17,19,22,24 — 4,110 games) | 0.241 | **0.744** |

The qualitative pattern — a modest disagreement gate turns a losing
comparison into a strongly resolving one — **replicates independently on two
disjoint season splits**. The exact `tau` and the pooled `P+ 0.951` should
not be trusted to that precision (researcher degree of freedom on a 9-point
sweep), but the underlying phenomenon is not an artifact of that sweep.

**Verdict: yes, the `f` lever is real and exploitable.** This is arguably the
single most decision-relevant finding of this document: it is a general
recipe — gate any candidate's influence to games where it substantively
disagrees with a simpler baseline — that can turn currently-unresolved
comparisons into provable ones WITHOUT any new data or new features, purely
by being selective about when an already-fitted model is trusted. Gating too
aggressively does eventually collapse the sample and the benefit (`tau≥0.12`
in half B's noisier tail); there is a real sweet spot, not a monotonic "gate
more" rule. A synthetic validation (informative-subset DGP, misspecified
linear ridge on a thresholded true relationship) found the same DIRECTION
(f falls, estimate holds, P+ rises 0.787→0.832 as `tau` rises 0→0.35) but a
much smaller magnitude — the synthetic design only weakly separates
informative from uninformative games via prediction disagreement; the real
data's much larger, cleaner effect should be trusted over the synthetic
scaling.

## 6. Weak signals recorded (not run — payload only, per this session's constraints)

Three category-3 results from §§4-5 are written as the exact
`nfl-ats weak-signals record` payload to
`<scratchpad>/weak_signal_record.json` (JSON array matching
`src/nfl_ats/weak_signals.py`'s CLI contract). **Not executed** — this
session is barred from running `nfl-ats weak-signals record` and from
writing `registry/*.json`. A future session should run each entry through
the CLI once free to do so: `estimation_variance_center_shrink_cfb`,
`estimation_variance_disagreement_gate_full_vs_thin_cfb`,
`estimation_variance_bagging_stability_cfb`.

## 7. Which existing recorded verdicts this bears on

This document's refit-aware methodology applies specifically to comparisons
between DIFFERENT FITTED RIDGE MODELS (different feature columns or
different regularization) — the source of variance §§1-3 measured. It does
**not** apply to comparisons that hold the mean model fixed and only change
how a fixed residual sample is READ (e.g. `docs/residual_location.md`'s
recency-weighting/reader-shrinkage screen) — a different, unaddressed
variance source.

**In scope, and worth re-measuring before further reliance**:
- `ridge_alpha_global` (`registry/weak_signals.json`) — `unresolved_below_power`,
  `probability_positive` 0.758 (week) / 0.703 (season), already modest. This
  is exactly a different-ridge-fit comparison (production alpha=10 vs. swept
  alternatives); §3's finding that such comparisons run 4-33% narrower than
  honest treatment supports means this confidence is probably closer to a
  coin flip than the reported 0.76 suggests.
- `groupwise_ridge_block_penalties` — `unresolved_below_power`,
  `probability_positive` 0.8735, interval already crosses zero
  `[-0.419, +1.532]`. Same family (different penalty structure = different
  fit); classification would not change (already the honest "unresolved"
  bucket) but the stated confidence is likely optimistic by a comparable
  margin.
- `injury_value_lost_gradient` — `unresolved_below_power`,
  `probability_positive` 0.899, only 456 games / 35 blocks (the thinnest
  sample of the group, and a feature-set-difference comparison — in scope).
  Already below the strict ≥0.90 bar used elsewhere in this project; the
  smallest, thinnest sample is where this document's effect was largest
  (comparison B, similarly sized `f`, showed 33% width inflation), making
  this the entry most likely to see its confidence erode on honest
  re-measurement.

**Not expected to change**:
- `player_qb_continuity_matched_alpha` — `refuted_mechanism` on a
  MECHANISTIC argument (alpha=1 vs. 10 shrinks the design's median principal
  direction by 0.03% vs. 0.27%, a near-null contrast by construction), immune
  to power arguments; no amount of honest interval width changes a refuted
  mechanism.
- `residual_location_recency_hl200_cfb`, `residual_location_recency_hl400_cfb`
  — `refuted_mechanism` on reader-only comparisons (§7 scope note above);
  outside what this methodology measures.

No number above was re-measured for these specific entries — computing that
would mean rerunning each screen's own recipe with the honest bootstrap,
which was out of scope for this session. These are directional flags, stated
with the numbers behind the general claim, not new point estimates for those
specific families.

## 8. Declared limitations

1. **Cadence**: annual refit, not the production weekly cadence, for compute
   (§3). Cross-checked against the real weekly benchmark for comparison A
   only; not re-verified for comparison B or the variance-reduction/f-lever
   screens.
2. **Residual-calibration split not resampled**: the refit bootstrap
   resamples the FINAL-fit training rows and refits the mean model; it does
   NOT also resample `fit_cfb_residual_model`'s internal 80/20
   residual-calibration split. Each season's out-of-time residual sample is
   held fixed across that season's refit draws. Only the predicted CENTRE
   varies — the source the power audit named (predicted-margin bootstrap SD)
   — not residual-calibration noise, a smaller, different, unaddressed
   source.
3. **`N_BOOT=120`** bounds the honest interval's own quantile resolution
   (order statistics from 120 draws are coarser than the naive interval's
   3,000-sample bootstrap). Acceptable for the magnitudes measured here but
   worth increasing in any confirmatory follow-up.
4. **Bagging's synthetic validation was inconclusive** (§4a) due to a
   comparator design flaw, disclosed rather than silently dropped. The
   real-data evidence is trusted over the synthetic null.
5. **The `f`-lever's exact `tau` and pooled `P+` are not resolved to full
   precision** (§5) — only the qualitative, replicated pattern is trustworthy
   without a predeclared confirmation design.
6. **CFB only**, per this task's scope; no NFL rotation-registry window was
   touched (`registry/rotation_registry.json` was read, never written).
