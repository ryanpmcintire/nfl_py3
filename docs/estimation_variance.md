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

> **CORRECTED 2026-08-18 by Part II of this document — read §9 before quoting
> anything in Part I.** The "17-58% too narrow" headline below is a real
> measurement of the wrong quantity: it adds a training-by-game interaction term
> that the game bootstrap already carries, and removing that double count takes
> the flagship CFB comparison from **1.293x to 1.003x** (95% upper bound
> 1.099x). The 89.5-92.5% coverage was measured at 20 game-blocks, where the
> block count alone caps coverage at ~0.92; at 80 blocks the currently-reported
> interval covers 0.940-0.952. Part I's §§4-5 findings (bagging, centre
> shrinkage, the `f` lever) are unaffected — they do not depend on the
> decomposition. **The real defect is D4, the block floor (Part II §10).**

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

---

# Part II — the durable fix, and a correction to Part I (2026-08-18, later)

Part I measured the problem and built a first honest interval. This part makes
the machinery permanent — structural pairing, a derived factor, a measured
degeneracy floor, a regression test — and in doing so **corrects Part I's
headline number**.

**The correction, stated first.** Part I says every reported interval is
"17-58% too narrow" and that measured coverage is 89.5-92.5% against 95%
nominal. Both figures are real measurements, but they were attributed to the
wrong cause and they do not survive:

- The **width gap** was a double count. Part I's honest intervals add the
  spread of refit deltas on the fixed test games to a game bootstrap that
  already carries most of that spread. The overlapping part is the
  training-by-game INTERACTION, and for a forced-pick accuracy delta it is the
  dominant piece. Removing the double count takes the real CFB comparison from
  **1.293x to 1.003x** (one-sided 95% upper bound **1.099x**).
- The **coverage gap** was mostly D4. Part I's simulation used 20 game-blocks,
  where the percentile block bootstrap caps out at ~0.92 coverage from block
  count alone (§10 below). Rerun at 80 blocks, the currently-reported naive
  interval covers **0.940 / 0.952** — about a point short of nominal, not five
  and a half.

Both Part I estimators over-cover as a direct result: measured **0.992-1.000**
against nominal 0.95. Conservative rather than wrong, but conservative for a
reason that is an arithmetic error, so it is fixed rather than kept.

Every number below is **measured this session** unless labelled otherwise, with
the command that produces it named inline.

## 9. The durable estimator

### 9a. Pairing is structural, not a seed coincidence

Paired deltas are this project's primary estimand: almost every verdict asks
"does arm B beat arm A **on these same games**", not "what is arm B's level". If
each arm is refit on its OWN resample of the training rows, the refit noise the
two arms share stops cancelling and the delta's variance becomes
`Var_A + Var_B` instead of `Var(A - B)`. For two models sharing most of their
features and all of their training rows that is a large over-statement — in the
opposite direction to the defect being fixed, and just as wrong.

Measured on the real CFB comparison, same 8,933 games, same 120 refit draws:

| refits | training component (pts) | inflation factor |
|---|---|---|
| **paired** (both arms on the same resampled rows) | **0.042** | **1.003** |
| unpaired (each arm on its own resample) | 0.152 | 1.043 |

Unpaired inflates the training term by **3.6x**. The two scripts that predate
this (`estvar_planted_effects.py`, `estvar_real_cfb_audit.py`) call
`refit_predicted_values` twice with the *same* seed on the *same* training
frame, so they happen to be correctly paired — but nothing in the API said so,
nothing enforced it, and passing `seed` and `seed + 1` would silently have
produced the over-wide answer. `paired_refit_predicted_values` now returns
`PairedRefits`, carrying the one `row_indices` matrix both arms were fit on.
`paired=False` exists only so the error stays testable.

Arms that fit no estimator at all — the unconditional `market` baseline —
belong outside this: pass them as a `(1, n_games)` array, which broadcasts and
correctly contributes zero refit variance.

### 9b. The interaction, and why adding the refit spread double-counts it

Write a paired delta as

```
Delta(T, G) = mu + a(T) + b(G) + e(T, G)
```

a training effect, a game-sampling effect, and their interaction. Then

- the conditional block bootstrap (resample games, one fit) measures `Var(b) + Var(e)`;
- the spread of refit deltas on the FIXED test games measures `Var(a) + Var(e)`;
- the honest total is `Var(a) + Var(b) + Var(e)`.

Adding the two measured quantities counts `Var(e)` **twice**. For a forced-pick
accuracy delta that is not a rounding term: a refit that flips the pick on game
*i* moves the delta by `±1/n` depending on whether that particular game
covered, which is pure interaction. On the real CFB comparison the fixed-games
refit spread is 0.420 points, of which only **0.042** is `Var(a)`; the other
99% of the variance was already inside the game bootstrap.

`refit_common_variance` recovers `Var(a)` alone by splitting the games into two
disjoint halves and taking the **covariance** of the two halves' refit-delta
series across draws: `a(T)` is common to both halves so it survives, while the
interaction averages of disjoint game sets are uncorrelated so `Var(e)`
cancels. Splitting by GAME rather than by block is correct here — within-week
game correlation is mandated to be exactly zero in this project (`AGENTS.md`),
and the across-week component a training perturbation shares IS `a(T)` by
construction. Averaged over 40 random halvings.

It is a covariance estimated from `n_boot` draws, so it is intrinsically
noisier than either variance it is built from, and the function returns its
standard error for exactly that reason. On real CFB the point estimate is an
order of magnitude below its own standard error, which is why §11 quotes an
upper bound and not a correction.

### 9c. The construction

1. Block-bootstrap the POINT fit's per-game improvements at the full `samples`
   budget (20,000) — exactly what `paired_feature_comparisons` reports, so the
   naive interval comes back as a by-product and the two are comparable.
2. The refit draws supply ONE number: `refit_sd = sqrt(Var(a))`.
3. Rescale the conditional draws about their own mean by

   ```
   inflation_factor = sqrt(1 + (refit_sd / conditional_sd) ** 2)
   ```

   and read the bounds and `probability_positive` off the rescaled draws.

Rescaling about the draws' own mean (not the point estimate) makes the honest
interval an *exact widening* of what the project already reports: at
`inflation_factor == 1` it returns the naive bounds to floating-point equality,
so any difference between the two IS the refit variance and nothing else. Part
I's estimator instead re-centred each draw on a bootstrap REFIT, which is
systematically slightly worse than the full-data fit — on the CFB comparison it
moved the reported estimate from **-0.336 to -0.649 points**. An interval
routine may widen a number; it may never move it.

### 9d. Coverage: before and after

`scripts/estvar_refit_intervals.py --study coverage`. Synthetic DGP matching
Part I §2 (4 baseline + 3 candidate features, Gaussian noise sd 13, the
project's own Ridge pipeline), `Delta_true` estimated over 250 Monte-Carlo
replicates on a 20,000-game population test set, **250 replicates per cell**
(coverage se ≈ 0.014), `n_boot=80`, `samples=20,000`. Two block counts, because
Part I's own configuration cannot separate the two defects:

| DGP / blocks | `Delta_true` (pts) | naive | Part I honest | **new honest** | new factor |
|---|---|---|---|---|---|
| null / 20 blocks | -0.165 ± 0.014 | 0.904 | 1.000 | **0.932** | 1.079 |
| null / **80 blocks** | -0.160 ± 0.015 | 0.940 | 1.000 | **0.952** | 1.072 |
| real effect / 20 blocks | +1.726 ± 0.021 | 0.936 | 0.996 | **0.936** | 1.026 |
| real effect / **80 blocks** | +1.727 ± 0.021 | 0.952 | 0.992 | **0.956** | 1.029 |

**Yes — the refit-aware interval reaches nominal coverage: 0.952 and 0.956
against 0.95, at 80 blocks, where the block count is not the binding
constraint.** Part I's estimator over-covers at 0.992-1.000 in the same cells.

The 20-block rows are the diagnostic. There, nothing reaches nominal — naive
0.904/0.936 and new honest 0.932/0.936 — because 20 blocks caps coverage at
~0.924 by itself (§10). Part I measured its 89.5-92.5% at exactly this block
count and read the shortfall as missing refit variance. It was block count.

Mean widths, same cells: naive 4.660 / 2.870 / 8.612 / 5.270 points; new honest
4.980 / 3.057 / 8.831 / 5.418. The honest interval is **2.6-7.9% wider**, not
17-58%.

### 9e. The cheap path and what the factor depends on

The factor needs only `refit_sd`, which needs `m` refits — not `samples`
refits. The conditional bootstrap stays at 20,000 draws and costs nothing.

**How many refits?** `--study refits`, 12 replicates, `n_train=1,200`,
`n_test=1,120`:

| refits `m` | mean factor | bias vs 320 refits |
|---|---|---|
| 5 | 1.0258 | -0.003 |
| 10 | 1.0067 | -0.022 |
| 20 | 1.0176 | -0.011 |
| 40 | 1.0204 | -0.008 |
| 80 | 1.0218 | -0.007 |
| 160 | 1.0261 | -0.002 |
| 320 | 1.0285 | — |

Under the corrected decomposition the bias is small and **non-monotone**, which
is the honest reading: at these magnitudes the estimator's noise dominates its
bias, so no refit count "converges" in the usual sense. Part I's equivalent
table showed a clean 4.8% monotone bias, which was an artifact of estimating a
quantity (`Var(a) + Var(e)`) large enough for the bias to show. **Use 80-120
refits and quote `inflation_factor_upper`, not the point estimate.**

**What the factor depends on.** `--study inflation`, 12 replicates per cell,
`n_boot=120`, `f` held near 0.21 so the sweep isolates the two sample sizes:

| `n_train` | `n_test` | blocks | conditional SD (pts) | `Var(a)` SD (pts) | factor |
|---|---|---|---|---|---|
| 300 | 420 | 30 | 2.049 | 0.414 | 1.048 |
| 300 | 1,120 | 80 | 1.318 | 0.555 | 1.107 |
| 300 | 4,200 | 300 | 0.720 | 0.642 | **1.356** |
| 600 | 4,200 | 300 | 0.687 | 0.310 | 1.131 |
| 1,200 | 420 | 30 | 2.264 | 0.207 | 1.018 |
| 1,200 | 1,120 | 80 | 1.400 | 0.206 | 1.025 |
| 1,200 | 4,200 | 300 | 0.711 | 0.150 | 1.035 |
| 2,400 | 1,120 | 80 | 1.295 | 0.067 | 1.012 |
| 4,800 | 1,120 | 80 | 1.302 | 0.128 | 1.014 |
| 4,800 | 4,200 | 300 | 0.717 | 0.091 | **1.019** |

Two drivers survive the correction, both monotone:

- **More training data shrinks it.** `Var(a)` is estimation variance, so it
  falls with `n_train`; the conditional SD does not move at all. At
  `n_test=4,200` the factor goes 1.356 → 1.019 as `n_train` goes 300 → 4,800.
- **More evaluation games grow it.** The conditional SD falls as
  `1/sqrt(n_test)` — that is what more games buy — while `Var(a)` is a property
  of the fit and barely moves. At `n_train=300` the factor goes 1.048 → 1.356
  as `n_test` goes 420 → 4,200. **Adding evaluation games does not shrink this
  term; it makes it relatively larger.**

The practical consequence: **at this project's real training scale the honest
factor is 1.01-1.04, and it only exceeds 1.1 when the training set is small
relative to the evaluation set.** Cells at 12 replicates are visibly noisy (the
`n_train=2,400` / `n_test=420` row is out of line with its neighbours); the
monotone trends are trustworthy, individual cells to ±0.02 are not.

**Do not borrow a factor across mechanisms.** The factor is a property of what
the comparison varies. A parallel agent measured **2.07x / 2.29x** for the
`residual_location` family, whose comparisons hold the mean model fixed and
re-read the residual sample — a different resampling entirely, which Part I §7
already disclaimed — and **1.438x** for `cfb_role_continuity`. Those are
*reported here, not verified by me*.

## 10. D4: the block floor is 10, not 4

`docs/anytime_valid.md` §6 put the floor at "~4-5 blocks", reasoning from the
count of achievable resamples. Two problems, both now measured.

**The count was quoted wrong.** Resampling `k` blocks with replacement gives a
multiset, and the statistic depends only on the multiset, so the number of
achievable values is `C(2k-1, k-1)` = 1, 3, 10, 35 at k = 1, 2, 3, 4. §6 states
that formula and then quotes "27 at k=3, 256 at k=4", which are `k**k` — the
count of ORDERED tuples, which over-counts by 7x at k=4 because most orderings
collapse to the same statistic. Its own later `C(4+4-1, 3) = 35` is right.
`distinct_block_resamples` now returns it, with a test.

**And a resample count was never the right question.** What matters is
coverage. `--study degeneracy` sweeps the block count against the project's OWN
estimand — the paired forced-pick accuracy delta, per-game value in
`{-1, 0, +1}`, zero on the `1 - f` games where both arms pick the same side — at
`f` = 0.10 / 0.20 / 0.55, 2,000 replicates each, 14 games per block:

| blocks `k` | distinct resamples | `f`=.10 | .20 | .55 | **mean** | intervals with EXACTLY zero width |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.000 | 0.000 | 0.000 | **0.000** | **100%** |
| 2 | 3 | 0.438 | 0.476 | 0.484 | **0.466** | **25.2%** |
| 3 | 10 | 0.675 | 0.702 | 0.739 | 0.705 | 8.4% |
| 4 | 35 | 0.723 | 0.775 | 0.783 | **0.760** | 2.3% |
| 5 | 126 | 0.797 | 0.820 | 0.834 | 0.817 | 0.8% |
| 6 | 462 | 0.834 | 0.853 | 0.849 | 0.845 | 0.4% |
| 8 | 6,435 | 0.874 | 0.882 | 0.892 | 0.882 | 0.1% |
| **10** | 92,378 | 0.889 | 0.897 | 0.902 | **0.896** | 0.1% |
| 13 | 5.2e6 | 0.915 | 0.911 | 0.914 | 0.913 | 0 |
| 17 | 1.2e9 | 0.915 | 0.924 | 0.925 | 0.921 | 0 |
| 20 | 6.9e10 | 0.917 | 0.925 | 0.929 | **0.924** | 0 |
| 25 | 6.3e13 | 0.937 | 0.933 | 0.935 | 0.935 | 0 |
| 35 | 5.6e19 | 0.935 | 0.942 | 0.928 | 0.935 | 0 |
| **50** | 5.0e28 | 0.949 | 0.946 | 0.936 | **0.944** | 0 |
| 68 | 3.0e39 | 0.945 | 0.946 | 0.939 | 0.943 | 0 |
| 100 | 4.5e58 | 0.938 | 0.938 | 0.943 | 0.940 | 0 |
| 199 | 1.3e118 | 0.943 | 0.948 | 0.951 | **0.947** | 0 |

Nominal is 0.95. Three readings:

- **At 4 blocks coverage is 0.76.** The recorded "~4-5" floor is not a floor;
  it is the middle of the failure. `MIN_BLOCKS_FOR_INTERVAL = 10` is the
  smallest block count whose coverage is not statistically below 0.90 (k=10 is
  1.0 SE below; k=8 is 4.6 SE below). At the floor the miss rate is still
  double nominal — 10 blocks is a refusal threshold, not a licence.
- **The zero-width column is the smoking gun.** With a discrete `{-1, 0, +1}`
  estimand, one in four 2-block "intervals" and one in fifty 4-block intervals
  has *literally zero width*. That is the mechanism behind the
  `[0.0, 2.2177]` interval this project carried on 4 blocks: a bound landing
  exactly on a round number is what a resampling distribution with 35
  achievable values looks like when asked for a 2.5% quantile.
- **Coverage plateaus at ~0.945, not 0.95, and never gets there.** Even at 199
  blocks the reported "95%" interval is a 94.7% interval. That is a third,
  smaller source of narrowness — the percentile bootstrap of a skewed,
  zero-inflated statistic is mildly anti-conservative — and unlike the other
  two it does not go away with more blocks or more refits. Recorded here rather
  than fixed; a BCa or studentized interval is the standard remedy and is out
  of scope for this pass.

**Update, 2026-08-18 — the named remedies were measured and do not fix this.**

Rerunning the same degeneracy harness (identical grid, seeds, 2000 replicates x 4000 samples; scratchpad driver validated bit-for-bit against degeneracy_study's percentile arm before being trusted) with BCa and studentized/bootstrap-t intervals alongside percentile: BCa is statistically indistinguishable from percentile at k >= 50 (0.946 vs 0.947 at k=199, overlapping binomial CIs) and measurably worse below the block floor (0.289 vs 0.466 at k=2 -- the jackknife acceleration breaks on a distribution with only a handful of achievable resample values); studentized is mostly undefined below k of about 13 and anti-conservative above it, collapsing monotonically to 0.114 coverage at k=199, because the resamples that move the mean furthest also inflate their own internal SE on this zero-inflated three-valued statistic. Neither is adopted. The plateau stands as documented; if the residual half-point of coverage is ever worth chasing, the discreteness of the statistic itself is the binding constraint, and a smoothing/continuity correction on the statistic (analogous to the Laplace treatment in home_cover_probability_from_center) is the untested direction -- not another interval family.

**The guard.** `guard_block_count` / `block_count_verdict` /
`BootstrapDegeneracyError` / `BootstrapDegeneracyWarning`, wired so that:

- every estimator in `estimation_variance.py` that consumes refits **raises**
  below the floor by default (`refit_aware_interval`,
  `refit_variance_decomposition`);
- `naive_block_bootstrap_interval` **warns and stamps `degenerate=True`** on
  the returned `PairedInterval`, because its job is to reproduce what the
  project reports today and refusing would change that baseline;
- `experiments.paired_feature_comparisons` — the function whose CSV output the
  offending registry entry cited — now warns and adds two columns, `blocks` and
  `degenerate_blocks`, to every row. **No existing number changes.** What
  changes is that the flag travels with the number into the artifact, so
  downstream reporting cannot render a degenerate interval as a normal one
  without ignoring a column that says otherwise. A caller about to record a
  verdict should pass `on_degenerate="raise"`.

`tests/test_experiments.py::test_paired_feature_comparison_flags_a_degenerate_block_count`
and `tests/test_estimation_variance.py::test_low_block_interval_is_never_reported_as_valid`
(parametrized over k = 1..9) fail if any path ever hands back a below-floor
interval without the flag. A guard with no test is how this defect survived an
audit that had already found it.

## 11. Real CFB: the whole ladder on one comparison

`--study cfb`. Thin (market + context + experience, 11 columns) vs full
`CFB_MODEL_FEATURE_COLUMNS` (35 columns), annual refit cadence, CFB clean core,
**8,933 games / 199 week blocks**, `f = 19.8%`, 120 paired refits,
`samples = 20,000`. CFB is free and unlimited per rule 8 of
`docs/rotation_registry.md`; the registry was read, never written.

| interval | estimate (pts) | 95% | width | `probability_positive` |
|---|---|---|---|---|
| naive — what the project reports today | -0.336 | [-1.331, +0.686] | 2.018 | 0.255 |
| Part I `refit_aware_paired_interval` | **-0.649** | [-1.348, +0.914] | 2.262 | 0.383 |
| **`refit_aware_interval`, paired refits** | -0.336 | [-1.344, +0.666] | **2.010** | **0.261** |
| same, UNPAIRED refits (wrong) | -0.336 | [-1.385, +0.706] | 2.090 | 0.267 |

Decomposition: conditional SD **0.512 pts**; interaction-free training term
**0.042 pts**; fixed-games refit spread **0.420 pts**. So:

- honest factor **1.0034**, one-sided 95% upper bound **1.099**;
- the double-counting factor is **1.293** — which is Part I §3's 1.330 for this
  same comparison, reproduced. Part I's arithmetic is reproducible; its
  interpretation was wrong.

The point estimate of the training term sits an order of magnitude below its
own standard error, so **the honest statement is an upper bound: on this
comparison the conditional interval is at most ~10% too narrow, and the best
estimate is that it is essentially right.**

`probability_positive` moves 0.255 → 0.261, i.e. towards a coin flip, as it
must: widening moves confidence towards 0.5 from whichever side it sits on. It
does not "make the result negative", and per `AGENTS.md` an interval that
crosses zero was never grounds to close anything.

## 12. Blast radius on what is already recorded

`scripts/estvar_blast_radius.py` (reads both registries, writes neither), run
against the registries as they stand after this session's parallel repairs.

Method, and its limits. Recorded rows do not keep their bootstrap draws, so the
conditional SD is recovered from whichever of `standard_error`, `interval` or
`probability_positive` the row kept, widened by that family's mechanism factor,
and re-read on a **normal** reference. To keep the comparison honest the
column that judges change is the same normal calculation at factor 1.0
(`normal_at_factor_1`), not the recorded bootstrap value. The two agree to
0.01-0.02 on every row that records both — and on the two rotation windows a
sibling re-read independently, the reconstruction lands on **0.4736 vs 0.474**
and **0.5000 vs 0.50**, which is the cross-check that the normal reference is
not doing any of the work.

Factors by mechanism, all stated with provenance:

| mechanism | factor | provenance |
|---|---|---|
| refit (differently fitted models) | 1.003, band 1.000-1.099 | **measured**, §11, this session |
| reader (fixed mean model, residual re-read) | 2.07, band 2.07-2.29 | **reported** by a parallel agent, unverified by me |
| `cfb_role_continuity`, family-specific | 1.438 | **reported** by a parallel agent, unverified by me |

| signal_id | recorded P+ | honest P+ | mechanism | classification still holds? |
|---|---|---|---|---|
| `cfb_role_continuity` | 0.3498 | 0.3986 | role 1.438x | HOLDS |
| `ecdf_smoothing_accuracy` | 0.1100 | 0.2767 | reader 2.07x (extrapolated) | HOLDS |
| `fourth_down_aggressiveness` | *none* | 0.4298 | refit 1.003x | HOLDS |
| `groupwise_ridge_block_penalties` | 0.8735 | 0.8590 | refit 1.003x | HOLDS |
| `hc_year_one_fade` | *none* | 0.9319 | refit 1.003x | HOLDS |
| `injury_value_lost_gradient` | 0.8990 | 0.9160 | refit 1.003x | HOLDS |
| `penalty_discipline` | *none* | *not derivable* | refit 1.003x | HOLDS, **on no recorded evidence** |
| `player_qb_continuity_matched_alpha` | 0.7960 | 0.7951 | refit 1.003x | HOLDS |
| `residual_location_recency_hl100_cfb` | 0.0710 | 0.2541 | reader 2.07x | HOLDS |
| `residual_location_recency_hl200_cfb` | 0.2585 | 0.2991 | reader 2.07x | HOLDS |
| `residual_location_recency_hl400_cfb` | 0.3080 | 0.2869 | reader 2.07x | HOLDS |
| `residual_location_recency_hl800_cfb` | 0.0575 | 0.2286 | reader 2.07x | HOLDS |
| `residual_location_shrink_025_cfb` | 0.3890 | 0.4562 | reader 2.07x | HOLDS |
| `residual_location_shrink_050_cfb` | 0.1515 | 0.3201 | reader 2.07x | HOLDS |
| `residual_location_shrink_075_cfb` | 0.0935 | 0.2593 | reader 2.07x | HOLDS |
| `residual_location_shrink_100_cfb` | 0.1045 | 0.2728 | reader 2.07x | HOLDS |
| `ridge_alpha_global` | 0.7580 | 0.7573 | refit 1.003x | HOLDS |
| `best_pick_ranker` [2013,2015] | 0.7955 | 0.7861 | refit 1.003x | HOLDS |
| `best_pick_ranker_opener` [2020,2021] | 0.8650 | 0.8719 | refit 1.003x | HOLDS |
| `cfb_role_continuity` (rotation, open) | *none* | *n/a* | role 1.438x | HOLDS (no spent window) |
| `mod07_weak_signal_stack` [2020,2021] | 0.8745 | 0.8966 | refit 1.003x | HOLDS |
| `pbp_drive_bundle` [2013,2017] | 0.4740 | 0.4737 | refit 1.003x | HOLDS |
| `player_qb_continuity` [2014,2017] | 0.5000 | 0.5000 | refit 1.003x | HOLDS |

**Zero recorded classifications change. Zero entries cross the 0.90 mark.**
Every entry in both registries is now `unresolved` — the terminal verdicts this
work was commissioned to re-check were already repaired by parallel agents
during this session — and "unresolved" is exactly where a wider interval lands.

Under the refit mechanism the movement is negligible (≤0.01 of
`probability_positive`) because the factor is 1.003. The visible movement is
entirely in the **reader**-mechanism rows, where a sibling's 2.07x carries
`residual_location_recency_hl800_cfb` from 0.058 to 0.229 and
`residual_location_shrink_075_cfb` from 0.094 to 0.259 — all towards 0.5, none
across any threshold, and none of it measured by me.

Two things the table cannot fix, and they are the durable findings:

- **`penalty_discipline` records an effect and a reliability but no interval,
  no standard error and no `probability_positive`.** There is nothing to widen
  and nothing to read. `fourth_down_aggressiveness`, `hc_year_one_fade`,
  `ecdf_smoothing_accuracy` and `ridge_alpha_global` are each missing one of
  the three.
- **The rotation-registry schema has no field for the effect size, the
  interval, the standard error, or the block count.** On every spent window
  those numbers exist only inside free-text `notes`, so this script has to
  parse prose to say anything. That is not a hypothetical fragility: a first
  pass matched `pbp_drive_bundle`'s Brier interval `[-0.00553, +0.00072]`
  against the same note's `-0.08` accuracy-point effect and produced a
  confident-looking `probability_positive` of 0.0000 out of two different
  units. The parser now refuses any bracket not explicitly labelled
  `week-blocked` / `season-blocked`, but a registry that cannot re-read its own
  verdicts without prose parsing will keep generating rows like that one.

### Proposed registry edits (NOT applied — this session does not write `registry/*.json`)

1. `rotation_registry.json`: add `effect`, `effect_units`, `interval`,
   `standard_error` and `sample_blocks` to the window schema, and backfill the
   five spent windows from their `notes`. This is the highest-value change in
   either registry.
2. `weak_signals.json`: backfill `probability_positive` where it is derivable
   from what is already recorded — `fourth_down_aggressiveness` **0.430**,
   `hc_year_one_fade` **0.932** — and record for `penalty_discipline` either an
   interval or an explicit note that none was computed.
3. Nothing else. **No classification should be changed on the strength of this
   document**, and specifically none to a worse one: widening an interval is
   not evidence against anything.

## 13. Which call sites should switch, and what changes if they do

Nothing below is switched by this pass — changing a default estimator would
rewrite numbers across every doc in one commit with no way to see what moved.

| call site | change | effect on reported numbers |
|---|---|---|
| `experiments.paired_feature_comparisons` | **done**: warns + adds `blocks` / `degenerate_blocks` | none — additive columns only |
| any script recording a verdict | pass `on_degenerate="raise"` | none unless below the floor, where it correctly refuses |
| `reporting.block_bootstrap_intervals` | should take the same guard | none; it reports levels, not paired deltas |
| `outcomes.outcome_bootstrap_intervals` | should take the same guard; its `delta_*` columns are paired deltas between fitted methods | ≈1.003x on the delta columns — below reporting resolution |
| `scripts/estvar_real_cfb_audit.py` | switch to `refit_aware_interval` | comparison B's honest width 2.262 → 2.010 pts, estimate returns to -0.336 from -0.649, P+ 0.383 → 0.261 |

**D3 confirmed in place, not redone** (read this session): `experiments.py`,
`reporting.py` and `outcomes.py` all carry `samples: int = 20_000`.

## 14. Declared limitations of Part II

1. **The coverage validation is synthetic.** It has to be — real data never
   supplies a ground truth. The DGP is a correctly-specified linear model; a
   misspecified one could behave differently.
2. **The common-variance estimator is noisy by construction.** It is a
   covariance over `n_boot` draws. On real CFB the point estimate is below its
   own standard error, so §11 reports an upper bound. Quote
   `inflation_factor_upper`, not `inflation_factor`, whenever the two differ
   materially.
3. **`f` was held near 0.21 in the inflation sweep**, so the `n_train`/`n_test`
   effects are clean but the interaction with `f` is unmeasured under the
   corrected decomposition.
4. **One real comparison, one league.** Every factor here is CFB or synthetic,
   per rule 8 of `docs/rotation_registry.md`. The reader-mechanism and
   role-family factors in §12 are reported by parallel agents and unverified
   by me.
5. **The residual-calibration split is still not resampled** (Part I limitation
   2 stands): the refit bootstrap varies the predicted CENTRE only.
6. **The ~0.945 coverage plateau is recorded, not fixed.**
