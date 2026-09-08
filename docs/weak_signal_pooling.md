# The weak-signal evidence machinery, and four defects fixed 2026-09-08

Four defects were found in the arithmetic that turns bootstrap draws into
recorded evidence (`docs/audit_20260908.md`, findings B1–B4). They are
documented together because they are not four unrelated bugs: **every one of
them pushed in the same direction, converting measurements that said nothing
into resolved-looking negatives.** That is precisely the failure mode
`AGENTS.md`'s binding invariant exists to prevent, so fixing them is
correctness work on the instrument, not tuning.

Nothing in this work closed a line of work, reclassified a signal, or rewrote
a recorded number. Where a stored value is now known to have been produced by
a defective convention, it is **flagged for re-measurement** and left alone.

---

## D1. The pooling command could not run at all

`_binomial_two_sided_p` computed each outcome's probability as
`math.comb(total, k) * 0.5**total`. `math.comb` returns an exact int; `0.5**n`
underflows to `0.0`; and converting the int operand to float overflows above
~1.8e308.

**Measured:** fine at n=1020, `OverflowError` at n≥1030. The eligible NFL pool
is **n=1489**, so `nfl-ats weak-signals pool` — the command `AGENTS.md` tells
every session to re-run for the current number — was unexecutable. That is why
the pooled headline in `AGENTS.md` sat at a 2026-09-05 reading.

**Fix.** `nfl_ats.evidence_conventions.binomial_two_sided_p`. The construction
is unchanged (sum every outcome no more likely than the observed one). Two
things moved:

- For a fair coin the pmf is symmetric about `n/2`, so "no more likely than
  observed" is the **exact integer predicate** `|2k − n| ≥ |2·observed − n|`.
  No floating-point tolerance is involved, so there is no tie-inclusion
  ambiguity.
- The surviving terms are summed from `math.lgamma` in log space, where every
  term is a probability in [0, 1] and cannot overflow.

**Verified.** Reproduces `AGENTS.md`'s fixture `327 / 628 → 0.31846963`
exactly, and matches `scipy.stats.binomtest` on all 499 tested
`(favourable, total)` pairs including n=1489. (scipy is only a transitive
dependency here, via scikit-learn, so it is used to verify, not to implement.)

This also fixed a **second, latent bug**: the old absolute tolerance
`pmf(k) <= observed + 1e-12` swept in every outcome whose probability was below
1e-12, so deep-tail p-values were overstated by orders of magnitude
(e.g. `p(0, 51)` returned 1.18e-12 instead of 8.88e-16).

---

## D2. `probability_positive` charged the zero atom against the candidate

Every screen reports `probability_positive`: the share of resamples in which
the candidate beat the baseline. It was computed as `np.mean(draws > 0.0)` — a
**strict** inequality — at ~100 sites.

A paired accuracy delta is `(candidate wins − baseline wins) / games` over the
handful of games where the two arms *disagree*. That statistic has a large atom
of probability at exactly zero, and in the limiting case a candidate making
**identical picks on every game** puts all of its mass there. Under the strict
`>` such a candidate records `probability_positive = 0.0` — the strongest
negative the scale can express, awarded to a no-op.

**Measured:** **115** such rows sit live in the NFL `accuracy_points` pool;
**346** across the whole registry (`effect` exactly 0.0, `interval` exactly
`[0, 0]`, `P+` exactly 0.0 — the three conditions coincide exactly).

**Convention adopted.** The zero atom is split evenly, because "no difference
at all" is evidence for neither arm:

```
P+ = P(draws > 0) + 0.5 · P(draws == 0)
```

A pure no-op scores **0.5** ("this told us nothing"), and a candidate is pushed
below 0.5 only by resamples in which it actually lost. With no ties the new and
old conventions agree exactly, so no real result is softened.

**Independent corroboration.** Two registry rows
(`overlay_single_addition_to_played_union_forward_holdout` and
`xlg05_transfer_partial_pooled`) were already recorded by hand at `P+ = 0.5`
for exactly this case, with reasoning in their `classification_evidence`; and
`scripts/xlg05_transfer_screen.py` had independently diagnosed the defect in a
comment and worked around it locally with the identical
`P+ = P(better) + 0.5·P(tie)` formula. This convention is what the project was
already reaching for by hand.

**Delivery.** One shared helper,
`nfl_ats.evidence_conventions.probability_positive_from_draws`, called from
every site rather than 100 edited copies. It takes `axis=` for the
column-per-metric callers and `ignore_nan=` for the callers that previously
used `np.nanmean` (a resample that failed to produce a number stays a failed
resample rather than becoming a loss). Grep for
`probability_positive_from_draws` to find every user of the convention.

Sites deliberately **not** rewritten: six that average already-computed
probabilities (`np.mean(naive_pp)` and friends), and
`scripts/spread_regime_opener_eval.py:351`, which compares against a non-zero
threshold and is a different quantity.

---

## D3. The sign test scored exact ties against the candidate

`WeakSignal.favours_candidate` is `effect > 0.0`, and `sign_test` counted
everything else as a baseline win. That is not a tie-breaking convention, it is
a one-directional thumb on the scale — and it was heavy.

**Measured on the live registry:**

| convention | favourable / total | rate | two-sided p |
|---|---|---|---|
| as coded (ties → baseline) | 576 / 1489 | 38.7% | 2.24e-18 |
| ties excluded (standard) | 576 / 1281 | 45.0% | 3.45e-04 |

Both readings lean baseline, so **no direction flipped** — but this is the
difference between "resolved" and "leaning", inflated by fourteen orders of
magnitude on p.

**Convention adopted: ties are excluded from the test.** This is the classical
sign-test construction (Dixon–Mood). A tie carries zero information about
direction, so conditioning on the informative comparisons is both the standard
choice and the one that keeps the binomial *exact*, rather than forcing a
half-integer count through it.

**Both readings are reported**, so no number is ambiguous about which
convention produced it: `favouring_candidate` / `favouring_baseline` /
`ties` / `informative_signals` / `share_favouring_candidate` alongside
`favouring_candidate_half_credit` and `share_favouring_candidate_half_credit`
(ties split evenly, matching D2's zero-atom convention), plus an explicit
`tie_convention` string.

`favours_candidate` itself is unchanged — it is honestly named — but its
docstring now says out loud that it is **not** the negation of
`favours_baseline`, and a three-way `direction` property (+1 / 0 / −1) was
added for callers that need the tie bucket.

---

## D4. Inverse-variance pooling was inverted for this population

`pooled_effect` weighted each signal by `1 / SE²` with the SE taken at face
value from the recorded interval. Textbook inverse-variance weighting is right
when the reported standard errors are trustworthy. Here they systematically are
not, and they fail in the one direction that does maximum damage: these
intervals come from **block bootstraps of mined cells, and a bootstrap band
shrinks as the cell it resamples gets smaller and more degenerate.** The least
informative entries therefore received the most weight.

**Measured before the fix**, on the NFL `accuracy_points` pool:
`roof_battery_visiting_dome_open_vs_closed_opener` — a **three-game** cell
whose own `classification_evidence` says in as many words that its narrowness
"is an artifact of resampling a 3-point sample, not statistical power" — held
**99.997%** of the fixed-effect weight. Kish's effective sample size was
**1.0**: a 1,369-signal pool worth one signal. `--method fixed` accordingly
reported `excludes_zero: TRUE` on a number that was that single cell's own
estimate to five decimals — and `excludes_zero` is exactly the field a session
would cite to justify a terminal verdict.

### What was rejected, and why

- **A minimum sample size.** Excluding a cell because it is small is a
  threshold decision, and `AGENTS.md` forbids discarding a signal for being
  underpowered. Rejected outright; nothing is dropped.
- **Rebuilding every variance from `sample_games`** (pure Hunter–Schmidt
  sample-size weighting). This removes the pathology but throws away *genuine*
  precision differences: these effects are `raw_gap × (slate share)`, so a rule
  touching 0.5% of games legitimately has both a small effect and a small SE,
  and `SE²·n` is not constant across the pool. Measured, it drove the pooled
  read to −0.123 with `excludes_zero: true` — trading one artifact for another.
  Rejected.

### What was adopted: a variance **floor**, derived from the pool

Weights stay inverse-variance. Each entry keeps its recorded precision, but
**never below what its own recorded sample size can support**:

1. An honest estimator's standard error scales as `σ/√n`, so `SE²·n` should be
   roughly constant across a commensurable pool. The scale `σ²` is estimated by
   **median** of `SE²·n`, so a handful of degenerate bands cannot set it.
2. Each entry's log-ratio to that curve is computed, and the cutoff is derived
   from the pool rather than fixed: the log-ratios' median minus three
   MAD-based standard deviations — the ordinary robust-outlier rule. (If more
   than half the pool records an identical ratio the MAD collapses to zero and
   says nothing, so the fallback scale is the mean absolute deviation, also
   derived from the pool.)
3. An entry below the cutoff has its variance floored to that point. It is
   **still pooled**, at the weakest precision its sample supports, and it is
   **flagged** for re-measurement. A three-game cell keeps a three-game cell's
   voice: small, but real.

The flag and the floor read the same curve, so the list of rows flagged for
re-measurement can never disagree with the list of rows that were floored.

`sample_games` is used, falling back to `sample_blocks` converted at the pool's
own median games-per-block, falling back to the pool's median sample size (and
counted in `sample_sizes_imputed`). When *no* entry carries either field — old
rows, synthetic test pools — the legacy inverse-variance weighting applies and
the output says so (`weighting: inverse_variance_fallback_no_curve`) rather
than inventing sample sizes.

### Before and after

Both readings are emitted on every call: the superseded scheme is reported
under `legacy_inverse_variance` so the change stays auditable, and
`--weighting inverse_variance` reproduces it in full.

| | `--method fixed` legacy | `--method fixed` fixed | `--method random` legacy | `--method random` fixed |
|---|---|---|---|---|
| pooled effect | −0.049949 | −0.008434 | −0.011037 | **−0.015488** |
| 95% interval | [−0.049999, −0.049899] | [−0.021093, +0.004225] | [−0.027731, +0.005658] | **[−0.035841, +0.004864]** |
| `excludes_zero` | **TRUE** | false | false | false |
| max weight share | **99.997%** | 8.84% | 2.07% | **1.17%** |
| effective signals (Kish) | **1.0** | 38.3 | 138.2 | **217.8** |
| most influential | the 3-game roof cell | `bias_battery_short_week` | the 3-game roof cell | `bias_battery_short_week` |

15 rows were floored and flagged.

---

## The live pooled read

Measured 2026-09-08 under the fixed code,
`nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`:

- **Pooled (random effects, sample-floored): −0.0155 accuracy points, 95%
  [−0.0358, +0.0049], `probability_positive` 0.068**, on 1,369 pooled signals,
  heterogeneity τ² = 0.0088.
- **Sign test: 576 candidate / 705 baseline among 1,281 informative signals**
  (208 exact ties excluded), p = 0.00034; on the half-credit reading, 680 /
  809 of 1,489.

The pile leans slightly to the baseline and is **not resolved**. Per the
binding rule, that is not grounds to close anything, and this document closes
nothing. The interval also still overstates precision — the `overlap_warnings`
are extensive (928,837 pairwise shared-season warnings) because the pool
contains many correlated decompositions of shared windows — so the sign test
and the per-entry rows remain the safer read.

`probability_positive` is now reported on the pooled estimate itself, so the
binary "contains zero" never has to be quoted.

---

## Registry rows flagged for re-measurement

`combination_report` now carries a `needs_remeasurement` block, surfaced by the
pool command. **No stored value was rewritten**: the original bootstrap draws
were never kept, so most of these rows cannot be honestly recomputed from what
is on disk.

Within the NFL `accuracy_points` scope (eligible + invalidated):

| bucket | count | correctable? |
|---|---|---|
| `zero_atom_probability_positive` | **115** | Yes, deterministically → 0.5 |
| `strict_zero_with_nonzero_effect` | 80 | No — needs the draws |
| `implausible_standard_error` | 15 | No — re-measure |

Registry-wide, the first bucket is **346** rows.

**Proposed correction, not applied.** The 346 rows in the first bucket are
unambiguous: `effect` exactly 0.0, `interval` exactly `[0, 0]`, `P+` exactly
0.0 means every resample was an exact tie, so the corrected value is exactly
0.5 and no stored draws are required to know it. Two rows in the registry were
already recorded at 0.5 by hand for this exact case. Applying it would need an
owner decision plus an audit line per row; it is **not** applied here.

The other two buckets are genuinely unknown and are listed as such. Flagging a
measurement for re-measurement is not a verdict on the signal: no row here is
refuted, bounded by a control, or reclassified.
