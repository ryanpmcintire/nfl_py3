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

**Applied later the same day.** The 346 rows in the first bucket were restated
to 0.5 with an append-only `corrections` entry each (commit `819ff7f`), and the
`zero_atom_probability_positive` bucket is now empty. The section below covers
what happened to the other two buckets.

---

## 2026-09-08, second pass: the strict-zero rows were re-measurable after all

The paragraph above says the remaining rows "cannot be honestly recomputed from
what is on disk". That is true of the *draws* and false of the *measurement*.
The MOD-18 research lanes each kept their scored per-game frame — the candidate
and baseline pick for every game, with the season and week the block bootstrap
resamples on — so the identical bootstrap can simply be run again. Nothing has
to be assumed about the draws, because the draws can be regenerated.

`scripts/weak_signal_zero_atom_remeasure.py` does this. For each lane it rebuilds
the cell grid from `scored.parquet` (or the lane's `paired.parquet`) using the
lane's own grouping code and the same shared `comparison()` helper, at the same
20,000 draws and the same seed, and then **requires every field except
`probability_positive` to reproduce the recorded cell exactly** before the
corrected value is allowed to stand. Effect, interval bounds, standard error,
sample size, block count and flip count all have to come back bit-for-bit.

**Measured: 463 rows reproduced, 0 drifted.** Not one cell in nine lane grids
came back with a different effect or interval, at a tolerance of 1e-9. That
equality is the whole argument — it is what makes this a re-measurement of the
same measurement rather than an edit of a recorded number. **457 of the 463
needed a corrected `probability_positive`**; the other six already carried the
value the re-measurement produced.

| lane | source still on disk | rows verified | rows corrected |
|---|---|---|---|
| R (`r1`, residual slope) | `artifacts/research/laneR/scored.parquet` | 92 | 92 |
| U (`r2`, shrunk slope) | `artifacts/research/laneU/scored.parquet` | 64 | 64 |
| I (`s5`, home-side prior) | `artifacts/research/laneI/scored.parquet` | 55 | 55 |
| V (`oos1`, out-of-sample declaration) | `artifacts/research/laneV/scored.parquet` | 54 | 54 |
| G (`s4`, side-aware) | `artifacts/research/laneG/scored.parquet` | 44 | 44 |
| T (`kl1`, key-line lattice) | `artifacts/research/laneT/scored.parquet` | 38 | 38 |
| T (home-side mapping) | session scratchpad `laneT/paired.parquet` | 32 | 32 |
| S (`S1`/`S2`, buckets + seasons) | session scratchpad `laneS/paired.parquet` | 42 | 41 |
| Q (`Q1`/`Q2`, buckets + seasons) | session scratchpad `laneQ/paired.parquet` | 42 | 37 |
| **total** | | **463** | **457** |

**Every correction moved the same way: up.** Measured across all 457:
minimum **+0.00010**, median **+0.04537**, mean **+0.06152**, maximum
**+0.18477**, and **457 of 457 non-negative** — exactly what a defect that
charged the zero atom against the candidate predicts. The **75** rows that
recorded `probability_positive` of exactly **0.0** re-measure between
**0.00028 and 0.18477**; the worst of them,
`mod18_home_side_location_v1_r2_r2b_pick_home_card_2020_2025`, was filed as
"no chance the candidate is better" when the honest reading is **18.5%**.

Each row was re-recorded through `nfl-ats weak-signals record --replace`,
carrying its stored payload back verbatim with the one field changed. Diffed
against the pre-pass registry, **457 rows moved `probability_positive` and not
one of them moved any other field** — description, source, effect, interval,
standard error, sample size, classification, family, notes, plain summary and
category all came back byte-identical. (Other rows in the file changed during
the same window; those are other lanes recording, not this pass.)

### What this changes for the decision, and what it does not

**It does not move the pool, measured rather than asserted.** Pooling the same
registry twice — once as it now stands, once with only these 457
`probability_positive` values reverted — returns the identical number to the
last digit: **−0.042455 accuracy points, 95% [−0.088496, +0.003587],
`probability_positive` 0.0354**, on 1,569 pooled signals, and the identical sign
test (636 candidate / 840 baseline among 1,476 informative signals, 213 ties
excluded, p = 1.21e-07). That is expected from the construction:
`pooled_effect` weights by effect and standard error, and the sign test reads
the effect's sign; a row's own `probability_positive` is not an input to either.

**The pile of small signals still leans slightly to the baseline, and it is
still not resolved.** Per the binding rule that is not grounds to close
anything, and nothing here closes anything. (The pooled read is also not
comparable to the −0.0155 on 1,369 signals recorded earlier in this document:
other lanes added 200 eligible rows to the registry between the two readings.
Re-run the command for the current number.)

**It changes what 457 rows say, one at a time.** `AGENTS.md` requires each
result to be reported as its `probability_positive` rather than as the binary
"contains zero", so that per-row number is the one a session reads when deciding
whether a candidate is worth playing. Every one of these 457 rows was
understating the candidate's side, and 75 of them by the largest margin the
scale allows. A row that reads 0.0 gets discarded on sight; the same row reading
0.18 is a small unresolved lean that stays in the pile. That is the failure mode
the four defects were fixed to stop, showing up one row at a time instead of in
the headline.

Measured on the pool report, before and after, same registry:
`needs_remeasurement.strict_zero_with_nonzero_effect` **80 → 5**;
`zero_atom_probability_positive` stays 0; `implausible_standard_error` is
unchanged at 20 (it grew from 15 during the session as other lanes recorded).

### Inventory: what is still not re-measured

| bucket | rows | verdict |
|---|---|---|
| re-measured this pass | 75 of the 80 `strict_zero_with_nonzero_effect` rows | done |
| re-measurable, costs a screen re-run | 5 strict-zero rows + the `implausible_standard_error` rows | not done here |
| unreproducible (source gone) | 0 | — |

**Nothing in either flagged bucket turned out to be unreproducible.** Every
source named by a flagged row still exists on disk. The five strict-zero rows
left (`bias_battery_short_week_opener`, `movement_leads_rising_total_dog`,
`week1_dog_on_production`, `pol09_best_pick_composed_v1_tiebreak_dispersion_vs_alphabetical`,
`roof_battery_visiting_dome_open_vs_closed_opener`) come from screens rather
than lanes, and a screen stores per-cell summaries rather than the per-game
arms — so they need the screen re-run, not a rebuild from a parquet. All of the
screens involved (`scripts/roof_decision_screen.py`,
`scripts/nfl_bias_battery_screen.py`, `scripts/movement_leads_battery.py`,
`scripts/schedule_flag_on_production.py`,
`scripts/best_pick_composed_rule_eval.py`, and `nfl-ats experiment run`) route
through code that now calls `probability_positive_from_draws`, so re-running
them produces the corrected value directly. Each re-run also rewrites its whole
battery, which is why they are a separate piece of work.

### The `implausible_standard_error` flag is measuring the wrong thing

Measured on the live pool (at 1,376 NFL `accuracy_points` rows with a usable
band): the median `SE² · n` is **599.7**, and all 15 rows flagged at that
moment sit between **0.0000 and 3.0** — two to seven orders of magnitude below
the curve. Reading their sources explains why, and it is not a bad band:

`artifacts/body_clock_night_screen/20260821T222542Z/results.json` records
`body_clock_night_west_road_ge2000et` as `raw_gap_pts` −6.21 on `n_flag` 119 of
`n_total` 4,317, scaled to a `full_slate_effect_pts` of −0.171 by
`fraction_of_slate` 0.0276. The registry row stores that scaled effect and its
scaled band — correct — alongside `sample_games` **119**, the flagged subset.
The plausibility curve then compares a band scaled to the whole 4,317-game slate
against a 119-game sample size. `SE² · n` is understated by both the wrong `n`
and the squared slate fraction.

This is the limitation D4 already names ("these effects are `raw_gap × slate
share`, so a rule touching 0.5% of games legitimately has both a small effect
and a small SE, and `SE²·n` is not constant across the pool"), now measured. The
flag is a **commensurability defect** — two conventions for expressing an
`accuracy_points` effect, per-subset-game and per-card-game, sharing one pool —
not evidence that these 15 measurements are wrong. Two of the 15
(`body_clock_night_dose_1700_1959` at `n_flag` 3,
`divisional_rematch_revenge_early_w1to6` at `n_flag` 4) are genuinely tiny cells
as well. The fix belongs in the recording convention (store the population the
effect is scaled to) or in the curve (fit against that population), not in the
rows; flooring them is conservative but is answering a question they were not
asked. **This is a defect report, not a closure**: no row here is refuted,
bounded by a control, or reclassified, and all 15 stay `unresolved_below_power`.

Re-running those screens would fix their `probability_positive` — every one of
them reaches the number through `scripts/_common.py`, `nfl_ats.clv` or
`nfl_ats.experiment_runner`, all of which now call
`probability_positive_from_draws` — but it would **not** clear this flag, because
the flag is about which population `sample_games` names. Fixing the flag is a
recording-convention change, and it belongs to whoever owns
`src/nfl_ats/weak_signals.py` and the screen recorders.

### Reproducing this pass

```
python scripts/weak_signal_zero_atom_remeasure.py --report out.json           # verify only
python scripts/weak_signal_zero_atom_remeasure.py --report out.json --apply   # re-record
```

The apply step is idempotent — it skips any row already carrying the
re-measured value — so an interrupted pass is resumed by running it again, and
`--from-report` reuses an earlier verification instead of re-running the
bootstraps.
