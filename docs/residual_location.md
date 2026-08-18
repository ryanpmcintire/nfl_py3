# The residual ECDF's location offset — characterization, mechanism, and a CFB screen

Written 2026-08-18, addressing the project's #2 open lead
(`docs/pool_edge_plan.md`, "Where to look next" item 2): the production pick
rule beats `sign(predicted_market_residual)` by **+2.12 accuracy points**,
95% `[+0.24, +4.17]`, `probability_positive` **0.990**, and the whole margin
is the *location* of a ~100-2,500-draw unweighted trailing-20% holdout
nobody had modelled. This document characterizes that offset, names the
mechanism, and screens two concrete remedies (recency weighting, location
shrinkage) on the CFB benchmark. **No NFL rotation-registry window is
assigned or spent by this document or by producing it** — every number
below is either CFB (free, rule 8) or an informational NFL read that never
calls `assign_window`/`record_look`, exactly as `docs/ecdf_smoothing.md`
established for this same research thread.

## 1. Where the ECDF lives, and its true size over time

`MarginModel.residuals` (`src/nfl_ats/margin.py:399-411`) holds one
out-of-time residual sample per fitted model. It is built once per
walk-forward week by `fit_margin_model` (NFL, `margin.py:585-653`) and its
CFB mirror `fit_cfb_residual_model` (`src/nfl_ats/cfb_benchmark.py:85-149`):
sort the eligible training rows chronologically, hold out the trailing 20%
(`distribution_fraction=0.20`), fit a *temporary* Ridge on the leading 80%
only, and the residual sample is `actual_target[late 20%] -
temporary_model.predict(late 20%)` — genuinely out-of-time. The `estimator`
actually used to score games is then refit on **all** of that week's
training data (leading 80% + trailing 20% combined), so the residual sample
and the deployed centre come from two different fits. `MarginModel.predict`
adds this fixed sample to every game's predicted centre and reads
`home_cover_probability` off the discretized empirical CDF
(`margin._smoothed_probability`, a Laplace/Krichevsky-Trofimov
continuity-corrected count: `(successes + 0.5) / (n + 1)`).

This is refit **every scored week** of an expanding walk-forward window, so
the sample size grows monotonically with training history:

| League | Seasons walked | `distribution_rows` range | Growth |
|---|---|---|---|
| CFB (`fit_cfb_residual_model`, `min_train_games=500`) | 2007-2025 (2006 has too few prior games to seed the floor) | **102 → 2,499** | ~700 CFB games/season means the trailing-20% slice itself spans multiple seasons by the 2020s |
| NFL (`fit_margin_model`, `market_residual`, `player` profile, `min_train_games=500`) | 2011-2025 | **102 → 883** | matches the 460-886 range `docs/ecdf_smoothing.md` reported from the frozen artifact; the final week here (2025 wk 18, 883 draws) reproduces a residual mean of **+0.849**, in the same neighborhood as that document's **+0.898** figure for the actual frozen artifact (small differences are a slightly different data snapshot/profile run, not a discrepancy in mechanism) |

Source: `scripts/residual_location_screen.py`, run artifact
`artifacts/residual_location/20260818T115234Z/` (`cfb_location.csv`,
`nfl_location.csv`, `diagnostics.json`).

## 2. Location by season: noisy local drift around zero, not a stable constant

Every walk-forward week's `model.residuals` mean/median was recorded and
averaged within each season. Selected seasons (full tables in
`artifacts/residual_location/20260818T115234Z/season_location_table.csv`
and `nfl_season_location_table.csv`):

**CFB** (mean of weekly residual means, points; `residual_std` ≈ 15.5
throughout):

| Season | Mean loc. | Season | Mean loc. | Season | Mean loc. |
|---|---|---|---|---|---|
| 2009 | −0.64 | 2015 | **−1.00** | 2021 | +0.32 |
| 2010 | +0.63 | 2016 | −0.47 | 2022 | −0.05 |
| 2011 | −0.27 | 2017 | −0.27 | 2023 | −0.23 |
| 2012 | +0.11 | 2018 | +0.11 | 2024 | −0.01 |
| 2013 | +0.21 | 2019 | +0.63 | 2025 | +0.53 |
| 2014 | −0.63 | 2020 | +0.52 | | |

**NFL, informational** (`player` profile; `residual_std` ≈ 12.4-12.6
throughout, matching the project's independently-measured ATS sd of
12.78-13.13):

| Season | Mean loc. | Season | Mean loc. | Season | Mean loc. |
|---|---|---|---|---|---|
| 2011 | +0.36 | 2016 | **−1.57** | 2021 | **−1.31** |
| 2012 | −0.06 | 2017 | −0.27 | 2022 | −0.32 |
| 2013 | +0.87 | 2018 | −0.02 | 2023 | +0.47 |
| 2014 | +0.40 | 2019 | −0.55 | 2024 | +1.13 |
| 2015 | **−1.62** | 2020 | **−1.15** | 2025 | +1.03 |

**Reading this**: the offset is **not** a stable non-zero constant — it
crosses zero and flips sign repeatedly, season to season, in both leagues.
Pooled across all 280 CFB walk-forward weeks the grand mean is
**−0.0006** (essentially exactly zero); the season-level swings (std of
weekly means 0.62 in CFB, 1.34 in NFL) are the visible part. It is also
not simply IID noise: there are multi-season runs in the same direction
(CFB 2014-2017 negative, 2018-2020 positive; NFL 2019-2022 negative,
2023-2025 positive). Some of that run structure is mechanical, not a
macro trend: because `distribution_rows` grows every week, the trailing
calibration window itself has grown to span **3+ CFB seasons** by the
2020s, so adjacent seasons' weekly estimates share most of their
underlying draws and are not independent evidence of a trend — a caveat
that applies to reading run-length here, not to the season-level point
estimates themselves. **Verdict for the "stable vs. drifting vs. noise"
question**: local, mean-reverting drift, not a stable additive bias. That
rules out "shrink toward a nonzero constant" as a sensible remedy and
points at recency-sensitivity as the right frame — screened in §4.

## 3. The mechanism: an intercept correction the point model doesn't carry, and it's (almost) all recency

**Named mechanism**: the residual sample's location is standing in for a
recency-aware intercept that the deployed Ridge model does not have. The
temporary calibration-split model is fit **only on the older ~80%** of
that week's training data; the residual sample it produces is exactly
`actual[late 20%] − temp_model.predict(late 20%)`. Because the temp
model's intercept is unpenalized (sklearn `Ridge` centers and fits it by
OLS on the training mean, out-of-time here), that residual mean decomposes
almost exactly as:

```
residual_mean ≈ [mean(target, late 20%) − mean(target, early 80%)]        (raw target drift the temp model never saw)
              − [mean(temp_model.predict, late 20%) − mean(temp_model.predict, early 80%)]   (drift the FEATURES already explained)
```

This is directly measurable without touching the fitted model at all: for
every walk-forward week, `_raw_drift` in
`scripts/residual_location_screen.py` computes the first (model-free) term
using nothing but the raw `ats_margin` split at the same 80/20 boundary.
Regressing the *actual* `residual_mean` the model produced against this
purely arithmetic `raw_drift`:

| League | n (weeks) | correlation | slope | intercept |
|---|---|---|---|---|
| CFB | 280 | **0.944** | 1.127 | −0.056 |
| NFL (informational) | 260 | **0.615** | 0.996 | −0.236 |

CFB's raw target drift alone explains **89%** of the variance in the
model's out-of-time residual mean (R² = 0.944²), with a slope
indistinguishable from 1 — the residual location is, almost entirely,
**the temporary model's inability to see how much the target's average has
already moved between the data it was fit on and the data right in front
of the walk**. NFL's relationship is weaker (R² = 0.38) because the richer
`player` feature profile explains away more of the raw drift through
features (the second term above is larger), but the sign and near-unity
slope hold.

**Why the production rule then beats `sign(predicted_market_residual)`**:
the *deployed* `estimator` is refit on the full expanding window every
week with no time-decay, so its own intercept is an unweighted average
over the model's **entire** training history — it lags the current era by
construction, more so as the window grows. The out-of-time residual
sample, drawn from only the trailing 20%, is close to a crude,
un-engineered recency correction bolted onto that otherwise
recency-blind model. It is not a designed feature; it is what happens to
survive when the calibration split is chronological. That is why thresholding
the ECDF-derived probability (which carries this correction) beats
thresholding the raw point residual (which does not) on 11.8% of games
where the two disagree.

## 4. CFB screen: recency weighting and location shrinkage on the same residual draws

Two OPT-IN alternative readers of the identical residual draws were added
in `src/nfl_ats/residual_location.py` (never imported by `margin.py`; the
frozen active model is unchanged whether or not this module is loaded,
same guarantee `calibration.py` already carries):

- **`recency_weighted_survival`** — exponential recency weights
  `0.5 ** (games_ago / half_life)` applied *within* the existing trailing-20%
  calibration sample (most recent draw weight 1, weight halves every
  `half_life` games back). Swept `half_life ∈ {100, 200, 400, 800}` games,
  spanning the full `distribution_rows` range measured in §1.
- **`shrunk_survival`** — shrinks the sample's mean toward zero by a
  fraction before reading the ECDF: `shifted = residuals − shrink_fraction
  × mean(residuals)`. `shrink_fraction=0` reproduces production exactly
  (test-pinned in `tests/test_residual_location.py`); `shrink_fraction=1`
  fully removes the location correction. Swept
  `shrink_fraction ∈ {0.25, 0.5, 0.75, 1.0}`.

Both were screened with the exact `fit_cfb_residual_model` recipe the
frozen XLG-03/MOD-16 CFB benchmark uses (Ridge α=10, `min_train_games=500`,
2006-2025 walk-forward), scoring identical weeks/games under all 9 arms
(baseline `ecdf` + 8 candidates) so only the probability *reader* differs.
8,933 clean-core games, `paired_feature_comparisons`, season-blocked
(project convention — matches the +2.12 headline's own blocking; 13 season
blocks) as primary, week-blocked as corroborating. Full CSV:
`artifacts/residual_location/20260818T115234Z/cfb_paired_comparisons.csv`.

**Accuracy (primary), season-blocked, CFB clean core** (positive = beats
current production `ecdf`):

| Candidate | Δ accuracy (pts) | 95% CI | `probability_positive` | Week-blocked `P+` |
|---|---|---|---|---|
| `recency_hl100` | −0.43 | [−1.08, +0.14] | 0.071 | 0.143 |
| `recency_hl200` | **−0.55** | **[−1.08, −0.06]** | **0.014** | 0.062 |
| `recency_hl400` | **−0.56** | **[−0.91, −0.19]** | **0.0005** | **0.015** ([−1.08, −0.06]) |
| `recency_hl800` | −0.17 | [−0.37, +0.05] | 0.058 | 0.157 |
| `shrink_025` | −0.03 | [−0.33, +0.25] | 0.389 | 0.394 |
| `shrink_050` | −0.24 | [−0.70, +0.26] | 0.152 | 0.143 |
| `shrink_075` | −0.38 | [−0.96, +0.16] | 0.094 | 0.095 |
| `shrink_100` | −0.35 | [−0.88, +0.21] | 0.105 | 0.134 |

**Brier (secondary), season-blocked**: every recency arm resolves negative
(`probability_positive = 0.000` for all four half-lives, e.g. `recency_hl100`
−0.00102 `[−0.00126, −0.00077]`) — recency weighting robustly *worsens*
calibration on CFB, not just accuracy. Shrinkage is calibration-neutral
(all four `shrink_*` Brier intervals cross zero, `P+` 0.22-0.54).

**Reading this, decision-first**: every one of the 8 candidate arms has a
negative accuracy point estimate against the current unweighted,
unshrunk production ECDF, in every evaluation window tested
(`clean_core`, `thin_2006_2011`, `regime_2020`) and under both blockings.
Two arms — `recency_hl200` and `recency_hl400` — **resolve** negative:
their 95% interval excludes zero under the project's own primary
(season-blocked) convention, and `recency_hl400` excludes zero under both
blockings simultaneously, backed by a Brier interval that also excludes
zero. Per AGENTS.md's taxonomy, an excluded-zero interval on the *wrong*
side is a genuine close, not a "crosses zero, ignore it" case — these two
are recorded as **refuted** (wrong sign, resolved on the well-powered CFB
instrument). The other six arms (the two remaining half-lives, and all
four shrink fractions) cross zero and are **unresolved, leaning negative**
— category 3, recorded not discarded, per the binding rule.

**Why recency weighting specifically backfires, given §3's mechanism**:
the calibration slice IS already "the recent 20%" — there is no further-back
data being pulled in by weighting it more; down-weighting its own older
members only shrinks the *effective* sample size (fewer effective degrees
of freedom in an already-thin 100-2,500 draw sample) without moving the
window's centre toward anything fresher, since the window can't extend
past "now." That trades away real information for no location gain, which
is exactly what a uniformly negative Brier readout across every half-life
says. Shrinkage doesn't have that failure mode (it never discards draws),
which is why its Brier reading is neutral rather than uniformly negative —
but §2 already shows the location is mean-reverting noise more often than
a stable bias worth partially removing, so shrinking it toward zero has no
clear upside either, and the point estimates lean the same (negative)
direction as the location grows more removed.

## 5. Recommendation

**Do not spend an NFL rotation-registry window on recency-weighting or
location-shrinking the residual ECDF as implemented here.** The CFB screen
— this project's well-powered instrument for exactly this question — leans
negative on every candidate and resolves negative on two of them. Spending
one of the four remaining NFL windows to confirm a family that already
screens this consistently negative would repeat the mistake
`docs/ecdf_smoothing.md` avoided (that document declined to reach for its
scarce opener windows given a merely-leaning-negative CFB read; this one
now has an actually-resolved negative on the identical instrument for two
of eight arms).

**This does not close the underlying lead.** §3's mechanism finding is
positive evidence, not a nuisance: the location offset is real, explicable
(R²=0.89 CFB, 0.38 NFL against a purely arithmetic drift term), and it is
already earning the measured +2.12 points in production. What's refuted is
narrower — *reweighting the same small calibration-slice draws by their
own internal recency* — not "recency-aware modelling of the location in
general." The mechanism points at a different, untested lever: the
**deployed mean model's own intercept is what's recency-blind** (an
unweighted average over its entire, ever-growing training history), not
the calibration slice. A recency-decayed *training sample weight* on the
mean model itself (or an explicit multi-season fixed effect), which would
let the model's own centre track the current era instead of asking a
100-2,500-draw side sample to patch it after the fact, has never been
screened and is where §3's decomposition says the real leverage is.

### Predeclaration a future session would freeze (recency-weighted mean model, not the ECDF reader)

Not run here; written so a future session can execute mechanically if this
lead is picked back up.

- **Candidate**: fit `fit_cfb_residual_model`/`fit_margin_model`'s Ridge
  with per-row sample weights `0.5 ** (games_ago / half_life)` (same
  `games_ago` convention as `nfl_ats.residual_location.recency_weights`)
  applied to the **temporary and final estimators**, not the residual
  reader. The residual sample keeps its current unweighted treatment
  unchanged (already screened negative above), isolating this as a new,
  independent lever.
- **Screen first, on CFB, exactly as this document did** — sweep half-life
  over the same `{100, 200, 400, 800}` grid (or wider; §1's CFB
  `distribution_rows` now runs to 2,499, so longer half-lives such as 1600
  are worth adding), report season-blocked `paired_feature_comparisons`
  accuracy as primary, Brier as secondary. Only predeclare an NFL window if
  the CFB read clears the same bar `docs/ecdf_smoothing.md` used
  (`probability_positive ≥ 0.75`) on at least one half-life.
- **Grade**: `nflverse_spread` (matches `ecdf_smoothing`'s reasoning: the
  screening methodology uses `game_features_player.parquet`'s
  `spread_line`, not the archived opener snapshot, and the opener pool's
  three windows are too scarce to spend on an unscreened candidate).
- **Rotation registry**: per `nfl_ats.rotation.eligible_blocks`'s
  deterministic earliest-block rule, with no inherited windows this would
  currently assign the same `[2011, 2013]` window `ecdf_smoothing` was
  assigned (verify against `nfl-ats rotation status` at declaration time,
  since other families may have changed the ledger by then; this document
  does not call `assign_window` and reserves nothing).
- **Frozen decision rule**: primary metric is paired forced-pick accuracy
  improvement vs. the current unweighted-training-sample production model,
  week-blocked `paired_feature_comparisons` on the assigned window; clears
  only if `probability_positive ≥ 0.75` on accuracy (same bar,
  same justification: a 3-season window is underpowered for a full
  interval-excludes-zero read on an effect this size). Brier is a secondary
  coherence check that does not override the accuracy rule in either
  direction.

## 6. Weak signals recorded (not run — payload only, per this session's constraints)

Eight category-3/refuted results from §4 are written as the exact
`nfl-ats weak-signals record` payload to
`<scratchpad>/weak_signal_record.json` (JSON array, one object per signal,
fields matching `src/nfl_ats/weak_signals.py`'s CLI contract). **Not
executed** — another process owns `registry/weak_signals.json` this
session. A future session (or the process that owns the file) should run
each entry through `nfl-ats weak-signals record` once free to do so.

## 7. Declared limitations

1. The recency-weighting and shrinkage candidates here only ever re-read
   the **existing** calibration-slice draws differently; neither changes
   what the temporary/final estimators are fit on. §5's proposed next step
   (weight the estimators' training rows) is a materially different
   candidate and was not screened by this document.
2. `shrink_fraction`'s location is computed from the sample **mean**, not
   the median the production `>= 0.5` decision actually thresholds. The two
   are close for this residual sample (CFB `residual_std` ≈ 15.5, broadly
   symmetric per the project's own prior Gaussian-fit finding in
   `docs/ecdf_smoothing.md`), but a median-based shrink target was not
   separately screened.
3. The §3 decomposition's "raw drift" term is deliberately model-free (pure
   arithmetic on `ats_margin`) so it can be computed without touching the
   fitted estimator; it is a **diagnostic**, not a claim that a model
   conditioning on it would recover the same numbers exactly. The
   R²-from-one-univariate-regression figures (0.89 CFB, 0.38 NFL) should be
   read as "how much of the location this one mechanism explains," not as
   a fully identified causal decomposition.
4. NFL numbers throughout (§1, §2, §3) are informational only, matching
   `docs/ecdf_smoothing.md`'s established convention for this research
   thread — no NFL accuracy screen was run in this document (§4 is
   CFB-only by design), and no NFL rotation-registry window was touched.
5. The season-level location tables in §2 are constructed by averaging
   weekly point estimates within a season; as noted there, these are not
   independent draws once `distribution_rows` grows past roughly one
   season's worth of games (CFB from the mid-2010s on), so the visible
   multi-season "runs" should not be read as strong evidence of a slow
   macro trend without accounting for that overlap.
