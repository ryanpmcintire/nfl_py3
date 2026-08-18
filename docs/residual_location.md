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

## 8. 2026-08-18 re-measurement: do the two terminal verdicts survive?

Written the same day as §4-§7, after a separate instrument audit
(`docs/estimation_variance.md`, `docs/anytime_valid.md`) found three defects
in the measurement instrument these two rows were closed with: **D2**, the
block bootstrap never refits, so every reported interval understates its own
width (measured 17-58% too narrow on other comparisons); **D3**, the old
`samples=2000` default carried seed-to-seed jitter of 0.02-0.03 points
(defaults raised to 20,000 project-wide the same day); **D1**, the
out-of-time residual/calibration step can attenuate or invert small effects
(`docs/purged_cv.md` §3). `docs/revisit_list.md` Tier 1 flagged
`residual_location_recency_hl200_cfb` and `residual_location_recency_hl400_cfb`
— the only two `refuted_mechanism` (terminal) verdicts in this family — for
re-measurement, not because a reversal was expected, but because a terminal
verdict resting on a since-audited instrument should be checked rather than
assumed. **All numbers below are measured this session**
(`scripts/residual_location_reaudit.py`, run output
`<scratchpad>/residual_location_reaudit/residual_location_reaudit.json`,
reusing one CFB walk-forward pass via a new `weekly_capture` hook added to
`run_cfb` — §4's original numbers are preserved above, untouched).

**Bottom line: neither terminal verdict survives.** Both reclassify from
`refuted_mechanism` to `unresolved_below_power`.

### 8.1 Reproduction

`scripts/residual_location_reaudit.py` reran the exact recorded configuration
(`samples=2000, seed=20260818`) and matched §4's numbers **bit-for-bit**:
`recency_hl200` −0.5485 pts `[−1.0774, −0.0562]` P+ 0.014; `recency_hl400`
−0.5597 pts `[−0.9143, −0.1928]` P+ 0.0005. The original measurement is not in
question — only whether its interval was honest.

### 8.2 D3: seed/sample-count jitter (samples=20000, 4 seeds)

Re-run at the corrected default (`samples=20000`) across four seeds
(`20260818, 1, 2, 3`):

| Arm | Lower range (pts) | Upper range (pts) | `P+` range |
|---|---|---|---|
| `recency_hl200` | [−1.051, −1.032] | [−0.066, −0.056] | [0.0127, 0.0142] |
| `recency_hl400` | [−0.909, −0.900] | [−0.193, −0.185] | [0.0011, 0.0014] |

Jitter at the new default is small (interval edges move by ≤0.02 pts,
`P+` by ≤0.003) and **does not explain** what follows — the honest-interval
result in §8.3 is roughly twenty times larger than this jitter. D3 is fixed,
confirmed, and not the reason either verdict changes.

### 8.3 The honest, mechanism-targeted interval

`docs/estimation_variance.md` §7 states its own refit-aware bootstrap
(resample training rows, refit the mean Ridge model) **does not apply** to
this family: all nine arms share one mean model and differ only in how a
fixed out-of-time residual sample is *read*, so refitting the model changes
nothing about what separates them. That document's own §8 limitation #2
names the source it left unaddressed: resampling
`fit_cfb_residual_model`'s internal 80/20 residual-calibration split itself.
That is exactly what a reader-only family's comparison turns on, so this
session built that bootstrap instead of borrowing D2's number: per week,
independently resample (with replacement, chronological order preserved so
recency weighting stays meaningful) that week's own out-of-time residual
sample, recompute both readers off the resample with the mean model's
centres held fixed, and combine with an independent block resample of games
in the same outer loop (`nfl_ats.estimation_variance.refit_aware_paired_interval`,
generalized from "refit the model" to "resample the calibration draws").
Validated against the production reader functions on an identity resample
(zero-noise case) before use — max abs. difference `8.9e-16`, floating-point
noise. `n_boot=2000`; matches the recorded 8,933-game paired sample exactly
(an initial pass over-included 160 push games with undefined `home_cover`
that `experiments.paired_feature_comparisons` drops but the first cut of this
script did not — caught and fixed in `WeeklyCache.capture` before these
numbers were finalized; the fix moved `P+` by 0.0000 and the point estimate
by <0.02 pts, i.e. it was never the story, but the 8,933 game count now
matches exactly).

| Arm | Estimate (pts) | 95% CI, season-blocked (primary) | `P+`, season | `P+`, week-blocked (corroborating) |
|---|---|---|---|---|
| `recency_hl200` | −0.59 | `[−1.45, +0.67]` | **0.2585** | 0.281 |
| `recency_hl400` | −0.49 | `[−1.03, +0.62]` | **0.3080** | 0.3105 |

Both terminal `P+` values move from deep in the "resolved negative" range
(0.014, 0.0005) to squarely in "unresolved, leaning negative" — the same
bucket their six siblings already occupy.

**Width inflation measured here vs. D2's borrowed range**: `recency_hl200`
2.074x, `recency_hl400` 2.290x — **larger** than D2's own measured 1.037x-1.575x
range from a *different* comparison type (different fitted Ridge models, not
readers of one fixed sample). This is the expected direction, not a
contradiction: D2 explicitly disclaims covering this family (§7 quoted
above), and a small (102-2,500-draw), highly leveraged calibration sample
read two different ways is a noisier comparison than two different Ridge
fits scored on the same centres. As a cruder, disclaimed comparator, applying
D2's blanket range directly to the *recorded* naive interval brackets the
same conclusion less precisely, and the honest way to state that range is in
`probability_positive`, not in where an interval edge falls: across D2's low
(1.037x) to high (1.575x) end, `recency_hl200` moves from P+ 0.019 to 0.106
and `recency_hl400` from P+ 0.001 to 0.043. Every one of those readings is
well under a coin flip and none of them is a refutation — the whole span
lands in the same place the mechanism-targeted bootstrap does (P+ 0.259 /
0.308), just less precisely and with a wider spread. The mechanism-targeted
bootstrap above is the one this document treats as authoritative, per D2's
own stated scope.

### 8.4 D1 cross-check: sign-only is degenerate for this family, by construction

`sign(predicted_margin − spread_line)` never reads `model.residuals` — it
depends only on the mean model's centre and the market line, both identical
across all nine arms by construction (one shared `model.predict` call feeds
every reader). **Confirmed numerically, not assumed**: zero disagreements
between the sign pick and the `ecdf` baseline's sign pick, across all eight
candidate arms, on all 9,093 raw clean-core games. A literal sign-only paired
contrast between any two readers in this family is therefore **fully
degenerate** (`f=0` by construction) — unlike `cfb_role_continuity`'s
different-feature-columns family, where the sign-only ablation is a valid,
non-degenerate D1 check.

The closest valid substitute: each reader's departure from a synthetic
sign-only arm (season-blocked; week-blocked corroborates within 0.02 pts):

| Reader vs. sign-only | Δ (pts) | 95% CI | `P+` |
|---|---|---|---|
| `ecdf` | **+0.60** | `[−0.07, +1.28]` | 0.960 |
| `recency_hl200` | +0.06 | `[−0.64, +0.75]` | 0.564 |
| `recency_hl400` | +0.04 | `[−0.62, +0.72]` | 0.545 |

Exact algebraic identity (a consistency check, not new information):
`0.6045 − 0.0560 = 0.5485` and `0.6045 − 0.0448 = 0.5597` — precisely the two
recorded headline effects, confirming the decomposition is self-consistent.

**Reading this**: the production `ecdf` reader earns essentially the whole
of its edge over doing nothing (`P+` 0.960 vs. sign-only). Recency weighting
does **not invert** that edge — both `recency_hl200` and `recency_hl400`
still lean positive against sign-only (`P+` 0.56, 0.55, both near a coin
flip) — but it gives back roughly 90% of it. This is a narrower reading than
D1's original planted-effect finding (a full sign inversion): recency
weighting attenuates the calibration step's benefit rather than reversing
it. **Inferred**: D1's originally-identified instability source — ridge-fit
noise contaminating the calibration split (`docs/purged_cv.md` §3) — is
*shared identically* by every reader within a week (the residual sample
itself is one fixed draw, unweighted or reweighted only after the fact), so
it differences out of a reader-vs-reader paired comparison and cannot be
what separates `ecdf` from `recency_hl200`/`hl400`. What differs is which of
that fixed sample's draws are up- or down-weighted before the ECDF is read —
a real but narrower channel, consistent with §4's original diagnosis
("recency weighting... shrinks the *effective* sample size... without moving
the window's centre toward anything fresher").

### 8.5 Power arithmetic

`f` = fraction of clean-core games where the arm's forced pick differs from
`ecdf`'s; `MDE80 = 280·√(f/n)`, `n=9,093` (the raw clean-core game count used
for pick comparisons, vs. the 8,933 push-filtered paired-accuracy count in
§8.3 — `f`/MDE80 don't depend on outcome validity, so this ~1.7% difference
is immaterial):

| Arm | `f` | MDE80 (pts) | Recorded \|effect\| (pts) |
|---|---|---|---|
| `recency_hl200` | 9.25% | 0.893 | 0.5485 — **below** MDE80 |
| `recency_hl400` | 5.32% | 0.677 | 0.5597 — **below** MDE80 |
| sign-only vs. `ecdf` | 0% (exact, matches §8.4) | undefined | n/a |

Both recorded effects sit below their own pipeline's 80%-power detection
threshold — an independent line of evidence, using only the *naive* pipeline's
own arithmetic, that neither was ever powerful enough to be called resolved.

### 8.6 Re-classification

Per `AGENTS.md`'s taxonomy, only two things justify a terminal close: a
resolvably wrong sign, or bounded-by-a-positive-control. Neither arm clears
either bar under honest treatment:

- **`recency_hl200`**: `refuted_mechanism` does **not** survive. Honest `P+`
  0.2585 (season) / 0.281 (week) — the interval excludes nothing decisively;
  D3 jitter is too small to explain the change; the effect is below its own
  MDE80. **Reclassify to `unresolved_below_power`.**
- **`recency_hl400`**: `refuted_mechanism` does **not** survive. Honest `P+`
  0.3080 (season) / 0.3105 (week). Same reasoning. **Reclassify to
  `unresolved_below_power`.**

Proposed exact `registry/weak_signals.json` edits (not applied by this
session — CFB re-measurements do not touch the registry; the orchestrator
serializes registry writes):

```json
{
  "residual_location_recency_hl200_cfb": {
    "classification": "unresolved_below_power",
    "classification_evidence": "2026-08-18 re-audit (scripts/residual_location_reaudit.py, docs/residual_location.md sec 8): the naive interval under-covers for this reader-only family MORE than docs/estimation_variance.md's borrowed 17-58% range (measured width inflation 2.074x here, via a mechanism-targeted bootstrap that resamples the out-of-time residual-calibration sample itself -- the source docs/estimation_variance.md sec 7-8 explicitly flags as unaddressed by its training-row-refit bootstrap for reader-only families). Under that honest interval P+ rises from 0.014 to 0.2585 (season-blocked, primary) / 0.281 (week-blocked). D3 seed jitter at samples=20000 is negligible (P+ range 0.0127-0.0142 across 4 seeds) and does not explain the change. The recorded effect (-0.5485 pts) also sits below its own naive-pipeline MDE80 (0.893 pts, f=9.25%). Sign-only D1 check is degenerate for this family by construction (f=0; predicted_margin never depends on the reader) -- closest valid contrast (vs. a synthetic sign-only arm) shows recency weighting attenuates but does not invert the calibration step's edge (ecdf P+ 0.960 vs. sign-only; hl200 P+ 0.564). Reclassified from refuted_mechanism (terminal) to unresolved_below_power, joining recency_hl100/hl800 and all four shrink_* siblings.",
    "effect": -0.5933,
    "effect_units": "accuracy_points",
    "interval": [-1.4480, 0.6703],
    "probability_positive": 0.2585,
    "standard_error": 0.5404,
    "sample_blocks": 13,
    "sample_games": 8933,
    "source": "docs/residual_location.md sec 4 (original) + sec 8 (2026-08-18 re-audit); scripts/residual_location_reaudit.py; artifacts/residual_location/20260818T115234Z/cfb_paired_comparisons.csv (original naive numbers, preserved); <scratchpad>/residual_location_reaudit/residual_location_reaudit.json (re-audit)"
  },
  "residual_location_recency_hl400_cfb": {
    "classification": "unresolved_below_power",
    "classification_evidence": "2026-08-18 re-audit, same design as recency_hl200_cfb (see that entry for the full method). Measured width inflation 2.290x. Honest P+ rises from 0.0005 to 0.3080 (season-blocked, primary) / 0.3105 (week-blocked). D3 seed jitter is negligible (P+ range 0.0011-0.0014). Recorded effect (-0.5597 pts) sits below its own naive-pipeline MDE80 (0.677 pts, f=5.32%). Sign-only D1 check is degenerate by construction; closest valid contrast: hl400 P+ 0.545 vs. sign-only (ecdf P+ 0.960). Reclassified from refuted_mechanism (terminal) to unresolved_below_power.",
    "effect": -0.4926,
    "effect_units": "accuracy_points",
    "interval": [-1.0317, 0.6203],
    "probability_positive": 0.3080,
    "standard_error": 0.4214,
    "sample_blocks": 13,
    "sample_games": 8933,
    "source": "docs/residual_location.md sec 4 (original) + sec 8 (2026-08-18 re-audit); scripts/residual_location_reaudit.py; artifacts/residual_location/20260818T115234Z/cfb_paired_comparisons.csv (original naive numbers, preserved); <scratchpad>/residual_location_reaudit/residual_location_reaudit.json (re-audit)"
  }
}
```

All other fields (`recorded_at`, `description`, `league`, `seasons`, `notes`)
are unchanged from the existing entries.

### 8.7 The family as a whole (descriptive, signs already seen)

With this reclassification, **all eight arms** in the family — four
half-lives, four shrink fractions — are `unresolved_below_power`, and all
eight have negative point estimates on the identical 8,933-9,093 CFB games
(§4's table: `hl100` −0.43, `hl200` −0.55/−0.59 honest, `hl400` −0.56/−0.49
honest, `hl800` −0.17, `shrink_025` −0.03, `shrink_050` −0.24, `shrink_075`
−0.38, `shrink_100` −0.35). This is a consistent, worth-noting descriptive
pattern, stated plainly as **descriptive only, not a fresh pooled finding**:
`AGENTS.md`'s pooling license requires the family to be declared before the
signs are seen, and here every sign was already visible before this
re-measurement started. The eight arms are also **not independent trials** —
same 8,933 games, same underlying calibration mechanism, nested designs (the
shrink sweep and half-life sweep both re-read one shared residual sample) —
so no aggregate `probability_positive` or sign-test is computed from this;
doing so would misrepresent eight correlated reads of one sample as eight
independent experiments. The practical upshot: §5's recommendation (do not
spend an NFL rotation-registry window on recency-weighting or shrinking the
residual ECDF reader as implemented here) is **unchanged** by this
reclassification — no arm's point estimate flipped positive, only the
terminal-vs-unresolved status of two of them changed. What did change: no
row in this family may any longer be cited as a *resolved* negative; every
row is category-3, unresolved, leaning negative, and stays open per
`AGENTS.md`'s binding rule against closing on a crossing-zero interval.

### 8.8 Declared limitations of this re-measurement

1. The honest interval (§8.3) holds the mean model's predicted centre fixed
   and only resamples the residual-calibration draws — correct for isolating
   this family's actual variance source (§8.3's argument from
   `docs/estimation_variance.md` §7-8), but it does not also refit the mean
   Ridge model, so any (believed-small, per §7's scope note) contribution
   from mean-model refit variance to this specific comparison is not
   captured here.
2. `n_boot=2000` for the honest interval, smaller than the naive interval's
   20,000-sample bootstrap; §8.2 shows the naive interval's own jitter at
   this order of magnitude is small, and `n_boot=2000` was chosen for
   compute, not because 2,000 is independently known sufficient — a future
   session could raise it cheaply if a verdict ever sits close to a gate
   again.
3. A first pass of this script's honest interval included 160 push games
   with undefined outcomes that the recorded naive interval excludes; caught
   and fixed before any number here was finalized (§8.3), but recorded as a
   limitation of the *process*, not just a footnote, since it is exactly the
   kind of silent scope mismatch `AGENTS.md`'s "verify before quoting"
   discipline exists to catch.
4. This section does not re-derive `docs/estimation_variance.md`'s D2 range
   itself, only applies it as a disclaimed comparator; the mechanism-targeted
   bootstrap in §8.3 is the number this document treats as authoritative.
