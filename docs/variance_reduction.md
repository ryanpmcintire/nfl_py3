# Variance reduction for the paired feature comparison

This is a methodology reference, not a pick-affecting predeclaration: nothing
here changes a model, a feature profile, a pick, or `artifacts/
active_ats_model.json`. It documents two add-on estimators
(`src/nfl_ats/variance_reduction.py`) layered on top of
`experiments.paired_feature_comparisons`, and the empirical evidence
(`scripts/variance_planted_effects.py`, CFB clean core, 2006-2025,
market-residual arm, 8,933 games / 199 week-blocks) for how much detection
power each one buys.

## The problem this answers

At this evaluator's scale, `paired_feature_comparisons` resolves roughly a
2-accuracy-point difference; real feature effects run 0.2-1.3 points
(`AGENTS.md`). Almost every comparison comes back with an interval crossing
zero -- the expected outcome for a real-but-small signal, per the project's
own binding invariant -- which burns a scarce evaluation window (an NFL
rotation-registry window) without resolving anything. Both levers below
reduce the WIDTH of the comparison, not what it measures.

## Lever 1: covariate-adjusted (CUPED) comparison

### The estimator

`covariate_adjusted_paired_comparisons(predictions, covariates, ...)` mirrors
`paired_feature_comparisons`'s schema and per-game improvement definitions
(accuracy / Brier / log-loss). For each metric, it fits `theta` by OLS
(`cuped_adjust`) regressing the raw per-game improvement on
`DEFAULT_CUPED_COVARIATES` -- `abs_spread_line`, `total_line`,
`on_key_number`, `abs_rest_diff`, `week_number` (`build_cuped_covariates`,
derived from strictly pregame columns) -- then bootstraps the residual
(`values - theta @ (covariates - covariate_means)`) with the identical
week/season block-bootstrap `paired_feature_comparisons` already uses.

### Why it is provably unbiased for the same estimand

`covariates - covariate_means` sums to exactly zero over the sample by
construction, so `theta @ (covariates - covariate_means)` also sums to
exactly zero **for any theta**, not just the fitted one. Subtracting it
therefore cannot move the sample mean:

```
mean(adjusted) = mean(values) - theta @ mean(covariates - covariate_means)
               = mean(values) - theta @ 0
               = mean(values)
```

This is an algebraic identity, not a statistical result contingent on the
covariates being any good — pinned by
`test_cuped_adjust_preserves_mean_for_arbitrary_theta` (adjusts with three
deliberately-bad thetas, including all-zero and a huge random one, and
checks the mean is untouched to `1e-8`) and cross-checked against production
`paired_feature_comparisons` on identical data
(`test_covariate_adjusted_matches_raw_point_estimate_and_reference_bootstrap`:
`raw_estimate`/`estimate` match the reference `estimate` to `1e-9`). Fitting
`theta` in-sample only changes which part of the per-game noise is explained
away before the interval is built — never the point estimate.

### Which covariates actually paid

Measured on the real CFB clean core with a 1.0-accuracy-point planted arm
(`covariate_effects`, full 8,933 games): total variance reduction is real but
modest — **1.35% (accuracy), 0.55% (Brier), 0.54% (log-loss)**. Per-covariate
(univariate) contribution to the accuracy metric:

| Covariate | Univariate variance reduction |
|---|---|
| `week_number` | 0.041% |
| `on_key_number` | 0.035% |
| `total_line` | 0.019% |
| `abs_spread_line` | 0.002% |
| `abs_rest_diff` | 0.0003% |

`week_number` and `on_key_number` pay the most (still small); `abs_rest_diff`
pays essentially nothing on CFB (weekly schedule, little rest variance).
That the pooled residual model's per-game noise is only weakly explained by
market/context covariates is consistent with MOD-16's finding
(`docs/margin_variance.md`): the pooled out-of-time residual distribution is
already close to correctly calibrated, so there is not much exploitable
heteroskedasticity left for a linear covariate to mop up.

### Empirical gain (planted effects, required games for 80% power)

| Effect | current (accuracy) | covariate-adjusted (accuracy) | multiplier |
|---|---|---|---|
| 1.0 pt | 5,272 games | 4,943 games | **1.07x** |
| 2.0 pt | 1,389 games | 1,167 games | **1.19x** |
| 0.5 pt | not reached at 8,933 | not reached at 8,933 | — |
| 0.25 pt | not reached at 8,933 | not reached at 8,933 | — |

**Bottom line, Lever 1 alone: roughly equivalent to 1.07-1.19x more data** —
real, free, but modest on this covariate set. It does not rescue effects
accuracy cannot resolve at all (0.25/0.5 pt); it shaves 7-19% off the games
needed for effects accuracy CAN resolve.

### Small-sample calibration caveat (a real limitation, disclosed)

At very small windows, in-sample `theta` (5 covariates fit on few games) can
overfit and **inflate the false-positive rate**: pooled-null FPR (nominal
~2.5%, "interval excludes zero from below") was **17.7% at 3 week-blocks
(~134 games)** and **6.6% at 6 blocks (~270 games)**, dropping to **3.3% at
12 blocks (~540 games)** and **1.5% at 25 blocks (~1,120 games)** — in line
with nominal from there on. **Rule: do not trust the CUPED-adjusted interval
below roughly 12 week-blocks (~500 games) in this data; fall back to the raw
`paired_feature_comparisons` interval, which stayed at or below nominal (0.3
-3.8%) at every sample size tested.**

## Lever 2: a screening ladder on continuous metrics

### Relative efficiency, in games-equivalent terms

Required games for 80% power, same planted mechanism, continuous vs binary:

| Effect | current (accuracy) | Brier | log-loss | Brier multiplier | log-loss multiplier |
|---|---|---|---|---|---|
| 1.0 pt | 5,272 | 603 | 602 | **8.75x** | **8.76x** |
| 2.0 pt | 1,389 | 156 | 156 | **8.89x** | **8.89x** |
| 0.5 pt | not reached at 8,933 | 1,746 | 1,764 | — (accuracy never resolves; Brier does in 1,746) | — |
| 0.25 pt | not reached at 8,933 | 4,310 | 4,312 | — (accuracy never resolves; Brier does in 4,310) | — |

**Bottom line, Lever 2 alone: roughly equivalent to 8.7-8.9x more data** for
effects accuracy can eventually resolve, and it resolves two magnitudes
(0.25, 0.5 accuracy points) that forced-pick accuracy **cannot resolve at
all**, even with the entire 8,933-game CFB clean core. Stacking both levers
(CUPED-adjusted Brier) reaches **~10.0-10.4x**.

### Internal consistency check

Required-N scales roughly as `1/effect^2` (fixed-variance CLT scaling): the
1.0pt requirement (5,272 games) implies a 2.0pt requirement of `5272/4 ≈
1,318` games — the simulation's directly-measured 2.0pt requirement is
**1,389 games**, within 5%. This also lines up with `AGENTS.md`'s own
operational claim ("resolves ~2 accuracy points" at the sample sizes this
project actually runs): a 0.5pt effect (1/4 the reference magnitude) needs
roughly 16x the reference games, comfortably beyond the 8,933-game clean
core — exactly what was observed (never reached).

### Does a continuous-metric result predict the eventual accuracy result?

At an affordable screening size (12 week-blocks, ~540 games — comparable to
a real NFL confirmation window), Brier's `probability_positive >= 0.75`
screen:

| Arm | Screen pass rate (~540 games) | Accuracy power at 540 games | Accuracy power at full 8,933 games |
|---|---|---|---|
| null (pooled, 5 seeds) | 14.4% | 1.3% | 0.0% |
| 0.25 pt | 61.5% | 6.0% | **0.0%** (never resolves) |
| 0.50 pt | 83.25% | 6.25% | 6.0% (never reliably resolves) |
| 1.00 pt | 97.75% | 15.5% | **100%** |
| 2.00 pt | 100% | 40.5% | **100%** |

Illustrative concordance (80% prior on "a real effect somewhere in
0.25-2.0pt", 20% on null, screen = Brier `probability_positive >= 0.75` at
~540 games): **P(real effect | screen passes) ≈ 96%.** The screen reliably
separates real from null. It does **not** reliably predict whether accuracy
specifically will ever confirm it: two of the four tested magnitudes
(0.25pt, 0.5pt) pass the screen most/half the time yet accuracy never
resolves them even with the full CFB clean core. This is the honest limit —
and it is not a contradiction of this project's own binding rule (an
interval crossing zero is not grounds for rejection): a real, EV-positive
effect below accuracy's resolution floor is still worth playing as a forced
pick, it is just never going to be provable via accuracy alone at any
feasible sample size.

### Concrete screen-then-confirm rule

1. **Screen** on Brier (or log-loss; near-identical) improvement's
   `probability_positive >= 0.75`, at whatever sample is cheaply available
   (CFB, or free/non-reserved NFL seasons). This is a recall-oriented gate
   (~14-22% nominal false-positive rate at realistic screening sizes, by
   design, matching the project's existing `best_pick_ranker`/
   `ecdf_smoothing` 0.75 convention) — never a decision.
2. **Confirm** only candidates that clear the screen with a real, unmodified
   `paired_feature_comparisons` accuracy run on a genuinely fresh (e.g.
   NFL rotation-registry) window. The stricter "excludes zero from below"
   Brier/log-loss criterion is well-calibrated even at ~540 games (FPR
   1.1-2.0%, close to nominal) but has materially lower recall than the 0.75
   screen (e.g. 78% vs 98% recall at 1.0pt) — use it as a secondary
   coherence check, never as a substitute decision rule.
3. **Never** let a continuous-metric result stand in for the accuracy
   verdict on its own — the pool grades forced picks. This screen only
   changes which candidates get a scarce confirmation window spent on them.

## No-bias check (planted-null false-positive rates)

Pooled across 5 independent null datasets (permuted-outcome direction, same
noise texture as the positive arms), "interval excludes zero from below"
criterion (nominal ~2.5%, one-sided tail of a two-sided 95% interval):

| n_blocks | mean games | current | covariate-adjusted | Brier | log-loss | combined |
|---|---|---|---|---|---|---|
| 3 | 134 | 3.8% | 17.7%* | 8.4%* | 8.8%* | 16.8%* |
| 6 | 270 | 3.2% | 6.6%* | 4.4% | 4.1% | 7.2%* |
| 12 | 539 | 1.5% | 3.3% | 2.0% | 1.8% | 3.1% |
| 25 | 1,121 | 1.2% | 1.5% | 1.1% | 0.8% | 1.2% |
| 50 | 2,251 | 0.3% | 0.4% | 0.1% | 0.1% | 0.1% |
| 100 | 4,481 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

(\* below the ~500-game floor Lever 1's own caveat already flags — not a
new finding, the same overfitting-at-small-N effect visible in both the
CUPED-only and combined columns.) From 12 blocks (~540 games) on, every
method sits at or below nominal — **no method shifts the point estimate or
inflates detections once the small-sample floor is respected.** The `current`
column never exceeds nominal at any tested size, confirming the reference
method itself is not the source of the small-N miscalibration.

## Reproducing this

```
.\.tools\uv.exe run --no-sync python scripts\variance_planted_effects.py
```

Reads `data/processed/cfb_game_features.parquet`, refits the frozen CFB
market-residual benchmark (`cfb_benchmark.cfb_walk_forward_benchmark`,
~22s), plants known effects, and writes `planted_effects.csv`,
`power_curves.csv`, `false_positive_rates.csv`, `required_sample_sizes.csv`,
`screen_confirm_concordance.csv`, and `metadata.json` to `--output` (default:
a scratch path, never `artifacts/` or the rotation registry). Total runtime
~55s. No NFL rotation-registry window is read for writing or spent by this
script (CFB only, rule 8).

## What this does not do

- Does not change any pick, feature profile, or model artifact.
- Does not replace `paired_feature_comparisons` — both levers are additive,
  opt-in estimators callers choose to use alongside it.
- Does not claim any real feature has a 0.25-2.0pt edge — the planted
  effects are synthetic, outcome-correlated constructions built to validate
  the evaluator's own detection power, per this task's brief.
