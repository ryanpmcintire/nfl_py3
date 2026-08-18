# ECDF smoothing — predeclaration for an `nflverse_spread` confirmation window

Predeclared 2026-08-17/18, before any NFL rotation-registry window is
assigned or spent for this family. The CFB screen and the NFL "informational"
walk-forward below are BOTH free of the registry: CFB carries no NFL
confirmation cost at all (rule 8), and the NFL run never calls
`assign_window`/`record_look` — it exists only to choose a method and set an
honest expected effect size, exactly as MOD-16 used the CFB benchmark before
its own predeclaration (`docs/margin_variance.md`). No NFL rotation-registry
window has been assigned or spent by this document or by producing it.

## Where this comes from, and what "the ECDF" actually is

`margin.fit_margin_model` (mirrored in `cfb_benchmark.fit_cfb_residual_model`)
fits the mean model on the full training set but reads cover/win/loss
probabilities off a SEPARATE, smaller sample: the trailing 20% of training
games, held out chronologically so the residuals used for the distribution
are genuinely out-of-time rather than in-sample fit residuals (which would
be systematically too small). `MarginModel.residuals` holds those draws, and
`margin._smoothed_probability` reads a discretized empirical CDF off them —
a Laplace/Krichevsky-Trofimov continuity-corrected count, not a fitted
density:

```
successes = count(center + residuals > line)
probability = (successes + 0.5) / (n_residuals + 1)
```

For the current frozen active model (`market_residual`, player profile, full
2009-2025 training) `n_residuals = 886`. That count is not fixed — it is
`int(0.20 * training_rows)` and grows every week of an expanding walk-forward
window. Across the standing 2018-2025 backtest artifact
(`artifacts/margins/20260817T200603Z`) it ranges from 460 to 883 games, and
takes the value **518 exactly** for 16 games at one particular week roughly
midway through that walk (confirmed by direct inspection of
`distribution_rows` in that artifact — not a special constant, just the
draw count the 20%-holdout split naturally produces partway through an
expanding window). It is "only" a few hundred draws by construction: the
alternative (using in-sample residuals from the full training set) would be
larger but optimistic, since the model that produced the point prediction
would be scoring its own fit.

## Why smoothing can move picks even though rescaling cannot

`docs/pool_edge_plan.md` records, correctly, that any method whose whole
effect is to **rescale** the point prediction — shrinkage, regularization,
recalibration by a positive scalar — cannot flip `sign(predicted residual)`.
That result is real and is NOT contradicted here. It does not apply to ECDF
smoothing because **the pool's actual forced pick is not
`sign(predicted_market_residual)`.** It is
`home_cover_probability >= 0.5` (`nfl_ats.pool.build_ats_pool_card`), and
`home_cover_probability` is the empirical **median** of `center + residuals`
compared against the line, not the point residual compared against zero.
Those two decision rules already disagree today, on the unsmoothed
production model: measured directly on the 2018-2025 backtest artifact,
**250 of 2,127 market_residual predictions (11.8%)** have
`sign(predicted_market_residual) != (home_cover_probability >= 0.5)`,
because the held-out residual sample has a nonzero, sampling-noisy mean/
median (it is correcting for the temporary fit model's own out-of-time
bias — measured mean +0.898 on the current full-history fit, +0.036 for the
underlying `fair_margin` residual pool). Roughly 9.7% of games already sit
within 0.01 of the 0.5 threshold and 4.5% within 0.005.

Smoothing re-estimates the residual distribution's **location and shape**
from the same draws — it denoises exactly the median/threshold-crossing
point a small ECDF estimates noisily. That is a different lever from
rescaling the centre, and it is why it can flip picks: any game whose
`line - predicted_margin` sits between the raw ECDF's noisy median and the
smoothed distribution's estimated median flips sides. It cannot, and does
not, change every pick uniformly (unlike a scale change), and it is
concentrated in exactly the games nearest that boundary (confirmed below).

## Method selection (screened on CFB, free, before any candidate was chosen)

`nfl_ats.calibration` (this item's owned module) adds
`fit_residual_smoother`/`smoothed_home_cover_probability`, an OPT-IN reader
of the same residual draws with four methods:

- `ecdf` — the CONTROL arm; reproduces `margin._smoothed_probability`
  exactly (pinned in `tests/test_calibration_ecdf_smoothing.py`, floating
  precision).
- `gaussian` — analytic `N(mean, std)` fit to the residuals.
- `gaussian_kde` — nonparametric kernel density (`scipy.stats.gaussian_kde`,
  default bandwidth).
- `skew_normal` — `scipy.stats.skewnorm.fit`.

None of this touches `margin.py`; the frozen active model is bit-identical
whether or not `calibration.py` is imported (same test).

`scripts/ecdf_smoothing.py` walks all three candidates against the `ecdf`
control on identical weeks, sharing the identical mean model (only the
probability read differs), on the CFB benchmark's clean core (8,933 paired
games, 2006-2025 walk-forward, `min_train_games=500`, matching the frozen
XLG-03/MOD-16 recipe exactly) and, informationally, on the NFL 2009-2025
walk-forward (3,919/3,818 paired games; this reuses already-computed
production math and is not a new registry look).

### Screening result: Brier/log-loss improve; forced-pick accuracy does not (and CFB — the well-powered instrument — leans negative)

CFB clean core, week-blocked paired comparison (`positive = smoothed
better`; `artifacts/ecdf_smoothing/20260818T000600Z/cfb_paired_comparisons.csv`):

| Candidate | Brier improvement | P(Brier+) | Log-loss improvement | P(log-loss+) | Accuracy improvement | P(accuracy+) |
|---|---|---|---|---|---|---|
| gaussian | +0.000114 | 0.896 | +0.000233 | 0.900 | -0.00325 pts | **0.126** |
| gaussian_kde | +0.000089 | 0.964 | +0.000179 | 0.964 | -0.00224 pts | **0.085** |
| skew_normal | +0.000107 | 0.938 | +0.000219 | 0.939 | -0.00291 pts | **0.082** |

Season-blocked intervals agree (accuracy P(positive) 0.06-0.15 for all three
methods). **This is the opposite pattern from what "costs nothing" implies**:
calibration improves with high confidence, but the metric this whole project
is built around (forced-pick accuracy, per the project's own stated bar) is
resolvably more likely to get *worse* than better, on the one instrument
with enough games to resolve it. Read against
`docs/pool_edge_plan.md`'s "three kinds of negative": this is not (yet) a
category-1/2 close, because the interval does not exclude zero and the NFL
evidence below is genuinely mixed — but it is a real, adverse lean that the
original one-line recorded finding ("costs nothing") did not have the power
to see.

NFL 2009-2025, informational, week-blocked
(`artifacts/ecdf_smoothing/20260818T000600Z/nfl_paired_comparisons.csv`):

| Candidate | Brier improvement | P(Brier+) | Accuracy improvement | P(accuracy+) |
|---|---|---|---|---|
| gaussian | +0.001590 | **1.000** | +0.00079 pts | 0.558 |
| gaussian_kde | +0.001230 | **1.000** | -0.00026 pts | 0.420 |
| skew_normal | +0.001477 | **1.000** | -0.00210 pts | 0.255 |

The Brier improvement here (+0.0012 to +0.0016, P essentially 1.0) is the
same magnitude and direction as the recorded finding ("buys Brier -0.0015 at
P=0.998", opposite sign convention: their negative = their Brier score fell,
i.e. improved) — this is read as an independent, close reproduction of the
calibration half of that finding. But accuracy on NFL is a near-coin-flip for
every method (P 0.42-0.56 except skew-normal), i.e. **unresolved, not
"costs nothing"**: unresolved and "zero cost" are different verdicts, and
NFL alone cannot tell them apart here at this sample size. The CFB evidence,
which can resolve it, leans against "zero cost."

**Candidate selected for the confirmation: `gaussian`.** It has the smallest
(least negative) CFB accuracy readout of the three, the fewest interval
outliers, and the only NFL accuracy point estimate that leans positive —
while still delivering the class's core Brier/log-loss gain. This is also
the theoretically motivated choice: the project has already independently
measured the ATS residual as near-Gaussian (sd 13.13,
`docs/pool_edge_plan.md`), so a Gaussian smoother should track the true
distribution more faithfully than a flexible nonparametric fit (KDE,
skew-normal) that can chase small-sample noise in a 460-900 draw sample.
`gaussian_kde` and `skew_normal` are recorded here as rejected-on-evidence
alternatives, not deleted — if `gaussian` fails its confirmation, they
remain candidates but would need their own predeclaration.

### Pick movement is concentrated exactly where the mechanism predicts

Flip rate (`gaussian` vs the `ecdf` control), by `key_numbers.line_bucket`:

| Bucket | CFB flip rate | NFL flip rate |
|---|---|---|
| `\|line\| < 3` | 8.85% | **10.44%** |
| `\|line\| = 3` | 6.07% | 5.62% |
| `3.5 <= \|line\| <= 6.5` | 6.05% | 7.61% |
| `\|line\| = 7` | 7.75% | 6.48% |
| `\|line\| > 7` | 7.74% | 7.46% |

Flipped games have systematically smaller `|predicted_market_residual|` than
the population: NFL mean 0.777 vs 2.214 overall (median 0.664 vs 1.676); CFB
mean 0.344 vs 1.131 overall. Both patterns are exactly what the mechanism
above predicts — flips cluster in near-pick'em lines and in games whose point
prediction already sits close to the decision boundary, not randomly across
the slate, and least often at the `\|line\| = 3` key number specifically
(the market's single most information-dense line). Overall `gaussian` flip
rate: 7.43% (CFB, 676/9,093), 7.86% (NFL, 308/3,919).

## Hypothesis for the confirmation window

`gaussian` residual smoothing improves the calibrated `home_cover_probability`
(lower Brier/log-loss) and, on a fresh out-of-sample NFL window, forced-pick
accuracy is **not worse** than the unsmoothed ECDF control by a margin the
window can resolve. The screening evidence above makes the accuracy question
genuinely open (CFB leans negative, NFL is a coin flip) — that is exactly why
this needs its own predeclared window rather than shipping on either
measurement.

## Grade and window

**Grade: `nflverse_spread`** (not `opener`), for two reasons. (1) The
screening methodology above used the standard `game_features_player.parquet`
`spread_line`, matching the `nflverse_spread` data contract exactly, not the
archived Tuesday-opener snapshot. (2) The opener pool is scarce (six seasons,
three 2-season windows total) and two of its three windows
(`best_pick_ranker_opener`, `mod07_weak_signal_stack`, both `[2020, 2021]`)
are already spent by other families; spending one on a candidate whose
well-powered instrument leans negative on the headline metric is a real cost
this project should not pay before the cheap, broad `nflverse_spread` pool
has answered the accuracy question. This mirrors the `best_pick_ranker` →
`best_pick_ranker_opener` precedent: confirm broad first, then decide whether
an opener-graded follow-up (a new family, e.g. `ecdf_smoothing_opener`,
`inherits: [ecdf_smoothing]`) is worth one of the three scarce opener
windows.

**Training policy**: forward-chaining, `min_train_games=500`, identical to
every other `nflverse_spread` family (`fit_margin_model`, ridge alpha 10,
player feature profile — the frozen active model's own recipe, `target=
"market_residual"`), so the only thing that differs between arms is the
probability read (`method="ecdf"` vs `method="gaussian"`), exactly as in the
screening runs above.

**Inherits**: none. This is a genuinely new family — no existing spent
window constrains it.

**Contamination**: with no inherited windows and `MIN_ELIGIBLE_START_SEASON
= 2011`, `nfl_ats.rotation.eligible_blocks`/`assign_window`'s deterministic
earliest-block rule (traced directly against `src/nfl_ats/rotation.py`,
not executed) assigns **`[2011, 2013]`** — entirely outside the mined
2018-2025 ledger, so `acknowledges_mined_2018_2025=False` is correct at
declaration. Seasons 2011-2012 currently have zero usage by any family in
`registry/rotation_registry.json`; 2013 has been used by two other families,
which rule 4 permits (windows retire per-family, not globally).

## Frozen decision rule

Unlike MOD-16 (whose candidate was pick-invariant by construction, so it
could gate on calibration alone), this candidate can and does move picks, so
**the primary metric is paired forced-pick accuracy improvement** on the
assigned window (`gaussian` vs `ecdf`, week-blocked,
`paired_feature_comparisons`). The candidate **clears** only if
`probability_positive` on accuracy is at least **0.75** (matching the
SPEC-5/`best_pick_ranker` screening bar already established in this
registry) — not merely if the interval excludes zero, since a 3-season
window is underpowered for a full interval-excludes-zero read on a signal
this size. Brier and log-loss improvement are reported as secondary
coherence checks and do NOT override the accuracy rule in either direction:
a Brier win with an accuracy loss is a **close**, not a partial success,
given this project's own stated bar (forced-pick accuracy vs the coin flip,
never a calibration metric). One run; no method, feature, or split retuning
after seeing the window's results — a different method is a new
predeclaration.

## Commands to run (NOT executed by this document)

```powershell
.\.tools\uv.exe run --no-sync nfl-ats rotation declare `
  --name ecdf_smoothing `
  --description "Gaussian residual-distribution smoothing vs the raw out-of-time ECDF: does it improve forced-pick accuracy (primary) and Brier/log-loss (secondary) on a fresh nflverse_spread window? See docs/ecdf_smoothing.md." `
  --grade nflverse_spread

.\.tools\uv.exe run --no-sync nfl-ats rotation assign --name ecdf_smoothing
```

Expected assignment: `[2011, 2013]` (default 3-season `nflverse_spread`
width; verify against `nfl-ats rotation status` at run time, since another
family may have changed the ledger between now and then).

After scoring the assigned window with `scripts/ecdf_smoothing.py`'s
`nflverse_spread`/player-profile recipe against `[2011, 2013]` (a new run,
not the CFB/informational-NFL screen above):

```powershell
.\.tools\uv.exe run --no-sync nfl-ats rotation record `
  --name ecdf_smoothing `
  --artifact <artifacts/ecdf_smoothing/<confirmation-run-id>> `
  --verdict <confirmed|closed_negative|unresolved> `
  --probability-positive <accuracy P(positive) from that run> `
  --notes "<one-line summary of the accuracy/Brier/log-loss result>"
```

## Declared limitations

1. The screening evidence above (CFB + informational NFL) is used only to
   pick the method and set an honest prior on effect size, per this
   project's own established practice of screening distribution work on CFB
   first (MOD-16). It is not the confirmation, and NFL 2009-2025 was never
   registered as a look.
2. Only `home_cover_probability` (the two-way forced-pick threshold) is
   smoothed. Push/three-way probabilities keep `margin._three_way_
   probabilities`' existing integer-rounding treatment; a smoothed push
   model is out of scope and would be a separate predeclaration (relevant to
   MOD-05, not this item).
3. `gaussian_kde`/`skew_normal` are not carried into the confirmation. If
   `gaussian` closes negative, either alternative would need its own fresh
   predeclaration, not a retry inside this one.
4. The 3-season `[2011, 2013]` window is comparatively small (fewer games
   than the CFB screen); the 0.75 `probability_positive` bar is set as a
   screening gate for that reason, matching `best_pick_ranker`'s precedent,
   not as a claim that the effect is fully resolved at that sample size.
