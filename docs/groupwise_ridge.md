# Group-wise ridge penalties

Written 2026-08-17. The lead came out of a re-read of the closed
`player_qb_continuity` experiment: that comparison ran the candidate at ridge
`alpha = 1.0` against a baseline at `alpha = 10.0` and returned exactly 0.000.
Re-derived alpha-matched, the candidate arm is worth **+1.1033** accuracy points
and the alpha change alone **−1.1033**. They cancelled exactly. **The window
stays spent and that result is not re-scored here.** What is actionable is the
implication: a single global ridge penalty is an assumption, and this project
has never tested it.

Everything below was measured on the **CFB benchmark** (12,206 games, rotation
rule 8 — unreserved, no NFL confirmation window spent).

---

## 1. The structural check, stated before anything was built

`docs/pool_edge_plan.md` records the corollary that closed MOD-06:

> the pick is `sign(predicted residual)`, and rescaling by a positive scalar
> cannot change a sign, so a method whose whole effect is "be more
> conservative" cannot move a single pick.

**Verdict: group-wise penalties escape it.** The corollary governs maps of the
form `yhat -> c * yhat` with `c > 0` constant across games. Group-wise ridge is
not such a map.

Generalized ridge solves `b(L) = (X'X + L)^-1 X'y` with `L = diag(lambda_1 ...
lambda_p)`. For two penalty vectors to give predictions related by a positive
scalar *as functions*, we would need `b(L1) = c * b(L0)`. In an orthogonal
standardised design (`X'X = diag(d_j)`) the solution is
`b_j = d_j * b_j^OLS / (d_j + lambda_j)`, so proportionality requires
`(d_j + lambda_j^1) / (d_j + lambda_j^0)` to be the *same* constant for every
`j` at once. That fails as soon as the `lambda_j` differ across blocks and the
`d_j` are not all equal. Differential shrinkage changes the **direction** of the
coefficient vector, not just its length, so the prediction becomes a different
linear functional of the features and its sign can flip.

Two things sharpen this for our actual design rather than a textbook one:

- **A *global* alpha change is also not a pure rescale.** After
  `StandardScaler` the design would have to be orthonormal with equal `d_j` for
  that to hold. It is not — see the next point — which is exactly why the
  `player_qb_continuity` alpha mismatch cost 1.1033 points instead of 0. The
  ROADMAP's MOD-06 entry states the corollary slightly too broadly; ridge
  penalty changes are *not* in the class of methods it closes. That is an
  existence proof on this very pipeline that penalty changes move picks.
- **The design is rank-deficient by construction.** Every team-state metric
  ships as `home_x`, `away_x`, `diff_x` with `diff = home - away` exactly. On
  the CFB contract that is 8 exact linear dependencies: complete-case rank is
  **27 of 35 columns**. On the NFL active-model contract (`market_residual` /
  `player`), 79 declared columns become 142 after median imputation with
  indicators, of which only **71 directions are non-null**. In a rank-deficient
  design the penalty structure is what selects among otherwise-equivalent
  solutions, so it has more leverage here than in a well-conditioned one.

**Empirical confirmation.** At the frozen `alpha = 10`, the strongest declared
arm (`market_light_10`) flips **54 of 11,989** scored CFB picks (0.45%) against
the uniform baseline, with 44 sign flips in the raw predicted residual. Non-zero
is the whole structural claim: a positive rescale flips exactly zero, and the
regression test `test_a_positive_rescale_flips_nothing` pins that contrast.
At penalty levels where the penalty actually bites, the flip counts are far
larger (Section 4).

So the answer to "does this fall to MOD-06" is **no** — but see Section 5,
because escaping the corollary is not the same as being worth anything.

---

## 2. Why `ridge_alpha = 10.0` turned out to be the real finding

Before running the block grid, one diagnostic: ridge shrinks the principal
direction with eigenvalue `d` by `alpha / (d + alpha)`. Measure the eigenvalues
of the standardised design and you learn what the penalty is actually doing.

| design | n | transformed cols | non-null directions | median eigenvalue | **median shrinkage at alpha=10** |
|---|---:|---:|---:|---:|---:|
| CFB, training < 2012 | 2,904 | 60 | 38 | 1.18e3 | 0.0084 |
| CFB, training < 2018 | 6,919 | 63 | 41 | 3.54e3 | 0.0028 |
| CFB, training < 2025 | 11,738 | 63 | 41 | 6.02e3 | **0.0017** |
| NFL active model, training < 2018 | 2,304 | 142 | 71 | 1.82e3 | 0.0055 |
| NFL active model, training < 2026 | 4,431 | 142 | 71 | 3.48e3 | **0.0029** |

(Medians are over non-null directions only; the null ones are shrunk to zero at
any alpha and carry no signal by construction.)

**At the frozen `alpha = 10`, the median direction of the NFL active model is
shrunk by 0.29%.** Even the weakest decile of real directions is shrunk by only
16%. The project's production model is, to three figures, unregularised least
squares on a rank-deficient design; `alpha = 10` is an undocumented default that
does nothing except pick a solution inside the null space, which does not affect
predictions because the null space comes from an exact identity that holds in
scoring rows too.

That has a direct consequence for this screen: **reallocating a penalty that is
inert cannot do anything, whatever the ratio.** Testing block ratios only at
`alpha = 10` would have been a rigged screen guaranteed to return "no effect".
So the grid is run at three global levels, and the two extra ones are derived
from the table above rather than chosen: at the final CFB cut, `alpha = 1e3`
puts median shrinkage at 0.14 and `alpha = 1e4` at 0.62 (0.46 and 0.89 at the
earliest cut), bracketing the range where a penalty has any leverage at all.

It also reframes the lead. The `player_qb_continuity` +1.1033 / −1.1033 pair was
the difference between two models whose coefficients differ by a fraction of a
percent, measured on 997 games. That is a coin-flip metric's noise sensitivity,
not a signal — a useful calibration on how much accuracy movement this
evaluator produces from nothing.

---

## 3. Design

**Implementation** (`src/nfl_ats/margin.py`). Generalized ridge with penalty
`alpha * m_j` on column `j` is implemented *exactly* by scaling column `j` by
`1/sqrt(m_j)` and running an ordinary `Ridge(alpha)`: with `S = diag(1/sqrt(m))`
and `g` the coefficients on the scaled design, `b = S g` and
`alpha * ||g||^2 = alpha * sum_j m_j b_j^2`. Algebraically
`S (S X'X S + alpha I)^-1 S = (X'X + alpha diag(m))^-1`. No approximation.

- `GroupPenaltyScaler` sits between `StandardScaler` and `Ridge`. It must come
  after standardisation — standardising afterwards would divide the scaling
  straight back out.
- Missing-value indicator columns arrive named `missingindicator_<source>` and
  inherit their source column's multiplier, so a block's missingness flags are
  penalised with the block. This needs the pipeline's output container set to
  pandas, which is done **only** on the group-penalty path.
- `resolve_feature_groups` labels columns using the project's own
  `FEATURE_FAMILIES`; no new taxonomy is introduced, and an unclaimed column
  raises rather than silently defaulting.

**Strictly opt-in.** `make_margin_estimator(..., column_penalties=None)` returns
the identical three-step pipeline it always has, and `fit_margin_model` defaults
to `None`. Pinned three ways in `tests/test_margin_groupwise.py`, and verified
once directly against `git show HEAD:src/nfl_ats/margin.py` on the real NFL
feature table: predictions, residual distribution, and metadata are all
bit-identical for both the `base` and `player` profiles. `margin_model_metadata`
emits the `column_penalties` key only when penalties are in use, so a frozen
run's payload does not gain a field.

**Normalisation.** Every arm's multipliers are divided by their count-weighted
geometric mean, so all arms carry the same *average* penalty and only the
*allocation* across blocks differs. Without this, a "market light / state heavy"
arm would also be a global-alpha change, and the result could not be attributed
to the block structure — the very confound that produced the lead.

**The predeclared grid** (`scripts/groupwise_ridge_screen.py`, fixed before any
result was read). CFB blocks are `market` (2 columns), `context` (7),
`experience` (2), `offense` (12), `defense` (12).

| arm | market | context / experience | offense / defense |
|---|---:|---:|---:|
| `uniform` (baseline) | 1 | 1 | 1 |
| `market_light_3` | 1/3 | 1 | 3 |
| `market_light_10` | 1/10 | 1 | 10 |
| `market_heavy_3` | 3 | 1 | 1/3 |
| `market_heavy_10` | 10 | 1 | 1/10 |
| `state_heavy_3` | 1 | 1 | 3 |
| `state_light_3` | 1 | 1 | 1/3 |

The grid is deliberately **symmetric**: `market_light_*` is the lead's
hypothesis (market columns are a near-sufficient statistic and want a light
penalty; rolling team-state columns are noisy and want a heavy one), and
`market_heavy_*` is its equal and opposite. If both directions help, the
movement is noise. Run at global `alpha` in {10 (frozen), 1e3, 1e4} = 21
walk-forwards.

**Correctness gate.** The `uniform` arm at `alpha = 10` must reproduce
`cfb_walk_forward_benchmark` bit-for-bit. It does: max `|delta residual| =
0.000e+00` over all 11,989 scored games.

---

## 4. Results

All numbers: CFB clean core (8,933 evaluated games of 9,093 scored), paired
week-blocked bootstrap against `uniform` **at the same global alpha**, 2,000
resamples over 199 blocks. `probability_positive` is the fraction of resamples
favouring the arm. Positive always means the arm is better.

### 4a. Pick flips — the diagnostic that separates this from MOD-06

Flips are counted on the scored pick (`home_cover_probability >= 0.5`) over all
11,989 games.

| arm | flips @ alpha=10 | @ alpha=1e3 | @ alpha=1e4 |
|---|---:|---:|---:|
| `market_light_3` | 30 (0.25%) | 447 (3.7%) | 973 (8.1%) |
| `market_light_10` | 54 (0.45%) | 798 (6.7%) | **1,502 (12.5%)** |
| `market_heavy_3` | 77 (0.64%) | 706 (5.9%) | 911 (7.6%) |
| `market_heavy_10` | 241 (2.0%) | 1,394 (11.6%) | 1,464 (12.2%) |
| `state_heavy_3` | 29 (0.24%) | 402 (3.4%) | 716 (6.0%) |
| `state_light_3` | 39 (0.33%) | 455 (3.8%) | 703 (5.9%) |

A positive rescale flips **zero**. Group-wise penalties flip up to an eighth of
the slate. **The MOD-06 corollary does not apply to this method.** The flip
count also tracks the penalty level exactly as the eigenspectrum predicts:
near-nothing where the penalty is inert, thousands of picks where it bites.

For reference, moving the *global* alpha alone from 10 to 1e4 (`uniform` vs
`uniform`) flips 2,406 picks (20.1%) — so both axes move picks, and neither is
the rescale MOD-06 closed.

### 4b. Forced-pick accuracy

Clean-core accuracy by arm and penalty level:

| arm | log(m_state / m_market) | alpha=10 | alpha=1e3 | alpha=1e4 |
|---|---:|---:|---:|---:|
| `market_light_10` | +4.61 | 0.5151 | 0.5183 | **0.5213** |
| `market_light_3` | +2.20 | 0.5156 | 0.5137 | 0.5160 |
| `state_heavy_3` | +1.10 | 0.5155 | 0.5137 | 0.5140 |
| `uniform` | 0 | 0.5160 | 0.5158 | 0.5161 |
| `state_light_3` | −1.10 | 0.5160 | 0.5155 | 0.5135 |
| `market_heavy_3` | −2.20 | 0.5161 | 0.5161 | 0.5119 |
| `market_heavy_10` | −4.61 | 0.5156 | 0.5098 | 0.5106 |

Paired week-blocked deltas at alpha=1e4:

| arm | accuracy delta | 95% | `probability_positive` |
|---|---:|---|---:|
| `market_light_10` | **+0.526 pts** | [−0.230, +1.319] | **0.9155** |
| `market_light_3` | −0.011 | [−0.664, +0.657] | 0.4925 |
| `state_heavy_3` | −0.202 | [−0.750, +0.369] | 0.2290 |
| `state_light_3` | −0.258 | [−0.748, +0.221] | 0.1300 |
| `market_heavy_3` | −0.414 | [−1.024, +0.162] | 0.0790 |
| `market_heavy_10` | −0.549 | [−1.317, +0.168] | 0.0690 |

**This is a category-3 result — unresolved below detection power, not a
positive.** The interval contains zero; the best arm is the best of **21**
configurations, so it carries a selection discount; and the dose-response is
**not monotone** (accuracy dips below `uniform` at +1.10 and +2.20 before
jumping up at +4.61), which a real gradient would not do. What *is* meaningful
is the ordering of the extremes: the hypothesised direction is the best arm and
its equal-and-opposite control is the worst, at both non-inert penalty levels.

### 4c. Brier and log loss — here the effect is resolvable

Paired week-blocked deltas at alpha=1e4, clean core:

| arm | log(m_state / m_market) | Brier delta | 95% | `P+` | log-loss delta | `P+` |
|---|---:|---:|---|---:|---:|---:|
| `market_light_10` | +4.61 | **+0.000304** | [+0.000055, +0.000589] | 0.9915 | +0.000630 | 0.9930 |
| `market_light_3` | +2.20 | +0.000181 | [+0.000030, +0.000348] | 0.9870 | +0.000374 | 0.9895 |
| `state_heavy_3` | +1.10 | +0.000137 | [+0.000026, +0.000261] | 0.9925 | +0.000287 | 0.9940 |
| `uniform` | 0 | 0 | — | — | 0 | — |
| `state_light_3` | −1.10 | −0.000180 | [−0.000304, −0.000070] | 0.0005 | −0.000366 | 0.0005 |
| `market_heavy_3` | −2.20 | −0.000227 | [−0.000368, −0.000105] | 0.0005 | −0.000461 | 0.0000 |
| `market_heavy_10` | −4.61 | −0.000357 | [−0.000588, −0.000161] | 0.0005 | −0.000724 | 0.0000 |

**Perfectly monotone in the direction parameter, antisymmetric about zero, and
every one of the six intervals excludes zero.** Three arms in the hypothesised
direction improve probability quality; the three equal-and-opposite controls
degrade it by a matching amount. That is a dose-response curve, not a lucky
draw, and it is the strongest form of evidence this screen can produce: the
falsification controls did what a real effect requires them to do.

The **magnitude is very small**: +0.0003 Brier on a base of 0.2498, i.e. 0.12%
relative. For scale, MOD-08's ECDF-smoothing finding is worth Brier −0.0015 —
five times larger — on NFL data.

At alpha=10 none of this exists: every arm's Brier delta is under 3.4e-5, every
accuracy delta under 0.09 points, and every interval straddles zero. The block
structure has nothing to allocate when the penalty is inert.

### 4d. The headline comparison against the config actually running

Everything above compares arms at a fixed alpha, which isolates the block
allocation. The deployment question crosses both axes: how does the best
configuration compare to what is running today? `scripts/groupwise_ridge_headline.py`,
`market_light_10 @ alpha=1e4` against the frozen `uniform @ alpha=10`.

| metric | clean core (8,933) | 95% week-blocked | `P+` | all (11,780) | `P+` |
|---|---:|---|---:|---:|---:|
| forced-pick accuracy | 0.5213 vs 0.5160 (**+0.537 pts**) | [−0.419, +1.532] | 0.8735 | +0.314 pts | 0.7515 |
| Brier | 0.24947 vs 0.24997 (**+0.000496**) | [+0.000051, +0.000943] | 0.9840 | +0.000804 | 0.9930 |
| log loss | 0.69206 vs 0.69303 (**+0.000963**) | [+0.000052, +0.001868] | 0.9795 | +0.001698 | 0.9925 |

Season-blocked intervals agree in sign and are slightly wider (accuracy `P+`
0.8295; Brier `P+` 0.9655).

**And the single most honest number in this document:** the change flips
**2,232 of 11,989 picks (18.6%)**, and on the 2,195 flipped-and-evaluated games
the candidate is right **1,116** times against the baseline's **1,079**. That is
**50.8%** on the picks it moves. The mechanism relocates a fifth of the slate
and has essentially no directional skill on what it relocates. Mean |predicted
residual| falls from 1.13 to 0.74 — the candidate is a much more conservative,
better-calibrated model that is not a better *picker*.

Decomposing the Brier gain: global alpha alone (`uniform` 10 -> 1e4) is worth
+0.000192, week-blocked [−0.000327, +0.000672], `P+` 0.767 — **unresolved**. The
block allocation at alpha=1e4 is worth +0.000304, [+0.000055, +0.000589],
`P+` 0.9915 — **resolvable**. So the group-wise part is the load-bearing half of
the calibration gain, which is the one thing here that genuinely belongs to this
method rather than to the alpha correction.

---

---

## 5. Verdict

**1. Structural: group-wise ridge escapes the MOD-06 corollary. Decisively.**
It is not a rescale — it changes the direction of the coefficient vector — and
the implementation confirms it rather than asserting it: 18.6% of scored picks
move against the frozen configuration, 12.5% against a same-alpha uniform
baseline. A positive rescale moves zero. **The ROADMAP's MOD-06 entry states its
corollary too broadly** and should be narrowed: ridge penalty changes, global or
group-wise, are *not* in the class of methods it closes. This matters beyond
this screen, because that sentence is currently being used as a general filter.

**2. Forced-pick accuracy: category 3, unresolved below detection power. Not a
positive, and not a negative either.** +0.537 points against the frozen config,
week-blocked [−0.419, +1.532], `probability_positive` 0.8735. Four reasons not
to believe it: the interval contains zero; it is the best of 21 predeclared
configurations plus one cross-axis selection; the dose-response in accuracy is
non-monotone; and **the picks it moves are won at 50.8%** (1,116 of 2,195). The
last is the decisive one. A mechanism that relocates a fifth of the slate at a
coin flip is not a picking improvement, whatever its headline number does.

**3. Calibration: a real, resolvable effect — and small.** Brier and log loss
show a perfectly monotone, antisymmetric dose-response across the symmetric
grid, with all six intervals excluding zero and the equal-and-opposite
falsification controls degrading by a matching amount. This is as clean as
evidence gets here. It is also worth +0.0003 Brier (0.12% relative) — a fifth of
MOD-08's ECDF-smoothing effect. Per MOD-06's own note, calibration and
confidence ordering are where a shrinkage-family method can pay, and the pool's
**Best Pick** lever is exactly a confidence-ordering problem
(`docs/pool_edge_plan.md`: our ordering is currently flat, top-1 scored 48.6%
over 107 weeks). **That, not headline accuracy, is where this belongs.**

**4. The finding that outranks the whole screen: `ridge_alpha = 10.0` is inert.**
On the NFL active model at the current training size, the median principal
direction is shrunk by **0.29%**, and the weakest decile of real directions by
16%. The production model is unregularised least squares in all but name. This
is an undocumented default doing nothing, which is a defect under the standing
rule that an unjustified number is broken until derived — and it explains the
lead: the `player_qb_continuity` ±1.1033 pair was two models differing by a
fraction of a percent, giving a calibration of how much accuracy this evaluator
manufactures from nothing.

It also sets the order of work. **Deriving the global penalty comes first;
allocating it across blocks comes second.** Screening block ratios on top of an
arbitrary global level answers the second question before the first, and the
results above show why: at the frozen alpha the entire block axis is a no-op
(every accuracy delta under 0.09 points, every Brier delta under 3.4e-5, every
interval containing zero).

### Does this deserve an NFL confirmation window?

**No — not now, and not for accuracy.** Opener-graded capacity is three windows
in the whole 2020-2025 pool. Spending one on a category-3 accuracy result that
is a max over 22 configurations, whose moved picks run at 50.8%, would be
precisely the mistake `docs/pool_edge_plan.md` warns against. The recommended
sequence instead:

1. **Derive `ridge_alpha`** (free, CFB + non-reserved seasons): walk-forward
   selection on strictly prior data, or a target shrinkage level read off the
   eigenspectrum. Both the frozen 10.0 and this screen's 1e4 are undefended
   numbers.
2. **Re-screen the block grid on top of the derived alpha**, still on CFB. The
   present result says the block axis only exists once the penalty is live, so
   this is the honest version of the experiment.
3. **Route the calibration gain to the Best Pick ranker**, where the metric is
   top-k-per-week accuracy and a Brier/log-loss improvement is the right
   currency. This needs no headline-accuracy claim and no window.
4. Only if (2) survives with a monotone accuracy dose-response and `P+` past the
   0.90 bar MOD-07 used should a window be drawn — as a **new** family
   (`groupwise_ridge_penalties`), inheriting nothing from `player_qb_continuity`
   (different hypothesis), drawing the earliest eligible `nflverse_spread` block.

**Not recommended: a `registry/weak_signals.json` entry.** That registry's value
is that each signal's *sign* is one clean bit under the null. The sign of a
maximum over 21 arms is not, so folding it in would corrupt the sign test the
registry exists to run. The symmetric-grid dose-response is recorded here
instead, which is a stronger claim than the registry could hold anyway.

## Files

- `src/nfl_ats/margin.py` — `GroupPenaltyScaler`, `column_penalty_multipliers`,
  `resolve_feature_groups`, `margin_feature_groups`; `column_penalties`
  threaded through `make_margin_estimator`, `fit_margin_model`, `MarginModel`,
  and `margin_model_metadata`.
- `tests/test_margin_groupwise.py` — 49 tests: frozen-path bit-identity, the
  closed-form generalized-ridge identity, indicator-block inheritance, block
  coverage for all 32 target/profile combinations, and the pick-flip contrast
  against a positive rescale.
- `scripts/groupwise_ridge_screen.py` — the predeclared grid and the CFB sweep.
- `scripts/groupwise_ridge_headline.py` — best arm vs the frozen configuration.
