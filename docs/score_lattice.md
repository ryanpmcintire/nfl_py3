# Correct-score lattice for the pool tiebreaker (MOD-05, WP23)

ROADMAP row MOD-05 says the key-number lattice is real and large but buys
nothing for ATS cover probability, and closes with an instruction rather than
a dead end: *"Build it for push probability, alternative-line/half-point
questions and correct-score products — **not** for ATS accuracy."* This
document is the correct-score-products half of that instruction, applied to
the one place in the repository where an exact final score is actually
published: `nfl-ats tiebreaker`'s exact-score mode list.

**§1 below is a predeclaration. It was written and saved before a single
evaluation number existed.** §2 (Results) was appended afterwards.

---

## §1 Predeclaration (frozen 2026-09-01, before any outcome number)

### 1.1 What is being compared

**Comparator (the shipped arm).** `nfl_ats.tiebreaker.build_report`'s
`common_scores`: the kernel-weighted neighborhood of historically similar
market shapes, with each historical game's *actual* exact final
`(home_score, away_score)` counted at its kernel weight, ranked by weight and
then by score for determinism. This is what the tiebreaker report prints today
under "most common exact finals in the neighborhood (exact-score metric,
weighted)".

**Candidate (the lattice).** A joint distribution over the integer score
lattice, conditioned on the market `(spread, total)`, built from the empirical
joint distribution of **(home margin, total) residuals from the market** over
the training history, recentred on the blended guess and smoothed onto the
lattice:

1. Take the *same* kernel-weighted neighborhood the comparator uses — same
   triangular kernel, same `(1.0, 1.5)` base bandwidths, same continuous
   widening schedule, same Kish effective-sample-size floor of 150, same
   bisection. `nfl_ats.score_lattice` imports `_neighborhood`,
   `kernel_weights` and `effective_sample_size` from `nfl_ats.tiebreaker`
   rather than reimplementing them, so the two arms cannot drift apart.
2. For each neighborhood game *i* with weight `w_i`, form its market
   residual pair

   ```
   r_margin_i = actual_margin_i - spread_line_i
   r_total_i  = actual_total_i  - total_line_i
   ```

   and recentre it on the target game's blended guess:

   ```
   margin_i' = guess_margin      + r_margin_i
   total_i'  = guess_total_line  + r_total_i
   ```

   Equivalently, and this is how it is implemented because the lattice lives
   in score space, `home_i' = guess_home + (actual_home_i - implied_home_i)`
   and `away_i' = guess_away + (actual_away_i - implied_away_i)`. The two
   forms are the same linear bijection `(margin, total) <-> (home, away)`;
   nothing is added or dropped by choosing one.
3. Spread each recentred point onto the integer lattice with a **product
   triangular kernel of bandwidth exactly 1 score point per coordinate**:

   ```
   K(h, a; h_i', a_i') = max(0, 1 - |h - h_i'|) * max(0, 1 - |a - a_i'|)
   ```

   **This bandwidth is derived, not chosen.** A triangular kernel whose
   half-width equals the lattice spacing is the unique mass-preserving linear
   interpolation of a real-valued point onto that lattice: for any real `x`,
   `sum over integers n of max(0, 1 - |n - x|) == 1` exactly. So each game's
   weight `w_i` is distributed over the at-most-four surrounding integer
   score pairs and nothing is created or destroyed. Any wider bandwidth would
   be a free parameter; any narrower one would not be mass preserving.
   (AGENTS.md: underived constants are defects. This introduces none.)
4. Restrict to the **feasible score set**, enumerated from the training
   finals themselves and never from a hand list: `V = {every team score that
   has occurred in the training data}`, and the lattice support is
   `S = V x V`. NFL scoring makes 1 and 4 unreachable in practice and the
   data says so on its own; the walk-forward version of `V` also grows over
   time, which is the honest behaviour.
5. Normalise over `S`.

The candidate's **products** are then the top-3 exact finals (`argmax` by
probability, ties broken by score tuple exactly as the comparator does), the
modal and median totals implied by the same distribution, and `P(push)` at
any line — `P(home margin == line)`, which is identically zero at a
half-point line by construction.

**The centre is held identical between the arms.** In the walk-forward there
is no weekly forecast and no totals model view, so both arms are centred on
`(spread_line, total_line)`. The contrast is therefore purely
raw-final-counting vs residual-recentred-lattice-smoothing, with no confound
from the blend weights.

### 1.2 Declared tension, stated before the numbers

MOD-05's own finding is that the key-number lattice lives in the **raw**
margin and that "the ATS residual is one line-varying convolution away from
smooth". Recentring by the market residual therefore *smears* the 3/7 spikes
by the amount each neighborhood game's line differs from the target's. So
there is a real mechanism by which the candidate could lose top-1 hit rate
even while being a better calibrated distribution. That prediction is written
here before the result, not after it.

For that reason a **secondary arm** is declared now: `raw` — identical in
every respect except that step 2 is skipped (each neighborhood game's actual
final is smoothed onto the lattice with no recentring). It isolates
"smoothing" from "recentring". It is a second look and is disclosed as such;
the primary arm for the decision rule is the residual lattice.

### 1.3 Evaluation protocol

- **Chronological walk-forward, one prediction per lined game.** Target
  games are every completed game with a recorded spread and total in
  **2012-2025**. Training for a target game in `(season, week)` is every
  lined final strictly before `(season, week)` in chronological order —
  never the target week itself, never the future.
- **Warm-up: seasons 2009, 2010 and 2011 are training-only** (801 lined
  finals). The first scored game is 2012 week 1, whose training set is those
  three seasons. Chosen so the first evaluation neighborhood can clear the
  ESS floor of 150 from a full three-season base.
- **Feasible score set is recomputed from the training slice at every
  target week.** A leakage test pins this.
- Within-week correlation is zero for this project (owner mandate), so week
  blocks are the resampling unit.

### 1.4 Metrics, frozen

**(a) Exact-score hit rate** — the headline, and the decision metric.
Per game, two Bernoulli indicators for each arm: `top1_hit` (the arm's
single best exact final equals the realised final) and `top3_hit` (the
realised final is among the arm's top three). Paired per-game difference
`candidate - comparator`, in **accuracy points** (percentage points), with
`nfl_ats.clv.week_blocked_bootstrap` over `(season, week)` blocks,
2,000 resamples, reporting the estimate, the 95% interval and
`probability_positive`.

**(b) Log loss** on the realised final. Both arms are turned into
distributions over the same support `S` and scored with `-log P(realised)`.
Zero counts are handled identically for both arms by **one pseudo-observation
spread uniformly over `S`**:

```
P(h, a) = (W(h, a) + 1 / |S|) / (W.sum() + 1)
```

where `W` is that arm's own un-normalised weight grid over `S`. One
pseudo-count in total is the minimum amount of smoothing that makes the score
finite, expressed in the same units as the data; the formula is identical for
both arms and each side sums to exactly 1 over `S`, so the comparison stays
paired. A realised final outside `S` receives the same floor an in-support
zero-count cell receives, for both arms; the count of such games is reported. Reported as a **log-loss improvement** (comparator minus candidate,
so positive favours the candidate), paired, week-blocked.

A **secondary diagnostic** is declared with it: log loss restricted to the
games where the comparator assigns non-zero weight to the realised final. The
full-sample number will be dominated by the comparator's zeros — a mode list
is not a probability distribution — and that diagnostic separates "the
comparator has zeros" from "the lattice is sharper where both are alive".

**(c) Closest-total absolute error.** The shipped closest-total answer is
`round(weighted median actual total)`. Two lattice answers are reported
against it: the lattice's **modal** total (`argmax_T P(total = T)`) and the
lattice's **median** total. Median is the |error|-minimising statistic and
mode is not, so the modal number is expected to be the worse of the two; both
are reported rather than only the flattering one. Paired, week-blocked, in
total points, reported as `mae_improvement` (comparator minus candidate).
Declared expectation: **this must not be resolvably worse**, i.e. the
candidate is not allowed to buy exact-score hits by wrecking the closest-total
answer.

**(d) `P(push)` calibration.** Mean lattice `P(home margin == spread_line)`
against the realised push rate, bucketed by `nfl_ats.key_numbers.line_bucket`
(`|line| < 3`, `|line| = 3`, `3.5 <= |line| <= 6.5`, `|line| = 7`,
`|line| > 7`) — reusing the existing bucketing rather than inventing one.
Descriptive; no decision hangs on it.

**Positive control.** A third arm, `oracle_total`, identical to the residual
lattice except that it conditions on the realised total: every lattice cell
with `h + a != actual_total` is zeroed and the rest renormalised. It must
drive the exact-score hit rate up sharply. Its purpose is to prove the
instrument can detect a hit-rate improvement at all, so that a null on the
real candidate can be classified honestly rather than by assumption.

### 1.5 Decision rule, frozen

The pool submits a tiebreaker either way, so this is an expected-value
decision, not a threshold-clearing one (AGENTS.md: "a promotion bar is not a
decision bar").

> **If `probability_positive` > 0.5 on metric (a) top-1 exact-score hit
> rate, the tiebreaker report's exact-score section switches to the lattice's
> top-3, keeping the neighborhood mode list as a second line. Otherwise
> `tiebreaker.py` is not touched.**

Metric (c) is a guard: if the lattice's *median* total were resolvably worse
on closest-total error, that would be reported prominently alongside the
switch, because the pool's tiebreak metric is not recorded anywhere in this
repository and both metrics have to keep working.

### 1.6 Recording

Family `score_lattice_tiebreaker`, league `nfl`, seasons 2012-2025, one
registry entry per metric:

| name | metric | units |
|---|---|---|
| `score_lattice_top1_exact` | (a) top-1 exact-score hit rate | `accuracy_points` |
| `score_lattice_top3_exact` | (a) any-of-top-3 exact-score hit rate | `accuracy_points` |
| `score_lattice_log_loss` | (b) log loss on the realised final | `log_loss_improvement` |

Classification is `unresolved_below_power` unless the binding AGENTS.md
taxonomy is *literally* met: a terminal classification needs a RESOLVED wrong
sign (the whole interval below zero), zero split-half reliability, or a
positive control proven able to detect an effect that size and finding it
absent. An interval containing zero is never grounds to reject, fail or close
anything.

### 1.7 Artifacts

`artifacts/score_lattice/<UTC stamp>/` holds `predictions.csv` (one row per
scored game per arm, prediction-level output preserved), `summary.json` (the
bootstrap tables and per-season breakdown) and `manifest.json` (inputs,
protocol, git head).

---

## §2 Results

Everything below is **measured 2026-09-01** by
`.\.tools\uv.exe run --no-sync python scripts\score_lattice_eval.py --mode
positive-control` then `--mode screen`. Artifacts:
`artifacts/score_lattice/20260901T192512Z` (positive control),
`artifacts/score_lattice/20260901T192552Z` (screen — the run the registry rows
cite) and `artifacts/score_lattice/20260901T192852Z` (screen, re-run after the
secondary arm's push column was added to the descriptive table; its bootstrap
table is bit-identical, verified field by field). Each carries an
`artifact_provenance()` stamp and an experiment-registry row under
`registry/experiments/score-lattice-eval/`.

Scope: **3,829 lined games, 2012-2025, 299 week blocks**, one prediction per
game, training strictly before the target week, warm-up 2009-2011 (801 lined
finals, training-only).

### 2.1 What this implies for the decision, before what is wrong with it

**1. The frozen decision rule fires NO, and `tiebreaker.py` was not touched.**
Top-1 exact-score hit rate is **-0.209 accuracy points, 95% [-0.495, +0.077],
`probability_positive` 0.0615** — the rule required `probability_positive`
> 0.5. Playing the lattice's top-3 in place of the shipped mode list would be
taking the 6/94 side of that bet, and on the top-3 product the sign is
resolved against it outright. The published Week 1 exact-score line is
unchanged.

**2. The closest-total half of the tiebreak goes the OTHER way, and it is
resolved.** The lattice's **median** total lands **0.263 total points closer**
to the realised total than the shipped weighted median — 10.547 vs 10.810 MAE,
week-blocked 95% **[+0.162, +0.366]**, `probability_positive` **1.0000**, on
the same 3,829 paired games. That is a free improvement to the other half of
the tiebreaker, sitting behind an unanswered question this repository has
never recorded: *what metric does the pool actually break ties on?* If it is
closest-total, this is worth switching on and the switch is one line. It was
not wired here because the predeclared decision rule hung on the exact-score
metric, and moving a goalpost after seeing the numbers is the failure mode
AGENTS.md exists to stop.

**3. MOD-05's push-probability and alternative-line products now exist in
code, and the measurement says which lattice to build them from — the
UN-RECENTRED one.** The un-recentred arm is *analytically identical* to the
shipped mode list on exact scores (§2.3), so adopting it publishes nothing
new and moves no printed score, yet it answers `P(push)`, `P(margin = m)` and
any alternative-line question — none of which a mode list can answer at all —
and it is materially better calibrated at exactly the key numbers MOD-05 is
about: at `|line| = 3` its gap to the realised push rate is **-0.81pp against
the recentred lattice's -3.10pp**, and at `|line| = 7` **-1.84pp against
-2.47pp** (§2.5). The instruction on the MOD-05 row — *build it for push
probability, alternative-line/half-point questions and correct-score products*
— is satisfied by `nfl_ats.score_lattice` with `recentre=False`.

**4. The reason the candidate lost is the one written down before the run.**
§1.2 predicted, in writing, that recentring by the market residual would smear
the raw key-number spikes MOD-05 measured. It does, and the smear is now
quantified in two independent places: the exact-score top-3 loss (§2.2) and
the key-number push under-pricing (§2.5). This is a mechanism confirmed, not a
null.

### 2.2 Metric (a) and (b): the frozen paired comparison

Candidate minus comparator, week-blocked bootstrap, 2,000 resamples, 299
blocks, 3,829 games. Positive favours the lattice throughout.

| metric | estimate | 95% low | 95% high | `probability_positive` |
|---|---|---|---|---|
| **(a) top-1 exact score** (accuracy points) | **-0.209** | -0.495 | +0.077 | **0.0615** |
| **(a) any-of-top-3 exact score** (accuracy points) | **-1.175** | -1.636 | **-0.689** | **0.0000** |
| **(b) log loss on the realised final** (improvement) | **+0.560** | +0.441 | +0.683 | **1.0000** |
| (b) same, restricted to games where the comparator is alive | -1.946 | -2.086 | -1.807 | 0.0000 |
| **(c) closest-total MAE, lattice MEDIAN total** (improvement) | **+0.263** | +0.162 | +0.366 | **1.0000** |
| (c) closest-total MAE, lattice MODAL total (improvement) | -0.420 | -0.575 | -0.271 | 0.0000 |
| secondary arm (no recentring), top-1 | 0.000 | 0.000 | 0.000 | — |
| secondary arm (no recentring), top-3 | 0.000 | 0.000 | 0.000 | — |
| secondary arm (no recentring), log loss | 0.000 | 0.000 | 0.000 | — |

Raw rates: top-1 **0.418% lattice vs 0.627% mode list**; top-3 **0.783% vs
1.959%**; mean log loss **9.217 vs 9.777**; closest-total MAE **10.547
(lattice median) / 11.230 (lattice mode) vs 10.810 (shipped)**.

**The (b) win is entirely the comparator's zeros, and that is said out loud
rather than banked.** The mode list assigns zero weight to the realised final
in **55.1%** of games (it is alive in 44.9%); on the 44.9% where both arms
have real mass, the lattice is **resolvably worse** (-1.946, whole interval
below zero). The correct reading of the +0.560 headline is *a mode list is
not a probability distribution*, which is a true and useful statement about
the shipped tool but is not evidence that the recentred lattice is sharper.

The modal-total row is worse than the median-total row exactly as §1.4
predicted: the median minimises absolute error and the mode does not. Both
are printed so the flattering one cannot be quoted alone.

### 2.3 The secondary arm has an analytic answer: smoothing alone is the identity

Every secondary-arm delta is **exactly 0.000**, not approximately. That is not
a bug and not a coincidence — it is the mass-preserving bandwidth doing what
it was derived to do. With no recentring, each neighborhood game's point is
already an integer final, the triangular kernel's fractional part is zero, and
all of its weight lands on its own cell. The lattice therefore *reproduces*
`weighted_score_counts` cell for cell, and `tests/test_score_lattice.py::
test_without_recentring_the_lattice_is_exactly_the_shipped_mode_list` pins it.

The consequence for MOD-05 is the useful part: **the entire measured effect,
in both directions, is the recentring. None of it is the smoothing.** So the
un-recentred lattice is a strictly free upgrade in *form* — same exact scores,
plus a distribution — and the recentring is the thing that has to earn its
place metric by metric. On totals it earns it (+0.263, resolved); on exact
scores and key-number pushes it does not.

### 2.4 Positive control

`--mode positive-control`, same instrument, same 3,829 games, same blocks. The
`oracle_total` arm conditions the lattice on the realised total:

| metric | estimate | 95% low | 95% high | `probability_positive` |
|---|---|---|---|---|
| oracle top-1 exact score (accuracy points) | **+6.346** | +5.554 | +7.122 | 1.0000 |
| oracle any-of-top-3 (accuracy points) | **+16.271** | +15.086 | +17.624 | 1.0000 |
| oracle log loss (improvement) | +4.241 | +4.121 | +4.359 | 1.0000 |

Raw oracle rates: top-1 **6.973%**, top-3 **18.229%**, log loss **5.537**.

**What the control does and does not license.** It proves the instrument can
detect an exact-score hit-rate movement of roughly **6 accuracy points**. It
was never shown able to detect a **0.2-point** one, so `bounded_by_control` is
*not* an admissible closure for metric (a) top-1 and is not claimed. Top-1 is
recorded `unresolved_below_power`. Metric (a) top-3 closes on a different and
independently admissible ground — its whole interval sits below zero — so it
is recorded `refuted_mechanism` / `wrong_sign_resolved`.

### 2.5 Metric (d): `P(push)` calibration by line bucket

Descriptive, both lattice arms. The secondary arm's column was added to this
table *after* the primary arm's calibration was seen — disclosed rather than
quietly folded in; metric (d) was declared descriptive with no decision
attached, and it cannot rescue the candidate because the decision rule had
already fired NO on §2.2.

| line bucket | games | lattice `P(push)` | un-recentred arm | realised | lattice gap | un-recentred gap |
|---|---|---|---|---|---|---|
| `3.5 <= \|line\| <= 6.5` | 1,293 | 1.530% | 1.153% | 0.696% | +0.83pp | +0.46pp |
| `\|line\| > 7` | 879 | 1.665% | 0.913% | 1.593% | +0.07pp | -0.68pp |
| `\|line\| < 3` | 877 | 1.857% | 1.091% | 0.570% | +1.29pp | +0.52pp |
| `\|line\| = 3` | 549 | 5.830% | 8.120% | **8.925%** | **-3.10pp** | **-0.81pp** |
| `\|line\| = 7` | 231 | 4.452% | 5.086% | **6.926%** | **-2.47pp** | **-1.84pp** |

The realised push rate at `|line| = 3` is **8.93%** and at `|line| = 7` is
**6.93%** — MOD-05's key-number lattice, visible in the outcome column. The
recentred lattice under-prices both by 2.5-3.1 percentage points and
over-prices the non-key buckets; the un-recentred arm tracks them far more
closely. Buckets are `nfl_ats.key_numbers.line_bucket`, reused unchanged.

### 2.6 Per season

`modes` = shipped mode list, `lattice` = recentred candidate, `oracle` =
positive control. Rates in percent; the un-recentred arm equals `modes`
exactly in every season (§2.3) and is omitted.

| season | games | modes top-1 | lattice top-1 | modes top-3 | lattice top-3 | oracle top-1 | oracle top-3 | all-history fallback | log-loss gain |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 267 | 1.498 | 0.375 | 2.247 | 1.498 | 7.491 | 19.101 | 89.5% | +0.498 |
| 2013 | 267 | 1.124 | 0.000 | 3.371 | 0.375 | 5.618 | 19.101 | 61.0% | +0.326 |
| 2014 | 267 | 0.000 | 0.000 | 1.873 | 0.375 | 5.243 | 15.730 | 46.1% | +0.091 |
| 2015 | 267 | 0.749 | 0.375 | 1.498 | 0.375 | 2.622 | 16.854 | 25.1% | +0.262 |
| 2016 | 267 | 1.124 | 0.375 | 2.247 | 0.749 | 4.869 | 18.727 | 22.8% | +1.006 |
| 2017 | 267 | 0.749 | 0.375 | 1.498 | 1.124 | 7.865 | 16.854 | 26.2% | +0.455 |
| 2018 | 267 | 0.749 | 1.124 | 2.247 | 1.498 | 9.363 | 17.978 | 34.5% | +0.736 |
| 2019 | 267 | 0.000 | 0.375 | 0.749 | 0.375 | 7.491 | 17.228 | 23.6% | +0.778 |
| 2020 | 269 | 1.115 | 0.000 | 2.230 | 0.000 | 8.178 | 19.703 | 33.5% | +0.797 |
| 2021 | 285 | 0.000 | 0.702 | 1.404 | 0.702 | 6.316 | 19.649 | 26.0% | +0.640 |
| 2022 | 284 | 0.352 | 0.704 | 3.169 | 1.761 | 6.338 | 17.958 | 14.8% | +0.629 |
| 2023 | 285 | 0.000 | 0.351 | 2.105 | 0.351 | 7.719 | 16.842 | 16.8% | +0.493 |
| 2024 | 285 | 0.702 | 0.351 | 1.404 | 0.702 | 7.719 | 20.351 | 8.8% | +0.633 |
| 2025 | 285 | 0.702 | 0.702 | 1.404 | 1.053 | 10.526 | 18.947 | 16.1% | +0.488 |

Top-1 is a sub-1% event on either arm, so single-season rows are two or three
games wide and should be read as texture, not as season stability. The top-3
column is where the comparator's advantage is consistent: the mode list is
ahead or level in **12 of 14** seasons.

### 2.7 Week 1 2026: DEN @ KC

Market **KC by 2.5, total 43** (snapshot `20260901T130035Z`, 11 books); blended
guess margin **+2.76**, blended guess total **43.04** (the same
`MODEL_RESIDUAL_WEIGHT` 0.2 / `TOTALS_RESIDUAL_WEIGHT` 0.1 blend the shipped
report uses). Kernel-weighted neighborhood: effective 150 games from 206
positively-weighted rows, bandwidths ±1.04 margin / ±1.58 total. Support
`|S|` = 3,721 feasible finals (61 distinct team scores, from the data).

**Lattice top-5 exact finals**, with the un-recentred arm — which is the
shipped mode list, normalised — beside them:

| exact final | recentred lattice | un-recentred (= shipped mode list) |
|---|---|---|
| KC 17 - DEN 23 | 1.682% | 2.202% |
| KC 20 - DEN 23 | 1.434% | **2.574%** |
| KC 23 - DEN 20 | 1.357% | 2.081% |
| KC 28 - DEN 24 | 1.178% | 1.552% |
| KC 20 - DEN 17 | 1.015% | 0.993% |

The shipped report's three printed modes are KC 20 - DEN 23, KC 17 - DEN 23,
KC 23 - DEN 20, which is the un-recentred column's own top three. **Nothing
about the published card changes.**

**Push and alternative-line products** (what MOD-05 asked for):

- `P(push at the market line KC -2.5)` = **0.0000%** — exactly zero, by
  construction, because a half-point line cannot push. That is the right
  answer and the mode list cannot state it.
- If the line moved to **KC -3**: `P(push)` = **6.83%** (recentred) /
  **7.95%** (un-recentred). §2.5 says the un-recentred number is the better
  calibrated of the two at `|line| = 3`, where the realised rate over
  2012-2025 is 8.93%.
- If the line moved to **KC -2**: `P(push)` = **4.79%** / **3.92%**.
- Key-number mass, recentred / un-recentred: margin **+3** 6.83% / 7.95%;
  **-3** 5.23% / 7.18%; **+7** 3.23% / 3.53%; **-7** 3.62% / 4.85%; a **tie**
  1.39% / 0.96%. The recentred arm holds less mass on every one of the four
  key numbers and more on the tie — the smearing, on one live game.
- Lattice modal total **40**, median total **41**; the shipped closest-total
  guess is **KC 22 - DEN 19**, total **41**. The two agree this week.

### 2.8 Recorded

Family `score_lattice_tiebreaker`, league `nfl`, seasons 2012-2025, all via
`nfl-ats weak-signals record` under the session's cross-process lock:

| name | effect | units | interval | `probability_positive` | classification |
|---|---|---|---|---|---|
| `score_lattice_top1_exact` | -0.208932 | `accuracy_points` | [-0.495, +0.077] | 0.0615 | `unresolved_below_power` |
| `score_lattice_top3_exact` | -1.175242 | `accuracy_points` | [-1.636, -0.689] | 0.0000 | `refuted_mechanism` (`wrong_sign_resolved`) |
| `score_lattice_log_loss` | +0.560004 | `log_loss_improvement` | [+0.441, +0.683] | 1.0000 | `unresolved_below_power` |
| `score_lattice_closest_total` | +0.262732 | `mae_improvement` | [+0.162, +0.366] | 1.0000 | `unresolved_below_power` |

§1.6 named the first three. The fourth is metric (c), frozen in §1.4 but not
listed in §1.6; it is recorded and the addition is disclosed here, because the
row left unrecorded would otherwise have been the only *positive* one.
`score_lattice_log_loss` and `score_lattice_closest_total` are both resolvably
positive and carry no closing ground — the classification enum has no
"resolved positive" state, so `unresolved_below_power` is the non-terminal
option and the evidence field says so explicitly.

### 2.9 What is wrong with it

- **31.4% of scored games fell back to the "all history" unweighted
  neighborhood**, concentrated early (89.5% of 2012, 61.0% of 2013, 46.1% of
  2014, under 27% from 2015 on) because a 3-season warm-up cannot fill a
  150-ESS neighborhood at every market shape. It degrades both arms
  identically, so the paired contrast stays fair, but the early seasons are a
  weaker test of the real tool, which has 4,630 games behind it.
- **Exact-score hit rate is a sub-1% event**, so metric (a) top-1 rests on 16
  lattice hits against the mode list's 24, out of 3,829 games. The interval is
  honest about that; the point estimate should not be read as a precise -0.21.
- **The log-loss headline is not what it looks like** (§2.2), and the
  restricted-support diagnostic that says so was predeclared precisely because
  the headline would otherwise be quotable in the wrong direction.
- **11 of 3,829 realised finals (0.29%) fell outside the training-derived
  feasible support** and were scored at the shared floor. Symmetric between
  arms, but it is a floor value rather than a modelled one.
- **The walk-forward has no model or totals-model view**, so it grades the
  lattice *construction* with both arms centred on the raw market. The live
  tiebreaker centres on the blended guess. That is the right contrast for this
  question and the wrong one for "how good is the shipped guess" — the latter
  is the module docstring's existing framing, unchanged: a tiebreaker guess is
  a coin toss weighted a few points in your favour, not a prediction.
- **The push-calibration table has no interval.** It is a descriptive
  five-bucket comparison, not a paired bootstrap, and no registry row claims
  it. A predeclared paired push-calibration screen (Brier or log loss on the
  push indicator, recentred vs un-recentred) is the obvious next measurement
  and was not run here.
