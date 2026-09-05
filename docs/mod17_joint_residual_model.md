# MOD-17 research half: a joint margin/total residual model

Predeclared 2026-09-05, before any number in this document past this line was
computed. Lane AC, overnight fleet.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Decide on expected value:
`probability_positive` above 0.5 favours the candidate; grade play/no-play at
the OPENER.

## The owner's question

2026-09-05, verbatim: "if our spread prediction disagrees with the total
prediction, we need to understand why we dont have a unified model" and
"wouldn't [separate models looking fine] only be true because we have
significant errors in both of those models?"

This is the research half of ROADMAP MOD-17 (the served-numbers-are-one-
lattice contract is the other half, already landed). The question here is
whether fitting margin and total residuals as one statistical object beats
fitting them as two, and how big that gain could ever be given how correlated
their errors already are.

## Frozen contract

**Population.** `data/processed/game_features_weak_stack.parquet` (production
margin table; read 2026-09-05: it is a strict column superset of both the
totals wave-1 allowlist and the totals wave-2 drive-pace allowlist -- verified
by set difference, zero missing columns either way) joined against its own
`home_score`/`away_score`/`spread_line`/`total_line`/`result`/`ats_margin`
columns. Regular season only (`game_type == "REG"`), rows with both targets
finite:

- `margin_residual = ats_margin` (already the production margin target,
  `result - spread_line`).
- `total_residual = (home_score + away_score) - total_line` (identical
  definition to `nfl_ats.totals`'s target).

**Feature set for every "union"/"joint" fit.** The union of the production
margin `weak_stack` profile's 90 columns and the totals wave-2 24-column
drive-pace family (`nfl_ats.totals_wave2.WAVE2_DRIVE_FEATURES`) -- 114 columns
total, verified zero overlap between the two source lists. The totals wave-1
41-column allowlist (`nfl_ats.totals.TOTALS_FEATURES`, including `total_line`
itself) is already a strict subset of the 90 margin columns (verified by set
difference: zero totals-wave-1 columns absent from margin `weak_stack`), so
"production margin features plus `total_line` plus totals' wave-2 columns" is
exactly this union; no separate reconciliation step is needed.

**Pipeline.** `SimpleImputer(median, add_indicator=True) -> StandardScaler ->
Ridge(alpha=10.0)` -- production's exact recipe, reused unmodified from
`nfl_ats.totals.make_totals_estimator` (this is also `margin.py`'s frozen
default pipeline for the `ridge` model). `alpha=10.0` throughout; no new value
is swept.

**Stage 1 ("the joint model"): a multi-output ridge.** `sklearn.linear_model
.Ridge` fit once on the 114-column union design matrix against a 2-column
target `[margin_residual, total_residual]`. **Predeclared mathematical fact,
checked by a unit test before any real number is read:** ordinary ridge
regression with a 2-D target is column-independent -- each output column's
fitted coefficients solve `(X'X + alpha I)^-1 X'y_j` on its own, with no term
that depends on the other column. So this "joint model" arm is provably
IDENTICAL to fitting two separate single-target ridges on the same 114-column
union design matrix ("marginal-on-union"). The comparison this arm actually
measures is therefore **the effect of the wider union feature set**, not of
joint estimation -- and this document states that distinction before any
number is read, specifically so a positive result here cannot later be
mis-sold as evidence of "jointness".

**Stage 2 ("reduced-rank / SUR-lite", the cheap variant that can actually
show coupling).** A second-stage ridge (`alpha=10.0`, same recipe) regressing
each target on the OTHER model's stage-1 out-of-fold prediction plus its own:
`margin_residual ~ [predicted_margin_residual, predicted_total_residual]`
(2 inputs) and symmetrically for `total_residual`. Trained walk-forward on
strictly earlier blocks of stage-1's own out-of-sample output (so a block's
stage-2 fit never sees a stage-1 prediction that used its own or a later
block's data). This is the one arm that can show a real cross-target
coupling effect, because it is not decomposable into independent per-column
fits the way stage 1 is.

**Protocol -- two separate walk-forwards, matching the two harnesses already
governing each side of production:**

- *Total side*: expanding-window walk-forward by `(season, week)` block over
  the FULL scored population (`nfl_ats.totals.chronological_blocks`/
  `design_matrix`, `min_train_games = 500`, reused unmodified), predicting
  every block once 500 prior games exist. Same population, same guard, same
  pipeline constructor as `docs/totals_model.md`'s already-run regime; this
  screen's own re-run of the wave-1 baseline on this table must reproduce
  that regime's numbers (checked below, "Verified this session").
- *Margin side*: graded at the Tuesday opener via the SAME game/week set
  `nfl_ats.clv.opener_pick_evaluation` already produces for the production
  `weak_stack` baseline (the 1,537-game, ~2020-2025 archive) -- not
  re-derived from the odds archive a second time. For every archived week,
  a joint-model fit is trained on every population game strictly before that
  week's earliest kickoff (same cutoff rule `opener_pick_evaluation` uses)
  and scored at both the archived Tuesday-opener and closing spread, with
  only `spread_line` swapped between the two (identical, previously-declared
  approximation: every other feature, including `total_line`, stays at its
  close-era value). **Scope decision, disclosed rather than silently
  taken:** the joint arm is graded by the SIGN rule only
  (`predicted_margin_residual > 0`), the predeclared historical rule in
  `docs/opener_evaluation.md`, not by the calibrated probability rule
  production actually plays -- building a calibrated out-of-time residual
  distribution for a two-output estimator is a materially larger, separately
  scoped piece of work. The baseline is read under the SAME sign rule for
  parity, and its own probability-rule number is reported for context only.

**Arms reported:**

1. **Base marginal margin model** -- production `weak_stack` (90 columns,
   single target), opener sign rule, via `opener_pick_evaluation` unmodified.
2. **Joint margin output** -- stage 1's margin column (114-column union,
   two-target ridge; mathematically the same as a 114-column single-target
   ridge, per the predeclared fact above), opener sign rule, on the identical
   archive game/week set.
3. **Base served total** -- production's frozen served blend,
   `total_line + 0.1 * predicted_residual` (`k = 0.1`, `docs/totals_model.md`,
   totals wave-1 41 columns), re-fit fresh on this table's population so the
   game set matches exactly.
4. **Joint total output** -- stage 1's total column (114-column union), its
   OWN MAE-minimizing blend weight from the same k-sweep `nfl_ats.totals`
   already uses (mirrors `nfl_ats.totals_wave2`'s wave-2-vs-wave-1 comparator
   convention: baseline graded at its frozen k, candidate at its own swept
   minimum).

Plus, not scored as a headline arm but reported for part 3 of the owner's
question: stage 2's out-of-sample R² on both targets, and each stage-1 arm's
own out-of-sample R² relative to a "trust the market fully" (predict zero
residual) baseline, so the reader can see how much of each residual's
variance either model reaches.

**Metrics.**

- Margin side: paired per-game sign-rule correctness, week-blocked bootstrap
  (`nfl_ats.clv.week_blocked_bootstrap`, 20,000 resamples, seed 20260905),
  `accuracy_points` units, `probability_positive` reported.
- Total side: paired per-game |error| improvement, week-blocked bootstrap
  (`nfl_ats.totals.bootstrap_improvement` / `nfl_ats.totals_wave2
  .bootstrap_wave_vs_wave`, 2,000 resamples, seed 20260901 matching the
  existing totals regime's seed), `mae_improvement` units (positive = the
  candidate's absolute error is smaller), `probability_positive` reported.
- Part 1's two correlations (predicted-vs-predicted, realised-vs-realised)
  are reported with per-season breakdowns and a SEASON-blocked bootstrap
  (`week_blocked_bootstrap(..., block="season")`), per the task's explicit
  instruction to block on season for this part; `correlation` units in the
  registry.

**Positive control.** One union feature column (`home_point_diff`, an
existing, weak, already-present column, chosen the same way
`nfl_ats.totals_wave2`'s own positive control chose
`home_drive_points_per_drive` -- an arbitrary, pre-chosen, already-present
member of the design matrix) is replaced by that row's own `margin_residual`
value (unit slope, zero noise -- identical method to
`nfl_ats.totals_wave2.run_positive_control`) for BOTH stage-1 walk-forwards
(the opener-graded margin arm and the full-population total arm). Expected
shape, frozen before running: the margin-side accuracy delta reads hugely
positive (the instrument can plainly detect an effect this large when it is
actually there); the total-side delta is not predeclared to be large, because
what is leaked is margin truth, not total truth, and the two residuals are
not known in advance to correlate strongly enough for a margin leak to also
resolve the total.

**Rotation / promotion disclosure.** This runs the opener side on the SAME
1,537-game Tuesday-opener archive several other confirmations have already
graded arms against (e.g. `docs/player_arrests_policy_eval.md`). Per that
document's precedent, this is disclosed as a promotion-style look on REUSED
history, not a fresh confirmation, and spends no rotation window. No
`nfl-ats rotation record-look` entry is made for it.

**Recording.** Three `nfl-ats weak-signals record` entries: the margin-side
opener delta (`accuracy_points`, family `mod17_joint_residual`), the
total-side MAE delta (`mae_improvement`, family `mod17_joint_residual`), and
the realised-residual correlation (`correlation`, family
`mod17_joint_residual`, informational -- there is no "candidate" for a
structural correlation between two residuals, so `favours_candidate` is not
a meaningful read for that one row and its notes say so explicitly). Every
entry classified `unresolved_below_power` unless the measured numbers meet an
admissible closing ground (a whole-interval-below-zero wrong sign, or the
positive control failing to move as predicted -- neither is expected, per the
shape above, but this document commits to classifying by the rule regardless
of which way the run lands).

**Decision rule, fixed before the run.** Promote the joint margin output only
if its opener `probability_positive` exceeds 0.5 with a non-negative point
estimate; promote the joint total output only if its `probability_positive`
(vs the served k=0.1 blend) exceeds 0.5 with a non-negative point estimate.
Anything else stays exactly where it is: recorded, not played, and not framed
as a negative.

## Verified this session, before scoring

- `data/processed/game_features_weak_stack.parquet`: 4,902 rows, 286 columns.
  Every one of `nfl_ats.totals.TOTALS_FEATURES` (41) and
  `nfl_ats.totals_wave2.WAVE2_DRIVE_FEATURES` (24) is present; zero missing.
- `nfl_ats.margin.margin_feature_columns("market_residual", "weak_stack")`
  returns 90 columns, of which all 41 totals wave-1 columns (including
  `total_line`, `spread_line`) are already members -- so "production margin
  features plus `total_line`" needed no additional column.
- The 24 `WAVE2_DRIVE_FEATURES` share zero names with the 90 margin columns,
  so the union is exactly 90 + 24 = 114 columns.

## Status

**RUN 2026-09-05.** `scripts/mod17_joint_residual_screen.py` ran once
end-to-end; every number below is **measured**
(`artifacts/mod17_joint_residual/20260905T160219Z/results.json`,
`registry/experiments/mod17-joint-residual-screen/20260905T160219Z.json`),
command:

```
.\.tools\uv.exe run --no-sync python scripts\mod17_joint_residual_screen.py
```

## Results (added after the run, 2026-09-05)

### Part 1: correlation of the two residuals

**Realised residuals** (the structural quantity), 4,431 regular-season games
2009-2025: Pearson r = **+0.0304**, season-blocked bootstrap (17 seasons,
2,000 resamples) 95% **[+0.0063, +0.0539]**, `probability_positive` **0.994**.
Small, but the season-blocked interval sits entirely above zero -- a game
that beats the total does lean, very slightly, toward also favouring the
side the margin model already leans toward. Per-season range -0.049 (2020)
to +0.104 (2014); 11 of 17 seasons positive.

**Predicted residuals** (the joint model's own two out-of-sample point
predictions, 3,919 games after the 500-game warm-up): Pearson r =
**+0.1344**, season-blocked bootstrap (15 seasons) 95% **[-0.0434,
+0.2650]**, `probability_positive` **0.812**. This is roughly 4x the
realised correlation -- expected and not itself informative about football:
the two predictions share the same 114-column feature set, the same
training folds, and the same imputer/scaler, so correlated model NOISE
inflates the apparent link well above the true structural correlation
measured directly on outcomes. Read the realised-residual number for the
football question, not this one.

### Part 2: joint vs marginal, four arms

**Margin side** (opener sign rule, 1,503 graded games, 107 week blocks,
production `weak_stack` baseline vs the joint model's margin column on the
114-column union): candidate accuracy 52.23% vs baseline 52.96%. Week-blocked
bootstrap (20,000 resamples, seed 20260905): **-0.732 accuracy points, 95%
[-2.831, +1.411], `probability_positive` 0.243.** Per-season deltas: 2020
+0.91, 2021 -2.97, 2022 -2.02, 2023 0.00, 2024 +3.38, 2025 -3.75 -- no
consistent direction. Baseline's own probability-rule opener accuracy for
context: 54.09% (sign rule 52.96%); this run compares sign rule to sign rule
throughout, per the predeclaration.

**Total side** (3,919 games, base served blend at the frozen k=0.1 vs the
joint model's total column at its own swept-minimum k, which is also 0.1):
week-blocked bootstrap (2,000 resamples, 260 week blocks, seed 20260901):
**+0.00491 mae_improvement, 95% [-0.00662, +0.01646], `probability_positive`
0.791.** Per-season deltas mixed: 8 of 15 seasons positive, largest movers
2013 (+0.0946) and 2018 (+0.0327). By the predeclared decision rule
(`probability_positive` > 0.5, non-negative point estimate), **this meets
the promotion bar for the joint total output; the margin side does not.**

**What this implies before what is wrong with it:** at `probability_positive`
0.791 the joint total output is the favourite over the served k=0.1 blend,
so the EV-correct action is to treat it as the better of the two -- not to
wait for its interval to clear zero, which the taxonomy above already rules
out as a bar. **What is wrong with treating this as a real "jointness" win:**
the raw (unblended) out-of-sample fit is measurably WORSE with the wider
114-column union than with the wave-1 41-column allowlist alone (see Part 3
below, `total_baseline_r2_vs_market` -0.0162 vs `total_union_r2_vs_market`
-0.0762) -- and, because the multi-output ridge is column-independent (the
predeclared fact this module is built around, pinned by
`tests/test_joint_residual_model.py::test_multi_output_ridge_matches_two_independent_single_target_fits`),
"joint total output" here is mathematically the SAME thing as "fit a
single-target ridge on the union features." The paired win is real but
riding on a small blend weight (k=0.1) that shrinks a noisier raw signal by
90% before it ever reaches the served number; it is not evidence that
letting the two targets inform each other helped, only that the extra
columns' small residual signal survived shrinkage marginally better than
wave-1's on this particular population. Recorded and promotable by the
rule; not oversold as more than that.

### Part 3: how market-dominated is each model, and how big could joint gain ever be

Out-of-sample R² against the "trust the market fully" (predict-zero-residual)
baseline, full population, same 500-game floor:

| target | baseline features | R² | union (114-col) features | R² |
| --- | --- | --- | --- | --- |
| margin | production `weak_stack` (90 cols) | **-0.0572** | union | **-0.0653** |
| total | totals wave-1 (41 cols) | **-0.0162** | union | **-0.0762** |

Both models are worse than trusting the market outright as RAW point
estimates, at every feature-set size tried -- the same shape
`docs/totals_model.md` and `MODEL_RESIDUAL_WEIGHT`'s docstring already
established (production margin MAE 10.00 vs market 9.91; total wave-1 raw
model MAE 10.5495 vs market 10.4249). The wider union feature set makes the
RAW fit worse, not better, on both targets -- what small paired gain exists
(Part 2's total-side result) survives only because the served blend shrinks
the raw signal by 90% before it is used.

Given the realised-residual correlation of only +0.0304, the theoretical
ceiling on how much one target's TRUE value could ever help predict the
other is bounded near `r^2` ≈ 0.09% of variance -- a ceiling so low that no
joint-modelling trick can turn "both models are market-dominated" into
"either model reaches the market," let alone beat it by a wide margin. This
directly answers the owner's question: the two marginal models are not
"looking fine" because of hidden, mutually compensating errors that a joint
model would expose -- there simply is not much shared structure between the
two residuals to exploit in the first place.

**The one arm that actually tests coupling** (stage 2, the SUR-lite
regression of each residual on BOTH stage-1 predictions, 3,711 games after
its own 200-row warm-up) confirms this directly. Stage 2 improves R² a lot
over stage 1's noisy raw output (margin -0.0622 -> -0.0027; total -0.0727 ->
-0.0002) -- but a controlled comparison shows this gain is pure
shrinkage/recalibration, not cross-target information: a "solo" version that
regresses each target on ONLY its own stage-1 prediction (no access to the
other target at all) does AT LEAST AS WELL -- margin R² **-0.00082 (solo) vs
-0.00266 (coupled)**, total R² **+0.00022 (solo) vs -0.00019 (coupled)**.
Solo beats coupled on both targets. Giving the two models explicit access to
each other's predictions did not help, and on this measurement mildly hurt --
a clean, controlled null on the coupling question specifically, consistent
with the tiny realised correlation above.

### Positive control

Leaking realised margin truth into one union feature column
(`home_point_diff`, unit slope, zero noise) and re-running both harnesses:
margin-side opener accuracy jumps to 96.94% (candidate) vs 52.96% (baseline),
delta **+43.979 accuracy points, 95% [+41.338, +46.627],
`probability_positive` 1.0**; margin full-population R² **0.99994**. As
predeclared, the total side barely moves: R² -0.0758 (leaked) vs -0.0762
(unleaked); total-side bootstrap **+0.00496, `probability_positive` 0.7915**
(vs +0.00491/0.791 unleaked) -- leaking MARGIN truth does not resolve the
TOTAL residual, exactly as predeclared given the two residuals' weak
correlation. `shape_matches_expectation: true`. The harness has power to
detect a huge effect when one is actually present, which is the point of
running it: the near-null reads above are not an instrument failure.

### Decision

Per the predeclared rule: **do not promote the joint margin output**
(`probability_positive` 0.243, negative point estimate). **The joint total
output meets the promotion bar** (`probability_positive` 0.791, positive
point estimate) -- disclosed above as riding on a small, shrinkage-heavy
blend weight rather than a real jointness or raw-fit improvement, but the
rule is EV, not a promotion-bar aesthetic, so it is recorded as promotable.
Classification for all three registry entries: `unresolved_below_power`
(nothing here is a resolved wrong sign -- the margin-side interval's upper
bound is above zero -- and no positive-control failure was observed to bound
anything by a control). No rotation window spent (reused Tuesday-opener
archive, `docs/player_arrests_policy_eval.md` precedent).

Registry entries: `mod17_joint_residual_margin_opener` (`accuracy_points`,
family `mod17_joint_residual`), `mod17_joint_residual_total_blend`
(`mae_improvement`, family `mod17_joint_residual`), and
`mod17_joint_residual_realised_correlation` (`correlation`, family
`mod17_joint_residual`, informational -- no "candidate" applies to a
structural correlation between two residuals).
