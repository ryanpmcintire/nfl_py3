# The surgical injury candidate: a distribution-derived gate, its measured `f`, and a frozen predeclaration

Written 2026-08-18. Follows `docs/estimation_variance.md` (the `f`-lever
recipe: gating a candidate's influence to games where it disagrees with a
simpler baseline can turn a losing or unresolved comparison into a resolving
one) and `docs/injury_value_lost.md` (the candidate: `injury_value_lost_narrowed`,
+1.316 accuracy points, `probability_positive` 0.8875 on the already-spent
456-game `[2020, 2021]` window, split-half reliability 0.87-0.93, survives a
market-move control and a drop-QB stress test, `unresolved_below_power`).

This document builds the **surgical** version: a mechanism-derived gate that
defers to a simpler baseline pick on games where the injury construct itself
has little to say. It derives the gate's threshold from the DISTRIBUTION of
value lost, not from any accuracy sweep; measures the resulting `f` two ways
(mechanically on NFL, and with real scored accuracy on a CFB analog, since
NFL admits no further look here); and freezes an executable predeclaration
for the one NFL opener window this family would draw if a future session
chooses to spend it.

**One-line verdict:** the gate is real, mechanism-derived, and moves things
in the predicted direction on every check available -- but modestly, not
dramatically. NFL `f` falls only 1.24x (5.70% -> 4.61%) on the one window
that can be inspected mechanically, and the same style of gate applied to an
already-closed-negative CFB comparison cuts the loss by roughly half without
resolving it (-0.67 -> -0.37 points, `probability_positive` 0.026 -> 0.063) --
though the direct, sharper comparison (gated vs. ungated, same real outcomes)
shows gating itself helped at `probability_positive` **0.881** (+0.30 pts).
Recommendation: do not spend a window on the gated form now; if this family's
next opener window is spent, spend it on the plainer, already-better
-evidenced `injury_value_lost_narrowed` candidate first, per that document's
own predeclaration, and hold the gated form as a frozen, ready-to-run
follow-up rather than a preferred first look.

---

## 1. The gate: definition, and how the threshold was derived

### 1.1 What is gated, and on what axis

`docs/injury_value_lost.md` section 4 isolates the cleanest available
candidate: the `player_value` profile on `game_features_player_value.parquet`
(fixed-prior injury severity -- manifest `player_feature_version: "v2"`)
against the frozen `player` profile on `game_features_player.parquet` (also
fixed-prior severity). The two tables differ by **exactly** two columns:

```
diff_injury_skill_epa_value_lost
diff_injury_defense_disruption_value_lost
```

(`src/nfl_ats/surgical_gating.py::VALUE_LOST_DIFF_COLUMNS`). Both are
pregame-available -- built from severity x role-share x value-rate, all
measurable before kickoff (`players.py::_injury_value_features`).

The gate axis is **not** how much the fitted probabilities disagree
(`estimation_variance.gate_by_disagreement`, already screened in
`docs/estimation_variance.md` section 5 on a different, unrelated CFB
comparison). It is the injury **construct's own magnitude** -- a game where
almost no value is missing from either team cannot carry information the
candidate needs, whatever the fitted model happens to output there. Defined
as (`surgical_gating.raw_value_magnitude`):

```
raw_value_magnitude = |diff_injury_skill_epa_value_lost|
                     + |diff_injury_defense_disruption_value_lost|
```

read directly from the columns the ridge model itself consumes -- no manual
home/away reconstruction, no window-specific standardization, so the number
and its units are identical in every season. This portability matters: a
frozen threshold that required recomputing a sample standard deviation over
whichever window happened to be scored would not be one fixed rule, it would
be re-derived every time it was used.

The gate itself (`surgical_gating.gate_by_value_lost_magnitude`):

```
gated_pick(game) = candidate_pick(game)  if raw_value_magnitude(game) >= threshold
                  = baseline_pick(game)  otherwise
```

`baseline_pick` = the frozen `player` profile's forced pick (arm A);
`candidate_pick` = the `player_value` profile's forced pick (arm D, the
value-lost-only isolation). The gate is agnostic to whether picks are passed
as probabilities or already-forced booleans; it only ever selects one array's
value elementwise, never blends them.

### 1.2 Deriving the threshold from the distribution, not from accuracy

`scripts/surgical_value_lost_distribution.py` computes `raw_value_magnitude`
over the full leak-safe history: 4,431 completed REG games, 2009-2025
(`data/processed/game_features_player_value.parquet`, via
`modeling.regular_season_rows` + `result.notna()` -- the identical filter
`docs/injury_value_lost.md` section 3.1 already used for this table's
split-half reliability, and the same "free descriptive statistic" category:
no model fit, no pick, no outcome touched, no rotation window spent). This
script never reads a spread, a pick, or a game result.

The distribution:

| quantity | value |
|---|---|
| games | 4,431 |
| fraction exactly zero | **29.86%** |
| mean | 1.90 |
| median (unconditional) | 1.40 |
| P75 / P90 / P95 / P99 | 2.96 / 4.80 / 5.95 / 8.37 |
| max | 17.66 |

A point mass at exactly zero (no listed value-lost differential of either
kind) plus a long, strongly right-skewed tail (mean > median) -- exactly the
"sparse and lumpy" shape the task brief predicted for an injury construct.

**Derivation rule: the conditional median** -- the median of
`raw_value_magnitude` taken only over games where it is strictly nonzero
(`surgical_gating.derive_conditional_median_threshold`). Of the three
sanctioned methods (natural break / quantile fixed in advance / the point
where reliability is strong), the median is the one fixed quantile that
carries no further researcher degree of freedom: there is no "why 75 and not
70" question to answer for it. The **conditional** form is necessary, not
optional: the unconditional population median lands inside or at the edge of
the zero point-mass whenever that mass exceeds half the population, which
turns out to be true for this recipe's CFB analog (55.7% zero -- see section
3) though not for the NFL construct itself (29.86% zero). A rule that
silently changed definition between leagues would not be one general recipe;
the conditional median is the one definition of "the typical *positive*
reading" that is well-defined in both without adapting after the fact.

**Result: `VALUE_LOST_MAGNITUDE_THRESHOLD = 2.247849687590416`**, pinned
exactly in `tests/test_surgical_gating.py::test_threshold_is_pinned_to_its_derived_value`.
This sits at the 65.1st percentile of the full population (35.07% of all
4,431 games clear it) -- well past the 29.86th-percentile end of the zero
mass, so it is not an artifact of that point mass. **Not derived by sweeping
candidate thresholds against any accuracy outcome** -- no accuracy number for
any NFL window was computed before this constant was fixed.

---

## 2. `f` before and after, measured mechanically on NFL (no accuracy)

### 2.1 Why this is admissible, and what it structurally cannot produce

The task brief is explicit: a gated variant is a NEW variant, and the one
window available to inspect it (`[2020, 2021]`, spent for
`mod07_weak_signal_stack`) may not be scored for the gated form and reported
as evidence -- that would be indistinguishable from iterating a new candidate
until it wins on a window already used to select this family, which
`docs/mod07_stack.md` and `docs/availability_confirmation.md` both already
rule out for this exact family. Reconstructing the window's ALREADY-TAKEN
look, however, is precedented as free re-read (`scripts/availability_ablation.py`,
`scripts/availability_mechanism_screen.py`: "attribution on data already
looked at costs no window", quoting `docs/pool_edge_plan.md`).

`scripts/surgical_gate_reread.py` threads this needle by construction, not by
promise: it reconstructs arms A and D, asserts the reproduction matches
`docs/injury_value_lost.md` section 4's published numbers byte-for-byte
(456 games, 35 weeks, accuracies 0.51316/0.52632, delta +1.3158,
`probability_positive` 0.8875, 26 disagreements -- all matched to machine
precision), and then **deletes every correctness/outcome column from scope**
before computing anything new. Everything after that point reads only
`pick_home_at_open` (a pregame-timestamped boolean) and the pregame
`raw_value_magnitude` covariate. No accuracy, no delta-points, no
`probability_positive` for the gated variant exists anywhere in this
script's output -- structurally, not by omission.

### 2.2 What it measures

| quantity | ungated | gated (`threshold = 2.2478`) |
|---|---|---|
| `f` (fraction of 456 games the pick differs from baseline A) | **5.70%** (26/456) | **4.61%** (21/456) |
| MDE80 (`280 * sqrt(f/456)`) | 3.13 pts | 2.81 pts |
| gate-active fraction on this window | -- | 56.1% |

`f` falls **1.24x**, and MDE80 improves by the same factor. Of the 26 games
where arms A and D disagreed, the gate **retains 21 (80.8%)** and **drops 5**
-- consistent in direction with the already-published finding
(`docs/availability_confirmation.md` section 3.2, reproduced exactly in
`docs/injury_value_lost.md` section 2: `value_magnitude` rank-biserial
correlation with disagreement +0.248, p=0.016) that disagreement games skew
toward higher injury magnitude. But the magnitude is modest: this window's
own gate-active fraction (56.1%) sits well above the full population's
35.07%, because 2020-2021 carried unusually volatile injury/availability
reporting (a plausible pandemic-era effect, not investigated further here),
so a threshold derived from the full 17-season population is markedly less
selective on these two specific seasons than it would be on a typical
window.

**This is a materially weaker `f`-reduction than `docs/estimation_variance.md`
section 5's headline** (probability-disagreement gating cut CFB `f` 9.6x,
19.8% -> 2.06%). That comparison gated on how far two FITTED models'
opinions diverged; this gate acts on the injury CONSTRUCT's own magnitude, a
mechanistically cleaner axis for this specific candidate but, measured here,
a less aggressive filter in practice.

---

## 3. Validating the RECIPE on CFB -- free, unlimited, and fully scored

### 3.1 Why CFB can test the recipe but not the candidate

`docs/injury_value_lost.md` section 5 already establishes, by direct data
audit, that no CFB source carries pregame injury/availability data -- so the
injury candidate itself cannot be screened there, not weakly, not as a
proxy. What CAN be tested is the general claim: *"gate a sparse/lumpy
candidate's influence to games where its own construct fired materially,
using a threshold derived from the construct's distribution alone" -- does
this help, as a recipe, independent of which sparse feature or which
league?* CFB is free and unlimited for this (`docs/rotation_registry.md`
rule 8) and already has a real, already-scored, already-**closed_negative**
comparison built on a genuinely sparse/lumpy construct: `cfb_role_continuity`
(`docs/cfb_role_features.md`; closed at the CFB benchmark, no NFL window ever
drawn, so re-reading it touches no NFL ledger at all).

### 3.2 The analog construct and the already-saved arms

Role-continuity `absent_mass` (`src/nfl_ats/cfb_role_features.py:462-472`):
share-weighted mass of previously-established role holders (dropback/carry)
who did not show up this game. Sparse (55.7% of games exactly zero) and
lumpy (long right tail, max 2.95) in the same shape as the NFL construct.
The per-game gating covariate is the **sum** of `absent_mass` across both
teams and both action types, read straight from the already-saved
`artifacts/cfb_role_experiments/20260817T110541Z/role_continuity.parquet` --
deliberately the total rather than a home-minus-away differential, because
reconstructing the home/away split requires a play-by-play team-id join
(`attach_role_continuity`) not available from the saved artifact alone; the
total is the natural single-scalar analog of "how much established role mass
is absent from this game."

The baseline/candidate arms are the ALREADY-SCORED, ALREADY-SAVED predictions
from the same artifact directory (`predictions.parquet`: `market_residual`
vs `market_residual_roles`, clean_core evaluation window, walk-forward --
exactly what produced the closed_negative verdict). No model is refit;
`scripts/surgical_cfb_recipe_validation.py` only re-derives the threshold
(same rule as section 1.2, `derive_conditional_median_threshold`, computed on
`role_continuity.parquet`'s own full 8,951-game population) and re-scores the
already-computed probabilities under the gate.

Here the unconditional median collides with the zero mass exactly as
predicted (55.7% > 50%, so the unconditional median is precisely 0.0 and
unusable). Conditional median: **`0.6614064187738985`**.

### 3.3 Result -- fully admissible, real accuracy, because CFB is free

| quantity | ungated | gated (`threshold = 0.6614`) |
|---|---|---|
| games | 8,933 (clean_core; matches `docs/cfb_role_features.md`'s own headline count exactly) | same |
| `f` | 9.96% | 3.93% (**2.54x reduction**) |
| MDE80 | 0.94 pts | 0.59 pts |
| accuracy delta vs baseline | **-0.67 pts** | **-0.37 pts** |
| week-blocked 95% | reproduces the recorded `[-1.33, +0.01]` almost exactly (`[-1.337, +0.011]`) | `[-0.852, +0.118]` |
| `probability_positive` | **0.026** | **0.063** |
| disagreements | 890 total | 351 retained, 539 dropped |

Gating **more than doubled `probability_positive`** (0.026 -> 0.063) and cut
the loss roughly in half (-0.67 -> -0.37 points) -- the predicted direction,
and not a small movement. But it did **not** resolve the comparison or flip
its sign: the gated CFB arm is still solidly negative, `probability_positive`
still far below any bar that would call it promising on its own. This is
real, honest evidence about the RECIPE, not the candidate: magnitude-gating a
sparse/lumpy feature reliably sheds some dilutive noise, but for THIS
recipe (conditional-median threshold, magnitude axis, not probability
-disagreement) the effect measured here is a partial rescue, not the dramatic
sign-flip `docs/estimation_variance.md` section 5 found for
probability-disagreement gating on an unrelated comparison.

### 3.4 The sharper question: does gating help, directly

Sections 3.3's two arms are each compared against the true baseline
separately, which is the right comparison for "is this arm good" but a
noisier one for "did gating help" (a difference-of-two-differences). The
direct, paired comparison -- gated arm vs. the UNGATED candidate itself, both
still scored against the same real outcomes -- answers that question with
one bootstrap instead of two:

| comparison | estimate | week-blocked 95% | `probability_positive` |
|---|---|---|---|
| gated vs. ungated candidate (direct) | **+0.30 pts** | [-0.17, +0.79] | **0.881** |

This is the cleanest evidence the recipe produces: gating the already
-closed-negative CFB candidate **improved it**, directly, at
`probability_positive` 0.881 -- a real, fairly strong lean, even though the
resulting gated arm is still net-negative against the true baseline (section
3.3). The two findings are not in tension: a gate can reliably improve on a
losing candidate (retaining the better-behaved subset of games, discarding
the noisier one) without the result clearing zero against an independent
baseline, if the underlying candidate's edge in its best-behaved games is
itself not large enough. This is exactly the modest-not-dramatic shape
predicted going in, now with a number attached: real, replicable-in-direction
improvement from gating, well short of resolution.

---

## 4. Prediction, stated before any NFL accuracy is available to check it

Combining what is legitimately known:

1. **Already-published** (not re-derived here): the tercile gradient on the
   spent `[2020, 2021]` window (`docs/availability_confirmation.md` section
   3.1, reproduced in `docs/injury_value_lost.md` section 2) shows the
   value-magnitude effect concentrates heavily in the top third (+5.26 points)
   versus the bottom third (-0.66 points) -- directionally consistent with
   gating helping.
2. **Measured here, mechanically**: NFL `f` falls only 1.24x under this
   specific (distribution-derived, not tercile-matched) threshold -- a modest
   filter, not an aggressive one, on the one window inspectable.
3. **Measured here, with real accuracy**: the same style of gate (magnitude
   axis, conditional-median threshold) produces a real but partial
   improvement on an unrelated CFB comparison -- `probability_positive` more
   than doubles but stays well short of resolving against the true baseline.
   The direct comparison (gated vs. ungated candidate, section 3.4) is
   stronger: `probability_positive` 0.881 that gating helped at all.

**Prediction**: if the gated NFL candidate were scored on a future window,
its `probability_positive` would plausibly move up somewhat from the ungated
form's 0.8875 (already high), but the evidence available does not support
expecting a dramatic jump to near-certainty, and does not rule out a smaller
sample-size-driven move in either direction given only 21 of 456 games remain
gate-active-and-disagreeing. The honest expectation, calibrated against
section 3's CFB check, is **modest improvement, not resolution** -- and this
is a prediction, not a result; no NFL accuracy number for the gated candidate
exists anywhere in this repository, and none is computed in this document.

---

## 5. The frozen predeclaration

Executable by a future session with zero further design choices.

> **Family:** `injury_value_lost_narrowed_surgical`
> **Inherits:** `injury_value_lost_narrowed` (contaminates the same 456-game
> `[2020, 2021]` informative-only evidence), which itself inherits
> `mod07_weak_signal_stack` (contaminates `[2020, 2021]`, spent).
> **Grade:** `opener`
> **acknowledges_mined_2018_2025:** `true`
>
> **Exact feature definition:** the `player_value` profile on
> `game_features_player_value.parquet` (fixed-prior severity) as the
> candidate arm, contrasted against the frozen `player` profile on
> `game_features_player.parquet` as the baseline arm -- identical to
> `docs/injury_value_lost.md` section 4's arms D and A. No feature, model, or
> profile changes from that document; only the pick-selection rule below is
> new.
>
> **Exact gate:** `nfl_ats.surgical_gating.gate_by_value_lost_magnitude`,
> applied to the two arms' `pick_home_at_open` booleans (or, equivalently,
> to their `correct_at_open` booleans directly, since for a fixed outcome
> selecting-the-pick and selecting-the-correctness commute -- confirmed
> equivalent by construction: `gated_correct(game) = A_correct(game)` where
> `raw_value_magnitude(game) < threshold`, else `D_correct(game)`), with
> `magnitude = nfl_ats.surgical_gating.raw_value_magnitude` computed from
> `game_features_player_value.parquet` for the assigned window's games, and
> `threshold = nfl_ats.surgical_gating.VALUE_LOST_MAGNITUDE_THRESHOLD`
> (`2.247849687590416`, frozen, pinned in `tests/test_surgical_gating.py`,
> not to be recomputed against the confirmation window).
>
> **Exact profile / evaluation machinery:** `clv.opener_pick_evaluation` for
> both arms exactly as `scripts/availability_ablation.py::arm` calls it
> (`ridge`, `ridge_alpha=10.0`, `target="market_residual"`,
> `min_train_games=500`); `rotation.confirmation_split` for the forward
> -chained training/window split (do not use the spent-window reconstruction
> helper -- that is only for re-reading an already-spent window).
>
> **Predeclared metric:** `probability_positive` from
> `clv.week_blocked_bootstrap` (`block="week"`, `samples=2000`,
> `seed=20260818`) on the paired contrast `gated - A` (gated variant vs. the
> frozen `player` baseline), computed over the assigned window only.
>
> **Predeclared thresholds** (matches `injury_value_lost_narrowed`'s own
> convention, itself matching MOD-07's): `probability_positive >= 0.90` ->
> confirmed; `<= 0.10` -> closed_negative; otherwise unresolved.
>
> **Prior evidence to cite, not re-litigate:** section 2's mechanically
> -measured `f` (5.70% -> 4.61%, 1.24x) and section 3's CFB recipe validation
> (`probability_positive` 0.026 -> 0.063 on an unrelated comparison) --
> informative about the recipe's expected strength, not a look at this
> candidate, and must not be treated as the confirmation result.
>
> **Window this would draw:** `[2022, 2023]` if assigned before
> `injury_value_lost_narrowed` (the ungated form); `[2024, 2025]` -- the
> LAST remaining opener block in the entire 2020-2025 pool -- if assigned
> after it. Confirm via `rotation status` at assignment time rather than
> assuming; the two families' declared `inherits` chains make the order
> load-bearing (section 6).
>
> **Do not assign until:** the free, zero-cost 2026 prospective evidence for
> `mod07_weak_signal_stack` (`docs/availability_confirmation.md` section 4)
> has accrued enough weeks to be informative, per that document's own
> already-stated deferral for the ungated form. Nothing in this document
> changes that reasoning; if anything it strengthens it, since the gate's
> measured benefit here is real but modest, not the kind of decisive
> mechanism that would justify jumping the queue.

---

## 6. Window arithmetic: this family is not free to run alongside the ungated one

`registry/rotation_registry.json` (read-only this session, confirmed
unmodified) shows the 2020-2025 opener pool holds three 2-season blocks:
`[2020, 2021]` (spent, by two unrelated families), `[2022, 2023]`, and
`[2024, 2025]` -- **two remaining for the entire project.**
`injury_value_lost_narrowed` (the ungated form) is already predeclared to
draw `[2022, 2023]` as its earliest eligible block
(`docs/injury_value_lost.md` section 7). This document's surgical family
honestly inherits `injury_value_lost_narrowed`, so if the ungated form spends
`[2022, 2023]` first, the gated form's earliest eligible block becomes
`[2024, 2025]` -- the project's **last** opener window, for anything, ever
(within the current 2020-2025 archive coverage).

**Consequence, stated plainly**: confirming both the ungated and gated forms
of this one candidate would consume the entire remaining opener-graded
research budget of the project. That is not disqualifying by itself -- a
window never spent is worth exactly zero, per the task's own framing -- but
it does mean the two variants are in direct competition for scarce capacity,
not independent asks, and the order matters.

---

## 7. Recommendation, argued on expected value

**Do not spend a window on the gated (surgical) form now.** Not because it
fails to clear a confidence bar -- `AGENTS.md`'s own binding rule says a
promotion bar is not a decision bar -- but because the expected research
value of spending one of only two remaining opener windows on THIS variant,
right now, is lower than the alternatives available with the same resource:

1. **The measured benefit of gating, on both checks available, is real but
   modest, not decisive.** A 1.24x `f`-reduction (NFL, mechanical) and a
   partial-not-resolving improvement on the CFB recipe check
   (`probability_positive` 0.026 -> 0.063, still solidly negative) are
   genuine, directionally-supportive evidence for the mechanism -- but they
   are far short of the kind of result (e.g. `docs/estimation_variance.md`
   section 5's 9.6x `f`-reduction and full sign-flip) that would justify
   preferentially spending a scarce, irreplaceable window on the REFINEMENT
   ahead of the base hypothesis.
2. **The base hypothesis already has stronger existing evidence and a
   free evidence stream in flight.** `injury_value_lost_narrowed` already
   carries `probability_positive` 0.8875 (informative, not confirmatory) on
   456 games, split-half reliability 0.87-0.93, and survives two independent
   stress tests -- more NFL evidence than exists anywhere for the gated form,
   which has none (by design; none may be computed here). `mod07_weak_signal_stack`
   is simultaneously accruing free, zero-multiplicity-discount evidence every
   week of the live 2026 season (`docs/availability_confirmation.md` section
   4) at no window cost at all. That stream will sharpen the picture on
   BOTH variants before either window needs to be spent.
3. **Spending on the gated form first forecloses the option to spend on the
   simpler, better-evidenced form at the better window.** Section 6:
   whichever variant is assigned first claims `[2022, 2023]`; the other is
   pushed to `[2024, 2025]`, the project's last opener block. Since the
   ungated form's evidence is currently stronger and the gate's incremental
   benefit here is unproven at NFL scale, the option value clearly favors
   letting the ungated form claim the earlier, unremarkable-either-way block
   first if and when a window is spent on this family at all.
4. **This is not "never spend it."** A window never spent has zero value,
   and 285 cards go out every week regardless of what research is or is not
   validated. The predeclaration in section 5 is frozen and ready specifically
   so that IF the ungated form's `[2022, 2023]` confirmation (or the 2026
   prospective stream) comes back promising-but-short-of-the-promotion-bar in
   a way a sharper cut could plausibly resolve -- the exact shape a gate is
   built to fix -- the gated form is sitting ready to spend `[2024, 2025]`
   with no further design work, immediately, rather than needing a fresh
   research cycle at that point.

**What could not be measured without spending a window, stated explicitly:**
the gated candidate's own `probability_positive` on any NFL confirmation
window. Every number in sections 2 and 4 is either mechanical (picks only,
no outcomes) or measured on a different sport entirely; no accuracy figure
for this specific candidate, gated, on real NFL outcomes, exists anywhere,
and producing one is exactly the one thing this document is not allowed to
do without spending one of the two windows the recommendation above says not
to spend yet.
