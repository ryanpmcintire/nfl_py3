# Best Pick ranker (SPEC-5) — predeclaration, screen, and result

Written 2026-08-17, executing `docs/opus_execution_specs.md` § SPEC-5.

> **STATUS: the screen ran on [2013, 2015] and the window is SPENT**
> (`registry/rotation_registry.json`, verdict `unresolved`). Results and
> the decision are in **§ Screen result** at the end. Sections 1-6 below
> are the audit trail, preserved verbatim: the predeclaration, the
> machinery proof, and the earlier stop that this run resolved. Read them
> as history — the "window was not spent" language in § "Why the window
> was not spent" describes the state on the morning of 2026-08-17, before
> rule 9 existed.

The short version: three signals were predeclared, one (`sweep_robustness`)
cleared the 0.75 screen gate at 0.7955 and earns a single opener-graded
confirmation; the other two moved the wrong way and are closed.

## Question

The pool pays one Best Pick per week directly. Our confidence ordering is
flat: taking the weekly top-|residual| pick scored 48.6% over 107 weeks —
worse than picking at random among our own picks. SPEC-5 asks whether any
of three predeclared signals orders pick quality well enough to be worth
using for that single weekly choice.

## Predeclared candidates (exactly three, no variants)

1. **`calibrated_probability`** — the pick-side cover probability after
   Platt calibration on the chronological walk-forward stream
   (`calibration.calibrate_cover_prediction_stream`, method `platt`). The
   pick side is fixed by the *raw* probability (`>= 0.5` → HOME); Platt is
   monotone in the raw probability but can move the 0.5 crossing point, so
   a pick-side calibrated probability slightly below 0.50 is expected and
   is not a bug.
2. **`key_number_distance`** — `key_number_distance(spread_line)` minus
   `key_number_distance(fair_spread)`, using `clv.KEY_NUMBERS = (3.0, 7.0)`
   and `clv.key_number_distance` (the repo's existing helper; the
   `key_numbers.py` module's `DEFAULT_KEY_NUMBERS` is the wider
   `(1,2,3,4,6,7,10,14)` set and is *not* what SPEC-5's "{3, 7}" means).
   Positive when our number sits on a key number the market misses.
3. **`sweep_robustness`** — from `MarginModel.line_sweep`, the width in
   points of the contiguous interval around the quote over which the
   *pick's* cover probability stays `>= 0.50`. The pick side is fixed at
   the quote; for AWAY picks the sweep's `home_cover_probability` is
   complemented. The run is anchored at offset 0 (where the condition holds
   by construction) and extended across contiguous 0.5-point grid steps.

## Predeclared metrics and gate

Per signal: top-1-per-week accuracy; Kendall tau between the signal and
pick correctness across all picks; and `probability_positive` for
(top-1 accuracy − all-pick accuracy) > 0 from
`clv.week_blocked_bootstrap` (week blocks, 2,000 samples, seed 20260817).

Gate: any candidate with `probability_positive >= 0.75` on the screen earns
**one** opener-graded confirmation window. No candidate clearing 0.75 →
record `closed_negative` for the screen window and stop; no new signals on
the same window, no widening the window.

## Machinery, built and proved (stage A)

Script: `scratchpad/best_pick_ranker.py` (session scratchpad; promoted to
`scripts/best_pick_ranker.py` at resolution — see the final section). It:

- runs the standard walk-forward evaluator (`outcomes.walk_forward_outcomes`,
  `market_residual` only, `base` profile, ridge alpha 10, `min_edge` 0.02,
  `min_train_games` 500 — the frozen active configuration except for the
  profile, which SPEC-5 fixes to `base` because player features do not
  exist pre-2016 and the ranker does not need them);
- re-runs the identical weekly loop to emit a `line_sweep` per scored game,
  because the evaluator does not emit sweeps. **Proved identical**: the
  sweep loop's `predicted_margin` reproduces the evaluator's with
  `max |diff| = 0.0` across all scored games. The sweep is a re-evaluation
  of the same weekly fit at alternative lines, never a fresh (leakier) fit;
- marks the week's top-1 pick **once, before any resampling**, and lets
  `week_blocked_bootstrap` resample whole weeks over that fixed flag.
  Recomputing the ranking inside the metric function would silently
  collapse a week drawn twice into one contribution.

Debug run — **2018–2020, non-reserved seasons inside the already-ledgered
2018–2025 mined era, chosen for convenience, not evidence.** These numbers
are a plumbing check only; they carry the standing 130–150-look discount,
they were not predeclared, and no verdict rests on them:

| signal | top-1 acc | all-pick acc | delta | Kendall tau | P(positive) |
|---|---|---|---|---|---|
| calibrated_probability | 49.02% | 52.60% | −3.58 pts | −0.023 | 0.284 |
| key_number_distance | 60.78% | 52.60% | +8.18 pts | +0.013 | 0.920 |
| sweep_robustness | 50.98% | 52.60% | −1.62 pts | −0.028 | 0.410 |

51 weeks, 749 resolved picks, 51 top-1 picks, 500 bootstrap samples. Note
how wide this already is: `key_number_distance`'s week-blocked interval on
*three full seasons of warm data* is [−5.2, +21.2] points. Read it as
"the code runs", nothing more.

Known limitation of signal 3: the sweep grid is ±4 points in 0.5 steps, so
`sweep_robustness` is floored at 4.0 and **censored at 8.0** (55 of 749
debug picks hit the ceiling). Widening the grid would change the signal's
definition and was not predeclared, so the censoring stands.

## Why the window was not spent

SPEC-5 assigns the screen to the earliest eligible `nflverse_spread` block.
Confirmed by an in-memory dry run (nothing written): the earliest eligible
block is **[2009, 2011]**. That is exactly the problem.

The feature table begins in **2009**. So:

- `confirmation_split` yields an **empty training frame** for [2009, 2011] —
  there is no completed game strictly before the window's first gameday.
- With the standard `min_train_games = 500`, the evaluator cannot score a
  week until 500 completed regular-season games have accumulated *inside*
  the window. Measured structurally (schedule arithmetic only, no model
  scored): **17 of the window's 51 weeks are scorable, 256 games** —
  scoring cannot start until roughly week 16 of 2010. SPEC-5's "~48 weeks"
  assumption does not hold.
- Worse, `calibrated_probability` **cannot be computed at all**.
  `calibrate_cover_prediction_stream` needs 400 prior out-of-sample
  prediction rows before a week can be calibrated (the frozen player
  selection budget warms its calibrator by starting the raw stream in 2016
  for a 2018 evaluation). The entire [2009, 2011] prediction stream is 256
  rows and there is nothing earlier in the world to warm it with. The call
  raises; it does not degrade.

Making the screen runnable therefore requires changing `min_train_games`,
`min_calibration_games`, or both. Neither is predeclared in SPEC-5, and the
choice determines whether signal 1 exists at all and how many weeks the
screen has — that is a methodology decision about *what a signal means*, not
a mechanical ambiguity. Per the spec's own standing instruction ("if a spec
turns out to be wrong or ambiguous in a way that changes a methodology
decision, STOP and leave the decision to the owner"), and because spending a
window on a misunderstanding is unrecoverable, the look was not taken.

**Negative-result discipline note:** nothing has been recorded as
`closed_negative` here, because nothing was scored. Recording a verdict for
a run that never happened would be worse bookkeeping than recording none.

## Options for the owner (each is a decision, not a default)

1. **Drop signal 1 from the screen** and run the remaining two on
   [2009, 2011] at standard settings — 17 weeks, 17 top-1 picks. Honest,
   but the interval on 17 picks will be enormous; `unresolved` is close to
   guaranteed.
2. **Relax `min_train_games`** (e.g. to 256, one full season) so scoring
   starts at 2010 week 1 — ~34 weeks — and still drop signal 1, which
   remains uncomputable at 400 calibration rows.
3. **Assign a later window** than the earliest-eligible rule produces, so
   the family gets real pre-window training and a warm calibrator (e.g.
   [2012, 2014] with the raw stream started in 2009). This departs from the
   registry's deterministic earliest-eligible rule and must be an explicit,
   logged exception to `docs/rotation_registry.md`'s window mechanics — it
   is the only option under which all three predeclared signals are
   computable as written.
4. **Amend SPEC-5** to predeclare the warm-up policy (raw-stream start
   season and `min_calibration_games`) and then run the screen.

Option 3 or 4 is what the machinery is built for; the script takes
`--raw-start-season`, `--min-train-games`, and `--min-calibration-games`
so whichever policy is chosen is a one-line invocation.

## Ledger state after this session

`registry/rotation_registry.json`: **unchanged**. `best_pick_ranker` is not
declared, holds no window, and [2009, 2011] remains available to it.

## Resolution (Fable, 2026-08-17)

Every factual claim above was independently reproduced against the real
code and data (see `docs/opus_session_blockers.md` and its adjudication).
The stop was correct. The root cause was not ambiguity but an authoring
error in SPEC-5: it assigned the pool's first block while promising sample
sizes only a later block can deliver.

Decisions, all executed:

1. **Option 4 is taken, as a permanent rule rather than a one-off.**
   Warm-up eligibility is now **binding rule 9** of
   `docs/rotation_registry.md`, enforced in `rotation.py`
   (`MIN_ELIGIBLE_START_SEASON`, then 2013): no window may start before 500
   walk-forward training games plus 400 calibration prediction rows
   (~900 games; four 256-game seasons) of history exist in front of it.
   *(Later the same day the calibration constant was derived at 200 rather
   than inherited at 400, cutting the requirement to 700 games and moving
   the floor to 2012; see `docs/rotation_registry.md` rule 9. The
   [2013, 2015] assignment below was made under the 400 constant and is
   recorded as it ran.)*
   `rotation assign` now lands `best_pick_ranker` on **[2013, 2015]**
   deterministically — no exception logged, no discretion retained.
2. **The warm-up requirement is 900 games, not 500** — signal 1 stays in
   the screen. All three predeclared signals run as written.
3. **Predeclared invocation** (verified end-to-end on the real table
   before this resolution shipped: predictions begin 2011 week 1; 512
   prediction rows precede 2013 week 1; the window scores 768 games over
   51 weeks, 17 per season; per-week calibration histories run 496-1,228
   rows):

   ```
   python scripts/best_pick_ranker.py --start-season 2013 --end-season 2015
       --raw-start-season 2011
   ```

   The 2011-2012 stream rows are warm-up plumbing, never evidence.
4. **Disclosure carried forward:** [2013, 2015] sits inside
   `pbp_drive_bundle`'s spent [2013, 2017]. Rule 4 permits the overlap
   (windows retire per-family); the screen write-up must state it.
5. The runner moved from the session scratchpad to
   `scripts/best_pick_ranker.py` (unchanged logic; import/typing cleanup
   only) so it cannot evaporate with the temp directory.

The screen is now a mechanical execution: declare, assign, run the
invocation above, record the look, write the results here.

## Screen result (2026-08-17, [2013, 2015] — window spent)

Executed exactly as predeclared, immediately after `rotation declare` and
`rotation assign` (which landed [2013, 2015] with no override, as rule 9
predicts):

```
python scripts/best_pick_ranker.py --start-season 2013 --end-season 2015
    --raw-start-season 2011
```

Artifact: `artifacts/best_pick_ranker/screen_2013_2015.json` (plus the
per-pick parquet beside it). Sample: **51 weeks, 748 resolved picks, 51
top-1 picks**, matching the predeclared expectation exactly. The sweep
loop reproduced the evaluator's weekly fit to `max |diff| = 0.0`, so the
signals are computed off the same model the evaluation scored.

| signal | top-1 acc | all-pick acc | delta | Kendall tau | week-blocked 95% | P(positive) |
|---|---|---|---|---|---|---|
| `sweep_robustness` | **54.90%** | 49.33% | **+5.57 pts** | +0.042 | [−8.36, +19.09] | **0.7955** |
| `key_number_distance` | 43.14% | 49.33% | −6.19 pts | −0.016 | [−18.69, +6.60] | 0.170 |
| `calibrated_probability` | 41.18% | 49.33% | −8.16 pts | −0.025 | [−20.78, +4.03] | 0.0925 |

### What this says

**`sweep_robustness` clears the gate — barely, and on a wide interval.**
0.7955 ≥ the predeclared 0.75, so it earns its one opener-graded
confirmation. But the interval spans 27 points and comfortably contains
zero, and the Kendall tau across all 748 picks is +0.042 with p = 0.19 —
the rank correlation is not itself resolvable. The honest reading is
"promising enough to spend one opener window on, nowhere near
established", which is exactly why the registry verdict is `unresolved`
rather than `confirmed`. The gate was fixed at 0.75 before any of these
numbers existed; honouring it here is the point of the registry, and
so is refusing to upgrade the verdict because the number happened to
land on the right side.

**The two intuitive signals failed, and failed in the same direction.**
Both `calibrated_probability` and `key_number_distance` made the weekly
top-1 pick *worse* than picking arbitrarily among our own picks — by 8.2
and 6.2 points. This is the flat-confidence finding from
`docs/pool_edge_plan.md` (top-|residual| scored 48.6% over 107 weeks)
reproducing on untouched seasons, and it now extends to calibrated
probability specifically: recalibrating the number does not rescue the
ordering, because the ordering was never carrying information. Per the
predeclared stop rule these two are closed; they are not to be retuned
on this window or any other.

**Read the delta, not the level.** All-pick accuracy in this window is
**49.33%** — the `base` profile (no player features, as SPEC-5 fixes it)
is a below-coin-flip classifier on 2013-2015. The predeclared metric is
the paired delta (top-1 − all-pick), and that is what the gate scored, but
`sweep_robustness`'s 54.90% top-1 level is measured against a weak parent
in an old regime. It is not a forecast of 54.90% Best Pick accuracy in
2026.

### Disclosure (required by SPEC-5)

[2013, 2015] sits inside `pbp_drive_bundle`'s spent [2013, 2017]. Rule 4
permits this — windows retire per-family, and the two hypotheses are
independent — but these seasons have been mined once before by another
family, and this result carries that discount.

### Next step, predeclared

`sweep_robustness` earns **one** opener-graded confirmation: declare
`best_pick_ranker_opener` (inherits `best_pick_ranker`,
`--acknowledge-mined`), assign ⇒ [2020, 2021], and evaluate the same
frozen signal's top-1 at the opener grade. ≥ 0.75 there means using it to
choose the Best Pick in 2026 — a pool-play decision, not a model change,
so no activation is involved. Below 0.75 closes the family.

The screen runner grades against the nflverse spread and cannot do this as
written; the opener arm needs the `clv.opener_pick_evaluation` machinery,
which is a build task, not a new methodology decision.

## Opener confirmation (2026-08-17, [2020, 2021] — window spent)

Family `best_pick_ranker_opener` (inherits `best_pick_ranker`,
`--acknowledge-mined`); `rotation assign` ⇒ **[2020, 2021]**, the earliest
eligible opener block.

```
python scripts/best_pick_ranker.py --grade opener --feature-profile player
    --start-season 2020 --end-season 2021
    --features data/processed/game_features_player.parquet
```

Artifact: `artifacts/best_pick_ranker/opener_2020_2021.json`.

**Profile decision.** SPEC-5 says "evaluate the SAME frozen signal top-1 at
the opener grade" without naming a feature profile, and the two readings
differ: `base` replicates the screen's model exactly, `player` ranks the
picks we actually publish. **`player` was chosen**, because the gate's own
consequence — "use it for Best Pick in 2026" — is a choice among the
card's picks, and confirming a ranker on picks we will never make would
answer the wrong question. The signal's *definition* is unchanged; the
model generating the picks it ranks is the deployed one.

Every pick is re-formed at the archived Tuesday opener consensus and
settled against it. The sweep loop reproduced `opener_pick_evaluation`'s
weekly fit to `max |diff| = 0.0`, so signal and grade come from one model.

| metric | value |
|---|---|
| weeks / resolved picks / top-1 picks | 35 / 456 / 35 |
| `sweep_robustness` top-1 accuracy | **60.0%** (21 / 35) |
| all-pick accuracy at the opener | 51.32% |
| delta | **+8.68 points** |
| week-blocked 95% interval | [−7.00, +22.88] |
| `probability_positive` | **0.865** |
| Kendall tau (all 456 picks) | +0.067 (p = 0.099) |

**Verdict: `confirmed`** — 0.865 clears SPEC-5's predeclared 0.75
confirmation gate. Per the predeclaration, the consequence is: **use
`sweep_robustness` to choose the Best Pick in 2026.** That is a pool-play
decision, not a model change — no activation, no new active model, the
card is unaffected.

### What "confirmed" does and does not mean here

It means the signal cleared the bar that was fixed before any of these
numbers existed, twice, on two independent windows and two different
grades. It does **not** mean the effect is established:

- **35 top-1 picks.** The interval runs [−7.00, +22.88]. Nine extra correct
  picks out of 35 is the whole result, and a 30-point-wide interval is what
  35 observations buy.
- **The rank correlation is not resolvable.** Kendall tau across all 456
  picks is +0.067 at p = 0.099 — suggestive, not significant. The signal
  looks better at picking one winner per week than at ordering the field,
  which is at least the right shape for the job, but could equally be the
  top-1 selection getting lucky.
- **The window is inside the mined era**, acknowledged at declaration. The
  standing ~130–150-look discount applies.
- **Consistency is the strongest part.** Screen +5.57 points at 0.7955
  against the nflverse spread on 2013–2015; confirmation +8.68 points at
  0.865 against the opener on 2020–2021. Two disjoint windows, two grades,
  same direction, both clearing. That is more than either number alone.

The honest summary: this is the best-supported pool-play lever the project
has, and it rests on 86 top-1 picks total. Use it — the alternative
(picking arbitrarily, since all picks score ≈52.5%) has no evidence behind
it at all and this does — but expect the 2026 realized number to land
nearer the all-pick rate than 60%.

### Now spent

`best_pick_ranker` [2013, 2015] and `best_pick_ranker_opener` [2020, 2021]
are both permanently spent. The family is closed as `confirmed`. Re-scoring
either window, or tuning the sweep grid (still censored at 8.0 points) and
re-running, is inadmissible. Prospective 2026 Best Pick results are the
next real evidence and need no window.

## Tier-2 re-read (2026-08-18) — the hardest look at the project's only `confirmed` verdict

Per `docs/revisit_list.md` Tier 2: **re-read only.** No model refit, no
rotation window touched, no registry file written by this session. Every
number below is **measured this session** from the already-stored parquet
artifacts (`artifacts/best_pick_ranker/opener_2020_2021.picks.parquet`,
`screen_2013_2015.picks.parquet`) with
`scripts/best_pick_ranker_tiebreak_audit.py`
(`.\.tools\uv.exe run --no-sync python scripts/best_pick_ranker_tiebreak_audit.py`,
output `artifacts/best_pick_ranker/tiebreak_reread_20260818.json`, gitignored,
reproduce on demand). The original numbers above are preserved verbatim;
nothing here overwrites them.

### 1. Ties, verified from the artifact directly

`docs/pool_format_levers.md` §2.1 reported **24 of 35** confirmation weeks and
**39 of 51** screen weeks were `sweep_robustness` ties. *(measured:
`scripts/best_pick_ranker_tiebreak_audit.py`, grouping each artifact by
`(season, week)` and counting weeks where more than one game shares the
maximum score)* — **confirmed exactly**: 24/35 (68.6%) on the opener
confirmation, 39/51 (76.5%) on the screen, with the max sitting on the
0.5-grid's 8.0-point censoring ceiling in 31/35 and 45/51 weeks respectively.
The instruction to verify this "a previous session stated backwards" was
heeded: it is **not** backwards. Ties are the majority outcome in both
windows, not an edge case.

Top-1 accuracy under each tie-break rule *(measured, same script)*:

| window | recorded (alphabetical) | tie-break-agnostic (average over tied candidates) | delta: recorded vs. all-pick | delta: tie-agnostic vs. all-pick |
|---|---|---|---|---|
| opener confirmation (35 wk) | 60.0% (21/35) | **52.24%** | +8.68 pts | **+0.92 pts** |
| screen (51 wk) | 54.90% (28/51) | **58.38%** | +5.57 pts | **+9.05 pts** |

Both figures reproduce `docs/pool_format_levers.md` §2.1's numbers (52.24%,
+0.92 / 58.38%, +9.05) to the digit, from an independent recomputation off
the raw picks rather than a copy of the earlier write-up. A week-blocked
bootstrap recompute of the tie-agnostic series (20,000 samples, seed
20260818) gives the confirmation window's tie-agnostic delta a naive
`probability_positive` of **0.553** — a coin flip — versus the recorded
alphabetical delta's naive recompute of 0.884 (close to the original 0.865;
the small gap is seed/sample-count noise, consistent with D3). On the screen
window the tie-break luck ran the other way, as the earlier write-up already
noted: tie-agnostic naive `probability_positive` there is **0.976**, higher
than the recorded 0.7955.

### 2. Honest, D2-widened `probability_positive`

D2 (`docs/estimation_variance.md`) measured naive-vs-honest interval width
inflation of **1.037x-1.575x** across two synthetic ground-truth DGPs and two
real CFB comparisons — the "17-58% too narrow" headline. Refitting the 35
weekly ridge models to measure this directly on the Best Pick screen would be
a re-run, which Tier 2 disallows, so this is a **sensitivity check**, not a
new measurement: hold the point estimate fixed, back out the naive interval's
implied SE under a normal approximation, scale that SE by each of D2's
measured factors, and recompute `probability_positive`. *(measured:
`scripts/best_pick_ranker_tiebreak_audit.py::d2_sensitivity`, applied to the
recorded artifact values `estimate=0.08684210526315783`,
`bootstrap_lower=-0.06997804357245584`, `bootstrap_upper=0.2288232557466831`
from `artifacts/best_pick_ranker/opener_2020_2021.json`.)*

| inflation factor | source | honest `probability_positive` (as-recorded, +8.68) | honest `probability_positive` (tie-agnostic, +0.92) |
|---|---|---|---|
| 1.037x | D2 comparison A (large disagreement fraction) | 0.864 | 0.554 |
| 1.17x | D2 headline floor | 0.835 | 0.548 |
| 1.330x | D2 comparison B | 0.804 | 0.542 |
| 1.575x | D2 headline ceiling (null DGP) | **0.765** | **0.536** |

Two separate findings, and they point the same direction but for different
reasons:

- **Applied to the recorded (+8.68, alphabetical) number alone**, D2 widening
  does not flip the confirmation: honest `probability_positive` ranges
  0.765-0.864, staying above SPEC-5's 0.75 gate across the entire measured
  inflation range, though the top of that range (0.765) sits only 0.015 above
  the line — one more turn of the inflation crank would cross it.
- **Applied to the tie-break-agnostic (+0.92) number** — the estimate that
  actually isolates the ranker's signal from alphabetical luck —
  `probability_positive` is **0.536-0.554 across the whole D2 range**: this
  never came close to clearing 0.75, with or without D2's widening. D2 was
  never the deciding defect here; the tie-break artifact was, and it was
  already large enough on its own.
- **This is an inferred sensitivity check, not a refit measurement.** A real
  refit-aware bootstrap could also move the central estimate (D2 §3's
  comparison B did, non-monotonically), which holding the estimate fixed
  cannot capture. Treat the ranges above as the right order of magnitude, not
  a resolved number to the third decimal.

### 3. Paired power arithmetic

35 top-1 picks is `sqrt(0.25/35) = 8.45` points of SE *(measured, matching
`docs/pool_format_levers.md`'s independently-stated 8.45)*. The recorded
+8.68-point delta is **1.03 SE** from zero — nowhere near a conventional
two-sided resolution (`1.96 x 8.45 = 16.6` points would be needed). At this
instrument's resolution on 35 weeks, only effects at or above roughly 16-17
points are distinguishable from noise at 95%; an 8-9 point effect, the size
actually recorded, is expected to look like this by chance alone even if the
true effect were zero.

**The fragility is concrete, not abstract.** The recorded top-1 is 21/35
correct. *(measured, same script)*: flipping exactly **3** of those 21 picks
from correct to incorrect — three games, out of 456 total resolved picks in
the window — drops top-1 accuracy to 18/35 (51.43%) and the delta from
+8.68 points to **+0.11 points**, i.e. from "the project's only confirmed
result" to "indistinguishable from nothing." A 3-pick swing on a
Binomial(35, 0.5) count has a standard deviation of `sqrt(35*0.25) = 2.96`
picks — a 3-pick swing is almost exactly 1 SD, an ordinary and expected
amount of week-to-week noise, not a rare event. The entire "+8.68, confirmed"
finding rests on a coin-flip-sized number of games landing the way they did.

Per the binding project rule, this is not grounds to reject the signal — an
interval crossing zero, or a result close to the noise floor, is the expected
shape of a real-but-small effect, not evidence of no effect. It is grounds to
stop calling the number "confirmed."

### 4. What the verdict should become

Applying `docs/revisit_list.md`'s D5 frame — a promotion bar governs what the
docs may claim, never which card gets played, and the pool is forced picks —
these are two separate questions with two separate answers.

**The registry classification**: `best_pick_ranker_opener` should
**downgrade from `confirmed` to `unresolved`.** The number that decides
this is the tie-break-agnostic honest `probability_positive`,
**0.536-0.554** (§2 above) — the estimate that isolates what
`sweep_robustness` itself contributes once alphabetical luck is removed. That
number never approached SPEC-5's own predeclared 0.75 confirmation gate, at
any point in D2's measured inflation range. The recorded 0.865 was real
arithmetic on real data, but it measured the alphabetical tie-break's luck at
least as much as it measured the ranker: a 5,000-draw Monte Carlo over
uniformly-random tie-breaks *(measured, same script,
`monte_carlo_random_tiebreak`)* puts the confirmation window's recorded
60.0% at the **95.4th percentile** of its own tie-break-luck distribution
(mean 52.2%, 5-95% range [42.9%, 60.0%] — the recorded result sits right at
the top edge of that range). `docs/pool_format_levers.md` independently
reported "88th percentile" using a similar Monte Carlo with a different
sampling design; both agree the recorded draw is in the upper tail, i.e.
unusually lucky, not typical. **Proposed registry edit** (not applied by this
session — no `registry/*.json` write per the task constraints):
`registry/rotation_registry.json` → `families.best_pick_ranker_opener.status`
`"confirmed"` → `"unresolved"`, and `windows[0].verdict` `"confirmed"` →
`"unresolved"`, with a `windows[0].notes` addendum citing this section and
`docs/pool_format_levers.md` §2.1.

**The play decision — separate question, different answer.** Both windows'
tie-break-agnostic deltas are **positive** (+0.92 opener, +9.05 screen) —
the sign is consistent across two disjoint windows and two grades even after
removing the artifact that inflated the opener number, which is more than
either of the two alternatives can say: `calibrated_probability` and
`key_number_distance` both scored **negative** deltas at the screen (−8.16
and −6.19 points, `probability_positive` 0.0925 and 0.170) and were closed
under SPEC-5's own predeclared stop rule — they do not get a second look on
this window. "No ranker" (arbitrary nomination) is by construction a zero
-edge strategy. Per `AGENTS.md`'s "a promotion bar is not a decision bar":
a ranker with a +0.92-point honest edge and `probability_positive` near 0.55
is still the right card to play when every alternative is worse or is zero.
**`sweep_robustness` should keep choosing the Best Pick for 2026 Week 1**,
budgeted at its honest **+0.9-point** edge, not +8.68, and reported with
`probability_positive` near a coin flip, not 0.865. `docs/pool_format_levers.md`
§6 item 2 (surface the tie instead of hiding it — already live for 2026 Week
1's ARI@LAC/WAS@PHI tie) remains the correct complementary fix: since a
majority of weeks are ties, most Best Pick weeks are, honestly, a reproducible
coin flip between the tied games, and the card should say so.

**Summary verdict**: not "stays confirmed", not "refuted" — **the signal is
directionally real (positive tie-agnostic estimate on both independent
windows) but the recorded magnitude and confidence are artifacts of the
alphabetical tie-break and an understated interval, and the registry
classification should say `unresolved`, not `confirmed`.**

## 2026-08-18: the weekly NOMINATION rule switches to the measured v2 winner

Owner decision, same day as the re-read above. The pool is forced picks, so
choosing which already-picked game gets the week's bonus points costs
nothing — an unjustified nomination is free money left unclaimed, whichever
rule chooses it (`AGENTS.md`, "edge means beating 50%" / forced-pick
framing). `sweep_robustness` (§ above) is itself measured signal-free at a
fair test: tie-agnostic +0.92 points, honest `probability_positive`
**0.536–0.554**, nowhere near a confirmation gate. A same-day exploratory
screen (**read**, session scratchpad `scratchpad/bestpick_opener/predeclaration.md`
+ `results.md`, script `scripts/best_pick_opener_ranker_eval.py`) measured a
stronger lean for a different chooser on the same 107-week opener
population `docs/opener_evaluation.md` already used:

- **`dispersion_filtered_candidate`** (chooser 6): the alpha=2000 candidate
  probability's distance from 0.5 (`docs/ridge_alpha.md` § 4's named
  walk-forward Brier optimum, confirmed at the opener grade by
  `artifacts/ridge_alpha_promotion/20260818T221459Z`), restricted to that
  week's below-median cross-book opener `spread_std` games (fallback to the
  full week on missing/degenerate dispersion data). Scored **+3.92 points**
  vs its unfiltered parent (chooser 4), `probability_positive` **0.813**,
  week-blocked interval **[−3.92, +11.76]**, 102 paired weeks.

This is **not a resolution**: the interval contains zero, the sample is 102
weeks, and it is the **third reuse** of the same 107 opener weeks this
session (ridge_alpha promotion, the odds-microstructure battery, this
ranker screen) — the multiplicity discount compounds each look and is
stated explicitly everywhere this number is quoted. But `probability_positive`
0.813 is the single strongest lean measured anywhere in either screen
section, clears the 0.75 "worth a dedicated look" bar this project has used
elsewhere (SPEC-5's own screen gate), and — unlike `sweep_robustness` — has
**no positive alternative on the table that beats it**: the two other
predeclared v1 alternatives (`calibrated_probability`, `key_number_distance`)
both scored outright negatives. **EV rationale**: swapping a measured-negative-adjacent,
signal-free incumbent for a measured, positive-leaning (if unresolved)
challenger is EV-positive on a forced pick even though neither rule clears a
promotion bar — per `AGENTS.md`, a promotion bar gates registry claims, not
which card gets played (the same principle the § 4 "play decision" analysis
above already applied to keep `sweep_robustness` in play over its own
alternatives).

**What actually ships is a composition, not chooser 6 verbatim.** Chooser 6's
own predeclared tie-break is ascending `game_id`, same as every other
chooser in that screen. A *separate* chooser (8, `dispersion_tiebreak`, run
on the *unfiltered* week) tested breaking ties by lower dispersion instead,
and it read `probability_positive` 0.0 — flagged in `results.md` as a
5-week degenerate artifact (only 5 of 107 weeks ever tie on
`candidate_prob_distance`, and this sample's tie-break happened to lose all
5), not a real negative. The production rule the owner specified composes
the two pieces: chooser 6's filter, PLUS a dispersion tie-break applied
*inside* the filtered pool. **That exact composition was never itself
scored as one chooser** — implemented anyway per the owner's explicit
instruction, flagged here and in the implementation
(`src/nfl_ats/best_pick_nomination.py`'s module docstring) rather than
silently presented as identical to the measured chooser 6.

**Implementation**: `src/nfl_ats/best_pick_nomination.py` (new module,
`nfl_ats.best_pick.py` stays untouched and frozen — SPEC-5's confirmation
depends on `sweep_robustness`'s exact definition never drifting). The
alpha=2000 probability is fit walk-forward at publish time via
`nfl_ats.outcomes.fit_margin_models_for_week` (same training-cutoff
discipline every other weekly forecast uses); dispersion is read from the
LOCAL market snapshot store the weekly pipeline already populates via
`odds-ingest` (`nfl_ats.market_data.tuesday_opener_quotes`, extended with an
`opener_std` column — the historical evidence's `spread_std` has no
production equivalent since live captures carry no decision-label
structure). **Sides never change** — every game's forced pick still comes
from the active model exactly as before; only which ONE game gets the ★
Best Pick mark moves. `publishing.py` computes BOTH rules' nominations every
week (`NOMINATION_V2_ENABLED = True` switches which one is actually
marked/played; v1 remains the fallback whenever v2's infrastructure is
unavailable — no feature table on the forecast, no market snapshot, not
enough walk-forward training history yet — mirroring the coach-fade
overlay's "missing input degrades, never fails the publish" contract). The
published card discloses the method in plain language
(`"nominated by calibrated probability among low-disagreement games"`,
verbatim, plus a fallback/tie sentence when either applies).

**2026 scores both arms.** v1's nomination needs no new tracking — it is
already recorded, unchanged, via the active model's own `is_best_pick` flag
on the primary paper-decision ledger. v2's weekly nominee is separately
recorded to the prospective challenger ledger under `best_pick_nomination_v2`
(`artifacts/prospective/challengers.json`, pinned to the active model's
configuration fingerprint, mirroring `hc_year_one_fade_overlay`'s pattern
exactly) via `nfl_ats.best_pick_nomination.record_nomination_challenger_decisions`,
wired into `publish-predictions --record-decisions` alongside the existing
two ledger writes. No promotion decision is implied by a partial season —
this settles a nomination-rule decision already made, not a candidate
awaiting a threshold, and the registry verdict on `best_pick_ranker_opener`
(`unresolved`) is unchanged by this switch.
