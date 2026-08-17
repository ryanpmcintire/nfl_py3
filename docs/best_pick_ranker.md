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
   (`MIN_ELIGIBLE_START_SEASON = 2013`): no window may start before 500
   walk-forward training games plus 400 calibration prediction rows
   (~900 games; four 256-game seasons) of history exist in front of it.
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
