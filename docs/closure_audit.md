# Terminal-closure audit: what `ROADMAP.md` closed that `revisit_list.md` never saw

Written 2026-08-18. `docs/revisit_list.md` triaged the verdicts recorded in
`registry/weak_signals.json` and `registry/rotation_registry.json` against the
five defects found in the measurement instrument (D1-D5, restated below). It
never looked at `ROADMAP.md`'s own closure table (the "Sensitivity-aware
review of completed experiments" table, ~line 391-404) or the closure
language scattered through the phase tables — those verdicts were decided
with the same defective instrument and were never triaged. This document is
that triage. It recomputes what it can from artifacts already on disk; it
proposes registry writes and `ROADMAP.md` text edits without making them,
per this task's constraints.

**Every claim below is tagged measured / read / reported / inferred, per
`AGENTS.md`.** Note on scope: `docs/evaluator_power.md`, named in this task's
instructions, does not exist in this repository (**measured**:
`Read` on that path fails; **measured**: `git log --all -- docs/evaluator_power.md`
returns nothing — it was never committed). The MDE80 formula and the
power/resolution facts this task attributes to it live instead in
`docs/estimation_variance.md` and `src/nfl_ats/estimation_variance.py`
(**read**), which is what this document cites throughout.

## The five defects, restated for reference

- **D1** — the probability-calibration step may attenuate or invert small
  effects. Demonstrated only on planted effects (+1.3 came back -0.7,
  wrong sign); the one real-data reading found only a ~0.1-point gap.
  **Unverified on real effects; another agent is settling this.** Flagged
  as conditional below wherever it would matter.
- **D2** — naive block-bootstrap intervals are 17-58% too narrow (measured
  range in `docs/estimation_variance.md` real-CFB comparisons: 3.7% to
  33.0%; synthetic null DGP: up to 57.5%). `probability_positive` is
  overstated project-wide.
- **D4** — the bootstrap is degenerate below ~4-5 blocks (`docs/anytime_valid.md`
  §6).
- **D5** — a 0.90/0.75 threshold governs what docs may CLAIM, never what gets
  PLAYED (forced-pick pool).

## Summary

**7 terminal closures were audited** beyond the two rows already identified
(the Participation RAPM row and the PER-07 4th-down framing, both handled in
depth below). Of those 7: **4 have admissible grounds** (refuted mechanism or
control-bounded, and survive a D2 sanity check) — Conditional margin variance
(MOD-16), Opponent-adjusted PBP/matchup transfer-only closure, Pace/possession
forecast (PBP-09), and the Broad 48-row player-selection-grid closure. **2 do
not** and should be reclassified `unresolved_below_power` — Participation
offense/defense RAPM and the PageRank/HITS schedule graph. **1 is a different
kind of closure the taxonomy doesn't govern at all** — Pick-popularity input
(POL-04), closed because the data does not exist, not because an effect was
measured and found absent. Two more rows (Snap-weighted player value; Beta/other
calibration) were checked and are **already correctly framed as open**, not
closures — no change needed beyond one flagged reasoning caveat.

**The one item most worth actually re-running: Participation offense/defense
RAPM.** Unlike every other item here, it (a) is nowhere close to bounded by
power — MDE80 at this sample is **~1.43 accuracy points** against an observed
effect of **-0.43**, a 3.3x gap — so the accuracy channel it was closed on
cannot distinguish "no effect" from the effect actually claimed; (b) its two
"resolved" Brier/log-loss channels sit only **8-10% of a bootstrap-width
away** from re-crossing zero, well inside D2's demonstrated 3.7-33.0% real
range, so they are the single most fragile "resolved negative" found in this
sweep; and (c) it blocks **live, in-progress work** (PER-05/PER-09, both
🚧) rather than a dead line, and `docs/availability_confirmation.md`'s own
mechanism screen already found the closely related `value_magnitude` axis of
the SAME broader family mechanistically resolves (r=+0.248, p=0.016) — a
value-weighted redesign of the RAPM construct, screenable on CFB first at
zero rotation-window cost (rule 8), is a natural, cheap next step that
nothing here recommends skipping ahead to games-collection for.

---

## 1. Participation offense/defense RAPM — the highest-consequence finding

**Recorded**: `ROADMAP.md` ~line 401 ("Keep closed in current form"), the
`participation` arm of `player_experiments`/PER-09. **Not** in
`registry/weak_signals.json` at all — no entry exists (**measured**:
`registry/weak_signals.json` has 16 entries, none named for this family).
[**updated 2026-08-18**: the registry now has 19 entries — this doc's own
proposals below added `participation_offense_defense_rapm` and
`graph_schedule_rating_brier`, and parallel work added `cfb_role_continuity`
(**measured**: diffing the current file against commit `cfa3ecc`, which
matches this section's original count of 16, shows exactly those three
additions and no removals). This family itself is still unrecorded.]

**Stated grounds** (ROADMAP verbatim): *"Accuracy fell 0.43 points and Brier
worsened with a season-blocked interval excluding zero in the wrong
direction."*

**Underlying artifact, recovered**: `artifacts/participation_experiments/
20260813T132030Z/paired_comparisons.csv` (**measured**, still on disk):

| metric | block | estimate | 95% CI | crosses zero? |
|---|---|---|---|---|
| accuracy_improvement | week | -0.4337 pts | [-1.5319, +0.6250] | yes |
| accuracy_improvement | season | -0.4337 pts | [-1.0130, +0.0960] | yes (barely, upper +0.096) |
| brier_improvement | week | -0.000825 | [-0.001817, +0.000059] | yes |
| brier_improvement | **season** | -0.000825 | **[-0.001712, -0.000065]** | **no** |
| log_loss_improvement | week | -0.001792 | [-0.003915, +0.000064] | yes |
| log_loss_improvement | **season** | -0.001792 | **[-0.003711, -0.000177]** | **no** |

The ROADMAP sentence conflates two different channels. Read plainly, "Brier
worsened with a season-blocked interval excluding zero" is true only of
Brier/log-loss season-blocking; the accuracy claim ("Accuracy fell 0.43
points") is a point estimate whose own interval **crosses zero at both
blockings** — which `AGENTS.md`'s binding rule says is explicitly not
grounds for closure on its own.

**f, MDE80 (measured, `scripts` none exist for this; recomputed directly
from `artifacts/participation_experiments/20260813T132030Z/predictions.parquet`
this session)**:

- Pick-disagreement fraction `f` (player_value vs. player_participation,
  `home_cover_probability >= 0.5` tie convention) = **5.55%** on 2,127 paired
  games (the artifact's own recorded n is 2,075; the small gap is from how
  the raw parquet's arm-intersection was taken and does not change the
  conclusion).
- `MDE80 = 280 * sqrt(f/n)` (`src/nfl_ats/estimation_variance.py:374`,
  formula per this task's brief) = **280 * sqrt(0.0555/2127) ≈ 1.43 accuracy
  points**.
- Observed effect: **-0.43 points** — well below MDE80. **The accuracy
  channel this closure leans on cannot resolve an effect this size in
  either direction; a closure grounded partly in "accuracy fell" is
  grounded in a channel with no power to have shown otherwise.**

**Normal-approximation `probability_positive` (inferred, computed this
session from the recorded intervals via `resolved_standard_error()`'s own
`width / (2 * 1.96)` convention, `src/nfl_ats/weak_signals.py:144-155` — not
a re-run of the bootstrap)**:

| metric/block | inferred P+ (favours candidate) |
|---|---|
| accuracy, week | 0.215 |
| accuracy, season | 0.063 |
| brier, week | 0.042 |
| brier, season | 0.025 |
| log-loss, week | 0.039 |
| log-loss, season | 0.023 |

All lean against the candidate — this is a real directional lean, not a
coin flip — but none of the accuracy readings reach a resolved 0.05/0.95
tail on their own, and the "resolved" Brier/log-loss season readings are
fragile (next point).

**D2 sensitivity (inferred; applying the width-inflation logic
`docs/estimation_variance.md` derives, not a re-run of the refit-aware
bootstrap on this specific comparison)**: the fractional interval-widening
needed to push each "resolved" season-blocked bound back across zero:

| metric | widening needed to re-cross zero |
|---|---|
| brier, season | **7.9%** |
| log-loss, season | **10.0%** |

`docs/estimation_variance.md`'s own measured real-CFB range is **3.7% to
33.0%** width inflation, and it explicitly states the understatement is
**worst** for exactly this shape of comparison — "small, surgical,
single-feature-family additions" (§3) — which this is (two added columns).
An 8-10% widening is comfortably inside, and below the middle of, that
measured range. **These two "resolved negative" readings are the most
fragile in this entire sweep** and most likely evaporate under the honest
refit-aware bootstrap.

**Admissible ground?** **NEITHER.** Not refuted mechanism (no split-half
reliability failure was tested; the disagreement rate is small, not zero,
and the sign is consistent but statistically unresolved by its own power).
Not control-bounded: RWB-15's own NFL positive-control audit
(**read**, `ROADMAP.md` line 93, and reproduced in the "Recommended
execution order" section) found a **0.5-point** true effect clears the
week-blocked interval only **3/8** times and a **1-point** effect only
**2/8** times on comparisons of this scale — i.e., the instrument is
independently shown **not** to reliably detect effects at or below 0.43
points, which is the opposite of a positive control proving detection.
**Recommendation: reclassify `unresolved_below_power` (category 3), record
it, and do not treat the Brier/log-loss "resolution" as settled until
re-measured with the honest refit-aware bootstrap
(`src/nfl_ats/estimation_variance.py`, already built, unused on this
comparison).**

### Proposed `ROADMAP.md` text change (not applied — propose only, per this task's constraints)

**Old** (~line 401, cell text only):
> Accuracy fell 0.43 points and Brier worsened with a season-blocked interval
> excluding zero in the wrong direction. Position-unit or matchup hierarchy
> would be a new model, preferably learned with CFB; alpha retuning is not
> admitted.

**New:**
> Accuracy fell 0.43 points, week-blocked 95% [-1.53, +0.63] and season-blocked
> [-1.01, +0.10] — both cross zero, and MDE80 at this sample (f=5.5%,
> n=2,075) is ~1.4 points, well above the observed effect, so accuracy alone
> cannot resolve this either way. Brier and log-loss resolve worse under the
> naive season-blocked bootstrap ([-0.00171, -0.00007] and [-0.00371,
> -0.00018]), but both sit within 8-10% of re-crossing zero — inside the
> 3.7-33.0% width understatement `docs/estimation_variance.md` measures for
> comparisons this size — so this is a lean, not a settled negative. Recorded
> as `participation_offense_defense_rapm`
> (`registry/weak_signals.json`), category 3, rather than closed. Position-unit
> or matchup hierarchy would be a new model, preferably learned with CFB;
> alpha retuning is not admitted.

---

## 2. PER-07, 4th-down aggressiveness — the banned conclusion

**Recorded**: `ROADMAP.md` ~line 155, and `registry/weak_signals.json` entry
`fourth_down_aggressiveness` (**measured**: `python -c "import json;
json.load(open('registry/weak_signals.json'))"`). Entry: `classification:
unresolved_below_power`, `effect: -0.038` (ats_points), `interval: [-0.423,
0.417]`, `probability_positive: null`, `sample_games: 4175`,
`sample_blocks: 17`, `seasons: [2009, 2025]`, `source: ROADMAP.md#PER-07`.

**The banned sentence** (ROADMAP verbatim): *"Resolving 0.174 points at 95%
would need roughly 24,000 games — about 90 NFL seasons (CFB's 12,500 does
not close that either)."* Per `docs/scaling_and_transfer.md` (**read**), this
project is model-limited, not data-limited (forced-pick accuracy flat across
a 100-fold range of training-set size, three convergent measurements) — the
sentence's own conclusion ("needs N more games") is exactly the form this
task's brief identifies as banned, and this document does not repeat it.

**What is already correct and must be preserved**: 4th-down aggressiveness
is genuinely reliable (year-over-year +0.320 on 512 team-season pairs), is
explicitly **not closed**, and "belongs in the weak-signal stacker or
nowhere" — that conclusion survives unchanged; only the "wait for more
games" framing needs to go.

**Recoverable continuous evidence**: no artifact/script for the market-test
regression was found on disk (**measured**: grep across `docs/`, `scripts/`,
`artifacts/` for coach/4th-down/aggressiveness terms found only ROADMAP prose
and the registry entry — no standalone doc or CSV). What IS recoverable is
`probability_positive`, via the same normal approximation used above
(**inferred**, from the registry's own recorded interval): `SE = 0.84 /
(2*1.96) = 0.2143`, `P+ = P(effect > 0) ≈ 0.430`. This is a near-coin-flip
lean, consistent with "unresolved" — not evidence the true effect is zero,
and not grounds to treat the closed hypothesis differently than the row
already (correctly) concludes.

**Admissible ground?** N/A — this row is **not closed** (status 🚧, and the
row's own text already says "NOT closed"). The defect here is the banned
data-limited conclusion, not a false closure.

### Proposed `ROADMAP.md` text change (not applied — propose only)

**Old** (~line 155, tail of the PER-07 cell):
> Resolving 0.174 points at 95% would need roughly **24,000 games -- about 90
> NFL seasons** (CFB's 12,500 does not close that either). So this can never
> be confirmed as a standalone candidate and must not be spent on a window;
> it belongs in the weak-signal stacker or nowhere.

**New:**
> Per `docs/scaling_and_transfer.md` this project is model-limited, not
> data-limited — forced-pick accuracy is flat across a 100-fold range of
> training-set size — so more games would not resolve this even if they
> existed, and the ~24,000 required will never exist (NFL yields ~4,431, CFB
> ~12,500). This can never be confirmed as a standalone candidate and must
> not be spent on a window; it is recorded
> (`fourth_down_aggressiveness`, `registry/weak_signals.json`,
> `probability_positive` ≈0.43 inferred from the recorded interval) and
> belongs in the weak-signal stacker, not in waiting.

---

## 3. PageRank/HITS schedule graph — inadmissible, and the underlying artifact is missing entirely

**Recorded**: `ROADMAP.md` ~line 402 ("Keep closed in current form") and
MOD-15 (line 192). Detail lives only in `docs/modeling.md` lines 33-63
(**read**). Not in `registry/weak_signals.json`.

**Stated grounds**: *"Graph candidates were selected in 0/8 outer seasons and
worsened probability/margin diagnostics."*

**What `docs/modeling.md` actually reports**: season-blocked Brier
improvement **-0.000186, 95% [-0.000376, +0.000005]** — the upper bound is
**positive**; this interval does **not** exclude zero (**read**, lines
50-56). Cover Brier was worse in **7 of 8** fair-margin seasons and **6 of
8** market-residual seasons (**read**, lines 58-63) — a directional pattern,
not a resolved interval. Two-sided sign-test p-values for those splits
(**inferred**, exact binomial): 7/8 → p≈0.070; 6/8 → p≈0.289.
Normal-approximation `probability_positive` on the one interval given
(**inferred**, same method as above): **≈0.028** — a strong lean, but the
interval itself still technically crosses zero, so a claim of "excluded"
would be wrong even though the lean is real.

**No artifact on disk.** Unlike the RAPM case, no `artifacts/*graph*` or
`artifacts/*pagerank*` directory exists (**measured**: searched `artifacts/`
for both terms, zero matches), and no dedicated doc holds the underlying
CSV/parquet. This is the same "headline number, no artifact" failure mode
`docs/availability_confirmation.md` already caught once for the MOD-07 0.899
figure — it recurs here, independently, and is worse: for RAPM the number
was at least recomputable; for this one it is not, from this session's
constraints (no rotation window, no re-run).

**Admissible ground?** **NEITHER**, on the strict letter — no split-half
reliability test was run (not `refuted_mechanism`), and no positive control
proves the diagnostics used here can detect an effect this size (not
`bounded_by_control`). But this is the closest thing in the sweep to a
legitimate third category the taxonomy doesn't explicitly name: **a
consistent directional replication across multiple quasi-independent splits**
(0/8 selections, 7/8 and 6/8 season signs, two separate model comparisons
agreeing). That is real evidence, in the same spirit RWB-16's own sign-test
logic treats as informative — just not resolved to the project's stated
standard, and not verifiable at all right now because the artifact is gone.
**Recommendation: reclassify `unresolved_below_power`, record with the
number available, and do not claim "worsened diagnostics" as if it were a
resolved interval** — it is a consistent lean under selection and sign
counts, not an excluded-zero finding.

### Proposed `ROADMAP.md` text change (not applied — propose only)

**Old** (~line 402):
> Graph candidates were selected in 0/8 outer seasons and worsened
> probability/margin diagnostics. CFB may support player/unit graphs, but it
> does not warrant rerunning this team graph.

**New:**
> Graph candidates were selected in 0/8 outer seasons; cover Brier was worse
> in 7/8 and 6/8 seasons across two model comparisons (one-sided sign-test
> p≈0.07 and p≈0.29), but the season-blocked Brier interval itself
> ([-0.000376, +0.000005]) does not exclude zero. A consistent negative
> lean across replications, not a resolved refutation — recorded as
> `graph_schedule_rating_brier` (`registry/weak_signals.json`), category 3.
> CFB may support player/unit graphs, but the lean gives no reason to rerun
> this specific team graph without a new mechanism.

---

## 4. Conditional margin variance (MOD-16 screen) — admissible, correctly closed

**Recorded**: `ROADMAP.md` ~line 397 and line 193 (MOD-16); full detail in
`docs/margin_variance.md` (**read**, entire file). **Stated grounds**: a
frozen, predeclared CFB screen (8,933 clean-core games, 2006-2025) with a
decision rule fixed *before* scoring — "clears only if the week-blocked 95%
interval excludes zero from below." It did not clear; it resolved in the
**wrong** direction on both primary and secondary metrics.

| metric | block | estimate | 95% CI | width-inflation needed to re-cross zero |
|---|---|---|---|---|
| cover log-loss (primary) | week | -0.000335 | [-0.000558, -0.000122] | 56.0% |
| cover log-loss (primary) | season | -0.000335 | [-0.000607, -0.000075] | 28.2% |
| Brier | week | -0.000148 | [-0.000239, -0.000058] | 64.1% |
| Brier | season | -0.000148 | [-0.000263, -0.000037] | 32.7% |

(**measured**: read directly from `docs/margin_variance.md`; widening
percentages **inferred**, same method as above.)

**This is the strongest-powered closure in the sweep.** It is an
8,933-game CFB screen (not the thinner NFL 2,075-game scale), both week- and
season-blocking agree in direction, both Brier and log-loss agree, and the
decision rule was **frozen before the run** — the exact predeclaration
discipline `AGENTS.md` asks for elsewhere. The needed widening (28-64%
across the four cells) mostly sits at or above the top of D2's
real-CFB-measured range (3.7-33.0%), and the two week-blocked cells (56%,
64%) exceed even the synthetic-null maximum (57.5%) in one case. This is
also the kind of comparison `docs/estimation_variance.md` §7 explicitly
scopes **out** of its own methodology — it holds the mean model fixed and
varies only how the residual's width is read, "a different, unaddressed
variance source" the refit-aware bootstrap does not claim to fix. **Note the
season-blocked margin (28-33%) is closer to the edge than the week-blocked
one (56-64%) and is the one worth a second look if this is ever revisited,
but nothing here recommends prioritizing that.**

**Admissible ground?** **Yes — closest to `refuted_mechanism` in spirit**
(a resolvably wrong-signed result on a well-powered, frozen-rule screen,
coherent across two metrics and two blockings), though not literally a
split-half reliability failure. **No action needed; do not reopen.**

---

## 5. Opponent-adjusted PBP and matchup effects — legitimately control-bounded, confirmed correct

**Recorded**: `ROADMAP.md` ~line 399 ("Revisit only through transfer").
**Read**, `docs/cfb_opponent_adjustment.md` lines 217-233: a deliberate-leak
positive control (the adjustment fit on the **entire** 2006-2025 history, so
it sees the future) moved clean-core margin MAE by **+0.0129, `probability_
positive` 0.984**, while the honest arm moved it by only -0.0003,
`probability_positive` 0.463. This is a textbook positive control: the
identical harness, on the identical games, with the identical bootstrap,
detects a known-present effect at high confidence and fails to detect the
honest one — proving the null is a measured absence, not underpowered
silence.

**Admissible ground? Yes — `bounded_by_control`, cleanly.** This is the one
row this task's brief flagged as likely-correct in advance, and the read
confirms it. **No action needed. Do not reopen**, and the "revisit only
through transfer" framing (use CFB to find a low-dimensional mechanism, then
freeze one NFL transfer test) is the right shape for any future attempt.

---

## 6. Pace and possession forecast (PBP-09) — admissible, refuted mechanism

**Recorded**: `ROADMAP.md`, Phase 3 table, PBP-09 row (not in the
sensitivity-review table, but a terminal "closed on measurement, not built"
claim). **Read**, ROADMAP verbatim: on 1,024 games (2009-2012, no window
spent), forecasting game pace from both teams' prior pace reaches only
**R² 0.041**; `corr(plays, |margin|) = -0.20` — the **wrong sign** for the
premise that more plays would signal something about margin (blowouts have
*fewer* plays, from clock-killing and kneel-downs) — and
`corr(drives, |margin|) = +0.004`, indistinguishable from zero.
"Independently reproduced by a second measurement before this row was
changed."

**Admissible ground? Yes — `refuted_mechanism`.** The correlation is
resolvably the **wrong sign** for the hypothesis (ground 1 of the binding
taxonomy is explicitly "wrong sign"), and R²=0.041 with a negative-sign
control leaves no forecastable quantity for a downstream ATS mechanism to
use. This is not an interval-crossing-zero closure at all; it is a direct
sign refutation with an independent replication. **No action needed.**

---

## 7. Broad 48-row player selection grid — admissible, a different kind of ground

**Recorded**: `ROADMAP.md` ~line 404. **Stated grounds**: *"The nested
selector failed and pooled winners are multiplicity-exposed... the grid
itself is not independent evidence."* Confirmed in `docs/modeling.md:135-139`
(**read**): "QB+continuity/alpha-1/uncalibrated... reached 52.63%... chosen
only after all 48 rows were visible."

**Admissible ground?** This does not map cleanly onto either named ground —
it is not claiming a measured negative at all, refuted or otherwise. It is a
**multiplicity/selection-validity** claim: the grid's best row was chosen
*post hoc* out of 48 candidates, so its headline number is a selection
artifact, not evidence the underlying features don't work. This is the
**opposite** direction of concern from D1/D2 (which worry real positives get
read as negatives); here the worry is a false positive from search, and nothing
in D1/D2/D4/D5 touches that concern. **Correctly scoped already** — the row
explicitly preserves the individual rows as "one mechanistic hypothesis," it
only retires the *grid as an evidentiary method*. **No action needed.**

---

## 8. Two rows checked and already correctly open (no closure, no change needed)

- **Snap-weighted player value** (`ROADMAP.md` ~line 398, "Redesign, then
  reopen"): already states the +0.10-point extension "was too small to
  resolve" — category 3 in its own words, already open for redesign. No
  false-closure language to fix.
- **Beta/other calibration** (`ROADMAP.md` ~line 400, "Revisit only after a
  stronger signal"): not a hard closure, and cites no interval-crossing-zero
  claim. One reasoning caveat worth flagging for whenever this is revisited:
  its premise ("calibration... does not create side information") rhymes
  with the "rescaling can't change a pick" argument `ROADMAP.md`'s own MOD-06
  entry (line 183) explicitly **retracted** on 2026-08-17 (the production
  rule reads a nonzero-median residual sample, so rescaling *can* flip
  picks), and `registry/weak_signals.json`'s `ecdf_smoothing_accuracy` entry
  independently shows a distribution-shape change moves **8.8%** of forced
  picks. Not urgent — nothing is blocked on this row — but the next time it
  is revisited, do not lean on the retracted argument.

## 9. Pick-popularity input (POL-04) — a closure the taxonomy doesn't govern

**Recorded**: `ROADMAP.md` ~line 231, ❌ "Closed for lack of data, not
effort." **Read**: Splash's pick-distribution feature unlocks game-by-game
only as each game kicks off (structurally unavailable before the Tuesday
lock), no odds API sells betting percentages, and no free ticket/handle feed
exists. This is a **data-nonexistence** closure — nothing was measured and
found absent; the input the row wants does not exist to measure. The binding
taxonomy in `AGENTS.md` governs closing a line of work on a **measured**
negative; it has nothing to say about a line that cannot be measured at all.
**No action needed; flagged only so this sweep's count is honest about what
it is and is not.**

---

## What should be recorded as weak signals

All three commands below are **proposals only** — this session does not
write `registry/*.json` per this task's constraints.

**1. Update `fourth_down_aggressiveness`** to carry the inferred
`probability_positive` (the entry already exists; needs `--replace`):

```
nfl-ats weak-signals record --replace \
  --name fourth_down_aggressiveness \
  --description "Coach go-for-it rate above situational expectation, lagged a season." \
  --source "ROADMAP.md#PER-07" \
  --effect -0.038 --effect-units ats_points \
  --classification unresolved_below_power \
  --league nfl --season-start 2009 --season-end 2025 \
  --interval-low -0.423 --interval-high 0.417 \
  --probability-positive 0.43 \
  --sample-games 4175 --sample-blocks 17 --reliability 0.32 \
  --classification-evidence "The completely-unpriced hypothesis is +0.174 pts and sits INSIDE the interval. Resolving it at 95% would need games this project will never have; per docs/scaling_and_transfer.md the project is model-limited, not data-limited, so this is a stacker input, not a wait." \
  --notes "probability_positive is INFERRED this session via normal approximation from the already-recorded interval (SE = width / (2*1.96)), not a bootstrap re-run. Naive figure without the season-norm correction is +0.666."
```

**2. New entry, Participation offense/defense RAPM** (accuracy channel; the
Brier/log-loss channels are flagged as fragile in the notes rather than
recorded as separate resolved entries, since they are the least trustworthy
numbers in this whole audit):

```
nfl-ats weak-signals record \
  --name participation_offense_defense_rapm \
  --description "Participation-based RAPM (ridge over play participation, offense/defense) added on top of the player_value composite." \
  --source "artifacts/participation_experiments/20260813T132030Z/paired_comparisons.csv" \
  --effect -0.4337 --effect-units accuracy_points \
  --classification unresolved_below_power \
  --league nfl --season-start 2018 --season-end 2025 \
  --interval-low -1.5319 --interval-high 0.6250 \
  --probability-positive 0.215 \
  --sample-games 2075 --sample-blocks 141 \
  --classification-evidence "MDE80 at f=5.5%, n=2127 is ~1.43 accuracy points, 3.3x the observed -0.43 -- accuracy alone cannot resolve this. Season-blocked Brier [-0.001712,-0.000065] and log-loss [-0.003711,-0.000177] naive intervals exclude zero, but need only 7.9% and 10.0% width inflation to re-cross it -- inside docs/estimation_variance.md's measured 3.7-33.0% real range, the most fragile 'resolved' readings in this project's audit." \
  --notes "probability_positive is INFERRED via normal approximation from the recorded interval, not a bootstrap re-run. Week-blocked accuracy P+=0.215, season-blocked P+=0.063 (also inferred) -- reported for completeness, not resolved. RWB-15's own NFL positive-control audit finds a true 0.5pt effect clears the week-blocked interval only 3/8 times and a 1pt effect only 2/8 times at this scale, so this is independently shown to be below this instrument's detection power, not bounded by a control."
```

**3. New entry, PageRank/HITS schedule graph** (lower priority — nothing is
blocked on this one, but it is currently mis-stated as "worsened
diagnostics" without qualifying that the cited interval does not itself
exclude zero):

```
nfl-ats weak-signals record \
  --name graph_schedule_rating_brier \
  --description "Temporal PageRank/HITS schedule-graph features added to market_context." \
  --source "docs/modeling.md" \
  --effect -0.000186 --effect-units brier \
  --classification unresolved_below_power \
  --league nfl --season-start 2018 --season-end 2025 \
  --interval-low -0.000376 --interval-high 0.000005 \
  --probability-positive 0.028 \
  --classification-evidence "Interval does not exclude zero (upper bound +0.000005); 0/8 nested-selection wins and 7/8, 6/8 same-sign season splits across two model comparisons are the real evidence, sign-test p~0.07 and p~0.29 -- a consistent lean, not a resolved refutation." \
  --notes "No stored artifact exists for this screen (searched artifacts/ for graph/pagerank terms, zero matches) -- only ROADMAP/docs/modeling.md prose survives. probability_positive is INFERRED via normal approximation from the one recorded interval, not re-run. sample_games/sample_blocks omitted: not stated in the surviving prose and not independently recoverable this session."
```

## Which one is most worth actually re-running

**Participation offense/defense RAPM.** Restated with the reasoning
together: it is the only item in this sweep where (1) the channel the
closure leans on (accuracy) is proven underpowered by more than 3x rather
than merely narrow, (2) the channels that DO nominally resolve (Brier,
log-loss) are fragile enough (7.9-10.0% width away from flipping) to sit
inside the low-middle of this project's own measured real-world interval
understatement, not at some untestable extreme, and (3) it gates **active**
work — PER-05 and PER-09 are both still 🚧 — rather than a line nobody is
touching. The cheapest next step is not collecting more NFL games (banned
reasoning, and MDE80 says it would not help at any plausible scale this
project could reach anyway): it is re-testing a **value-weighted** RAPM
variant on CFB first, free under rotation rule 8, informed directly by
`docs/availability_confirmation.md` §3's finding that the `value_magnitude`
axis of this same broader family already passes a mechanism test the
plain-participation axis fails.
