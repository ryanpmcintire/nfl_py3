# Following the heavy handle, split by how hard it overrides the model

Written 2026-09-14, before any number in **Results** was computed. Occasioned
by 2026 Week 1: `handle_follow_0_70` flipped ARI +9.5 to LAC -9.5 at the
Saturday 12:15 ET last-call pass, onto a side the served model priced at
**0.362**, and it lost. The other three flips that pass overrode sides the
model priced at 0.512, 0.507 and 0.509 — near coin flips.

## Question

`docs/handle_follow_on_card.md` promoted H1 (follow the side holding >= 70% of
the spread money) on 19-18 across 37 flipped picks: +0.38 accuracy points,
week-blocked [-3.00, +4.28], `probability_positive` 0.5695. That measurement
pooled every flip regardless of how confident the card was in the pick being
overridden. **Does following the heavy handle still pay when it overrides a
pick the card holds with conviction, or is its whole value in the coin
flips?**

Per the owner's standing directive (2026-09-13, signals are tested in splits),
both halves are measured and decided together. This changes a **SIDE**, so it
is a served-policy question, not a display one.

## Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero". Within-week correlation is ZERO; week-blocked bootstrap.
Decide on expected value: the card is forced, so the arm with the higher
expected accuracy is served; 0.90-style thresholds govern only what the docs
may claim.

## Population

The frozen lane-S population, reused read-only and not rebuilt:
`artifacts/handle_follow_on_card/20260909T232503Z/population.parquet` — 260
opener-scored REG games, 40 `(season, week)` blocks, seasons 2020-2025, of
which **133 carry handle** (2023/30, 2024/71, 2025/32). It already carries the
served nine-member card state (`served_pick_home`, `correct_served`,
`served_flipped`), the served opener probability
(`home_cover_probability_at_open`) and the split percentages.

The run **reproduces H1 first** — 73 fires, 37 flips, +0.38 accuracy points —
and aborts if any of those three disagree with the stored `result.json`. No
re-implementation.

**Multiplicity, stated plainly:** this is a second look at lane S's own
population, splitting a rule already being played. No rotation window is
assigned or spent. Every number carries that discount.

## The split variable, defined before any score

**Override distance** = the served card's own probability for the pick the
handle rule is about to overturn, minus 0.5. It is known pregame, it is the
number the card prints, and it is exactly "how sure was the card about the
side we are throwing away". ARI at LAC scored 0.138; the week's other three
flips scored 0.012, 0.007 and 0.009.

The cut is **the median override distance across H1's own flips**, computed
from the data, not chosen. No tuned constant.

## Predeclared arms (exactly three plus a control)

Sides outside the arm's own flip set are untouched in every arm.

- **G0 — H1 as served (incumbent).** Every H1 flip.
- **G1 — shallow overrides only.** H1 flips whose override distance is at or
  below the median; deep overrides are left on the card's own side.
- **G2 — deep overrides only.** The complement: H1 flips above the median.
- **Control — perfect foresight** on H1's flip set, to show the harness
  resolves an effect at this flip count.

## Predeclared grade and metric

Opener-graded forced-pick accuracy on the 260-game population, paired per game
against G0, week-blocked bootstrap on `(season, week)`, 20,000 resamples, seed
20260914, effect in accuracy points. Season-blocked reported alongside. Report
the effect, the interval and `probability_positive` before any verdict, and
report the flipped-pick records, which are the whole evidence base.

## Decision rule, fixed in advance

The card is forced. Whichever of G0/G1/G2 has the higher expected accuracy is
served; `probability_positive > 0.5` is the whole bar. If G1 leads G0, the
handle rule is gated to shallow overrides and the deep ones stay with the
model. A dead heat leaves the incumbent alone. Nothing here may close H1: a
single losing week is not a ground, and neither is an interval containing
zero.

## Results

Measured 2026-09-14. Artifact
`artifacts/handle_follow_override_distance/20260914T180713Z/`
(`summary.json`, `flips.csv`). Command:
`uv run python scripts/handle_follow_override_distance_eval.py`.

**H1 reproduced exactly before anything was split**: 73 fires, 37 flips,
+0.38461538461538464 accuracy points, identical to lane S's stored
`result.json`. The run aborts otherwise.

Median override distance across H1's own flips: **0.0261**. The split is 19
shallow / 18 deep.

**Decision, revised 2026-09-14: no gate is served.** G1 was briefly shipped
as `HANDLE_FOLLOW_MAX_OVERRIDE_DISTANCE`; that constant was removed the same
day because it was fitted and scored on the same 37 flips (see the retraction
under the records table). The handle rule itself is off the served card
(`HANDLE_FOLLOW_SERVED = False`) and keeps recording as a challenger.

| Arm | Flips | Accuracy | Effect vs G0 | 95% week-blocked | `probability_positive` | Season-blocked P+ |
|---|---:|---:|---:|---|---:|---:|
| **G0 incumbent (all H1 flips)** | 37 | 57.69% | — | — | — | — |
| **G1 shallow overrides only** | 19 | **59.23%** | **+1.54** | [-1.59, +4.68] | **0.834** | 0.811 |
| G2 deep overrides only | 18 | 55.77% | -1.92 | [-4.91, +0.78] | 0.091 | 0.234 |
| Positive control (foresight) | 19 | 64.62% | +6.92 | [+3.92, +10.15] | 1.000 | 0.992 |

Against the untouched card rather than against G0: G1 is +1.92 [-0.78, +4.91],
P+ 0.909; G2 is -1.54 [-4.68, +1.59], P+ 0.167.

### The records, which are the whole evidence base

| Flip group | n | Card's record on those | Following the handle |
|---|---:|---:|---:|
| All flips (G0) | 37 | 18-19 | 19-18 |
| **Shallow (G1)** | 19 | 7-12 | **12-7** |
| **Deep (G2)** | 18 | 11-7 | **7-11** |

**Retracted 2026-09-14, later the same day (audit, measured).** The earlier
text here called the shallow/deep contrast "two populations with opposite
signs, not noise". It is noise-compatible: partitioning the same 37 flips
into a 19/18 split at random, with no reference to override distance, gives
a contrast at least as lopsided as 12-7 versus 7-11 with exact
hypergeometric probability **0.194**, about one time in five. The split
point (0.0261) was the in-sample median of those same 37 flips, so the gate
was chosen on the flips it was scored on. The shallow-versus-deep records
are kept below as a description; they support no gate.

Distance ranges: shallow [-0.100, +0.026], deep [+0.028, +0.139].

### Week 1, the game that occasioned this

ARI at LAC sat at override distance **+0.1379** — the top of the deep range
(max 0.1394). Verified against the recorded revision rows, the gate stops that
flip and leaves the other three of that pass alone:

| game | model's prob for its own side | override distance | under the gate |
|---|---:|---:|---|
| ARI at LAC | 0.638 | +0.1379 | **gated, stays ARI** |
| CHI at CAR | 0.512 | +0.0121 | still fires |
| DAL at NYG | 0.507 | +0.0073 | still fires |
| DEN at KC | 0.509 | +0.0091 | still fires |

### Registry

Family `handle_follow_override_distance_v1`, all `unresolved_below_power`:
`handle_follow_shallow_override_only` (+1.5385, P+ 0.8335),
`handle_follow_deep_override_only` (-1.9231, P+ 0.0914),
`handle_follow_override_foresight_control` (+6.9231, P+ 1.0).

**Nothing here closes H1.** G2's week-blocked interval reaches +0.78, above
zero, so the sign is not resolved and `wrong_sign_resolved` is inadmissible;
the foresight control resolves at 6.9 points and bounds nothing at this scale.
The rule stays served, gated.

### What was shipped

Nothing, as of the same-day revision. The distance constant that was shipped
first was removed from `pick_refresh.py` because it was an in-sample split
of 37 flips. Any future gate on this rule is chosen on seasons it is not
scored on and served only as a paired challenger first.
