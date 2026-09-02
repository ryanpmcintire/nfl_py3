# PER-13 Stage 2: the durability prior, stacked on PRODUCTION — predeclaration

Work package WP26, ROADMAP row **PER-13**, **Stage 2**. Stage 1 lives in
`docs/per13_durability_prior.md` and spent no ATS window. This document spends
one, and it is written **before any ATS accuracy, cover-rate or
`probability_positive` number produced by this comparison exists**. Sections
1–6 are the predeclaration; **§7 was added after the look** and changes nothing
above it.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". A promotion threshold governs only what the docs may
CLAIM; it never governs which card is PLAYED, which is expected value.

## 1. The question, and why it is a REPLACEMENT rather than an addition

Stage 1 met its frozen EV gate: six per-player durability columns improved
out-of-season availability Brier from **0.09087474 to 0.08332335**, a paired
improvement of **+0.0075514**, week-blocked 95% [+0.0067549, +0.0083081],
`probability_positive` **1.000** on 52,382 player-games over 2015–2024, with a
placebo at **−0.0000871** and a prior-seasons-only variant retaining 51% of the
effect (**read**, `docs/per13_durability_prior.md` §9.2, §9.5). Its §9.6
recommendation, verbatim: "the recommended Stage 2 is an ATS on-production test
on a rotation-assigned window, measured on top of what is actually PLAYED".
This document is that test.

Stage 1 improved a *probability*, not a feature. The production ATS chain never
consumes that probability directly — it consumes nine game-level injury columns
that are each built by multiplying a per-player severity by a role share and
summing over a team's visible injury report. That severity **is** the
availability model's output: `players.py::_injury_unavailability` (**read**,
`src/nfl_ats/players.py`:691–695) returns the learned cell probability where the
season-lagged lookup has one and the hand-authored `fixed_unavailability` prior
otherwise, and every one of the nine columns multiplies exactly that number
(**read**, `src/nfl_ats/players.py`:833 for the seven unavailability columns and
:948 for the two value-lost columns).

So the honest Stage 2 candidate does not *add* a column. It **swaps the
severity input** and rebuilds the same nine columns with the same aggregation
code. Adding the durability-augmented columns *alongside* the production ones
would put two near-collinear measurements of one construct into a ridge and
answer a different question — "does the residual between two availability
models help?" — instead of the question PER-13 asks, which is "is the better
P(plays) better?". The replacement also keeps the comparison at one degree of
freedom: the two arms carry the **same number of columns with the same names'
semantics**, so nothing here can be credited to the candidate merely having a
wider design matrix, which is the artifact Stage 1's own placebo was built to
rule out.

## 2. The disclosed prior, in full, before the look

**The measured conversion rate from availability Brier to ATS points is poor,
and it is the most important number in this section.** PER-11's parent
improvement was **0.00444** on this exact metric and bought **+0.10 ATS
points**, week-blocked [−0.63, +0.78] (**read**, `docs/data_feasibility.md`:213–216
and `docs/per13_durability_prior.md` §9.6 item 1). Naively scaling this
candidate's 0.00755 lands near **a fifth of an accuracy point** — an order of
magnitude below this evaluator's ~2-point resolution. Stage 1's own write-up
says so out loud and adds: "Stage 2 is worth its window because the pool is
forced picks and EV is EV, **not** because a visible ATS gain should be
expected. Anyone quoting this result as an edge is misquoting it."

That prior is not a reason to decline the window — declining a candidate whose
intermediate mechanism is measured at P+ 1.000 is taking the far side of a bet
the evidence does not support — but it **is** a reason to freeze, here and now,
that a small or negative point estimate at this sample size is the expected
shape and settles nothing. Two sibling on-production looks run under the same
template this month came back **−0.935** and **−0.668** accuracy points with
week-blocked P+ 0.122 and 0.189 (**read**, `docs/graph_team_stat_on_production.md`
§7 and `docs/graph_team_stat_def_ypp_on_production.md` §7); both were recorded
`unresolved_below_power` and both families stayed open. The same discipline
applies here regardless of sign.

The project's standing lesson, restated because it is the reason this test
exists at all: **composition is not the signal.** A component that improves an
intermediate target on its own can go flat or negative once stacked on the
chain that is actually PLAYED, because that chain already explains some of the
same variance. PER-09, PER-12 and the participation-RAPM candidate all died at
exactly this step (**read**, `docs/per13_durability_prior.md` §9.6 item 2).

## 3. The augmented P(plays): frozen definition

Let *i* be one visible injury row at a game's decision time (kickoff − 24h, the
cutoff production already applies).

**Base probability `p_base(i)`.** Exactly what production computes today:
`nfl_ats.players._injury_unavailability(i)`. Unmodified, unrecalibrated, and
the reference point for everything below. This is deliberate — production's own
number is what the candidate must beat, not a re-fitted stand-in for it.

**Durability columns `x(i)`.** The six columns frozen in
`docs/per13_durability_prior.md` §4 and implemented in
`src/nfl_ats/durability_prior.py`, unchanged and not refit: `durability_residual`,
`durability_listed_active_residual`, `durability_rate_logit_offset`,
`durability_log_observations`, `roster_absence_rate_logit_offset`,
`roster_reserve_rate_logit_offset`. Their point-in-time contract is inherited
verbatim: outcome history is bounded by **kickoff** against the row's own
decision cutoff, roster history by strictly earlier `(season, week)`, and every
shrinkage strength is re-derived per fold from that fold's strictly-prior
seasons. **All six are exactly 0.0 when the player has no prior history**, which
is the property §5's identity check and §6's disclosure both rest on.

**Coefficients, walk-forward by week.** For each `(season, week)` *w*, fit
`StandardScaler` → `sklearn.linear_model.LogisticRegression(C=1.0,
solver="lbfgs", max_iter=1000)` — the same estimator Stage 1 used, at the same
hyperparameters — on the design matrix `[clipped_logit(p_base), x_1..x_6]` over
**every availability-outcome row whose game kickoff is strictly earlier than
w's earliest decision cutoff**, with target `unavailable`. A fit requires at
least `MIN_TRAIN_ROWS = 2,000` labelled rows (Stage 1's own constant);
otherwise no model exists for that week and the offset below is exactly 0.

The fold is by **week** rather than by season — the one deliberate departure
from Stage 1's fold shape, declared here with its reason before the window is
drawn. Two reasons, neither of them about outcomes: (a) production's own ATS
chain refits weekly and forward-chains on kickoff, so a weekly-expanding
availability fit matches the cadence of the thing being stacked on; (b) a
season fold would leave the *first* season that carries any participation label
at all with no fittable model, which needlessly deletes the earliest support the
rotation registry is able to assign. The fold is strictly point-in-time either
way: every training row's kickoff precedes the scored week's decision time.

**The offset.** From the fitted model, take only the six durability
coefficients and unstandardise them, `b_j = β_j / scale_j`, then

```
δ(i) = Σ_j b_j · x_j(i)
```

The refit `base_logit` coefficient and the intercept are **discarded on
purpose**. Keeping them would recalibrate production's own probability, and the
recurrence-hazard sibling already measured that recalibration accounts for
essentially all of a naive comparison's apparent gain (**read**,
`docs/recurrence_hazard_features.md`:102–104). Discarding them isolates the
*durability information* — the only thing PER-13 claims — and it is what makes
the identity property below exact rather than approximate.

**The augmented probability, as an odds-ratio update.**

```
p_dur(i) = p_base(i)·e^δ / ( p_base(i)·e^δ + 1 − p_base(i) )
```

Three properties, all of them intentional and all of them tested:

1. `δ = 0` ⟹ `p_dur = p_base` **exactly** (no clipping, no drift).
2. `p_base ∈ {0, 1}` ⟹ `p_dur = p_base`: a player the report says is Out stays
   out, and a player with a clean full practice stays at production's 0.0. The
   durability prior may not overrule the designation's endpoints.
3. `p_dur ∈ (0, 1)` always, so no downstream column can go out of range.

## 4. The nine swapped columns, and the candidate table

These are the nine — and the only nine — production `weak_stack` feature
columns derived from the availability model (**read**,
`src/nfl_ats/constants.py`:230–238 and :250–253 for the metric names, :581 and
:583 for their entry into `FEATURE_FAMILIES`, and :785–788 for
`FEATURE_SETS["football_weak_stack"]` / `["full_weak_stack"]`):

| # | production column | family |
| --- | --- | --- |
| 1 | `diff_injury_offense_unavailability` | `player_injuries` |
| 2 | `diff_injury_defense_unavailability` | `player_injuries` |
| 3 | `diff_injury_special_teams_unavailability` | `player_injuries` |
| 4 | `diff_injury_offensive_line_unavailability` | `player_injuries` |
| 5 | `diff_injury_skill_unavailability` | `player_injuries` |
| 6 | `diff_injury_front_unavailability` | `player_injuries` |
| 7 | `diff_injury_secondary_unavailability` | `player_injuries` |
| 8 | `diff_injury_skill_epa_value_lost` | `player_values` |
| 9 | `diff_injury_defense_disruption_value_lost` | `player_values` |

Every other injury-adjacent column in the table (`player_continuity`,
`player_qb`) is built from rosters, snaps and quarterback history and never
touches `_injury_unavailability`; those are left alone.

**The candidate columns** carry the same names with a `_durability` suffix and
are built by re-running the *same* enrichment code — `players.py::
enrich_with_player_features`, at the production builder's own defaults and
pinned to the production table's own snapshots — twice: once with `δ ≡ 0` and
once with the real `δ`. The candidate column is then

```
<name>_durability = <production column> + ( rebuilt_δ − rebuilt_δ≡0 )
```

with a missing difference treated as 0. This construction is chosen over
"just use the rebuilt column" for one reason: it is **additive against
production by construction**, so every production column in the candidate table
comes back bit-identical and the candidate differs from production by exactly
the durability transport and nothing else — no library-version drift, no
snapshot drift, no re-derivation error. As a separate check the `δ ≡ 0` rebuild
is compared against the production table's nine columns and the reproduction
result is reported in §7 either way.

The candidate table is
`data/processed/game_features_weak_stack_durability.parquet`: the PRODUCTION
`data/processed/game_features_weak_stack.parquet` plus exactly these nine new
columns, merged additively on `game_id`. Built on the PRODUCTION table directly,
never on `weak_stack_v3`/`_surface`/`_v4`/`_graph_*`/`_fluview`/`_illness` —
the same reason every sibling states: stacking onto a profile already refused or
still undecided would confound the answer to "does this add to what is actually
played."

## 5. The candidate profile: `weak_stack_durability`

A new `MarginFeatureProfile`, `weak_stack_durability` = production
`weak_stack`'s exact feature set (`FEATURE_SETS["football_weak_stack"]` /
`["full_weak_stack"]`) with the nine columns of §4 **removed and their
`_durability` twins put in their place**. Same column count, same construct,
different P(plays). Never referenced by the active model. Never mixed with any
other candidate profile.

**The identity property, frozen as a test.** In any game where every visible
injured player has `δ = 0` — no prior history, or no fittable model for that
week — the nine candidate columns equal production's bit-identically and the
two arms are the *same model*. `tests/test_per13_durability_production_feature.py`
pins this, along with the leakage regression AGENTS.md requires for every new
feature family: a future season's outcomes, arbitrarily flipped, may not change
any earlier game's candidate column.

## 6. The comparison, grade, window and rotation family

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_durability` | production with the durability P(plays) |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values (**read**,
`artifacts/active_ats_model.json`), fit with `nfl_ats.margin.fit_margin_model`
— the full production profile, not a single-feature model. Only the nine
columns' contents differ. Forward-chaining only: each week is predicted from a
model trained strictly on games that kicked off before that week's earliest
kickoff.

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta in `accuracy_points`, `pick_correct` against
`home_cover_probability >= 0.5` — the same probability rule production plays.

**Grade.** Close-graded. This is a screen, not a play/no-play decision. Per the
binding "grade the decision at the opener" rule, nothing here may settle a
play/no-play or promotion call, and a terminal classification
(`refuted_mechanism` / `bounded_by_control`) is reserved for an opener-graded
look. Every verdict from this document is `unresolved_below_power` regardless
of sign.

**Family and window.** A new rotation family, `per13_durability_on_production`,
close-graded, declared with **no inheritance** and **without
`--acknowledge-mined`**. No inheritance because PER-13 Stage 1 spent no ATS
window and declared no rotation family — there is nothing to inherit, and
naming an unrelated family as a parent to move the window would be a fiction.
`nfl-ats rotation assign` therefore hands this family the **earliest eligible
close-graded block at or after the warm-up floor**; the block is reported in §7
from the CLI's own output and is never hand-picked.

**How an early block is handled, disclosed before it is drawn.** The
durability columns rest on participation labels, and the snap-count source
begins in **2013** (**read**, `docs/per13_durability_prior.md` §3: `snap_counts.parquet`
covers 2013–2025). Roster-status history reaches back to 2009, so column 6
(`roster_reserve_rate_logit_offset`) can be non-zero earlier than the rest, but
the odds offset of §3 additionally needs a fittable prior pool of ≥2,000
labelled rows, which cannot exist before participation labels do. The
consequence, stated plainly: **in seasons before the columns' data support, all
six columns are 0 for every player, `δ = 0`, and the candidate is bit-identical
to production.** That is a design consequence of the feature family's data
reach meeting the registry's warm-up floor, not a defect and not something to
paper over. §7 reports (a) the share of scored rows per season with any
non-zero durability history and (b) the share of games in the window where the
two arms' picks actually differ. **If those shares are zero across the assigned
block, the look cannot separate the arms and says nothing whatever about the
mechanism** — it is recorded `unresolved_below_power` with that reason stated in
full, never as a closure, and the family stays open for its next window. Under
the taxonomy above, "the instrument could not separate the arms" is not one of
the two admissible closing grounds.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary (within-week game correlation is zero by owner mandate), season-blocked
reported beside it and never averaged with it. Same `BOOTSTRAP_SAMPLES=1000` /
`SEED=20260826` constants the sibling on-production scripts use, for
comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to the
sibling documents: both arms' models are fit ONCE on the REAL `ats_margin`, and
only the grading margin is shuffled within week, so 200 draws cost no extra
fits. This null is **not** centred on zero by design (it preserves each week's
realised home-cover rate and the arms may carry different home-pick rates) and
is reported alongside the bootstrap-vs-zero interval, never instead of it.

**Positive control, run BEFORE the real screen.** One swapped column,
`diff_injury_offense_unavailability_durability`, is temporarily replaced by the
realised `ats_margin` — a deliberate, large leak — so the harness must show an
obvious, large effect. This proves the FULL-PROFILE ridge fit can detect a real
effect of meaningful size with the candidate column embedded in ~270 other
production features. A "no effect" reading from a blind instrument would mean
nothing; this check exists so that possibility is ruled out first, and it is
also what would later make `bounded_by_control` an available closing ground.

**Order of operations, frozen:** null → positive control → screen, each run
exactly once.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold — `probability_positive`
above 0.5 favours playing the candidate over the baseline, and predeclared
thresholds govern only what a doc may CLAIM. This run is close-graded, so it
settles no play/no-play decision by itself; what it DOES settle is whether the
Stage 1 finding, measured the honest way, still looks worth an eventual
opener-graded confirmation look.

## Recording

One `nfl-ats weak-signals record` entry, `--effect-units accuracy_points`,
family **`per13_durability_on_production`** — a different pooling bucket from
Stage 1's `per13_durability_prior_stage1`, stated explicitly because AGENTS.md's
commensurability rule forbids pooling a Brier improvement on 52,382
player-games with an accuracy-point delta on a few hundred games.
Classification **`unresolved_below_power`** at this close grade regardless of
sign; a resolved wrong sign here is reported as continuous evidence, never as a
`refuted_mechanism` closure, because that closure is reserved for the opener
grade. `--reliability` carries Stage 1's measured residual-trait split-half
figure, 0.793.

One `nfl-ats rotation record --name per13_durability_on_production --verdict
unresolved` call spends the assigned window, carrying the same paired effect,
interval and `probability_positive`.

## 7. Results (added after the look, 2026-09-01)

Nothing above this line was changed. Every number below is **measured this
session**; artifacts:

- candidate table build: `artifacts/per13_durability_on_production/20260901T194607Z/build.json`
- null check: `artifacts/per13_durability_on_production/20260901T195229Z/results.json`
- positive control: `artifacts/per13_durability_on_production/20260901T195246Z/results.json`
- screen: `artifacts/per13_durability_on_production/20260901T195302Z/results.json`

### 7.1 The assigned window

Declared with no `--inherits` and no `--acknowledge-mined`, then assigned. The
CLI's own output:

```
"assigned": "per13_durability_on_production",
"grade": "close", "inherits": [], "acknowledges_mined_2018_2025": false,
"windows": [{"seasons": [2011, 2013], "state": "assigned",
             "window_kind": "contiguous", "assigned_at": "2026-09-01"}],
"remaining_eligible_windows": 2
```

**[2011, 2013]** — the earliest eligible close-graded block at the warm-up floor
`MIN_ELIGIBLE_START_SEASON = 2011` (**measured**,
`nfl_ats.rotation.MIN_ELIGIBLE_START_SEASON`). Never hand-picked: a fresh family
with no inheritance has touched nothing, so `assign_window` returns the earliest
block by construction. The two blocks still eligible afterwards are
**[2014, 2016]** and **[2015, 2017]** (**measured**,
`rotation.eligible_blocks`).

### 7.2 The candidate table, and whether it reproduces production

The rebuild ran twice on the production table's own pinned snapshots (player
`20260812T200527Z`, player-value `20260813T121050Z`, PBP `20260812T142851Z`,
features `data/processed/game_features_pbp.parquet`), 167.3 s and 167.0 s.

**The `δ ≡ 0` rebuild reproduces production's nine columns bit-identically on
all 4,902 rows** — maximum absolute difference **0.0** on every one of the nine
(`build.json` → `reproduction`). So the candidate table differs from production
by the durability transport and by nothing else: no snapshot drift, no library
drift, no re-derivation error. The additive-merge check passes too: every one of
production's 275 columns comes back bit-identical and exactly nine are added.

The offset panel: **62,206 training rows, 76,763 target player-games, 60,207
non-zero offsets, 201 weeks fitted, 13.6 s.** Across the whole table, 3,017 of
4,902 games carry at least one moved column.

### 7.3 Data support, per season (the §6 disclosure, measured)

| season | injury player-games | share with non-zero durability history | share with a non-zero offset δ |
| --- | --- | --- | --- |
| 2009 | 17 | 100.0% | 0.0% |
| 2010 | 4,262 | 99.4% | 0.0% |
| 2011 | 4,748 | 99.1% | **0.0%** |
| 2012 | 5,266 | 99.4% | **0.0%** |
| 2013 | 4,913 | 99.0% | **58.8%** |
| 2014 | 4,912 | 99.0% | 99.3% |
| 2015 | 5,009 | 99.4% | 99.4% |
| 2016 | 4,928 | 99.2% | 99.2% |
| 2017 | 4,948 | 99.3% | 99.6% |
| 2018 | 4,961 | 99.4% | 99.8% |
| 2019 | 5,202 | 99.5% | 99.7% |
| 2020 | 5,414 | 99.3% | 99.9% |
| 2021 | 5,348 | 99.6% | 99.8% |
| 2022 | 5,432 | 99.6% | 99.7% |
| 2023 | 5,451 | 99.6% | 99.7% |
| 2024 | 5,952 | 99.5% | 99.5% |

The two columns diverge exactly where §6 said they would. Roster-status history
reaches back to 2009, so the durability columns themselves are non-zero for
~99% of rows from 2010 on; but the **offset** additionally needs a fitted model,
and participation labels — hence training rows — begin in 2013. The first week
with ≥2,000 strictly-prior labelled rows lands part-way through 2013, which is
why 2013 comes in at 58.8% and 2011–2012 at exactly zero.

Inside the assigned window that means:

| season | window games | share with a moved column |
| --- | --- | --- |
| 2011 | 256 | **0.0%** |
| 2012 | 256 | **0.0%** |
| 2013 | 256 | **58.2%** |

**149 of 768 window games (19.4%)** can separate the arms at all. On the other
619 the two arms are literally the same model, and the screen confirms it:
**83.1% of graded games get bit-identical home-cover probabilities from the two
arms.** The picks themselves differ on **1.3%** — 10 of 746 graded games.

### 7.4 Instrument checks

**Null check** (`--mode null`, 200 within-week permutations, real candidate
feature, not leaked): mean **−0.190** accuracy points, sd 0.390, 95%
[−0.804, +0.536], observed +0.536. A sane, finite distribution — the harness
produces a null, not a crash or a degenerate spike. It is not centred on zero,
exactly as §6 said it would not be, and its negative centre is consistent with
the two arms carrying slightly different home-pick rates (53.6% vs 54.2%).

**Positive control** (`--mode positive-control`,
`diff_injury_offense_unavailability_durability` replaced by the realised
`ats_margin`): candidate accuracy **1.000** against the baseline's 0.4987, paired
delta **+50.134** accuracy points, week-blocked P+ **1.000**, 95%
[+46.428, +53.652], at the 100.0th percentile of its own null (which centres at
+2.713 under the leak treatment). **The full-profile ridge fit is not blind**: it
detects a real effect of meaningful size in exactly the column the candidate
modifies, with ~270 other production features present. A null reading from this
instrument would therefore have been informative rather than vacuous.

### 7.5 The screen

`--mode screen`, artifact
`artifacts/per13_durability_on_production/20260901T195302Z/results.json`.

| quantity | value |
| --- | --- |
| baseline (`weak_stack`) accuracy | 0.49865951742627346 |
| candidate (`weak_stack_durability`) accuracy | 0.5040214477211796 |
| paired delta | **+0.536 accuracy points** |
| week-blocked 95% CI | **[−0.264, +1.363]** |
| week-blocked `probability_positive` | **0.872** |
| season-blocked 95% CI | [+0.000, +1.600] |
| season-blocked `probability_positive` | 0.672 |
| n games / weeks / seasons | 746 / 51 / 3 |
| home-pick rate | baseline 53.6%, candidate 54.2% |
| observed vs its own permutation null | **93.0th percentile** (null mean −0.190) |

Both arms carry nearly the same home-pick rate, so — like the two on-production
siblings and unlike the bare-baseline 38-family screen — this measurement
carries very little of the home-tilt artifact.

**Post-hoc breakdown** (the SAME fitted models, re-graded; not a second look and
not part of the frozen comparison):

| season | graded games | baseline | candidate | delta |
| --- | --- | --- | --- | --- |
| 2011 | 245 | 0.461224 | 0.461224 | **+0.000** |
| 2012 | 251 | 0.513944 | 0.513944 | **+0.000** |
| 2013 | 250 | 0.520000 | 0.536000 | **+1.600** |

The whole effect is 2013's, because 2011 and 2012 are bit-identical arms by
construction. **On the 10 games where the picks differ, the candidate is right
on 7 and the baseline on 3.** That four-game net is the entire +0.536.

### 7.6 What this implies for the decision, before what is wrong with it

**On EV grounds this measurement favours the candidate, and that is the only
decision rule this project uses.** `probability_positive` is **0.872** against
an EV break-even of 0.5. AGENTS.md's own worked example is this number: "the
pool is FORCED PICKS… declining a candidate that is 87% likely better is not
caution — it is taking the other side of an 87/13 bet." The point estimate,
both blockings, and the permutation-null percentile all point the same way, and
the positive control proves the instrument could have seen the opposite. **This
is the first of this session's on-production screens whose expected value points
at the candidate rather than at the status quo** — the two graph-feature
siblings came back at P+ 0.122 and P+ 0.189.

What it does **not** do is promote anything. This run is close-graded, and the
binding rule is that a play/no-play decision is graded at the OPENER; the
predeclaration froze `unresolved_below_power` regardless of sign, and that is
what was recorded. The concrete next step it earns is an **opener-graded
confirmation look on the family's next window, [2014, 2016]** — where §7.3
measures 99.0–99.2% offset coverage instead of this window's 19.4%, so the same
mechanism gets roughly five times the exposure per game scored.

**Now what is wrong with it. Five things, and none of them reverses the
direction.**

1. **The entire effect is four net games out of ten flipped picks, all in one
   season.** 2011 and 2012 contributed exactly zero by construction, so the
   746-game denominator flatters the precision of the +0.536 figure: the honest
   sub-population is 2013's 250 graded games, and inside that the arms disagree
   on 10. Ten flips at 7–3 is, on its own, a two-sided binomial p of 0.34. The
   week-blocked P+ of 0.872 is a statement about the paired difference across
   the whole window, and it is real, but nobody should read "+0.536 points" as
   an effect size that would replicate at that magnitude.
2. **A window was spent at 19.4% feature coverage.** §6 disclosed the risk and
   froze the handling before the block was drawn, which is why this is
   reportable rather than a silent null — but the accounting is that the CLI's
   earliest-eligible rule and this family's late data support are in tension,
   and the support table in §7.3 needed no ATS window to produce. The
   generalisable lesson for the next family whose feature starts late: measure
   per-season support before declaring.
3. **A positive close-graded reading is exactly where this project has been
   burned before.** MOD-07 was refused on a close-graded comparison and then
   promoted at the opener, and the standing correction is that the close and the
   opener disagree systematically. That cuts both ways: a close-graded positive
   is no more a promotion than a close-graded negative is a refutation.
4. **Stage 1's own conversion warning still stands.** PER-11's 0.00444 Brier
   improvement bought +0.10 ATS points; naive scaling of this candidate's
   0.00755 predicts about a fifth of a point. The +0.536 measured here is above
   that prediction, on a window where only a fifth of games could move at all,
   which is more likely small-sample luck than evidence the conversion rate is
   better than PER-11 measured. Freezing that expectation now: a [2014, 2016]
   look should be expected to land nearer +0.2 than +0.5.
5. **The 2013 half-window that the 2,000-row training floor cost is real.**
   99.0% of 2013's rows carry non-zero durability columns; only 41.2% lack the
   fitted offset. A lower floor would have given 2013 full coverage. That choice
   was frozen before the block was drawn and is not revisited post hoc, but it
   is the one design decision that measurably cost coverage here.

### 7.7 Registry

Recorded as `per13_durability_on_production_ats`, family
`per13_durability_on_production`, league nfl, seasons 2011–2013,
`--effect-units accuracy_points`, effect **+0.5361930294906166**, interval
[−0.26390487284567443, +1.3627731363673892], `probability_positive` **0.872**,
reliability **0.793** (Stage 1's measured residual-trait split-half figure), 746
sample games, 51 blocks, category `health`, classification
**`unresolved_below_power`**.

That classification is the predeclared one and it is also the only admissible
one. `refuted_mechanism` is out twice over: the sign is POSITIVE, so
`wrong_sign_resolved` cannot apply, and the trait's split-half reliability is
0.793, not zero. `bounded_by_control` is out because the control DETECTED its
planted effect (+50.134 points, P+ 1.000) rather than failing to; it proved the
instrument responsive, which is the opposite of bounding an absent effect. The
registry row must **not** be pooled with `per13_durability_prior_availability_brier`:
that entry is a Brier improvement on 52,382 player-games and this one is an
accuracy-point delta on 746 games, which AGENTS.md's commensurability rule
forbids combining.

One `nfl-ats rotation record --name per13_durability_on_production --verdict
unresolved` call spent the [2011, 2013] window, carrying the same effect,
interval and `probability_positive`. The family's status stays **open** with
[2014, 2016] and [2015, 2017] still eligible.
