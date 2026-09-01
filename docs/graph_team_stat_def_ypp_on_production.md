# Graph `team_stat` def_yards_per_play, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the
same rule that governs `docs/graph_ratings_v2_screen.md`,
`docs/graph_ratings_v2.md`, and its own sibling document,
`docs/graph_team_stat_on_production.md` (the `off_sack_rate` version of
exactly this experiment, completed earlier today). **Sections 1-6 are the
predeclaration** and contain no accuracy, cover-rate, or `probability_positive`
number against NFL outcomes. **Section 7 was added after the look** and
reports what it found; it changes nothing above it.

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

## 1. What this closes, why it is not a repeat of the sack-rate look, and why it is not the same question as section 8 of the screen

`docs/graph_ratings_v2_screen.md` §4 declared the comparator of first resort
as the raw `home - away` differential, and its §8 measured all 38 cluster
representatives against that bare comparator on a 2011-2013 window,
close-graded. It also recorded the project's standing lesson, restated in
ROADMAP.md and AGENTS.md: **"composition is not the signal"** — an overlay or
feature positive alone can go negative once stacked on the chain that is
actually PLAYED, because the played chain already explains some of the
variance a bare-baseline comparison credits to the candidate. The marginal
that decides is the one measured on top of what is played.

`docs/graph_team_stat_on_production.md` ran exactly that stacked measurement
for `off_sack_rate` — the family that led the screen by the NAIVE, zero
-reference reading (+2.949 points, P+ 0.987) — and found it went negative on
production (-0.935 pts, week-blocked P+ 0.122, [2014, 2016]). This document
declares the analogous stacked measurement for a *different* family,
`def_yards_per_play` — the family that leads the screen by the CONSERVATIVE,
null-adjusted reference instead (95.5th percentile of its own within-week
permutation null, +2.145 points against a null centred at only +0.279 —
**read**, `docs/graph_ratings_v2_screen.md` §8, "The three readings worth
carrying forward": "`def_yards_per_play` leads by the conservative reference
... despite ranking second against zero. This is the ordering the null
reference changes."). Per `docs/pool_edge_plan.md`'s 2026-08-31 ranked agenda
(item 1): this is a **distinct bet, not a repeat of the one that just lost**
— `off_sack_rate` was the family whose apparent edge the null reference
showed was ~40% artifact (92.5th percentile against zero's own reading vs its
raw +2.949); `def_yards_per_play` is specifically the cell the null reference
judges *least* artifact-contaminated of the three the doc names, which is
exactly the property that should matter once the comparison moves from "beats
a bare baseline" to "beats production".

It answers a different question than §8 did: does the graph-propagated
`def_yards_per_play` signal add anything on top of the full PRODUCTION
`weak_stack`/`market_residual`/ridge/alpha-10 chain (**read**,
`artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`), rather
than on top of a zero-feature market baseline?

## 2. The candidate feature, unchanged from the screen

`graph_v2_team_stat_def_yards_per_play_katz_diff`: the `team_stat` arm of
`src/nfl_ats/graph_ratings_v2.py` (`edge_signal="team_stat"`,
`signal_column="def_yards_per_play"`), at the SAME structural configuration
frozen in `docs/graph_ratings_v2_screen.md` §5 and inherited, not refit, here
— identical to the `off_sack_rate` sibling document's own inheritance:

```
alpha=0.85, half_life_weeks=8.0, max_row_l1=1.0, prior_weight=1.0,
min_games=16, propagation="signed_katz", injury_beta=0.0
```

No new structural choice is made in this document. `def_yards_per_play` is a
`STATE_METRICS` family (`src/nfl_ats/constants.py`), carried in the feature
table under the standard prefix convention (`home_def_yards_per_play`,
`away_def_yards_per_play`) exactly like `off_sack_rate` — not one of the five
suffix-form families `docs/graph_ratings_v2_screen.md` §1 names, so no
`signal_column_pair` override is needed, matching the sibling module's own
default-prefix construction. The feature is computed once, leak-safe exactly
as proven in `tests/test_graph_ratings_v2.py` (ratings for week `w` read the
graph through week `w-1` only), over the full production `weak_stack` feature
table's completed games, then additively joined back onto that table by
`game_id` — the same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features` and the
`off_sack_rate` sibling module already established for `weak_stack_v4` and
`weak_stack_graph_sack` respectively.

## 3. The candidate profile: `weak_stack_graph_def_ypp`

A new `MarginFeatureProfile`, `weak_stack_graph_def_ypp` = production
`weak_stack`'s exact feature set (`FEATURE_SETS["football_weak_stack"]` /
`FEATURE_SETS["full_weak_stack"]`) plus exactly one new column,
`graph_v2_team_stat_def_yards_per_play_katz_diff`. Built on
`data/processed/game_features_weak_stack.parquet` (the PRODUCTION table)
directly, never on `weak_stack_v3`/`_surface`/`_v4`/`_graph_sack` — mirroring
`weak_stack_graph_sack`'s own declared reason verbatim: stacking a candidate
onto a profile already refused or still undecided would confound the answer
to "does this add to what is actually played." Never referenced by the active
model. Never mixed with any other candidate profile.

## 4. The comparison

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_graph_def_ypp` | production + the one graph feature |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`, `target="market_residual"`
fixed at the active model's own values (`artifacts/active_ats_model.json`) —
only `feature_profile` differs, isolating the graph column's marginal
contribution against everything the production chain already explains. Both
are fit with `nfl_ats.margin.fit_margin_model`, the same estimator production
itself uses, not a single-feature model — this is the whole point of
"on top of production" rather than "on top of a bare baseline."

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring `docs/graph_ratings_v2_screen.md` §6 and
the `off_sack_rate` sibling document §5 in full: this is a screen, not a
play/no-play decision. Per the binding "grade the decision at the opener"
rule, nothing here may settle a play/no-play or promotion call; a terminal
classification is reserved for an opener-graded look, exactly as
`scripts/graph_team_stat_record.py`'s `classify()` already enforces for the
sibling screen.

**Window and family.** `graph_ratings_v2_team_stat` (the 38-family screen's
own rotation family) has already spent its first window ([2011, 2013],
verdict `unresolved`) on the bare-baseline comparator, and the rotation
registry hard-refuses a family re-looking at seasons it or its inheritance
chain has already touched (`nfl_ats/rotation.py::_validate`, the "no re-look"
invariant). Its remaining ~2 eligible windows are reserved for that family's
OWN predeclared bare-baseline question (`docs/graph_ratings_v2_screen.md` §8's
"three readings worth carrying forward"), not spent here on a different
comparator.

This document therefore declares a **new, narrowly-scoped rotation family**,
`graph_def_ypp_on_production`, close-graded, declared with
`inherits=("graph_ratings_v2_team_stat",)` — the SAME single-parent lineage
the `off_sack_rate` sibling family (`graph_off_sack_rate_on_production`)
declared, not a lineage naming that sibling itself. This is disclosed rather
than glossed over: `graph_off_sack_rate_on_production` inherits from
`graph_ratings_v2_team_stat` too and has, as of today, already spent a
[2014, 2016] window under that lineage — but per `nfl_ats/rotation.py`'s own
documented design (`_inherited_names` walks the declared `inherits` edges
upward only; a family is never held responsible for what an unrelated sibling
that shares the same parent has drawn, "rule 4 makes windows retire
per-family, so independent families are explicitly allowed to draw the same
seasons"), this family's touched-season set is computed from
`graph_ratings_v2_team_stat`'s OWN windows, not from every family that
inherits it. So `rotation assign` may legally hand this family the SAME
[2014, 2016] block the sack-rate sibling just used, or a different one — the
CLI computes the earliest eligible block at declaration time, and the actual
assigned block is confirmed in section 7, not asserted here. A same-window
outcome would not be a defect: the original 38-family screen itself scored
`off_sack_rate` and `def_yards_per_play` on the identical [2011, 2013] window
as one look, so two different graph-feature candidates measured on the same
window is the established, comparable shape for this line of work, not a
re-look at a question already answered.

## 6. Uncertainty and instrument checks, reusing the screen's own design unchanged

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read, never averaged with it. Same tool, same
`BOOTSTRAP_SAMPLES=1000`/`SEED=20260826` constants
`scripts/graph_team_stat_screen.py` already uses, for comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to
`docs/graph_ratings_v2_screen.md` §6 and the `off_sack_rate` sibling document
§6: both arms' models are fit ONCE per week on the REAL `ats_margin`; only the
grading margin is shuffled within week for the null, so 200 draws costs no
extra model fits. This null is **not** centred on zero by design (it preserves
each week's realized home-cover rate, and the two arms may carry different
home-pick rates), and is reported ALONGSIDE the bootstrap-vs-zero interval,
never instead of it.

**Positive control**, run BEFORE the real screen: the candidate profile's one
new column (`graph_v2_team_stat_def_yards_per_play_katz_diff`) is temporarily
REPLACED by the realized `ats_margin` — a deliberate, large leak — so the
harness must show an obvious, large effect. This proves the FULL-PROFILE
ridge fit (not just a single-feature model, per the screen's own instrument
check) can detect a real effect of meaningful size when one is actually
present. A "no effect" reading from a blind instrument would mean nothing;
this check exists so that possibility is ruled out first.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run
is close-graded, so it settles no play/no-play decision by itself; what it
DOES settle is whether the family's screen-stage finding, measured the honest
way (stacked on what is actually played, not a bare baseline), still looks
worth an eventual opener-graded confirmation look — mirroring
`docs/graph_ratings_v2_screen.md` §8's and the `off_sack_rate` sibling
document's "what this implies for the decision" framing exactly.

## Recording

One `nfl-ats weak-signals record` entry, `effect_units=accuracy_points`,
family `graph_def_ypp_on_production` (a DIFFERENT pooling bucket from both
`graph_ratings_v2_team_stat` and `graph_off_sack_rate_on_production`, stated
explicitly: all three measure graph-propagated `team_stat` constructs but
against non-commensurable comparators or different underlying families, and
AGENTS.md's commensurability rule forbids pooling them together).
`unresolved_below_power` at this close grade regardless of sign, exactly as
`scripts/graph_team_stat_record.py::classify` already enforces for the
sibling screen — a resolved wrong sign at this grade is reported as
continuous evidence, never as a `refuted_mechanism` closure, because that
closure is reserved for the opener grade.

One `nfl-ats rotation record --name graph_def_ypp_on_production
--verdict unresolved` call spends the assigned window, carrying the same
paired effect, interval, and `probability_positive`.

## 7. Results (added after the look, 2026-08-31)

Rotation family `graph_def_ypp_on_production` was declared inheriting
`graph_ratings_v2_team_stat`, then assigned by `nfl-ats rotation assign`
(never hand-picked): the earliest eligible block is **[2014, 2016]** (749
games, 51 weeks) — the SAME block the `off_sack_rate` sibling family drew,
exactly as section 5 disclosed was possible (that family's own spent window
is invisible to this family's touched-season computation, since it is a
sibling under `graph_ratings_v2_team_stat`, not an ancestor of this family).
Two instrument checks ran first.

**Null check** (`--mode null`, 200 within-week permutations, real candidate
feature, not leaked): mean **+0.263** accuracy points, sd 0.681, 95%
[-1.068, +1.602], observed -0.668. A sane, finite distribution — the harness
produces a null, not a crash or a degenerate spike. (This differs from the
`off_sack_rate` sibling's own null-check reading, +0.416 pts, because the
candidate model here carries a different feature and therefore a slightly
different home-pick rate, which is exactly what shifts a within-week
permutation null's centre.)

**Positive control** (`--mode positive-control`, the candidate's one column
replaced by the realized `ats_margin`): paired delta **+48.999** accuracy
points, week-blocked P+ **1.000**, 95% [+45.661, +51.921], sitting at the
100.0th percentile of its own null (which itself centres at +2.652 pts under
the leak treatment). The full-profile ridge fit is not blind to a real effect
of meaningful size, even with the graph column embedded in ~270 other
production features.

**The real screen** (`--mode screen`, artifact
`artifacts/graph_team_stat_def_yards_per_play_on_production/20260831T173427Z/results.json`):
production `weak_stack` accuracy on this paired population is 51.00%
(baseline_accuracy 0.5100133511348465), `weak_stack_graph_def_ypp` is 50.33%
(candidate_accuracy 0.5033377837116155). Paired candidate-minus-baseline delta
**-0.668** accuracy points. Week-blocked 95% CI **[-2.243, +0.804]**,
week-blocked `probability_positive` **0.189**. Season-blocked 95% CI
[-1.210, 0.000], season-blocked `probability_positive` 0.000 — reported
alongside the week-blocked read per section 6's own discipline, but read with
the same caution `docs/era_weighting_promotion.md` gives its own
2-season-block secondary: with only 3 season blocks in this window the
season-blocked bootstrap has very little combinatorial diversity, so its
near-zero-touching upper bound is a low-power artifact of block count, not a
sharper answer than the week-blocked primary. Home-pick rate: baseline
37.6%, candidate 36.7% — close together, so (like the `off_sack_rate`
sibling's own reading, and unlike the bare-baseline screen's 55-67% home-pick
arms) this measurement carries very little of the home-tilt artifact that
discounted the earlier screen. Against its own permutation null (mean
+0.263), the observed delta sits at the **8.0th percentile** — in the
negative tail of its own null, though with only 200 draws and no multiplicity
correction across the two graph-column experiments run today, one family
landing in a tail is not itself a surprising event.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the
candidate, the only decision rule this project uses — this measurement's
week-blocked P+ 0.189 does **not** favour adding the graph
`def_yards_per_play` feature on top of what is actually played: both the
point estimate and the probability mass lean toward the status quo
(production `weak_stack` alone) over the candidate. This is **not** a
closure: the week-blocked CI upper bound is positive (+0.804), so the
interval is not resolved entirely on the wrong side of zero, and the positive
control demonstrated the harness CAN see a real effect — it did not
demonstrate that an effect the size of the screen's own conservative-reference
+2.145 reading would have been reliably detected at this window's 749-game
sample (the week-blocked CI here spans about 3.0 points, wider than that
reading), so `bounded_by_control` is not available either. The result is
recorded `unresolved_below_power`, exactly as the predeclared grade
discipline requires, and the family stays **open**.

The honest reading, stated plainly: `def_yards_per_play` was named in
`docs/pool_edge_plan.md`'s ranked agenda specifically because it is the
family the null reference judges LEAST artifact-contaminated of the three the
screen carried forward — the bet that a bare-baseline positive reading here
would be more likely to survive being stacked on production than
`off_sack_rate`'s was. It did not survive either, and it did not survive by a
somewhat larger margin on the week-blocked point estimate (-0.668 pts here
vs -0.935 pts for `off_sack_rate`) while landing at a noticeably worse
percentile against its own null (8.0th here vs 6.0th for `off_sack_rate` --
both negative-tail, comparable). This is the same "composition is not the
signal" pattern recorded before: a component's screen-stage standing (against
a bare baseline, or against its own permutation null) does not predict its
marginal contribution once stacked on a chain that already explains much of
the same variance. Nothing here refutes the graph transform as a mechanism
for `def_yards_per_play` (no resolved wrong sign — the week-blocked interval
still contains positive values — and no split-half-reliability failure was
tested), and nothing here proves it inert (P+ 0.189 is not P+ 0.00, and the
interval carries real positive mass). It is simply unresolved at this sample
size, which per the binding taxonomy is the expected shape for a real small
signal and is never grounds to close the line of work on its own.

Combined with `docs/graph_team_stat_on_production.md`'s own result, TWO of
the screen's three named conservative-reference/naive-reference leaders
(`off_sack_rate`, `def_yards_per_play`) have now both failed to clear EV
against production at this window, close-graded. Per `docs/pool_edge_plan.md`
item 2 in the same ranked agenda, `off_rush_epa_per_play` — the family with
the single highest reliability figure recorded for this construct
(`graph_input_screen_off_rush_epa_per_play`, 0.987) but whose screen-stage
reading the doc itself disclosed as "most of its apparent edge is plausibly
the artifact" — is the next predeclared candidate in this line of work, not
a repeat of either family already measured here.

