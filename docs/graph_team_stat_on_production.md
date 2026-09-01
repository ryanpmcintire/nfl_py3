# Graph `team_stat` off_sack_rate, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/graph_ratings_v2_screen.md` and
`docs/graph_ratings_v2.md`. **Sections 1-6 are the predeclaration** and
contain no accuracy, cover-rate, or `probability_positive` number against NFL
outcomes. **Section 7 was added after the look** and reports what it found;
it changes nothing above it.

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

## 1. What this closes, and why it is not the same question as section 8 of the screen

`docs/graph_ratings_v2_screen.md` §4 declared the comparator of first resort
as the raw `home - away` differential, and its §8 measured all 38 cluster
representatives against that bare comparator on a 2011-2013 window,
close-graded. It also recorded the project's standing lesson, restated in
ROADMAP.md and AGENTS.md: **"composition is not the signal"** — an overlay or
feature positive alone can go negative once stacked on the chain that is
actually PLAYED, because the played chain already explains some of the
variance a bare-baseline comparison credits to the candidate. The marginal
that decides is the one measured on top of what is played.

This document declares that stacked measurement for exactly one family --
`off_sack_rate`, the best cell of the 38 by the naive zero-reference (+2.949
points, P+ 0.987) and third by the conservative permutation-null reference
(92.5th percentile; **read**, `docs/graph_ratings_v2_screen.md` §8, "The three
readings worth carrying forward") -- and only that one. It answers a
different question than §8 did: does the graph-propagated
`off_sack_rate` signal add anything on top of the full PRODUCTION
`weak_stack`/`market_residual`/ridge/alpha-10 chain (**read**,
`artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`), rather
than on top of a zero-feature market baseline?

## 2. The candidate feature, unchanged from the screen

`graph_v2_team_stat_off_sack_rate_katz_diff`: the `team_stat` arm of
`src/nfl_ats/graph_ratings_v2.py` (`edge_signal="team_stat"`,
`signal_column="off_sack_rate"`), at the SAME structural configuration frozen
in `docs/graph_ratings_v2_screen.md` §5 and inherited, not refit, here:

```
alpha=0.85, half_life_weeks=8.0, max_row_l1=1.0, prior_weight=1.0,
min_games=16, propagation="signed_katz", injury_beta=0.0
```

No new structural choice is made in this document. The feature is computed
once, leak-safe exactly as proven in `tests/test_graph_ratings_v2.py`
(ratings for week `w` read the graph through week `w-1` only), over the full
production `weak_stack` feature table's completed games, then additively
joined back onto that table by `game_id` -- the same additive-merge
discipline `nfl_ats.forecast_weather_features.attach_forecast_weather_features`
already established for `weak_stack_v4`.

## 3. The candidate profile: `weak_stack_graph_sack`

A new `MarginFeatureProfile`, `weak_stack_graph_sack` = production
`weak_stack`'s exact feature set (`FEATURE_SETS["football_weak_stack"]` /
`FEATURE_SETS["full_weak_stack"]`) plus exactly one new column,
`graph_v2_team_stat_off_sack_rate_katz_diff`. Built on
`data/processed/game_features_weak_stack.parquet` (the PRODUCTION table)
directly, never on `weak_stack_v3`/`_surface`/`_v4` -- mirroring
`weak_stack_v4`'s own declared reason verbatim: stacking a candidate onto a
profile already refused or still undecided would confound the answer to "does
this add to what is actually played." Never referenced by the active model.
Never mixed with any other candidate profile.

## 4. The comparison

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_graph_sack` | production + the one graph feature |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`, `target="market_residual"`
fixed at the active model's own values (`artifacts/active_ats_model.json`) --
only `feature_profile` differs, isolating the graph column's marginal
contribution against everything the production chain already explains. Both
are fit with `nfl_ats.margin.fit_margin_model`, the same estimator production
itself uses, not a single-feature model -- this is the whole point of
"on top of production" rather than "on top of a bare baseline."

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring `docs/graph_ratings_v2_screen.md` §6 in
full: this is a screen, not a play/no-play decision. Per the binding "grade
the decision at the opener" rule, nothing here may settle a play/no-play or
promotion call; a terminal classification is reserved for an opener-graded
look, exactly as `scripts/graph_team_stat_record.py`'s `classify()` already
enforces for the sibling screen.

**Window and family.** `graph_ratings_v2_team_stat` (the 38-family screen's
own rotation family) has already spent its first window ([2011, 2013],
verdict `unresolved`) on the bare-baseline comparator, and the rotation
registry hard-refuses a family re-looking at seasons it or its inheritance
chain has already touched (`nfl_ats/rotation.py::_validate`, the
"no re-look" invariant). Its remaining ~2 eligible windows are reserved for
that family's OWN predeclared bare-baseline question (`docs/graph_ratings_v2_screen.md`
§8's "three readings worth carrying forward"), not spent here on a different
comparator.

This document therefore declares a **new, narrowly-scoped rotation family**,
`graph_off_sack_rate_on_production`, close-graded, declared with
`inherits=("graph_ratings_v2_team_stat",)` so it (a) cannot legally re-draw
[2011, 2013] -- disclosed lineage, enforced by the registry, not by
convention -- and (b) is honestly recorded as a variant of that line of work
rather than an unrelated fresh family. Its window is **assigned by
`nfl-ats rotation assign`**, never hand-picked -- the CLI computes the
earliest eligible block at declaration time. Given the current registry
state (parent window [2011, 2013] spent, mined seasons 2018-2025 blocked
since this family does not acknowledge them), the earliest eligible
size-3 block is expected to be **[2014, 2016]**; the actual assigned block is
confirmed in section 7, not asserted here.

## 6. Uncertainty and instrument checks, reusing the screen's own design unchanged

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read, never averaged with it. Same tool, same
`BOOTSTRAP_SAMPLES=1000`/`SEED=20260826` constants
`scripts/graph_team_stat_screen.py` already uses, for comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to
`docs/graph_ratings_v2_screen.md` §6: both arms' models are fit ONCE per
week on the REAL `ats_margin`; only the grading margin is shuffled within
week for the null, so 200 draws costs no extra model fits. This null is
**not** centred on zero by design (it preserves each week's realized
home-cover rate, and the two arms may carry different home-pick rates), and
is reported ALONGSIDE the bootstrap-vs-zero interval, never instead of it.

**Positive control**, run BEFORE the real screen: the candidate profile's one
new column (`graph_v2_team_stat_off_sack_rate_katz_diff`) is temporarily
REPLACED by the realized `ats_margin` -- a deliberate, large leak -- so the
harness must show an obvious, large effect. This proves the FULL-PROFILE
ridge fit (not just a single-feature model, per the screen's own instrument
check) can detect a real effect of meaningful size when one is actually
present. A "no effect" reading from a blind instrument would mean nothing;
this check exists so that possibility is ruled out first.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold --
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run
is close-graded, so it settles no play/no-play decision by itself; what it
DOES settle is whether the family's screen-stage finding, measured the honest
way (stacked on what is actually played, not a bare baseline), still looks
worth an eventual opener-graded confirmation look -- mirroring
`docs/graph_ratings_v2_screen.md` §8's "what this implies for the decision"
framing exactly.

## Recording

One `nfl-ats weak-signals record` entry, `effect_units=accuracy_points`,
family `graph_off_sack_rate_on_production` (a DIFFERENT pooling bucket from
`graph_ratings_v2_team_stat`, stated explicitly: the two measure the same
construct against two non-commensurable comparators -- bare market baseline
vs. the full production chain -- and AGENTS.md's commensurability rule
forbids pooling them together). `unresolved_below_power` at this close grade
regardless of sign, exactly as `scripts/graph_team_stat_record.py::classify`
already enforces for the sibling screen -- a resolved wrong sign at this
grade is reported as continuous evidence, never as a `refuted_mechanism`
closure, because that closure is reserved for the opener grade.

One `nfl-ats rotation record --name graph_off_sack_rate_on_production
--verdict unresolved` call spends the assigned window, carrying the same
paired effect, interval, and `probability_positive`.

## 7. Results (added after the look, 2026-08-31)

Rotation family `graph_off_sack_rate_on_production` was declared inheriting
`graph_ratings_v2_team_stat`, then assigned by `nfl-ats rotation assign`
(never hand-picked): the earliest eligible block, exactly as predicted in
section 5, is **[2014, 2016]** (749 games, 51 weeks). Two instrument checks
ran first.

**Null check** (`--mode null`, 200 within-week permutations): mean
+0.416 accuracy points, sd 0.940, 95% [-1.338, +2.270]. A sane, finite
distribution -- the harness produces a null, not a crash or a degenerate
spike.

**Positive control** (`--mode positive-control`, the candidate's one column
replaced by the realized `ats_margin`): paired delta **+48.999** accuracy
points, week-blocked P+ **1.000**, 95% [+45.661, +51.921], sitting at the
100.0th percentile of its own null. The full-profile ridge fit is not blind
to a real effect of meaningful size, even with the graph column embedded in
~270 other production features.

**The real screen** (`--mode screen`, artifact
`artifacts/graph_team_stat_off_sack_rate_on_production/20260831T150339Z/results.json`):
production `weak_stack` accuracy on this paired population is 51.00%
(baseline_accuracy 0.510013), `weak_stack_graph_sack` is 50.07%
(candidate_accuracy 0.500668). Paired candidate-minus-baseline delta
**-0.935** accuracy points. Week-blocked 95% CI **[-2.625, +0.809]**,
week-blocked `probability_positive` **0.122**. Season-blocked 95% CI
[-2.419, +1.992], season-blocked `probability_positive` 0.248. Home-pick
rate: baseline 37.6%, candidate 36.7% -- much closer together than the
bare-baseline screen's 55-67% home-pick arms, so this measurement carries far
less of the home-tilt artifact that discounted the earlier +2.949 reading.
Against its own permutation null (mean +0.416), the observed delta sits at
the **6.0th percentile** -- a mild negative lean, not an extreme tail value,
and nowhere near a resolved wrong sign at either blocking.

### What this implies for the decision, before what is wrong with it

On EV grounds -- `probability_positive` above 0.5 favours playing the
candidate, the only decision rule this project uses -- this measurement's
week-blocked P+ 0.122 does **not** favour adding the graph `off_sack_rate`
feature on top of what is actually played: the point estimate and the
probability mass both lean toward the status quo (production `weak_stack`
alone) over the candidate. This is **not** a closure: the week-blocked CI
upper bound is positive (+0.809), so the interval is not resolved entirely on
the wrong side of zero, and the positive control demonstrated the harness
CAN see a real effect -- it did not demonstrate that an effect the size of
the screen's own +2.949 reading would have been reliably detected at this
window's 749-game sample (the week-blocked CI here spans about 3.4 points,
wider than that reading), so `bounded_by_control` is not available either.
The result is recorded `unresolved_below_power`, exactly as the predeclared
grade discipline requires, and the family stays **open**.

The honest reading, stated plainly: the screen-stage's bare-baseline
`off_sack_rate` finding (+2.949 points against a zero-feature market
baseline) does **not** survive being measured the way that actually decides
anything -- stacked on top of what is played. This is precisely the
"composition is not the signal" pattern the project has recorded before: a
component positive alone can flip sign once stacked on the played chain,
because the chain already explains variance a bare-baseline comparison
credited to the candidate. Nothing here refutes the graph transform as a
mechanism (no resolved wrong sign, no split-half-reliability failure was
tested), and nothing here proves it inert (the interval still contains
positive values with real mass, P+ 0.122 is not P+ 0.00). It is simply
unresolved at this sample size, which per the binding taxonomy is the
expected shape for a real small signal and is never grounds to close the
line of work on its own.

