# Graph `team_stat` off_rush_epa_per_play, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/graph_ratings_v2_screen.md`, `docs/graph_ratings_v2.md`,
and its two sibling documents, `docs/graph_team_stat_on_production.md` (the
`off_sack_rate` version of exactly this experiment) and
`docs/graph_team_stat_def_ypp_on_production.md` (the `def_yards_per_play`
version), both completed 2026-08-31. **Sections 1-6 are the predeclaration**
and contain no accuracy, cover-rate, or `probability_positive` number against
NFL outcomes produced by this comparison. **Section 7 was added after the
look** and reports what it found; it changes nothing above it.

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

## 1. What this asks, what the prior already says, and why it is still worth a window

`docs/graph_ratings_v2_screen.md` §4 declared the comparator of first resort as
the raw `home - away` differential, and its §8 measured all 38 cluster
representatives against that bare comparator on a 2011-2013 window,
close-graded. It also recorded the project's standing lesson, restated in
ROADMAP.md and AGENTS.md: **"composition is not the signal"** — an overlay or
feature positive alone can go negative once stacked on the chain that is
actually PLAYED, because the played chain already explains some of the variance
a bare-baseline comparison credits to the candidate. The marginal that decides
is the one measured on top of what is played.

This document declares that stacked measurement for `off_rush_epa_per_play`,
item 2 of `docs/pool_edge_plan.md`'s 2026-08-31 ranked agenda. **The prior is
disclosed in full before the look, not after it.** Three readings bear on it,
and they do not agree:

**(a) The screen-stage reading for this cell is the WEAKEST of the three cells
the screen carried forward, by the reference that strips the artifact.**
**Read**, `docs/graph_ratings_v2_screen.md`:280 (the per-family table row):
`off_rush_epa_per_play` delta **+1.609** accuracy points, `P+` (vs zero)
**0.911**, week 95% CI **[-0.669, +3.963]**, null mean **+1.450**, percentile
**53.5**. **Read**, the same doc:310, in "The three readings worth carrying
forward", verbatim: "`off_rush_epa_per_play` reads P+ 0.911 against zero and
sits at the **53.5th** percentile of its own null (+1.609 observed, +1.450 null
centre). Essentially all of its apparent edge is the artifact. This single row
is the clearest argument for having built the permutation reference at all."
The registry entry agrees to the digit (**read**, `registry/weak_signals.json`,
`graph_team_stat_off_rush_epa_per_play`: effect 1.6085790884718498,
`probability_positive` 0.911, interval [-0.6693889652570011,
3.9631157983761227], family `graph_ratings_v2_team_stat`, seasons [2011, 2013],
746 games, 51 blocks; its own note states "the observed delta sits at the 53.5th
percentile of its own null" and "CLOSE-graded: may not settle a play/no-play
decision"). So on the screen's conservative reference, this cell's apparent
edge is very nearly all home-tilt artifact. That is not a reason to skip the
look — an artifact-inflated bare-baseline reading says nothing about a stacked
marginal in either direction — but it IS the honest prior, and it is stated
here before the outcome, not retrofitted to it.

**(b) A genuinely disjoint window reads the same sign, on the single highest
reliability figure in the family.** **Read**, `registry/weak_signals.json`,
`graph_input_screen_off_rush_epa_per_play`: effect **+1.996007984031936**
accuracy points, `probability_positive` **0.828**, interval
[-1.9286444202484647, 5.605683193310111], reliability
**0.9865554026446922**, family `graph_input_screen`, seasons **[2020, 2025]**,
**n=1,503** games, 107 blocks; its note reads "Holdout window (opener-graded,
2020-2025) is the decision-relevant figure recorded as effect/interval above,
n=1503" and "Selection window (close-graded, 2013-2019) delta_accuracy +0.805
pts, week-blocked P+ 0.671, n=1740". That is a different construct from (a) —
a single-feature ridge on the standardized raw differential versus a
zero-feature market baseline, not the graph-vs-raw contrast — measured on
seasons that do not intersect (a)'s 2011-2013 at all, and graded at the
OPENER. Two non-overlapping windows, same sign, and the highest split-half
reliability recorded anywhere in this family.

**(c) Both siblings of this exact experiment came back negative.** **Read**,
`registry/weak_signals.json`: `graph_team_stat_off_sack_rate_on_production`
effect **-0.9345794392523364** accuracy points, `probability_positive`
**0.122**, interval [-2.625192684247802, 0.8088456708064953], family
`graph_off_sack_rate_on_production`, seasons **[2014, 2016]**, 749 games; and
`graph_def_ypp_on_production` effect **-0.6675567423230975**,
`probability_positive` **0.189**, interval [-2.2429668524699014,
0.804289544235925], family `graph_def_ypp_on_production`, seasons
**[2014, 2016]**, 749 games. Neither is a closure (both intervals carry
positive mass; both are recorded `unresolved_below_power`), but both lean to
the status quo, and both were the cells that led the screen by the *naive* and
the *conservative* reference respectively. This cell led by neither.

**Inferred**, stated as reasoning and not as evidence: with (a) saying nearly
all the screen-stage edge is artifact and (c) saying the two better-positioned
siblings did not survive the stack, my expectation is a muted or negative
on-production read here. `docs/pool_edge_plan.md`'s own agenda item 2 says the
same thing in its own words: "highest reliability in the family, but the doc's
own disclosure means expect a more muted on-production read." I am recording
that expectation BEFORE the number so that whatever comes back cannot be
narrated as having been anticipated after the fact. The EV case for spending
the window anyway is (b): the highest reliability in the family, replicated on
a disjoint opener-graded window, is exactly the property that is supposed to
predict survival, and this is the only cell in the family that has it. If
reliability does not predict survival here, that is itself worth knowing.

It answers a different question than §8 did: does the graph-propagated
`off_rush_epa_per_play` signal add anything on top of the full PRODUCTION
`weak_stack`/`market_residual`/ridge/alpha-10 chain (**read**,
`artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`), rather
than on top of a zero-feature market baseline?

## 2. The candidate feature, unchanged from the screen

`graph_v2_team_stat_off_rush_epa_per_play_katz_diff`: the `team_stat` arm of
`src/nfl_ats/graph_ratings_v2.py` (`edge_signal="team_stat"`,
`signal_column="off_rush_epa_per_play"`), at the SAME structural configuration
frozen in `docs/graph_ratings_v2_screen.md` §5 and inherited, not refit, here —
identical to both sibling documents' own inheritance:

```
alpha=0.85, half_life_weeks=8.0, max_row_l1=1.0, prior_weight=1.0,
min_games=16, propagation="signed_katz", injury_beta=0.0
```

No new structural choice is made in this document. `off_rush_epa_per_play` is a
`STATE_METRICS` family (**read**, `src/nfl_ats/constants.py`:145, inside the
`STATE_METRICS` tuple opened at :142), carried in the feature table under the
standard prefix convention — **measured** this session on
`data/processed/game_features_weak_stack.parquet` (4,902 rows, 275 columns,
seasons 2009-2026): `home_off_rush_epa_per_play` and
`away_off_rush_epa_per_play` are both present. It is therefore not one of the
five suffix-form families `docs/graph_ratings_v2_screen.md` §1 names, so no
`signal_column_pair` override is needed, matching both sibling modules' own
default-prefix construction. The feature is computed once, leak-safe exactly as
proven in `tests/test_graph_ratings_v2.py` (ratings for week `w` read the graph
through week `w-1` only), over the full production `weak_stack` feature table's
completed games, then additively joined back onto that table by `game_id` — the
same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features` and both
sibling modules already established.

## 3. The candidate profile: `weak_stack_graph_off_rush_epa`

A new `MarginFeatureProfile`, `weak_stack_graph_off_rush_epa` = production
`weak_stack`'s exact feature set (`FEATURE_SETS["football_weak_stack"]` /
`FEATURE_SETS["full_weak_stack"]`) plus exactly one new column,
`graph_v2_team_stat_off_rush_epa_per_play_katz_diff`. Built on
`data/processed/game_features_weak_stack.parquet` (the PRODUCTION table)
directly, never on
`weak_stack_v3`/`_surface`/`_v4`/`_graph_sack`/`_graph_def_ypp` — mirroring
both siblings' declared reason verbatim: stacking a candidate onto a profile
already refused or still undecided would confound the answer to "does this add
to what is actually played." Never referenced by the active model. Never mixed
with any other candidate profile.

## 4. The comparison

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_graph_off_rush_epa` | production + the one graph feature |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values
(`artifacts/active_ats_model.json`) — only `feature_profile` differs, isolating
the graph column's marginal contribution against everything the production
chain already explains. Both are fit with `nfl_ats.margin.fit_margin_model`,
the same estimator production itself uses, not a single-feature model — this is
the whole point of "on top of production" rather than "on top of a bare
baseline."

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring `docs/graph_ratings_v2_screen.md` §6 and
both sibling documents in full: this is a screen, not a play/no-play decision.
Per the binding "grade the decision at the opener" rule, nothing here may
settle a play/no-play or promotion call; a terminal classification is reserved
for an opener-graded look, exactly as `scripts/graph_team_stat_record.py`'s
`classify()` already enforces for the sibling screen.

**Window and family.** `graph_ratings_v2_team_stat` (the 38-family screen's own
rotation family) has already spent its first window ([2011, 2013], verdict
`unresolved`) on the bare-baseline comparator, and the rotation registry hard-
refuses a family re-looking at seasons it or its inheritance chain has already
touched (`nfl_ats/rotation.py::_validate`, the "no re-look" invariant). Its
remaining eligible windows are reserved for that family's OWN predeclared
bare-baseline question (`docs/graph_ratings_v2_screen.md` §8's "three readings
worth carrying forward"), not spent here on a different comparator.

This document therefore declares a **new, narrowly-scoped rotation family**,
`graph_off_rush_epa_on_production`, close-graded, declared with
`inherits=("graph_ratings_v2_team_stat",)` — the SAME single-parent lineage
both sibling families (`graph_off_sack_rate_on_production`,
`graph_def_ypp_on_production`) declared, not a lineage naming either sibling
itself. This is disclosed rather than glossed over: both siblings inherit from
`graph_ratings_v2_team_stat` too and have each already spent a [2014, 2016]
window under that lineage — but per `nfl_ats/rotation.py`'s own documented
design (`_inherited_names` walks the declared `inherits` edges upward only; a
family is never held responsible for what an unrelated sibling that shares the
same parent has drawn, "rule 4 makes windows retire per-family, so independent
families are explicitly allowed to draw the same seasons"), this family's
touched-season set is computed from `graph_ratings_v2_team_stat`'s OWN windows,
not from every family that inherits it.

**The expected window is therefore [2014, 2016]** — that is what BOTH siblings
drew from the identical lineage position, so it is what the earliest-eligible-
block computation should produce again. This is stated as an **expectation to
be confirmed by the CLI, not a choice**: the block is ASSIGNED by
`nfl-ats rotation assign`, never hand-picked, and whatever it returns
(including `remaining_eligible_windows`) is pasted verbatim into §7. A
same-window outcome would not be a defect: the original 38-family screen itself
scored `off_sack_rate`, `def_yards_per_play` and `off_rush_epa_per_play` on the
identical [2011, 2013] window as one look, so three graph-feature candidates
measured on the same window is the established, comparable shape for this line
of work, not a re-look at a question already answered. It also makes this
result directly comparable, game for game, with both siblings' — which is the
whole point of running the third cell at all.

## 6. Uncertainty and instrument checks, reusing the screen's own design unchanged

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read, never averaged with it. Same tool, same
`BOOTSTRAP_SAMPLES=1000`/`SEED=20260826` constants
`scripts/graph_team_stat_screen.py` already uses, for comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to
`docs/graph_ratings_v2_screen.md` §6 and both sibling documents' §6: both arms'
models are fit ONCE per week on the REAL `ats_margin`; only the grading margin
is shuffled within week for the null, so 200 draws costs no extra model fits.
This null is **not** centred on zero by design (it preserves each week's
realized home-cover rate, and the two arms may carry different home-pick
rates), and is reported ALONGSIDE the bootstrap-vs-zero interval, never instead
of it. For this cell the null reference matters more than for either sibling:
§1(a) above is precisely the finding that this family's screen-stage edge was
almost entirely the null's own offset.

**Positive control**, run BEFORE the real screen: the candidate profile's one
new column (`graph_v2_team_stat_off_rush_epa_per_play_katz_diff`) is
temporarily REPLACED by the realized `ats_margin` — a deliberate, large leak —
so the harness must show an obvious, large effect. This proves the FULL-PROFILE
ridge fit (not just a single-feature model, per the screen's own instrument
check) can detect a real effect of meaningful size when one is actually
present. A "no effect" reading from a blind instrument would mean nothing; this
check exists so that possibility is ruled out first.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run is
close-graded, so it settles no play/no-play decision by itself; what it DOES
settle is whether the family's screen-stage finding, measured the honest way
(stacked on what is actually played, not a bare baseline), still looks worth an
eventual opener-graded confirmation look — mirroring
`docs/graph_ratings_v2_screen.md` §8's and both sibling documents' "what this
implies for the decision" framing exactly.

## Recording

One `nfl-ats weak-signals record` entry, `effect_units=accuracy_points`, family
`graph_off_rush_epa_on_production` (a DIFFERENT pooling bucket from
`graph_ratings_v2_team_stat`, `graph_input_screen`,
`graph_off_sack_rate_on_production` and `graph_def_ypp_on_production`, stated
explicitly: all of them measure graph-propagated `team_stat` constructs but
against non-commensurable comparators or different underlying families — this
one's comparator is the FULL PRODUCTION CHAIN, not a bare market baseline and
not a raw differential — and AGENTS.md's commensurability rule forbids pooling
them together). `unresolved_below_power` at this close grade regardless of
sign, exactly as `scripts/graph_team_stat_record.py::classify` already enforces
for the sibling screen — a resolved wrong sign at this grade is reported as
continuous evidence, never as a `refuted_mechanism` closure, because that
closure is reserved for the opener grade.

One `nfl-ats rotation record --name graph_off_rush_epa_on_production --verdict
unresolved` call spends the assigned window, carrying the same paired effect,
interval, and `probability_positive`.

## 7. Results (added after the look, 2026-09-01)

Rotation family `graph_off_rush_epa_on_production` was declared inheriting
`graph_ratings_v2_team_stat`, then assigned by `nfl-ats rotation assign` (never
hand-picked). The CLI's own response, verbatim on the fields that matter
(**measured**, `nfl-ats rotation assign --name graph_off_rush_epa_on_production`,
run this session under the shared registry lock):

```
"assigned": "graph_off_rush_epa_on_production",
"remaining_eligible_stratified_seasons": 1,
"remaining_eligible_windows": 0,
"status": "open",
"windows": [ { "assigned_at": "2026-09-01", "seasons": [2014, 2016],
               "state": "assigned", "window_kind": "contiguous" } ]
"grade_pools": { "close":  { "default_window_size": 3, "seasons": [2009, 2025],
                             "total_windows": 5, "unspent_blocks": [],
                             "unspent_windows": 0 },
                 "opener": { "default_window_size": 2, "seasons": [2020, 2025],
                             "total_windows": 3, "unspent_blocks": [],
                             "unspent_windows": 0 } }
"season_usage" (post-record): 2011:2 2012:1 2013:3 2014:6 2015:6 2016:5
                              2017:2 2020:5 2021:5 2022:1 2023:1 2025:1
```

The assigned block is **[2014, 2016]** (749 games, 51 weeks) — exactly what §5
declared as the expectation to be confirmed, and the SAME block both sibling
families drew, for the reason §5 disclosed in advance (a sibling's spent window
is invisible to this family's touched-season computation, since it is a sibling
under `graph_ratings_v2_team_stat`, not an ancestor of this family). Note
`remaining_eligible_windows: 0` after this assignment: this family has spent its
one contiguous close block, and only a two-leg stratified option
(`remaining_eligible_stratified_seasons: 1`) remains under this lineage. Two
instrument checks ran first.

**Null check** (**measured**, `--mode null`, 200 within-week permutations, real
candidate feature, not leaked; artifact
`artifacts/graph_team_stat_off_rush_epa_per_play_on_production/20260901T184209Z/results.json`):
mean **+0.196** accuracy points, sd 0.512, 95% [-0.668, +1.202], observed
-0.935. A sane, finite distribution — the harness produces a null, not a crash
or a degenerate spike. Its centre (+0.196) is nearer zero than either sibling's
(+0.416 for `off_sack_rate`, +0.263 for `def_yards_per_play`), which is what a
smaller home-pick-rate gap between the two arms produces.

**Positive control** (**measured**, `--mode positive-control`, the candidate's
one column replaced by the realized `ats_margin`; artifact
`.../20260901T184225Z/results.json`): candidate accuracy **100.0%** against
baseline 51.00%, paired delta **+48.999** accuracy points, week-blocked P+
**1.000**, 95% [+45.661, +51.921], season-blocked 95% [+48.000, +49.597], P+
1.000, sitting at the 100.0th percentile of its own null (which itself centres
at +2.652 pts under the leak treatment). The full-profile ridge fit is not blind
to a real effect of meaningful size, even with the graph column embedded in ~270
other production features.

**The real screen** (**measured**, `--mode screen`, run exactly once; artifact
`artifacts/graph_team_stat_off_rush_epa_per_play_on_production/20260901T184239Z/results.json`):
production `weak_stack` accuracy on this paired population is **51.00%**
(`baseline_accuracy` 0.5100133511348465), `weak_stack_graph_off_rush_epa` is
**50.07%** (`candidate_accuracy` 0.5006675567423231). Paired
candidate-minus-baseline delta **-0.935** accuracy points
(`delta_accuracy` -0.009345794392523364). Week-blocked 95% CI **[-1.998,
+0.135]**, week-blocked `probability_positive` **0.037**. Season-blocked 95% CI
**[-1.195, -0.800]**, season-blocked `probability_positive` **0.000** — reported
alongside the week-blocked read per §6's own discipline, but read with the same
caution `docs/era_weighting_promotion.md` gives its own 2-season-block
secondary: with only 3 season blocks in this window the season-blocked bootstrap
has almost no combinatorial diversity, so its entirely-below-zero interval is a
low-power artifact of block count, not a sharper answer than the week-blocked
primary, and within-week game correlation is zero by owner mandate, which is
what makes the week block the honest reference here. Home-pick rate: baseline
**37.63%**, candidate **36.72%** — close together, so (like both siblings, and
unlike the bare-baseline screen's 55-67% home-pick arms) this measurement
carries very little of the home-tilt artifact that discounted the earlier
screen. Against its own permutation null (mean +0.196, sd 0.512), the observed
delta sits at the **0.5th percentile** — the deepest negative-tail placement of
the three siblings (`off_sack_rate` 6.0th, `def_yards_per_play` 8.0th) — though
with 200 draws, three uncorrected families and no multiplicity correction across
the graph-column experiments, one family landing in a tail is not itself a
finding.

One numerical curiosity worth stating so nobody mistakes it for a bug or for
corroboration: the paired delta here, -0.009345794392523364, is **bit-identical
to the `off_sack_rate` sibling's**. Both are exactly -7 net games out of the
same 749, on the same window, on an accuracy grid whose step is 1/749. Same
grid, same denominator, different feature — a coincidence on a coarse discrete
scale, not a shared computation and not evidence that the two features behave
alike. Their intervals and null percentiles differ (`off_sack_rate` [-2.625,
+0.809] P+ 0.122 at the 6.0th percentile; this one [-1.998, +0.135] P+ 0.037 at
the 0.5th), which is what actually distinguishes them.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the candidate,
the only decision rule this project uses — this measurement's week-blocked P+
**0.037** does **not** favour adding the graph `off_rush_epa_per_play` feature
on top of what is actually played. Both the point estimate and nearly all the
probability mass lean to the status quo (production `weak_stack` alone). Taking
the other side of a 96/4 read would be the mistake here, exactly as declining an
87/13 read would be in the opposite direction: the decision follows the expected
value, and at this window the expected value says leave the card alone.
Production `weak_stack` is unchanged by this work; no card, model, or artifact
was touched.

This is **not** a closure. The week-blocked CI upper bound is positive
(+0.135), so the interval is not resolved entirely on the wrong side of zero and
`wrong_sign_resolved` is unavailable — and the binding grade rule settles it
independently anyway: this is a CLOSE-graded look, and a close grade may not
settle a play/no-play decision or draw a terminal classification regardless of
sign, exactly as `scripts/graph_team_stat_record.py::classify` already enforces
for the sibling screen. The season-blocked secondary CI does sit wholly below
zero, and it is named here rather than buried, but it is a 3-block bootstrap at
a grade that closes nothing, not a resolution. The underlying trait is the most
reliable in the whole family (split-half 0.987), so
`no_split_half_reliability` is not available either. And the positive control
demonstrated the harness CAN see a real effect — it did not demonstrate that an
effect the size of the disjoint holdout's own +1.996 reading would have been
reliably detected at this window's 749-game sample (the week-blocked CI here
spans about 2.1 points, wider than that reading), so `bounded_by_control` is not
available. The result is recorded `unresolved_below_power`, exactly as the
predeclared grade discipline requires, and the family stays **open** — though
with `remaining_eligible_windows: 0`, any future look under this lineage would
have to be the stratified two-leg option.

The honest reading, stated plainly and against the prior §1 wrote down before
the number existed: `off_rush_epa_per_play` was ranked second on
`docs/pool_edge_plan.md`'s agenda **because of its reliability**, not because of
its screen-stage magnitude — the screen doc itself had already said "essentially
all of its apparent edge is the artifact," and §1 recorded my expectation of a
muted or negative on-production read before the run. That expectation held. What
is new, and what the window actually bought, is a direct test of the premise
behind ranking it at all: **the highest split-half reliability in the family
(0.987), replicated in sign on a disjoint opener-graded window (+1.996 pts, P+
0.828, 2020-2025, n=1,503), did not predict survival on top of production.**
Reliability says a trait is measured consistently; it says nothing about whether
the production chain already prices it. That distinction is the transferable
lesson here, and it is the same "composition is not the signal" pattern recorded
twice before today, now with the family's best-measured member rather than its
biggest-magnitude one.

All three of the screen's named carry-forward cells — `off_sack_rate` (naive
-reference leader), `def_yards_per_play` (conservative-reference leader), and
now `off_rush_epa_per_play` (reliability leader) — have been measured on top of
production on the same [2014, 2016] window, close-graded, and all three read
negative: -0.935 (P+ 0.122), -0.668 (P+ 0.189), -0.935 (P+ 0.037). Nothing here
refutes the graph transform as a mechanism (no resolved wrong sign at the
primary reference, no reliability failure, no control bound), and nothing here
proves it inert. **Inferred**, and offered as reasoning rather than evidence:
the consistent direction across three differently-selected cells on one shared
window is more suggestive than any single cell, but they are three correlated
looks at the same 749 games, not three independent votes, so the honest summary
is that the graph-on-production line has not produced a positive read and has
not been refuted either. It is unresolved at this sample size, which per the
binding taxonomy is the expected shape for a real small signal and is never
grounds to close the line of work on its own.

### Recorded

- `nfl-ats weak-signals record --name graph_off_rush_epa_on_production --family
  graph_off_rush_epa_on_production --classification unresolved_below_power
  --effect -0.9345794392523364 --effect-units accuracy_points --interval-low
  -1.9975371518901712 --interval-high 0.13513513513513514
  --probability-positive 0.037 --sample-games 749 --sample-blocks 51
  --season-start 2014 --season-end 2016` →
  `{"classification": "unresolved_below_power", "effect": -0.9345794392523364,
  "favours_candidate": false, "recorded": "graph_off_rush_epa_on_production",
  "total_signals": 616}`. Its note states explicitly that the comparator is the
  full production chain and the entry is therefore **not poolable** with
  `graph_ratings_v2_team_stat`, `graph_input_screen`, or either sibling family.
- `nfl-ats rotation record --name graph_off_rush_epa_on_production --verdict
  unresolved --probability-positive 0.037 --effect -0.9345794392523364
  --effect-units accuracy_points --interval-low -1.9975371518901712
  --interval-high 0.13513513513513514 --sample-blocks 51` → window
  `[2014, 2016]` moved to `"state": "spent"`, `"verdict": "unresolved"`,
  `"spent_at": "2026-09-01"`, `"closing_ground": null`; family
  `"status": "open"`.
