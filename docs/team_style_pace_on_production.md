# Team-style pace mismatch, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/illness_on_production.md`,
`docs/graph_team_stat_def_ypp_on_production.md` and
`docs/fluview_on_production.md`. **Sections 1-6 are the predeclaration** and
contain no accuracy, cover-rate or `probability_positive` number against NFL
outcomes from this comparison. **Section 7 was added after the look** and
reports what it found; it changes nothing above it.

Ranked #3 of four in `docs/on_production_sweep_20260901.md` section 1.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator. Decisions
are expected value: `probability_positive` above 0.5 favours the candidate;
predeclared thresholds govern only what docs may CLAIM. Grade play/no-play
decisions at the OPENER; a close-graded look settles no play decision and is
recorded `unresolved_below_power` regardless of sign. Never say something
"needs N more games". Within-week game correlation is ZERO by owner mandate.

## 1. What this closes, and why pace is not another quality feature

### 1.1 The team-style channel has never been stacked on production

`docs/team_style.md` (PBP-08) predeclared and froze five team-style cells and
scored every one of them against a **bare market baseline** — a subset
cover-rate gap scaled to the full slate. **Read**,
`registry/weak_signals.json`, the two cells that matter here:

| cell | effect (accuracy points) | week-blocked 95% | `probability_positive` | reliability | n | n_flag |
|---|---|---|---|---|---|---|
| `team_style_pace_mismatch_dog_cover` | +0.229 | [-0.559, +1.040] | 0.711 | **0.489** | 4313 | 1018 |
| `team_style_short_game_identity` | +0.350 | [-0.256, +0.951] | 0.870 | 0.408 | 8634 | 2070 |

Reliability 0.489 is the **highest in the team-style battery** (**read**,
`scripts/team_style_screen.py` `reliability_notes`: `seconds_per_play_pace`
YoY Pearson r +0.489, 95% [+0.405, +0.567], n=512 team-season pairs; against
`short_pass_share` +0.408, `deep_share` +0.306, `shotgun_rate_faced` +0.278).
AGENTS.md makes reliability the decisive field — an unreliable trait is refuted
because no sample size rescues it — so the battery's most reliable input is the
one worth spending a window on, and it was chosen on that basis rather than on
its point estimate (see section 6, where the cost of that choice is disclosed
before any result).

The project's own recorded lesson — "composition is not the signal"
(AGENTS.md, ROADMAP.md) — is that a component positive alone can go negative
once stacked on the chain that is actually PLAYED, because the played chain
already explains some of the variance a bare-baseline comparison credits to
the candidate. **This document asks the marginal question that decides:** does
the pace-mismatch indicator add anything on top of the full PRODUCTION chain
(**read**, `artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`,
`model_id: "d1f07d773475dc58"`)? No team-style construct has ever been asked
that question — every on-production test that exists came from the graph-
propagated `team_stat` channel or the CDC/club illness channels
(`docs/on_production_sweep_20260901.md` section 1.1).

### 1.2 Pace is a style axis, and the flag is a shape a ridge cannot form

The project's build filter (`.claude` memory `team-quality-is-already-priced`,
ROADMAP.md PBP-05) says features that only measure team quality better are
bounded near zero. That filter is the reason most candidates are not worth a
window. It does not bind here, for two independent reasons.

**First, tempo is not quality.** `scripts/team_style_features.py`'s own module
docstring states the design intent — the nine style dimensions "are
deliberately QUALITY-ORTHOGONAL: none of them measure how good a team is, only
how it chooses to play, so they sit outside the measured team-quality ceiling
(ROADMAP.md PBP-05)". `seconds_per_play_pace` is season sum(drive
time-of-possession) / sum(drive play count) (**read**, `docs/team_style.md:90`),
league-season-centred, so leaguewide tempo drift over 2009-2025 cannot read as
a team identity.

**Second, the flag is an ABSOLUTE GAP, `|home − away|`, and a linear ridge
cannot form one.** Even if both teams' centred pace values were present as
columns, a linear model can form `home − away` but not `|home − away|`; the
mechanism `docs/team_style.md:204-210` predeclares is a variance mechanism (a
pace mismatch compresses total offensive possessions, and fewer possessions
favour the dog), which is symmetric in the sign of the gap. That is precisely
the functional form a linear ridge is blind to.

**And production carries no pace column at all.** **Measured** this session:

```
./.tools/uv.exe run --no-sync python -c "from nfl_ats.constants import FEATURE_SETS; \
  full = FEATURE_SETS['full_weak_stack']; print(len(full)); \
  print([c for c in full if 'pace' in c or 'seconds_per_play' in c])"
-> 90
-> []
```

`FEATURE_SETS["full_weak_stack"]` holds **90 columns** and none of them is a
pace column in any form. **One correction to the sweep doc, stated rather than
glossed:** `docs/on_production_sweep_20260901.md:147` says "`pbp_off_pass_rate`
and `pbp_off_proe` are in" that 90-column set. **Measured**, they are not —
`[c for c in FEATURE_SETS['full_weak_stack'] if 'pass_rate' in c or 'proe' in c
or 'pbp' in c]` returns `[]`, and the full 90-column listing carries no
play-calling-tendency column of any kind (the closest are EPA/yards-per-play
rate statistics, which are quality measures). The correction makes the case for
this experiment **stronger**, not weaker: production carries no tendency or
tempo feature whatsoever, so this is the first play-style column the chain has
ever been offered. The sweep doc is owned by another program and is not edited
here; this paragraph is the correction of record.

## 2. The column, the reused source, and the one deliberate deviation

`team_style_pace_mismatch_flag` = **1** when
`|home_prior_season_seconds_per_play_pace_centered −
away_prior_season_seconds_per_play_pace_centered|` is at or above the
top-quartile threshold, **0** otherwise, **NaN** where either team has no
prior-season pace value or the threshold is undefined.

**Source, reused not rebuilt.** `data/pbp/team_style/team_season_style.parquet`
— **measured** this session: 544 rows, seasons 2009-2025, one row per
(season, team), carrying `seconds_per_play_pace` and
`seconds_per_play_pace_centered` with zero nulls in the centred column. That
table is the exact artifact `scripts/team_style_features.py` writes and
`scripts/team_style_screen.py` reads; this module reads the parquet directly
rather than importing the builder, so no new structural choice is made and the
`scripts/` package does not join `mypy src`'s import graph. Each game's home and
away team is joined to that team's **prior season** row (`season + 1`, the
same one-season forward shift `scripts/team_style_screen.py::_prior` performs),
and both sides are canonicalised through
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`, matching the screen.

The feature is additively joined onto the production feature table by `game_id`
with `validate="one_to_one"` — the same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features`,
`nfl_ats.illness_production_feature` and every sibling candidate module already
established: every pre-existing column comes back bit-identical, only the one
new column is added.

### 2.1 The deviation: an expanding threshold, declared before scoring

`scripts/team_style_screen.py` computes its top-quartile threshold as
`game["pace_diff_abs"].quantile(0.75)` over the **whole 2009-2025 panel**.
For a screen that scores one pooled cover-rate gap that is a defensible pooled
statistic, but it is a **mild look-ahead**: a 2012 game would be flagged with a
cut estimated partly from 2020 data. A pregame feature column may not carry
that.

**This column's threshold is recomputed expanding over strictly prior seasons
only**: for a game in season S, the threshold is the 75th percentile of the
absolute pace gap across every game in the panel whose season is **strictly
less than S**. Where no prior seasons carry a defined gap, the threshold is
undefined and the flag is NaN.

Three consequences, declared here rather than discovered afterwards:

1. **This makes the column a slightly different quantity from the registered
   cell.** `team_style_pace_mismatch_dog_cover` is a whole-panel-threshold
   flag; `team_style_pace_mismatch_flag` is an expanding-threshold flag. They
   agree on most games and disagree near the cut. The registered cell's numbers
   are context for why this window is worth spending, never a prediction of
   what this column will score.
2. **A noisier threshold can only attenuate toward the null.** The
   expanding cut is estimated on fewer games than the pooled cut, especially in
   the first eligible seasons, so it misclassifies some near-threshold games in
   both directions. Independent misclassification of a binary regressor
   attenuates its fitted coefficient toward zero; it cannot manufacture an
   effect that is not there. Whatever this column scores is therefore a
   *conservative* reading of the construct, not an inflated one.
3. **The leakage test pins it.** `tests/test_team_style_pace_production_feature.py`
   proves on a synthetic panel that a season-S game's flag uses only season
   `< S` pace values AND a threshold estimated only from seasons `< S`:
   injecting an extreme pace value into season S or later cannot change any
   season-S flag, while injecting one into season S-1 does change it (so the
   test proves a cutoff, not a builder that never returns anything).

**Two smaller declared differences from the screen's own population.** The
screen dropped true pick'ems (`spread_line == 0`) because its value column
`dog_cover` is undefined there; this column is a feature, not a value column,
so no pick'em restriction applies — the flag is defined for every game whose
two teams have prior-season pace. And the screen folded a missing prior-season
pace into `False`; a feature column may not, because "no prior-season pace" and
"a measured small gap" are different states and only the model's own
training-fold median (`fit_margin_model`) may decide what to do with the first.
Unseen games come back **NaN**, exactly as `nfl_ats.illness_production_feature`
and `nfl_ats.fluview_production_feature` already treat their own missing
coverage.

## 3. The candidate profile

One new `MarginFeatureProfile`, `weak_stack_team_style_pace` = production
`weak_stack`'s exact feature set plus **exactly one** new column — the same
"one new column" shape `weak_stack_graph_sack`, `weak_stack_graph_def_ypp`,
`weak_stack_fluview_home/_away` and `weak_stack_illness_away/_home` use.

**Measured** this session:
`margin_feature_set("market_residual", "weak_stack_team_style_pace")` resolves
to `full_weak_stack_team_style_pace`, a **91-column** set = the 90-column
`full_weak_stack` production set plus exactly `team_style_pace_mismatch_flag`
(`sorted(set(FEATURE_SETS['full_weak_stack_team_style_pace']) -
set(FEATURE_SETS['full_weak_stack']))` → `['team_style_pace_mismatch_flag']`).

It is built on `data/processed/game_features_weak_stack.parquet` — the
PRODUCTION table — **directly**, never on
`game_features_weak_stack_v3/_surface/_v4/_graph_*/_fluview/_illness.parquet`,
mirroring `weak_stack_graph_sack`'s own declared reason verbatim: stacking a
candidate onto a profile already refused or still undecided would confound the
answer to "does this add to what is actually played." The widened table is
written to its own path
(`data/processed/game_features_weak_stack_team_style_pace.parquet`); the
production table is never touched. Never referenced by the active model. Never
mixed with any other candidate profile.

## 4. The comparison

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_team_style_pace` | production + the one pace-mismatch column |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values
(`artifacts/active_ats_model.json`); only `feature_profile` differs, isolating
the column's marginal contribution against everything the production chain
already explains. Both are fit with `nfl_ats.margin.fit_margin_model` — the
same estimator production itself uses, not a single-feature model — which is
the whole point of "on top of production" rather than "on top of a bare
baseline". Both are fit and scored on the **same games in the same weeks**,
forward-chaining only: each week is predicted from a model trained strictly on
games that kicked off before that week's earliest kickoff.

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring both sibling documents. Per the binding
"grade the decision at the opener" rule, nothing here may settle a play/no-play
or promotion call; every recorded classification is `unresolved_below_power`
regardless of sign.

**Window and family.** A new rotation family, `team_style_pace_on_production`,
close-graded, declared with **no `--inherits`**: the team-style battery has
never held a rotation family at all. **Verified** with
`nfl-ats rotation status` before declaring — the 16 existing families are
`best_pick_ranker`, `best_pick_ranker_opener`, `cfb_role_continuity`,
`combined_stacker`, `era_weighting_half_life_8`,
`fluview_elevated_on_production`, `fluview_home_elevated_opener`,
`graph_def_ypp_on_production`, `graph_off_rush_epa_on_production`,
`graph_off_sack_rate_on_production`, `graph_ratings_v2_team_stat`,
`illness_on_production`, `mod07_weak_signal_stack`, `movement_expansion_v1`,
`pbp_drive_bundle` and `player_qb_continuity`, and none of them is a
team-style family. There is therefore no lineage to declare and nothing to
inherit; PBP-08's bare-baseline screen was scored outside the rotation registry
entirely.

Declared **without** `--acknowledge-mined`, because the deterministic
earliest-eligible close block is not expected to intersect the 2018-2025 mining
ledger; if the CLI refuses, that refusal is the authority and section 7 records
what actually happened.

The window is **ASSIGNED by `nfl-ats rotation assign`**, never hand-picked
(`src/nfl_ats/rotation.py::assign_window`: "the lowest-starting block of the
requested size inside the grade's pool that starts at or after the warm-up
floor ... There is no hidden choice and nothing to tune"). The assigned block
is confirmed in section 7, not asserted here.

## 6. Uncertainty, instrument checks, and the power caveats stated up front

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read reported beside it, **never averaged with
it**. Same `BOOTSTRAP_SAMPLES=1000` / `SEED=20260826` constants the sibling
on-production scripts already import from `scripts/graph_input_screen.py`, for
comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to the
sibling documents: both arms' models are fit ONCE per week on the REAL
`ats_margin`; only the grading margin is shuffled within week for the null, so
200 draws cost no extra model fits. This null is **not** centred on zero by
design — it preserves each week's realized home-cover rate, and the two arms
may carry different home-pick rates — and is reported ALONGSIDE the
bootstrap-vs-zero interval, never instead of it.

**Positive control**, run BEFORE the real screen: the candidate profile's one
new column is temporarily REPLACED by the realized `ats_margin`, a deliberate
large leak, so the harness must show an obvious large effect. This proves the
FULL-PROFILE ridge fit can detect a real effect of meaningful size when one is
actually present, even with the candidate column embedded in 90 other
production features. A "no effect" reading from a blind instrument would mean
nothing; this check exists so that possibility is ruled out first. If the
control does not fire, the screen is not run and the instrument is reported
blind.

### 6.1 What is weak about this candidate, disclosed before any result

**It fires on the broadest slice of the four constructs in this sweep, and it
carries the lowest registered `probability_positive` of the four.** **Read**,
`registry/weak_signals.json`: `team_style_pace_mismatch_dog_cover` fires on
**1018 of 4313 games — 23.6% of the slate** — the broadest of the four
constructs in `docs/on_production_sweep_20260901.md`, and its registered
`probability_positive` is **0.711**, the *lowest* of the four. It was ranked #3
and selected on **reliability** (0.489, the battery's highest) rather than on
point estimate or P+. That is a deliberate, declared trade: AGENTS.md makes
reliability decisive because it is the one field that can refute a trait
outright, but a construct chosen on reliability is not thereby a construct with
a large expected effect, and nothing here should be read as predicting one.

A 23.6% firing rate cuts both ways and both directions are stated here. It
gives the column real support inside a three-season window — roughly 175-185
firings on a ~750-game block, far more than the illness sibling's home arm had
— so this is not a sparse-indicator power problem. But a broad flag is also a
blunt one: the top quartile of pace gaps is not an extreme condition, and if
the mechanism lives only in genuinely extreme mismatches, a quartile cut
dilutes it. The quartile cut is inherited unchanged from the frozen
`docs/team_style.md` predeclaration and is **not** re-tuned here; re-cutting it
after seeing an outcome number is exactly the practice these documents exist to
prevent.

**The expanding threshold is thinnest in the window's first season.** Section
2.1's cut for season S is estimated from games in seasons `< S` only. Since a
game's pace gap needs both teams' prior-season pace, the earliest season
carrying a defined gap is 2010, so the earliest season carrying a defined
threshold is 2011 — and that 2011 threshold rests on a single season of gaps
(~256 games) rather than the pooled panel. Per section 2.1's consequence (2)
this attenuates toward the null; it cannot manufacture an effect.

**Coverage floor.** 2009 games have no prior-season pace and 2010 games have
no prior-season threshold, so both come back NaN by construction. Any assigned
window starting in 2011 or later is fully covered; a window reaching earlier
would carry structural NaN, and section 7 reports the measured coverage inside
whatever block is assigned rather than assuming it.

### 6.2 The sibling arm this document deliberately does not run

`team_style_short_game_identity` sits at `probability_positive` **0.870** with
reliability **0.408** (**read**, `registry/weak_signals.json`) — a higher P+
than the pace cell — and it is **not** an arm of this experiment. The reason is
structural, not a judgement about its merit: it is a **TEAM-level** flag scored
on a long table with one row per (game, side) and value column `team_covered`
(**read**, `scripts/team_style_screen.py::build_long_table`), whereas every
column in the production feature table is **game-level**, one row per
`game_id`. Turning it into a feature column requires a home/away split — two
columns, `home_short_game_identity` and `away_short_game_identity`, or a signed
difference — which is a different column shape, a different number of arms, and
a multiplicity question this window has not declared. It is left for a future
look with its own predeclaration; naming it here, before scoring, is what stops
it from being quietly added later if this arm disappoints.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run is
close-graded, so it settles no play/no-play decision by itself; what it DOES
settle is whether the team-style channel's screen-stage finding, measured the
honest way (stacked on what is actually played, not a bare baseline), still
looks worth an eventual opener-graded confirmation look.

## Recording

ONE `nfl-ats weak-signals record` entry, `effect_units=accuracy_points`,
`league=nfl`, family **`team_style_pace_on_production`**, classification
`unresolved_below_power` at this close grade regardless of sign, carrying
`reliability=0.489` (the construct's own recorded YoY figure) and
`category=onfield`.

**This family is NOT poolable with the bare-baseline team-style battery**, and
the record's notes say so explicitly. Both measure the same underlying pace
construct, but against **non-commensurable comparators**: the battery scored a
subset cover-rate gap against a bare market baseline scaled to the full slate,
while this scores a paired accuracy delta against the full production ridge
chain. AGENTS.md's commensurability rule — "pooled inputs must be commensurable
(same units, same scale, same population) and the family must be declared
before the signs are seen" — forbids pooling them into one estimate. The
expanding-threshold deviation of section 2.1 is a second, independent reason:
the two are not even the same column.

ONE `nfl-ats rotation record --name team_style_pace_on_production --verdict
unresolved` call spends the assigned window, carrying the same paired effect,
interval and `probability_positive`.

## 7. Results (added after the look, 2026-09-01)

Rotation family `team_style_pace_on_production` was declared close-graded with
**no inheritance** (the team-style battery holds no rotation family), then
assigned by `nfl-ats rotation assign` — never hand-picked. The earliest
eligible close-pool block is **[2011, 2013]**, the same block the illness
sibling drew, which is expected and not a defect: `assign_window` hands every
fresh close-graded family with no inheritance the deterministic
lowest-starting eligible block, and rule 4 makes windows retire per-family, so
independent families are explicitly allowed to draw the same seasons. Both
instrument checks ran first, in the declared order.

**Coverage inside the window.** **Measured** (`--mode screen` console output
and `result.flag_coverage` in the artifact): 768 REG games,
`team_style_pace_mismatch_flag` non-missing on **100.000%** of them, firing on
**23.698%** (182 games). That reproduces the registered bare-baseline screen's
own 23.6% firing rate (1018/4313) almost exactly, which is the reproduction
check this differently-thresholded construction had to pass. Section 6.1's
coverage floor did not bite: the window starts in 2011, the first season the
expanding threshold is defined. **Measured** across the whole widened table
(`scripts/build_weak_stack_team_style_pace_table.py` console output): coverage
89.106% of 4,902 rows — 2009 and 2010 are structurally NaN as declared — and a
pooled firing rate of 19.574%, with per-season rates ranging 9.7% (2018) to
30.3% (2014).

**Null check** (`--mode null`, 200 within-week permutations, real candidate
feature, not leaked): mean **-0.259** accuracy points, sd 1.162, 95%
[-2.416, +1.749]. A sane, finite, non-degenerate distribution — the harness
produces a null, not a crash or a spike.

**Positive control** (`--mode positive-control`, the candidate's one column
replaced by the realized `ats_margin`): paired delta **+50.134** accuracy
points, week-blocked P+ **1.000**, 95% [+46.428, +53.652], candidate accuracy
100.000%, sitting at the 100.0th percentile of its own null (which itself
centres at +2.713 pts under the leak treatment). The full-profile ridge fit is
not blind to a real effect of meaningful size even with the pace column
embedded in 90 other production features.

**The real screen** (`--mode screen`, artifact
`artifacts/team_style_pace_on_production/20260901T194505Z/results.json`), 746
paired games over 51 weeks, 3 seasons. Production `weak_stack` accuracy on this
paired population is **49.866%** (`baseline_accuracy` 0.49865951742627346) —
the identical baseline figure the illness sibling measured on the same block,
as it must be, since it is the same unmodified production arm on the same
games.

| quantity | value |
|---|---|
| candidate (`weak_stack_team_style_pace`) accuracy | **51.877%** |
| paired candidate-minus-baseline delta | **+2.011 accuracy points** |
| week-blocked 95% CI (primary) | [-0.400, +4.667] |
| week-blocked `probability_positive` | **0.944** |
| season-blocked 95% CI (secondary) | [-2.000, +6.531] |
| season-blocked `probability_positive` | 0.857 |
| percentile of its own permutation null | **97.5th** (null centre -0.259) |

**Home-pick rate: baseline 53.646%, candidate 53.776%** — 0.13 points apart.
Per the `home-tilt-null-artifact` lesson, a paired delta between two arms with
very different home-pick rates carries an offset that the bootstrap does not
remove; these two arms are as close together as any pair measured in this line
of work, so this reading carries essentially none of that artifact. That is
also why the permutation null matters here and it is reported: the observed
+2.011 sits at the **97.5th percentile** of a null built by shuffling settle
margins within week, which preserves each week's realized home-cover rate. The
reading survives its own null, not just its distance from zero.

The season-blocked secondary is reported beside the week-blocked primary and
**never averaged with it**, and it is read with the same caution the sibling
documents give their own 3-season secondaries: with only 3 season blocks the
season-blocked bootstrap has very little combinatorial diversity, so neither
its width nor its P+ 0.857 is a sharper answer than the week-blocked primary.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the candidate,
the only decision rule this project uses — **this measurement favours adding
the pace-mismatch column over the status quo, at P+ 0.944.** This is a
FORCED-PICK pool: 285 cards must be submitted either way, so declining a
candidate that is ~94% likely better is not caution, it is taking the other
side of a 94/6 bet.

The point estimate, **+2.011 accuracy points on top of what is actually
played**, is the **largest positive on-production marginal recorded in this
line of work**, by a factor of about 2.5 over the previous best. **Read**, the
rotation registry (`nfl-ats rotation status`) and the sibling section 7s:

| construct | on-production delta | week-blocked P+ |
|---|---|---|
| graph `off_sack_rate` | -0.935 pts | 0.122 |
| graph `off_rush_epa_per_play` | -0.935 pts | 0.037 |
| graph `def_yards_per_play` | -0.668 pts | 0.189 |
| FluView away elevated | 0.000 pts | 0.403 |
| FluView home elevated | +0.969 pts | 0.792 |
| illness home ≥2 | +0.268 pts | 0.662 |
| illness away active ≥1 | +0.804 pts | 0.908 |
| **team-style pace mismatch** | **+2.011 pts** | **0.944** |

The pattern the illness sibling named — health channels survive stacking,
graph-propagated team statistics do not — now extends in the direction the
`team-quality-is-already-priced` build filter predicts. The three constructs
that failed were all restatements of team quality the production chain already
prices. The three that survived are a health channel, a health channel, and now
a **play-style** channel: the first tendency or tempo column production has
ever been offered (section 1.2, **measured**: none of the 90 production columns
is a pace or play-calling column). And the pace flag is the one candidate in
the set whose functional form — an absolute gap, `|home − away|` — a linear
ridge provably cannot construct from its own inputs. Those are exactly the two
properties the build filter says should predict a surviving marginal, and they
did.

**What this does not settle, and what should happen next.** This run is
**close-graded**, and per the binding "grade the decision at the opener" rule a
close-graded look settles no play/no-play or promotion decision regardless of
sign. The close is the market at its sharpest and systematically understates
pool-relevant edge, so the opener-graded number is the one that would decide a
card change — and it is not yet measured. Nothing in this document authorises a
card change. The honest next step is an **opener-graded confirmation look on a
disjoint window**, which is a new family with its own predeclaration, not a
re-look; this family retains 2 eligible close-pool windows and stays **open**.

**Caveats, after the implication and not instead of it.** The week-blocked
interval's lower bound is -0.400, which per the binding taxonomy is the
expected shape for a real small signal at this evaluator's ~2-point resolution
and is never grounds to close or discount a line of work; the classification
recorded is `unresolved_below_power` and no closing ground is admissible. The
effect is measured on ONE three-season block of 746 games and on ONE
construction of the column. The expanding-threshold deviation (section 2.1)
means this is not the identical quantity the registry's bare-baseline cell
scored, so the +0.229 there and the +2.011 here are two different measurements
of one mechanism, not a replication in the strict sense — though the direction
of the deviation's bias is known and works against the finding, since a noisier
cut attenuates toward the null. The season-blocked secondary's interval reaches
to -2.000, wider on the low side than the week-blocked primary, which is the
low-power block-count artifact described above rather than a contradiction.
And this is one arm on one window with no multiplicity correction — but it is
also the only arm this document declared, named before scoring, with the
sibling `team_style_short_game_identity` explicitly deferred in section 6.2
precisely so it could not be swapped in afterwards.

**Recorded.** One weak-signal entry
(`team_style_pace_mismatch_on_production`, family
`team_style_pace_on_production`, `unresolved_below_power`, reliability 0.489;
the recorder reported `total_signals` 643 immediately after this write, and
other programs are writing concurrently, so the live count is higher) and one
rotation verdict
(`--verdict unresolved`, window [2011, 2013] spent, 2 eligible windows
remaining, family open). Both notes state the window, mode, null centre and
observed percentile, control magnitude, both arms' home-pick rates, the flag's
coverage and firing rate inside the window, the expanding-threshold deviation,
and that this family is **not poolable** with the bare-baseline team-style
battery because the comparator differs.
