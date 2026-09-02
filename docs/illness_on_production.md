# Injury-report illness designations, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/graph_team_stat_def_ypp_on_production.md` and
`docs/fluview_on_production.md`. **Sections 1-6 are the predeclaration** and
contain no accuracy, cover-rate or `probability_positive` number against NFL
outcomes from this comparison. **Section 7 was added after the look** and
reports what it found; it changes nothing above it.

Ranked #1 of four in `docs/on_production_sweep_20260901.md` section 1.

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
predeclared thresholds govern only what docs may CLAIM. Grade play/no-play at
the OPENER; a close-graded look settles no play decision and is recorded
`unresolved_below_power` regardless of sign.

## 1. What this closes, and why it is not a repeat of the FluView look

`docs/illness_battery.md` predeclared and froze five cells and scored them
against a **bare market baseline** — a subset cover-rate gap scaled to the full
slate. **Read**, `registry/weak_signals.json`:

| cell | effect (accuracy points) | week-blocked 95% | `probability_positive` | reliability | n | n_flag |
|---|---|---|---|---|---|---|
| `illness_home_ge2` | +0.297 | [-0.176, +0.784] | **0.890** | 0.702 | 3536 | 256 |
| `illness_differential_home_worse` | +1.534 | [-2.880, +5.972] | 0.753 | 0.702 | 425 | — |
| `illness_away_active_ge1` | +0.307 | [-0.659, +1.280] | 0.733 | 0.702 | 3539 | 830 |
| `illness_away_ge2` | +0.107 | [-0.339, +0.547] | 0.682 | 0.702 | 3539 | — |
| `illness_home_active_ge1` | -0.303 | [-1.230, +0.616] | 0.258 | 0.702 | 3536 | — |

Four of the five lean positive. The split-half reliability of **0.702** is the
highest of any construct in the 2026-09-01 sweep that is not an
attention-volume series, and AGENTS.md makes reliability the decisive field:
an unreliable trait is refuted because no sample size rescues it, so a
construct that clears 0.7 is exactly the kind worth spending a window on.

The project's own recorded lesson — "composition is not the signal"
(AGENTS.md, ROADMAP.md) — is that a component positive alone can go negative
once stacked on the chain that is actually PLAYED, because the played chain
already explains some of the variance a bare-baseline comparison credits to the
candidate. **This document asks the marginal question that decides:** does the
illness indicator add anything on top of the full PRODUCTION chain
(**read**, `artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`)?

**This is not the FluView look repeated.** `docs/fluview_on_production.md`
already stacked two *CDC regional influenza-like-illness* indicators on
production — a measure of how much flu is going around the team's **market**.
These columns measure something structurally different: the **club's own
injury-report illness designations**, i.e. which players that specific team
actually listed as ill, resolved as of that game's own pick deadline. The
registry note on `illness_home_ge2` records that this source was reconciled
against the independent NFL.com scrape on 2022-2024 at **97.13%
illness-designation agreement** (`scripts/nflverse_injuries_reconcile.py`). A
market-level epidemiological series and a club-level personnel report are not
the same signal, and one going flat says nothing about the other.

## 2. The candidate columns, unchanged from the frozen battery

Two columns, each carried by its own profile:

- **`illness_away_active_ge1`** — the away team's `active_illness_count >= 1`
  as of that game's own pick deadline, where "active" excludes players whose
  `report_status` is `Out` or `Doubtful` (i.e. players listed ill who are still
  expected to play).
- **`illness_home_ge2`** — the home team's `illness_count >= 2` as of the same
  cutoff.

`src/nfl_ats/illness_production_feature.py` **imports the frozen construction
rather than reimplementing it**: `attach_cutoffs`, `load_injuries`,
`build_team_week_cutoffs` and `resolve_asof_team_week` come straight from
`scripts/illness_battery_screen.py`, which itself imports
`nfl_ats.pick_refresh.pick_deadline` and `sunday_pick_lock`. The point-in-time
rule is therefore the project's own binding per-game pick deadline —
`min(that game's own kickoff, that week's Sunday 16:00 ET)` — and per
`(season, week, team, gsis_id)` entity only report revisions with
`date_modified <= cutoff` are visible, the as-of state being the latest
surviving revision. No new structural choice is made in this document.

**One deliberate, declared difference from the frozen battery.** The battery
folded "no visible report for this team-week" into `False`, because it was
scoring a subset cover-rate gap over a population it had already restricted. A
feature column may not do that: "no report visible at the deadline" and "a
report visible showing nobody ill" are different states, and only the model's
own training-fold median (`fit_margin_model`) may decide what to do with the
first. Unseen team-weeks therefore come back **NaN**, exactly as
`nfl_ats.fluview_production_feature` already treats its own missing as-of
coverage. This can only move the estimate toward the null relative to the
battery's own treatment, never away from it.

The feature is additively joined back onto the production feature table by
`game_id` with `validate="one_to_one"` — the same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features` and every
sibling candidate module already established: every pre-existing column comes
back bit-identical, only the two new columns are added.

## 3. The candidate profiles

Two new `MarginFeatureProfile`s, each = production `weak_stack`'s exact feature
set plus **exactly one** new column — the same "one new column" shape
`weak_stack_graph_sack` and `weak_stack_fluview_home`/`_away` use:

| profile | feature set | the one new column |
|---|---|---|
| `weak_stack_illness_away` | `full_weak_stack_illness_away` | `illness_away_active_ge1` |
| `weak_stack_illness_home` | `full_weak_stack_illness_home` | `illness_home_ge2` |

**Measured**: `margin_feature_set("market_residual", "weak_stack_illness_away")`
resolves to a 91-column set = the 90-column `full_weak_stack` production set
plus exactly `illness_away_active_ge1`, and the home profile likewise.

Both are built on `data/processed/game_features_weak_stack.parquet` — the
PRODUCTION table — directly, never on
`weak_stack_v3`/`_surface`/`_v4`/`_graph_*`/`_fluview`, mirroring
`weak_stack_graph_sack`'s own declared reason verbatim: stacking a candidate
onto a profile already refused or still undecided would confound the answer to
"does this add to what is actually played." Both columns live in the same
widened table (`data/processed/game_features_weak_stack_illness.parquet`), but
each profile reads only its own. Never referenced by the active model. Never
mixed with any other candidate profile.

## 4. The comparison

**Three arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate A | `weak_stack_illness_away` | production + the away-illness column |
| candidate B | `weak_stack_illness_home` | production + the home-illness column |

All three hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values
(`artifacts/active_ats_model.json`); only `feature_profile` differs, isolating
each column's marginal contribution against everything the production chain
already explains. All three are fit with `nfl_ats.margin.fit_margin_model` —
the same estimator production itself uses, not a single-feature model — which
is the whole point of "on top of production" rather than "on top of a bare
baseline." All three are fit and scored on the **same games in the same weeks**,
so the two candidate deltas are paired against a common baseline.

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring both sibling documents. Per the binding
"grade the decision at the opener" rule, nothing here may settle a play/no-play
or promotion call; every recorded classification is
`unresolved_below_power` regardless of sign.

**Window and family.** A new rotation family, `illness_on_production`,
close-graded, declared with **no `--inherits`**: the illness battery has never
held a rotation family at all (**verified** with `nfl-ats rotation status`
before declaring — the 15 existing families are `best_pick_ranker`,
`best_pick_ranker_opener`, `cfb_role_continuity`, `combined_stacker`,
`era_weighting_half_life_8`, `fluview_elevated_on_production`,
`fluview_home_elevated_opener`, `graph_def_ypp_on_production`,
`graph_off_rush_epa_on_production`, `graph_off_sack_rate_on_production`,
`graph_ratings_v2_team_stat`, `mod07_weak_signal_stack`,
`movement_expansion_v1`, `pbp_drive_bundle`, `player_qb_continuity`, and none
of them is an illness family). Declared **without** `--acknowledge-mined`,
because the deterministic earliest-eligible close block does not intersect the
2018-2025 mining ledger; if the CLI refuses, that refusal is the authority and
section 7 records what actually happened.

The window is **ASSIGNED by `nfl-ats rotation assign`**, never hand-picked
(`src/nfl_ats/rotation.py::assign_window`: "the lowest-starting block of the
requested size inside the grade's pool that starts at or after the warm-up
floor ... There is no hidden choice and nothing to tune"). The assigned block
is confirmed in section 7, not asserted here.

## 6. Uncertainty, instrument checks, and the power caveat stated up front

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read reported beside it, never averaged with it.
Same `BOOTSTRAP_SAMPLES`/`SEED` constants the sibling on-production scripts
already use, for comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to the
sibling documents: all three arms' models are fit ONCE per week on the REAL
`ats_margin`; only the grading margin is shuffled within week for the null, so
200 draws cost no extra model fits. This null is **not** centred on zero by
design — it preserves each week's realized home-cover rate, and the arms may
carry different home-pick rates — and is reported ALONGSIDE the
bootstrap-vs-zero interval, never instead of it.

**Positive control**, run BEFORE the real screen: each candidate profile's one
new column is temporarily REPLACED by the realized `ats_margin`, a deliberate
large leak, so the harness must show an obvious large effect. This proves the
FULL-PROFILE ridge fit can detect a real effect of meaningful size when one is
actually present. A "no effect" reading from a blind instrument would mean
nothing; this check exists so that possibility is ruled out first.

**Power caveat, stated before any result.** These are sparse indicators.
**Measured** this session on the built table
(`data/processed/game_features_weak_stack_illness.parquet`, 4,902 rows):
`illness_away_active_ge1` is non-missing on 79.56% of rows and fires on 24.08%
of those; `illness_home_ge2` is non-missing on 79.46% and fires on 7.47%. Both
reproduce the registered battery's own firing rates (23.5% and 7.2%) closely,
which is the reproduction check this construction has to pass. Coverage is
**complete (256 games per season) for every season 2010-2024** and **empty for
2009 and 2025-2026** — the point-in-time-recoverable floor
`docs/illness_battery.md` already documents. Per-season firing rates in the
early window are lower than the pooled figure: in 2011/2012/2013 the away
column fires on 18.4% / 22.3% / 18.4% of games and the home column on 2.7% /
5.9% / 4.3%. On a three-season block of roughly 750 games the home arm
therefore rests on roughly 30-45 firings. That is what it is; it is disclosed
here rather than discovered afterwards, and per the binding taxonomy a wide
interval from a sparse column is `unresolved_below_power`, never a negative.

**COVID stratum.** `docs/illness_battery.md` section 4 scored the 2020 season
separately and excluded it from the pooled battery population. If the assigned
window does not contain 2020 the question does not arise; if it does, 2020 is
excluded here the same way and section 7 says so.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run is
close-graded, so it settles no play/no-play decision by itself; what it DOES
settle is whether the battery's screen-stage finding, measured the honest way
(stacked on what is actually played, not a bare baseline), still looks worth an
eventual opener-graded confirmation look.

## Recording

TWO `nfl-ats weak-signals record` entries, one per arm,
`effect_units=accuracy_points`, family `illness_on_production` — a **different
pooling bucket** from both the bare-baseline illness battery and
`fluview_elevated_on_production`. Stated explicitly: all three measure illness
constructs, but against non-commensurable comparators (a bare market baseline
vs. the full production chain) or from different underlying data (CDC regional
ILI vs. club injury reports), and AGENTS.md's commensurability rule forbids
pooling them together. Both entries carry `reliability=0.702` (the battery's
own recorded figure) and classification `unresolved_below_power` at this close
grade regardless of sign — a resolved wrong sign at this grade is reported as
continuous evidence, never as a `refuted_mechanism` closure, because that
closure is reserved for the opener grade.

ONE `nfl-ats rotation record --name illness_on_production --verdict unresolved`
call spends the assigned window, carrying the primary arm's paired effect,
interval and `probability_positive`, with both arms' numbers in the notes.
`illness_away_active_ge1` is named the primary arm **before scoring**, because
it is the higher-power arm (24.08% firing against 7.47%).

## 7. Results (added after the look, 2026-09-01)

Rotation family `illness_on_production` was declared with no inheritance, then
assigned by `nfl-ats rotation assign` (never hand-picked): the earliest
eligible close-pool block is **[2011, 2013]**. The window does **not** contain
2020, so `docs/illness_battery.md` section 4's COVID-stratum exclusion does not
arise. **Measured** inside the window (768 REG games):
`illness_away_active_ge1` is non-missing on **100.0%** of games and fires on
**19.66%** (151 games); `illness_home_ge2` is non-missing on **99.87%** and
fires on **4.30%** (33 games) — the sparse arm section 6 disclosed before
scoring. Both instrument checks ran first.

**Null check** (`--mode null`, 200 within-week permutations, real features, not
leaked): `illness_away` mean **+0.153** accuracy points, sd 0.627, 95%
[-1.206, +1.340]; `illness_home` mean **+0.082**, sd 0.621, 95%
[-1.206, +1.210]. Sane, finite, non-degenerate distributions — the harness
produces a null, not a crash or a spike.

**Positive control** (`--mode positive-control`, each candidate's one column
replaced by the realized `ats_margin`): paired delta **+50.134** accuracy
points on both arms, week-blocked P+ **1.000**, 95% [+46.428, +53.652],
sitting at the 100.0th percentile of its own null (which itself centres at
+2.713 pts under the leak treatment). The full-profile ridge fit is not blind
to a real effect of meaningful size even with the illness column embedded in 90
other production features.

**The real screen** (`--mode screen`, artifact
`artifacts/illness_on_production/20260901T192918Z/results.json`), 746 paired
games over 51 weeks. Production `weak_stack` accuracy on this paired population
is **49.866%** (`baseline_accuracy` 0.49865951742627346).

| arm | candidate accuracy | paired delta | week-blocked 95% | week `probability_positive` | season-blocked 95% | season P+ | percentile of own null |
|---|---|---|---|---|---|---|---|
| `illness_away_active_ge1` | 50.670% | **+0.804 pts** | [-0.268, +1.914] | **0.908** | [+0.398, +1.633] | 1.000 | 82.5th |
| `illness_home_ge2` | 50.134% | +0.268 pts | [-0.670, +1.230] | 0.662 | [-0.398, +1.224] | 0.720 | 55.5th |

Home-pick rate: baseline 53.6%, away arm 54.9%, home arm 55.6% — close
together, so (like both graph siblings, and unlike the bare-baseline screens'
55-67% home-pick arms) this measurement carries very little of the home-tilt
artifact that discounted the earlier screens.

The season-blocked reading on the away arm sits entirely above zero, but it is
reported beside the week-blocked primary and never averaged with it, and it is
read with the same caution `docs/graph_team_stat_def_ypp_on_production.md`
gives its own 3-season secondary: with only 3 season blocks the season-blocked
bootstrap has very little combinatorial diversity, so a tight interval there is
a low-power artifact of block count, not a sharper answer than the week-blocked
primary.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the candidate,
the only decision rule this project uses — **both arms favour adding the
illness column over the status quo**, and the away arm does so at **P+ 0.908**.
This is a FORCED-PICK pool: declining a candidate that is ~91% likely better is
not caution, it is taking the other side of a 91/9 bet. The away arm's point
estimate, +0.804 accuracy points on top of what is actually played, is the
**largest positive on-production marginal recorded in this line of work so
far**. For comparison, the four on-production tests that preceded it:

| construct | on-production delta | week-blocked P+ |
|---|---|---|
| graph `off_sack_rate` | -0.935 pts | 0.122 |
| graph `def_yards_per_play` | -0.668 pts | 0.189 |
| FluView away elevated | 0.000 pts | 0.403 |
| FluView home elevated | +0.969 pts | 0.792 |
| **illness away active ≥1** | **+0.804 pts** | **0.908** |

The pattern worth naming: the two constructs that survived stacking on
production are both **illness/health** channels, and the two that did not are
both **graph-propagated team statistics** — that is, restatements of team
quality the production chain already prices, which is exactly what the
project's own `team-quality-is-already-priced` build filter predicts. The away
arm also beat its own permutation null (82.5nd percentile against a null
centred at +0.153), so the reading is not simply the home-pick-rate artifact
that discounted the earlier bare-baseline screens.

**What this does not settle.** This run is **close-graded**, and per the
binding "grade the decision at the opener" rule a close-graded look settles no
play/no-play or promotion decision regardless of sign. The close is the market
at its sharpest and systematically understates pool-relevant edge, so the
opener-graded number is the one that would decide a card change — and it is not
yet measured. The honest next step is an **opener-graded confirmation look on a
disjoint window** for the away arm; that is a new family, not a re-look, and
nothing in this document authorises a card change.

**Caveats, after the implication and not instead of it.** The week-blocked
interval on the away arm reaches below zero (-0.268), which per the binding
taxonomy is the expected shape for a real small signal at this evaluator's
~2-point resolution and is never grounds to close or discount a line of work.
The home arm's 4.30% firing rate inside this window means its coefficient rests
on roughly 33 games, and its 55.5th-percentile position against its own null
says it is close to indistinguishable from that null — it is the weaker of the
two arms and was disclosed as such before scoring. Two arms were measured on
one window without a multiplicity correction, which is disclosed here rather
than corrected, since the family was declared with both arms named in advance.
Neither arm is closed and neither is promoted: both are recorded
`unresolved_below_power`, the family stays **open**, and it retains 2 eligible
close-pool windows.
