# Subreddit fan-attention cells, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/illness_on_production.md`,
`docs/graph_team_stat_def_ypp_on_production.md` and
`docs/fluview_on_production.md`. **Sections 1-6 are the predeclaration** and
contain no accuracy, cover-rate or `probability_positive` number against NFL
outcomes from this comparison. **Section 7 is added after the look** and
reports what it found; it changes nothing above it.

Ranked #2 of four in `docs/on_production_sweep_20260901.md` section 1.3.

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

## 1. What this closes: the attention channel has never touched production

`docs/arctic_shift_ats_battery.md` predeclared and froze five cells and scored
them against a **bare market baseline** — a subset cover-rate gap scaled to the
full slate. **Read**, `registry/weak_signals.json` (all five, verified this
session):

| cell | effect (accuracy points) | week-blocked 95% | `probability_positive` | reliability | n_population | n_flag |
|---|---|---|---|---|---|---|
| `reddit_home_comment_ratio_elevated` | +0.329 | [-0.207, +0.883] | **0.885** | 0.992 (ratio) | 3365 | 357 |
| `reddit_away_spike_value` | +0.214 | [-0.224, +0.644] | **0.832** | 0.992 (volume) | 3412 | 225 |
| `reddit_away_comment_ratio_elevated` | +0.186 | [-0.330, +0.696] | 0.761 | 0.992 (ratio) | 3347 | 294 |
| `reddit_home_spike_fade` | -0.058 | [-0.486, +0.367] | 0.394 | 0.992 (volume) | 3428 | 217 |
| `reddit_spike_gap_home_worse` | -1.346 | [-6.206, +3.446] | 0.297 | 0.992 (volume) | 376 | 182 |

Three of the five lean positive (0.885 / 0.832 / 0.761). The leader's
season-blocked secondary is **95% [+0.078, +0.603], P+ 0.996** (**read**, its
registry `notes`), and `n_excluded_missing` is 952 for that cell — both facts
carried forward into section 6 rather than left in a footnote.

**Why this look and not another battery cell.** The project's recorded lesson —
"composition is not the signal" (AGENTS.md, ROADMAP.md) — is that a component
positive alone can go negative once stacked on the chain that is actually
PLAYED, because the played chain already explains some of the variance a
bare-baseline comparison credits to the candidate. This document asks the
marginal question that decides: does a subreddit-attention indicator add
anything on top of the full PRODUCTION chain (**read**,
`artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`,
`model_id: "d1f07d773475dc58"`)?

**The attention channel has never been stacked on production in any form.**
**Read**, `docs/on_production_sweep_20260901.md` section 1.1: the four
on-production tests that existed before this sweep came from exactly two
channels — graph-propagated team statistics (`weak_stack_graph_sack`,
`_graph_def_ypp`, `_graph_off_rush_epa`) and CDC regional influenza-like
illness (`weak_stack_fluview_home`/`_away`). Neither Reddit nor GDELT nor
Wikipedia pageviews has ever been a column in a production-stacked profile.
**Measured**, `src/nfl_ats/constants.py` `FEATURE_SETS["full_weak_stack"]`: 90
columns, none of them an attention series. So this is not a re-run of anything;
it is the first time the channel meets the model that is actually played.

**Distinct from the closed shared-variance gate.**
`docs/arctic_shift_gate.md` failed its shared-variance leg against Wikipedia
pageviews (pooled log-scale r=0.7319 on RAW VOLUME) and explicitly left the ATS
question open: "Correlation here measures construct overlap, not predictive
value; an ATS battery remains a separate decision." Per AGENTS.md, construct
overlap is not an admissible closing ground. One of the two arms below is a
**comment-to-post ratio**, which the gate never measured at all: it is
scale-free in volume by construction, so the overlap finding does not reach it.

**This sweep is not a foregone negative.** **Reported** (unverified by me;
orchestrator-measured this session, artifact
`artifacts/illness_on_production/20260901T192918Z/results.json`): sibling #1 of
this same sweep, `illness_away_active_ge1`, came back **+0.804 accuracy points
at week-blocked P+ 0.908** on production, against the two graph constructs that
preceded it at -0.935 (P+ 0.122) and -0.668 (P+ 0.189). The framing here is
therefore neutral, not defensive.

## 2. The candidate columns, and the two declared deviations from the battery

Two columns, each carried by its own profile, named identically to the
registry cells so the lineage from bare-baseline screen to on-production
stacking is legible from the column name alone:

- **`reddit_home_comment_ratio_elevated`** — the home team's Tuesday-ending
  7-day window comment-to-post `ratio_z >= 2.0`, where the z-score uses a
  trailing baseline over that team's own strictly prior games this season.
- **`reddit_away_spike_value`** — the away team's Tuesday-ending 7-day window
  post+comment `volume_z >= 2.0`, same trailing-baseline construction.

`src/nfl_ats/reddit_attention_production_feature.py` **imports the frozen
construction rather than reimplementing it**: `load_subreddit_daily_counts`,
`build_team_game_long`, `attach_game_level` and `SPIKE_THRESHOLD` are imported
directly from `scripts/arctic_shift_battery_screen.py` (whose
`build_team_game_long` in turn calls its own `window_sum`), which in turn
imports `TRAILING_MIN_GAMES` / `TRAILING_WINDOW_GAMES` from
`scripts/attention_battery_screen.py` and `SUBREDDITS_ALL` from
`scripts/arctic_shift_battery_fetch.py`. The frozen screen script itself is not
edited.

**Point-in-time safety by construction.** The window ends on the **Tuesday of
the game's own week** (`window_end = gameday - ((weekday - 1) mod 7) days`,
`window_start = window_end - 6 days`), so for every game the window closes at
least five days before a Sunday kickoff and never later than the day of a
Tuesday game. The trailing baseline is `shift(1)` before the rolling window and
resets per `(team, season)`, so it never includes the current window. Both
facts are inherited unchanged from the frozen battery and are pinned by a
leakage regression test in both directions
(`tests/test_reddit_attention_production_feature.py`).

**Two deliberate, declared deviations from the frozen battery.**

1. **Missing coverage comes back NaN, not `False`.** The battery folded "this
   team-week has no computable baseline" into the excluded population
   (`has_baseline_*`), because it was scoring a subset cover-rate gap over a
   population it had already restricted. A feature column may not fold it into
   `False`: "no subreddit coverage at all" and "coverage showing no spike" are
   different states, and only the model's own training-fold median
   (`fit_margin_model`) may decide what to do with the first. Rows without a
   computable z therefore come back **NaN**, exactly as
   `nfl_ats.illness_production_feature` and `nfl_ats.fluview_production_feature`
   already treat their own missing as-of coverage. This can only move the
   estimate toward the null relative to the battery's own treatment, never away
   from it.
2. **The baseline sequence is not filtered on the outcome.** The battery's
   `load_games` restricted to REG games with a `spread_line` **and a non-null
   `home_cover`** — i.e. it dropped pushes and unplayed games, an
   outcome-dependent restriction. A pregame feature column must not replicate
   that, so the trailing-baseline sequence here is every REG row of the feature
   table with a `spread_line`, in each team's own chronological order. Non-REG
   (playoff) rows are given no column value (NaN) and are never scored anyway
   (`nfl_ats.modeling.regular_season_rows`). The difference is confined to a
   handful of push games per season shifting position inside an 8-game rolling
   window; it is declared here rather than discovered afterwards.

The feature is additively joined back onto the production feature table by
`game_id` with `validate="one_to_one"` — the same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features` and every
sibling candidate module already established: every pre-existing column comes
back bit-identical, only the two new columns are added.

## 3. The candidate profiles

Two `MarginFeatureProfile`s, each = production `weak_stack`'s exact feature set
plus **exactly one** new column — the same "one new column" shape
`weak_stack_graph_sack`, `weak_stack_fluview_home`/`_away` and
`weak_stack_illness_away`/`_home` use:

| profile | feature set | the one new column |
|---|---|---|
| `weak_stack_reddit_ratio_home` | `full_weak_stack_reddit_ratio_home` | `reddit_home_comment_ratio_elevated` |
| `weak_stack_reddit_spike_away` | `full_weak_stack_reddit_spike_away` | `reddit_away_spike_value` |

**Measured** this session:
`margin_feature_set("market_residual", "weak_stack_reddit_ratio_home")`
resolves to a 91-column set = the 90-column `full_weak_stack` production set
plus exactly `reddit_home_comment_ratio_elevated`, and the away-spike profile
likewise.

Both are built on `data/processed/game_features_weak_stack.parquet` — the
PRODUCTION table — directly, never on
`weak_stack_v3`/`_surface`/`_v4`/`_graph_*`/`_fluview`/`_illness`, mirroring
`weak_stack_graph_sack`'s own declared reason verbatim: stacking a candidate
onto a profile already refused or still undecided would confound the answer to
"does this add to what is actually played." Both columns live in the same
widened table (`data/processed/game_features_weak_stack_reddit.parquet`), but
each profile reads only its own. Never referenced by the active model. Never
mixed with any other candidate profile.

## 4. The comparison

**Three arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate A | `weak_stack_reddit_ratio_home` | production + the home comment-ratio column |
| candidate B | `weak_stack_reddit_spike_away` | production + the away volume-spike column |

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

**Grade.** Close-graded, mirroring all three sibling documents. Per the binding
"grade the decision at the opener" rule, nothing here may settle a play/no-play
or promotion call; every recorded classification is `unresolved_below_power`
regardless of sign.

**Window and family.** A new rotation family, `reddit_attention_on_production`,
close-graded, declared with **no `--inherits`**: the Arctic Shift battery has
never held a rotation family at all. **Verified** with `nfl-ats rotation status`
before declaring — the 16 existing families are `best_pick_ranker`,
`best_pick_ranker_opener`, `cfb_role_continuity`, `combined_stacker`,
`era_weighting_half_life_8`, `fluview_elevated_on_production`,
`fluview_home_elevated_opener`, `graph_def_ypp_on_production`,
`graph_off_rush_epa_on_production`, `graph_off_sack_rate_on_production`,
`graph_ratings_v2_team_stat`, `illness_on_production`,
`mod07_weak_signal_stack`, `movement_expansion_v1`, `pbp_drive_bundle` and
`player_qb_continuity`, and none of them is an attention family. Declared
**without** `--acknowledge-mined`, because the deterministic earliest-eligible
close block does not intersect the 2018-2025 mining ledger; if the CLI refuses,
that refusal is the authority and section 7 records what actually happened.

The window is **ASSIGNED by `nfl-ats rotation assign`**, never hand-picked
(**read**, `src/nfl_ats/rotation.py:967-975`, `assign_window`: "the
lowest-starting block of the requested size inside the grade's pool that starts
at or after the warm-up floor ... There is no hidden choice and nothing to
tune"). The assigned block is confirmed in section 7, not asserted here.

## 6. Uncertainty, instrument checks, and the caveats stated up front

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate) and
season-blocked as a secondary read reported beside it, **never averaged with
it**. Same `BOOTSTRAP_SAMPLES`/`SEED` constants the sibling on-production
scripts already use, for comparability.

**Within-week permutation null**, 200 permutations, identical mechanism to the
sibling documents: all three arms' models are fit ONCE per week on the REAL
`ats_margin`; only the grading margin is shuffled within week for the null, so
200 draws cost no extra model fits. This null is **not** centred on zero by
design — it preserves each week's realized home-cover rate, and the arms may
carry different home-pick rates (the `home-tilt-null-artifact` lesson) — and is
reported ALONGSIDE the bootstrap-vs-zero interval, never instead of it.

**Positive control**, run BEFORE the real screen: each candidate profile's one
new column is temporarily REPLACED by the realized `ats_margin`, a deliberate
large leak, so the harness must show an obvious large effect. This proves the
FULL-PROFILE ridge fit can detect a real effect of meaningful size when one is
actually present. A "no effect" reading from a blind instrument would mean
nothing; this check exists so that possibility is ruled out first. **If the
control does not fire, the screen is not run and section 7 says the instrument
was blind.**

### 6.1 Three caveats, disclosed before any result

**(a) The 0.992 reliability is a subreddit-series figure, not a predictor
figure.** It is the split-half reliability of the raw per-team-game
`window_volume` (and separately `comment_post_ratio`) across a team-season odd/
even week split (`docs/arctic_shift_ats_battery.md` section 6). It is
**plausibly dominated by the stability of fanbase size** — r/eagles is reliably
louder than r/Jaguars in both halves of every season — rather than by any
game-level trait that predicts covering. It must not be read as "a
0.992-reliable predictor". It is recorded as the battery reported it, both
registry entries will carry this caveat in their `--notes`, and this sentence
is the disclosure. What the figure *does* rule out is the one closing ground
that no sample size rescues: `no_split_half_reliability` is unavailable here.

**(b) Thin firing rates, and 952 excluded rows.** **Read**,
`registry/weak_signals.json`: the home-ratio cell fires on 357 of 3,365
population games (**10.61%**) with 952 further games excluded as missing; the
away-spike cell fires on 225 of 3,412 (**6.59%**) with 905 excluded. The
assigned close block holds roughly 750 REG games, so the away arm's ridge
coefficient rests on a few dozen firings. Per the binding taxonomy, a wide
interval from a sparse column is `unresolved_below_power`, never a negative,
and this is disclosed here rather than discovered afterwards.

**(c) Early-archive Reddit volume is real but small.** **Measured** this
session from `data/raw/arctic_shift/*_comments_timeseries_full.json` (all 32
subreddits in `SUBREDDITS_ALL`): **30 of 32** team subreddits carry non-zero
comment volume across 2011-2013, with league totals of **145,963 / 634,771 /
1,895,412** comments in 2011 / 2012 / 2013, against **15,935,082** in 2025 —
roughly a 100x growth from 2011 to 2025. The two teams with zero 2011-2013
comment volume are **LA** (`r/LosAngelesRams`, a subreddit that did not exist
while the franchise was in St. Louis) and **WAS** (`r/Commanders`, created at
the 2022 rename); both have zero posts as well, so their trailing standard
deviation is zero and every one of their rows self-excludes to NaN rather than
being read as "no chatter" — the mechanism section 2's deviation 1 describes.

What that implies for an early assigned window: the metric is a
**within-`(team, season)` trailing z-score**, so it is scale-free with respect
to Reddit's growth — a 2011 spike is measured against that same team's own 2011
baseline, not against 2025 traffic. What early volume does affect is **noise**:
a team-week of 40 comments against a trailing mean of 25 is a much coarser
quantity than 4,000 against 2,500, and small integer counts make the ratio
metric jumpier. So the honest expectation is a **noisier column in an early
window than the pooled 2009-2026 registry read was built on**, with two of 32
teams contributing nothing. That is a disclosed property of the instrument on
this window; it is **not** a reason to reject a result, and per the binding
rules it is never phrased as needing more games. The per-season coverage and
firing rates actually realized inside the assigned window are reported in
section 7.

### 6.2 Measured coverage of the built table (pre-outcome)

**Measured** this session on `data/processed/game_features_weak_stack_reddit.parquet`
(4,902 rows, built by `scripts/build_weak_stack_reddit_table.py` before any
model was fit):

| column | coverage (non-missing) | fires on covered rows |
|---|---|---|
| `reddit_home_comment_ratio_elevated` | 70.685% of 4,902 rows | **10.620%** |
| `reddit_away_spike_value` | 71.807% of 4,902 rows | **6.562%** |

Both firing rates reproduce the registered battery's own figures (10.61% and
6.59%) to within 0.03 points, which is the reproduction check this
reconstruction had to pass. Coverage below 100% is the trailing-baseline
requirement plus playoff rows plus the zero-volume franchises, exactly as
section 2 and 6.1(c) describe.

Per-season REG coverage and firing, also measured before any model was fit:

| season | ratio covered / 256 | ratio fires | spike covered / 256 | spike fires |
|---|---|---|---|---|
| 2009 | **0** | — | **0** | — |
| 2010 | 132 | 12.879% | 175 | 14.857% |
| 2011 | 205 | 10.732% | 209 | 11.962% |
| 2012 | 208 | 13.942% | 211 | 15.166% |
| 2013 | 208 | 15.865% | 213 | 13.146% |
| 2014 | 210 | 14.762% | 214 | 8.879% |
| 2015 | 216 | 11.574% | 216 | 5.556% |

**2009 is empty**: the Arctic Shift daily-count archive carries no usable
2009 volume for any team, so every 2009 row is NaN. Coverage stabilizes at
roughly 80-84% of the slate from 2011 onward — the missing fifth is the two
zero-volume franchises plus each team's first two games of a season, which
structurally have no trailing baseline. If the assigned window includes 2009
that is a material power loss and section 7 says so; if it starts at 2011 or
later it does not arise.

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
`effect_units=accuracy_points`, `category=attention`, family
`reddit_attention_on_production` — a **different pooling bucket** from the
bare-baseline `arctic_shift_battery` family. Stated explicitly and before
scoring: both measure the same underlying subreddit series, but against
**non-commensurable comparators** (a bare market baseline scaled to the full
slate vs. a paired forced-pick accuracy delta on top of the full production
chain), and AGENTS.md's commensurability rule — "pooled inputs must be
commensurable: same units, same scale, same population" — forbids pooling them
together. Both entries carry `reliability=0.992` with caveat (a) spelled out in
`--notes`, and classification `unresolved_below_power` at this close grade
regardless of sign: a resolved wrong sign at this grade is reported as
continuous evidence, never as a `refuted_mechanism` closure, because that
closure is reserved for the opener grade.

ONE `nfl-ats rotation record --name reddit_attention_on_production --verdict
unresolved` call spends the assigned window, carrying the primary arm's paired
effect, interval and `probability_positive`, with both arms' numbers in the
notes. **`reddit_home_comment_ratio_elevated` is named the primary arm before
scoring**, because it is both the higher-`probability_positive` (0.885 vs
0.832) and the higher-firing-rate (10.61% vs 6.59%) arm in the registered
battery.

## 7. Results (added after the look, 2026-09-01)

Rotation family `reddit_attention_on_production` was declared with no
inheritance and **no `--acknowledge-mined`** (the CLI accepted it), then
assigned by `nfl-ats rotation assign` — never hand-picked. The earliest
eligible close-pool block is **[2011, 2013]**, exactly the deterministic block
`docs/on_production_sweep_20260901.md` predicted for every family declared in
this sweep. The window contains no 2009 season, so section 6.2's empty-2009
caveat does not arise. The family retains **2** eligible close-pool windows and
4 eligible stratified seasons after this look.

**Measured** inside the window (768 completed REG games):

| column | non-missing | fires (of covered) | fires (of window) |
|---|---|---|---|
| `reddit_home_comment_ratio_elevated` | 621 / 768 (**80.86%**) | 84 (**13.53%**) | 10.94% |
| `reddit_away_spike_value` | 633 / 768 (**82.42%**) | 85 (**13.43%**) | 11.07% |

Both columns fire **more often** inside this early window than the pooled
2009-2026 registry rates (10.61% and 6.59%) — which is the noise property
section 6.1(c) disclosed in advance, pointing the way it actually pointed:
smaller integer counts in the early archive produce more 2-sigma excursions
against a team's own trailing baseline. It raises the number of firings behind
each coefficient (a power gain) and it makes each firing a noisier event (a
signal loss); both are stated here, neither is being traded off silently.

Both instrument checks ran first, on this same window.

**Null check** (`--mode null`, 200 within-week permutations, real features, not
leaked; artifact
`artifacts/reddit_attention_on_production/20260901T194642Z/results.json`):
`reddit_ratio_home` mean **+0.001** accuracy points, sd 0.460, 95%
[-0.938, +0.938]; `reddit_spike_away` mean **+0.029**, sd 0.693, 95%
[-1.076, +1.478]. Sane, finite, non-degenerate distributions — the harness
produces a null, not a crash or a spike.

**Positive control** (`--mode positive-control`, each candidate's one column
replaced by the realized `ats_margin`; artifact
`artifacts/reddit_attention_on_production/20260901T194709Z/results.json`):
paired delta **+50.134** accuracy points on both arms, week-blocked P+
**1.000**, 95% [+46.428, +53.652], season-blocked [+48.000, +53.878] P+ 1.000,
sitting at the 100.0th percentile of its own null (which itself centres at
+2.713 pts under the leak treatment). The full-profile ridge fit is not blind
to a real effect of meaningful size even with the attention column embedded in
90 other production features. The screen was therefore run.

**The real screen** (`--mode screen`, artifact
`artifacts/reddit_attention_on_production/20260901T194738Z/results.json`), 746
paired games over 51 weeks and 3 seasons. Production `weak_stack` accuracy on
this paired population is **49.866%** (`baseline_accuracy`
0.49865951742627346). The 22-game gap between 768 window games and 746 paired
games is pushes, which `nfl_ats.clv.pick_correct` grades as NaN for every arm
alike.

| arm | candidate accuracy | paired delta | week-blocked 95% | week `probability_positive` | season-blocked 95% | season P+ | percentile of own null |
|---|---|---|---|---|---|---|---|
| `reddit_home_comment_ratio_elevated` | 50.938% | **+1.072 pts** | **[+0.133, +2.125]** | **0.979** | [+0.408, +1.600] | 1.000 | **99.0th** |
| `reddit_away_spike_value` | 50.134% | +0.268 pts | [-1.078, +1.594] | 0.614 | [-0.816, +1.200] | 0.605 | 64.0th |

Home-pick rate: baseline **53.65%**, ratio arm **54.04%**, spike arm
**55.73%**. The primary arm moves the home-pick rate by 0.4 points, so
essentially none of its delta can be the home-tilt artifact that discounted the
bare-baseline attention screens (`home-tilt-null-artifact`); the second arm's
2.1-point tilt is larger and its point estimate should be read with that in
mind.

The season-blocked reading on the primary arm sits entirely above zero, but it
is reported beside the week-blocked primary and **never averaged with it**, and
it is read with the same caution `docs/illness_on_production.md` and
`docs/graph_team_stat_def_ypp_on_production.md` give their own 3-season
secondaries: with only 3 season blocks the season-blocked bootstrap has very
little combinatorial diversity, so a tight interval there is a low-power
artifact of block count, not a sharper answer than the week-blocked primary.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the candidate,
the only decision rule this project uses — **both arms favour adding their
column over the status quo**, and the primary arm does so at **P+ 0.979** with
a week-blocked interval whose whole span sits above zero (+0.133 to +2.125).
This is a FORCED-PICK pool: 285 cards must be submitted either way, so
declining a candidate that is ~98% likely better is not caution, it is taking
the other side of a 98/2 bet.

`reddit_home_comment_ratio_elevated` at **+1.072 accuracy points on top of what
is actually played** is the **largest positive on-production marginal recorded
in this line of work so far**, and the first whose week-blocked interval clears
zero. The five on-production tests that precede it:

| construct | on-production delta | week-blocked P+ |
|---|---|---|
| graph `off_sack_rate` | -0.935 pts | 0.122 |
| graph `def_yards_per_play` | -0.668 pts | 0.189 |
| FluView away elevated | 0.000 pts | 0.403 |
| FluView home elevated | +0.969 pts | 0.792 |
| illness away active ≥1 | +0.804 pts | 0.908 |
| **reddit home comment ratio elevated** | **+1.072 pts** | **0.979** |

The pattern worth naming, extended by this result: the constructs that survive
stacking on production are **health and attention** channels — things the
market's own price does not already encode as team strength — while the two
that did not survive are **graph-propagated team statistics**, i.e.
restatements of team quality the production chain already prices. That is
exactly what the project's own `team-quality-is-already-priced` build filter
predicts, and it now has four positive and two negative observations behind it.
The primary arm also beat its own within-week permutation null (99.0th
percentile against a null centred at +0.001), so the reading is not the
home-pick-rate artifact that discounted the bare-baseline screens.

**The honest next step**: an **opener-graded confirmation look on a disjoint
window** for the home-ratio arm. That is a new rotation family, not a re-look
at this one, and nothing in this document authorises a card change.

**What this does not settle.** This run is **close-graded**, and per the
binding "grade the decision at the opener" rule a close-graded look settles no
play/no-play or promotion decision regardless of sign. The close is the market
at its sharpest and systematically understates pool-relevant edge, so the
opener-graded number is the one that would decide a card change — and it is not
yet measured.

**Caveats, after the implication and not instead of it.** (i) The 0.992
reliability behind both arms is a *subreddit-series* split-half figure,
plausibly dominated by fanbase-size stability rather than a game-level trait;
it must not be read as "a 0.992-reliable predictor", and both registry entries
carry that caveat in their notes. What it does do is make
`no_split_half_reliability` unavailable as a closing ground. (ii) The primary
arm's coefficient rests on 84 firings in 621 covered games; the second arm's on
85 in 633, and its 64.0th-percentile position against its own null says it is
close to indistinguishable from that null — it is the weaker of the two arms
and the predeclaration named it second before scoring. (iii) Two arms were
measured on one window without a multiplicity correction, disclosed here rather
than corrected, since the family was declared with both arms named in advance.
(iv) Two of 32 franchises (LA, WAS) contribute nothing in this window because
their subreddits did not exist yet, so roughly 19% of window games carry NaN in
the column and are imputed by `fit_margin_model`'s own training-fold median.
(v) Only 3 season blocks exist, so the season-blocked secondary is a low-power
read, not a confirmation.

**Recorded.** Neither arm is closed and neither is promoted. Both are recorded
`unresolved_below_power` with no closing ground
(`reddit_home_comment_ratio_elevated_on_production`,
`reddit_away_spike_value_on_production`, family
`reddit_attention_on_production`, category `attention`), the rotation verdict
is `unresolved`, the family stays **open**, and it retains 2 eligible
close-pool windows. Both registry entries state explicitly that this family is
**not poolable** with `arctic_shift_battery`, because the comparator differs.
