# Graph `team_stat` — CFB cross-league replication (predeclaration)

**Status:** predeclared 2026-09-01, BEFORE any outcome sign was computed (one
disclosed exception in section 9). Sections 1–9 are frozen; section 10
(Results) is appended after the look and nothing above it is edited afterwards.

**Owning work package:** WP8. Files: this document,
`src/nfl_ats/graph_team_stat_cfb_feature.py`,
`scripts/graph_team_stat_cfb_replication.py`,
`tests/test_graph_team_stat_cfb_feature.py`,
`artifacts/graph_team_stat_cfb_replication/`.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decisions are expected value. `probability_positive` above 0.5 favours the
candidate; predeclared thresholds govern only what a document may CLAIM, never
which card is played. This experiment is CFB, so it changes no card either way
(section 8).

---

## 1. The question, and why CFB answers it for free

The graph `team_stat` family (`src/nfl_ats/graph_ratings_v2.py`, signed-Katz
propagation of a pregame team-stat differential over the schedule graph)
produced the weak-signal registry's most one-sided family: **measured** this
session by reading `registry/weak_signals.json` — **83** cells carry
`family == "graph_input_screen"` (NFL 2020–2025, opener-graded), of which
**62** have `probability_positive > 0.5`. (The task brief that commissioned
this work package said "82 cells, 63 favour the candidate"; the counted values
are 83 and 62. The 82 is the survivor count after one cell closed at the
screen's own Gate 1, and 63-vs-62 is a boundary-counting difference. The
qualitative point — a family leaning heavily one way — is unchanged.) Its three
strongest cells were then stacked on the actual NFL production chain and all
went the other way:

| NFL cell | on-production delta | week-blocked P+ | source |
|---|---|---|---|
| `off_sack_rate` | −0.935 accuracy pts | 0.122 | **read** `docs/graph_team_stat_on_production.md` §7 |
| `def_yards_per_play` | −0.668 accuracy pts | 0.189 | **reported** `docs/graph_team_stat_def_ypp_on_production.md` §7 (not re-verified this session) |
| `off_rush_epa_per_play` | running in parallel (WP2) | — | — |

Both readings were taken on the same close-graded `[2014, 2016]` NFL window.
The unanswered question is not "is this particular NFL window unlucky" — it is
**whether the graph transform adds anything to a market-residual model at
all**. That question is answerable on college football, where:

- the schedule graph is roughly **five times sparser**: **measured** (this
  session, from `data/processed/cfb_game_features.parquet` and
  `data/processed/game_features_weak_stack_v4.parquet`) CFB 2015 has 126 FBS
  teams / 679 games / 679 distinct opponent pairs, an edge density of 0.0862 of
  all possible pairs; NFL 2015 has 32 teams / 256 REG games / 208 distinct
  unordered pairs, density 0.4194. CFB 2023: 0.0846. NFL 2023: 0.4516.
- the graph is **heavily clustered by conference**: **measured** 72.83% of the
  12,500 rows in the CFB feature table are `conference_game == 1`.
- **no NFL evaluation window is spent.** ROADMAP.md names "CFB-replicated
  mechanisms" as one of three admissible paths forward (**read**
  `ROADMAP.md:611`).

If the transform is doing real opponent-adjustment work, a sparser, more
clustered graph is where it should have the MOST to add over the raw
statistic — the raw stat is least comparable across a fragmented schedule.
If it adds nothing here either, the NFL on-production negatives stop looking
like one unlucky window.

This replication mirrors `scripts/cfb_surface_familiarity_screen.py`
(the `cfb_surface_familiarity_turf_venue_visitor_split` entry in
`registry/weak_signals.json`) in its conventions: `--league cfb`, the XLG-03
clean-core population, a week-blocked bootstrap primary with a season-blocked
secondary, `probability_positive` always reported, and a predeclaration frozen
before any sign is seen.

---

## 2. Population and data (all measured this session)

Source table: `data/processed/cfb_game_features.parquet`, built by
`nfl-ats cfb-build-features` (12,500 rows, seasons 2006–2025, 142 distinct team
names / 137 distinct `home_id` values). No CFBD API credits are spent by this
work package; every read is from that local snapshot.

- **Spreads and settled margins**: `spread_line`, `result`, `ats_margin` are
  non-null for **all 12,500 rows in all 20 seasons** (measured: per-season
  counts of `spread_line`, `result`, `ats_margin` equal the per-season row
  count for every season 2006–2025). There is no missing-market prerequisite
  problem.
- **Scored window** = the XLG-03 clean core,
  `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` = 2012–2019 + 2021–2025,
  reused verbatim, never redeclared. **Measured**: 9,093 rows, all with
  non-null `ats_margin`. 2020 is excluded by that constant (sparse-provider
  regime), and 2006–2011 by it (thin-line regime).
- **Graph build corpus** = the FULL table, 2006–2025. The graph walk-forward is
  leak-safe by construction (section 4), so the pre-2012 seasons are free
  warm-up for the propagation state and are never scored.
- **Team identity**: the graph uses `home_id`/`away_id` (ESPN team ids) rather
  than team NAMES. **Measured**: 5 of 137 `home_id` values map to more than one
  `home_team` string across seasons (program rebrands), while every
  `home_team` string maps to exactly one id — so names would split a rebranded
  program into two graph nodes and ids do not. This is adaptation A1
  (section 5).
- **Neutral sites / FCS opponents**: the CFB feature table is already
  restricted to regular-season FBS-vs-FBS games by `cfb_features.py`, so FCS
  opponents never appear as graph nodes and cannot be added. 327 of 12,500 rows
  are `neutral_site == 1` (measured); they are KEPT — unlike the surface
  replication, nothing here depends on a "home venue" and the benchmark's own
  feature contract carries `neutral_site` as a control.

---

## 3. Cells (three, one per team-stat column) — and the honest column mapping

The CFB feature table's team-state columns are exactly
`nfl_ats.cfb_features.CFB_STATE_METRICS` (**read**
`src/nfl_ats/cfb_features.py:66-76`): `off_epa_per_play`,
`off_success_rate`, `off_explosive_rate`, `off_plays_per_game`,
`def_epa_per_play`, `def_success_rate_allowed`, `def_explosive_rate_allowed`,
`def_plays_per_game_allowed`. Each appears as `home_<m>`, `away_<m>`,
`diff_<m>`; **measured**, each is non-null on 12,473 of 12,500 rows (9,072 of
9,093 in the clean core).

**There is no CFB yards-per-play column, no rush/pass split, and no sack data
at all in this table.** Saying so plainly matters more than forcing a
three-for-three mapping:

| # | NFL screen cell | CFB cell used here | How close, honestly |
|---|---|---|---|
| C1 | `def_yards_per_play` (NFL screen P+ 0.711; on-production −0.668, P+ 0.189) | **`def_epa_per_play`** | Same construct class — defence-side per-play efficiency allowed. Not the same statistic. The NFL screen scored `def_epa_per_play` itself at P+ 0.583, so the NFL side of this pairing is directly quotable. |
| C2 | `off_rush_epa_per_play` (NFL screen P+ 0.828, the family's 3rd-highest cell) | **`off_epa_per_play`** | Same construct class — offence-side EPA per play — but ALL plays, because CFB has no rush/pass split here. The NFL screen scored `off_epa_per_play` itself at P+ 0.659. |
| C3 | `off_sack_rate` (NFL screen P+ 0.695; on-production −0.935, P+ 0.122) | **`off_success_rate`** | **NOT a sack-rate analogue.** CFB has no sack or pressure column, so `off_sack_rate` has no CFB counterpart and this cell does not pretend otherwise. `off_success_rate` is declared as the third-best available team-stat column; its nearest NFL screen sibling is `pbp_off_success_rate` (P+ 0.709). |

All NFL P+ figures above are **read** from `registry/weak_signals.json` this
session. Three cells, declared here, before any CFB sign is computed; no fourth
cell may be added after seeing these three.

---

## 4. The graph column, and the frozen structural config

For each cell the treatment column is the signed-Katz rating differential from
`nfl_ats.graph_ratings_v2.add_graph_ratings_v2_features` with
`edge_signal="team_stat"`, `signal_column=<cell>`, i.e. the
`graph_v2_team_stat_<cell>_katz_diff` column.

The structural configuration is frozen at exactly the values
`scripts/graph_team_stat_screen.py::FROZEN_STRUCTURE` uses on NFL (**read**
that file, lines 82–91), reproduced with no CFB retuning:

```
alpha              = 0.85
half_life_weeks    = 8.0
max_row_l1         = 1.0
prior_weight       = 1.0
min_games          = 16
propagation        = "signed_katz"
injury_beta        = 0.0
offseason_retention = DEFAULT_OFFSEASON_RETENTION (module default, unchanged)
```

`min_games=16` is **reachable on CFB and needs no adaptation**: the gate counts
games seen cumulatively across the whole corpus, and the thinnest CFB season in
the table still carries 294 games over 11 weeks, so only week 1 of the first
build season (2006) goes unrated. **Measured**: with this exact config on
`def_epa_per_play`, the katz-diff column is non-null on 12,459 of 12,500 rows
overall and on **every one of the 9,093 clean-core rows** (per-season non-null
counts equal per-season row counts for 2012–2025 except 2006, which loses its
first 41 rows to the warm-up gate). Build time 4.2 s for the full corpus.

**Leak safety.** `add_graph_ratings_v2_features` assigns every game in week `w`
a rating read from the graph accumulated through week `w−1`, and only folds
week `w`'s own edges in after the whole week is assigned. Week `w`'s stat
differentials are pregame quantities (`cfb_features.py` builds them as
strictly-lagged span-8 EWMs over earlier completed games), so the edge weight
is knowable before kickoff and the propagation is opponent-ADJUSTING rather
than outcome-absorbing. This is asserted by a leakage regression test, not by
this paragraph — see section 7.

---

## 5. Declared adaptations (exactly two, each with its reason)

- **A1 — team identity is `home_id`/`away_id`, not `home_team`/`away_team`.**
  Reason: measured, 5 of 137 ids carry more than one team-name string across
  seasons (rebrands); names would split one program into two graph nodes and
  silently reset its propagated rating. No structural hyperparameter changes.
- **A2 — the graph is built over 2006–2025 and scored only on 2012–2019 +
  2021–2025.** Reason: the walk-forward is leak-safe, so warm-up seasons cost
  nothing and are not an evaluation window; the scored window is the frozen
  XLG-03 clean core, unmodified.

Nothing else is adapted. In particular `min_games` stays at 16 (it is
reachable, section 4) and no hyperparameter is refit on CFB.

---

## 6. Comparator, grade, metric, uncertainty

**Comparator (primary).** The XLG-03 frozen market-residual benchmark arm
against the same arm plus the one graph column — the benchmark's OWN
estimator, not a single-feature model, mirroring how the surface-familiarity
replication scored its feature inside the established CFB instrument:

- `benchmark` = `nfl_ats.cfb_benchmark.fit_cfb_residual_model(training,
  feature_columns=CFB_MODEL_FEATURE_COLUMNS)` — Ridge alpha 10, out-of-time
  residual distribution, min 500 training games, frozen.
- `benchmark_plus_graph` = the same call with
  `feature_columns=CFB_MODEL_FEATURE_COLUMNS + (graph_col,)`. `cfb_benchmark.py`
  explicitly sanctions this extension point (**read** its
  `fit_cfb_residual_model` docstring: "a declared candidate family … may extend
  it without touching the frozen benchmark path").

Note the comparator is already strict: `CFB_MODEL_FEATURE_COLUMNS` CONTAINS
`home_<cell>`, `away_<cell>` and `diff_<cell>`, so the graph is never credited
for what the raw statistic already earned. This is the CFB analogue of the NFL
**on-production** question, which is the one that went negative.

**Secondary comparators**, both reported and both recorded in the notes field,
neither the headline:

- `graph_only − raw_only`: single-feature CFB residual model on the graph
  column minus the same on `diff_<cell>`. The analogue of the NFL bare-baseline
  `graph_team_stat_screen` direction.
- `graph_only − market`: the same graph-only model minus
  `nfl_ats.margin.fit_market_baseline`. The analogue of the NFL
  `graph_input_screen` direction.

**Grade.** The CFB benchmark's own spread: `settle_margin = result −
spread_line`, identical to `ats_margin`, picks made by the production
probability rule `home_cover_probability >= 0.5` and graded with
`nfl_ats.clv.pick_correct` (forced picks, pushes drop out). A secondary
opener-graded read at `spread_open` is also computed — **measured**, 9,076 of
9,093 clean-core rows carry a non-null `spread_open` — and reported beside the
primary, never averaged with it, per the project's "grade the decision at the
opener" rule. `spread_line` remains the PRIMARY here only because it is the
frozen XLG-03 benchmark's own grade and the whole point is commensurability
with that instrument.

**Metric.** Paired candidate-minus-reference forced-pick accuracy, reported in
`accuracy_points` (percentage points), the registry's `--effect-units
accuracy_points`.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, 1,000 samples, seed
20260901: `block="week"` is the PRIMARY (within-week correlation is zero by
owner mandate, so the week block is the honest unit) and `block="season"` is
reported beside it, never averaged in. `probability_positive` is reported for
both; the binary "contains zero" is never the verdict.

**Within-week permutation null.** 200 draws, settle margins shuffled within
each (season, week), reusing the design of
`scripts/graph_team_stat_screen.py::null_distribution`. This null is
deliberately NOT centred on zero — within-week permutation preserves each
week's realised cover rate, so arms with differing home-pick rates have a
non-zero expected null delta — and it is read as the conservative reference
ALONGSIDE the bootstrap, never instead of it.

**Positive control.** `--mode positive-control` replaces the graph column with
the realised `ats_margin`. An instrument that cannot see that leak is blind and
its null result would mean nothing. Run BEFORE the screen.

**Run order, binding.** `--mode null` first, then `--mode positive-control`,
then `--mode screen` exactly once per cell.

**Era split (report only, no extra registry rows).** The clean core spans an
obvious boundary — the 2020 season the benchmark itself excludes. Per the owner
rule "era magnitude, not presence", each cell's delta is also reported
separately for era 1 = 2012–2019 and era 2 = 2021–2025, with magnitudes, and a
weaker era reading is never described as an absence.

---

## 7. Leakage test (release-blocking)

`tests/test_graph_team_stat_cfb_feature.py` must prove, on synthetic
CFB-shaped frames:

1. **Week `w` reads only through `w−1`.** Blanking and violently perturbing a
   future week's `home_<cell>`/`away_<cell>` values leaves every prior week's
   graph column byte-identical, and current-week outcome/stat changes cannot
   change the current week's own ratings.
2. **Join correctness.** The graph column is joined back onto the CFB frame by
   `game_id`, in the caller's original row order, with no row loss, no
   duplication, and no value reassigned to a different `game_id` when the input
   is shuffled.
3. **Identity mapping (A1).** Rebranded programs — one id, two names — remain a
   single graph node, and the id substitution never mutates the caller's frame.
4. **Cell contract.** Only the three declared cells are accepted; an undeclared
   column name is refused rather than silently graphed.

---

## 8. Decision rule

Expected value, as everywhere in this project: `probability_positive` above 0.5
favours the candidate.

**CFB is replication evidence and never by itself changes an NFL card.** No
result from this document promotes, demotes, or edits any NFL feature profile,
`artifacts/active_ats_model.json`, or `CURRENT_PREDICTIONS.md`. What it can do
is change what the NFL evidence MEANS: a CFB reading that lands on the same
side as the NFL on-production negatives makes "one unlucky window" a weaker
reading of those negatives; a CFB reading on the other side makes it a
stronger one. Either way the next NFL action is a separate, separately
predeclared decision.

---

## 9. Recording, and what is disclosed

Every cell is recorded with `nfl-ats weak-signals record`, `--league cfb`,
`--family graph_team_stat_cfb_replication`, `--effect-units accuracy_points`,
`--category onfield`, one entry per cell named
`graph_team_stat_cfb_<cell>`, with `--interval-low/--interval-high` from the
week-blocked bootstrap and `--probability-positive` from the same. Every write
goes through the session's cross-process registry lock.

Classification is `unresolved_below_power` for every cell **unless** a terminal
ground is literally met: `wrong_sign_resolved` requires the WHOLE week-blocked
interval to sit on the wrong side of zero, and `positive_control_bound`
requires the positive control to have proven detection at that effect size and
the effect to be absent. Reliability is not measured for these cells, so
`no_split_half_reliability` is unavailable and is never claimed. If a record
command errors, the verdict is wrong and the cell is reclassified
`unresolved_below_power`.

**Disclosed pre-predeclaration computation.** While establishing feasibility
(section 4), one outcome-touching diagnostic was computed and is disclosed here
rather than concealed: the raw Pearson correlation between the
`def_epa_per_play` katz-diff column and `ats_margin` on the clean core is
**−0.0046** (measured), and its correlation with the raw `diff_def_epa_per_play`
column is **+0.6189** (measured — the graph is materially different from the
raw statistic, which is the premise of the whole comparison). Neither number is
this experiment's outcome metric, which is a paired forced-pick accuracy delta
from a walk-forward fit; a raw correlation of a feature with the target does
not determine the sign of a paired accuracy delta inside a 31-feature ridge.
No accuracy, cover rate, or paired delta was computed before this document was
written.

---

## 10. Results (added after the look, 2026-09-01)

_Nothing above this line was edited after a sign was seen, with one exception
recorded in place: the section 1 figure "82 cells, 63 favour" was corrected to
the measured 83 and 62 after reading the registry directly. That is a count of
the NFL prior, not an outcome of this experiment._

Every number below is **measured** this session. Artifacts live under
`artifacts/graph_team_stat_cfb_replication/<cell>_<mode>/<timestamp>/results.json`.

### 10.1 Instrument checks (run first, both passed)

**Null** (`--mode null`, 200 within-week permutations, close grade), primary
comparison:

| cell | null mean | null sd | null 95% |
|---|---|---|---|
| `def_epa_per_play` | −0.005 pts | 0.140 | [−0.269, +0.280] |
| `off_epa_per_play` | +0.003 pts | 0.227 | [−0.448, +0.392] |
| `off_success_rate` | −0.017 pts | 0.228 | [−0.381, +0.459] |

All three are centred on zero. The two secondary comparisons carry non-zero
null means (+0.067 to +0.991) exactly as the design predicts — within-week
permutation preserves each week's cover rate, so arms with different home-pick
rates have a non-zero expected null delta. That is the artifact this null is
built to expose, not a defect.

**Positive control** (`--mode positive-control`, the graph column replaced by
the realised `ats_margin`): paired delta **+48.405** accuracy points,
week-blocked P+ **1.000**, 95% [+47.374, +49.464], at the **100.0th percentile**
of its own null; opener-graded +44.896 pts. Identical across all three cells by
construction — the leak column is the same regardless of which cell it replaces.
The instrument is not blind to a real effect embedded among 31 benchmark
features. It is **not** thereby proven able to resolve a 0.2-point effect, so
`positive_control_bound` is inadmissible for every cell below.

### 10.2 The screen — primary comparison (`benchmark_plus_graph` − `benchmark`)

199 weeks fitted, 13 seasons, 8,933 graded games (9,093 clean-core rows minus
160 pushes; **measured**).

| cell | delta (close) | week 95% | week P+ | season P+ | delta (opener) | opener P+ | picks moved |
|---|---|---|---|---|---|---|---|
| `def_epa_per_play` | **−0.011 pts** | [−0.287, +0.280] | **0.467** | 0.425 | +0.056 pts | 0.667 | 175 / 9,093 = 1.92% |
| `off_epa_per_play` | **+0.022 pts** | [−0.443, +0.490] | **0.535** | 0.520 | +0.000 pts | 0.474 | 415 / 9,093 = 4.56% |
| `off_success_rate` | **−0.179 pts** | [−0.685, +0.341] | **0.266** | 0.147 | −0.101 pts | 0.333 | 464 / 9,093 = 5.10% |

Reference (benchmark) accuracy is 51.5952% on this paired population at the
close and 51.7199% at the opener. Observed deltas sit at the 47.5th, 50.5th and
24.5th percentiles of their own permutation nulls — the third is a mild negative
lean, not a tail value.

### 10.3 The screen — secondary comparisons (the NFL bare-baseline directions)

`graph_only` − `raw_only` (the `graph_team_stat_screen` direction: graph column
alone versus the raw differential alone, same estimator):

| cell | delta (close) | week 95% | week P+ | season P+ | picks moved |
|---|---|---|---|---|---|
| `def_epa_per_play` | +0.369 pts | [−0.475, +1.189] | 0.798 | 0.770 | 1,547 / 9,093 = 17.01% |
| `off_epa_per_play` | +0.291 pts | [−0.493, +1.141] | 0.765 | 0.702 | 1,561 / 9,093 = 17.17% |
| `off_success_rate` | +0.694 pts | [−0.340, +1.638] | 0.897 | **0.983**, season 95% [+0.067, +1.279] | 2,384 / 9,093 = 26.22% |

`graph_only` − `market`: +1.052 / +0.918 / +0.996 pts, week P+ 0.896 / 0.879 /
0.895. **Read this one with a caveat, stated before the number is used:**
**measured**, the CFB market arm picks home on 92.5% of games, because CFB
spread odds in this table are flat −110/−110 (median both sides, 8,255 of
12,500 rows carrying odds at all), which puts the no-vig home probability at
exactly 0.500 and the `>= 0.5` forced-pick rule on home. That comparator is
therefore close to an always-home arm on CFB and is the weakest of the three
reads. The NFL `graph_input_screen` comparator does not have this property.

### 10.4 Era magnitudes (report only, close grade, primary comparison)

| cell | 2012–2019 (n=5,349) | 2021–2025 (n=3,584) |
|---|---|---|
| `def_epa_per_play` | +0.075 pts, P+ 0.632 | −0.140 pts, P+ 0.168 |
| `off_epa_per_play` | +0.094 pts, P+ 0.638 | −0.084 pts, P+ 0.379 |
| `off_success_rate` | +0.094 pts, P+ 0.662 | −0.586 pts, P+ 0.148 |

Consistent in shape: every cell is mildly positive in the earlier era and mildly
negative in the recent one, with the largest recent-era magnitude on
`off_success_rate`. Per the owner rule, these are magnitudes, not
presence/absence claims; no era is described as showing "no effect".

### What this implies for the decision, before what is wrong with it

**Two different questions got two different answers, and the split is the
finding.**

1. **The NFL bare-baseline direction REPLICATES on CFB.** Opponent-adjusting a
   team statistic through the graph beats using that statistic raw, on all
   three cells, at week-blocked P+ 0.765 / 0.798 / 0.897 — and it does so while
   moving 17–26% of the picks, so this is not a rounding artifact. On EV
   grounds, if the choice is "raw statistic or graph-adjusted statistic as your
   one feature", the graph-adjusted version is the side to take in all three
   cells. That is a real, transferable statement about the transform, made on
   new football at no NFL window cost, and it is the first independent
   corroboration the `graph_input_screen` family has.

2. **The NFL on-production direction does NOT reverse on CFB — but it also is
   not the harm the NFL numbers suggested.** Stacked on a model that already
   carries the raw statistic, the graph column is worth −0.011, +0.022 and
   −0.179 accuracy points. Two of three straddle 0.5 on P+; the third leans
   negative at P+ 0.266. Nothing here favours adding the column on top of a
   model that already has the raw stat, so on EV the NFL decision not to stack
   it stands, now with cross-league support rather than one window's word.

3. **The most decision-relevant number is the WIDTH, not the sign.** The NFL
   on-production readings were −0.935 pts with a week-blocked interval of
   [−2.625, +0.809] and −0.668 pts on 749 games. The CFB readings are ±0.3 to
   ±0.7 pts wide on 8,933 games, roughly **four times tighter**. At that
   resolution the answer is not "the graph hurts"; it is "on top of a model
   that already has the raw statistic, the graph column is worth approximately
   nothing, and the true value is inside roughly ±0.5 accuracy points." The
   NFL −0.935 is comfortably inside the noise of a true effect near zero. So
   the honest reading of the NFL on-production negatives changes: they are
   consistent with a null, not evidence of harm, and no NFL result should be
   described as showing the graph *damages* a production chain.

4. **Why the two directions differ is mechanically visible in the pick-move
   counts.** Alone, the graph column moves 17–26% of picks and helps. Added to
   the benchmark, it moves 1.9–5.1% of picks and does nothing — because the
   benchmark already carries `home_`, `away_` and `diff_` forms of the same
   statistic plus seven other team-state metrics, and the ridge has already
   extracted the opponent-adjustment-shaped part of that information. The graph
   is a better single feature than the raw statistic; it is not additional
   information once a full team-state contract is present.

Now what is wrong with it. The `off_sack_rate` cell — the NFL cell with the
strongest on-production negative — has **no CFB counterpart at all**, so the
single most-informative replication is the one that could not be run;
`off_success_rate` is a declared substitute, not a stand-in for it. The three
cells share one window and overlapping football, so they are correlated
decompositions and are deliberately **not pooled**. The market comparator is
near-always-home on CFB (section 10.3). The primary grade is the close, as the
predeclaration froze it to stay commensurable with XLG-03; the opener secondary
tells the same story (+0.056 / +0.000 / −0.101) so nothing turns on that
choice, but the project's own "grade at the opener" rule means the close-graded
figures should never be quoted alone. And split-half reliability was not
measured for these three CFB traits, so `no_split_half_reliability` is
unavailable as a ground and no closure of any kind is claimed.

### 10.5 Recorded

Three entries, `--league cfb`, `--family graph_team_stat_cfb_replication`,
`--category onfield`, seasons 2012–2025, all
**`unresolved_below_power`** — no interval sits entirely on one side of zero
(so `wrong_sign_resolved` is inadmissible), the positive control proves
detection at +48 points but not at ±0.2 points (so `positive_control_bound` is
inadmissible), and reliability was not measured (so
`no_split_half_reliability` is unavailable):

| name | effect | interval | P+ |
|---|---|---|---|
| `graph_team_stat_cfb_def_epa_per_play` | −0.0112 | [−0.2870, +0.2797] | 0.467 |
| `graph_team_stat_cfb_off_epa_per_play` | +0.0224 | [−0.4430, +0.4901] | 0.535 |
| `graph_team_stat_cfb_off_success_rate` | −0.1791 | [−0.6847, +0.3413] | 0.266 |

Registry went from 621 to 624 signals. The secondary comparisons are carried in
each entry's `notes` rather than as separate rows, because a row per comparison
would triple-count one window.
