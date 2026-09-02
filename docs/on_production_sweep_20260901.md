# On-production sweep, 2026-09-01: which registry constructs have never been
# stacked on what is actually played, and which four are worth the window

**Section 1 is a predeclaration.** It was written before any of the four
experiments it ranks produced a single ATS number against production, and it
contains no accuracy, cover-rate or `probability_positive` figure from any of
them. Sections 2 and 3 were added after the looks and change nothing above
them.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". Decisions are expected value: `probability_positive`
above 0.5 favours the candidate. A predeclared threshold governs only what the
docs may CLAIM, never which card is PLAYED. Grade play/no-play at the OPENER; a
close-graded look settles no play decision and is recorded
`unresolved_below_power` regardless of sign.

## 1. The survey, the exclusions, and the ranked four

### 1.1 What "on production" means and why the sweep exists

The project's recorded lesson — "composition is not the signal"
(AGENTS.md, ROADMAP.md, `.claude` memory `composition-is-not-the-signal`) — is
that a construct positive against a bare market baseline can go negative once
stacked on the chain that is actually PLAYED: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`
(**read**, `artifacts/active_ats_model.json`). Four such on-production tests
exist as of today and all four leaned negative or flat:

| test | doc | reported result |
|---|---|---|
| graph `off_sack_rate` | `docs/graph_team_stat_on_production.md` | -0.935 pts, P+ 0.122 |
| graph `def_yards_per_play` | `docs/graph_team_stat_def_ypp_on_production.md` | -0.668 pts, P+ 0.189 |
| graph `off_rush_epa_per_play` | `docs/graph_team_stat_off_rush_epa_on_production.md` | in flight by another program this session |
| FluView home/away elevated | `docs/fluview_on_production.md` | home +0.969 P+ 0.792; away 0.000 P+ 0.403 |

All four came from the same two channels (graph-propagated team stats; CDC
regional illness). **Read**, `registry/weak_signals.json`: the NFL
`accuracy_points` pile holds **543 entries**, of which **121** are the two
graph screens (`graph_input_screen` 83, `graph_ratings_v2_team_stat` 38). The
remaining ~420 span health, attention, schedule, environment, on-field style,
special teams, officiating, market microstructure and modelling — and almost
none of them have ever been measured as a feature column on top of production.
This sweep asks which of those are worth a rotation window.

### 1.2 Exclusions, and the reason for each

**Already tested on production (a feature column or profile exists):**

| construct | where it was already stacked |
|---|---|
| graph `team_stat` off_sack_rate / def_yards_per_play / off_rush_epa_per_play | `weak_stack_graph_sack` / `_graph_def_ypp` / `_graph_off_rush_epa` |
| FluView home/away market elevated | `weak_stack_fluview_home` / `_fluview_away`; `fluview_home_elevated_opener` |
| **surface switch / surface familiarity** | `weak_stack_surface` (MOD-08). **Read**, `src/nfl_ats/surface_switch_tilt_overlay.py:224`: `surface_switch_flag` = away team's modal home surface is grass AND the venue is turf — the *identical* cell to `surface_familiarity_r1_turf_venue_visitor_split` and `weather_battery_surface_switch_grass_to_turf`. Registered as `surface_switch_feature_arm`, P+ 0.6181. This is the one high-P+ environment construct with league replication (`cfb_surface_familiarity_turf_venue_visitor_split` P+ 0.9156) and it is **already spent** on production. |
| raw forecast weather (temp/wind/precip prob/outdoor) | `weak_stack_v4`, six continuous columns |
| observed weather | `weak_stack_oracle_weather` (positive control only, never promotable) |
| division revenge, sandwich spot, post-blowout letdown/bounce, prior-season penalty rate, thursday-pure, return-trip hangover | `weak_stack_v3` (**read**, `src/nfl_ats/constants.py:411-445`) |
| MOD-06 position-prior shrinkage | `weak_stack_js_prior` |
| overlay-subset cells (`overlay_subset_production_plus_*`, `overlay_leave_one_out_*`, `overlay_composition_reselection`) | measured on the production chain already, as pick-level subsets |

**Excluded by the brief:** `injury_value_lost_gradient` / `_narrowed` (gated by
a predeclaration until the 2026 prospective look lands), Best-Pick ranker
cells, `movement_*` cells (line-movement channels, not features), every
`*_oracle_*` control, `weak_stack_v3`/`v4`/`mod07`/`combined_stacker` variants,
and the CFB league.

**Excluded on data coverage.** A fresh close-graded rotation family with no
inheritance draws the **earliest eligible block**, which is deterministic
(**read**, `src/nfl_ats/rotation.py:967-1006`, `assign_window`: "the
lowest-starting block ... that starts at or after the warm-up floor") and is
**[2011, 2013]** for every family declared here. Any construct whose source
data does not reach back to 2011 is therefore unusable without contriving an
inheritance edge:

- **Referee / penalty-crew tendencies** (`penalty_crew_high_flag_heavy_underdog_opener`
  P+ 0.920 rel 0.370, `penalty_crew_holding_tilt_run_heavy` P+ 0.902,
  `referee_battery_penalty_rate_*` P+ 0.703/0.653 — the single best-looking
  family outside health/attention, and already opener-graded). **Measured**,
  `ls data/raw/officials/`: two snapshots, and the registered seasons are
  **2016-2025**. There is no crew identity for 2011-2013. This is the sweep's
  most painful exclusion and it is a *data* exclusion, not a verdict — the
  family stays open and is the first candidate the moment an officials family
  can legally draw a 2016+ block.

**Excluded on firing rate — the flag is too thin for a ridge coefficient on a
3-season window.** The assigned block holds roughly 750 REG games. A cell that
fires on 1% of the slate contributes ~8 rows and cannot move a 90-feature ridge
fit:

- `wxtot_precip60_top_total` — the highest single `probability_positive` in the
  whole non-graph NFL pile (0.9978, +0.224 pts, 95% [+0.073, +0.374]). **Read**,
  its registry `description`: `n_flag=50` across 2009-2025, i.e. ~1.2% of the
  slate, ~9 games in a 3-season window. Its raw ingredient
  (`forecast_precip_prob_pct`) is already inside `weak_stack_v4`; only the
  triple interaction is new, and it is unmeasurable at this window size. Kept
  open, not closed.
- `environmental_battery_aqi_high_outdoor` — P+ 0.926 but `n_flag=23`.
- `ffc_adp_cellA_highadp_underdog_back_ppr_w14` (P+ 0.942, n=275),
  `sagarin_battery_top_decile_close` (P+ 0.891, n=297),
  `movement_attribution_pop_*` (n=123-257): all too thin, and the movement
  cells are excluded as channels anyway.

**Surveyed and ranked below the cut** (recorded here so the next session does
not re-survey them): the **body-clock**, **travel/rest**, **bye**, **roof** and
**venue-milestone** batteries. **Read**, `registry/weak_signals.json`: every
one of their ~35 NFL `accuracy_points` cells carries `reliability: null`, the
strongest lean in the group is `pt_post_mnf_sunday_era_2009_2017` at P+ 0.884,
and that cell **flips sign across eras** (`pt_post_mnf_sunday_era_2018_2025`
P+ 0.257) — a stability problem, not a power problem. `travel_rest_*` tops out
at P+ 0.753, `bye_overval_fade_full_slate_post2011` at 0.838 (and its own sham
placebo control sits at 0.351, which is the right shape), `roof_battery_*` at
0.829 with `visiting_dome_open_vs_closed` resolved on the wrong side. Also
below the cut: `special_teams_*` (P+ up to 0.955 but reliability 0.065-0.109
and the CFB replication **contradicts** — `special_teams_return_top_quartile`
P+ 0.955 NFL vs `cfb_special_teams_return_top_quartile` P+ 0.032);
`close_game_luck_*` (reliability 0.132-0.163); `ol_continuity_*` and
`txn_*` (majority of cells lean negative); `pbp08_protection_mismatch`
(P+ 0.9785 with an interval that excludes zero and a clean mirror null at
P+ 0.555 — genuinely the best *evidence design* in the pile, but reliability
0.063 is close enough to the `no_split_half_reliability` boundary that an
on-production test is a poor EV bet, and it is already live as a pick-level
tilt overlay); `motivation_ladder_tank_zone_wk14_18` (P+ 0.933, but
`reliability: null` and the cell is restricted to weeks 14-18 among the
bottom-two league-wide records).

### 1.3 The ranked four

Ranked on the brief's own criteria in order: split-half reliability (≥ 0.5
preferred, because AGENTS.md makes an unreliable trait the one thing no sample
size rescues), `probability_positive` on the construct's strongest honest read,
replication, and whether the column can actually be built as a pregame column
on `data/processed/game_features_weak_stack.parquet` (4,902 rows, 2009-2026,
275 columns — **measured** this session) from local data.

| # | construct / new family | channel | split-half reliability | strongest registered read | firing rate | why it is worth the window |
|---|---|---|---|---|---|---|
| 1 | `illness_on_production` | health (team's OWN injury-report illness designations) | **0.702** | `illness_home_ge2` +0.297 pts, P+ **0.890**; `illness_away_active_ge1` +0.307 pts, P+ 0.733 | 7.2% and **23.5%** of the slate | Highest reliability of any construct in the sweep that is not an attention-volume series. Four of the battery's five cells lean positive (0.890 / 0.753 / 0.733 / 0.682). Distinct from FluView — that measures CDC *regional* ILI, this measures the *club's own* illness designations, reconciled at 97.13% against the NFL.com scrape. Production carries no illness feature. Data covers 2009-2025 (**measured**, `data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet`, 90,752 rows). |
| 2 | `reddit_attention_on_production` | attention (fan-forum volume and comment-to-post ratio) | **0.992** (ratio arm; disclosed caveat below) | `reddit_home_comment_ratio_elevated` +0.329 pts, P+ **0.885**; `reddit_away_spike_value` +0.214 pts, P+ 0.832 | 10.6% and 6.6% | Longest span in the registry (2009-2026, n≈3,400). Three of five cells lean positive. The attention channel has **never** been stacked on production in any form — neither Reddit nor GDELT. **Measured** this session: 30 of 32 team subreddits carry non-zero volume in 2011-2013 (146K / 635K / 1.90M comments/yr), and the metric is a within-(team, season) trailing z-score, so it is scale-free with respect to Reddit's growth. |
| 3 | `team_style_pace_on_production` | on-field STYLE (tempo), not quality | **0.489** | `team_style_pace_mismatch_dog_cover` +0.229 pts, P+ 0.711 | **23.6%** | Highest reliability in the team-style battery. The construct is an **absolute gap** (top-quartile \|home − away prior-season centred `seconds_per_play_pace`\|), which a linear ridge **cannot** form from its inputs — and production carries no pace feature at all.[^ts] Directly relevant to the "team quality is already priced" build filter, because tempo is a style axis, not a quality axis. Cache already built (**measured**, `data/pbp/team_style/team_season_style.parquet`, 544 rows, 2009-2025). |
| 4 | `redzone_reversion_on_production` | on-field REVERSION (fade last season's over-performer) | **0.407** | `redzone_reversion_c3_third_down_over_fade` +0.366 pts, P+ **0.872** | ~25% | Second-highest P+ among broad-firing cells with a recorded reliability. The mechanism is mean reversion on a prior-season rate, not a better measurement of current quality, so it is not obviously bounded by the "team quality is already priced" ceiling — though that is the honest risk and it is disclosed in the worker's own §2. Built from the local PBP snapshot; full 2009+ coverage. |

[^ts]: **Correction, appended 2026-09-01 after the look, to a claim this row
    originally made.** Row 3 first read: "`FEATURE_SETS["full_weak_stack"]` has
    90 columns; `pbp_off_pass_rate` and `pbp_off_proe` are in,
    `seconds_per_play_pace` is not." The worker challenged it and the
    orchestrator re-ran the check: **measured**,
    `[c for c in FEATURE_SETS["full_weak_stack"] if "pass_rate" in c or "proe" in c or "pbp" in c]`
    returns `[]`. The 90-column production feature set carries **no**
    play-by-play, play-calling-tendency, or pace column of any kind. The
    original claim was wrong — it described the wider
    `game_features_weak_stack.parquet` *table* (275 columns), not the feature
    *set* the model actually reads. The correction strengthens rather than
    weakens the case for this candidate: the pace-mismatch column is the first
    play-style feature production has ever been offered, not an addition to
    existing tendency features. Recorded here rather than silently edited, per
    AGENTS.md's "verify before quoting" rule.

**Two constructs carry two arms each** (`illness`: home and away;
`reddit`: home ratio and away spike), following `docs/fluview_on_production.md`'s
own precedent exactly — one widened table, two candidate profiles, each holding
**exactly one** new column, both scored against the same production baseline on
the same rotation window. Constructs 3 and 4 carry one arm each. Six candidate
arms in total, four rotation families, four windows.

**Disclosed caveats, before any result:**

- The Reddit reliability of 0.992 is a **subreddit-series** split-half figure.
  It is plausibly dominated by the stability of fanbase size rather than by a
  game-level trait, so it should not be read as "this is a 0.992-reliable
  predictor". It is recorded as the battery reported it, and this sentence is
  the disclosure.
- Every quartile threshold in constructs 3 and 4 is recomputed **expanding over
  strictly prior seasons** rather than over the whole 2009-2025 panel the
  original screens used. The screens' global quantile is a mild look-ahead in
  threshold estimation; a feature column may not carry it, and each worker
  ships a leakage regression test that pins this.
- None of the four families acknowledges the 2018-2025 mining ledger, because
  the deterministic earliest-eligible block, [2011, 2013], does not intersect
  it. The actual assigned block is confirmed per family in each worker's own
  §7 and in section 2 below, never asserted here.
- Each family is a **separate pooling bucket** from the bare-baseline family it
  descends from: the comparator differs (production chain vs. bare market
  baseline) and AGENTS.md's commensurability rule forbids pooling them.

## 2. Results across the four constructs (added after the looks, 2026-09-01)

Every family was declared with no inheritance and had its window **ASSIGNED**
by `nfl-ats rotation assign`, never hand-picked. Because `assign_window` is
deterministic — the lowest-starting eligible block at or above the warm-up
floor — **all four drew the same block, [2011, 2013]**, and therefore scored
the **same 746 paired games over 51 weeks against the same baseline**:
production `weak_stack` at **49.866%** (`baseline_accuracy`
0.49865951742627346, identical to the last digit in all four artifacts —
**measured**, cross-checked across files).

| # | construct / arm | delta (pts) | week-blocked 95% | week P+ | own-null pct | home-pick rate (baseline 53.65%) | reliability |
|---|---|---|---|---|---|---|---|
| 2 | `reddit_home_comment_ratio_elevated` | **+1.072** | **[+0.133, +2.125]** | **0.979** | **99.0th** | 54.04% | 0.992* |
| 3 | `team_style_pace_mismatch_flag` | **+2.011** | [-0.400, +4.667] | **0.944** | **97.5th** | 53.78% | 0.489 |
| 1 | `illness_away_active_ge1` | **+0.804** | [-0.268, +1.914] | **0.908** | 82.5th | 54.95% | 0.702 |
| 4 | `redzone_third_down_over_fade_diff` | +0.670 | [-0.408, +1.854] | 0.849 | 74.5th | 56.12% | 0.407 |
| 2 | `reddit_away_spike_value` | +0.268 | [-1.078, +1.594] | 0.614 | 64.0th | 55.73% | 0.992* |
| 1 | `illness_home_ge2` | +0.268 | [-0.670, +1.230] | 0.662 | 55.5th | 55.60% | 0.702 |

\* a subreddit-*series* split-half figure, plausibly dominated by fanbase-size
stability rather than a game-level trait — see
`docs/reddit_attention_on_production.md` section 6. It rules out
`no_split_half_reliability` as a closing ground; it is not a claim that this is
a 0.992-reliable predictor.

Every construct ran its positive control first and every one returned
**+50.134 points at P+ 1.000** (100.0th percentile of its own leak-treatment
null) — the harness is demonstrably not blind. Every within-week permutation
null was finite and non-degenerate.

**Placed against the full on-production ledger** (**measured**, read back from
`registry/rotation_registry.json`):

| family | window | delta | P+ |
|---|---|---|---|
| `graph_off_sack_rate_on_production` | [2014, 2016] | -0.935 | 0.122 |
| `graph_off_rush_epa_on_production` | [2014, 2016] | -0.935 | 0.037 |
| `graph_def_ypp_on_production` | [2014, 2016] | -0.668 | 0.189 |
| `fluview_elevated_on_production` | [2011, 2025] | 0.000 | 0.403 |
| `redzone_reversion_on_production` | [2011, 2013] | +0.670 | 0.849 |
| `illness_on_production` | [2011, 2013] | +0.804 | 0.908 |
| `reddit_attention_on_production` | [2011, 2013] | +1.072 | 0.979 |
| `team_style_pace_on_production` | [2011, 2013] | +2.011 | 0.944 |

All eight `closing_ground: null`, all `unresolved`, all families open.

## 3. What this implies for the card, before what is wrong with it

**The decision first.** Every one of the six arms has `probability_positive`
above 0.5, and four are above 0.84. On the only decision rule this project uses
— expected value in a forced-pick pool where 285 cards get submitted either way
— each of those four favours its candidate over the status quo, the strongest
at 98/2. Nothing here is a reason to *decline* any of them, and the four
leaders are the strongest set of candidates this line of work has produced.

**The pattern is interpretable and it matches a rule the project already wrote
down.** The three constructs that failed on production are all
graph-propagated restatements of *team quality*, which the
`team-quality-is-already-priced` build filter predicts are bounded near zero
because the played chain already prices them. The constructs that cleared EV
comfortably are an **availability/health** channel, an **attention** channel, a
**play-style/tempo** channel and a **prior-season reversion** channel — and
**measured**, none of those four has any representation in the 90-column
production feature set. That is a usable build filter going forward: *add
channels production cannot see, not better measurements of what it already
prices.*

**Now the thing that must be checked before anyone acts on this, stated plainly
because it is the biggest risk in the whole sweep.** Six of six arms came back
positive. Uniformity that complete is itself evidence that wants explaining,
and two properties of this window make a shared artifact plausible:

1. The production baseline on [2011, 2013] is **49.866% — below a coin flip.**
   A fit sitting at a bad operating point can be improved by perturbations
   carrying no information at all.
2. **Every single arm raised its home-pick rate** relative to the baseline
   (53.65% → 53.78%-56.12%). The project's own `home-tilt-null-artifact` lesson
   is that paired deltas between arms with differing home-pick rates carry an
   offset.

The per-construct instrument checks do **not** close this off. The positive
control proves the harness can SEE a real effect; it says nothing about whether
the harness reports a positive delta for a column that has none. **No negative
control — a signal-free column added the same way on the same window — was run
for any construct in this sweep.** Until one is, the correct reading of these
six numbers is "consistent with four real channel effects, and also consistent
with an additive artifact of this window and this fit."

The within-week permutation nulls are the partial defence that *was* run, and
they are not zero-centred by design, so the **own-null percentile column in
section 2 is the more honest ranking than raw P+**. It separates the six much
more sharply: `reddit_home_comment_ratio_elevated` (99.0th) and
`team_style_pace_mismatch_flag` (97.5th) stand clear of their own nulls;
`illness_away_active_ge1` (82.5th) and `redzone_third_down_over_fade_diff`
(74.5th) are moderately above; the remaining two (64.0th, 55.5th) are close to
indistinguishable from theirs.

**These are not four independent votes.** All four families drew the same
window and scored the same 746 games against the same baseline, so the results
are correlated by construction. No pooled estimate and no "six-of-six agree"
count may be computed across them — AGENTS.md's commensurability rule and the
overlap warning in `weak-signals pool` both apply. Each is recorded in its own
family for exactly this reason.

**Also true, and load-bearing for what to build next:** the two strongest
columns are near-orthogonal to each other. **Measured** on the window's 768 REG
games, `illness_away_active_ge1` vs `team_style_pace_mismatch_flag` Pearson
r = **0.0094** (151 and 182 firings, 37 joint). That makes a combined profile a
well-motivated experiment — but the project's own "composition is not the
signal" lesson is precisely that separately-positive components can go negative
stacked, so additivity is a hypothesis to test, never a sum to assume.

**Nothing here changes the card, and nothing here may.** All four looks are
**close-graded**, and the binding "grade the decision at the opener" rule means
a close-graded look settles no play/no-play or promotion decision regardless of
sign. All six arms are recorded `unresolved_below_power` with no closing
ground; all four families stay open with 2 eligible windows each.

### Recommended order of next work

1. **Run the negative control first.** A seeded signal-free column at matched
   firing rate, added through the identical harness on [2011, 2013], repeated
   over ~12 draws. It is cheap, it gates the interpretation of all six recorded
   results, and until it exists the sweep's headline should not be repeated
   outside this document. (A script for this was drafted and deliberately
   **deleted unrun** when the session budget ran out, rather than left in the
   tree unexecuted.)
2. **Opener-graded confirmation, on disjoint windows, for the two arms that
   clear their own nulls** — `reddit_home_comment_ratio_elevated` and
   `team_style_pace_mismatch_flag`. Each is a NEW family, not a re-look. The
   opener is the grade the pool settles on and the only grade that may move a
   card.
3. **The combined two-column profile**, only after (1) and (2), and predeclared
   as its own family.
4. **Officiating crew**, once `docs/referee_assignments_capture.md`'s capture
   makes a 2016+ window drawable — the one construct this sweep had to exclude
   on data coverage rather than on evidence.
