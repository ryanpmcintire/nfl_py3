# PBP coaching-trait reliability screen (LEAD-26 / LEAD-27 / LEAD-30)

Lane J of the 2026-09-05 overnight fleet. Scope: **reliability only**. This
document predeclares three Phase 12 play-by-play coaching-trait leads whose
ROADMAP rows each say "reliability first, ATS look second," then reports
what was measured. **No ATS window is run anywhere in this document or its
code.** That is a later lane's job, gated on whatever this file measures.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Verdicts flow only through `nfl-ats weak-signals record`, never
through prose.

Consistent with that taxonomy, **every signal recorded from this screen
uses `classification unresolved_below_power`**, regardless of the measured
value. This screen measures reliability; it does not adjudicate whether a
low reliability would ever earn the `no_split_half_reliability` closing
ground. That is deliberately left to a separate, dedicated decision, not
bundled into a first measurement pass.

## Scope and what "reliable" means here

For each trait, "reliable" means **split-half correlation on the
team-season unit**, measured two independent ways:

1. **Within-season, odd/even week.** A team-season's games are split by
   week parity; the trait's mean value in odd weeks is correlated (Pearson
   and Spearman) against its mean value in even weeks, across all
   team-seasons. Spearman-Brown corrects the resulting half-length
   correlation up to a full-season-length reliability estimate
   (`(2r)/(1+r)`).
2. **Season-to-season, same franchise.** A team's season-*t* value is
   correlated against its own season-*t+1* value (same on-file team code;
   `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` folds OAK/LV, SD/LAC, and
   STL-SL/LA into one continuous franchise, matching the alias convention
   `nfl_ats.pbp.build_pbp_team_game_metrics` already applies elsewhere in
   this codebase). No Spearman-Brown step-up applies here (it is not a
   half-length split); the raw season-to-season Pearson/Spearman r is the
   reliability figure.

Both methods use a **season-blocked bootstrap** (2,000 draws, fixed seed
`20260905`, one seed per metric so nothing is re-tuned): whole seasons are
resampled with replacement, not individual team-season rows, because
team-seasons sharing a season are not independent draws -- they share the
rule year, the ball, the officiating crop. `probability_positive` is the
fraction of the 2,000 draws with correlation above zero.

Both methods also run a **null**: which team's "even-week" (or "season
*t+1*") value is paired with which team's "odd-week" (or season-*t*) value
is shuffled *within* each season block, 2,000 times, and the resulting
correlation distribution's mean is reported. This preserves each season's
own value distribution but destroys the true team-level pairing, so a
sound reliability estimator should center this null near zero -- and, as
reported below, three of the four metrics do; one does not, which is
itself a reported finding, not a discarded one.

Engine: `nfl_ats.pbp_coaching_traits.paired_split_half_reliability` (generic
over any team/season/value_a/value_b/block_season pairing);
`compute_trait_reliability` runs both methods for one metric;
`run_all_trait_reliabilities` runs all four. Builders, reliability engine,
and null are pure functions, unit-tested in
`tests/test_pbp_coaching_traits.py` (18 tests: each builder on a synthetic
PBP frame, the LEAD-30 opportunity filter's exact boundaries, a leakage
regression test per rolling builder, and split-half math on a known frame
including a perfect-correlation case and a real-vs-null-shuffle case).

Data: the newest `data/pbp/raw/*/` snapshot
(`data/pbp/raw/20260817T184927Z/`, seasons 2009-2025, regular season only
via `nfl_ats.pbp.load_pbp_snapshot`, 781,712 REG plays -- **measured**,
`scripts/pbp_trait_reliability_screen.py` stdout).

## Population caveat shared by all three traits

Every trait here excludes some rare edge case where a team-game has zero
qualifying plays (e.g. a fully-penalty opening "drive," or a game somehow
missing a recorded Q3/Q4 first play). These drops are small and are not a
selection-on-outcome concern -- they are a data-completeness floor, not a
filter chosen after seeing results. Counts of surviving team-games/team-
seasons are reported with every number below (AGENTS.md: never state a
constraint without citing evidence, and always report the denominator).

---

## LEAD-26: scripted-drive efficiency (opening-drive TD rate, opening-drive EPA/play)

**Predeclaration (frozen before scoring).** "Opening drive" = a team's own
minimum `fixed_drive` id in a game. `fixed_drive` is a whole-game,
alternating counter in nflverse's schema -- **read**,
`data/pbp/raw/20260817T184927Z/season=2009/plays.parquet`, game
`2009_01_BUF_NE`: NE's opening possession is `fixed_drive == 1`, BUF's is
`fixed_drive == 2`. Built on `nfl_ats.pbp.build_drive_table`, which applies
the v1 analysis-play filter (real pass/rush snaps with valid EPA/WP, no
kneels/spikes/aborted plays) -- the same convention already used for every
other `PBP_STATE_METRICS` EPA/play quantity in production
(`pbp_off_epa_per_play`). Two metrics:

- `opening_drive_td_rate` -- team-season share of games whose opening drive
  ended in a `fixed_drive_result` of exactly `"Touchdown"`.
- `opening_drive_epa_per_play` -- team-season **play-weighted** EPA/play on
  the opening drive (sum of opening-drive EPA over the season, divided by
  sum of opening-drive plays over the season -- not a mean of per-game
  ratios, which would over-weight short drives).

Predeclared direction for the later ATS look (already in ROADMAP.md, cited
here, not re-decided): **BACK elite-script teams.**

Builders: `build_opening_drive_team_games`, `build_opening_drive_team_seasons`,
`build_opening_drive_rolling` (rolling team-week, strictly-prior games only).

**Results (measured, `scripts/pbp_trait_reliability_screen.py`, artifact
`artifacts/pbp_trait_reliability/20260905T040350Z/results.json`).**
544 team-seasons, 2009-2025.

| metric | method | n_units | n_seasons | Pearson r | 95% CI | P+ | Spearman ρ | Spearman-Brown | null mean r | null SD |
|---|---|---|---|---|---|---|---|---|---|---|
| `opening_drive_td_rate` | within-season odd/even | 544 | 17 | +0.1878 | [+0.1310, +0.2391] | 1.0000 | +0.1762 | **+0.3162** | +0.0152 | 0.0414 |
| `opening_drive_td_rate` | season-to-season | 512 | 16 | +0.1614 | [+0.0909, +0.2306] | 1.0000 | +0.1364 | n/a | +0.0031 | 0.0429 |
| `opening_drive_epa_per_play` | within-season odd/even | 544 | 17 | +0.1733 | [+0.0770, +0.2705] | 1.0000 | +0.1547 | **+0.2954** | +0.0001 | 0.0425 |
| `opening_drive_epa_per_play` | season-to-season | 512 | 16 | +0.1723 | [+0.0946, +0.2493] | 1.0000 | +0.1733 | n/a | -0.0102 | 0.0451 |

Both metrics: real correlation clearly positive both ways, null centers at
essentially zero (0.0001 to 0.0152, well inside the null's own ~0.04 SD) --
the estimator behaves as expected and the reliability reads clean.

---

## LEAD-27: third-quarter adjustments (Q3 point differential)

**Predeclaration (frozen before scoring).** `q3_point_diff` = a team's own
points scored in the third quarter minus its opponent's, for one game.
Derived from `score_differential` (posteam's score minus defteam's score,
nflverse's own authoritative running gap -- correct for defensive/special-
teams scores too, unlike a play-type-specific tally) read at the **first
play of Q3** and the **first play of Q4**, converted to a fixed home-team
perspective (`home_lead_pre = score_differential` if posteam is home, else
its negation) so both the entering-Q3 and entering-Q4 states are on the
same scale, then split back out per team with the correct sign. A game
lacking any recorded Q3 or Q4 play is dropped (did not occur in this
dataset in practice).

Predeclared direction for the later ATS look (already in ROADMAP.md):
**BACK persistent 3Q winners full-game.** The trait under test here is its
*persistence*, i.e. this reliability measurement, not any single game's Q3
result.

Builders: `build_third_quarter_point_diff_team_games`,
`build_third_quarter_team_seasons` (team-season mean), `build_third_quarter_rolling`
(rolling team-week, strictly-prior games only).

**Results (measured).** 544 team-seasons, 2009-2025.

| metric | method | n_units | n_seasons | Pearson r | 95% CI | P+ | Spearman ρ | Spearman-Brown | null mean r | null SD |
|---|---|---|---|---|---|---|---|---|---|---|
| `q3_point_diff` | within-season odd/even | 544 | 17 | +0.2385 | [+0.1746, +0.2964] | 1.0000 | +0.2574 | **+0.3851** | +0.0016 | 0.0435 |
| `q3_point_diff` | season-to-season | 512 | 16 | +0.2281 | [+0.1408, +0.3172] | 1.0000 | +0.2169 | n/a | -0.0015 | 0.0453 |

This is the **strongest** of the four metrics on the raw within-season
split (r=+0.2385, Spearman-Brown +0.3851) and the null again centers at
essentially zero. Third-quarter point differential persists across
odd/even weeks and across adjacent seasons for the same franchise more
than either opening-drive metric does.

---

## LEAD-30: fourth-down aggressiveness (go-for-it rate)

**Predeclaration (frozen before scoring; exact text from the task brief).**
4th down, `1 <= ydstogo <= 3`, `yardline_100` between 30 and 70 inclusive
(outside compressed field-goal range, not so deep that a punt is the only
sane option). `play_type` in `{run, pass, punt, field_goal}` is the eligible
opportunity population; `no_play` (penalty-nullified) snaps are **excluded**
because the intended call cannot be recovered from `play_type` alone once a
penalty voids the play. Going for it = `play_type` in `{run, pass}`.
`fourth_down_go_rate` = total go-for-it snaps / total eligible opportunities
per team-season (an opportunity-weighted rate, not a mean of per-game
ratios).

Predeclared direction for the later ATS look (already in ROADMAP.md): **the
interaction** (BACK aggressive underdogs, FADE aggressive favorites) --
"the interaction IS the family, never pooled." This screen measures only
the reliability of the underlying trait (does a team's go-for-it rate
persist), not the interaction itself; the interaction is squarely the later
lane's job once/if this trait clears the reliability bar.

**A real bug was found and fixed before any number was recorded.** The
first pass of `build_fourth_down_opportunities` additionally required
nflverse's `play == 1` indicator, copying that requirement from
`nfl_ats.pbp.analysis_plays` (where it correctly keeps only scrimmage
snaps for EPA aggregation). On the real 2009-2025 data this silently
discarded **every punt and every field-goal attempt** -- nflverse marks
`play == 0` for kicking plays by convention -- leaving only "go" outcomes in
the eligible population. First real-data run (**measured**, artifact
`artifacts/pbp_trait_reliability/20260905T040022Z/results.json`, the
pre-fix run, kept on disk for the record):
`go_for_it` was constant at 1.0 across all 467 within-season units
(`ConstantInputWarning`, correlation undefined, `pearson_probability_positive
= 0.0000` because every bootstrap draw of a constant array is NaN, not
because of a genuine negative finding). This is a data-pipeline defect
caught by an impossible output (a "100% aggressiveness" rate is not
football), not a post-hoc redefinition chosen because of a disliked
result -- the fix (drop the `play == 1` gate; `no_play` is already excluded
via the `play_type` allow-list) restores exactly the definition that was
predeclared in prose above. A direct regression test for this bug
(`play=0` on punt/field-goal rows, matching real nflverse data) is now in
`tests/test_pbp_coaching_traits.py::test_fourth_down_opportunity_filter_boundaries`.

Builders: `build_fourth_down_opportunities`, `build_fourth_down_team_games`,
`build_fourth_down_team_seasons`, `build_fourth_down_rolling` (rolling
team-week, strictly-prior opportunities only).

**Results after the fix (measured, artifact
`artifacts/pbp_trait_reliability/20260905T040350Z/results.json`).** 544
team-seasons; 543 usable for the within-season split (one team-season had
too few opportunities in one half to compute a mean under `min_per_half=1`);
2009-2025.

| metric | method | n_units | n_seasons | Pearson r | 95% CI | P+ | Spearman ρ | Spearman-Brown | null mean r | null SD |
|---|---|---|---|---|---|---|---|---|---|---|
| `fourth_down_go_rate` | within-season odd/even | 543 | 17 | +0.3145 | [+0.2074, +0.3828] | 1.0000 | +0.3013 | **+0.4785** | **+0.2059** | 0.0348 |
| `fourth_down_go_rate` | season-to-season | 512 | 16 | +0.4124 | [+0.2760, +0.5012] | 1.0000 | +0.4175 | n/a | **+0.3096** | 0.0304 |

**This is the strongest raw reliability of the four metrics, but its null
does NOT center near zero** -- unlike all three other metrics (whose nulls
sit at 0.0001 to 0.0152), `fourth_down_go_rate`'s label-shuffle null sits at
+0.2059 (within-season) and +0.3096 (season-to-season), 5-6 SDs above its
own null distribution's center of mass for the other traits. **Measured**
interpretation, stated plainly: shuffling which team's value pairs with
which team's value, *within* a season, cannot destroy a trend that runs
*across* seasons -- and fourth-down aggressiveness has a well-known
leaguewide secular increase over 2009-2025 as analytics-driven decision-
making became mainstream. Any two teams' rates in, say, 2023 both sit
elevated relative to 2010 simply because they are both from 2023, which
inflates the pooled-across-seasons correlation even after within-season
team-label shuffling. The raw r (+0.3145 / +0.4124) is a real, positive,
measured number and is NOT discarded or down-weighted by this observation
(an inflated-vs-null point estimate is not "contains zero" and is not one
of the two admissible closing grounds) -- but a reader should treat the
**excess over the null** (roughly +0.31 - 0.21 ≈ +0.11 within-season; +0.41
- 0.31 ≈ +0.10 season-to-season) as the more conservative read of
team-specific (as opposed to era-wide) persistence. This is exactly the
kind of nuance the later ATS lane should carry forward, not a reason to
treat the trait differently in this recording pass.

---

## Which traits earn an ATS look

Stated plainly, before caveats, per the task's own bar ("any non-zero
reliability with P+ > 0.5 earns a look; the later lane decides"): **all
four recorded metrics earn an ATS look.** Every one of the eight
method x metric combinations above (4 metrics x 2 split methods) measured
`probability_positive = 1.0000` with the whole 95% season-blocked bootstrap
interval on the positive side of zero. None of the three ROADMAP rows
(LEAD-26, LEAD-27, LEAD-30) is being flipped to ✅ by this file -- their own
definition of done explicitly includes the ATS look, which this lane does
not run -- but the reliability gate each row names is now measured, not
open, and it came back positive for every metric.

Caveat, restated for visibility: `fourth_down_go_rate`'s reliability is
real but partly conflates team persistence with a leaguewide era trend
(see above); the other three metrics' nulls behave as expected.

## Recorded signals

Each entry below was written via `nfl-ats weak-signals record` with
`--classification unresolved_below_power` (this screen adjudicates nothing;
see the taxonomy section above) and `--family pbp_coaching_traits`. Per
entry: `--effect`/`--interval-low`/`--interval-high`/`--probability-positive`
are the **within-season odd/even** Pearson r and its season-blocked
bootstrap CI (the headline split-half method); `--reliability` is that
same split's Spearman-Brown full-length correction; `--sample-games` is the
count of team-seasons used in the within-season split; `--sample-blocks` is
the count of distinct seasons. The season-to-season figures and both nulls
are reported above and in the source artifact, not duplicated into the
registry's single-effect schema.

| weak-signals name | effect (raw within-season r) | 95% CI | P+ | reliability (Spearman-Brown) | sample_games | sample_blocks |
|---|---|---|---|---|---|---|
| `pbp_trait_opening_drive_td_rate_reliability` | +0.1878 | [+0.1310, +0.2391] | 1.0000 | +0.3162 | 544 | 17 |
| `pbp_trait_opening_drive_epa_reliability` | +0.1733 | [+0.0770, +0.2705] | 1.0000 | +0.2954 | 544 | 17 |
| `pbp_trait_q3_point_diff_reliability` | +0.2385 | [+0.1746, +0.2964] | 1.0000 | +0.3851 | 544 | 17 |
| `pbp_trait_fourth_down_go_rate_reliability` | +0.3145 | [+0.2074, +0.3828] | 1.0000 | +0.4785 | 543 | 17 |

Source for all four: `artifacts/pbp_trait_reliability/20260905T040350Z/results.json`
(post-fix run); the exact `nfl-ats weak-signals record` commands run are
reproduced in the lane report.

## Files

- `src/nfl_ats/pbp_coaching_traits.py` -- builders + reliability engine
  (pure functions).
- `scripts/pbp_trait_reliability_screen.py` -- orchestration; writes
  `artifacts/pbp_trait_reliability/<UTC stamp>/results.json` via
  `write_experiment_artifact`; records nothing itself.
- `tests/test_pbp_coaching_traits.py` -- 18 tests: each builder on a
  synthetic PBP frame, the LEAD-30 opportunity filter's exact boundaries
  (including the `play=0` regression case for the bug above), a leakage
  regression test per rolling builder, and split-half math on a known
  frame (perfect correlation; real-signal-vs-null-shuffle).
