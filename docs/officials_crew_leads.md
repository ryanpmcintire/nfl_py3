# Officiating-crew leads: LEAD-31/32/33/34 (Phase 12)

**Predeclared 2026-09-05, before any outcome in this family was computed.**
Every construct below is fixed (population, thresholds, seed, sample counts)
before the first number is read. Family names: `officials_home_bias_reliability`
(Stage 1 reliability only), `crew_home_bias_on_production`,
`crew_second_meeting_favorite_on_production`, `rookie_crew_underdog_on_production`
(Stage 2 rotation families).

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
validator. Verdicts flow ONLY through `nfl-ats weak-signals record` and
`nfl-ats rotation record`, never through prose.

## Data sources and reuse (read, not rebuilt)

- **Officials + per-game penalty counts**: `data/raw/officials/20260819T190537Z/officials.parquet`
  (2,896 REG `Referee`-position rows, 2015-2025) and its sibling
  `game_penalties.parquet` (3,028 games, columns `game_id, season, week,
  season_type, home_team, away_team, penalties_total, penalties_on_home,
  penalties_on_away`) -- both already built by `docs/referee_battery.md`
  (2026-08-19). **Measured** (`officials.parquet` schema read this session):
  columns are `game_id, game_key, official_name, position, jersey_number,
  official_id, season, season_type, week` -- no crew-designation or
  all-star marker of any kind. **Measured**: `game_penalties.parquet`'s own
  `game_id` is ALREADY the standard `game_features`-shaped id (e.g.
  `"2015_01_BAL_DEN"`); only `officials.parquet`'s `game_id` is the legacy
  numeric GSIS id, so the crosswalk (`officials.parquet` -> newest
  `schedules.parquet`'s `old_game_id` -> standard `game_id`) is needed only
  to attach `official_name` to a `game_penalties.parquet` row, exactly the
  join `nfl_ats.experiment_runner._build_referee_trait_data` already
  performs. This module's loader reuses that same crosswalk, never
  re-derives it.
- **Penalty YARDS, home/away split -- NOT available locally.** **Measured**
  (`data/pbp/raw/20260817T184927Z/season=2015/plays.parquet`, 45 columns):
  the local trimmed PBP snapshot carries `penalty`/`penalty_yards` but NOT
  `penalty_team` -- confirmed the same gap `docs/referee_battery.md` already
  documented for the ORIGINAL count-based battery. Without `penalty_team`, a
  penalty's yardage cannot be attributed to the home or away side from any
  locally-snapshotted table; `game_penalties.parquet` itself only ever
  persisted counts (`penalties_total/on_home/on_away`), never yards, because
  building it required a fresh (non-persisted) `nflreadpy.load_pbp()` pull
  with `penalty_team` that this session's no-new-network-fetch rule does not
  extend to re-running. **LEAD-32 is therefore built on penalty COUNTS
  only, not yards** -- this is the documented fallback the task's own
  escape hatch names ("if the raw PBP lacks penalty columns, use whatever
  the referee battery used and say so"). Disclosed here, not hidden.
- **Referee tenure / `prior_seasons_experience`**: reused verbatim from
  `nfl_ats.experiment_runner._build_referee_trait_data` (its `game_trait`
  table already carries a per-game `prior_seasons_experience`, a 0-indexed
  count of distinct PRIOR dataset-visible seasons that official appears as
  `Referee` -- the exact quantity `docs/referee_battery.md`'s own
  `referee_rookie_home_cover`/`referee_veteran_home_cover` cells already use
  and reliability-classify `not_applicable` for the same reason stated
  there: a monotonically increasing career-stage counter is not a trait
  that can correlate year-over-year in the split-half sense).
- **Split-half reliability harness**: reused verbatim from
  `nfl_ats.pbp_coaching_traits` (`build_odd_even_halves`,
  `build_season_to_season_pairs`, `paired_split_half_reliability`,
  `_season_blocked_bootstrap`, `_within_season_label_shuffle_null`) -- the
  Wave 4 PBP coaching-trait battery's own generic reliability machinery
  (season-blocked bootstrap, Spearman-Brown correction, within-season
  label-shuffle null), which already implements exactly this lead's
  Stage-1 methodology and is generic over the grouping-column name. This
  module supplies `official_name` renamed to the literal column name
  `"team"` before calling it -- a column-name compatibility shim, not a
  semantic claim that a referee crew is a team.
- **Tuesday-opener consensus spread**: `nfl_ats.schedule_flag_features.default_opener_lines`
  (never the nflverse schedule's own closing `spread_line`) -- the same
  store every sibling on-production candidate grades against.
- **Existing live vehicle**: `nfl_ats.crew_tilt_refresh_overlay` already
  demonstrates the late-week refresh-channel pattern for a crew trait
  (Wednesday-published crew assignment, applied only inside
  `min(kickoff, Sunday 16:00 ET)`, never altering the played pick). Any
  future live wiring of these three Stage-2 candidates follows that same
  pattern; this session builds and screens the candidates, it does not wire
  a new live overlay.

## LEAD-33: late-season all-star crews -- SKIPPED, source gap

`officials.parquet`'s columns (listed above, read directly) carry no
crew-designation, all-star, or Pro-Bowl-selection field of any kind, and no
other locally-snapshotted table in this repo carries one either (`grep`,
this session, found none). Per this task's explicit instruction, NO proxy
is invented (a week-17/18-assignment heuristic was explicitly ruled out by
the task itself as an invented substitute, not a measured marker). LEAD-33
stays `🔬` on `ROADMAP.md` with this gap named in its dated note; it is not
scored this session.

## Stage 1: reliability, no rotation window spent

### LEAD-32: directional home-cooking reliability

**Construct**: per REG game with a matched head referee, `home_minus_away`
= `penalties_on_home - penalties_on_away` (a game-level COUNT
differential, sign flipped from `docs/referee_battery.md`'s own `mean_diff`,
which is `away - home`; correlation is invariant to a simultaneous sign
flip of both halves of a pair, so this is a labelling choice, not a new
measurement). This is the DIRECTIONAL bias construct the task asks to keep
distinct from the battery's already-published RATE-level `mean_total`
cells.

**Reliability, two methods, both via the reused harness**:

1. **Within-season odd/even-week split-half** (the NEW measurement this
   session adds -- not previously computed for this construct): for every
   `(official_name, season)` unit with >= 1 game in odd weeks and >= 1 in
   even weeks, `value_a` = mean `home_minus_away` over odd-numbered weeks
   that season, `value_b` = the same over even-numbered weeks. Population:
   full 2015-2025 (no left-censoring concern -- unlike tenure, a within-season
   split needs no PRIOR-season data, so 2015 is usable here). Pearson r,
   season-blocked bootstrap (2,000 draws, fixed seed `20260905`),
   Spearman-Brown full-length correction, `probability_positive` (fraction
   of bootstrap draws with r > 0), and a within-season official-name-label
   shuffle null (2,000 shuffles) -- exactly `paired_split_half_reliability`'s
   contract.
2. **Season-to-season by referee**: `(official_name, season t)` ->
   `(official_name, season t+1)` pairs of the same per-season mean, run
   through the identical harness (no Spearman-Brown correction, since this
   is already full-length). This reproduces `docs/referee_battery.md`'s own
   `mean_diff` season-to-season Pearson r (-0.101, 158 pairs, read at that
   file's line 90) up to the harness's own numerical method (season-blocked
   bootstrap CI is new; the point estimate should match to floating-point
   precision since only a global sign flip separates the two constructs).

**Stage 2 gate (predeclared)**: LEAD-32 proceeds to Stage 2 only if the
within-season split-half reliability's `probability_positive` (method 1
above) is `> 0.5`. Per this project's EV-decision rule, `P+ > 0.5` favours
treating the trait as real enough to test on production; it is not a
promotion bar.

### LEAD-34: crew-familiarity second meetings (descriptive, reliability not applicable)

**Construct**: for each REG game with a matched referee, `second_meeting`
= `True` when that SAME `official_name` has already officiated, earlier in
the SAME season (by week order), a game involving the home team OR the away
team of the CURRENT game (deterministic boolean flag; no measurement noise
to split in half, hence `reliability: not_applicable`, matching the
existing battery's own treatment of `referee_veteran_home_cover`'s
career-stage counter). Reported: the flag's frequency (count and % of
games with a matched referee) and the mean `penalties_total` (both teams
combined) in flagged vs. non-flagged games -- descriptive only, per the
task's own instruction, not run through a hypothesis-test classifier.

### LEAD-31: rookie-referee tenure (censoring disclosure)

**Construct**: reused `prior_seasons_experience` (0-indexed count of prior
dataset-visible seasons as `Referee`) from
`nfl_ats.experiment_runner._build_referee_trait_data`. **Left-censoring**:
anyone appearing in the dataset's first season (2015) shows
`prior_seasons_experience = 0` regardless of true career length --
indistinguishable from a genuine debut. Disclosed count of censored vs.
genuine debuts, read directly from `officials.parquet`'s own per-official
minimum season. Eligible population for the Stage-2 flag is restricted to
`season >= 2016` (excludes the contaminated all-"rookies" 2015 slate,
matching `docs/referee_battery.md`'s own `referee_rookie_home_cover`
population choice exactly). "Rookie" = `prior_seasons_experience` in
`{0, 1}` (first OR second dataset-visible season, per the task's own flag
definition) within that restricted population; a game with no matched
referee, or with season `< 2016`, defaults to "not rookie" (flag
ineligible), stated as a population restriction, not silently guessed.

## Stage 2: ATS look on PRODUCTION, opener-graded, own rotation family each

All three read the Tuesday-opener consensus line (never the closing line).
Referee crew assignments are published Wednesday-Thursday of game week
(`docs/referee_assignments_capture.md`), i.e. AFTER the Tuesday lock --
so, per that document's own argument and `nfl_ats.crew_tilt_refresh_overlay`'s
existing precedent, these are **late-week REFRESH-channel candidates**,
playable only inside `min(kickoff, Sunday 16:00 ET)`, graded HERE at the
frozen Tuesday line exactly like every other refresh channel this repo has
screened. This session builds and screens the three candidates; it does
not wire a new live refresh overlay.

### `crew_home_bias_on_production` (LEAD-32, conditional on the Stage-1 gate above)

**Column**: `crew_home_bias_flag`, unsigned (`{0.0, 1.0}`), matching
`home_thursday_flag`/`new_stadium_home_flag`'s single-sided shape (this is
a BACK-HOME-ONLY signal, not a home/away comparison -- the same crew
officiates both sides). `1.0` when the home team's assigned crew's TRAILING
within-season home-bias (mean `home_minus_away` over that crew's own PRIOR
games this season ONLY, minimum 3 prior games, else ineligible) sits in the
top quartile of the global population of such trailing values (global
`pd.qcut(4)`, mirroring every other quartile-cut trait in this repo);
`0.0` otherwise, including "not yet 3 prior games this season" and
"referee unmatched". Direction: **BACK the home team** under a
high-trailing-home-bias crew (matches the task's predeclared direction and
`docs/referee_battery.md` cell 5's own mechanism).

**Leakage**: the trailing value for game *k* uses only games strictly
BEFORE *k* in that crew's own within-season order (`expanding().mean().shift(1)`)
-- this game's own penalty count can never reach its own flag, but
correctly CAN change a LATER game's flag in the same crew-season (both
directions tested).

### `crew_second_meeting_favorite_on_production` (LEAD-34)

**Column**: `crew_second_meeting_favorite_flag`, signed. `+1` when
`second_meeting` (defined above) AND the home team is the favorite at the
Tuesday opener (`tue_open_home_spread > 0`); `-1` when `second_meeting` AND
the away team is the favorite (`tue_open_home_spread < 0`); `0` otherwise
(not a second meeting, an exact opener pick'em, or a missing opener line).
Direction: **BACK the favorite** in a crew second-meeting (mirrors
`docs/schedule_flag_battery.md`'s `division_dog`/`week1_dog` dog-sign shape
but inverted to a favorite-sign, since the predeclared mechanism here is
"cleaner/less chaotic games favour the side already expected to win").

### `rookie_crew_underdog_on_production` (LEAD-31)

**Column**: `rookie_crew_underdog_flag`, signed. `+1` when the assigned
crew is a rookie crew (defined above, `season >= 2016` population) AND the
home team is the underdog at the Tuesday opener (`tue_open_home_spread <
0`); `-1` when rookie crew AND the away team is the underdog
(`tue_open_home_spread > 0`); `0` otherwise. Direction: **take the
UNDERDOG** under a rookie crew (chaos/variance mechanism, matching the
task's and `ROADMAP.md` LEAD-31's own predeclared direction exactly).

## Recording plan

Stage 1's `officials_home_bias_reliability` (LEAD-32's within-season
split-half `probability_positive`, effect units `correlation`) is recorded
via `nfl-ats weak-signals record --family officials_crew_traits --category
onfield`. LEAD-34's descriptive frequency/gap is NOT a weak-signal record
(no hypothesis test; the task calls it descriptive only) -- reported in
this doc's measured-results section and the ROADMAP note instead. Each
Stage-2 candidate that runs gets its own `nfl-ats rotation
declare/assign/record` family (opener grade) plus a paired
`nfl-ats weak-signals record --classification unresolved_below_power
--category onfield` regardless of which way the sign comes out, per the
taxonomy above -- an interval crossing zero is never grounds to skip
recording.

## Measured results

### LEAD-32 Stage 1: reliability (measured, `nfl_ats.officials_flag_features.officials_home_bias_reliability`)

Population: 2,892 matched REG games, 29 distinct referees, seasons
2015-2025 (11 seasons).

| Method | n units | Pearson r | 95% CI (season-blocked bootstrap, 2,000 draws, seed 20260905/20260906) | P+ | Spearman-Brown | Label-shuffle null (mean, sd) |
|---|---|---|---|---|---|---|
| Within-season odd/even-week split-half | 187 (official, season) pairs | -0.0518 | [-0.2126, +0.1108] | **0.325** | -0.1092 | -0.0125, 0.0744 |
| Season-to-season by referee | 158 pairs | -0.1015 | [-0.2388, +0.0513] | 0.085 | n/a (already full-length) | +0.0056, 0.0749 |

The season-to-season figure (-0.1015, 158 pairs) reproduces
`docs/referee_battery.md`'s own `mean_diff` season-to-season Pearson r
(-0.101, same 158 pairs) to 3 decimal places, as expected from the global
sign flip -- confirms the loader/crosswalk matches the existing battery's.

**Stage-2 gate outcome**: within-season split-half `probability_positive`
= **0.325**, below the predeclared `> 0.5` escalation threshold. Per this
doc's OWN pre-committed rule (fixed before this number was computed),
`crew_home_bias_on_production` (LEAD-32 Stage 2) is **NOT run** this
session. This is an engineering resource-allocation gate this doc set for
itself, NOT one of `AGENTS.md`'s two admissible closing grounds -- the
observed interval [-0.2126, +0.1108] crosses zero (the expected outcome for
a real small signal at this evaluator's resolution) and is recorded
`unresolved_below_power`, not refuted. `derive_crew_home_bias_features` /
`attach_crew_home_bias_features` remain built and tested
(`tests/test_officials_flag_features.py`) for any future re-run of this
gate against a larger officials snapshot.

Recorded: `nfl-ats weak-signals record --name officials_home_bias_reliability
--family officials_crew_traits --category onfield --classification
unresolved_below_power --effect-units correlation --effect -0.0518
--interval-low -0.2126 --interval-high 0.1108 --probability-positive 0.325
--sample-games 2892 --sample-blocks 11 --league nfl --season-start 2015
--season-end 2025`.

### LEAD-31: left-censoring (measured, `describe_referee_left_censoring`)

Of 29 distinct referees appearing 2015-2025: **17 (58.6%) show a 2015
debut** (left-censored -- true tenure unknown, indistinguishable from a
genuine rookie in this dataset) and **12 (41.4%) have a genuine
dataset-visible debut in 2016-2025**. The Stage-2 rookie flag is restricted
to `season >= 2016` for exactly this reason (see predeclaration above).

### LEAD-34: crew-familiarity frequency and penalty gap (measured, `describe_crew_familiarity`)

Of 2,892 matched REG games, **1,010 (34.9%) are a crew second meeting**
(the assigned crew already officiated a game involving one of the two
teams earlier that season). Mean `penalties_total` (both teams combined):
**12.08** in second-meeting games vs. **12.87** in first-meeting games, a
descriptive difference of **-0.80** penalties/game -- directionally
consistent with the predeclared "cleaner games" mechanism, reported
plainly as a descriptive comparison (not a hypothesis test; `reliability:
not_applicable` per the predeclaration).

### LEAD-33: confirmed source gap, skipped

Confirmed (read): `officials.parquet`'s 9 columns (`game_id, game_key,
official_name, position, jersey_number, official_id, season, season_type,
week`) carry no all-star/Pro-Bowl/crew-designation marker, and no other
locally snapshotted table in this repo does either. No proxy invented, per
the task's own instruction. Stays `🔬` on `ROADMAP.md`.

### Stage 2: LEAD-34 and LEAD-31 on-production screens (measured)

Both rotation-assigned window `[2020, 2021]`, opener grade, `--acknowledge-mined`
(pool entirely inside 2018-2025). Positive control (both, same
window/population, `--mode positive-control`): **+44.298 accuracy points**,
week-blocked `probability_positive` **1.000** -- harness proven sensitive
before either screen was read.

| Family | Verdict | Effect (pts, week-blocked) | Week 95% CI | P+ | n games / weeks |
|---|---|---|---|---|---|
| `crew_second_meeting_favorite_on_production` | unresolved | -0.6579 | [-4.2129, +3.2967] | 0.32515 | 456 / 35 |
| `rookie_crew_underdog_on_production` | unresolved | -1.0965 | [-2.6906, +0.2217] | 0.0456 | 456 / 35 |

`rookie_crew_underdog_on_production`'s interval upper bound (+0.2217) sits
fractionally above zero, so despite the low P+ this is NOT a RESOLVED
whole-interval-below-zero wrong sign -- the admissible closing ground
requires the WHOLE interval on the wrong side, which this narrowly misses.
Both remain `unresolved_below_power`, recorded via `nfl-ats rotation record`
and `nfl-ats weak-signals record`; both rotation windows are now spent.

Recorded: `nfl-ats weak-signals record` for both families, classification
`unresolved_below_power`, category `onfield`, league `nfl`, seasons
`[2020, 2021]`, `sample_games=456`, `sample_blocks=35`. Registry counts
after recording: 744 (`crew_second_meeting_favorite_on_production`), 745
(`rookie_crew_underdog_on_production`).
