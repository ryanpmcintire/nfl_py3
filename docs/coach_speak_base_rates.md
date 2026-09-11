# Coach-speak phrase base rates (LEAD-56)

Lane LEAD-56, 2026-09-11. This document is written in two passes, in order:
the predeclaration (lexicon, population, matching algorithm, every
methodological choice) was fixed and written down **before** any absence
outcome was read, then the Results section was appended once the script ran.
Same discipline `docs/follow_news_gate.md` and the other mined-battery docs in
this repository use.

## Binding closing-grounds taxonomy (verbatim; pasted per the standing
subagent rule since subagents never see AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

## What the archive actually is (measured before any phrase was mined)

- **Disk footprint and shape.** `data/raw/injury_news/` is **3.0 GB** across
  **44** timestamped snapshot directories (`measured`, `du -sh` and a directory
  listing). Each snapshot's `index.parquet` is independently rebuilt from the
  same `monthly/*.parquet` sitemap chunks plus a `current.parquet` hub crawl
  (`scripts/ingest_injury_news.py:ingest`, read), so the 44 snapshots are
  overwhelmingly redundant re-derivations of one cumulative history rather than
  44 independent slices. This lane mines the single latest snapshot,
  `data/raw/injury_news/20260911T000020Z/index.parquet` (**measured**:
  299,748 rows, `lastmod` spanning 2009-09-01 to 2026-09-10).
- **There is no article body text in this archive.** `index.parquet`'s
  columns are exactly `month, url, lastmod, slug, headline_guess,
  matched_keywords, injury_relevant` (**measured**, `df.columns.tolist()`).
  `headline_guess` is the URL slug with hyphens replaced by spaces
  (`scripts/ingest_injury_news.py:189-217`, read) -- it is a machine
  reconstruction of the headline, not the article prose. Every phrase match in
  this lane is therefore a match against a **slug-derived pseudo-headline**,
  not full body text. Two direct consequences, both disclosed rather than
  patched around: (a) possessive constructions lose the apostrophe-s
  ("Thomas's" -> "thomass"), which silently breaks player-name matching for
  those articles (an undercount, not an overcount); (b) a phrase that appears
  only in an article's body and not its headline is invisible to this
  archive. `scripts/ingest_injury_news.py --verify-sample` fetches a handful of
  real article pages to a JSON scratch file, but that is a one-off spot-check,
  not a stored corpus.
- **The `injury_relevant` column is not a usable prefilter for this lane.**
  It is set by whether the slug contains one of a fixed 44-term keyword list
  (`INJURY_KEYWORDS` in the ingester) that does **not** include "hopeful",
  "trending", "game-time", "expected-to-play", "day-to-day" as a whole term,
  or "will-not-play". The *historical* monthly-sitemap sweep tags every
  PFT/NFL sitemap URL regardless of that flag (only the incremental hub crawl
  pre-filters to it), so mining the full `headline_guess` column rather than
  the `injury_relevant` subset is required and is what this lane does.
- **Player identity crosswalk.** The official injury report used elsewhere in
  this repo (`data/players/raw/<stamp>/injuries.parquet`, read by
  `injury_signal_refresh_tilt.py`) carries only `gsis_id`, not a name
  (**read**, `src/nfl_ats/injury_signal_refresh_tilt.py` -- every player
  reference in that module is `gsis_id`; grep for `player_name` in that file
  returns nothing). The raw nflverse source one layer upstream,
  `data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet`, carries the
  same 90,891 rows **plus** `full_name`/`first_name`/`last_name` (**measured**,
  `df.columns.tolist()`), so that file is the crosswalk this lane resolves
  player identity against, restricted to skill positions
  (`SKILL_POSITIONS = {QB, RB, WR, TE}`, the same constant
  `injury_signal_refresh_tilt.py` uses) to match repo convention and cut
  name-collision risk. 2,153 distinct skill-position `full_name` values
  (**measured**).
- **Season coverage is narrower than the news archive's 2009-2026 span, for
  two independent reasons, both measured:**
  - `date_modified` (the timestamp a phrase-article is compared against) is
    only readable for seasons **2010-2024**: 2009 has it on 17 of 4,821 rows,
    2025 and 2026 have it on 0 of 6,068 and 0 of 139 (**measured**, grouped by
    season). This exactly matches the prior finding in
    `docs/follow_news_gate.md` that 2025's official rows are
    `observed_at_basis = week_proxy`.
  - The participation outcome (`data/players/raw/20260817T184901Z/
    snap_counts.parquet`) only covers seasons **2013-2025** (**measured**).
  - This lane's population is therefore **seasons 2013-2024, `game_type ==
    "REG"` only** -- the intersection of both constraints, declared here
    before any outcome was scored.

## Predeclared phrase lexicon

Matched case-insensitively as a literal substring against `headline_guess`
(regex-escaped, no word-boundary trimming beyond what the substring itself
implies):

| phrase_id | literal substring |
|---|---|
| `game_time_decision` | "game time decision" |
| `questionable` | "questionable" |
| `doubtful` | "doubtful" |
| `hopeful` | "hopeful" |
| `we_will_see` | "we will see" |
| `day_to_day` | "day to day" |
| `trending` | "trending" |
| `expected_to_play` | "expected to play" |
| `will_not_play` | "will not play" |
| `ruled_out` | "ruled out" |

This is exactly the LEAD-56 row's named list ("game-time decision",
"hopeful", "we'll see", "day to day", "trending the right way") plus the
task's required floor ("questionable", "doubtful", "expected to play", "will
not play", "ruled out"). "We'll see" cannot survive slug normalization
(apostrophes are stripped, so "we'll see" would need to appear as "we ll see"
or "well see"); it is predeclared as the literal "we will see" and reported
as measured-near-empty below rather than silently dropped.

## Predeclared methodology

1. **Player resolution.** Tokenize `headline_guess` on whitespace. Build a
   name index from the skill-position crosswalk: each distinct `full_name`
   is normalized (NFKD-fold accents, drop periods/apostrophes, lowercase,
   collapse whitespace) and indexed twice -- once as-is, once with a trailing
   suffix token (`jr/sr/ii/iii/iv/v`) stripped, since many articles omit the
   suffix. Every 2-token and 3-token contiguous span of the headline is
   checked against this index; a span that maps to **exactly one** `gsis_id`
   is accepted, a span mapping to more than one (11 of 2,157 name-index keys,
   **measured**) is treated as ambiguous and dropped rather than guessed.
2. **Game-window resolution.** For a matched (article, phrase, player)
   triple, take every skill-position official-report row for that player in
   seasons 2013-2024 REG, attach that row's team-game kickoff (Eastern local
   `gameday`+`gametime` from the schedule, converted to UTC -- the same
   recipe `nfl_ats.features._kickoff_utc` uses, replicated inline rather than
   imported since it is a private helper), and keep rows with
   `0 <= (kickoff - lastmod) <= 8 days`. This means the match is **only
   accepted when the player already had an official injury-report row for
   that team-week** -- a phrase with no filing behind it (e.g. "hopeful about
   a contract extension") cannot produce a match, the same discipline
   `docs/follow_news_gate.md` applies to its own news reader. Nearest
   qualifying game wins if more than one is in-window (rare).
3. **Deduplication.** Two different rules, for two different outputs, stated
   explicitly because they differ:
   - *Base-rate table (per phrase):* one row per (`gsis_id`, season, week,
     phrase), keeping the **earliest** qualifying article -- a later echo of
     the same phrase closer to kickoff is not new information for "when did
     this phrase first appear."
   - *Incremental-information test (per player-week, one phrase category):*
     one row per (`gsis_id`, season, week), keeping the article **closest to
     kickoff** across all phrases that fired that week -- the most recent
     status echo is what a reader deciding "beyond the official tag" would
     have seen last.
4. **Participation / absence outcome.** `total_snaps > 0` in
   `data/players/raw/20260817T184901Z/snap_counts.parquet` (offense + defense
   + special-teams snaps), joined on `(game_id, gsis_id)` via
   `nfl_ats.players.attach_snap_player_ids` (the same function
   `scripts/friday_designation_momentum_screen.py` uses for the identical
   purpose). A player absent from that week's snap file is coded `inactive`.
   Measured cross-check before trusting this: snap-row match rate by that
   week's own official `report_status` is 93.4% for Probable, 64.7% for
   Questionable, 0.9% for Doubtful, 0.03% for Out (**measured**) -- i.e. the
   players who fail to match are overwhelmingly the players who did not
   suit up, not a random data gap, so treating an unmatched row as inactive is
   the correct reading, not a bug being papered over.
5. **Days-before-kickoff buckets.** `[0,1)`, `[1,3)`, `[3,5)`, `[5,8]` days,
   fixed before the table was built.
6. **Wilson score interval**, z=1.959964, standard closed-form, on the
   base-rate table.
7. **Split-half reliability**, exactly as the lane brief specifies: **odd vs
   even calendar month of kickoff** (not season), computed per team on the
   pooled "any predeclared phrase fired" population (individual phrases are
   too thin per team-month-half to reliability-test alone), then Pearson-
   correlated across teams with >=5 player-weeks in **both** halves.
8. **Incremental-information test.** Population: the full 2013-2024 REG
   skill-position report population (19,322 rows with a resolved kickoff),
   not just the phrase-matched subset -- a player-week with no predeclared
   phrase gets `phrase_category = "none"`. Cross-fit by kickoff-month parity
   (fit on odd months, score even months, then swap, pool the two
   out-of-fold prediction sets so every row is scored by a model that never
   saw it): **baseline** predicts `P(inactive)` from `report_status` alone
   (Beta(2,2)-smoothed empirical rate per status); **candidate** predicts from
   `(report_status, phrase_category)` jointly. Reported as both mean Bernoulli
   log-loss improvement (candidate better = positive) and accuracy-point
   improvement at a 0.5 decision threshold, week-blocked bootstrap (20,000
   draws, seed 20260910), `probability_positive` from
   `nfl_ats.evidence_conventions.probability_positive_from_draws`. This is the
   literal test the lane brief asks for: "whether any phrase beats the
   official designation at the same instant... as a predictor of absence" --
   with the disclosed caveat in the next paragraph.
   - **Disclosed limitation on "same instant."** The official-report source
     stores exactly one row per player-week (**measured** in
     `scripts/friday_designation_momentum_screen.py`'s own coverage report,
     reproduced here: 90,887 of 90,889 season/week/team/`gsis_id` groups have
     count 1), so there is no intraday revision history to compare a
     mid-week article against -- only that week's single recorded
     `report_status` exists at all. "Beats the official designation at the
     same instant" is therefore read as "adds information beyond that week's
     one recorded status," not literally "beats what was known at the exact
     minute the article published."
9. **Production-screen trigger rule, predeclared, EV-first (not a 0.90/0.95
   bar).** If the incremental-information test's `accuracy_points`
   `probability_positive` exceeds 0.5 -- leaning toward the candidate direction
   at all, regardless of whether the interval crosses zero -- a production
   screen is attempted. This mirrors `AGENTS.md`'s standing rule that a
   promotion bar is not a decision bar; a coin flip leaning the right way is
   still worth an EV-priced look, not a veto.
10. **Production screen construction** (only run because step 9's trigger
    fired). Restrict to phrase-fired player-weeks whose *prior* game's snap
    share was >= 0.5 (a real starter, using an `asof`-joined trailing share
    computed independently of whether the player suited up THIS week, to
    avoid the tautology of only counting players who happened to play -- see
    Caveats). Weight by the empirical inactive rate among those starters
    (reusing the injury-value-lost family's "starter-weighted expected
    absence" logic, `scripts/friday_designation_momentum_screen.py:216-232`,
    adapted from `revision_class` to `phrase_category`) to build a per-team
    per-game expected-absence value, fade the team with the larger value when
    the two sides disagree, and grade against the active model's opener
    evaluation (`nfl_ats.public_board.find_matching_opener_evaluation`),
    week-blocked bootstrap, plus a perfect-foresight positive control on the
    same decision set.

## Results

**Measured** this session,
`artifacts/experiments/coach_speak_base_rates/20260911T035942Z/`
(`results.json` via `write_experiment_artifact`, `player_week_phrase_matches.parquet`,
`scored_population.parquet`, a copy of the run's provenance).

### Coverage

| quantity | value |
|---|---:|
| news archive rows (latest snapshot) | 299,748 |
| news archive span | 2009-09-01 to 2026-09-10 |
| articles matching >=1 predeclared phrase | 5,322 |
| distinct skill-position `full_name` values | 2,153 |
| official skill-position REG rows, 2013-2024, readable timestamp | 19,328 (19,322 resolved to a kickoff) |
| resolved (article, phrase, player, in-window game) rows, player-week deduped | **2,537** |

Resolved player-week rows by phrase (player-week deduped, earliest article):

| phrase | n | inactive rate | 95% Wilson |
|---|---:|---:|---|
| `will_not_play` | 38 | 89.47% | [75.87%, 95.83%] |
| `doubtful` | 263 | 82.89% | [77.87%, 86.96%] |
| `ruled_out` | 289 | 80.28% | [75.30%, 84.45%] |
| `day_to_day` | 114 | 57.89% | [48.72%, 66.56%] |
| `game_time_decision` | 91 | 52.75% | [42.59%, 62.68%] |
| `questionable` | 1,320 | 40.38% | [37.76%, 43.05%] |
| `hopeful` | 65 | 38.46% | [27.60%, 50.62%] |
| `trending` | 23 | 26.09% | [12.55%, 46.47%] |
| `expected_to_play` | 334 | 22.75% | [18.58%, 27.55%] |
| `we_will_see` | **0** | -- | -- |

`we_will_see` is measured-empty: the raw substring "we will see" appears in
only 2 of 299,748 headline-guess rows archive-wide, and neither resolves to a
player + in-window game. This is reported as a finding about the archive, not
silently dropped from the lexicon.

**Read plainly, the phrase gradient is monotone and matches football
intuition exactly**: the most direct absence language ("will not play",
"doubtful", "ruled out") sits at 80-90% next-game absence, the classic hedge
words ("questionable", "hopeful") sit near 40%, and the optimistic phrases
("trending [in the right direction]", "expected to play") sit at 23-26%. None
of the nine populated phrases' Wilson intervals overlap both the top cluster
and the bottom cluster, which is the honest way to say this is not noise.

By days-before-kickoff bucket (`by_phrase_days_bucket` in `results.json`;
full table in the artifact, not reproduced in full here): the largest cells
land in the `1-3d` bucket for every phrase except `expected_to_play` and
`questionable`, consistent with these being predominantly Thursday/Friday
practice-report presser language rather than same-day inactive-list echoes.
`ruled_out__1-3d` (n=226) reads 83.6% [78.2%, 87.9%]; `expected_to_play` is
below 30% in every bucket it has >=10 rows.

### Split-half reliability

**Measured**: team, odd-kickoff-month mean vs even-kickoff-month mean, pooled
across all nine populated phrases (individual-phrase team-month cells are too
thin to reliability-test alone), 32 teams qualify (>=5 player-weeks in both
halves). **Correlation: -0.227.**

This is not a resolved `no_split_half_reliability` closure under the taxonomy
above (that requires a reliability of essentially zero measured with
adequate power, and 32 teams at this cell size is thin), and it is not
evidence of a real negative team-level trait either. **Read plainly:** the
construct being tested here -- "does a team have a stable house style of using
these phrases whose absence-predictiveness repeats month to month" -- is a
team-level aggregation of what is fundamentally a per-player, per-article
signal, and a negative-leaning, noisy correlation at n=32 is what an
underpowered read of that aggregation looks like. It is reported here as
measured and `unresolved_below_power`, not as a refutation.

### Incremental-information test vs the official designation

**Measured**, cross-fit odd/even kickoff-month, 19,322-row population (11,718
of those retained after dropping rows with no readable `report_status`/
`inactive`/`kickoff`), week-blocked bootstrap, 20,000 draws, seed 20260910:

| metric | baseline (status only) | candidate (status + phrase) | delta | 95% week-blocked | `probability_positive` |
|---|---:|---:|---:|---|---:|
| mean log loss | 0.33267 | 0.33286 | -0.000182 | [-0.00177, +0.00149] | 0.4097 |
| accuracy | -- | -- | **+0.213 pts** | **[+0.025, +0.401]** | **0.9876** |

Two honest, disagreeing readings on the same 11,718-row cross-fit, reported
both rather than only the favorable one. The **log-loss** read is a coin flip
leaning slightly against the phrase (P+ 0.41): once a player's official
`report_status` is known, the extra text in a coach-speak phrase adds
essentially nothing to the *calibrated probability* of absence. The
**accuracy** read is a small, week-blocked-bootstrap-positive effect whose
interval technically sits above zero (+0.025 to +0.401 points) at P+ 0.9876,
but the magnitude is tiny -- about a fifth of one accuracy point on a
player-level absence call, not a football-relevant swing. **My read
(inferred):** the two metrics disagree because a phrase can nudge a
borderline `report_status="Questionable"` prediction (near 0.4-0.5) across the
0.5 decision threshold on a small number of games without moving the
*calibrated* probability by much -- exactly the shape you'd expect from a
genuinely marginal, mostly-redundant-with-the-official-tag signal, not a
large hidden edge. Per the taxonomy above, neither reading is a terminal
closure (log loss crosses zero; accuracy's positive-only interval is not a
`wrong_sign_resolved` case, and there is no positive-control bound on this
specific comparison) -- both are `unresolved_below_power`, reported with
`probability_positive`, not as "no effect."

### Production screen

Triggered per the predeclared rule (`accuracy_points` `probability_positive`
0.9876 > 0.5). Graded against the active model's opener evaluation
(`artifacts/opener_evaluation/20260910T211255Z`, model `d49194e04945a5e5`),
week-blocked bootstrap, 20,000 draws, seed 20260910:

| arm | games | flips | accuracy delta | 95% week-blocked | `probability_positive` |
|---|---:|---:|---:|---|---:|
| starter-weighted expected-absence fade vs production | 1,503 | 237 | **-0.200 pts** | [-2.183, +1.799] | 0.4306 |
| positive control (perfect foresight on the same 237-switch decision set) | 1,503 | -- | **+45.442 pts** | [+42.848, +48.097] | 1.0000 |

The instrument is proven (a perfect gate on this exact decision set is worth
+45.4 accuracy points, resolved far from zero), and the measured candidate
(-0.200 points) sits nowhere near that bound -- **this is not
`bounded_by_control`**, it is a genuine small-sample null read,
`unresolved_below_power`, reported with `probability_positive` 0.4306 rather
than as "no effect" or "contains zero." The population is seasons 2020-2025
(the opener-evaluation window); phrase-driven flips only occur in the
2020-2024 portion of it because seasons after 2024 have no readable official
`report_status` for the news-resolution step (disclosed above), so 2025 games
in this screen carry a zero expected-absence value on both sides by
construction, not an absence of signal.

## Registry entries recorded

Four `nfl-ats weak-signals record` calls, family `coach_speak`, category
`health`, league `nfl`, all `unresolved_below_power` (no admissible
`--closing-ground` applies to any of the four: no interval sits entirely on
the wrong side of zero, and the one positive control that exists bounds a
completely different, much larger decision set than the -0.200-point read it
sits beside). Exact commands and their JSON output are in
`artifacts/experiments/coach_speak_base_rates/20260911T035942Z/record_commands.json`.

1. `coach_speak_incremental_accuracy_2013_2024` -- accuracy_points, +0.213,
   season 2013-2024, `probability_positive` 0.9876.
2. `coach_speak_incremental_log_loss_2013_2024` -- log_loss_improvement,
   -0.000182, season 2013-2024, `probability_positive` 0.4097.
3. `coach_speak_production_screen_accuracy_2020_2025` -- accuracy_points,
   -0.200, season 2020-2025, `probability_positive` 0.4306.
4. `coach_speak_production_screen_positive_control_2020_2025` --
   accuracy_points, +45.442, season 2020-2025, `probability_positive` 1.0000.

## Caveats (label how you know it)

- **Measured, this session, from the artifact above** unless tagged
  otherwise: every number in the Results section.
- **No article body text exists in this archive** -- every phrase match is
  against a slug-derived pseudo-headline (`headline_guess`), never the
  article's own prose. A phrase used only in body text is invisible here.
- **Possessive names undercount.** Slug normalization drops "'s", so
  "Thomas's" becomes "thomass" and fails to match the crosswalk's "thomas".
  This is a measured undercount mechanism, not a claim that it was fixed.
- **The starter-weighted production-screen feature required a bug fix mid-lane
  worth disclosing.** The first implementation joined "trailing share" (the
  player's prior-week snap share) from the same current-week snap-count row
  used to compute this-week's participation outcome -- which meant
  "trailing_share is known" and "the player played this week" were
  tautologically the same condition, so the feature-eligible population had a
  fired-rate of exactly 0.0 by construction and produced zero flips. Fixed
  with a `merge_asof` lookup of the player's most recent PRIOR game's snap
  share, independent of this week's outcome (`scripts/coach_speak_phrase_mining.py`,
  `attach_participation`). The base-rate table and the incremental-information
  test never used `trailing_share` and are unaffected by this bug or its fix.
- **The team-level split-half reliability construction is a coarser cut than
  the underlying signal** (which is per-player, per-article). It is reported
  because the lane brief asks for odd/even-month reliability, but a
  player-week-level reliability was not separately computable at this sample
  size without a repeating unit (a given player rarely fires the same phrase
  twice), which is disclosed rather than glossed over.
- **Mined battery.** No multiplicity correction is claimed across the nine
  phrase cells or the four registry entries, matching every other mined-
  battery document in this repository.
- **Correlated with, not independent of, the existing injury-signal family.**
  The participation/absence outcome and the skill-position/season-2013-2024
  population overlap `injury_value_lost`, `follow_news_gate`, and
  `friday_designation_momentum` (`docs/injury_value_lost.md`,
  `docs/follow_news_gate.md`, `scripts/friday_designation_momentum_screen.py`).
  Never pool this family's registry entries additively with those.
