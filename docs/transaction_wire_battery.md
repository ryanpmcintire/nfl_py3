# Transaction-wire battery: predeclaration

Written 2026-08-26, **before any cover-rate sign in this battery has been
examined** (the classification/coverage measurements in sections 1-2 below
are, per this project's existing precedent in `docs/fluview_battery.md`
section 4 and `docs/team_style.md`'s "Reliability gate" section, the
admissible pre-scoring exception: they are computed on the PREDICTOR's own
distribution -- article dates, slug text, team-name matches -- never on a
cover-rate outcome). This document freezes cells, thresholds, and the
scoring population before any of it is scored against `home_cover`.

## Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(the whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never
the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign or interval
shape.

## 0. Mechanism

The owner's pool (Splash Sports) posts lines Tuesday, revises once
Wednesday, then FREEZES them for the week (`docs/opener_evaluation.md` line
129: "posts lines Tuesday morning, revises once Wednesday, then freezes
them"; `docs/observed_movement_channel.md` line 14: "line freezes Tuesday
noon (revised once Wednesday, then frozen for the week)"). Practice-squad
elevations, late signings, and injured-reserve moves that happen after that
freeze are the team publicly announcing which positions it is scrambling to
cover, days after the price stopped moving. **Picks stay editable to
kickoff, only the LINES freeze Tuesday/Wednesday** (owner-stated,
`picks-lock-at-kickoff` memory entry -- not re-derived here), so a
late-week channel built from this archive is playable even though the line
itself cannot react to it.

This is, to my knowledge, the first attempt to build ATS features from the
Pro Football Rumors (PFR) transaction-wire archive ingested in
`docs/pfr_transactions_sourcing.md`. That document's own scope was
ingestion + a source-coverage/additivity experiment against PFT injury
news; it explicitly did not run an ATS screen, and the registry's one
`pfr_transactions_sourcing` entry is a sourcing note, not a signal.

## 1. Timestamp basis (established first, per this task's own instruction)

**Read** (`data/raw/pfr_transactions/20260820T011126Z/manifest.json`,
`docs/pfr_transactions_sourcing.md` section 1): the sitemap `<lastmod>`
field is CONTAMINATED as a publish-date proxy -- a 2025-12/2026 site-wide
bulk retouch bumped many old posts' `<lastmod>` forward (a 2015-09-23
article carries `<lastmod>` 2025-12-26). It is recorded for transparency
but **never used as a timestamp in this battery**.

Two timestamps that ARE trustworthy, in increasing order of precision:

1. **`url_year`/`url_month`** (parsed from every URL's own `/YYYY/MM/slug`
   path, free, present for all 72,368 inventory rows): **measured** (two
   independent stratified samples, 195 + 130 real article fetches with
   JSON-LD extraction) to match the true `datePublished` year/month
   **100.0% of the time** in both samples
   (`date_verification_summary.json`, `date_verification_summary_relevant_only.json`).
   Month-granular only.
2. **Per-article JSON-LD `datePublished`** (fetched and cached per-slug into
   `sample_articles/<slug>.json`): exact to the second, ground truth. This
   is the ONLY basis this battery uses for anything that needs day/hour
   precision -- every window this battery builds (the 72-hour-before-kickoff
   window, the since-line-freeze window) is exactly that kind of question,
   so **every scored transaction row in this battery uses the per-article
   JSON-LD `datePublished`, never the month-only proxy**. The month-only
   proxy is used ONLY for the section-2 coverage/classification report
   below, which needs no day-level precision.

**Read** (`docs/pfr_transactions_sourcing.md` section 3, as of the
2026-08-20 session): the shared `sample_articles/` cache held 4,653 valid
dated files, of which 4,537 were `transaction_relevant` rows -- almost all
from a targeted bulk fetch (`scripts/pfr_bulk_date_fetch.py`) covering only
Aug(Y)-Jan(Y+1) for seasons Y in {2022, 2023, 2024, 2025}, **complete**
(4,361/4,361 target rows dated, zero fetch or extraction failures) for that
four-season scope.

**Measured this session** (2026-08-26): extended `scripts/pfr_bulk_date_fetch.py`
with a `--seasons` argument (non-breaking; default unchanged) and launched
it against the Aug(Y)-Jan(Y+1) window for seasons 2014-2021 (8 more
seasons, 9,080 target rows, 65 already cached before this run) --
`.\.tools\uv.exe run --no-sync python scripts/pfr_bulk_date_fetch.py --snapshot data/raw/pfr_transactions/20260820T011126Z --seasons 2014,2015,2016,2017,2018,2019,2020,2021 --max-fetches 15000`.
This run was still in progress (background process, ~1 request/second per
the source's `robots.txt` `Crawl-delay: 1`, ~2.5 hours for the full
9,080-row scope) when this document and the screen/record scripts below
were written; `scripts/transaction_wire_battery_screen.py`'s own printed
coverage table and the recorded registry entries report the actual
per-season completeness at the time each was run, not a number frozen
here.

**Predeclared completeness rule (binding, decided before any score was
seen):** a season is included in the SCORED population if and only if
100% of that season's `transaction_relevant` rows with `url_year`/`url_month`
in its Aug(Y)-Jan(Y+1) target window have a successfully cached, non-null
JSON-LD `datePublished` at scoring time. This is a coverage-completeness
rule, not a cherry-pick: a partially-fetched season is EXCLUDED WHOLESALE
rather than scored on whatever subset happened to be fetched first, because
a partial-coverage season would silently undercount every team-week that
happens not to have been fetched yet -- indistinguishable from "no
transaction happened" without this rule. Excluded seasons are reported by
name and by their measured coverage fraction, never silently dropped.

## 2. Transaction-type classification and coverage

**Measured** (`nfl_ats.transaction_wire_features.classify_transaction_slug`,
applied to all 29,414 `transaction_relevant` inventory rows, independent of
date-fetch coverage -- this is a predictor-only, pre-scoring measurement):

| category | n (all seasons 2014-2026) |
|---|---:|
| signing | 11,816 |
| other (not one of the 8) | 8,145 |
| release | 3,580 |
| trade | 3,009 |
| ir_placement | 1,445 |
| suspension | 635 |
| waiver_claim | 406 |
| ir_activation | 329 |
| practice_squad_elevation | 49 |

Classification is a **priority-ordered, single-category-per-slug**
approximation (full priority order and regexes in
`nfl_ats.transaction_wire_features.classify_transaction_slug`): a slug
naming two simultaneous events (e.g. `49ers-elevate-kerryon-johnson-place-jamycal-hasty-on-ir`,
two different players, two different transactions) is counted once, under
its higher-priority category. `practice_squad_elevation` is checked FIRST,
ahead of the much larger IR buckets, specifically because measuring this
session found elevation announcements are disproportionately compound with
an IR placement in the same headline -- reordering elevation ahead of IR
moved its count from 40 to 49 (the other 8,880-ish IR-pattern-matching
slugs did not also contain an elevation keyword, so this reordering is a
small, disclosed correction, not a large one).

**Read, disclosed limitation**: `practice_squad_elevation` is genuinely
thin at the SLUG level, not an artifact of this classifier. Sampling slugs
containing `practice-squad` (1,541 of them) shows the large majority are
either (a) practice-squad SIGNINGS (`ravens-sign-tony-jefferson-to-practice-squad`,
a roster-depth move, correctly classified `signing`, not `elevation`) or
(b) weekly ALL-TEAMS roundup posts (`nfl-practice-squad-updates-9-4-23`),
which match zero team nicknames (see below) and cannot be attributed to a
specific team from the slug at all. True single-player, single-team,
game-day elevation announcements with their own dedicated slug
(`bills-elevate-wr-john-brown`) are real but comparatively rare in this
corpus. The 72-hour-before-kickoff elevation cells (T6/T7 below) inherit
this thinness; their measured `n_flag` is reported honestly in the results,
including if it is small enough to leave a cell `insufficient_data` --
which is a coverage fact about this source, not a decision to drop the
cell (nothing in this battery is dropped pre-emptively).

**Per-season counts**: `scripts/transaction_wire_battery_screen.py`'s
printed output and its `artifacts/transaction_wire_battery/<UTC>/results.json`
carry the full per-season x per-category pivot table (13 seasons x 9
categories) so coverage is auditable season by season, not just pooled.

**Team attribution, measured** (`nfl_ats.transaction_wire_features.match_transaction_teams`,
applied to the same 29,414-row inventory): matching each slug's tokens
against a 32-key nickname dictionary (duplicated from
`nfl_ats.injury_signal_refresh_tilt.TEAM_NICKNAMES`, itself extending
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` -- see that module's own
docstring for why nickname matching needs only the 32 CURRENT codes while
`scripts/fluview_battery_ingest.py`'s state-based `STATE_BY_TEAM` needs 34
including OAK/SD/STL: a nickname does not change on relocation, a market
does):

| teams matched in slug | n rows | share |
|---|---:|---:|
| 0 (round-ups, no team name in slug) | 6,778 | 23.0% |
| 1 (the common case) | 20,377 | 69.3% |
| 2 (usually a trade, both sides) | 1,671 | 5.7% |
| 3+ (multi-team roundups) | 588 | 2.0% |
| **>=1 (team-attributable)** | **22,636** | **77.0%** |

A trade slug matching two teams contributes one churn EVENT to EACH team's
count (`nfl_ats.transaction_wire_features.explode_dated_transactions`) --
this battery counts roster-churn ACTIVITY, not signed value, so no
"traded away" vs. "traded for" direction is inferred from slug text alone.
The 23.0% zero-team-match rows (round-ups, "extra-points" link posts) are
excluded from every team-week feature below -- they cannot be attributed to
a team-week without parsing full article bodies, which are not fetched by
this ingestion (see `docs/pfr_transactions_sourcing.md` section 2 on the
`headline_from_slug` design). This is a genuine, disclosed undercount of
true team activity, not a defaulted zero.

## 3. Team-week feature construction (point-in-time-safe)

`nfl_ats.transaction_wire_features.build_team_week_population`: one row per
`(season, week, team)` for every REG schedule game, both home and away
sides, team codes canonicalized via `TEAM_ABBREVIATION_ALIASES` (matching
`nfl_ats.features._canonical_schedules`'s own normalization, duplicated
here since that helper is private). Each row carries:

- `kickoff_utc` -- combined `gameday` + Eastern `gametime`, in UTC
  (duplicated from `nfl_ats.features._kickoff_utc`, also private).
- `freeze_utc` -- **own-week Wednesday noon ET**, in UTC. Neither
  `docs/opener_evaluation.md` nor `docs/observed_movement_channel.md`
  states an exact hour for the Wednesday revision; noon ET is an
  **inferred** convention chosen for consistency with every other
  noon-anchored cutoff already in this repo (`own_week_tuesday_noon_utc` in
  `injury_signal_refresh_tilt.py`/`movement_attribution.py`/
  `injury_tuesday_cutoff_experiment.py`), shifted one day later. Disclosed
  edge case: a Tuesday-kickoff game (rare weather makeup) resolves to the
  PRIOR week's Wednesday under the "most recent Wednesday at or before
  kickoff" definition -- correct under that definition, not a bug
  (`tests/test_transaction_wire_features.py::test_own_week_wednesday_freeze_for_a_tuesday_kickoff_is_the_prior_week`).
- `window72_start_utc` -- `kickoff_utc - 72 hours`, the literal window this
  task's own example asked for ("elevations in the 72 hours before
  kickoff"). Disclosed: for a Thursday game specifically, Wed-noon freeze
  falls ~32 hours before kickoff, INSIDE the 72-hour window (for a
  Sunday/Monday game the two windows do not overlap, ~4-5 days apart) --
  the two features are not always disjoint, and this battery does not force
  them to be (`tests/test_transaction_wire_features.py::test_thursday_kickoff_freeze_instant_falls_inside_the_72h_window`).

`nfl_ats.transaction_wire_features.attach_transaction_counts` then attaches,
per team-week row, STRICT point-in-time-safe counts via a sorted-timestamp
binary search per team: `n_events_since_freeze` (all 8 typed categories,
`freeze_utc < precise_ts < kickoff_utc`), `n_<category>_since_freeze` per
category, `n_events_72h` and `n_<category>_72h` for the 72-hour window. Both
bounds are STRICT, and critically the right bound is always `< kickoff_utc`
-- **no transaction dated at or after kickoff can ever reach a feature**,
enforced by construction and covered by
`tests/test_transaction_wire_features.py`'s leakage regression tests
(a single-event boundary check, an exactly-at-freeze/exactly-at-kickoff
boundary check, and a 200-event randomized check that recomputes every
count independently and confirms it matches the strictly-pregame subset).

`net_roster_churn`, the aggregate metric cells T1-T3 below use, is
`n_events_since_freeze`: the count of all 8 typed categories (excluding
`other`) published after that week's freeze and before that team's own
kickoff.

## 4. Predeclared cells (7)

Population for every cell: NFL REG, seasons meeting section 1's
completeness rule, both `home_cover` sides (no `location == "Home"`
restriction -- unlike FluView's home-market illness mechanism, a team's own
roster churn is a property of the TEAM, not the market it happens to be
playing in, so neutral-site and displaced-stadium games are not excluded).
Close-graded via `schedules.spread_line` (`nfl_ats.features.add_ats_outcomes`,
pushes/missing dropped) -- the same close-graded convention
`docs/fluview_battery.md`/`scripts/nfl_weather_battery_screen.py` already
use for this style of measure-only lead-generation battery; nothing in this
document is a promotion decision, so AGENTS.md's opener-grading rule for
decisions does not apply here (that rule governs what gets PLAYED, not
every measurement).

Because churn events are sparse in a several-day window (unlike FluView's
continuous state-level illness rate), the "elevated" flag for every cell is
**binary presence** (`count >= 1`) rather than a decile threshold -- a
decile threshold would degenerate to approximately the same binary flag
once the underlying rate is this sparse, so binary presence is the more
transparent predeclaration. Method: joint week-blocked bootstrap (block =
`season*100+week`) PRIMARY, season-blocked bootstrap SECONDARY, both
`block_bootstrap_two_group`-identical to `scripts/nfl_weather_battery_screen.py`/
`docs/fluview_battery.md` (imported from `scripts/_common.py`, not
reimplemented). Full-slate effect scaling via
`nfl_ats.experiment_runner.scale_subset_effect`, `accuracy_points` units,
20,000 bootstrap samples, seed 20260826 (repo convention: today's date).
Within-week correlation is zero by owner mandate -- no ICC term.

**T1. `txn_home_churn_elevated`** -- home team has `n_events_since_freeze
>= 1` vs. 0, response `home_cover`. **Predicted sign: NEGATIVE** -- a team
with visible post-freeze roster churn (signings, releases, trades, IR
moves) is plausibly scrambling to cover a gap the frozen line never saw;
the market cannot react, so the home team specifically underperforms the
frozen number.

**T2. `txn_away_churn_elevated`** -- away team `n_events_since_freeze >= 1`
vs. 0, response `home_cover`. **Predicted sign: POSITIVE** -- mirror
mechanism: the away team's own scrambling favors the home side the frozen
line did not adjust for.

**T3. `txn_differential_home_worse_churn`** -- restricted to games where
**exactly one side** has `n_events_since_freeze >= 1` (home XOR away);
subset = home elevated & away not, complement = away elevated & home not,
response `home_cover`. **Predicted sign: NEGATIVE** -- the cleanest test of
the same mechanism, isolating games where only one side shows the exposure
(same differential-cell design as `docs/fluview_battery.md` F3).

**T4. `txn_home_ir_placement_since_freeze`** -- home team
`n_ir_placement_since_freeze >= 1` vs. 0, response `home_cover`.
**Predicted sign: NEGATIVE** -- IR placement is a confirmed, non-speculative
loss of a roster spot for the rest of that game (and typically several
more), a more severe and unambiguous signal than the aggregate churn count.

**T5. `txn_away_ir_placement_since_freeze`** -- away team
`n_ir_placement_since_freeze >= 1` vs. 0, response `home_cover`.
**Predicted sign: POSITIVE** -- mirror of T4.

**T6. `txn_home_ps_elevation_72h`** -- home team
`n_practice_squad_elevation_72h >= 1` vs. 0, response `home_cover`.
**Predicted sign: NEGATIVE** -- an elevation is typically driven by an
injury/absence to the starter the elevated player is covering for; the
team is publicly signaling an in-game-day gap the market never priced.
Section 2 already discloses this category is thin at the slug level, so
this cell's `n_flag` may be small; reported honestly either way.

**T7. `txn_away_ps_elevation_72h`** -- away team
`n_practice_squad_elevation_72h >= 1` vs. 0, response `home_cover`.
**Predicted sign: POSITIVE** -- mirror of T6.

Seven cells, not eleven (a full home/away/differential x 3-construct
design): the differential form is predeclared ONCE, for the aggregate churn
construct only (T3), matching FluView's own precedent of one differential
cell per battery rather than one per underlying construct -- IR placement
and PS elevation are each already the more specific, rarer signal, and a
differential restriction would shrink an already-thin population further
without a new hypothesis being tested.

## 5. Reliability check (measured, run before cover-rate scoring)

Split-half reliability of the underlying team-week churn-count trait, via
`nfl_ats.cfb_qb_dependence.split_half_reliability` (the same function
`docs/fluview_battery.md` section 6 / PBP-05 / PBP-08 / `injury_value_lost`
were built on), applied to the FULL team-week panel (every scored
team-week, not just flagged ones), `team_id` <- team code, `season` <-
season, `week` <- NFL week (the odd/even split), metric = the raw
`n_events_since_freeze` COUNT (not the binary flag) -- reused as a single,
conservative figure across all 7 cells, same precedent as FluView section 6
("the same figure applied to all 5 cells, since all 5 share the identical
underlying AS-OF-elevated construct"). This tests whether "this team is
running hot on roster churn this season-half" is a real, persistent
within-season trait, which is the assumption every cell's binary-presence
flag depends on. Per AGENTS.md, a `no_split_half_reliability` closing
ground requires this figure's CI to sit AT (not near) zero -- an interval
crossing zero here is, as everywhere else, not grounds to close.

## 6. Files

- `src/nfl_ats/transaction_wire_features.py` -- slug classification, team-
  nickname matching, team-week population construction, point-in-time-safe
  window counts. Covered by `tests/test_transaction_wire_features.py`,
  including the leakage regression tests required before this family can be
  scored.
- `scripts/pfr_bulk_date_fetch.py` -- extended this session with a
  `--seasons` argument (non-breaking; default unchanged at 2022-2025) to
  widen per-article JSON-LD date coverage to earlier seasons.
- `scripts/transaction_wire_battery_screen.py` -- builds the team-week
  panel, applies section 1's completeness rule per season, scores all 7
  cells, runs the reliability check, writes
  `artifacts/transaction_wire_battery/<UTC>/results.json` (measure-only, no
  registry writes), following `scripts/fluview_battery_screen.py`'s
  structure and its `nfl_ats.provenance` artifact/experiment-registry
  writing.
- `scripts/transaction_wire_battery_record.py` -- records all 7 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`,
  mechanically classified by `nfl_ats.experiment_runner.classify_subset_bias_result`,
  verifies after writing. Follows `scripts/fluview_battery_record.py`
  exactly.
