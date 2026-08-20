# QB news-visibility channel: reviving backup-QB with timestamped headlines

Written 2026-08-20. Predeclared BEFORE any coverage rate or screen-cell sign
was computed (Section 3 below is filled in only after this predeclaration is
frozen; Section 4 reports what actually ran).

## 0. Why this document exists

`docs/prospective_evidence.md`'s Tuesday-visibility audit (`backup_qb_fade_overlay`
section) found the overlay **permanently dead on arrival**: `home_qb_name`/
`away_qb_name` in `schedules.parquet` are populated only AFTER a game is
played (measured there: 272/272 null for all of 2026 REG, 0/272 null for all
of 2025), so no pregame timestamp can ever read the actual pregame starter
from that column, and the 816-team-week simulation showed the overlay
produces **zero live flips ever** under that data source.

**Owner correction, 2026-08-20**: pool picks are editable until kickoff --
only grading lines freeze Tuesday. This means the schedule column's
post-game-only nature is not fatal: it can still supply the historical GROUND
TRUTH (which team-games were actually backup starts, an outcome fact, always
valid for measurement), while a genuinely pregame-safe SIGNAL for live play
has to come from somewhere else -- specifically, the now-ingested timestamped
news corpora (`docs/injury_news_sourcing.md`'s ProFootballTalk archive,
`docs/pfr_transactions_sourcing.md`'s Pro Football Rumors transaction-wire
archive), read as of any point before kickoff, not just Tuesday noon.

## 1. Ground truth construct (unchanged from the existing overlay)

Reused verbatim, not re-derived: `own_qb_name != modal_qb` for a team once
that team has `>= 3` prior starts this season, ported through
`nfl_ats.experiment_runner._bias_battery_qb_backup_flag` /
`scripts/nfl_bias_battery_screen.py::_qb_backup_flag` /
`nfl_ats.backup_qb_fade_overlay.MIN_PRIOR_STARTS`. Same eligibility floor,
same running-counter definition, same team-game long table
(`_bias_battery_team_game_table`, `data/processed/game_features.parquet`
inner-joined with the newest `data/raw/*/schedules.parquet`). Window:
2015-2025 REG season, aligned to the two corpora's coverage (PFR transactions
start 2014 -- one warm-up season of margin before the ground-truth window;
PFT covers back to Sept 2009, no constraint).

## 2. News corpora on disk

| Source | Snapshot | Rows | Per-article timestamp |
|---|---|---:|---|
| ProFootballTalk (PFT) | `data/raw/injury_news/20260819T191639Z/index.parquet` | 299,739 | `lastmod`, real UTC timestamp, minute-granular, verified against JSON-LD `datePublished` (`docs/injury_news_sourcing.md` sec 3) |
| Pro Football Rumors (PFR) transactions | `data/raw/pfr_transactions/20260820T011126Z/index.parquet` | 72,368 | `url_year`/`url_month` (100% verified vs JSON-LD in a 325-article sample, month granularity, free); per-article JSON-LD `datePublished` (exact, requires a live fetch, `Crawl-delay: 1`) |

## 3. Matching rule, frozen before any cover-rate is computed

**Name match.** For each backup-start row, the ground-truth starting QB's
full name (as recorded in `schedules.parquet`) is normalized: lowercased,
punctuation (`.`, `'`, the curly apostrophe) stripped, remaining non-alphanumeric
runs collapsed to a single space (identical normalization to
`scripts/injury_tuesday_cutoff_experiment.py::_normalize_name`, reused for
consistency with the project's one existing precedent for this exact kind of
match). A candidate headline hits on NAME only when the full normalized name
(first + last, no suffix stripping) appears as a literal substring of the
candidate's normalized headline text (`headline_guess` for PFT,
`headline_from_slug` for PFR). This is the same conservative, precision-
favoring, last-name-undercounting convention `injury_tuesday_cutoff_experiment.py`
already uses and documents as a **lower bound** on true coverage -- carried
forward unchanged here, not re-derived.

**Keyword match.** A candidate headline hits on KEYWORD when its normalized
text contains at least one of the following, frozen before any cover-rate
was computed (union of three groups: injury/availability language reused
from `scripts/ingest_injury_news.py::INJURY_KEYWORDS`, new QB-specific
start/bench/depth-chart language, and roster-move language reused from
`scripts/ingest_transaction_news.py::TRANSACTION_KEYWORDS` -- a backup start
is not always injury-driven, so the keyword set must cover a benching/demotion
just as well as an injury):

```
injured-reserve, -on-ir, -ir-, questionable, doubtful, ruled-out, will-miss,
out-for-the-season, out-for-season, concussion, hamstring, quadricep, groin,
ankle, knee-injury, torn-acl, torn-mcl, shoulder-injury, achilles, surgery,
fracture, fractured, sprain, sprained, day-to-day, week-to-week,
injury-report, injury-designation, placed-on-ir, pup-list, exits-, exited-,
starting, starts, to-start, named-starter, gets-the-start, in-for, replace,
replaces, replacing, bench, benched, benching, demoted, new-starter, backup,
takes-over, relief, subs-in, activated, activate-, elevate, elevated,
elevation, practice-squad, signs, signed, waived, released, claimed,
suspended, suspension, inactive, inactives
```

**A match requires BOTH** the name hit and the keyword hit on the same
headline. A row's earliest qualifying match across the union of PFT and PFR
is the one used for bucketing. This rule is applied identically to both
corpora and is not redrawn after cover rates are seen.

**Pregame-safety enforcement.** A match only counts if its timestamp is
strictly before the game's own deadline (see Section 4's owner-corrected
`D = min(kickoff, that week's Sunday 16:00 ET)`; kickoff itself is
`data/raw/<snapshot>/schedules.parquet` `gameday` + `gametime`, combined and
localized America/New_York then converted to UTC, matching
`nfl_ats.features._kickoff_utc`'s construction). Post-game recaps that
happen to name the backup starter are excluded by construction -- this is a
coverage-of-FORESHADOWING measurement, not a coverage-of-mentions
measurement.

**Lookback window (added before any cover-rate number was treated as final --
see the addendum at the bottom of this section for why).** A match only
counts if its timestamp also falls within `[kickoff - 10 days, kickoff)`.
Without this bound, a journeyman backup QB's own name recurs across his
entire career against generic keywords ("signs", "activated", "waived",
"questionable") in totally unrelated seasons, and an unbounded search
returns an implausible ~94% "coverage" rate that is a name-collision
artifact, not foreshadowing of THIS game's start. Ten days mirrors
`scripts/injury_tuesday_cutoff_experiment.py::build_pft_match_table`'s own
precedent (`lookback_days=9.0`), rounded up slightly so a 10-day window
still fully covers a Thursday-night game's own-week Tuesday noon (the
earliest bucket) with a small cushion for a Sunday/Monday news item from the
tail of the PRIOR game week, exactly the kind of legitimate carry-over signal
the existing experiment already credits.

*Addendum, same session:* this document was frozen with the matching rule
above but WITHOUT a lookback bound. A first dry run (`--pfr-max-fetches 3`,
a deliberately tiny smoke test, not a reported coverage number) surfaced an
implausible 94.2%-by-Tuesday-noon reading, which is an obvious engineering
defect (name-collision over unbounded history), not a data-dependent
adjustment made after seeing a favorable or unfavorable sign -- the frozen
rule never specified "search a player's entire career," and the fix (bound
the search window) is the same kind of correction
`docs/injury_news_sourcing.md`'s own `lookback_days` parameter already
encodes. The lookback bound is added here, once, before any cover-rate
number in Section 6 was computed with it in place.

## 4. Time buckets, frozen before any cover-rate is computed

Four cutoffs per game, all convertible to UTC:

- `T` = that game's own-week Tuesday noon ET, using the SAME own-week
  convention `nfl_ats.clv.live_tuesday_openers` and
  `scripts/injury_tuesday_cutoff_experiment.py::team_week_tuesday_noon`
  already use (`(weekday - 1) % 7` days back from kickoff, anchored at
  noon ET rather than UTC midnight).
- `F` = that same week's Friday, 23:59:59 ET (`T`'s calendar date + 3 days).
  Chosen (not measured) as "end of the day the NFL's mandated final
  injury-practice report is filed" -- an inferred operational boundary, not
  a verified fact about the pool's own rules.
- `S` = Sunday 10:00 ET of the game's own calendar date **if** the game is
  played on a Sunday, **else** the game's own kickoff (verbatim task
  specification: "Sunday 10:00 ET (or the game's own kickoff for non-Sunday
  games)").
- `K` = the game's own kickoff (UTC).

**Owner correction, 2026-08-20 (same day, before Section 6's numbers were
finalized): the pool's actual per-game pick deadline is `D = min(K,
that week's own Sunday 16:00 ET)`, not raw `K`.** SNF and MNF lock EARLY at
Sunday 4pm ET, not at their own later kickoff -- "that week's own Sunday" is
the Tuesday-anchored market week's Sunday (`T`'s calendar date + 5 days),
which for a Monday game is the day BEFORE its own kickoff. `D` replaces `K`
as the enforcement boundary everywhere below (pregame-safety cutoff, the
10-day lookback window's anchor, and the `F`/`S` clamps); `K` is still
carried for reference/reporting. This affects Sunday games with a kickoff
after 4pm ET (late-afternoon and SNF slots, deadline a few minutes to ~4
hours earlier than their own kickoff) and, most, Monday games (deadline a
full day earlier). Materiality was checked directly, not assumed -- see the
addendum at the end of this section.

Each cutoff is clamped so the sequence is non-decreasing and never exceeds
the deadline: `T' = T`, `F' = min(F, D)`, `S' = min(S, D)`, and `D` itself.
(`T <= F' <= S' <= D` always holds -- proved in-session by construction, not
just assumed: `T < D` for every game including Thursday-night games, since
Tuesday noon of the same market week always precedes even the earliest
possible deadline of that week.)

A backup start's bucket is the FIRST of the following whose cutoff is
`>=` its earliest qualifying match time:

1. `by_tuesday_noon` (match `<= T`)
2. `by_friday` (`T <` match `<= F'`)
3. `by_sunday_10am_or_kickoff` (`F' <` match `<= S'`) -- **this is the task's
   named screen-eligible bucket** (the label is kept verbatim from the task
   even though the non-Sunday branch is now `D`, not raw kickoff)
4. `before_deadline` (`S' <` match `< D`) -- an honesty addition beyond the
   task's three named buckets, since Sunday games have a real gap between
   10:00 ET and their own deadline (`D`, typically their own kickoff at
   1pm/4:05pm/4:25pm ET, or 16:00 ET itself for an 8:20pm SNF kickoff) where
   a late-morning/early-afternoon inactives-list story could still land; for
   non-Sunday games this bucket is structurally empty (`S' = D`)
5. `never` (no qualifying match with match time `< D` in either corpus)

**Addendum: SNF/MNF materiality, measured directly (not assumed).** Of the
916 backup starts, 313 (34.2%: 258 late-Sunday/SNF, 55 Monday) had a
deadline `D` strictly earlier than their own kickoff `K` and were therefore
potentially exposed to the correction. Directly re-checked against the
FIRST (pre-correction, plain-kickoff) run's own detail table: only **3 of
916 (0.33%)** backup starts had their earliest qualifying match actually
fall in the affected `[D, K)` gap -- i.e. would have been wrongly counted as
"pregame-visible" under a plain-kickoff boundary. This is negligible by the
owner's own stated threshold ("if zero or negligible, a dated note
suffices"). The correction was applied and the full pipeline re-run anyway
(not just noted) -- every number in Section 6 below is from the
corrected-deadline run, not the plain-kickoff run.

## 5. The predeclared screen cell (run only if warranted by Section 4's numbers)

Population: the SAME eligible team-games `bias_battery_backup_qb_start` uses
(`>= 3` prior starts this team-season, 2015-2025 REG), from
`_bias_battery_team_game_table`.

Flag: `backup_start AND bucket IN {by_tuesday_noon, by_friday,
by_sunday_10am_or_kickoff}` -- i.e. news-visible by Sunday 10:00 ET (or
kickoff for a non-Sunday game), verbatim the task's specification. Complement:
every other eligible team-game (non-backup-starts, AND backup starts that
were not news-visible in time, e.g. `before_deadline` or `never`).

Value: `team_covered` (home_cover for the home side, `1 - home_cover` for the
away side), matching every other bias-battery cell.

Sign: `+1`, matching `bias_battery_backup_qb_start`'s own registered sign
convention exactly (so the two are directly comparable without a sign flip;
a negative reading here means the news-visible subset under-covers, same
direction as the two already-registered cells).

Scaling: `scale_subset_effect` convention (`nfl_ats.experiment_runner`) /
identically `scripts/nfl_bias_battery_screen.py::summarize_population`'s
`raw_gap_pts * fraction_of_slate`, where `fraction_of_slate = n_flag /
n_total` over the restricted eligible population (one-sided design, matching
`backup_qb_start`'s own `eligible`-restricted convention).

Interval: week-blocked joint bootstrap (`block = season*100 + week`), 20,000
resamples, seed 20260819 -- identical machinery to
`nfl_bias_battery_screen.py::block_bootstrap_two_group` /
`nfl_ats.experiment_runner._block_bootstrap_subset_gap`.

Comparator: `registry/weak_signals.json:bias_battery_backup_qb_start`
(-0.2731 pts, `probability_positive` 0.1684, close-graded 2009-2025) and
`bias_battery_backup_qb_start_opener` (-2.3578 pts, `probability_positive`
0.0982, opener-graded 2020-2025) -- both read from
`docs/backup_qb_fade_overlay.md` and the registry directly.

## Binding closing-grounds taxonomy (restated verbatim, applies to every verdict below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

---

## 6. Results, 2026-08-20 (measured this session)

Script: `scripts/qb_backup_news_visibility.py`. Artifact:
`artifacts/qb_news_channel/20260820T093852Z/result.json` (+
`backup_rows_detail.parquet`, the 916-row per-backup-start detail table).
Experiment-registry row: `registry/experiments/qb-backup-news-visibility/`
(written by `write_experiment_artifact`).

### 6.1 Ground truth (task item 1)

916 backup starts out of 4,588 eligible (`>= 3` prior starts this
team-season) team-games, 2015-2025 REG, identical construct to
`bias_battery_backup_qb_start`:

| Season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Backup starts | 73 | 56 | 67 | 72 | 75 | 83 | 74 | 107 | 99 | 103 | 107 |

### 6.2 News-visibility measurement (task item 2)

Matching rule and buckets frozen in Section 3-4 above BEFORE any cover rate
was treated as final (one implausible 94.2% dry-run reading, from an
unbounded-lookback bug, was caught and fixed before any number here was
computed with the fix in place -- see Section 3's addendum). Cumulative
coverage over the 916 backup starts, deadline-corrected:

| Bucket (cumulative) | n | rate |
|---|---:|---:|
| by Tuesday noon | 241 | 26.31% |
| by Friday | 366 | 39.96% |
| by Sunday 10:00 ET-or-deadline (**screen-eligible**) | 412 | 44.98% |
| before the deadline | 433 | 47.27% |
| never (not pregame-visible under this rule) | 916 | 100.00% (483 "never", 52.73%) |

**Comparator** (`docs/injury_news_sourcing.md` sec 5.1, official Wed-Fri
injury-report-only construct): 2.39% of Friday-final designations were
already headline-visible by Tuesday noon. This measurement's by-Tuesday-noon
rate (26.31%) is ~11x that comparator -- expected, since it searches outlet
news directly (PFT + PFR, 64-keyword QB-specific list, 10-day lookback)
rather than waiting on the NFL's mandated Wednesday-Friday report cadence,
and because a backup start is a broader/earlier-breaking event class
(injury OR benching OR a preceding roster move) than a single practice-report
designation. **Materially better: yes**, which is why Section 5's screen ran.

PFR contribution: 579 name+keyword+month-window-bounded candidates
identified locally (no fetch); 258 unique article URLs required a fetch for
day/hour precision, all 258 fetched successfully (`Crawl-delay: 1`, total
fetch time well under the 15-minute budget).

### 6.3 The screen cell (task item 3)

Ran per Section 5's predeclaration, deadline-corrected boundary:

| | |
|---|---:|
| Population (eligible team-games, 2015-2025) | 4,588 |
| Flagged (backup start, news-visible by Sunday 10am ET-or-deadline) | 412 |
| Complement | 4,176 |
| Subset cover rate | 49.51% |
| Complement cover rate | 50.14% |
| Raw gap (sign +1) | -0.6291 pts |
| Fraction of slate | 0.0898 |
| **Full-slate-scaled effect** | **-0.0565 accuracy points** |
| Week-blocked 95% CI (20,000 resamples, seed 20260819, 159 week-blocks) | [-0.4698, +0.3625] |
| Standard error | 0.2132 |
| **`probability_positive`** | **0.39145** |

**Comparators** (read from `registry/weak_signals.json`, same sign
convention, +1):

| Entry | Grade / window | Games | Effect (pts) | `probability_positive` |
|---|---|---:|---:|---:|
| `bias_battery_backup_qb_start` | close, 2009-2025 | 7,002 | -0.2731 | 0.1684 |
| `bias_battery_backup_qb_start_opener` | opener, 2020-2025 | 2,436 | -2.3578 | 0.0982 |
| `qb_news_backup_visible_by_deadline_screen` (this run) | close, 2015-2025 | 4,588 (412 flagged) | -0.0565 | 0.39145 |

**Reading it plainly.** The news-visible subset leans the SAME direction
(negative -- the flagged side still under-covers on average) as both
already-registered cells, so this does not contradict the existing
construct. But it is MUCH weaker here -- `probability_positive` 0.39 sits
close to a coin flip, versus 0.10-0.17 for the two broader constructs. My
guess, not measured: a backup start the market can already see coming in the
news may be partly priced by kickoff, while a "silent" backup start (never
bucket, 52.73% of the population) is the one a market-lagging mechanism
would predict under-covers MOST -- this measurement cannot distinguish that
from noise, and per the closing-grounds taxonomy below, it does not need to.

## Closing-grounds taxonomy (binding, restated verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

**Verdict applied here**: the [-0.4698, +0.3625] interval crosses zero --
the expected shape at this resolution, not a rejection. Neither admissible
closing ground is cleared (the interval does not sit entirely below zero;
no positive control was run). Recorded as `unresolved_below_power`, no
`closing_ground`, `registry/weak_signals.json:qb_news_backup_visible_by_deadline_screen`.

## 7. Provenance

- **Measured this session**: every number in Sections 6.1-6.3, the
  first (buggy, unbounded-lookback) dry run and its 94.2% reading, the
  10-day-lookback fix and re-run, the SNF/MNF materiality check (3/916,
  313/916 exposed), the deadline-corrected final run, the registry record.
- **Read this session**: `docs/prospective_evidence.md` (the audit that
  killed the schedule-column path), `docs/backup_qb_fade_overlay.md`,
  `docs/injury_news_sourcing.md` (full, including sec 5.1's PFT matching
  precedent), `docs/pfr_transactions_sourcing.md` (full),
  `scripts/nfl_bias_battery_screen.py`, `src/nfl_ats/backup_qb_fade_overlay.py`,
  `src/nfl_ats/experiment_runner.py` (the bias-battery long-table construct
  and subset-bias scaling/classification helpers),
  `scripts/injury_tuesday_cutoff_experiment.py` (name-normalization and
  PFT-matching precedent), `registry/weak_signals.json`.
- **Reported (owner)**: the two owner corrections dated 2026-08-20 (picks
  editable up to a per-game deadline; that deadline is `min(kickoff, Sunday
  16:00 ET)`), taken as binding instructions, not independently re-derived
  from a pool rules document.
- **Inferred**: the "foreshadowed backups may already be partly priced"
  explanation in 6.3 for why the news-visible subset's effect is weaker than
  the full backup-start population's -- my reasoning, not evidence.
- **Not done**: no overlay logic, live pick path, or refresh pipeline was
  touched or wired -- per the task's explicit scope, another agent owns that
  integration. No rotation-registry window was drawn or spent.
