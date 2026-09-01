# FluView home-market illness battery: predeclaration

Written 2026-08-19, **before any cover-rate sign in this battery has been
examined**, per the data-source scout's #3 recommendation
(`docs/archive/data_source_scout_v2.md`, "Recommended top 3 for immediate
ingestion" item 3): CDC Delphi FluView state-level influenza-like-illness
(ILI) history as a pregame covariate. Mechanism: team-level illness burden
is real (flu waves hit locker rooms, not just fans), is public weekly data,
and is plausibly unpriced at the Tuesday opener because no sportsbook is
known to model it. This document freezes cells, thresholds, and scaling
before scoring; the reliability check and the peak-week window below are
the two admissible pre-scoring exceptions (computed on the PREDICTOR's own
distribution, never on a cover-rate outcome), matching the precedent in
`docs/team_style.md`'s "Reliability gate" section.

## Binding taxonomy (owned verbatim, per AGENTS.md / CLAUDE.md)

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

## 1. API verification (measured, this session)

`https://api.delphi.cmu.edu/epidata/fluview/` -- free, no key, live.

- **Basic call**: `regions=pa&epiweeks=201040-201045` returned 6 real
  versioned rows (`release_date`, `issue`, `epiweek`, `lag`, `ili`, `wili`,
  provider/patient counts). **Measured**.
- **`issues` parameter is EXACT-MATCH, not "latest as of"**: querying a
  single `issues=201845` for `epiweeks=201840` returns only the row whose
  `issue` field equals 201845 exactly (or nothing, if that exact issue
  was never published) -- it does not search backward for the freshest
  prior issue. **Measured** (`ca`, epiweek 201840, issue 201845 -> exactly
  one row, `lag=5`). Consequence: reconstructing "what was known as of
  date X" requires fetching a RANGE of issues and filtering client-side,
  which is what the ingestion script does (see section 3).
- **Critical finding, changes the point-in-time-safety picture from
  `data_source_scout_v2.md`**: state-level VERSION HISTORY (multiple
  distinct issues per epiweek, i.e. real revision tracking) is
  effectively **absent before the 2017-18 season** and only becomes dense
  (near-weekly re-issue, `lag=0` issues appearing) starting **epiweek
  201740-201840**. **Measured**: a 50-week-wide `issues` window around
  epiweeks 201003/201103/.../201603 (2010-2016) returned **zero** rows
  each time -- meaning for those years, Epidata has captured only ONE
  issue ever (the eventual final revision, `lag` in the hundreds of
  weeks: e.g. `pa` epiweek 201040 shows `issue=201740, lag=365`; epiweek
  201542 shows `issue=201740, lag=102`). Epiweek 201703 shows the first
  crack (13 issues, `min_lag=37`); by 201740 `min_lag` reaches 0, and
  201801 onward shows dense weekly reissue (48-51 issues per epiweek in a
  50-week window). Cross-checked against Delphi's own published note
  (**read**, `https://cmu-delphi.github.io/delphi-epidata/api/fluview.html`):
  "State-level data was not publicly available from the CDC prior to the
  2010-2011 flu season (2010w40)" -- confirming the RAW-data floor the
  task specifies, but silent on the separate, more restrictive
  VERSION-HISTORY floor measured above. **Both floors are disclosed
  below; they are not the same number.**
- **Region codes**: lower-case two-letter USPS state abbreviations work
  directly (`pa`, `ca`, ...), confirmed against Delphi's own docs
  (**read**) which also list `nat`, `hhs1-10`, `cen1-9`, and city codes.
- **Rate limiting, fully explained (read, Delphi's own docs page)**:
  anonymous (no API key) access is capped at **60 requests/hour**, "public
  datasets only," and "only two parameters may have multiple selections."
  A free API key (self-service registration, email only, at
  `https://api.delphi.cmu.edu/epidata/admin/registration_form`) removes
  the rate limit entirely; none was configured this session (checked the
  environment, no `DELPHI_API_KEY`/similar). This session's own
  verification probing (roughly 30-40+ exploratory calls across the
  bisection/coverage checks above) consumed the hourly budget partway
  through, which is why the full 24-request ingest (23 states + `nat`,
  section 3) had to be sequenced around a wait for the window to clear
  rather than run immediately after verification. 24 requests fits
  comfortably inside 60/hour on its own; the constraint bound this
  session's own testing overhead, not the ingest itself. The ingestion
  script (section 3) uses one bulk request per state (entire 2010-2025
  range in one call, confirmed to return 33-34k rows in ~9MB / ~1.5s
  without truncation) specifically to make the real ingest cheap in
  request count, plus exponential backoff as a safety net for transient
  429s.
- **Bulk feasibility, measured**: one request for `regions=ca`,
  `epiweeks=201040-202552`, `issues=201040-202552` returned 33,430 rows,
  9.09MB, in 1.48s, `result: 1 (success)`, no truncation. One call per
  state is therefore sufficient for the full history.

**One more measured gap, found during ingestion**: the `ny` region (Buffalo's
state) returns `release_date: null` on EVERY row, for every epiweek/issue
-- confirmed live (`regions=ny&epiweeks=201040-201045` returns real
`ili`/`issue`/`lag` values but `"release_date": null` on all three sampled
rows), not a parsing artifact on this project's side. Because the as-of
algorithm (section 3) keys entirely on `release_date`, `ny` cannot be
resolved to ANY as-of value across the whole 2010-2025 range -- Buffalo's
home and away markets are both **entirely missing** in every cell below,
not defaulted to non-elevated. This is disclosed here, before scoring, as
a known coverage gap, not corrected around.

**Net effect on the point-in-time-safety claim**: the API's issue/lag
mechanism is real and exactly as strong as advertised for the 2017-18
season onward. For 2010-2017, Epidata simply never captured an early
version, so a strict as-of query correctly returns **missing**, not a
leaked final value -- see section 3's algorithm, which enforces this by
construction rather than assuming a fallback convention.

## 2. Team -> state mapping (static, season-invariant on nflverse codes)

`schedules.parquet` `home_team`/`away_team` codes already carry the
season-correct franchise code for relocations (`STL` retired after 2015,
`LA` used from 2016; `SD` retired after 2016, `LAC` used from 2017; `OAK`
retired after 2019, `LV` used from 2020 -- matching
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`), so a single static
code -> state dict needs no season conditioning:

| code | state | code | state | code | state | code | state |
|---|---|---|---|---|---|---|---|
| ARI | AZ | DET | MI | LA | CA | PHI | PA |
| ATL | GA | GB | WI | LAC | CA | PIT | PA |
| BAL | MD | HOU | TX | LV | NV | SD | CA |
| BUF | NY | IND | IN | MIA | FL | SEA | WA |
| CAR | NC | JAX | FL | MIN | MN | SF | CA |
| CHI | IL | KC | MO | NE | MA | STL | MO |
| CIN | OH | | | NO | LA (Louisiana) | TB | FL |
| CLE | OH | | | NYG | NJ | TEN | TN |
| DAL | TX | | | NYJ | NJ | WAS | MD |
| DEN | CO | | | OAK | CA | | |

23 unique states cover all 34 historical codes. **Two-team-or-more
states** (handled, not avoided, per task instruction): CA (up to 4
concurrent franchises across the window: LA/LAC/OAK/SF), FL (JAX/MIA/TB,
3), TX (DAL/HOU), OH (CIN/CLE), PA (PHI/PIT), NJ (NYG/NYJ), MD (BAL/WAS).
For these states, two different teams' home games in the same week draw
the identical state-level ILI reading -- an inherent granularity limit of
state- (not metro-) level surveillance, disclosed, not corrected.

**DC note**: Washington's stadium (FedExField / Northwest Stadium) is in
Landover, **Maryland**, not the District of Columbia -- `WAS` maps to
`MD`, not a nonexistent `DC` region code (Delphi's region list has no DC
entry for FluView state-level data in any case).

**International / neutral-site note**: `schedules.parquet`'s `location`
column already flags every game where the designated home team is NOT
playing in its own market -- confirmed (**measured**, this session) it
correctly catches not just the obvious London/Mexico City/Germany/Brazil/
Spain/Australia neutral games (Wembley, Azteca, Tottenham, Deutsche Bank
Park, Allianz Arena, Arena Corinthians, Bernabeu, Estadio Banorte, Stade
de France, Melbourne Cricket Ground) but also domestic displaced-stadium
games (2010 MIN home game at Ford Field after the Metrodome roof
collapse; 2020-21 SF home games at State Farm Stadium, AZ, under COVID
restrictions; several early Buffalo "Toronto Series" games at Rogers
Centre). **This battery restricts its population to `location == "Home"`
games only** (drops ~1-2% of the REG slate) -- at a neutral/displaced
site, the home team's own home-market crowd/environment component of the
mechanism does not apply, and using the away team's actual travel-market
state instead of their nominal home state would require data this
battery does not build. This is the "note DC/international" handling the
task asked for.

## 3. Point-in-time-safe as-of construction (measured algorithm)

Because `issues` is exact-match only (section 1), the ingestion script
(`scripts/fluview_battery_ingest.py`) pulls, per state, the FULL
2010-2025 multi-issue history in one bulk call, then the screen script
(`scripts/fluview_battery_screen.py`) builds a per-state **as-of
checkpoint table**: sort all `(epiweek, issue, release_date, ili)` rows by
`release_date` ascending, and at each `release_date` step, carry forward
whichever `(epiweek, ili)` pair has the highest `epiweek` released so far
(a running max). This produces a monotone-in-`known_epiweek` table indexed
by real calendar `release_date`.

**Decision cutoff**: the Tuesday of the game's own week, i.e. the most
recent Tuesday on or before `gameday` -- identical convention to
`scripts/attention_battery_screen.py`'s `tuesday_offset = (weekday - 1) %
7` (Monday=0). Every game in the same numbered NFL week shares this same
calendar Tuesday regardless of Thu/Sun/Mon scheduling.

**As-of value** = `merge_asof(cutoff_date, checkpoint_table, on=release_date,
direction="backward")`: the freshest ILI reading whose `release_date` is
`<= cutoff_date`. If no checkpoint row qualifies (true for nearly all of
2010-2017, per section 1's measured finding), the value is **missing**,
not a leaked final value -- the algorithm enforces the safety property by
construction rather than by a documented-but-unverified convention.

**Expected consequence, disclosed before scoring**: this battery will
show two season floors, not one:
- **Raw-data floor**: 2010-10 (CDC's own public-availability date for
  state-level ILINet, matching the task's stated instruction).
- **Point-in-time-recoverable floor**: measured this session at
  approximately the 2017-18 season -- games before that will have very
  high (likely near-total) missingness on the as-of feature and
  contribute little to the scored population. The 2010-2025 ingest is
  still run in full (per the task instruction), and per-season coverage
  is reported in the results JSON, but the EFFECTIVE scored window is
  expected to concentrate in 2017-18 through 2024-25 (8 seasons). This is
  a measured data-availability fact, not a design choice to exclude early
  seasons.

**Threshold ("top decile of that state's own history")**: computed ONCE
from the full panel of that state's own AS-OF values (one value per
state per NFL season-week, not per raw epiweek row, and not per game --
teams sharing a state share the same panel entry), at the 90th
percentile, per state. This mirrors `docs/team_style.md`'s frozen-
threshold convention (computed once from the full panel, not re-derived
per season, so it cannot silently drift as more data is recorded) and is
computed on the AS-OF values themselves (the values actually used to
score games), not on final revisions, since a decile threshold defined
against a different distribution than the one being classified would be
internally inconsistent. Missing AS-OF values do not count toward the
threshold or the flag; those rows are excluded from every cell (reported
as `n_missing_required_data`), not defaulted to non-elevated.

## 4. Late-season flu-peak window (measured, predictor-only, predeclared)

To predeclare "late-season flu-peak weeks" from data rather than folk
knowledge, the national (`nat`) FINAL ILI series 2010-2025 was averaged by
calendar week-of-year (**measured**, this session, before any cover-rate
sign was examined -- a predictor-distribution-only computation, the same
admissible exception `docs/team_style.md` uses for its reliability gate).
Top-quartile weeks by mean national ILI:

`{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53}` (ISO-style week-of-year
from the epiweek's own `WW` component; national mean ILI ranges from
2.93-4.93 in this set vs. a low of 1.04-1.20 in the summer trough weeks
24-33). This exact set is frozen and reused by both peak-week cells below.

## 5. Predeclared cells (5)

Population for all cells: NFL REG 2010-2025, `location == "Home"`
(section 2), close-graded via `schedules.spread_line`
(`nfl_ats.features.add_ats_outcomes`, pushes dropped, same convention as
`scripts/nfl_weather_battery_screen.py`). Method: joint week-blocked
bootstrap (block = `season*100+week`) PRIMARY, season-blocked bootstrap
SECONDARY -- both `block_bootstrap_two_group`-identical to
`scripts/nfl_weather_battery_screen.py` (read; same vectorized multinomial
block-resample algorithm, reused not reimplemented). Full-slate effect
scaling: `raw_gap_pts * fraction_of_slate` (identical arithmetic to
`nfl_ats.experiment_runner.scale_subset_effect`, `accuracy_points` units).
20,000 bootstrap samples, seed 20260819 (repo convention: today's date).
Within-week correlation is zero by owner mandate -- no ICC term.

Every cell is a **plain subset-vs-complement game-level test on
`home_cover`**, one flag per game built as described in section 3. Rows
with a missing required AS-OF value (either side, as applicable to that
cell) are excluded from BOTH the subset and complement of that cell, not
defaulted, and reported as `n_missing_required_data`.

**F1. `fluview_home_market_elevated`** -- home team's own state AS-OF ILI
in that state's own top decile (section 3) vs. not. **Predicted sign:
NEGATIVE** on `home_cover` -- elevated illness burden in the home team's
own market plausibly degrades the home team specifically (their own
locker room, their own community), and the market is not known to price
this.

**F2. `fluview_away_market_elevated`** -- away team's own state AS-OF ILI
in that state's own top decile vs. not. **Predicted sign: POSITIVE** on
`home_cover` -- mirror mechanism, the away team's own market burden
degrades them, favoring the home side.

**F3. `fluview_differential_home_worse`** -- restricted to games where
**exactly one side** is elevated (home XOR away, both sides' AS-OF values
required non-missing); subset = home elevated & away not, complement =
away elevated & home not. **Predicted sign: NEGATIVE** on `home_cover` --
isolates the relative/differential form of the same mechanism, the
cleanest test since it removes games where both or neither side carries
the exposure.

**F4. `fluview_peak_home_elevated`** -- restricted to section 4's
predeclared late-season peak weeks; home team's own state AS-OF ILI
elevated vs. not. **Predicted sign: NEGATIVE** -- same mechanism as F1,
tested where the underlying epidemiological signal is largest and most
differentiated across states/weeks.

**F5. `fluview_peak_away_elevated`** -- same peak-week restriction; away
team's own state AS-OF ILI elevated vs. not. **Predicted sign:
POSITIVE** -- mirror of F4, parallel to the F1/F2 relationship.

## 6. Reliability check (measured, run before cover-rate scoring)

Split-half reliability of the underlying state-ILI-elevated trait, via
`nfl_ats.cfb_qb_dependence.split_half_reliability` (reused directly, the
same function PBP-05/PBP-08/injury-value-lost figures were built on),
applied to the state-week AS-OF panel (`team_id` <- state code, `season`
<- NFL season, `week` <- state-week's own epiweek `WW` parity for the
odd/even split), metric = the raw AS-OF `ili` value (Pearson r is scale-
invariant, so raw vs. z-scored is immaterial to the reliability figure
itself). This tests whether "this state is running hot this
season-half" is a real, non-noise persistent trait within a season, which
is the assumption every cell's decile flag depends on. Reported in the
results JSON and in the registry `--reliability` field for every cell
(conservative: the same figure applied to all 5 cells, since all 5 share
the identical underlying AS-OF-elevated construct). Per AGENTS.md, a
`no_split_half_reliability` closing ground requires this figure's CI to
sit at (not near) zero -- an interval crossing zero here is, as
everywhere else, not grounds to close.

## 7. Files

- `scripts/fluview_battery_ingest.py` -- bulk per-state fetch, writes
  `data/raw/fluview/<UTC>/fluview_raw.parquet` + `manifest.json`
  (gitignored, per repo convention).
- `scripts/fluview_battery_screen.py` -- as-of construction, cell
  scoring, writes `artifacts/fluview_battery/<UTC>/results.json`
  (measure-only, no registry writes).
- `scripts/fluview_battery_record.py` -- records all 5 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`,
  verifies after writing.

## 8. Results (dated addendum, 2026-08-31)

**Provenance of this section**: this session found the battery already
fully built by an earlier one -- ingest (`data/raw/fluview/20260820T003258Z/`,
809,716 rows, 24/24 regions succeeded), scoring
(`artifacts/fluview_battery/20260820T003505Z/results.json`), and all 5 cells
already present in `registry/weak_signals.json` at `recorded_at: 2026-08-20`
(**read**, `registry/weak_signals.json`, keys
`fluview_home_market_elevated`/`fluview_away_market_elevated`/
`fluview_differential_home_worse`/`fluview_peak_home_elevated`/
`fluview_peak_away_elevated`). This section records the **measured
verification and refresh done this session**, not a from-scratch rebuild.

**Reproduction (measured, this session)**: re-ran
`scripts/fluview_battery_screen.py` against the existing raw FluView
snapshot and the current latest `schedules.parquet` snapshot
(`data/raw/20260824T115346Z/`, four days newer than the one originally
used) -- every cell's estimate, interval, and `probability_positive`
reproduced **bit-for-bit identical** to the already-recorded registry
values. The extra REG-season games available in the newer schedules
snapshot fall outside `SEASON_END=2025` and do not affect the scored
population. New artifact: `artifacts/fluview_battery/20260831T145604Z/results.json`.

**Decision: did not re-invoke `weak-signals record`.** `record_signal`
(`src/nfl_ats/weak_signals.py`) refuses to overwrite an existing name
without `--replace` by design ("a silently overwritten effect would let a
second look at the same signal masquerade as new evidence"). Since this
session's numbers are identical to the already-recorded ones, invoking
`--replace` would only update a timestamp while writing to a registry file
two other agents are concurrently editing this session -- an unforced risk
for zero evidentiary gain. The existing 2026-08-20 entries are the correct
record of this measurement; they were read back and verified in place
rather than rewritten.

**Reliability (measured, section 6)**: `n_state_seasons=190`,
`pearson_r=0.9636`, 95% CI `[0.9487, 0.9752]`, Spearman-Brown corrected
`0.9814`, `probability_positive=1.0000` -- the state-week AS-OF `ili` trait
is highly reliable within a season; the reliability gate is not the
constraint on any of these cells.

**Coverage by season (measured)**: 0% for 2010-2016 (confirms section 1's
version-history floor), 48.6% in 2017 (partial), then 87.7-97.3% every
season 2018-2025. Missingness in the full scored population:
`home_missing=1990/4009` (49.6%), `away_missing=1989/4009` (49.6%) --
matches the doc's predicted two-floor picture (section 3) almost exactly:
the raw-data floor (2010) contributes zero scorable rows, and the
point-in-time-recoverable floor concentrates the effective population in
2017-18 onward, as predeclared.

**One additional coverage fact, not in the original predeclaration
(measured, disclosed here rather than silently absorbed)**: the `fl`
checkpoint table's earliest release is 2021-10-15, four seasons later than
every other state's 2017-10-24 (`az`/`ca`/`co`/.../`wi` all start
2017-10-24; only `fl` and `ny` differ -- `ny` per the doc's own disclosed
gap, `fl` newly measured here). `fl` hosts JAX/MIA/TB. This narrows those
three teams' effective home/away coverage further than the rest of the
league for 2017-2021, on top of (not instead of) the already-disclosed `ny`
gap. It changes no threshold or population-membership rule -- the existing
missing-data handling (section 3) already excludes any row without a
qualifying as-of value -- it is disclosed as an added data-coverage fact.

**Per-cell measured numbers** (week-blocked primary; season-blocked
secondary in brackets; effect units `accuracy_points`; all five
`classification: unresolved_below_power`, per AGENTS.md -- no cell's
interval sits entirely on one side of zero, which is the expected outcome
at this evaluator's resolution, not grounds to reject):

| cell | n_population (excl. missing) | n_flag | effect | 95% CI (week-blocked) | P+ (week-blocked) | 95% CI (season-blocked) | P+ (season-blocked) |
|---|---|---|---|---|---|---|---|
| `fluview_home_market_elevated` | 2019 (1990) | 206 | +0.3090 | [-0.4092, +0.9491] | 0.8179 | [-0.4445, +0.8291] | 0.8599 |
| `fluview_away_market_elevated` | 2020 (1989) | 214 | +0.3681 | [-0.2566, +1.0005] | 0.8826 | [-0.2009, +1.0300] | 0.9328 |
| `fluview_differential_home_worse` | 124 (3885) | 58 | -0.6109 | [-10.2926, +9.2276] | 0.4529 | [-7.2620, +5.6387] | 0.4290 |
| `fluview_peak_home_elevated` | 583 (3426) | 162 | +0.3761 | [-1.9742, +2.6483] | 0.6228 | [-1.7043, +2.5106] | 0.6348 |
| `fluview_peak_away_elevated` | 588 (3421) | 170 | -0.1066 | [-2.1755, +2.0346] | 0.4693 | [-1.8688, +2.2225] | 0.4707 |

Reading this by predicted sign, never by "contains zero": F1
(home-elevated, predicted NEGATIVE) came out the WRONG predicted sign
(effect positive, P+=0.82 that it's actually positive) but the interval is
nowhere near resolved wrong-sign (upper bound +0.95, nowhere close to
"whole interval below zero") -- `unresolved_below_power`, not refuted. F2
(away-elevated, predicted POSITIVE) came out the RIGHT predicted sign with
P+=0.88, the strongest single cell here, also `unresolved_below_power`
since the interval still crosses zero. F3 (the differential/XOR cell,
predicted NEGATIVE) is the smallest population (124 games) and widest
interval by far, effect in the predicted direction but P+ barely above a
coin flip (0.45 for positive, i.e. ~0.55 for the predicted-negative
direction) -- clearly underpowered, `unresolved_below_power`. F4/F5 (peak
weeks) both sit near a coin flip in `probability_positive` with wide
intervals from small flagged subsets (162/170 games); no cell here is
bounded by a positive control, so none qualifies for
`bounded_by_control`, and no cell's whole interval sits on the wrong side
of zero, so none qualifies for `wrong_sign_resolved`. All five remain
open, exactly as AGENTS.md requires when neither closing ground is met.

**Data hygiene added this session (no scientific content changed)**:
`scripts/fluview_battery_ingest.py` now computes and writes an
`output_sha256` field in `manifest.json` (mirroring
`data/players/participation/raw/`'s per-partition SHA-256 convention,
collapsed to one hash since FluView's ingest writes a single unpartitioned
parquet file, not one per season); the field was also backfilled onto the
already-existing `data/raw/fluview/20260820T003258Z/manifest.json` by
hashing the parquet file already on disk, without re-fetching from the API
(re-fetching would have cost 24 more anonymous-rate-limited requests for a
snapshot whose historical 2010-2025 content cannot have changed). This is
an addition to the frozen doc's section 7 file-manifest convention, not a
deviation from any frozen cell, threshold, or scaling.

**Leakage regression + unit tests added this session**:
`tests/test_fluview_battery_leakage.py` (9 tests, all passing) -- canaries
proving (1) a revision issued after a game's decision-cutoff Tuesday is
never visible to that game's AS-OF feature, both at the checkpoint-table
level and end-to-end through `attach_asof_ili`; (2) a stale late re-issue
of an OLD epiweek never overwrites a NEWER epiweek's already-known as-of
value; (3) the measured `ny` (and now `fl`-adjacent) all-null-`release_date`
gap resolves to a missing value through the actual production code path,
never a leaked or defaulted one; (4) a missing AS-OF value or a
below-floor state threshold is flagged for exclusion, never silently
treated as "not elevated" evidence; (5) `compute_state_thresholds` drops
missing values before computing the decile and enforces the >=10-observation
floor; (6) `cdc_epiweek` matches the doc's own live-validated Delphi
example and handles both an ordinary and a 53-CDC-week year boundary; (7)
`build_state_week_panel` collapses a two-team state's home games in the
same week to one panel row, not one per game.

**Registry state**: `registry/weak_signals.json` total signal count
unchanged by this session's verification work (already 605 before and
after this session's read-only checks; the 5 FluView cells were already
counted in that total from 2026-08-20).
