# FluView home-market illness battery: predeclaration

Written 2026-08-19, **before any cover-rate sign in this battery has been
examined**, per the data-source scout's #3 recommendation
(`docs/data_source_scout_v2.md`, "Recommended top 3 for immediate
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
