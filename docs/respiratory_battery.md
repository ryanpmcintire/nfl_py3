# Total-respiratory-illness home-market battery: predeclaration

Written 2026-08-26, **before any cover-rate sign in this battery has been
examined**, extending ``docs/fluview_battery.md`` (state-level influenza-like
-illness, split-half reliability 0.9814, two cells beating the coin flip)
from influenza alone to total respiratory illness burden. Mechanism: ILI is
one disease; what plausibly matters for a locker room / a home crowd is the
whole seasonal respiratory-virus load in that market, not flu specifically.
This document freezes cells, sources, thresholds, and the point-in-time
construction before scoring, mirroring ``docs/fluview_battery.md``'s
structure and admissible pre-scoring exceptions (the peak-week window,
reused unchanged rather than re-derived -- see section 4).

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

## 1. Source survey and API verification (measured, this session)

**What was investigated**: the Delphi Epidata documentation
(`https://cmu-delphi.github.io/delphi-epidata/`) and the live
`https://api.delphi.cmu.edu/epidata/covidcast/` endpoint, specifically for
a state-level weekly respiratory-illness indicator with REVISION HISTORY
(an `issues`/`as_of` mechanism, the same requirement FluView's `issues`
parameter satisfied). **Decisive finding**: Delphi's `covidcast` API
carries the **NSSP** (National Syndromic Surveillance Program) source,
signals `pct_ed_visits_covid` / `pct_ed_visits_influenza` /
`pct_ed_visits_rsv` (the weekly percentage of emergency-department visits
associated with each pathogen) -- state-level, weekly, and genuinely
revision-tracked, i.e. exactly what the task's own worked example named.

- **Revision history, measured**: `covidcast`'s `issues=<range>` parameter
  (unlike FluView's exact-match-only `issues`) returns every distinct issue
  in the requested window directly. `regions=ca` (read: `geo_value=ca`)
  `pct_ed_visits_influenza` epiweek 202301 (an old, fully-backfilled week)
  carries **121 distinct issues** on file (first issue `lag=67`, i.e. 67
  weeks after the fact -- 2023 was still backfill-heavy). A THEN-recent
  epiweek, 202510 (early 2025), already had its **first issue at `lag=1`**
  -- one week after the observed week -- and the value itself genuinely
  revised over subsequent reissues (1.75 -> 1.85 across ~15 later issues,
  not just carried forward unchanged). This is denser, more-real-time
  revision behavior than FluView's own post-2018 floor (docs/fluview_
  battery.md section 1's densest observed cadence).
- **Geography**: state-level (`geo_type=state`) is directly supported, 51
  locations (50 states + DC) at every checked epiweek from launch onward.
  **Measured**: at the launch epiweek (202239) all 51 already report (no
  slow ramp-up), and at the current epiweek (202632) 49 of 51 report --
  the two missing are Iowa and South Carolina, **neither of which hosts an
  NFL franchise**, so this ingest's 23-state population is unaffected.
  Region codes are the same lower-case two-letter USPS abbreviations
  FluView uses, so `STATE_BY_TEAM` (`scripts/fluview_battery_ingest.py`,
  imported here, NOT redefined) applies unchanged.
- **Coverage floor, measured (the decisive scope limitation)**: NSSP
  state-level data begins at epiweek **202239** (late September 2022),
  confirmed via `covidcast`'s metadata endpoint
  (`https://api.delphi.cmu.edu/epidata/api.php?source=covidcast_meta`,
  `min_time: 202239` for every NSSP state-level signal) and cross-checked
  live. This is a hard start, not FluView's gradual pre-2010/pre-2017-18
  two-tier floor -- but it is much LATER than FluView's raw-data floor
  (2010) and even later than FluView's own point-in-time-recoverable floor
  (~2017-18). **Consequence, disclosed before scoring**: this battery can
  score at most NFL seasons 2022-2026 (~5 seasons) vs FluView's ~8
  effective seasons -- a real reduction in scored population, reported
  per-season in the results JSON, not corrected around.
- **The official combined signal is DEAD, measured**: `covidcast_meta`
  shows `pct_ed_visits_combined` (all three pathogens pre-summed by CDC)
  stopped updating at state level after epiweek **202439** (`max_time:
  202439`, `last_update` timestamp corresponding to June 2025, no further
  revisions since), while the three per-pathogen raw signals continue
  updating through the current epiweek (202632 at ingest time). **This
  battery therefore sums the three per-pathogen AS-OF values itself**
  (section 3) rather than using CDC's own combined signal, which cannot be
  extended past 2024w39.
- **Anonymous rate limit, measured**: covidcast enforces "only two
  parameters may have multiple selections" (matching FluView's own docs
  page, but measured independently here) -- a THIRD multi-selection
  parameter is rejected with HTTP 401 "Requested too many multiples for
  anonymous queries" (measured: `signal=<3 values>` + `issues=<range>` +
  `time_values=<range>` together = 401; any 2 of those 3 together = 200).
  Getting one state's full point-in-time history already spends both
  slots on `time_values=<range>` + `issues=<range>`, so this ingest needs
  one request per (state, signal) pair: 23 states x 3 signals = **69
  requests**, which -- unlike FluView's 24-request ingest -- EXCEEDS the
  60-requests/hour anonymous cap on its own. `scripts/respiratory_battery_
  ingest.py` paces requests at a fixed interval (~62s, ~58/hour) so the
  run does not trip the limiter, rather than relying on 429 backoff to
  recover from a blown quota (backoff is kept as a safety net only,
  unchanged from FluView's ingest).
- **No release_date field**: unlike FluView's raw response (which carries
  a real calendar `release_date`), `covidcast` rows for a weekly signal
  carry only an `issue` field in `YYYYWW` epiweek format. Section 3
  explains the conservative epiweek -> calendar-date conversion this
  battery uses to keep the existing `release_date`-keyed checkpoint
  machinery (`scripts/fluview_battery_screen.py`'s
  `build_checkpoint_tables` / `asof_lookup`, imported and reused
  UNMODIFIED) applicable without rewriting it.

## 2. Team -> state mapping (reused, not redefined)

Identical to `docs/fluview_battery.md` section 2: `STATE_BY_TEAM`
(`scripts/fluview_battery_ingest.py`) maps all 34 historical nflverse team
codes (including the OAK/SD/STL relocation aliases) to 23 unique states.
This battery imports that dict directly rather than restating it. The same
population restriction applies: `location == "Home"` REG games only (drops
neutral-site/displaced-stadium games, ~1-2% of the slate), for the same
reason FluView gave -- at a non-home site the home-market mechanism does
not apply and this battery does not build an away-market-of-travel
substitute. The same DC note applies (`WAS` -> `MD`, no `DC` region code
needed since Delphi's covidcast state list has no separate DC-as-home-
market entry any NFL team would map to either).

## 3. Point-in-time-safe as-of construction (measured algorithm)

**Per-pathogen checkpoint tables, reusing FluView's machinery verbatim**:
for each of the three NSSP signals (`pct_ed_visits_covid` / `_influenza` /
`_rsv`) and each state, the raw `(time_value, issue, value)` rows from
`scripts/respiratory_battery_ingest.py`'s snapshot are reshaped into the
SAME column schema FluView's raw frame carries (`region`, `epiweek` <-
`time_value`, `issue`, `release_date`, `ili` <- `value`), with
`release_date` derived from `issue` per the conversion below. The reshaped
frame is then passed UNCHANGED to
`scripts/fluview_battery_screen.py`'s `build_checkpoint_tables` and
`asof_lookup` (imported, not reimplemented) -- the identical
monotone-in-`known_epiweek` running-max checkpoint construction and
`merge_asof(cutoff_date, checkpoint, direction="backward")` lookup FluView
uses, per FluView's own section 3 (read that first).

**Issue -> release-date conversion (new, needed because `covidcast` has no
`release_date` field for a weekly signal)**: `issue` is a CDC epiweek
(`YYYYWW`). This battery maps it to the **Saturday ending that epiweek**
(the last calendar day of the Sunday-Saturday MMWR week), via
`epiweek_to_release_date()`, the algebraic inverse of `cdc_epiweek()`
(`scripts/fluview_battery_screen.py`, imported unchanged; its two internal
date helpers were hoisted to module level, unchanged in behavior, so this
inverse can reuse them rather than re-deriving the epiweek calendar from
scratch). **This is deliberately conservative, not precise**: NSSP's
documented cadence is "weekly, Friday mornings" (Delphi's own NSSP signal
page), which would put the true release a day or two BEFORE this
battery's assumed Saturday anchor. Using the epiweek's last day rather
than its likely-true release day can only make this battery's as-of
values MORE stale (never less), i.e. it can only cost detectable signal,
never leak future information -- the safe direction for a
point-in-time-safety assumption this battery cannot verify at the same
granularity FluView could (FluView had a literal `release_date` field to
check against; this battery does not, so it errs conservative instead of
assuming precision it cannot confirm).

**Total respiratory AS-OF value**: `respiratory_total = covid_asof +
flu_asof + rsv_asof`, computed **only when all three pathogen AS-OF values
are non-missing** at a game's decision cutoff; if any one of the three is
missing, `respiratory_total` is missing for that team/cutoff -- never a
partial two-pathogen sum, matching FluView's "missing not defaulted"
convention exactly (section 3 of that document).

**Decision cutoff**: identical to FluView -- the Tuesday of the game's own
week (`tuesday_offset = (weekday - 1) % 7`, Monday=0), same convention as
`scripts/attention_battery_screen.py` and `scripts/fluview_battery_
screen.py`.

**Threshold ("top decile of that state's own history")**: identical rule
to FluView section 3 -- the 90th percentile of that state's own AS-OF
`respiratory_total` panel (one value per state per NFL season-week),
computed once, frozen, not re-derived per season. Missing AS-OF values are
excluded from both the threshold computation and every cell (reported as
`n_excluded_missing`), never defaulted to non-elevated.

**Expected consequence, disclosed before scoring (revised below with the
MEASURED post-run reality)**: given section 1's measured 202239 floor,
this battery's scored population was expected to concentrate in NFL
seasons **2022 (partial, from roughly week 3 onward) through 2026**; the
nominal SEASON_START/SEASON_END passed to the screen script is 2022-2026,
and per-season AS-OF coverage is reported in the results JSON exactly as
FluView reports its own pre-2017-18 near-total missingness.

**Correction, measured 2026-08-26 after the real ingest+screen ran**: the
actual floor is much narrower than the pre-scoring estimate above. Coverage
by season (`artifacts/respiratory_battery/20260826T133901Z/results.json`,
**measured**): 2022 **0.0%**, 2023 **0.0%**, 2024 **97.0%**, 2025
**96.6%**. Spot-checked (**measured**, this session) to confirm this is a
genuine publication-timing fact, not a bug: for California/covid, EVERY
2022-season epiweek's (202240-202252) earliest available issue is
**202416** (`release_date` 2024-04-20) -- NSSP's public backfill for late
2022 was not released until April 2024, matching this document's own
section-1 measurement that epiweek 202301's first issue arrived at
`lag=67` (i.e. mid-2024). Because the as-of algorithm enforces "missing
until actually released" by construction (section 3, no fallback), NO game
played during the 2022 or 2023 seasons can ever have a non-missing
`respiratory_total` -- the games happened before NSSP had published
ANYTHING about that period. **This battery's effective scored window is
NFL seasons 2024-2025 only** (`n_state_seasons=44` in the reliability
check, `n_seasons=2` in every cell's season-blocked secondary), not the
nominal 2022-2026 -- a measured data-availability fact, disclosed here
exactly as it was found, not corrected around or hidden by loosening the
point-in-time-safety rule.

## 4. Late-season respiratory-peak window (REUSED, not re-derived)

This battery reuses `scripts/fluview_battery_screen.py`'s frozen
`PEAK_WEEKS` constant (`{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53}`,
imported unchanged) rather than re-deriving a fresh peak-week set from
NSSP's own national series. Two reasons, both stated plainly: (1) the
underlying fact this window encodes -- respiratory illness of ALL kinds
concentrates in the same winter calendar window in the US -- is a general
epidemiological fact FluView's own predictor-only measurement already
established from national ILI, not a flu-specific artifact this battery
would need to re-derive; (2) re-deriving a fresh window from NSSP's own
national series would require a second predictor-distribution-only
pre-scoring exception on this battery's OWN data, and reusing an
already-frozen, independently-measured window is the more conservative
choice (it cannot have been shaped, even innocently, by anything in this
battery's own respiratory cover-rate results, since it predates this
document).

## 5. Predeclared cells (5)

Population for all cells: NFL REG 2022-2026, `location == "Home"` (section
2), close-graded via `schedules.spread_line`
(`nfl_ats.features.add_ats_outcomes`, pushes dropped, identical convention
to FluView and `scripts/nfl_weather_battery_screen.py`). Method: joint
week-blocked bootstrap (block = `season*100+week`) PRIMARY, season-blocked
bootstrap SECONDARY -- both `block_bootstrap_two_group`-identical to
FluView / `scripts/nfl_weather_battery_screen.py` (imported from
`scripts/_common.py`, reused not reimplemented). Full-slate effect
scaling: `raw_gap_pts * fraction_of_slate` via
`nfl_ats.experiment_runner.scale_subset_effect` (imported). 20,000
bootstrap samples, seed 20260826 (repo convention: today's date). Within-
week correlation is zero by owner mandate -- no ICC term.

Every cell is a **plain subset-vs-complement game-level test on
`home_cover`**, one flag per game built as described in section 3. Rows
with a missing required AS-OF `respiratory_total` value (either side, as
applicable to that cell) are excluded from BOTH the subset and complement
of that cell, not defaulted, and reported as `n_excluded_missing`.

**R1. `respiratory_home_market_elevated`** -- home team's own state AS-OF
`respiratory_total` (covid+flu+rsv ED-visit %) in that state's own top
decile (section 3) vs. not. **Predicted sign: NEGATIVE** on `home_cover`
-- elevated total respiratory-illness burden in the home team's own
market plausibly degrades the home team specifically, and the market is
not known to price this. Mirrors FluView F1, generalized from ILI alone
to covid+flu+rsv combined.

**R2. `respiratory_away_market_elevated`** -- away team's own state AS-OF
`respiratory_total` in that state's own top decile vs. not. **Predicted
sign: POSITIVE** on `home_cover` -- mirror mechanism, the away team's own
market burden degrades them, favoring the home side. Mirrors FluView F2.

**R3. `respiratory_differential_home_worse`** -- restricted to games where
**exactly one side** is elevated (home XOR away, both sides' AS-OF values
required non-missing); subset = home elevated & away not, complement =
away elevated & home not. **Predicted sign: NEGATIVE** on `home_cover` --
isolates the relative/differential form of the same mechanism. Mirrors
FluView F3 (which scored on a small population, 124 games; this battery's
narrower 2022-2026 window makes an even smaller population plausible,
expected and reported, not corrected around).

**R4. `respiratory_peak_home_elevated`** -- restricted to section 4's
reused predeclared late-season peak weeks; home team's own state AS-OF
`respiratory_total` elevated vs. not. **Predicted sign: NEGATIVE** -- same
mechanism as R1, tested where the underlying epidemiological signal is
largest. Mirrors FluView F4.

**R5. `respiratory_peak_away_elevated`** -- same peak-week restriction;
away team's own state AS-OF `respiratory_total` elevated vs. not.
**Predicted sign: POSITIVE** -- mirror of R4, parallel to the R1/R2
relationship. Mirrors FluView F5.

## 6. Reliability check (measured, run before cover-rate scoring)

Split-half reliability of the underlying state-respiratory-elevated trait,
via `nfl_ats.cfb_qb_dependence.split_half_reliability` (reused directly,
same function FluView/PBP-05/PBP-08/injury-value-lost figures were built
on), applied to the state-week AS-OF `respiratory_total` panel (`team_id`
<- state code, `season` <- NFL season, `week` <- state-week's own epiweek
`WW` parity for the odd/even split), metric = the raw AS-OF
`respiratory_total` value (Pearson r is scale-invariant, so raw vs.
z-scored is immaterial). This tests whether "this state is running hot
this season-half, across covid+flu+rsv together" is a real, non-noise
persistent trait within a season -- the assumption every cell's decile
flag depends on. Reported in the results JSON and in the registry
`--reliability` field for every cell (conservative: the same figure
applied to all 5 cells, since all 5 share the identical underlying AS-OF-
elevated construct). Per AGENTS.md, a `no_split_half_reliability` closing
ground requires this figure's CI to sit AT (not near) zero -- an interval
crossing zero here is, as everywhere else, not grounds to close.

## 7. Files

- `scripts/respiratory_battery_ingest.py` -- per-(state, signal) NSSP
  fetch (69 requests, paced under the anonymous rate cap), writes
  `data/raw/respiratory/<UTC>/respiratory_raw.parquet` + `manifest.json`
  (gitignored, per repo convention).
- `scripts/respiratory_battery_screen.py` -- issue -> release-date
  conversion, per-pathogen as-of construction reusing FluView's checkpoint
  machinery verbatim, `respiratory_total` summation, cell scoring, writes
  `artifacts/respiratory_battery/<UTC>/results.json` (measure-only, no
  registry writes).
- `scripts/respiratory_battery_record.py` -- records all 5 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`, verifies
  after writing.
- `tests/test_respiratory_battery_leakage.py` -- leakage regression test:
  proves a revision issued after a game's decision-cutoff Tuesday cannot
  reach that game's `respiratory_total` AS-OF feature.
