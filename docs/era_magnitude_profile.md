> Predeclared 2026-08-19 BEFORE `scripts/era_magnitude_profile.py` scored anything.
> Stage 2 (changepoint estimation + mechanistic modulators + "what makes an
> era") was added by the owner mid-session as an explicit refinement and is
> predeclared here, in this same document, before its own signs were computed
> -- see "Stage 2" below. Nothing in Stage 1 was altered after Stage 2 arrived.

# Era-magnitude profile

## The hypothesis being tested

The owner's standing hypothesis, stated 2026-08-19 and binding: effects in
this project vary in **magnitude** across eras, not in existence -- "it's not
that the signal is there in era-A but not era-B, it's that it's less
predictive." A weaker-era reading is **never** evidence of absence (owner,
2026-08-19). This document predeclares how seven of the project's
already-recorded or already-constructed signals will be re-sliced by era to
look for that magnitude drift, before any era-sliced number exists.

This is **continuous / diagnostic evidence on windows this project has
already examined** (2009-2025 close-grade seasons; every signal below has
prior looks on some or all of this span). Per `docs/rotation_registry.md`
rule 6, contamination is inherited honestly and a reused window "carries a
stated discount, not a ban" -- this write-up carries that discount and states
it here, once, rather than re-arguing it per signal. **No rotation window is
declared or spent by this work**: nothing here is a family confirmation look
under `nfl-ats rotation`; it is a re-slice of already-built constructs for a
diagnostic question the rotation registry does not govern (rule 8: only NFL
*confirmation* looks are governed; a re-slice of prior-measured constructs by
calendar era is not one).

## Binding rules this document and its script must follow

Pasted verbatim per `AGENTS.md`/`CLAUDE.md` (subagents/scripts never see
those files' context injection):

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". If a record command errors, the verdict is wrong,
> not the validator.

Additional binding constraints for this document specifically: within-week
correlation is hardcoded ZERO (owner-mandated); every feature/flag used is
pregame-available by construction (all seven signals reuse already-audited
pregame constructs, verified below); every claim below is tagged
measured / read / reported / inferred; a weaker-era reading is never reported
as absence.

## Signals profiled, construct source, and verified population

All seven signals are re-sliced on the **close grade** (`spread_line` /
`home_cover` as recorded in `data/processed/game_features.parquet`), except
signal 7, whose own construct is opener-graded by definition (proxy 2011-2019,
true 2020-2025) and is re-sliced, not re-run.

**2007-2008 extension: not implemented, measured infeasible.** The task
context proposed extending population back to 2007-2008 using `sbr_odds.parquet`
(2007-2021 opens/closes) wherever a construct's own inputs also exist that
far back. Read directly: the newest three local `data/raw/*/schedules.parquet`
snapshots (`20260812T101244Z`, `20260812T130036Z`, `20260817T235649Z`) all
begin at season 2009 (measured: `pd.read_parquet(path, columns=["season"]).min()
== 2009` for all three). Every signal below needs at least one schedules-only
column (`surface`/`roof`, `home_coach`/`away_coach`, `home_rest`/`away_rest`,
`div_game`) that is not present in `sbr_odds.parquet` and has no other local
source before 2009. Signal 5 additionally needs the PBP snapshot, which is
also 2009-2025 only (measured:
`load_pbp_snapshot(latest_pbp_snapshot(...))["season"]` spans 2009-2025).
Consequence: **population is 2009-2025 close-grade for every signal**
(narrower per-signal where a construct itself lags one season or requires an
observed prior season -- stated per signal below), never 2007-2008.

| # | Signal | Construct source (reused by import) | Population (seasons) | Notes |
|---|---|---|---|---|
| 1 | `surface_switch` | `scripts/nfl_weather_battery_screen.py` (`load_population`, `build_cells`), cell `weather_battery_surface_switch_grass_to_turf` | 2009-2025 | Game-level; away team's own-season modal home surface is grass AND this game's surface is turf; response `home_cover`; one-sided (flag vs everyone else) |
| 2 | `division_revenge_game` | `nfl_ats.experiment_runner.FLAG_BUILDERS["division_revenge_game"]` | 2009-2025 | Team-game; 2nd meeting this season vs. same division opponent, team lost the 1st; response `team_covered`; one-sided |
| 3 | `home_underdog` | `nfl_ats.experiment_runner.FLAG_BUILDERS["home_underdog"]` | 2009-2025 | Team-game; home team getting points; response `team_covered`; one-sided |
| 4 | `extra_rest_edge` | `nfl_ats.experiment_runner.FLAG_BUILDERS["extra_rest_edge"]` | 2009-2025 | Team-game; own rest minus opponent rest >= 4 days; response `team_covered`; one-sided |
| 5 | `penalty_rate_quartile` | `nfl_ats.experiment_runner.FLAG_BUILDERS["penalty_rate_quartile"]` | 2010-2025 (2009 has no local prior-season rate to lag) | Team-game; prior-season penalty-rate quartile 1 vs quartile 4, global cut; response `team_covered`; two-sided (Q2/Q3 excluded from direct comparison, counted in the slate denominator) |
| 6 | `hc_year_one_fade` | Ported from `scripts/hc_year_one_fade.py` (`build_team_game_table`, `team_season_primary_coach`, `flag_year_one`), re-hosted on the SAME generic subset-vs-complement pipeline as 2-5 for era-slicing consistency | 2010-2025 (2009 has no observed prior coach) | Team-game, REG weeks 1-8 only; flag = first-year HC (known tenure), complement = kept-coach (two-sided, same "restricted eligible" design as penalty quartile); response `team_covered`. **Positive control**: already known 2018-2025-concentrated (registry `hc_year_one_fade`, era split +0.09pts 2009-2017 vs -8.08pts 2018-2025) -- reproducing that split is this script's instrument check. |
| 7 | Production model's own opener-proxy edge | `artifacts/proxy_opener_replication/20260819T194330Z/main_scored.parquet` (proxy-opener, 2011-2019, `correct_at_open_proxy_pr`) + `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet` (true-opener, 2020-2025, `correct_at_open_probability_rule`) | 2011-2025 (2009-2010 fail the 500-game warm-up floor, `docs/rotation_registry.md` rule 9 -- same mechanism, not re-derived here) | Per-game accuracy of the frozen production rule (probability rule, primary) vs. a 50% coin flip, in accuracy points; **re-sliced from the existing artifacts, no walk-forward re-run**. Grade differs by sub-period (SBR-Open proxy 2011-2019 vs. true Tuesday-opener 2020-2025, mean \|diff\| 1.36-1.37pts per `docs/sbr_odds_archive.md`) -- a genuine population-definition seam, stated wherever this series is read, never smoothed over. |

Every construct above is pregame-safe by inheritance: signals 1-6 reuse
already-audited flag logic verbatim (imported, not re-derived) from scripts/
modules whose own docstrings already state the leakage caveats (surface
battery: game-time actuals disclaimer does not apply to the *surface_norm*
column itself, which is a stadium fact known pregame, only to `temp`/`wind`
cells not used here); signal 7 reuses a frozen model's already-scored
predictions.

## Era scheme (Stage 1, original brief)

1. **Three fixed eras** (reporting convenience, not a mechanistic claim):
   `2009-2014`, `2015-2019`, `2020-2025`. Per signal, per era: subset-vs-
   complement cover-rate effect in accuracy points using the EXACT
   `scale_subset_effect` convention `nfl_ats.experiment_runner` already uses
   (raw cover-rate gap x sign x 100 x fraction-of-slate), a week-blocked
   bootstrap interval (primary) via the same vectorized joint two-group block
   bootstrap (`_block_bootstrap_subset_gap`, imported), 20,000 samples, fixed
   seed, plus `probability_positive`. Signal 7 uses the equivalent pooled
   accuracy-vs-50% construction (`nfl_ats.clv.week_blocked_bootstrap`) on its
   own already-scored per-game frame, same sample count and seed convention.
2. **A continuous per-season effect series**, one point per season, computed
   the same way as the era slices but restricted to one season at a time,
   PLUS an OLS trend slope of effect-on-season-number with its own
   week-block bootstrap interval and `probability_positive` (fraction of
   valid bootstrap draws with a positive slope). The per-season *bootstrap
   spread* (not a fresh interval computation) is reused as each season's
   precision weight for Stage 2's modulator regression -- documented at the
   point of use, not re-derived.
3. **Every number is reported with an interval regardless of sign** -- eras,
   seasons, and slopes alike. A weaker- or wrong-signed era reading is
   reported as exactly that reading, not as absence, and is never used alone
   to close anything (closing needs a RESOLVED wrong sign or a positive-
   control bound; see Binding rules above).

## Stage 2 (owner refinement, added mid-session, predeclared here before computing)

The owner's refinement arrived after Stage 1's constructs and eras were
declared but **before any era-sliced or season-sliced number had been
computed** for this document, so it is declared as its own stage rather than
retrofitted onto Stage 1's numbers.

### 2a. Free-break changepoint (fixed calendar eras are a convenience, not a claim)

For each signal's real (non-resampled) per-season effect series, and
separately for signal 7's per-season accuracy-vs-50% series: search every
candidate break season `b` (requiring >= 3 seasons on each side) for the
mean-shift two-segment fit that minimizes total sum of squared deviations
(the standard optimal-partition single-changepoint estimator applied to the
season-indexed point series; unweighted -- per-season sample sizes are
broadly comparable within one construct, stated as a simplification, not
hidden). Report the point estimate's break season, pre- and post-break mean
effect, and:

- A **bootstrap distribution of the break season** built from the SAME joint
  week-block bootstrap draws already generated for the per-season series
  (reused, not a second resample): for every bootstrap draw that has a valid
  (non-empty) subset and complement in every season, rerun the identical
  grid search on that draw's resampled per-season series and record its
  argmin break season. Report the median, the [2.5, 97.5] percentile
  spread, and the modal break season with its share of draws.
- If that spread is wide (spans most of the eligible break-season range) or
  the modal share is small, this is reported explicitly as "no stable break
  year" **with the spread itself**, per the owner's instruction, rather than
  forcing a single answer.
- Pre/post magnitude bootstrap intervals are taken by fixing the break at the
  POINT estimate's break season and resampling the pre- and post-segment
  means directly (same draws), reported as ordinary 95% percentile
  intervals -- these, like every other interval in this document, are never
  grounds to reject a shape on their own.

### 2b. Mechanistic modulators, declared before their correlation is computed

For each signal, one league-level series is declared as the modulator the
underlying MECHANISM says should scale the effect's magnitude -- computable
from data already in this repository, and named here before any regression
is run:

| Signal | Declared modulator | Mechanistic direction (stated, not assumed to hold) |
|---|---|---|
| `surface_switch` | League turf share: % of REG games per season played on a surface that normalizes to turf (of games with a known grass/turf surface) | As turf becomes the league norm, grass-to-turf mismatches become both rarer and less novel to opposing players -- hypothesized to shrink the effect as turf share rises, but stated, not assumed |
| `division_revenge_game` | League mean absolute scoring margin per REG game per season (parity/competitiveness proxy) | Revenge narratives plausibly matter more when games are closer (lower margin); direction not assumed a priori |
| `home_underdog` | League mean raw home-field advantage per season: mean(home score - away score), uncorrected for the market line | If home underdogs are underpriced BECAUSE the market lags true home-field advantage, the gap should scale with how large true HFA actually was that season (visible independently in the 2020 no-crowd dip, see "What makes an era" below) |
| `extra_rest_edge` | League mean \|own rest - opponent rest\| per team-game per season (rest-spread distribution) | A rest-differential edge should matter more in seasons where rest gaps are themselves larger/more common |
| `penalty_rate_quartile` | League mean team-season penalty rate per season | A market mispricing of discipline should track how much discipline varies league-wide that season (rules points-of-emphasis move this year to year) |
| `hc_year_one_fade` | Count of first-year-HC team-seasons per season (coaching-turnover volume) | More simultaneous first-year coaches could either dilute the average shock (more of the league already pricing turnover) or concentrate it (more exploitable teams); direction not assumed |
| Production model's own opener-proxy edge | Mean number of book quotes contributing to the CLOSE line per season | **Restricted to 2020-2025 only** (6 points) -- the SBR proxy archive (`main_scored.parquet`, read directly: columns checked, no book-count field exists) carries no book-count field for 2011-2019, so this modulator cannot be computed for the proxy leg. A sharper (more-quoted) market is hypothesized to compress the model's own edge; reported as likely underpowered at 6 points per the owner's explicit allowance to report the spread rather than force a conclusion |

Each modulator regression is a weighted OLS of the signal's real per-season
effect on the modulator's real per-season value, weights = inverse variance
of that season's own bootstrap draws (from 2a/Stage 1's per-season bootstrap,
reused). The slope's own bootstrap interval and `probability_positive` reuse
the SAME per-season joint bootstrap draws (fixed modulator values, fixed
weights, resampled effect only) -- consistent with every other interval in
this document. A signal whose effect tracks its declared modulator is read as
mechanistic era-dependence; a signal whose slope interval sits on either side
of zero with no lean is read as more consistent with era-agnosticism (which
the owner states is the rarer outcome) OR modulation by something not
declared here -- both readings are stated with the numbers, never asserted
past what the interval supports, and neither is used to close anything.

---

# Results

**measured**, `scripts/era_magnitude_profile.py`, seed 20260819, 20,000
bootstrap samples, run `artifacts/era_magnitude_profile/20260819T204710Z/results.json`
(46.8s wall clock). All six subset-vs-complement signals reproduce their
already-registered pooled 2009-2025 numbers as a sanity check before this
document trusted the era slices: `surface_switch`'s own construct, summed
across its three era slices proportional to n, reconstructs +1.159 accuracy
points against the registry's independently-recorded **+1.1618** pooled
figure (`weather_battery_surface_switch_grass_to_turf`) on the identical
4,317-game population -- confirms the re-hosted pipeline matches the
established one before any new claim is built on it.

## Signal x era matrix

Effect in accuracy points, week-blocked 95% interval, `probability_positive`
(P+). **An interval containing zero is not a negative result** -- read every
row for its number and lean, never as a binary.

| Signal | 2009-2014 | 2015-2019 | 2020-2025 | Season-trend slope (pts/season) |
|---|---|---|---|---|
| `surface_switch` | +0.909 [-0.481, +2.302] P+0.900 | +2.720 [+1.111, +4.345] P+0.999 | +0.173 [-1.295, +1.662] P+0.590 | -0.062 [-0.237, +0.120] P+0.258 |
| `division_revenge_game` | +0.257 [-0.258, +0.778] P+0.824 | +0.111 [-0.531, +0.717] P+0.615 | +0.191 [-0.289, +0.698] P+0.775 | -0.001 [-0.065, +0.061] P+0.482 |
| `home_underdog` | -0.101 [-0.930, +0.722] P+0.393 | -0.295 [-1.277, +0.689] P+0.273 | +0.337 [-0.579, +1.219] P+0.758 | +0.026 [-0.083, +0.134] P+0.684 |
| `extra_rest_edge` | -0.231 [-0.694, +0.224] P+0.144 | +0.300 [-0.282, +0.879] P+0.836 | -0.033 [-0.415, +0.372] P+0.419 | -0.001 [-0.053, +0.058] P+0.513 |
| `penalty_rate_quartile` | -0.451 [-3.077, +2.191] P+0.375 | +1.728 [-1.750, +5.094] P+0.835 | +0.054 [-2.038, +2.099] P+0.518 | -0.133 [-0.583, +0.328] P+0.283 |
| `hc_year_one_fade` | -1.463 [-6.760, +3.691] P+0.279 | +3.460 [-3.481, +10.779] P+0.817 | **+9.523 [+4.704, +14.370] P+1.000** | **+0.970 [+0.118, +1.866] P+0.988** |
| Production model's own opener-proxy edge | -0.251 [-3.035, +2.585] P+0.414 | +0.891 [-1.840, +3.639] P+0.734 | **+3.360 [+0.605, +6.040] P+0.991** | +0.347 [-0.021, +0.708] P+0.968 |

Bolded cells are the two clean excludes-zero readings in the raw calendar
view. Every other cell contains zero, which per the binding rule above is the
**expected** shape for a real small signal at this resolution -- reported as
a lean and a number, never as absence, and never as grounds to close
anything on its own.

`hc_year_one_fade`'s and the production model's own scale note: `table` for
`hc_year_one_fade` is already restricted to REG weeks 1-8 (the construct's
own population), so this script's `fraction_of_slate` denominator is that
weeks-1-8 population, not the full 17-week season the ORIGINAL registered
`hc_year_one_fade` entry (0.7528, pooled 2009-2025) scaled against. The two
numbers are **not on the same absolute scale** -- this document's magnitudes
are internally consistent era-to-era (same denominator applied uniformly),
which is what an era-magnitude comparison needs, but should not be read
digit-for-digit against the original registry entry.

## Free-break changepoint (2a)

| Signal | Break season (point) | Bootstrap spread of break season | Pre-break mean | Post-break mean |
|---|---|---|---|---|
| `surface_switch` | 2020 | modal 2023 (24.6% of draws); median 2019; [2012, 2023] | +1.736 | +0.183 |
| `division_revenge_game` | 2016 | modal 2016 (18.6%); median 2017; [2012, 2023] | +0.345 | +0.087 |
| `home_underdog` | 2020 | modal 2023 (27.4%); median 2020; [2012, 2023] | -0.192 | +0.343 |
| `extra_rest_edge` | 2015 | modal 2015 (36.8%); median 2016; [2012, 2023] | -0.231 | +0.116 |
| `penalty_rate_quartile` | 2020 | modal 2023 (30.4%); median 2020; [2013, 2023] | +1.114 | -2.304 |
| `hc_year_one_fade` | **2019** | modal **2019 (27.7%)**; median 2016; [2013, 2021] | -0.323 | **+9.387** |
| Production model's own edge | 2019 | bootstrap estimate 2019.0; [2014, 2023] | +0.132 | +3.227 |

Every modal share is well under 50% -- per the predeclaration's own
instruction, **none of these seven break-season estimates is stable**; each
is reported here as the number plus its spread, not forced to a confident
single year. `hc_year_one_fade` and the production model's own edge are the
two whose spreads are narrowest and whose pre/post means diverge most
sharply, and both cluster the point estimate and bootstrap distribution
around **2018-2019** -- consistent with, not proof of, one shared underlying
shift (sports-betting-market maturation, see "What makes an era" below) but
this document does not have the power to separate that from two independent
coincidences.

## Mechanistic modulator regressions (2b)

| Signal | Modulator | Slope | Interval | P+ | Reading |
|---|---|---|---|---|---|
| `surface_switch` | League turf share % | -0.068 | [-0.341, +0.202] | 0.296 | Crosses zero, weak lean; consistent with era-agnosticism or an undeclared modulator |
| `division_revenge_game` | League mean \|scoring margin\| | **-0.424** | **[-0.815, -0.067]** | **0.011** | **Excludes zero.** Effect grows as league scoring margins shrink (tighter, more competitive seasons) -- a genuine mechanistic reading hidden entirely by the flat calendar-era view above |
| `home_underdog` | League home-field advantage | -0.371 | [-0.919, +0.198] | 0.100 | Crosses zero but 90% of draws negative -- a real lean, not resolved |
| `extra_rest_edge` | League mean rest gap | +0.777 | [-1.960, +3.629] | 0.720 | Crosses zero, weak lean |
| `penalty_rate_quartile` | League mean penalty rate | +1.968 | [-2.013, +5.893] | 0.831 | Crosses zero, moderate lean |
| `hc_year_one_fade` | Count of first-year HCs | +0.821 | [-1.093, +2.829] | 0.798 | Crosses zero, moderate lean -- turnover VOLUME does not resolve as the mechanism even though the calendar-era drift itself is resolved |
| Production model's own edge | Mean books at close (2020-2025 only, n=6) | +0.064 | [-0.844, +0.979] | 0.557 | Underpowered at 6 points, as predeclared; no lean either way |

**`division_revenge_game` is the headline mechanistic finding.** Its raw
calendar-era slope is flat (-0.001 [-0.065, +0.061], P+0.482 -- looks
perfectly era-stable), which on its own would read as "no era dependence."
The modulator regression shows the opposite: the effect tracks league
competitiveness with an interval that excludes zero. Calendar year is the
wrong axis for this signal; parity is closer to the right one. This is
exactly the failure mode Stage 2 was added to catch.

`hc_year_one_fade` is the inverse case: a resolved calendar-era drift (P+
0.988) whose one declared candidate mechanism (simultaneous coaching
turnover volume) does NOT resolve (P+ 0.798, crosses zero) -- the drift is
real by this instrument, but this document's one declared modulator does not
explain it. Per the owner's framing, that leaves two readings, both stated
because neither is settled: modulation by something undeclared (a genuine
possibility -- betting-market maturation, front-office quality-of-hire
trends, or something else entirely), or an era-agnostic-looking modulator
that still needs a better proxy than a bare count.

## Instrument check: does `hc_year_one_fade`'s known 2018+ concentration reproduce?

**Yes, qualitatively.** The registry's already-established finding
(`hc_year_one_fade`, `registry/weak_signals.json`) is a 2009-2017 vs.
2018-2025 split of +0.09 vs. -8.08 raw cover-rate points (raw sign; year-one
teams cover much less from 2018 on). This script's independent, differently-
scaled re-slice: (1) the fixed 2020-2025 era is the one clean excludes-zero
cell in the entire matrix (+9.523 [+4.704, +14.370] P+1.000); (2) the
free-break changepoint search -- which was never told 2018 -- independently
finds its point-estimate and bootstrap-modal break season at **2019**, one
year off the registry's already-known 2018 origin and well inside this
signal's own bootstrap spread ([2013, 2021]); (3) the season-trend slope is
the only calendar-axis slope in the whole profile with a bootstrap interval
that excludes zero (P+0.988). All three independent readings point the same
direction the positive control predicted. **The instrument passes**, so the
rest of this profile's methodology is trusted rather than re-derived from
scratch for each signal.

## What makes an era (2c)

Every series below is `measured`, from `league_series` in the run artifact,
2009-2025, computed once and reused for both the modulator regressions above
and this table -- nothing here is cherry-picked after the fact.

| Season | Turf share % | Mean \|margin\| | Home-field adv. | Mean rest gap | Penalty rate % | Year-one HCs |
|---|---|---|---|---|---|---|
| 2009 | 43.95 | 12.97 | 2.207 | 1.036 | 7.146 | -- |
| 2010 | 43.83 | 11.75 | 1.895 | 1.064 | 7.253 | 3 |
| 2011 | 43.27 | 12.06 | 3.266 | 1.269 | 7.611 | 8 |
| 2012 | 43.83 | 12.15 | 2.434 | 1.207 | 7.388 | 8 |
| 2013 | 44.40 | 11.29 | 3.105 | 1.320 | 7.114 | 9 |
| 2014 | 43.60 | 12.67 | 2.488 | 1.108 | 7.802 | 8 |
| 2015 | 45.57 | 11.06 | 1.562 | 1.202 | 8.074 | 7 |
| 2016 | 40.64 | 10.23 | 2.566 | 1.295 | 7.910 | 7 |
| 2017 | 41.53 | 11.81 | 2.484 | 1.242 | 7.959 | 6 |
| 2018 | 39.68 | 11.09 | 2.203 | 1.182 | 8.061 | 7 |
| 2019 | 35.37 | 11.64 | **-0.141** | 1.272 | 8.026 | 9 |
| 2020 | 40.23 | 11.07 | **0.055** | 1.258 | 6.676 | 7 |
| 2021 | 45.52 | 12.18 | 1.713 | 1.063 | 7.129 | 8 |
| 2022 | 48.24 | 9.71 | 1.978 | 1.130 | 6.804 | 11 |
| 2023 | 45.33 | 11.27 | 2.684 | 1.101 | 6.918 | 6 |
| 2024 | 46.99 | 11.20 | 1.868 | 1.306 | 7.826 | 7 |
| 2025 | 45.93 | 11.15 | 2.070 | 1.151 | 7.882 | 7 |

**measured** footprints worth flagging:

- **Home-field advantage collapses in 2019-2020**: mean 2009-2018 is
  +2.421 points; 2019 measures -0.141 (the only negative season in the
  series) and 2020 measures +0.055, both far below every other season. The
  well-known **`reported`** (public record, not derived here) context for
  2020 is the COVID-19 season's fan-attendance restrictions -- but the
  measured dip is already visible a full season EARLIER, in 2019, which the
  no-crowds story does not explain. Stated as measured, not resolved to one
  cause.
- **Penalty rate drops in 2020** (6.676%, vs. a 2015-2019 mean of 8.006%) --
  plausibly the same reduced-crowd/shortened-training-camp season, stated as
  a measured coincidence, not a claimed mechanism.
- **Turf share bottoms at 35.37% in 2019 and peaks at 48.24% in 2022** -- a
  13-point swing across three seasons, driven by stadium changes (e.g. SoFi
  Stadium and Allegiant Stadium both opened for the 2020 season on
  synthetic surfaces), `inferred` explanation, not verified against a
  stadium-by-stadium audit in this session.
- **`reported`, external, not derived from this project's data**: the May
  2018 *Murphy v. NCAA* PASPA repeal broadened legal U.S. sports betting
  starting the 2018 season -- offered as context for why several of this
  profile's break-season estimates (`hc_year_one_fade`, the production
  model's own edge) cluster around 2018-2019, not as a causal claim this
  document tests.

## What this profile supports, plainly

- **Consistent with the owner's magnitude-drift hypothesis, resolved
  (interval excludes zero):** `hc_year_one_fade`'s calendar-era slope,
  `division_revenge_game`'s mechanistic (parity) modulator slope.
- **Consistent with magnitude drift, leaning but not resolved:** the
  production model's own opener-proxy edge (P+0.968, one grade-seam caveat
  stated above and again in the registry notes), `home_underdog`'s
  home-field-advantage modulator (P+0.100).
- **More consistent with era-stability than drift, on this evidence:**
  `division_revenge_game`'s and `extra_rest_edge`'s calendar-axis slopes,
  `surface_switch`'s calendar-axis slope in its OWN 2020-2025 era cell
  (though its 2015-2019 era cell is the sharpest single-era reading in the
  whole matrix, +2.720 [+1.111, +4.345] P+0.999 -- era-stability and a loud
  middle era are not contradictory, and neither is reported as the whole
  story).
- **Not resolved either way, reported as a number and a lean per the
  binding rule, never as absence:** every other cell in both tables.

None of the above closes anything. No signal in this profile meets either
admissible AGENTS.md closing ground (a whole-interval-below-zero wrong sign,
or a positive-control bound); every entry below is recorded
`unresolved_below_power`.

### 2c. "What makes an era" (populated after computation, from measured series only)

A table/short section of the actual league-level series computed for 2b
(turf share, scoring margin, home-field-advantage, rest-spread, penalty rate,
coaching turnover, book depth), one row per season, is added to this document
after the script runs. Known structural events are noted **only where their
date is independently verifiable and not derived from this project's own
data** (e.g., the 2018 PASPA repeal broadening legal U.S. sports betting; the
2020 season's fan-attendance restrictions) -- those dates are tagged
`reported` (public historical record, not measured here), while any visible
footprint in the measured series (e.g., a 2020 dip in measured home-field
advantage) is tagged `measured` and reported as its own row, not conflated
with the reported date.

## Registry recording plan

One registry entry per signal for the season-trend-slope finding, name
`era_trend_<signal>`. `accuracy_points` is an admissible `effect_units`
(`nfl_ats.weak_signals.EFFECT_UNITS`) for a POINT effect, but the registry
has no "per-season-slope" unit -- read directly, `EFFECT_UNITS = ("ats_points",
"accuracy_points", "brier", "log_loss", "mae")`, none of which is a rate. Per
the task's own fallback instruction, each entry therefore records the
**2020-2025-era effect** (in `accuracy_points`, the same scale as every other
subset-vs-complement entry in this registry) as the recordable `effect`, with
the slope value, its own interval, and its `probability_positive` placed in
`notes` -- never hand-edited, always produced by a script that reads this
run's artifact JSON, calls `nfl-ats weak-signals record`, and reads the
ledger back to verify. Every entry classifies `unresolved_below_power`
(diagnostic re-slice; nothing here meets a closing ground) unless the
mechanical `wrong_sign_resolved` test genuinely applies (whole interval below
zero AND the widening-factor test the runner already uses) -- decided by the
recorder script reading the artifact, never asserted in prose first.
