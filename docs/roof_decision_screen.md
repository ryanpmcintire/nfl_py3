# Retractable-roof open/closed decision screen

Source: `docs/archive/data_source_scout_v4.md` rank #6 ("Retractable-roof
open/closed decisions"), build-next rank 5 in that doc's "Top 5 for
immediate build" list. That doc's own effort estimate was "S for the
backtest" since the `roof` field already exists in the project's ingested
nflverse schedule data; this doc is that backtest.

Every claim below is tagged **measured** (run this session, command/path
given), **read** (a file opened this session, path/line given), **reported**
(unverified, a doc or a prior session's claim), or **inferred** (reasoning,
not evidence), per the binding `AGENTS.md` labeling rule.

## Binding closing-grounds taxonomy (restated verbatim, governs every verdict below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry hard-rejects inadmissible closures; if
a record command errors, the verdict is wrong, not the validator. Never use
95%/0.90 as a decision bar -- decide on expected value. Within-week game
correlation is ZERO by owner mandate.

Every cell below is classified `unresolved_below_power`. None is closed.

## Predeclaration

The full predeclaration was frozen, before any effect was computed, at
`<scratchpad>/roof_decision_screen/predeclaration.md` (this is a mined,
exploratory battery -- said explicitly, uncorrected multiplicity, per this
project's standing bias-battery precedent). It is reproduced in substance
below; population/data-availability facts (which teams are retractable,
which local table carries pregame forecast data) were inventoried before
locking cell definitions, since a data-availability check is not an
"effect."

## Population

**Retractable-roof teams (measured from data, not assumed):**
`home_team in {ARI, ATL, DAL, HOU, IND}` AND `roof in {open, closed}`, REG
season only. **Measured**, grouping `data/raw/20260817T235649Z/schedules.parquet`
by `home_team`/`stadium`: these five are the only home teams that ever show
BOTH `open` and `closed` roof tags in this data. ATL's pre-2017 Georgia Dome
games are tagged `dome` (fixed), not `open`/`closed`, so filtering to
`roof in {open,closed}` automatically restricts ATL to its Mercedes-Benz
Stadium era (2017+) with no extra season filter needed -- **measured**: ATL
shows `closed=55, dome=63, open=18`, and every one of the 63 `dome` rows is
`stadium=='Georgia Dome'` (2009-2016), every `open`/`closed` row is
`stadium=='Mercedes-Benz Stadium'` (2017+). LV (Allegiant Stadium) and
LA/LAC (SoFi Stadium) are fixed-roof domes in nflverse's own schema (always
`dome`, never `open`/`closed`) despite having partially-operable real-world
elements -- **measured**, treated here as controls, not retractable, since
this project's own data source classifies them that way. Three 2020 SF
"home" games relocated to State Farm Stadium (Santa Clara County COVID
restrictions) are tagged `closed` under `stadium=='State Farm Stadium'` but
`home_team=='SF'` -- **measured**; filtering by `home_team in
{ARI,ATL,DAL,HOU,IND}` automatically excludes them, which is the intended
behavior (those are a relocation artifact, not a genuine ARI home game).

**Measured** population: n=626 REG games, 2009-2025, all with a completed
result. Breakdown: ARI 126 closed / 11 open, ATL 55 closed / 18 open
(2017-2025 only), DAL 129 closed / 9 open, HOU 124 closed / 15 open, IND
108 closed / 31 open. Total closed=542, open=84.

**Pre-2020 completeness of the `roof` field** -- the scout doc (rank #6)
flagged this as an open question ("flagged 'NEW Feb 2020' in [nfldata's]
own changelog... needs direct inspection"). **Resolved this session,
measured**: the field is populated for every one of the 626 retractable
games back to 2009 (ARI's earliest closed/open tags are 2009), so the
"NEW Feb 2020" changelog note refers to when nfldata started EXPOSING the
column, not when the underlying value became available -- the doubt raised
in the scout doc does not hold up for this local snapshot.

**Full-league complement population:** REG season 2009-2025 team-game long
table (one row per team per side per game, 8,634 rows after push-dropping),
built the same way as `scripts/nfl_bias_battery_screen.py`'s long table.
Every cell's recorded `effect`/`interval` is scaled by
`fraction_of_slate = n_flag / n_total` against this FULL league population
-- the same convention every recorded `bias_battery_*`/`referee_battery_*`/
`attention_battery_*` signal already uses, so the recorded
`accuracy_points` figure stays commensurable with the rest of the registry
per AGENTS.md's binding commensurability rule (same units, same scale, same
population style -- it answers "how much would using this rule move overall
forced-pick pool accuracy," not just the within-subset gap). Raw (unscaled)
subset/complement cover rates are reported alongside for direct
interpretability.

**Weather-data availability finding (relevant to Cell 4):** **measured**,
`data/raw/20260817T235649Z/schedules.parquet`'s own `temp`/`wind` fields
are essentially 100% `NaN` for every retractable-venue game regardless of
roof state (0.18% notna for closed, 0.00% for open) -- nflverse only
populates them for `outdoors`-tagged games. The only local source with an
actual pregame FORECAST at the stadium's own coordinates, independent of
roof state, is `data/raw/forecast_archive/full_2020_2025/forecasts.parquet`
(`forecast_temp_f`/`forecast_wind_mph`, NDFD/GFS-MOS derived) --
**measured**, 100% populated for both open and closed retractable games in
that table. That table only covers 2020-2025, so Cell 4 is restricted to
that window; every other cell uses the full 2009-2025 `roof` field.

## Method

Week-blocked joint block bootstrap (block = `season*100+week`), 20,000
resamples, seed 20260820 -- the exact `block_bootstrap_two_group` method
from `scripts/nfl_bias_battery_screen.py`, duplicated (not imported) into
`scripts/roof_decision_screen.py` to keep the battery self-contained.
`probability_positive` (P+) = fraction of bootstrap draws favoring the
cell's own predeclared sign direction (for Cell 4, which has no
predeclared direction, the raw gap is reported as-is with no sign flip).

Two grades are reported for the two ATS cells most relevant to a
kickoff-refresh decision (Cells 1 and 2): the CLOSE grade (full 2009-2025
history, `game_features.parquet`'s own `home_cover`) and the OPENER grade
(AGENTS.md's binding "grade the decision at the opener" instruction),
built by calling `nfl_ats.experiment_runner._opener_graded_features`
directly -- **imported, not modified**, per this task's constraint against
touching `src/nfl_ats`. That function restricts to the paired 2020-2025
Tuesday-open/close archive (1,537 REG games) and overwrites
`spread_line`/`home_cover`/`ats_margin` with the OPENER line; roof/stadium
survive as simple passthrough columns.

Script: `scripts/roof_decision_screen.py`. Full JSON output:
`artifacts/roof_decision_screen/20260820T111226Z/results.json`.

## Cells

All effects are `accuracy_points` (full-slate-scaled), matching the
registry's existing convention. `raw_gap` is the unscaled subset-minus-
complement gap in percentage points (sign already applied to the
predeclared direction where one exists).

### Cell 1 -- `roof_battery_home_cover_open_vs_closed` (CLOSE grade)

Home team's own ATS cover rate at a retractable venue: subset = home
team's row, `roof=='open'`; complement = everyone else in the league.
Mechanism: an open roof strips these five franchises' climate-controlled
home identity, hypothesized to hurt HOME cover.

- **n**: n_flag=81 (of 84 raw open games, 3 pushes dropped), n_total=8,634,
  n_week_blocks=294.
- **Cover rates**: subset (open) 44.44% vs complement 50.05%.
- **Raw gap**: +5.608 pts (in the predeclared direction).
- **Effect (full-slate-scaled)**: **+0.0526 accuracy_points**.
- **95% week-blocked interval**: **[-0.0514, +0.1527]**.
- **P+**: **0.8293**.
- **Era split** (boundary = ATL's Mercedes-Benz Stadium entering the
  retractable set, not arbitrary): 2009-2016 n_flag=36, effect +0.0506pts,
  P+=0.7225; 2017-2025 n_flag=45, effect +0.0543pts, P+=0.7490. Subset
  cover rate is exactly 4/9=0.4444 in BOTH eras independently (16/36 and
  20/45, **measured**, verified per-season to rule out a bug) -- a genuine
  small-integer coincidence, not evidence of anything beyond itself.
  Direction and magnitude are consistent across both eras, not a
  weaker-era-masks-absence case (per the owner's era rule).
- **Operational timing**: BACKTEST. Roof status is announced ~90 minutes
  before kickoff (read, footballzebras.com's policy explainer, already
  cited in `docs/archive/data_source_scout_v4.md` rank #6) -- after this project's
  Tuesday line freeze, but before kickoff. Per project memory (picks stay
  editable to kickoff, only lines freeze Tuesday), this is a legitimate
  KICKOFF-ADJACENT refresh-pass input, **not** usable at the Tuesday lock.
  No live T-90 scrape pipeline exists in this repo.

### Cell 1 (OPENER grade) -- `roof_battery_home_cover_open_vs_closed_opener`

Same construct, graded against the Tuesday-opener line via
`_opener_graded_features`, 2020-2025 paired-archive intersection.

- **n**: n_flag=29, n_total=3,006, n_week_blocks=107.
- **Cover rates**: subset (open) **55.17%** vs complement 49.95%.
- **Raw gap**: -5.223 pts.
- **Effect**: **-0.0504 accuracy_points**.
- **95% interval**: **[-0.2513, +0.1479]**.
- **P+**: **0.2736**.

**The direction flips relative to the close-grade full-period cell.**
Stated plainly, not smoothed over: at CLOSE grade over 17 seasons the home
team covers worse when the roof is open (P+=0.83 favoring that
direction); at OPENER grade over the 2020-2025 window the home team
covers BETTER when open (P+=0.27 favoring the close-grade direction, i.e.
leaning the other way). n_flag=29 in a 6-season window is thin, and the
interval crosses zero at both grades, so neither number resolves the
other -- this is a genuine directional disagreement between two honest
gradings of the same mechanism, worth carrying forward as-is, not a case
where one grading is "right" and should override the other.

### Cell 2 -- `roof_battery_visiting_dome_open_vs_closed` (CLOSE grade)

Visiting fixed-dome team's own ATS cover rate at a retractable venue with
the roof open. Fixed-dome classification (**measured**, data-driven): a
`(team, season)` is "fixed dome" iff every `roof` value across that team's
OWN home games that season is exactly `dome` -- 81 such pairs found; this
correctly excludes ATL's own 2017+ seasons from counting as a dome visitor
while still counting its pre-2017 Georgia Dome seasons, and excludes MIN's
2014-2015 outdoor TCF Bank Stadium stint. Subset = away team's row,
`roof=='open'`, opponent's stadium in RETRACT, away team is fixed-dome that
season; complement = everyone else. Mechanism: a dome-only team is
hypothesized more disrupted by real outdoor conditions than an
already-acclimated outdoor team, predicting the VISITOR covers LESS when
open.

- **n**: n_flag=**10** (the thinnest cell in this battery), n_total=8,634,
  n_week_blocks=294.
- **Cover rates**: subset (visiting dome team, open) **80.0%** vs
  complement 49.97%.
- **Raw gap**: -30.03 pts (opposing the predeclared direction).
- **Effect**: **-0.0348 accuracy_points**.
- **95% interval**: **[-0.0580, 0.0000]** -- touches the zero boundary
  from below rather than cleanly excluding it.
- **P+**: **0.0147** (in the predeclared direction; the raw sample runs the
  opposite way).
- **Era split**: 2009-2016 n_flag=5, subset_cover=0.60, P+=0.2421;
  2017-2025 n_flag=5, subset_cover=**1.00**, P+=0.0000. Both halves are
  5-game cells.
- **Why this is NOT recorded as `wrong_sign_resolved`**, despite the
  interval technically not crossing zero: the taxonomy's "resolved" bar
  means genuine power, not a 10-game cell whose bootstrap is this close to
  degenerate (1 of 20,000 draws dropped for an empty arm at full period;
  125/20,000 and 104/20,000 dropped in the two era-split halves). Reported
  plainly as opposing the hypothesized mechanism at this sample size, not
  spun as a null and not closed.
- **Operational timing**: same as Cell 1.

### Cell 2 (OPENER grade) -- `roof_battery_visiting_dome_open_vs_closed_opener`

- **n**: n_flag=**3**, n_total=3,006, n_week_blocks=107. All 3 covered.
- **Cover rates**: subset 100% vs complement 49.95%.
- **Effect**: -0.0500 accuracy_points.
- **95% interval**: [-0.0500, -0.0499] -- superficially all-negative, but
  this is a **DEGENERATE** bootstrap (926 of 20,000 draws dropped for an
  empty arm -- the same "DEGENERATE: below the measured block-count floor"
  situation this registry already labels explicitly for other thin
  opener-grade cells, e.g. `bias_battery_backup_qb_start_opener`). Not
  invoking `wrong_sign_resolved`: 3 games cannot resolve a mechanism
  regardless of how narrow the resampled interval looks; the narrowness is
  an artifact of resampling a 3-point sample, not statistical power.
  Recorded for completeness per AGENTS.md ("never discard a signal for
  being underpowered"), not as an informative estimate on its own.

### Cell 3 -- `roof_battery_total_covered_open_vs_closed` (CLOSE grade, TOTALS market)

**This cell is a TOTALS (over/under) market signal, NOT ATS** -- flagged
explicitly here and in the registry notes. It uses the same units
(`accuracy_points`) and the same fraction-of-slate scaling convention as
every other entry, but a **different market/population**; per AGENTS.md's
binding commensurability rule ("same units, same scale, same population"),
**do not pool this figure blindly with the ATS `accuracy_points` entries
elsewhere in the registry** without separately accounting for the market
difference.

Game-level (not team-side) `total_covered = (home_score+away_score) >
total_line`; subset = retractable-venue games with `roof=='open'`;
complement = every other REG game 2009-2025. Mechanism: an open roof
exposes real wind/cold a total line (typically set assuming a controlled
environment for these venues) may not fully price, predicting UNDER covers
more (over-rate LOWER) when open.

- **n**: n_flag=84 (0 of the population's 4 total pushes fell in the open
  subset), n_total=4,385 games, n_week_blocks=294.
- **Over-rates**: subset (open) 47.62% vs complement 49.45%.
- **Raw gap**: +1.835 pts (in the predeclared direction).
- **Effect**: **+0.0351 accuracy_points**.
- **95% interval**: **[-0.1737, +0.2423]**.
- **P+**: **0.6363**.
- **Era split**: +0.0580pts (P+=0.6531, n_flag=38) 2009-2016 vs
  +0.0150pts (P+=0.5434, n_flag=46) 2017-2025 -- same direction, weaker
  magnitude in the later era; reported as a magnitude difference (owner's
  era rule), not treated as absence.
- **Operational timing**: same as Cell 1.

### Cell 4 -- `roof_battery_closed_benign_forecast_vs_open` (CLOSE grade, 2020-2025 only)

**Exploratory, no predeclared direction** (sign fixed at +1, raw gap
reported as-is -- stated in the predeclaration before scoring, not decided
after seeing the number). 2020-2025 only, gated by `forecast_archive`
coverage. "Benign" threshold, fixed before scoring: `forecast_temp_f >= 50`
AND `forecast_wind_mph <= 15` (a round, untuned "mild day" cutoff). Subset
= home team's row, `roof=='closed'`, forecast benign by that threshold --
i.e. a "comfort"/revealed-preference closure, not a weather-necessitated
one; complement = everyone else in the league.

- **n**: n_flag=145, n_total=3,164 (2020-2025 team-game rows),
  n_week_blocks=107.
- **Cover rates**: subset (closed-when-benign) 44.83% vs complement 50.25%.
- **Raw gap**: -5.421 pts.
- **Effect**: **-0.2484 accuracy_points**.
- **95% interval**: **[-0.6084, +0.1223]**.
- **P+**: **0.0870** (probability the gap is positive under the fixed +1
  sign, i.e. most draws run negative -- reported as an exploratory lean
  since no direction was predeclared, not as a confirmed or refuted
  finding).
- **Side count, anecdotal only, NOT bootstrapped** (too rare to test): 9
  games in 2020-2025 where the roof was OPEN despite a non-benign
  forecast.
- **Operational timing (the nuanced case)**: the benign/non-benign
  FORECAST classification is itself Tuesday-safe (NDFD/GFS-MOS forecasts
  are available days ahead of the Tuesday lock). But the cell as a whole is
  still gated on the actual roof OUTCOME (closed vs open), which is only
  known ~T-90 minutes before kickoff -- so despite having a Tuesday-safe
  half, this cell stays kickoff-adjacent overall, playable via a
  kickoff-editable refresh pass, **not** usable at the Tuesday lock.

## Registry entries recorded

All 6 recorded via `nfl-ats weak-signals record`, then confirmed present
with `nfl-ats weak-signals status` (measured: all 6 appeared on the first
check immediately after recording, and again on a second check afterward
to rule out a parallel-writer race -- this session ran concurrently with
other agents also writing to the same registry file, so the total signal
count moved between checks (281, then 293), but all 6 `roof_battery_*`
names were present both times). Classification is `unresolved_below_power`
for every entry -- none crosses either binding closing ground.

| name | grade | seasons | effect (accuracy_points) | 95% interval | P+ | n_flag / n_total |
|---|---|---|---|---|---|---|
| `roof_battery_home_cover_open_vs_closed` | close | 2009-2025 | +0.0526 | [-0.0514, +0.1527] | 0.8293 | 81 / 8,634 |
| `roof_battery_home_cover_open_vs_closed_opener` | opener | 2020-2025 | -0.0504 | [-0.2513, +0.1479] | 0.2736 | 29 / 3,006 |
| `roof_battery_visiting_dome_open_vs_closed` | close | 2009-2025 | -0.0348 | [-0.0580, 0.0000] | 0.0147 | 10 / 8,634 |
| `roof_battery_visiting_dome_open_vs_closed_opener` | opener | 2020-2025 | -0.0500 | [-0.0500, -0.0499] (degenerate) | 0.0000 | 3 / 3,006 |
| `roof_battery_total_covered_open_vs_closed` (totals market) | close | 2009-2025 | +0.0351 | [-0.1737, +0.2423] | 0.6363 | 84 / 4,385 |
| `roof_battery_closed_benign_forecast_vs_open` (no predeclared direction) | close | 2020-2025 | -0.2484 | [-0.6084, +0.1223] | 0.0870 | 145 / 3,164 |

## Wiring recommendation

**No cell here clears a strength bar that would justify wiring a live
refresh-pass overlay today**, and per AGENTS.md a promotion bar is not the
same question as a play decision -- but the pool is forced picks either
way, so the honest framing is expected value, not a pass/fail gate. Two
cells lean the same direction the mechanism predicts with real P+ weight
(Cell 1 close-grade P+=0.83, Cell 3 P+=0.64) at essentially free
`accuracy_points` (both intervals include zero, both are small, neither is
a strong signal on its own). One cell (Cell 1 opener-grade) leans the
opposite way in the exact window (2020-2025) that matters most for a live
system, which is the strongest reason to NOT wire this yet rather than any
crossing-zero objection: the two honest gradings of the same cell disagree
on direction, and per AGENTS.md's own "grade at the opener" instruction the
opener-grade number is the one that should carry more weight for a
kickoff-time decision, not the close-grade one, and it currently points the
other way.

**If a future session wires this as a refresh-pass overlay anyway** (e.g.
after pooling with other kickoff-adjacent channels, where the sign
disagreement matters less than it does alone), the sketch is:

1. **Live source needed** (unbuilt): a T-90-to-kickoff scrape of roof
   status for the week's retractable-venue games. `docs/archive/data_source_scout_v4.md`
   rank #6 already flags this as the separate, larger follow-up (effort M)
   beyond this backtest -- no such live source exists in this repo today.
   Candidate feeds: NFL.com/team beat-reporter pregame-info pages, or a
   direct footballzebras.com-style tracker; neither was verified live this
   session (that scout doc's own live-source verification was scoped to
   the backtest field, not a prospective feed).
2. **Trigger condition**: for a game at a retractable venue (ARI/ATL/DAL/
   HOU/IND home game), once roof status is observed between T-90 and
   kickoff, apply a small adjustment to the home team's cover probability
   -- direction and magnitude TBD pending which grade's sign a future
   session decides to trust (this doc deliberately does not resolve that
   disagreement).
3. **Pool interaction**: per the project's picks-editable-to-kickoff
   policy, this would run as a genuine late refresh pass, not a
   replacement for the Tuesday-locked line -- consistent with how the
   project's other kickoff-adjacent channels (late injury/weather/
   observed-movement) are already scoped in project memory.
4. **Do not treat Cell 2 (visiting dome team) as wireable at all right
   now**: n=10 (close) / n=3 (opener) is too thin for even a small
   refresh-pass nudge to be worth the added surface area; it is recorded
   for future pooling, not for near-term wiring.

## Files

- `scripts/roof_decision_screen.py` -- the measure-only screening script
  (never writes to `registry/`, never edits a tracked doc).
- `artifacts/roof_decision_screen/20260820T111226Z/results.json` -- full
  JSON output, every cell's full-period and era-split numbers.
- `<scratchpad>/roof_decision_screen/predeclaration.md` -- the frozen
  predeclaration, written before any effect was computed.
- `registry/weak_signals.json` -- the 6 `roof_battery_*` entries.
