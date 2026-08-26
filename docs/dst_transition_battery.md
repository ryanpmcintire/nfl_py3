# Daylight-saving transition battery: predeclaration

Written 2026-08-26, **before any cover-rate sign in this battery has been
examined**. Nobody in this repository has ever asked whether the DST
*transition itself* (the clock-change shock, not the correctness detail in
timezone arithmetic already handled by `zoneinfo` per
`docs/travel_rest_battery.md`) moves ATS outcomes. This document freezes
population, dates, cells, thresholds, and predicted signs before scoring,
matching the precedent in `docs/fluview_battery.md` and
`docs/travel_rest_battery.md`.

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

## Mechanism (inferred, stated as such)

The US "fall back" transition lands in early November, inside the NFL
season, every single year in this repo's data span. Circadian research
treats a clock transition as a multi-day sleep-schedule disruption, not an
instantaneous correctness detail. If that disruption is real and
asymmetric between the two teams in a game -- and the market does not
price it -- it should show up in `home_cover`. This is a hypothesis, not a
measurement; nothing below assumes it is true.

## 1. Transition dates (measured this session, via `zoneinfo`, not hardcoded)

The rule "first Sunday of November / second Sunday of March" changed in
2007 (Energy Policy Act of 2005); this repo's data starts in 2009, so the
whole span postdates the change, but the dates below were **computed**,
not assumed. Method: scan `America/New_York`'s UTC offset (via stdlib
`zoneinfo`, evaluated at noon local on each candidate day) day-by-day
across Oct 1 - Dec 1 (fall) and Feb 1 - May 1 (spring) for each season
2009-2025, and record the first date the offset changes. This is the exact
same "evaluate at the game's own date via `zoneinfo`" mechanism
`docs/travel_rest_battery.md` already uses for `tz_delta_eastbound` -- this
battery is the first to test the transition itself rather than only use it
as plumbing.

| season | spring transition (EST->EDT) | fall transition (EDT->EST) |
|---|---|---|
| 2009 | 2009-03-08 | 2009-11-01 |
| 2010 | 2010-03-14 | 2010-11-07 |
| 2011 | 2011-03-13 | 2011-11-06 |
| 2012 | 2012-03-11 | 2012-11-04 |
| 2013 | 2013-03-10 | 2013-11-03 |
| 2014 | 2014-03-09 | 2014-11-02 |
| 2015 | 2015-03-08 | 2015-11-01 |
| 2016 | 2016-03-13 | 2016-11-06 |
| 2017 | 2017-03-12 | 2017-11-05 |
| 2018 | 2018-03-11 | 2018-11-04 |
| 2019 | 2019-03-10 | 2019-11-03 |
| 2020 | 2020-03-08 | 2020-11-01 |
| 2021 | 2021-03-14 | 2021-11-07 |
| 2022 | 2022-03-13 | 2022-11-06 |
| 2023 | 2023-03-12 | 2023-11-05 |
| 2024 | 2024-03-10 | 2024-11-03 |
| 2025 | 2025-03-09 | 2025-11-02 |

Every transition in both columns is measured to fall on a Sunday (checked
programmatically, `weekday() == 6`, in the scan itself -- not merely
asserted from memory). `America/Phoenix` (Arizona) was measured to have a
constant UTC offset (`dst()` returns `0:00:00` at every sampled date,
including mid-summer) across the whole window -- confirming Arizona is a
structural non-participant in every transition, the basis for cells D3/D4
below.

## 2. Spring-transition / postseason overlap: measured to be EMPTY, dropped

One of the task's candidate cells was "the spring transition where it
overlaps the postseason." Before writing any cell for it, the population
was checked (a diagnostic on game *dates*, not on any cover-rate outcome --
the same admissible pre-scoring exception `docs/team_style.md`'s
reliability gate and this battery's own date-scan above both rely on).
**Measured**, from the newest `data/raw/*/schedules.parquet` snapshot: the
latest postseason (`game_type != 'REG'`) `gameday` for every completed
season 2009-2025 is `2026-02-08` (the Super Bowl for the 2025 season); the
full per-season list of latest postseason dates ranges `2010-02-07`
(season 2009) to `2026-02-08` (season 2025), i.e. always the first half of
February. The spring transition (table above) never falls earlier than
March 8 in any season. **No NFL game in 2009-2025 has ever been played
after that season's spring transition** -- the candidate cell has zero
population. This is a data fact, not a design choice, and it is disclosed
here instead of silently omitted: the postseason has finished, every
season, before the clocks change in March. This candidate is dropped and
no cover-rate outcome was ever examined for it.

## 3. Population and machinery (reused, not rewritten)

- Base population: `nfl_travel_rest_battery_screen.load_population`
  (imported directly, not reimplemented) against the newest
  `data/raw/*/schedules.parquet` snapshot and
  `registry/stadium_coordinates.json` -- REG 2009-2025, `add_ats_outcomes`
  for `home_cover` (pushes/missing spread dropped), `week_block =
  season*100+week`, and the already-built, already-validated
  `tz_delta_eastbound` column (DST-aware `zoneinfo` evaluation of venue vs.
  away-team-home UTC offset, per `docs/travel_rest_battery.md`) reused
  verbatim for cell D5. **Measured** scored population: 4,317 games (2009
  spring transition through the newest available 2025 season, after the
  push/missing-spread drop).
- Method reused verbatim from `scripts/fluview_battery_screen.py` /
  `scripts/nfl_travel_rest_battery_screen.py`: `_common.block_bootstrap_two_group`
  (joint week-blocked bootstrap, block = `season*100+week`, PRIMARY;
  season-blocked, block = `season`, SECONDARY), full-slate effect scaling
  via `nfl_ats.experiment_runner.scale_subset_effect` (imported, not
  reimplemented), `probability_positive` = fraction of bootstrap draws with
  gap > 0. 20,000 bootstrap samples, seed `20260826` (repo convention:
  today's date). Within-week correlation is zero by owner mandate -- no ICC
  term, week-blocking is a conservative convenience only.
- **New derived columns, this battery**: `fall_transition_date` (per-season,
  table above, mapped onto every row of that season), `days_since_fall_transition
  = (gameday - fall_transition_date).days`, and the mirror
  `days_since_placebo_anchor` using an anchor 21 days before the real
  transition (cell D6). These are calendar-day offsets computed from the
  MEASURED dates in section 1, not a hardcoded day-of-week rule.
- **Window definition, and why it is calendar-day- not schedule-week-based**:
  a flag of "same `week` number as the transition" would incorrectly
  include that week's Thursday game, which is always 3 days *before* the
  transition Sunday (pre-shock) and exclude the following Thursday game,
  which is 4 days *after* it (genuinely post-shock, still within a week of
  the change). The cells below instead flag `days_since_fall_transition in
  [0, 6]` -- a true 7-calendar-day window starting at the transition Sunday
  itself (that Sunday's slate, the following Monday-night game, and the
  following Thursday-night game), which is mechanistically the right
  boundary even though it does not line up with the schedule's own `week`
  column. Blocking (`week_block`) is unaffected by this choice -- blocking
  is a correlation-grouping device, not a restatement of the flag.

## 4. Predeclared cells (5 scored, 1 dropped in section 2)

All score `home_cover`, subset vs. complement, on the population and with
the method in section 3. `n_flag` figures below are **measured**
population-only diagnostics (no cover-rate outcome examined before
freezing this document, matching the fluview/travel_rest precedent).

**D1. `dst_fall_transition_shock`** (primary cell) -- population: full
scored REG 2009-2025 (n=4,317). Flag: `days_since_fall_transition in [0,
6]`. **Measured n_flag = 222 (5.1% of slate)**, spanning schedule weeks 8-10
depending on season (table measured, one late-season cluster per season, no
outlier). **Predicted sign: POSITIVE** on `home_cover`. Mechanism
(inferred): the national clock change hits both teams in a game
simultaneously, but the away team is *also* absorbing ordinary travel
disruption that week; layering a circadian shock on top of travel fatigue
plausibly costs the traveling side more than it costs a team sleeping in
its own bed and market, unaware of any DST-specific mechanism, should not
price this — mirrors the `travel_rest_eastbound_multizone` mechanism
already in the registry.

**D2.** Dropped -- see section 2 (zero population, not a mined-and-discarded
outcome).

**D3. `dst_arizona_home_shield`** -- restricted population: D1's flag=True
games where either the home team is Arizona (`ARI`, confirmed non-DST-observing
in section 1) or Arizona is not in the game at all (excludes ARI road games
from this specific comparison; the mirror case is D4). Flag: home team is
`ARI`. **Measured n_flag = 4**, complement = 210. Extremely thin -- disclosed
up front, not discovered after scoring; see section 5 on what this does to
interval trustworthiness. **Predicted sign: POSITIVE** on `home_cover` --
the purest form of the D1 mechanism: a home team that had literally zero
clock disruption, hosting a traveling opponent that had both ordinary
travel fatigue AND the national clock shock.

**D4. `dst_arizona_away_shield`** -- mirror of D3: restricted population is
D1's flag=True games where either the away team is `ARI` or Arizona is not
in the game at all. Flag: away team is `ARI`. **Measured n_flag = 8**,
complement = 210. Also thin, disclosed. **Predicted sign: NEGATIVE** on
`home_cover` -- an away team that had zero clock disruption (only its
ordinary travel burden) should outperform a typical away team of that week,
while the home team still absorbed the national shock in its own bed; the
home side's usual edge should be relatively suppressed.

**D5. `dst_transition_eastbound_interaction`** -- restricted population:
games with `tz_delta_eastbound >= 2` (the already-registered eastbound-travel
disadvantage construct, reused verbatim, not rebuilt; **measured** n=575 on
this battery's population, consistent with `travel_rest_eastbound_multizone`'s
own reported ~13% base rate). Flag: also falls in D1's window (`days_since_fall_transition
in [0, 6]`). **Measured n_flag = 39**, complement = 536. **Predicted sign:
POSITIVE** on `home_cover` -- if eastbound circadian disruption and the
national clock shock are both real and additive, the eastbound-travel edge
already hypothesized in `docs/travel_rest_battery.md` should be LARGER
specifically during the transition window than it is on an ordinary
eastbound-travel week elsewhere in the season.

**D6. `dst_placebo_shifted_window`** (negative/specificity control, see
section 6 on what this can and cannot close) -- population: same as D1 (full
scored REG 2009-2025). Flag: `days_since_placebo_anchor in [0, 6]`, where
`placebo_anchor = fall_transition_date - 21 days` -- an identically-shaped
7-day window, 3 weeks before the real transition, chosen as a round,
externally-set buffer (not tuned to any outcome) that lands solidly before
D1's window with zero calendar overlap (**measured**: 0 games in both
windows at once) while staying in the same general stretch of the season
(schedule weeks 5-7 depending on season, avoiding both the structurally
different week-1/2 window and the separately-studied Thanksgiving week).
**Measured n_flag = 236** (comparable size to D1's 222, by construction).
**Predicted sign: NULL** -- no DST mechanism operates on this window; if the
bootstrap/scaling machinery or a generic "this stretch of the season"
effect were driving D1, it should also appear here.

## 5. Block-count floor: read for every cell before treating any interval as a 95% interval

`estimation_variance.MIN_BLOCKS_FOR_INTERVAL = 10` (measured coverage of a
known truth: 0.000 at 1 block, 0.466 at 2, 0.760 at 4, 0.896 at 10, 0.944 at
50, against a nominal 0.95). The `n_blocks` field `_common.summarize` reports
is the number of DISTINCT blocks in the whole restricted population (subset
+ complement together), which is large and NOT the number that governs a
once-or-twice-per-season flag's actual resampling variability -- a block
that contains zero flagged rows contributes nothing to the subset mean no
matter how many times the bootstrap draws it. The screen script therefore
also reports, per cell and per blocking, `n_flag_blocks`: the count of
distinct blocks that contain at least one flag=True row. **This is the
number this battery's own numbers should be read against, not the raw
`n_blocks` field.** For D1/D6 (one ~7-day window per season, 17 seasons,
sometimes spilling across two schedule weeks), `n_flag_blocks` under
week-blocking is expected to land near but not exactly 17; under
season-blocking it is at most 17 by construction. For D3/D4 (ARI
home/away only), `n_flag_blocks` is at most 4 and 8 respectively --
**below the floor of 10 under any blocking**, so those two cells' intervals
are reported but explicitly flagged untrustworthy; the point estimate and
`probability_positive` are the only figures to read, per AGENTS.md ("If a
cell falls below that floor, report the point estimate and
probability_positive... do NOT treat it as a failure or a reason to stop").

## 6. Reliability check

`reliability_check.method = not_applicable` for every cell in this battery.
The flags here are calendar/schedule facts (a game's date relative to a
computed transition date, or whether Arizona is one of the two teams) --
not a persistent per-team trait with a year-over-year correlation to
split-half, matching the same `not_applicable` precedent
`docs/experiment_pipeline.md` documents for `home_underdog`,
`large_favorite`, and every situational (non-trait) `subset_bias` builder,
and matching `travel_rest_thursday_pure`'s identical posture in
`docs/travel_rest_battery.md`.

**On D6 and AGENTS.md's second closing ground.** D6 is a NEGATIVE
(specificity) control, predicted to show nothing -- it is not, by itself,
the "positive control proven able to detect an effect that size" AGENTS.md
names as the second admissible closing ground for the OTHER cells. A
positive control must be shown to have power to detect a REAL effect of
comparable magnitude; a placebo that is expected to be null and comes back
null demonstrates specificity (D1 is not just "any week in this part of the
season"), not detection power. **A null D6 result does not, by itself,
close D1, D3, D4, or D5** under rule 1; it can only strengthen or weaken
confidence that D1's mechanism, if present, is DST-specific rather than a
generic mid-season artifact. The only admissible closing grounds for any
cell in this battery remain a RESOLVED wrong sign (whole interval on the
wrong side of the predicted direction) or a genuine power-proving positive
control, neither of which this battery is designed to produce.

## 7. Files

- `scripts/dst_transition_battery_screen.py` -- measure-only screen; computes
  transition dates via `zoneinfo`, builds the 5 cells, writes
  `artifacts/dst_transition_battery/<UTC>/results.json`. Never writes the
  registry.
- `scripts/record_dst_transition_battery.py` -- records all 5 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`, reading
  every numeric field from the screen's output JSON (no hand-typed
  numbers), regardless of interval shape or block-count floor.

## Recording commitment

Every cell above records to `registry/weak_signals.json` as
`unresolved_below_power` (`league=nfl`, `effect_units=accuracy_points`,
`season_start=2009`, `season_end=2025`) unless a week-blocked interval
sits ENTIRELY on the wrong side of its predicted sign, in which case the
recorder script halts rather than auto-closing (matching
`scripts/record_travel_rest_battery.py`'s own safety check) -- a resolved
wrong sign is a human adjudication against the D1-mechanism-derived
predicted direction, not something this battery pre-authorizes itself to
declare. This battery spends no rotation-registry window (measure-only,
same posture as the fluview and travel/rest batteries) and does not touch
any production code path.
