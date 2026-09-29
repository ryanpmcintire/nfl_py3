# LEAD-73 unit 1: local pre-2023 source inventory

**Measured:** one offline probe per declared candidate; three inventory probes and zero statistical looks.
No outcomes loaded, fits, scoring, network requests, or registry writes.

Protocol was saved in `docs/lanes/lead73.md` before this run. Source policies were read before the probes.
Regular-season NFL games, 2009-2022; timestamp bounds follow
`scripts/sunday_market_probability_eval.py:22-68`: Monday history, Wednesday-or-later second capture,
deadline at the earlier of kickoff and Sunday 12:45 ET. Freeze is Tuesday noon ET. Equality with deadline is
excluded. File modification time and retrieval time never substitute for capture time.

## Candidate inventory

**Measured, Wayback:** 13 manifests, 203 entries, 189 distinct captures with existing SHA-256-verified raw
files; 0 entries rejected. Earliest raw capture: 2005-10-01T02:34:12+00:00. The locally identified archive
is VegasInsider; its game-level timing is below. No claim about archives absent from these local roots.

**Measured, VegasInsider:** 8 pre-2023 season tables, 12033 distinct rows, 9470 exact schedule-matched rows;
0 rows lack verified capture provenance. There are 684 games with a verified spread cell after freeze and
before deadline; 217 have two same-book captures with the second on/after Wednesday. **Inferred:** timing
evidence only; signed home-spread orientation and book comparability still need verification before reuse in
the four-term fit.

**Measured, cached Odds API:** 455 selected pre-2023 NFL spread manifests from 8834 local manifests; 0 quote
files rejected by integrity check; 87660 matched spread observations. There are 759 games with post-freeze,
pre-deadline quotes; 758 same-book pair games; 757 pair games from incumbent leaders. **Read:** new paid
acquisition remains cancelled (`docs/lanes/done/odds-api-key-deactivated.md:3`).

Wayback and VegasInsider share raw evidence; their coverage must not be added. The VegasInsider probe
inspects board tables, not an open/close label as timestamp proof. The Odds API probe excludes futures,
period markets, unverified quote files, missing book-update instants, and book updates after observation.

## Earliest usable capture relative to both gates

**Measured, Wayback/VegasInsider:** 2009-09-17T20:59:06+00:00; 52.985 hours after freeze; 67.765 hours
before deadline (freeze 2009-09-15T16:00:00+00:00, deadline 2009-09-20T16:45:00+00:00).

**Measured, Odds API:** 2020-09-10T21:55:00+00:00; 53.917 hours after freeze; 66.833 hours before deadline
(freeze 2020-09-08T16:00:00+00:00, deadline 2020-09-13T16:45:00+00:00).

## Coverage by season

These are source counts, not decisive-game or accuracy results. A pair means two distinct timestamps for one
game/book before deadline, with the later timestamp on/after Wednesday. A freeze pair also brackets Tuesday
noon. No requirement that the line changes is imposed. Incumbent leader keys are read from
`src/nfl_ats/sharp_book_movement_features.py:24`; zero matching VegasInsider keys does not establish that
its books have no information.

| Source | Season | Post-freeze games | Pair games | Freeze pair games | Leader pair games |
|---|---:|---:|---:|---:|---:|
| VegasInsider | 2009 | 34 | 20 | 0 | 0 |
| VegasInsider | 2010 | 92 | 24 | 12 | 0 |
| VegasInsider | 2011 | 143 | 46 | 35 | 0 |
| VegasInsider | 2012 | 48 | 45 | 24 | 0 |
| VegasInsider | 2013 | 57 | 0 | 0 | 0 |
| VegasInsider | 2014 | 100 | 12 | 0 | 0 |
| VegasInsider | 2015 | 99 | 23 | 10 | 0 |
| VegasInsider | 2016 | 111 | 47 | 11 | 0 |
| Odds API | 2020 | 237 | 236 | 228 | 235 |
| Odds API | 2021 | 253 | 253 | 251 | 253 |
| Odds API | 2022 | 269 | 269 | 269 | 269 |

**Measured:** cached Odds API decision-label counts: `{"mon_pre_mnf": 65, "sat_midday": 65,
"sun_early_close": 65, "sun_late_close": 65, "thu_pre_tnf": 65, "true_open": 65, "tue_open": 65}`.

## Decision and limits

**Measured:** 757 pre-2023 games have timestamp-admissible cached pairs from incumbent leaders.
**Inferred:** the local timestamp source gate clears for unit-2 preparation; model-logit/composition
coverage, exact move reconstruction, and the frozen four-term LOSO protocol still need verification before
fitting.

**Read:** the four terms are model logit, composition sum, market move, and move availability
(`docs/lanes/done/market-move-decomposition.md:30-33`). This assigned unit implements source inventory only.
No model has been refitted. No extension size or detection power is inferred from source counts.

In-sample, out-of-sample, their gap, per-fold coefficients, calibration, decisive-game record, effect
intervals, and probability_positive: **not estimated**; no outcome read or statistical look occurred.
Intervals do not apply to this finite local-file census. This source result neither closes nor promotes a
signal. AGENTS.md:65-85 requires admissible evidence for closure; AGENTS.md:87-105 requires one fitted
probability and out-of-season parameter choice before scoring.

Registry commands: none. Source inventory supplies no signal estimate to record; fabricating a neutral
effect or probability_positive would misstate the evidence. The orchestrator owns any later record command
and unit-2 fit.

Reproduce: `.tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit1.py`.
