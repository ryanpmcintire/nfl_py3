# Circadian body-clock screen (ENV-06) — predeclaration

Written **before** `scripts/body_clock_screen.py` scored any cover-rate
outcome. Only population counts (n_flag sizes, threshold feasibility,
missing-data checks) were examined before this document was frozen — no
cover rate, gap, interval, or probability_positive for any cell was
computed or looked at beforehand. Method, population, thresholds, blocking,
seed, and predicted directions are locked below exactly as in
`docs/travel_rest_battery.md`.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign.

## ROADMAP status (read 2026-08-21, before this work)

- `ENV-06` — ?? — "Circadian effects | Test local body-clock hypotheses
  with aggressive shrinkage". Never built before this session (**read**,
  ROADMAP.md line 287: one-line stub with no screen attached).

## Overlap check (why these cells are new ground)

Checked **before** designing cells, per the session mandate:

- `docs/travel_rest_battery.md` (ENV-03/ENV-04, **read** this session): its
  8 cells condition on travel distance, timezone-delta-eastbound,
  neutral-site, rest thresholds, and Thursday — **none conditions on
  kickoff time** (**measured**: no reference to `gametime` anywhere in that
  doc or `scripts/nfl_travel_rest_battery_screen.py`). Its near-null
  `travel_rest_eastbound_multizone` (tz_delta >= 2 eastbound, P+ 0.3239,
  **reported** from the doc) is a timezone-general construct with NO
  kickoff-time requirement — a different question from "does the West
  body clock specifically hurt at a 10am-biological-time kickoff".
- `bias_battery_west_coast_early_kickoff` and
  `bias_battery_west_coast_early_kickoff_opener` exist in
  `registry/weak_signals.json` (**read** this session, lines ~1134-1181):
  Pacific-timezone ROAD team, non-PT opponent, kickoff before 14:00 ET,
  NFL 2020-2025, opener-grade team-game framing, P+ 0.4057. This screen's
  primary cell overlaps that construct in spirit and is DISCLOSED as
  correlated with it, not independent of it — but differs on population
  (2009-2025 vs 2020-2025), grading frame (full-slate accuracy_points
  week-blocked bootstrap vs opener-grade paired team-games), metric
  (home_cover vs team-perspective bias), and body-clock definition
  (includes Arizona/Phoenix clocks, which that entry's "Pacific-timezone"
  wording may exclude). The two must never be sign-test-pooled as
  independent. The remaining five cells here (east-host mirror, late-window
  control, two era splits, midday dose-control) have no recorded
  counterpart at all.
- No other `body_clock*` name exists in `registry/weak_signals.json`
  (**measured**, grep this session).

## Data source and leakage posture

- Newest snapshot `data/raw/20260817T235649Z/schedules.parquet` (**read**;
  same snapshot the weather/travel batteries use).
- **Kickoff time exists directly in the data** — the `gametime` column
  (ET, per nflverse convention — **reported**, consistent with the observed
  13:00/16:05/16:25/20:xx TV-window clustering, **measured** value counts
  this session). Zero missing values across the 4,431 REG 2009-2025 rows
  (**measured**). No TV-window derivation was needed, so **no
  `registry/stadium_timezones.json` was built** — the mandate's fallback
  is moot.
- Body-clock and venue timezones come from the existing
  `registry/stadium_coordinates.json` IANA `tz` field (**read**; built
  2026-08-19 for ENV-03/04, 0 unresolved stadium names in this population,
  re-verified this session: **measured**, exhaustive diff returned an empty
  unresolved set). Away team's body clock = its own MODAL home stadium that
  season (`groupby(["home_team","season"])["stadium"]` mode over
  `location=='Home'` rows — same relocation-resolving convention as the
  travel battery: OAK→LV, SD→LAC resolve automatically).
- Pregame-safe: `gametime`, `gameday`, `stadium`, `location`, and team
  affiliation are all schedule facts known before kickoff. No leakage
  caveat.
- Population diagnostics measured before freezing (**counts only, no
  outcomes**): 4,431 REG rows; 61 neutral-site; C1 n=363 (era splits
  163/200); C2 n=231; C3 n=57; C6 n=279; C1 kickoff times are exclusively
  13:00 ET.

## Derived quantities

- `kick_min`: minutes-past-midnight ET parsed from `gametime`
  (`HH:MM`), e.g. 13:00 → 780.
- `away_body_tz`: IANA zone of the away team's modal home stadium that
  season. WEST body clock := tz in {`America/Los_Angeles`,
  `America/Phoenix`} — exactly SEA, SF, LAC, LAR, ARI, LV plus historical
  OAK/SD codes (**measured** team list this session); Denver (Mountain) is
  deliberately EXCLUDED (the mandate's six-team list omits it, and a 2-hour
  offset is a materially milder circadian shift than 3).
- Venue EAST := venue tz == `America/New_York`; venue WEST := venue tz in
  the same two-zone set.
- All cells additionally require `location == 'Home'` (true home/road
  games), so the 61 international/neutral games sit in every complement by
  construction rather than being dropped.

## Method (reused verbatim from `scripts/nfl_travel_rest_battery_screen.py`)

- `home_cover` from `nfl_ats.features.add_ats_outcomes` (pushes dropped).
- Subset-vs-complement full-slate-scaled effect: `(subset_cover −
  complement_cover) × 100 × fraction_of_slate`.
- Week-blocked joint bootstrap primary (block = `season*100+week`),
  season-blocked secondary (block = `season`), same
  `block_bootstrap_two_group` algorithm.
- **20,000 samples, seed 20260821** (mandated).
- `probability_positive` = fraction of bootstrap draws with gap > 0.

## The 6 predeclared cells

All score `home_cover` on the same REG 2009-2025 population (pushes/
missing spread dropped). Predictions fixed BEFORE scoring.

1. **`body_clock_west_road_early`** — away body clock WEST, true road game,
   kickoff < 14:00 ET (i.e. 10am Pacific biological time). n_flag=363
   (diagnostic count). Predicted: **positive home_cover edge** (classic
   documented circadian mechanism: West-coast-body-clock teams performing
   worse in early-window games).
2. **`body_clock_east_host_west_visitor_early`** — cell 1 ∩ venue tz EAST
   (the mirror: Eastern hosts receiving Western visitors at 1pm ET).
   n_flag=231. Predicted: **positive home_cover edge** (stricter version of
   cell 1; removes Central-host games where the visitor's shift is partly
   absorbed).
3. **`body_clock_west_host_east_visitor_late`** — CONTROL, expect null:
   venue tz WEST, away body clock EASTERN, true home game, kickoff
   ≥ 19:00 ET (late window/SNF/MNF — visitor's biological time is ~4pm,
   no circadian disadvantage). n_flag=57 (thin — disclosed, not hidden).
   Predicted: **null** (no direction claimed; a resolved wrong SIGN would
   be surprising, not disqualifying either way at this n).
4. **`body_clock_west_road_early_2009_2016`** — cell 1 restricted to
   2009-2016. n_flag=163. Stability split; same prediction as cell 1.
5. **`body_clock_west_road_early_2017_2025`** — cell 1 restricted to
   2017-2025. n_flag=200. Stability split; same prediction as cell 1.
6. **`body_clock_west_road_midday_control`** — away body clock WEST, true
   road game, 14:00 ≤ kickoff ET < 17:00 (12pm-1pm Pacific biological
   time). n_flag=279. Dose-response control: if cell 1's mechanism is
   kickoff-time-specific rather than a generic "West road team" effect,
   this should read WEAKER than cell 1. Predicted: positive but smaller
   than cell 1 (relative statement, not a standalone directional bet).

## Recording commitment

Every cell above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, `season_start=2009`, `season_end=2025`,
regardless of interval shape. This screen writes no registry JSON itself
and spends no rotation-registry window (measure-only, same posture as the
weather/travel batteries); it stamps a run log under
`registry/experiments/body-clock-screen/` via
`write_experiment_artifact`. The exact record command lines are returned
by the session, numbers passed through unmodified from the artifact JSON.
The only admissible alternative classification would be a RESOLVED wrong
sign (whole interval on the wrong side of the predicted direction) — none
is claimed here regardless of what the numbers show.

## Results (measured 2026-08-21, post-freeze)

Artifact:
`artifacts/body_clock_screen/20260821T182157Z/results.json` (**measured**,
seed 20260821, 20,000 draws; scored population 4,317 REG games after
dropping 114 pushes/missing-spread rows from 4,431; 0 unresolved stadium
names; 0 rows missing required cell inputs). All numbers below are
week-blocked primary full-slate accuracy points; season-blocked secondary
in parentheses.

| cell | n_flag | effect pts | week-blocked 95% CI | P+ | season-blocked P+ |
|---|---|---|---|---|---|
| `body_clock_west_road_early` | 352 | −0.1545 | [−0.6272, +0.3106] | 0.2588 | 0.2704 |
| `body_clock_east_host_west_visitor_early` | 227 | −0.2690 | [−0.6232, +0.0917] | 0.0714 | 0.0905 |
| `body_clock_west_host_east_visitor_late` (control) | 57 | +0.0265 | [−0.1471, +0.1991] | 0.6156 | 0.5976 |
| `body_clock_west_road_early_2009_2016` | 156 | −0.0308 | [−0.3268, +0.2738] | 0.4217 | 0.4450 |
| `body_clock_west_road_early_2017_2025` | 196 | −0.1175 | [−0.4583, +0.2272] | 0.2487 | 0.1887 |
| `body_clock_west_road_midday_control` | 270 | −0.1984 | [−0.5781, +0.1781] | 0.1487 | 0.1517 |

Reading (**inferred**, from the measured table above — no closure claimed):

- The control cell behaved as predeclared: near-null (+0.03 pts, P+ 0.62),
  so the instrument is not producing a spurious everywhere-positive drift.
- Every directional cell's point estimate leans OPPOSITE the predeclared
  direction (home covers LESS when hosting a West-body-clock visitor at an
  early kickoff), most strongly in the east-host mirror (P+ 0.0714 for the
  predicted direction). **None of these intervals sits entirely below
  zero** — the closest is the mirror at [−0.6232, +0.0917] — so per the
  binding taxonomy this is NOT a resolved wrong sign and cannot close
  anything; it is a category-3 unresolved lean, recorded as such.
- Era stability: both splits lean the same (anti-predicted) way, weaker in
  2009-2016 than 2017-2025, consistent with the pooled cell rather than
  contradicting it.
- Dose-response check did not order cleanly: the midday control (−0.1984)
  leans slightly MORE anti-predicted than the early primary (−0.1545), so
  the kickoff-time-specificity story is not supported at this resolution —
  reported plainly, not spun.
- These cells are correlated with each other (cells 2/4/5 are subsets of
  cell 1) and with `bias_battery_west_coast_early_kickoff(_opener)`; never
  sign-test-pool them as independent.

All six cells were recorded `unresolved_below_power` via
`nfl-ats weak-signals record` (exact command lines returned by the session;
numbers passed through unmodified from the artifact JSON above).
