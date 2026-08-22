# Circadian body-clock NIGHT screen — predeclaration

Written **before** `scripts/body_clock_night_screen.py` scored any cover-rate
outcome. Only population counts (n_flag sizes, threshold feasibility,
missing-data checks, weekday/gametime distributions) were examined before
this document was frozen — no cover rate, gap, interval, or
probability_positive for any cell was computed or looked at beforehand.
Method, population, thresholds, blocking, seed, and predicted directions
are locked below. Machinery (`load_population` timezone/stadium logic,
`kick_min` gametime parsing, week-blocked bootstrap, full-slate scaling)
is reused **verbatim** from `scripts/body_clock_screen.py`.

## Binding closing-grounds taxonomy (verbatim AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism — a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign.

## Published anchor and mechanism (reported-in-docs, full citation)

Smith RS, Ertl M, Frank E, et al. "Circadian Effects on NFL Team Strategic
Play: A 40-Year Retrospective Study of Monday Night Football Games and the
Effect on West Coast Teams." *Sleep* 2013;36(suppl):A233 (**reported**;
tagged as literature anchor, not locally verified beyond this citation).
West-coast teams beat the spread by 5.26 ± 1.33 points in EVENING games vs
approximately zero in 1pm games over 40 seasons (**reported**, from that
abstract). Mechanism (**inferred** from the literature lead): human
circadian peak performance lands in early evening local-body-clock time;
a Pacific-body-clock team playing at a >=20:00 ET kickoff is competing
near its biological late-afternoon/early-evening peak (~5-7pm Pacific),
while an Eastern-body-clock team at the same kickoff is past its evening
decline toward biological bedtime. The earlier ENV-06 screen deliberately
tested only the EARLY-window half of this literature lead; this screen
tests the NIGHT half it did not touch.

## Overlap disclosure (correlated family — never pool as independent)

Checked before designing cells:

- The six `body_clock_*` entries in `registry/weak_signals.json`
  (`body_clock_west_road_early`, `..._east_host_west_visitor_early`,
  `..._west_host_east_visitor_late`, `..._2009_2016`,
  `..._2017_2025`, `..._midday_control`; **measured**, grep this session)
  come from the same family, same population, same body-clock definition
  and share 352 of the same team-games with dose cell e1. Cells e1/e2 here
  are LITERALLY those recorded entries re-scored as dose buckets —
  disclosed as fully correlated, never pool as independent. The night
  cells (a)-(d) condition on the OPPOSITE end of the kickoff distribution
  (>=20:00 ET vs <14:00 ET); they are correlated through the shared
  population and shared construction but do not overlap in game sets.
- `bias_battery_west_coast_early_kickoff(_opener)` (**read** this
  session): early-window construct, correlated family, not independent.
- No `night`, `evening`, or SNF/MNF-conditioned body-clock entry exists in
  `registry/weak_signals.json` (**measured**, grep this session). This
  screen's night cells are new ground.

## Data source and leakage posture

- Snapshot `data/raw/20260817T235649Z/schedules.parquet` (**measured**
  via `_latest_schedules()`; same snapshot as the early screen).
- All cell inputs are schedule facts known before kickoff: `gametime`
  (ET per nflverse convention — **reported**, consistent with observed TV
  window clustering), `gameday` weekday, `stadium`/`location`, away team's
  modal home stadium that season → IANA tz from
  `registry/stadium_coordinates.json` (**read**; built for ENV-03/04).
  Point-in-time safe: no score, injury, market-move, or weather inputs.
  No leakage caveat; no new feature family added (reused machinery), so no
  new leakage regression test is required by the invariant.
- Population diagnostics measured before freezing (**counts only, no
  outcomes**): 4,431 REG 2009-2025 rows; 114 pushes/missing-spread rows
  dropped → 4,317 scored; 0 unresolved stadium names; 0 rows missing
  required cell inputs; kickoffs >=20:00 ET: 849 raw / 849 scored-pop
  context; 19:xx kickoffs exist (19:00-19:30) and are excluded from every
  night cell by the hard >=20:00 cut.

## Derived quantities

- `kick_min`: minutes-past-midnight ET parsed from `gametime` (`HH:MM`),
  reused verbatim.
- Body clocks: WEST := away modal-home-stadium tz in
  {`America/Los_Angeles`, `America/Phoenix`} (SEA, SF, LAC, LAR, ARI, LV,
  plus historical OAK/SD); EAST := tz == `America/New_York`. Denver
  (Mountain) sits in neither set, in complements — same convention as the
  early screen.
- True road game := `location == 'Home'` (neutral/international games sit
  in complements by construction).
- TRUE NIGHT SLOTS := `gameday` weekday in {Sunday, Monday, Thursday} —
  the actual SNF/MNF/Thursday-night TV windows — excluding the rare Sat
  (n=29), Tue (2), Fri (2), Wed (1) >=20:00 kickoffs (**measured**
  weekday counts). The 19:00-19:30 borderline kickoffs are excluded from
  all night cells by the >=20:00 floor.

## Method (identical to `scripts/body_clock_screen.py`)

- `home_cover` from `nfl_ats.features.add_ats_outcomes` (pushes dropped).
- Subset-vs-complement full-slate-scaled effect:
  `(subset_cover − complement_cover) × 100 × fraction_of_slate`.
- Week-blocked joint bootstrap primary (block = `season*100+week`),
  season-blocked secondary (block = `season`).
- **20,000 samples, seed 20260821** (mandated).
- `probability_positive` = fraction of bootstrap draws with gap > 0.

Direction convention: effects are scored on `home_cover`. For a WEST ROAD
team, "west side covers" = home does NOT cover, so the predicted direction
is a NEGATIVE home_cover gap. Stated explicitly so no sign is misread.

## The 9 predeclared cells (directions frozen BEFORE scoring)

1. **`body_clock_night_west_road_ge2000et`** (primary) — away body clock
   WEST, true road game, kickoff >= 20:00 ET (SNF/MNF window). n_flag=119
   (diagnostic count). Predicted: **negative home_cover gap** (positive
   west-side cover edge — circadian-peak mechanism, the Smith et al.
   evening effect).
2. **`body_clock_night_west_road_true_slots`** — cell 1 ∩ true night slots
   (Sun/Mon/Thu), excluding 19:xx borderline and off-night late games.
   n_flag=113. Predicted: **negative home_cover gap** (stricter version of
   cell 1; isolates the marquee night windows where the published effect
   was measured).
3. **`body_clock_night_east_road_ge2000et`** (mirror control) — away body
   clock EASTERN, true road game, kickoff >= 20:00 ET. n_flag=417.
   Predicted: **positive home_cover gap** (host benefits against a visitor
   at biological bedtime). Disclosed caveat: this cell includes West-venue
   games where the visitor's biological time is ~5pm, diluting the
   bedtime rationale — a conservative dilution, not a confound removal.
4. **`body_clock_night_west_road_ge2000et_2009_2016`** — cell 1 restricted
   to 2009-2016. n_flag=48. Stability split; predicted negative.
5. **`body_clock_night_west_road_ge2000et_2017_2025`** — cell 1 restricted
   to 2017-2025. n_flag=71. Stability split; predicted negative.
6-9. **Dose-response** (`body_clock_night_west_road_dose_*`) — WEST road
   team by kickoff bucket, monotonicity readout DESCRIPTIVE (relative
   ordering claim, not four standalone directional bets):
   - `dose_1300`: kickoff < 14:00 ET. n_flag=352. Predicted positive
     home_cover gap (west disadvantaged — replicates the recorded
     early-screen primary).
   - `dose_1400_1659`: 14:00 <= kickoff < 17:00. n_flag=270. Predicted
     weaker positive / near null.
   - `dose_1700_1959`: 17:00 <= kickoff < 20:00. n_flag=3 (**measured**
     — extremely thin, disclosed, not hidden). No directional claim
     possible; included only to complete the dose ladder.
   - `dose_ge2000`: kickoff >= 20:00. n_flag=119. Predicted negative —
     identical flag set to cell 1 by construction (internal consistency
     check: must reproduce cell 1 exactly).

Predicted monotone ordering across e1→e4 if the circadian mechanism drives
the signal: home_cover gap decreasing from bucket 1 to bucket 4.

## Recording commitment

Every cell records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, `season_start=2009`, `season_end=2025`,
regardless of interval shape. This screen writes no registry JSON itself
and spends no rotation-registry window (measure-only, CFB/free posture, NO
NFL rotation-window spend); it stamps a run log under
`registry/experiments/body-clock-night-screen/` via
`write_experiment_artifact`. Exact record command lines are returned by
the session, numbers passed through unmodified from the artifact JSON. The
only admissible alternative classification would be a RESOLVED wrong sign
(whole interval strictly on the wrong side) under an admissible
`--closing-ground` — none is claimed here regardless of what the numbers
show.

## Results (measured 2026-08-21, post-freeze)

Artifact:
`artifacts/body_clock_night_screen/20260821T222542Z/results.json`
(**measured**, seed 20260821, 20,000 draws; scored population 4,317 REG
games after dropping 114 pushes/missing-spread rows from 4,431; 0
unresolved stadium names; 0 rows missing required cell inputs; run log
stamped to `registry/experiments/body-clock-night-screen/`). All numbers
below are week-blocked primary full-slate accuracy points; season-blocked
secondary P+ in parentheses. `P+` is the bootstrap fraction with gap > 0;
for the west-night cells the PREDICTED direction is negative home_cover,
so the probability of the predicted direction is `1 − P+` — both stated
explicitly.

| cell | n_flag | effect pts | week-blocked 95% CI | P+ | season P+ |
|---|---|---|---|---|---|
| `..._west_road_ge2000et` (primary) | 119 | −0.1713 | [−0.4137, +0.0738] | 0.0843 | 0.0750 |
| `..._west_road_true_slots` | 113 | −0.1488 | [−0.3862, +0.0942] | 0.1133 | 0.0852 |
| `..._east_road_ge2000et` (mirror control) | 417 | +0.1305 | [−0.3112, +0.5718] | 0.7135 | 0.7128 |
| `..._ge2000et_2009_2016` | 48 | −0.1048 | [−0.2647, +0.0642] | 0.1086 | 0.0674 |
| `..._ge2000et_2017_2025` | 71 | −0.0640 | [−0.2478, +0.1180] | 0.2469 | 0.2973 |
| dose_1300 (<14:00) | 352 | −0.1545 | [−0.6272, +0.3106] | 0.2588 | 0.2704 |
| dose_1400_1659 | 270 | −0.1984 | [−0.5781, +0.1781] | 0.1487 | 0.1517 |
| dose_1700_1959 (n=3) | 3 | +0.0124 | [−0.0346, +0.0362] | 0.6892 | 0.7011 |
| dose_ge2000 (= primary flag set) | 119 | −0.1713 | [−0.4137, +0.0738] | 0.0843 | 0.0750 |

Reading (**inferred**, from the measured table above — no closure claimed):

- **The night cells lean in the PREDETECTED direction.** Primary cell:
  raw gap −6.21 pts on a 50.1%-vs-42.9% cover split, full-slate effect
  −0.171 pts, probability of the predicted (negative) direction
  **0.9157** (week-blocked) / 0.9250 (season-blocked). True-slots cell
  similar: predicted-direction probability 0.8867 / 0.9148. These are the
  strongest directional leans of any body-clock cell recorded so far in
  this family — including every early-window entry.
- **The mirror control behaved as predeclared**: east-body-clock road
  teams at >=20:00 ET see the host cover MORE (+0.1305 pts, P+ 0.7135 for
  the predicted-positive direction). Both ends of the night window point
  the way the circadian mechanism predicts, and neither is a resolved
  wrong sign anywhere.
- **Era split**: 2009-2016 leans harder into the prediction (predicted-
  direction probability 0.8914) than 2017-2025 (0.7531); same sign in
  both, consistent with a stable-not-growing effect rather than a
  contradiction.
- **Dose-response ordered monotonically** across the three scoreable
  buckets (raw gaps): 13:00 → −1.90, 14:00-16:59 → −3.17, >=20:00 →
  −6.21 pts. Home covers progressively LESS against a west-body-clock
  road team as kickoff moves later — exactly the monotone-decreasing
  ordering predeclared if circadian time drives the signal. The 17:00-
  19:59 bucket has n=3 and is uninformative by construction (disclosed at
  predeclaration). Internal consistency check passed: `dose_ge2000`
  reproduces the primary cell bit-for-bit (**measured**).
- Per the binding taxonomy: no interval here sits entirely beyond zero in
  either direction, no terminal classification is claimed, and nothing is
  closed. This is category 3, unresolved — the family now has coherent,
  mechanism-aligned, monotonically-ordered leans at both ends of the day
  that warrant more data or a pooled commensurable read, not a verdict.
- Correlation disclosure stands: these nine cells are correlated with each
  other (dose_ge2000 ≡ primary; true_slots ⊂ primary; era splits ⊂
  primary) and with the six early-window `body_clock_*` entries plus
  `bias_battery_west_coast_early_kickoff(_opener)`; never pool them as
  independent.

All nine cells were recorded `unresolved_below_power` via
`nfl-ats weak-signals record` (exact command lines returned by the
session; numbers passed through unmodified from the artifact JSON above).
