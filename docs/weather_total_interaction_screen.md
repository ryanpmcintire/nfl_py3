# Weather x market-expectation (betting total) interaction screen — predeclaration

Written **before** `scripts/weather_total_interaction_screen.py` scores
anything. Per AGENTS.md this is a mined/exploratory screen family: every cell
is predeclared to record `unresolved_below_power` regardless of interval shape
(an interval crossing zero is the EXPECTED outcome for a real small signal at
this evaluator's ~2-point resolution, never a rejection ground). Method,
population, cells, and predicted directions are locked before any cell's sign
is seen.

## Family and adjacency (disclosed up front)

Family: weather absolutes **conditioned on the betting TOTAL as the
passing-environment expectation**. Prior batteries tested weather absolutes
(`weather_battery_*`, ENV-01) and climatological gaps (`weather_followup_*`)
but never conditioned on the total.

Adjacent already-recorded signals, disclosed so nothing here is ever pooled as
independent of them:

- `forecast_weather_kn_precip_high_total_full` / `_pre2020` (recorded
  2026-08-20): outdoor AND kickoff-nearest forecast precip probability >= 60%
  AND `total_line >= 47` (fixed threshold), home_cover outcome, predicted
  POSITIVE home edge. This battery's precip cell shares the >= 60% precip
  threshold but conditions on a total TERCILE instead of a fixed 47, scores
  the FAVORITE-cover outcome instead of home_cover, and predicts the NEGATIVE
  direction — same mechanism family, correlated construction, one look each,
  do not sign-test-pool together.
- `weather_battery_high_wind_outdoor` (+0.16 pts P+ 0.757) and
  `weather_battery_high_wind_road_favorite`: wind >= 15 marginal cells. Every
  wind cell here is that same flag INTERACTED with a total tercile — a
  decomposition of a shared window, not an independent draw.

Mechanism under test: adverse weather hurts games the market expects to be
pass-heavy (high total) more than low-total grind-it-out games, where some
weather suppression is already in the price. Operationalized as: **the
favorite covers LESS often than its base rate when adverse weather hits a
top-tercile-total game** ("HIGH-total favorite covers LESS" — direction
predeclared exactly in those words by the task brief).

## Data sources and leakage posture

- Schedules: newest `data/raw/*/schedules.parquet`. REG 2009-2025 only.
  `total_line` measured present for 100% of REG games in every scored season
  2009-2025 (2026 partial season excluded by the window), so no season
  restriction beyond the standard window is needed.
- `temp`/`wind` are GAME-TIME ACTUALS, not pregame forecasts — the documented
  caveat inherited verbatim from `docs/weather_followup.md` and both parent
  scripts. Cells touching this game's own actual temp/wind are MECHANISM
  SCREENS, upper bounds on a future forecast-time feature, not themselves
  pregamable. Measured missingness on this snapshot (REG 2009-2025): ~25-29%
  of games per season 2009-2019, rising to 36% (2020), 33% (2021), 65%
  (2022), 45% (2023), 36% (2024), 35% (2025) — the documented 2022 gap
  reproduces at 65% here (the 49% figure quoted elsewhere counts a slightly
  different denominator); effective n per cell is reported from the run's own
  `n_missing_required_data` counters, not assumed.
- Precipitation: there is NO actual-precip field in schedules. The only
  precipitation source in the repo is the kickoff-nearest GFS-MOS forecast
  archive (`data/raw/forecast_archive/kickoff_nearest_2009_2025/
  forecasts.parquet`, `forecast_precip_prob_pct`, GFS MOS p06 falling back to
  p12; measured 1.2% missing, fetch_status ok on 4,379 of 4,431 rows). That
  archive is genuinely PREGAME-AVAILABLE, so the precip cell does NOT carry
  the actual-weather upper-bound caveat — it is the one cell here whose
  trigger could be consumed at prediction time (at close).
- `total_line` is the market's closing total: pregame-known only AT THE CLOSE,
  not at a Tuesday opener. Disclosed limitation for any future feature use.
- Tercile cuts are computed POOLED over the full scored population (all
  seasons 2009-2025 at once), not per-season. Consequences, disclosed: the
  cuts are not season-causal (they use later games' totals) and top-tercile
  membership skews toward recent high-scoring eras — but the cuts are a
  market-level normalization constant, outcome-independent, and no cover
  outcome enters their computation. A per-season cut is the obvious
  robustness variant and was NOT run (one predeclared construction).

## Population and outcome

REG 2009-2025, pushes/missing dropped via `add_ats_outcomes` (reused
verbatim), then games with `spread_line == 0` or missing `spread_line` or
missing `total_line` are excluded via the missing mask (no defined favorite /
no tercile assignment; counted and reported, not silently dropped).

Outcome: `favorite_cover` = 1 if the favorited team covered, 0 otherwise,
derived from `add_ats_outcomes`' `ats_margin` (= result - spread_line, home
perspective): favorite covers iff `ats_margin > 0` when `spread_line > 0`
(home favored) or `ats_margin < 0` when `spread_line < 0` (away favored).
This departs from the parent batteries' home_cover convention ON PURPOSE:
the predeclared direction is about the FAVORITE, which home_cover cannot
express without conflating favorite identity. The complement arm is all
scored games outside the flag, same subset-vs-complement convention as every
parent battery.

Tercile cuts: `q1 = quantile(total_line, 1/3)`, `q2 = quantile(total_line,
2/3)` on the scored population after the exclusions above. Top tercile =
`total_line > q2`; bottom tercile = `total_line < q1`; middle band belongs to
neither.

## The 4 predeclared cells (frozen before scoring)

All cells score `favorite_cover`, subset vs. complement, week-blocked joint
bootstrap primary (block = `season*100 + week`), season-blocked secondary
(block = `season`), 20,000 samples, seed `20260821`, full-slate-scaled effect
(`raw_gap_pts * fraction_of_slate`), method reused verbatim from
`scripts/nfl_weather_battery_screen.py::block_bootstrap_two_group`.
Effect sign convention: positive = favorite covers MORE in the flag subset
than the complement; the predicted directions below are NEGATIVE for cells
1, 2, 4 and NONE for the control.

1. **`wxtot_wind15_top_total`** — outdoor/open roof AND game-time actual wind
   >= 15 mph AND top-tercile total. **Predicted: NEGATIVE** (HIGH-total
   favorite covers LESS; wind degrades the pass-heavy-expected game's
   favorite disproportionately). Actual-wind upper-bound caveat applies.
2. **`wxtot_cold35_top_total`** — outdoor/open roof AND game-time actual temp
   <= 35F AND top-tercile total. **Predicted: NEGATIVE** (same mechanism,
   cold axis). Actual-temp upper-bound caveat applies.
3. **`wxtot_wind15_bottom_total`** — outdoor/open roof AND game-time actual
   wind >= 15 mph AND BOTTOM-tercile total. **CONTROL, no predicted
   direction**: the mechanism says weather is partially priced into
   low-total grind-it-out games, so this cell should read near null. It is a
   mechanism-discrimination control, not a discovery cell.
4. **`wxtot_precip60_top_total`** — outdoor/open roof AND kickoff-nearest
   GFS-MOS forecast precipitation probability >= 60% AND top-tercile total.
   **Predicted: NEGATIVE** (same mechanism, precip axis). Pregame-available
   trigger; NO actual-weather caveat. Adjacent to
   `forecast_weather_kn_precip_high_total_*` (disclosed above).

## Recording commitment

Every cell records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, regardless of interval shape, via exact
command lines returned by the run (numbers passed through unmodified from the
results JSON — no hand-typed values). The only exceptions admissible under
AGENTS.md would be a RESOLVED wrong sign (whole interval strictly above zero
on these negative-direction cells) or a positive-control bound, neither of
which this measure-only screen is designed to produce. Cell 3 (control) also
records `unresolved_below_power` — a near-null control is the mechanism's
predicted shape, not a failure.

---

## Results (measured 2026-08-21, post-predeclaration)

Run: `artifacts/weather_total_interaction_screen/20260821T182254Z/results.json`
(seed 20260821, 20,000 draws). Scored population **measured**: 4,313 games
(4,431 REG 2009-2025 minus 114 pushes/missing result, 4 pick'em, 0 missing
spread_line, 0 missing total_line). Pooled tercile cuts **measured**: low
43.0, high 46.5. Effective n per cell: the three actual-weather cells each
had 1,389 rows (32.2%) with missing temp/wind or non-outdoor roof — flag
forced False, included in complement (the documented 2022 gap dominates);
the precip cell had only 51 (1.2%). All numbers below **measured** from the
results JSON; week-blocked primary unless stated.

| cell | predicted | n_flag | effect (pts) | 95% CI | P+ | season-blocked P+ |
|---|---|---|---|---|---|---|
| `wxtot_wind15_top_total` | negative | 69 | **+0.0772** | [-0.1198, +0.2823] | 0.7799 | 0.8785 |
| `wxtot_cold35_top_total` | negative | 62 | -0.0071 | [-0.1767, +0.1655] | 0.4625 | 0.4529 |
| `wxtot_wind15_bottom_total` (control) | none | 150 | **+0.1366** | [-0.1460, +0.4221] | 0.8280 | 0.8889 |
| `wxtot_precip60_top_total` | negative | 50 | **+0.2243** | [+0.0726, +0.3741] | 0.9978 | 0.9995 |

Reads, all **inferred** from the measured table above:

1. **The precip cell resolves OPPOSITE to its predeclared negative
   direction**: the whole week-blocked interval sits above zero AND the
   season-blocked secondary does too ([+0.0964, +0.3484], P+ 0.9995) — the
   HIGH-total favorite covers MORE under forecast precip (68.0% vs 48.7%
   raw), not less. This sign-refutes the task brief's "weather hurts
   high-total favorites" mechanism on the precip axis and agrees in sign
   with the already-recorded `forecast_weather_kn_precip_high_total_*`
   home-edge cells (the favorite is usually the home team). The registry
   validator defines `wrong_sign_resolved` as an interval entirely BELOW
   zero (`src/nfl_ats/weak_signals.py:225`, positive-predicted convention)
   and cannot express this mirror-image closure, so per AGENTS.md ("if a
   record command errors, the verdict is wrong, not the validator") it
   records `unresolved_below_power` carrying the mirror-image resolution
   verbatim in its classification_evidence. Caveat: n_flag=50, mined
   family, one look.
2. **The control did NOT read null**: wind x bottom-total leans POSITIVE
   (+0.1366, P+ 0.828) — the same direction as wind x top-total
   (+0.0772), and larger. The wind axis does not discriminate between
   total terciles in the predicted direction; whatever wind signal exists
   here favors the favorite at BOTH total levels, which is closer to the
   parent battery's unsigned high-wind lean than to the priced-in-low-
   total mechanism. Mechanism discrimination fails on the wind axis.
3. Cold x top-total is indistinguishable from zero (-0.0071, P+ 0.46) —
   direction as predicted, magnitude nil; probability_positive 0.54 that
   the effect is negative.
4. Cells 1 and 4 have point estimates on the WRONG side of their
   predeclared negative direction, but only cell 4's week-blocked interval
   sits entirely above zero, so no `wrong_sign_resolved` record is
   admissible for cells 1-3 under any reading.

All four cells record `unresolved_below_power` (exact command lines in the
run handoff); none of this spends an NFL rotation window.
