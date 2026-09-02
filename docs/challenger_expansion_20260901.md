# Challenger expansion, 2026-09-01 (program ORCH-C)

Every EV-positive, live-computable registry cell that is not already tracked
should be a zero-cost 2026 prospective challenger. A prospective challenger
spends no rotation-registry window, changes no published pick, and generates
paired paper evidence for free. Declining one is an active bet that the cell is
worth exactly zero.

**A prospective registration is not a promotion.** Promotion is a separate
decision, graded at the opener, against the played chain. Nothing in this
document changes `artifacts/active_ats_model.json`, `CURRENT_PREDICTIONS.md`,
or any existing challenger.

---

## §1 Survey and ranking (written before any overlay was built)

### 1.1 Source and screen

**Measured this session** (`registry/weak_signals.json`, read 2026-09-01):
624 signal entries; **168** are `league=nfl`, `effect_units=accuracy_points`,
`probability_positive >= 0.70`. Those 168 were screened against four gates.

**(a) Not already tracked.** Matched by *construct*, not name, against the 27
entries in `artifacts/prospective/challengers.json` (read 2026-09-01; 21
`ACTIVE_PROSPECTIVE`, 4 `SUPERSEDED_BY_PROMOTION`, 1
`DEACTIVATED_STRUCTURAL_NO_OP`, 1 `CLOSED_BEFORE_ACTIVATION`).

**(b) Live-computable at the lock.** The pick deadline is
`min(kickoff, Sunday 16:00 ET)`; the Tuesday build is the opening statement.
Locally available at the Tuesday build: the nflverse schedule snapshot
(`data/raw/<ts>/schedules.parquet`), the PBP/feature tables, the odds archive,
and — from `SCHEDULE` in `scripts/capture_scheduler.py` (read 2026-09-01) —
odds captures (Tue/Thu/Sat/Sun/Mon), Action Network public betting (Sat/Sun),
NFL.com injury pages (Wed/Thu/Fri/Sat), the three `refresh-picks` passes, the
Tuesday player-arrests snapshot, and the Sunday backup. **Not** on a live
feed: CDC FluView, Wikipedia pageviews, Arctic Shift Reddit, AQI, FFC ADP,
officiating-crew assignments.

**(c) Well-defined overlay semantics.** A flag whose direction was
**predeclared before scoring**, so the overlay's direction is not chosen from
the same data that produced the effect.

**(d) Not a line-movement channel.** Everything in the
`observed_movement_*` / `opener_error_mining_movement_*` /
`movement_attribution_*` families is already carried by
`movement_rule_composed_v1` and `injury_signal_refresh_tilt`.

### 1.2 EV ranking

EV is scored as `(probability_positive - 0.5) x |effect| x reliability`, in
full-slate accuracy points. Reliability is the registry's own split-half /
year-over-year figure where the cell has one. For a **structural** flag — a
schedule gap, a standings position, a travel distance — there is no trait to
persist: the flag is observed exactly, so no reliability discount applies and
the factor is 1. (That last step is **inferred**, not measured; it is stated
so it can be argued with.)

| # | registry cell | P+ | effect (pts) | 95% interval | rel | n | seasons | EV | built? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `bye_overval_fade_full_slate_post2011` | 0.8375 | +0.5508 | [-0.567, +1.643] | struct. | 2171 | 2012-2025 | **0.186** | **W1** |
| 2 | `motivation_ladder_tank_zone_wk14_18` | 0.9334 | +0.3049 | [-0.079, +0.697] | struct. | 2768 | 2009-2025 | **0.132** | **W2** |
| 3 | `redzone_reversion_c3_third_down_over_fade` | 0.8719 | +0.3665 | [-0.259, +0.999] | 0.407 | 8634 | 2009-2025 | **0.056** | **W3** |
| 4 | `travel_rest_return_trip_hangover` | 0.7528 | +0.2120 | [-0.404, +0.831] | struct. | 4317 | 2009-2025 | 0.054 | **dropped — see 1.3a** |
| 5 | `team_style_short_game_identity` | 0.8704 | +0.3504 | [-0.256, +0.951] | 0.408 | 8634 | 2009-2025 | 0.053 | no — see 1.3 |
| 6 | `travel_rest_thursday_pure` | 0.7592 | +0.1349 | [-0.234, +0.502] | struct. | 4317 | 2009-2025 | 0.035 | no — same battery as #4 |
| 7 | `qb_age_second_year_jump` | 0.8548 | +0.2354 | [-0.195, +0.667] | 0.383 | 8634 | 2009-2025 | 0.032 | no — see 1.3 |
| 8 | `special_teams_return_top_quartile` | 0.9547 | +0.4986 | [-0.074, +1.080] | 0.109 | 8634 | 2009-2025 | **0.025** | **W5** |
| 9 | `attention_battery_both_cold` | 0.8568 | +0.5221 | [-0.441, +1.504] | 0.132 | 2246 | 2016-2025 | 0.025 | no — see 1.3 |
| 10 | `team_style_pace_mismatch_dog_cover` | 0.7111 | +0.2292 | [-0.559, +1.040] | 0.489 | 4313 | 2009-2025 | **0.024** | **W6** |
| 11 | `close_game_luck_turnover_under_rebound` | 0.9200 | +0.4092 | [-0.153, +0.969] | 0.132 | 8634 | 2009-2025 | **0.023** | **W4' (replaces W4)** |
| 12 | `illness_home_ge2` | 0.8897 | +0.2973 | [-0.176, +0.784] | **0.702** | 3536 | 2010-2024 | 0.081 | no — see 1.3, blocked |

Six built (W1-W6). Every interval above crosses zero; per AGENTS.md that is
the EXPECTED shape at this evaluator's ~2-point resolution and is never
grounds to decline. `probability_positive` is the decision figure, and every
one of the twelve is above 0.5.

**Rank 2 is a five-week signal.** `motivation_ladder_tank_zone_wk14_18` fires
only in weeks 14-18, so its 2026 ledger stays empty until December. It is
built anyway: EV is EV, and a challenger that costs nothing to carry does not
have to pay this month.

### 1.3a The registry's `description` field does not tell you which way a cell went

Found while writing the worker briefs, and it is the most reusable thing in
this document. **A registry entry's `description` states the cell's
PREDECLARED direction. Its `effect` and `probability_positive` are on
whatever scale that screen's own value column and `sign_dir` used.** Those two
are not the same statement, and three of the six cells assigned here had a
mismatch between them in one direction or the other. The only reliable read is
the screen artifact: `sign_dir`, `value_column`, and the raw
`subset_cover` / `complement_cover` pair.

Every cell in this batch was re-checked against its own artifact. Measured
this session:

| cell | artifact | `sign_dir` / value column | subset vs complement | verdict |
|---|---|---|---|---|
| `bye_overval_fade_full_slate_post2011` | `artifacts/bye_overvaluation_screen/post_fix_seed20260822/results.json` | `value_column: "fade_side_cover"`, "Predicted direction POSITIVE" | 0.5261 vs 0.5021 | **clean** — predeclared fade is the measured direction |
| `motivation_ladder_tank_zone_wk14_18` | `artifacts/motivation_ladder_screen/20260821T182643Z/results.json` | `sign_dir: -1` | 0.4444 vs 0.5030 | **clean, but the registry `description` is wrong** — it says the cell "leans OPPOSITE tank-fade prediction (tank teams over-cover)". The artifact says tank teams cover 44.4% against a 50.3% field: the predeclared FADE is exactly what was measured. |
| `redzone_reversion_c3_third_down_over_fade` | `artifacts/redzone_reversion_screen/20260821T181025Z/results.json` | `sign_dir: -1` | 0.4885 vs 0.5037 | **clean** |
| `special_teams_return_top_quartile` | `artifacts/special_teams_battery/20260819T232856Z/results.json` | `sign_dir: 1` | 0.5164 vs 0.4950 | **clean** |
| `team_style_pace_mismatch_dog_cover` | `artifacts/team_style_screen/20260819T210011Z/results.json` | `sign_dir: 1`, value column `dog_cover` | 0.5187 vs 0.5090 | **clean** |
| `close_game_luck_turnover_under_rebound` | `artifacts/close_game_luck_screen/20260821T182234Z/results.json` | `sign_dir: 1` | 0.5133 vs 0.4959 | **clean** |
| `travel_rest_return_trip_hangover` | `artifacts/travel_rest_battery/20260819T232521Z/results.json` | **no `sign_dir` key at all** — `add()` (`scripts/nfl_travel_rest_battery_screen.py:259-264`, read) stores no sign, so every cell in this battery reports the RAW `home_cover` gap | **0.5026 vs 0.4869** | **DROPPED — see below** |

**Why `travel_rest_return_trip_hangover` was dropped from the build.** Its
registry description predicts a "negative `home_cover` edge" — fade the home
team coming off its own long road trip. The artifact measures the opposite:
home teams cover **50.26%** in the flagged population against **48.69%** in
the complement, a raw gap of **+1.57 points**, and the recorded
`effect` +0.2120 is exactly that gap times the 0.1350 slate fraction, with no
sign flip anywhere in the pipeline. So `probability_positive` 0.7528 is the
probability that the HOME side does BETTER here, not worse.

That leaves no buildable rule today. Fading the home team follows the
predeclared direction but points against a P+ 0.75 measurement; backing the
home team follows the measurement but takes its direction from the same screen
that produced the effect, which is the mined-sign defect that already excluded
`team_style_short_game_identity` at rank 5.

**Nothing here closes the cell.** It stays `unresolved_below_power`, and
"predeclared direction refuted, opposite lean at P+ 0.75 on 583 flagged games"
is a more interesting result than the entry currently reads as. The next step
is a predeclared re-screen of the reversed mechanism, not a deletion. Its EV
slot went to `close_game_luck_turnover_under_rebound` (rank 11, verified
clean above), so the batch is still six.

**Two registry-hygiene follow-ups this program did not take** (editing
`registry/weak_signals.json` by hand is prohibited, and `--replace` on a
recorded cell is a judgement call above this program's authority):

1. `motivation_ladder_tank_zone_wk14_18`'s `description` states the measured
   lean backwards and should be corrected.
2. `travel_rest_return_trip_hangover`'s `description` should say that the
   measured direction contradicts the predeclared one.

### 1.3 Excluded, and exactly why

Nothing below is *closed*. These are build-eligibility calls, not verdicts;
every cell keeps its `unresolved_below_power` classification.

**Construct-duplicates of a live challenger** — the evidence is already being
accrued, a second overlay would double-count it:

- `surface_familiarity_r1_turf_venue_visitor_split` (P+ 0.9332, +1.458),
  `surface_familiarity_r3_era_2018_2025` (P+ 0.9578, +2.387),
  `weather_battery_surface_switch_grass_to_turf` (P+ 0.995) — all three are
  cited *by name* as the evidence for `surface_switch_tilt_overlay`
  (read: `artifacts/prospective/challengers.json`, that entry's
  `evidence.registry_source`).
- `wxtot_precip60_top_total` (P+ 0.9978, interval entirely above zero) —
  precip >= 60% AND top-tercile total. `forecast_weather_kn_precip_high_total_tilt`
  already fires on outdoor + live precip >= 60% + `total_line >= 47` (read:
  that entry's `status_reason`). Same construct, live already.
- `weather_battery_warm_team_cold_late`, `forecast_weather_*_temp_gap_cold_visitor`,
  `weather_followup_temp_gap_cold_visitor` — carried by
  `forecast_weather_kn_warm_team_cold_late_tilt` and `forecast_cold_visitor_tilt`.
- `era_trend_hc_year_one_fade`, `hc_year_one_fade` — the coach fade is a
  PROMOTED production component plus a live challenger.
- `pbp08_protection_mismatch`, `interim_hc_first_game`,
  `nflcom_refresh_out2_starters_on_chain`, `overlay_subset_production_plus_*`,
  `mod08_smooth_cdf_mapping`, `era_weighting_nfl_half_life_8`,
  `injury_value_lost_*`, `bias_battery_division_revenge_game*`,
  `divisional_rematch_revenge_*` — all already tracked.

**Line-movement channel (gate d):** `observed_movement_threshold_1_0` /
`_0_5` / `_sunday_am_realism`, `observed_movement_oracle_*`,
`movement_attribution_pop_*` (including the +17.07 injury cell),
`opener_error_mining_movement_agreement_*`. Carried by
`movement_rule_composed_v1` and `injury_signal_refresh_tilt`.

**Not computable before kickoff:**

- `qb_age_second_year_jump` (P+ 0.8548) and every other cell keyed to the
  game's own starting quarterback. `home_qb_name`/`away_qb_name` are
  **272/272 null for all of 2026 REG** in the schedule snapshot — the
  measurement that already retired `backup_qb_fade_overlay` as
  `DEACTIVATED_STRUCTURAL_NO_OP` (read: `docs/prospective_evidence.md`,
  "Tuesday-visibility audit"). No pregame hour fixes this; it needs a
  depth-chart feed the repo does not have.
- `odds_microstructure_H3_3_0a/0b/H3_3_1` (P+ 0.9999/0.9984/0.9326, effects
  +5.1/+5.7/+4.5) — these are **oracle sanity checks**: they use the line the
  market eventually moved to. They bound what movement capture is worth; they
  are not playable rules.

**Actual-weather screens, explicitly labelled "NOT pregame-available":**
`weather_followup_wind_gap_visitor` (P+ 0.8541),
`weather_battery_dome_team_outdoors_cold` (P+ 0.8249),
`weather_battery_extreme_cold`, `weather_battery_high_wind_outdoor`,
`weather_battery_thursday_outdoor_cold`. Their own registry notes call them
"upper bound for a forecast-time feature". A forecast-transposed version is a
new measurement, not a re-use of these constants — the correct next step is a
forecast re-screen in the `forecast_weather_kn_*` family, then a challenger.

**No predeclared direction — the overlay's sign would be mined:**

- `team_style_short_game_identity` (rank 5, P+ 0.8704, rel 0.408) — the
  registry description says **"UNSIGNED"**. Its effect is recorded as a
  magnitude with the sign fixed at +1 by convention, so the tilt direction
  would have to be read off the same screen that produced the effect.
- `environmental_battery_aqi_high_outdoor` (P+ 0.9264) — "NO predeclared
  direction ... sign fixed at +1 ... exploratory", and `n_flag=42` in 4,259
  games (opener re-screen `n_flag=9`), roughly 2-3 games a season.

Both stay recorded and unresolved. Either becomes buildable the moment a
predeclared-direction re-screen exists.

**No live data path** (each would need a new weekly fetch; the two
`forecast_weather_kn_*` challengers prove an outbound live call at publish
time is an accepted pattern, so these are cost calls, not impossibilities):

- `attention_battery_both_cold` (rank 9) — Wikipedia pageviews, window ends
  Tuesday of game week, point-in-time safe by construction. Best candidate of
  this group; the API is public and free.
- `reddit_home_comment_ratio_elevated` (P+ 0.8848, rel 0.992) — Arctic Shift
  bulk dumps, no live weekly feed.
- `fluview_home_market_elevated_on_production` (P+ 0.792, +0.969, rel 0.981,
  measured **on production**, the composition-correct read) and
  `fluview_away_market_elevated` (P+ 0.8826) — CDC FluView. Also: another
  program is editing the FluView surface today; deliberately left alone.
- `ffc_adp_cellA_highadp_underdog_back_ppr_w14` (P+ 0.9421, **+4.18 pts**,
  weeks 1-4) — the largest effect of any excluded cell and it fires in Week 1,
  but needs a 2026 preseason ADP ingest (`scripts/ingest_ffc_adp.py`) that has
  not been run for this season.
- `penalty_crew_high_flag_heavy_underdog_opener` (P+ 0.9204, rel 0.370,
  **opener-graded**, the primary grade) — needs the week's officiating-crew
  assignment pregame; no live capture exists.

**Refuted-direction reliability:** `vi_disp_homecover_top_vs_bottom_tercile`
(P+ 0.9671, +5.36) and `vi_dispersion_bottom_tercile_underdog` (P+ 0.9620)
both carry **reliability -0.0421**. A trait with no split-half reliability is
the one thing AGENTS.md does treat as a refuted mechanism — no sample size
rescues it. Not built; not re-closed here either, since the closure would have
to go through `weak-signals record`, and that is not this program's call.

**Blocked by a file this program may not edit:** `illness_home_ge2`
(P+ 0.8897, +0.2973, **reliability 0.702 — the highest of any unbuilt cell**).
Its as-of cutoff is already the project's own
`nfl_ats.pick_refresh.pick_deadline(kickoff, sunday_lock)`, and the NFL.com
injury pages it needs are captured Wed/Thu/Fri/Sat. It is a *refresh-time*
overlay, and every refresh-time recorder is wired inside
`src/nfl_ats/pick_refresh.py`, which is off-limits to this program. **This is
the single strongest recommendation for the next batch.**

**Model-level, not pick-level** (a feature/regressor change, not a
post-prediction transform — a different and much larger build):
the whole `graph_input_screen_*` family (10 cells, P+ 0.702-0.849, several
with reliability > 0.97) and `graph_team_stat_*` (P+ 0.729-0.987). Also
`opener_error_mining_slate_primetime` (P+ 0.874, +3.15) and its siblings:
these are heterogeneity reads on where the production rule already wins, which
is Best-Pick-eligibility territory, not a forced-pick transform.

### 1.4 Overlay semantics assigned (frozen before any code was written)

Every sibling overlay in this repo is a **parameter-free, post-prediction
pick flip**, not a probability nudge. That pattern is kept, deliberately:

- The composition code and the challenger recorders expect a side, not a
  magnitude.
- Inventing a tilt magnitude from the cell's own gap would be a constant
  derived from the same data that measured the effect. Under
  "underived constants are defects", the honest form is a rule with **no**
  fitted parameter, whose evidence is the cell's measured gap.
- The only numeric parameters any of these rules carries are the **screen's
  own predeclared thresholds** (>= 12-day gap; >= 1500 miles; rest <= 8;
  bottom-two record; weeks 14-18; top/bottom quartile of a named trait). Each
  is transcribed from the screen script that measured the cell, cited by file
  and line, never re-derived.

Where a flag can attach to either team, the rule fires only in the **clean
case** — flagged on exactly one side — mirroring `coach_fade_overlay` and
`backup_qb_fade_overlay`, which both exclude both-flagged games.

| worker | new challenger | rule (frozen) | direction is predeclared? |
|---|---|---|---|
| W1 | `bye_edge_fade_overlay` | Exactly one team off a strict bye (>= 12-day gap to its own previous game this season); if the model's forced pick IS that team, flip off it. REG only, 2012+ era read. | yes — "Market overprices the bye-holding side" |
| W2 | `tank_zone_fade_tilt_overlay` | Weeks 14-18; a team in the league's bottom two by record as of that week; if the pick IS that team, flip off it. | yes — predeclared NEGATIVE on `team_covered`; artifact confirms (`sign_dir: -1`, subset cover 0.4444 vs 0.5030) |
| W3 | `third_down_reversion_fade_overlay` | Prior-season centered 3rd-down conversion rate in the top quartile; if the pick IS that team, flip off it. Clean case only. | yes — predeclared NEGATIVE (fade) |
| W4' | `turnover_luck_rebound_tilt_overlay` | Prior-season centered turnover differential in the BOTTOM quartile; if the pick is NOT that team, flip onto it. Clean case only. | yes — `sign_dir: 1`, predeclared POSITIVE on `team_covered` (rebound) |
| W5 | `special_teams_return_tilt_overlay` | Prior-season `return_composite` in the top quartile; if the pick is NOT that team, flip onto it. Clean case only. | yes — predeclared POSITIVE on `team_covered` |
| W6 | `pace_mismatch_dog_tilt_overlay` | Top-quartile abs(home - away) prior-season centered `seconds_per_play`; if the pick is the favourite by `spread_line`, flip to the underdog. | yes — predeclared POSITIVE on `dog_cover` |

### 1.5 A wiring constraint this program cannot resolve itself

`tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`
(read, `tests/test_cli.py:657-673`) asserts that
`cli.PUBLISH_CHALLENGER_RESULT_KEYS` **exactly equals** the set of
`ACTIVE_PROSPECTIVE` challengers whose `weekly_recording_command` contains
`nfl-ats publish-predictions --record-decisions`. `src/nfl_ats/cli.py` is
off-limits to this program, so registering these six under that command would
turn the whole test suite red for every program working in this tree.

**What was done instead.** Each challenger's `weekly_recording_command` names
its own standalone recorder script (`scripts/record_<name>_challenger.py`),
which calls the identical `record_*_challenger_decisions` function the CLI
would call. The test stays green, the recorder is real and runnable today, and
the precedent already exists (`model_only_refresh_incumbent` and
`injury_signal_refresh_tilt` are both `ACTIVE_PROSPECTIVE` with a
non-publish recording command).

**What still has to happen before Week 1 locks on 2026-09-08.** Someone who
owns `cli.py` applies the patch in §3 — six imports, six map entries, six
fail-open `try/except` blocks — and flips the six `weekly_recording_command`
strings. Until then these six do not record automatically on the Tuesday run,
and each week that passes without it is a week of prospective evidence that
cannot be recovered.

---

## Appendix A — the `cli.py` wiring patch (ready to paste, NOT applied here)

`src/nfl_ats/cli.py` is off-limits to this program, so the six challengers
below register with a standalone recorder as their `weekly_recording_command`
and `tests/test_cli.py` stays green. To move them onto the Tuesday
`publish-predictions --record-decisions` path — which is what makes them record
automatically every week — apply all three edits together, then change each
challenger's `weekly_recording_command` in
`artifacts/prospective/challengers.json` to name the publish path. Half the
change breaks the test suite; do not land it partially.

**A1. Imports** (alphabetical position, matching the file's existing style):

```python
from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions
from nfl_ats.pace_mismatch_dog_tilt_overlay import (
    record_pace_mismatch_dog_tilt_challenger_decisions,
)
from nfl_ats.special_teams_return_tilt_overlay import (
    record_special_teams_return_tilt_challenger_decisions,
)
from nfl_ats.tank_zone_fade_tilt_overlay import (
    record_tank_zone_fade_tilt_challenger_decisions,
)
from nfl_ats.third_down_reversion_fade_overlay import (
    record_third_down_reversion_fade_challenger_decisions,
)
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    record_turnover_luck_rebound_tilt_challenger_decisions,
)
```

**A2. `PUBLISH_CHALLENGER_RESULT_KEYS`** — six new rows:

```python
    "bye_edge_fade_overlay": "bye_edge_fade_challenger_ledger",
    "tank_zone_fade_tilt_overlay": "tank_zone_fade_tilt_challenger_ledger",
    "third_down_reversion_fade_overlay": ("third_down_reversion_fade_challenger_ledger"),
    "turnover_luck_rebound_tilt_overlay": ("turnover_luck_rebound_tilt_challenger_ledger"),
    "special_teams_return_tilt_overlay": ("special_teams_return_tilt_challenger_ledger"),
    "pace_mismatch_dog_tilt_overlay": "pace_mismatch_dog_tilt_challenger_ledger",
```

**A3. Six fail-open recorder blocks** in `publish-predictions`, beside the
existing ones. Each follows the established shape exactly — a failure here is
reported but must never un-publish the card:

```python
        try:
            result["<key>"] = <record_fn>(_artifacts_root(), _data_root())
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["<key>"] = {"recorded": 0, "error": str(error)}
```

**Why this is time-critical.** Week 1 locks 2026-09-08. Every Tuesday that
passes without this wiring is a week of prospective evidence that cannot be
recovered — the recorders refuse any game at or past kickoff, by design
(`docs/prospective_evidence.md`, "The anti-backdating guarantee"). Until it
lands, each challenger records only when someone runs its
`scripts/record_*_challenger.py` before the lock.

---

## §2 What was built, and what the stacked back-test found

Six overlays, six test files, six back-test scripts, six standalone recorders,
six per-overlay docs, six new registry cells, six new `ACTIVE_PROSPECTIVE`
challengers. `artifacts/prospective/challengers.json` went from 27 entries to
33 by this program's writes (34 live, including one another program added);
**no existing entry was modified** (measured: `git show HEAD:...` diff, zero
changed keys).

Every overlay is a **parameter-free post-prediction pick flip** on the active
model's own card, dual-tracked in the prospective challenger ledger. Nothing is
wired into `publishing.py` or the production pick path, and no ledger row was
written — the first real write is the 2026-09-08 lock.

### 2.1 The stacked-on-production read

All six back-tests use the same frozen baseline,
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet` — 1,537 REG
games 2020-2025 graded at the **Tuesday opener**, 1,503 scored, 107 week
blocks, 6 season blocks — and stack the overlay on the **played** four-member
chain (coach fade, division revenge, player arrests, spread-gap fade), not on a
bare baseline. **This is a mined-seasons read on a window many families have
already used. It is context, not a gate**, and it did not decide any
registration.

| challenger | delta vs production | week-blocked 95% | P+ | season-blocked 95% | P+ | flips (net-new) | Week 1 |
|---|---|---|---|---|---|---|---|
| `bye_edge_fade_overlay` | **+0.5988** | [-0.467, +1.688] | **0.8474** | [+0.127, +1.418] | **0.9980** | 68 | 0 |
| `special_teams_return_tilt_overlay` | **+0.2661** | [-1.99, +2.57] | 0.5773 | [-1.18, +1.82] | 0.5877 | 283 | 1 |
| `tank_zone_fade_tilt_overlay` | 0.0000 | [-0.398, +0.454] | 0.4277 | [-0.522, +0.533] | 0.4011 | 16 (11) | 0 |
| `pace_mismatch_dog_tilt_overlay` | -0.4657 | [-1.952, +0.999] | 0.2498 | [-1.356, +0.627] | 0.1680 | 180 (120) | 1 |
| `third_down_reversion_fade_overlay` | -0.8649 | [-2.978, +1.192] | 0.2000 | [-2.183, +0.773] | 0.1448 | 319 (232) | 2 |
| `turnover_luck_rebound_tilt_overlay` | -0.8650 | [-2.170, +0.462] | 0.0867 | [-1.484, -0.145] | 0.0050 | — | 2 |

**Not one of the six met the closing bar, and all six were registered.** The
bar is a RESOLVED wrong sign — the whole interval below zero on **both**
blockings. `turnover_luck_rebound_tilt_overlay` is the only one that comes
close: its season-blocked interval [-1.484, -0.145] does sit entirely below
zero, but its week-blocked interval [-2.170, +0.462] does not, so the primary
blocking leaves it unresolved. It is recorded `unresolved_below_power`, and it
is also the one to watch: **P+ 0.0867 says the stacked form is ~91% likely
worse than the incumbent**, which is a real signal about composition even
though it closes nothing about the underlying cell (+0.4092, P+ 0.92 on its
own).

**Composition is not the signal, demonstrated twice.** `third_down_reversion`
reads **+1.3972 pts (P+ 0.866) solo against the bare baseline** and
**-0.8649 (P+ 0.200) stacked on what is actually played**. Same overlay, same
archive, opposite sign — exactly the failure mode that makes paired,
stacked challenger tracking the right instrument rather than solo screens.

**Why `tank_zone` reads exactly 0.0000.** Of the 10 scored games it moves that
production does not, the model's original pick was right on 5 and wrong on 5 —
they cancel game for game (measured). Its P+ 0.4277 is also depressed by ties:
the measured bootstrap split is 42.77% positive / **12.33% exactly zero** (no
moved game resampled) / 44.90% negative. Excluding ties that is 48.8%, a coin
flip, not a negative lean.

### 2.2 New registry cells (all `unresolved_below_power`, new families)

`bye_overval_fade_stacked_on_production`,
`motivation_ladder_tank_zone_stacked_on_production`,
`redzone_reversion_c3_stacked_on_production`,
`close_game_luck_turnover_stacked_on_production`,
`special_teams_return_stacked_on_production`,
`team_style_pace_mismatch_stacked_on_production`. Each was recorded through
`nfl-ats weak-signals record` under the cross-process lock, never by hand. No
rotation-registry window was spent by any of them.

### 2.3 Week 1 2026 disagreements with the incumbent (dry run, nothing recorded)

Six flips across four challengers, on the live card
`artifacts/margin_predictions/2026-week-01-20260824T120725Z`:

- `third_down_reversion_fade_overlay`: `2026_01_SF_LA` (SF→LA), `2026_01_TB_CIN` (CIN→TB)
- `turnover_luck_rebound_tilt_overlay`: JAX→CLE, MIA→LV
- `special_teams_return_tilt_overlay`: `2026_01_DEN_KC`
- `pace_mismatch_dog_tilt_overlay`: `2026_01_NO_DET` (DET→NO)
- `bye_edge_fade_overlay`: 0 — no team can be off a bye in its own opener
- `tank_zone_fade_tilt_overlay`: 0 — weeks 14-18 only

The two zeros are **by construction, not defects**, and both are pinned by a
test so a future reader cannot mistake a structurally-inert week for a null
result.

---

## §3 What this implies for the card

**First, what it implies.** The pool submits 285 forced picks either way, so
the question is never "is this proven" but "which side of the bet is favoured".

1. **`bye_edge_fade_overlay` is the one to look at hardest.** Stacked on the
   played chain it reads **+0.5988 accuracy points, P+ 0.8474 week-blocked and
   0.9980 season-blocked, with the season-blocked interval entirely above
   zero** — on 68 flips over six seasons at the opener, the grade the pool
   actually settles on. That is not a promotion recommendation, because this is
   a mined window and a promotion needs its own predeclared look. It IS the
   strongest stacked-on-production read this program produced, and at ~85/15 it
   is the side EV points to. It costs a schedule lookup and fires from the
   first bye week.
2. **Six Week-1 disagreements are already on the board** and will start
   generating paired evidence on 2026-09-08 — at zero window cost and with no
   change to the published card.
3. **`turnover_luck_rebound` stacked is ~91% likely worse than the incumbent.**
   Carrying it as a paper challenger is still right (it costs nothing and the
   underlying cell is P+ 0.92 on its own), but nobody should be tempted to play
   it on the card.

**Now what is wrong with it.** Every one of these six numbers comes from
2020-2025, a window already mined by many families; the intervals are wide;
five of the six point estimates would not survive a bar set anywhere above
"favoured". The stacked reads correlate with each other through the shared
baseline and must not be sign-test-pooled. And the whole batch records nothing
at all until the CLI wiring in Appendix A lands.

### Wiring status addendum (2026-09-02)

The Appendix A integration is now applied: all six standalone overlays run as
part of `nfl-ats publish-predictions --record-decisions`. Each result is
reported independently and a recorder failure is fail-open for the already
written card. An ordinary publish remains a no-write rehearsal for all six.
