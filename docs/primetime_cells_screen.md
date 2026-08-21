# Primetime situational cells screen — predeclaration

Written **before** `scripts/primetime_cells_screen.py` scored any cover-rate
outcome. Only population counts (n_flag sizes, threshold feasibility,
missing-data checks) were examined before this document was frozen — no
cover rate, gap, interval, or probability_positive for any cell was computed
or looked at beforehand. Method, population, blocking, seed, and predicted
directions are locked below exactly as in `docs/body_clock_screen.md`.

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

## Overlap check (why these cells are new ground)

Checked **before** designing cells, per the session mandate:

- `bias_battery_primetime_favorite` exists (**read** this session,
  `scripts/nfl_bias_battery_screen.py:376-385` and
  `registry/weak_signals.json:984`): Thu/Mon or Sun ≥20:00 ET **FAVORITE**,
  any venue, sign −1. This screen builds strictly distinct cells: away
  UNDERDOG (not favorite), response spot (prior-outcome conditioned, not
  role-conditioned), divisional-favorite interaction, and post-MNF calendar
  follow-on. The one deliberate partial overlap is disclosed under cell 4.
- `bias_battery_short_week` / `bias_battery_short_week_opener`
  (`registry/weak_signals.json:1059/1084`, **read** this session): own rest
  ≤ 5 days. Cell 5 (Sunday after Monday night) is thematically adjacent but
  **disjoint by construction**: Monday→Sunday is 6 days rest. **Measured**
  pre-freeze on the snapshot below: 0 of 588 post-MNF-Sunday team-games have
  `own_rest <= 5`. No window spend, no shared flag.
- `travel_rest_thursday_pure`, `weather_battery_thursday_outdoor_cold`,
  `attention_followup_both_cold_non_primetime` (**read** this session):
  none conditions on the primetime-window x role/prior-result interactions
  built here. No other `pt_*` or primetime-cell name exists in
  `registry/weak_signals.json` (**measured**, grep this session).

## Data source and leakage posture

- Newest snapshot `data/raw/20260817T235649Z/schedules.parquet` (**read**;
  same snapshot the body-clock/travel/weather screens use).
- Primetime identification uses schedule facts only: `weekday` +
  `gametime` (ET, nflverse convention). Pregame-safe: both are known before
  kickoff. Cells 2/3/5 use the team's OWN strictly-prior game this season
  (`gameday`-sorted shift(1) within (team, season)): its weekday and result
  are fully resolved before the current game's kickoff — point-in-time safe,
  no leakage caveat. Cell 1/4 use `spread_line` (the pregame line) and
  `div_game` (schedule fact).
- Population: REG 2009-2025, pushes/missing-spread dropped via
  `nfl_ats.features.add_ats_outcomes`, one row per team-game (long table,
  `scripts/nfl_bias_battery_screen.py::build_long_table` convention),
  canonicalized with `TEAM_ABBREVIATION_ALIASES`.
- **International windows separately flagged**: all 69 neutral-site games
  (**measured** pre-freeze) are EXCLUDED from every cell flag above (the
  primetime mask requires a true home/road game); 22 neutral-site team-game
  rows would otherwise hit the primetime mask (Monday-night London, late-
  Sunday Mexico City) and are reported as a named diagnostic count in the
  artifact rather than silently mixed into either arm.

## Derived quantities (frozen)

- `is_primetime`: `weekday ∈ {Thursday, Monday}` OR (`weekday == Sunday`
  AND kick ≥ 20:00 ET). Saturday excluded (stated limitation, identical to
  `bias_battery_primetime_favorite`). Friday/Wednesday/etc. excluded.
- `prior_margin` / `prior_weekday`: own previous REG game this season;
  rows with no prior game are excluded from BOTH arms of cells 2/3/5 via an
  eligibility mask (same convention as `backup_qb_start` in the bias
  battery), never defaulted into the complement.
- Tie as prior game: counts as neither off-loss nor off-win (eligible,
  unflagged).

## Method (reused verbatim from `scripts/body_clock_screen.py` /
`scripts/nfl_bias_battery_screen.py`)

- Value = `team_covered`; subset-vs-complement full-slate-scaled effect:
  `(subset_cover − complement_cover) × 100 × fraction_of_slate`.
- Week-blocked joint bootstrap primary (block = `season*100+week`),
  season-blocked secondary (block = `season`), same
  `block_bootstrap_two_group` algorithm.
- **20,000 samples, seed 20260821** (mandated).
- `probability_positive` = fraction of bootstrap draws favouring the
  predeclared direction (sign applied before the >0 test).
- Era splits (2009-2017 vs 2018-2025) are scored for EVERY cell in the
  artifact; item (e) below designates the strongest full-period cell's two
  splits for recording.

## The 7 recorded cells (directions frozen before scoring)

Population diagnostics measured pre-freeze (counts only): 8,634 team-game
rows; primetime rows 1,694; C1 n=540; C2 n=697; C3 n=866; C4 n=357; C5
n=588 (0 overlapping short-week); 125 primetime rows lack a prior game
(eligibility-masked out of C2/C3/C5 arms).

| # | name | flag | predicted direction |
|---|---|---|---|
| 1 | `pt_away_underdog` | primetime AND away AND `team_spread < 0` | **+1** — road dog covers: spotlight premium inflates the favorite's price |
| 2 | `pt_off_loss` | primetime AND prior game lost | **+1** — response spot: embarrassed-in-the-spotlight bounce-back covers |
| 3 | `pt_off_win` | primetime AND prior game won | **−1** — letdown: spotlights next week's opponent after the high |
| 4 | `pt_divisional_favorite` | primetime AND `div_game == 1` AND favorite | **−1** — hostility-spot hype inflates the divisional favorite's price. DISCLOSED: this is `bias_battery_primetime_favorite` ∩ divisional — a subset, correlated, never pool as independent |
| 5 | `pt_post_mnf_sunday` | current game Sunday AND own prior game (this season) was a Monday game | **−1** — short-week hangover (6-day turnaround, compressed prep). Disjoint from `bias_battery_short_week` (rest ≤ 5): measured 0-row overlap |
| 6 | `<strongest>_era_2009_2017` | strongest full-period cell ∩ seasons 2009-2017 | inherits parent sign |
| 7 | `<strongest>_era_2018_2025` | strongest full-period cell ∩ seasons 2018-2025 | inherits parent sign |

"Strongest" := largest |full-slate effect| in the full period among cells
1-5, designated mechanically after scoring; its direction is NOT re-chosen —
the parent's frozen sign carries to both splits.

## Recording commitment

Every cell above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, `season_start=2009`, `season_end=2025`,
regardless of interval shape. This session writes no registry JSON itself
(measure-only, no rotation-registry window spent, no NFL window spend); it
stamps a run log under `registry/experiments/primetime-cells-screen/` via
`write_experiment_artifact`. Exact record command lines are returned by the
session with numbers passed through unmodified from the artifact JSON. The
only admissible alternative classification would be a RESOLVED wrong sign
(whole interval on the wrong side of the predicted direction) — none is
claimed here regardless of what the numbers show.

---

## Results (measured 2026-08-21, `artifacts/primetime_cells_screen/20260821T184312Z/results.json`)

Population: REG 2009-2025, 4,431 games, 114 pushes/missing dropped, **8,634
team-game rows scored**; 22 international-window team-game rows excluded
from every primetime flag (**measured**, disclosed diagnostic); post-MNF ×
short-week overlap **0 rows** (**measured**). Week-blocked primary (294
blocks full-slate / 277 prior-eligible), season-blocked secondary (17),
20,000 samples, seed 20260821. Effects are SIGNED toward the predeclared
direction (positive = favours the prediction); all numbers below are
**measured** from the artifact JSON, passed through unmodified.

| cell | n_flag | effect pts | week-blocked 95% | P+ | season-blocked P+ |
|---|---|---|---|---|---|
| `pt_away_underdog` | 540 | −0.0618 | [−0.3281, +0.2108] | 0.3244 | 0.3478 |
| `pt_off_loss` | 697 | +0.0068 | [−0.2536, +0.2701] | 0.5159 | 0.5218 |
| `pt_off_win` | 866 | +0.0138 | [−0.2537, +0.2798] | 0.5362 | 0.5485 |
| `pt_divisional_favorite` | 357 | +0.0181 | [−0.2061, +0.2378] | 0.5612 | 0.5485 |
| `pt_post_mnf_sunday` | 586 | +0.0666 | [−0.2272, +0.3620] | 0.6705 | 0.6687 |

Item (e) — era splits of the mechanically-designated strongest cell,
`pt_post_mnf_sunday` (parent sign −1 carried, not re-chosen):

| era cell | n_flag | effect pts | week-blocked 95% | P+ |
|---|---|---|---|---|
| `pt_post_mnf_sunday_era_2009_2017` | 296 | +0.2546 | [−0.1592, +0.6733] | 0.8841 |
| `pt_post_mnf_sunday_era_2018_2025` | 290 | −0.1367 | [−0.5532, +0.2886] | 0.2566 |

Reading (**inferred**, from the measured tables above — no closure claimed):

- The strongest cell (`pt_post_mnf_sunday`) leans the predeclared hangover
  direction at P+ 0.6705 week-blocked / 0.6687 season-blocked — a real lean,
  far below any claim-worthy confidence.
- Era stability is POOR for that lead: fully driven by 2009-2017 (+0.2546,
  P+ 0.8841) with a sign flip to −0.1367 (P+ 0.2566) in 2018-2025. Reported
  plainly: whatever is there is not stable across eras, which caps how much
  the pooled lean can be trusted — an unresolved caveat, not a closure (no
  refuted-mechanism ground exists: the trait is a per-game situational
  condition with nothing to split-half, and no positive control was run).
- The response-spot pair behaves as near-mirrors (off-loss +0.0068 /
  off-win +0.0138, both P+ ~0.52-0.54), each individually tiny; their era
  splits also flip sign between eras. Any future look should treat the PAIR
  as one contrast, never pool them as independent signals.
- `pt_away_underdog` leans AGAINST its spotlight-premium prediction
  (P+ 0.3244), most strongly anti-predicted in 2009-2017 (P+ 0.0644) — a
  category-3 anti-lean, recorded as such, not a resolved wrong sign (the
  interval does not sit entirely below zero).
- Every interval here crosses zero, which at this evaluator's ~2-point
  resolution is the expected shape for real-but-small signals and closes
  nothing. All seven cells are category 3: recorded `unresolved_below_power`
  via `nfl-ats weak-signals record` (exact command lines returned by the
  session; numbers passed through unmodified from the artifact JSON above).
- Correlation disclosure for future pooling: cells 2/3 share an eligible
  population and are logical complements; cell 4 is a subset of
  `bias_battery_primetime_favorite`; cell 5 is thematically adjacent to
  `bias_battery_short_week` (measured 0-row overlap here). Never
  sign-test-pool these as independent.
