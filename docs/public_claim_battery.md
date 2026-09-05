# Public-handicapper claim replication battery (LEAD-57)

Written **before** `scripts/public_claim_battery_screen.py` scores anything
new, per AGENTS.md ("the family must be declared before the signs are
seen"). This single doc carries the predeclaration AND (appended after the
one run) the results table, per this task's instruction.

## Binding closing-grounds taxonomy (verbatim, restated for any subagent
that touches this file)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Verdicts
flow only through `nfl-ats weak-signals record`, never through prose.
Decide on expected value: `probability_positive` above 0.5 favours the
candidate. Report per-era magnitudes; a weaker-era reading is never absence.

## What this is

Twelve of the most commonly repeated public-handicapper "system" claims
(dome/cold, primetime dogs, bye-week angles, divisional dogs, big road
favorites, home dogs, letdown/bounce spots, Week 1 dogs, tanking teams,
short-week road teams, ATS streak regression), re-run on this project's own
2009-2025 archive with one predeclared boolean subset flag per claim, scored
**close-graded** (the archive's grading line is the closing nflverse spread,
disclosed once here per the parent battery's own convention, not
re-disclosed per cell). This is a **lead-generation screen on the full
population**, not a rotation-registry confirmation look: per
`docs/rotation_registry.md` rule 8, "CFB and non-reserved seasons stay
free... The registry governs NFL confirmation looks only" -- this screen is
neither a family's assigned confirmation window nor a play/no-play decision,
so no `nfl-ats rotation` window is drawn or spent. Grading at the close is
therefore a screen, not a decision (AGENTS.md: "Grade at the OPENER for any
play/no-play claim; close-graded reads are screens, not decisions").

**Predeclared modal expectation: decay, not replication.** Public
handicapper claims of this shape are folklore repeated across decades of
different markets; the modal outcome predeclared here, before any cell is
scored, is that most decay to a flat/mixed read against a modern,
efficiently-priced closing line. A decayed reading is not a negative --
every cell records regardless of shape, per the taxonomy above.

**First: two claims duplicate cells already recorded in this registry.**
Checked against `registry/weak_signals.json` and the existing bias/weather/
travel batteries (`scripts/nfl_bias_battery_screen.py`,
`scripts/nfl_weather_battery_screen.py`,
`scripts/nfl_travel_rest_battery_screen.py`, `scripts/nfl_forecast_weather_
screen.py`) before writing any new code, per this task's instruction not to
re-score a duplicate. Both are cited, not re-run; see "Claims 1 and 11
(duplicate citations)" below. The remaining ten are fresh cells, scored by
`scripts/public_claim_battery_screen.py`.

## Method (reused from `scripts/nfl_bias_battery_screen.py`, imported not
copy-pasted)

`scripts/public_claim_battery_screen.py` loads
`scripts/nfl_bias_battery_screen.py` as a module (`importlib`, the same
technique `scripts/attention_followup_screen.py` uses on its own parent) and
calls its `load_merged` / `build_long_table` / `add_history_features` /
`summarize_population` / `score_hypothesis` functions **by import**, to
guarantee bit-identical team-game long-table construction (`team_covered`,
`team_spread`, `team_is_favorite`, `div_game`, `weekday`, `gametime_hour`,
`own_rest`, `prior_win_pct`, `opp_prior_win_pct`, `prior_score_margin`, the
week-blocked joint block bootstrap, and the full-slate-scaling arithmetic) to
the parent battery. Only two genuinely new pregame-safe columns are added on
top (`prior_team_spread`, `ats_streak_len`; see claims 7 and 12 below) by a
small `add_claim_history_features` function in the new script -- the parent
module itself is not modified.

- Data: `data/processed/game_features.parquet` inner-joined on `game_id`
  with the newest `data/raw/*/schedules.parquet` snapshot -- identical join
  to the parent battery. REG season only, 2009-2025.
- Grading: `team_covered` (1/0, pushes dropped), the parent's team-level long
  table statistic, so home-side and away-side claims (e.g. home dogs vs. road
  favorites) use the same statistic from each flagged team's own perspective.
- Bootstrap: `block_bootstrap_two_group`, week-blocked (`season*100+week`)
  joint resampling, 20,000 samples, seed `20260905` (this battery's own
  date-seed, following the one-seed-per-battery convention every prior
  battery in this repo uses). `probability_positive` = fraction of resampled
  draws favoring the predeclared direction.
- Effect: `full_slate_effect_pts = sign * (subset_cover - complement_cover) *
  100 * fraction_of_slate` -- the same full-slate scaling as the parent
  battery, in accuracy-percentage-points (registry `accuracy_points` units,
  where e.g. `0.55` means 0.55 percentage points of full-slate accuracy
  impact, not 55%).
- **Era split reported on every fresh cell**: 2009-2017 vs. 2018-2025
  (`ERA_SPLITS`, imported from the parent, unchanged), full-period interval
  is the one recorded to the registry; per-era magnitudes are reported in
  the doc/notes regardless of shape (a weaker-era reading is never absence).
- **Leakage discipline**: every flag below is built ONLY from (a) pregame
  market/schedule facts already known before kickoff (`spread_line`,
  `div_game`, `weekday`, `gametime_hour`, `own_rest`, `week`), or (b) the
  team's own STRICTLY PRIOR games this season (`prior_win_pct`,
  `prior_score_margin`, `prior_team_spread`, `ats_streak_len`, all computed
  with `.shift(1)`/cumulative-prior arithmetic that excludes the current
  row's own outcome). No flag ever reads the current game's own
  `result`/`ats_margin`/`home_cover`/`team_covered`.
  `tests/test_public_claim_battery_screen.py::test_no_flag_depends_on_current_game_outcome`
  asserts this mechanically (each flag is recomputed with the current-game
  outcome columns shuffled; a leaking flag would change under the shuffle,
  a pregame-safe one cannot).

## Claims 1 and 11 (duplicate citations -- not re-scored)

**Claim 1 -- "Dome teams playing outdoors in cold (<40F) lose ATS" -> FADE
dome visitors outdoors cold.** This exact subset (away team's modal home
roof this season is dome/closed AND this game's roof is outdoor/open AND
temp<=40F) is already recorded twice, with the SAME predeclared direction
(both cells' descriptions read "predicted home_cover edge", i.e. fading the
dome visitor):
- `weather_battery_dome_team_outdoors_cold` (**read**,
  `registry/weak_signals.json`): actual game-time temp, REG 2009-2025, n=4,317
  games, n_blocks=294, effect **+0.1052 accuracy_points**, 95% CI
  **[-0.118, +0.3264]**, **P+ 0.8249**. Source:
  `scripts/nfl_weather_battery_screen.py`;
  `artifacts/nfl_weather_battery/20260819T155124Z/results.json`.
- `forecast_weather_dome_team_outdoors_cold` (**read**, same registry):
  Tuesday-noon forecast temp (genuinely pregame-available, narrower window
  2020-2025 since the forecast archive starts later), n=1,582, n_blocks=107,
  effect **-0.1130 accuracy_points**, 95% CI **[-0.6163, +0.3424]**,
  **P+ 0.3165**. Source: `scripts/nfl_forecast_weather_screen.py`;
  `artifacts/forecast_weather_screen/20260819T235541Z/results.json`.

Recorded as `public_claim_dome_cold_fade`, citing the actual-weather cell
(matches the claim's plain <40F description and the full 2009-2025
population) as the primary number, with the forecast-based sibling's
opposite-signed, narrower-window read disclosed in the same entry's notes --
the two together show this claim is genuinely unresolved, not a clean
positive.

**Claim 11 -- "Short-week (Thursday) road teams fail" -> FADE Thursday road
teams.** No cell in the registry is an exact Thursday x road-team
intersection, but two already-screened cells jointly bracket the mechanism
and point the same direction, so this is cited rather than re-scored (per
this task's instruction):
- `travel_rest_short_week_road` (**read**): `away_rest<=5` (the traveling
  side specifically on short rest, which is what a Thursday game does to the
  visitor), REG 2009-2025, n=4,317, n_blocks=294, effect
  **+0.0442 accuracy_points**, 95% CI **[-0.3116, +0.4041]**,
  **P+ 0.5933**. Source: `scripts/nfl_travel_rest_battery_screen.py`;
  `artifacts/travel_rest_battery/20260819T232521Z/results.json`.
- `travel_rest_thursday_pure` (**read**): any Thursday game regardless of
  side, same population, effect **+0.1349 accuracy_points**, 95% CI
  **[-0.2342, +0.5015]**, **P+ 0.7592**.

Recorded as `public_claim_thursday_road_fade`, citing
`travel_rest_short_week_road` (the closer proxy: it isolates the traveling
side, which is the actual claim) as the primary number, with
`travel_rest_thursday_pure`'s plain-calendar read disclosed in notes.

## Claims 2-10 and 12 (fresh cells, predeclared before scoring)

All ten use the parent's team-game long table (`team_covered` as the value
column, `sign` applied before scaling, week-blocked primary interval,
2009-2017/2018-2025 era split). Exact flag definitions, frozen before
`scripts/public_claim_battery_screen.py` was run:

2. **`public_claim_primetime_dog`** -- "Primetime underdogs (SNF/MNF/TNF)
   cover" -> BACK primetime dogs. Flag: `weekday in {Thursday, Monday} OR
   (weekday == Sunday AND gametime_hour >= 20)` (Saturday excluded, same
   stated limitation as the parent's own `primetime_favorite` cell) **AND**
   `team_spread < 0` (this team is the underdog; pick'ems at `team_spread ==
   0` fall to the complement, same convention as the parent's
   `team_is_favorite = team_spread > 0`). `sign = +1`. Related, not
   identical, to the already-recorded `bias_battery_primetime_favorite`
   (favorites-in-primetime fade, a different exact subset since dogs and
   favorites are disjoint populations with different complements).
3. **`public_claim_post_bye_back`** -- "Post-bye teams cover" -> BACK
   post-bye teams. Flag: `own_rest >= 12` (either side, home or road; the
   parent's own `short_week`/`extra_rest_edge` cells already use this same
   per-game `own_rest` field, sourced directly from `schedules.parquet`'s
   `home_rest`/`away_rest` columns -- a different, simpler source than
   `bye_overvaluation_screen.py`'s `build_bye_maps` helper, so that helper's
   2026-08-22 corrected-instrument note does not apply here, disclosed for
   anyone comparing the two). Eligible population: `own_rest.notna()`.
   **`sign = +1` -- the OPPOSITE of this project's own established bye
   finding.** The bye-overvaluation family already recorded, on
   finer-grained subsets of this same population: `bye_overval_home_edge_
   post2011` (home off strict bye only, effect -0.3304, i.e. FADE), `bye_
   overval_fade_full_slate_post2011` (fade whichever side holds the bye
   edge, effect +0.5508 for the fade direction), and `venue_milestone_post_
   bye_road` (road off bye only, effect -0.0010, flat). This predeclaration
   states that contradiction up front, as instructed, and lets this cell's
   own numbers speak rather than assuming the public claim must lose.
4. **`public_claim_division_dog`** -- "Divisional underdogs cover" -> BACK
   division dogs. Flag: `div_game == 1 AND team_spread < 0`. `sign = +1`.
5. **`public_claim_road_fav_big_fade`** -- "Road favorites of 7+ points fail
   to cover" -> FADE big road favorites. Flag: `is_home == False AND
   team_spread >= 7` (team_spread >= 7 already implies favorite by the
   parent's own `team_is_favorite = team_spread > 0` convention, so no
   separate favorite check is needed). `sign = -1`.
6. **`public_claim_home_dog_3plus`** -- "Home underdogs of 3+ cover" -> BACK
   home dogs 3+. Flag: `is_home == True AND team_spread <= -3`. `sign = +1`.
   Related, not identical, to the already-recorded, unthresholded `bias_
   battery_home_underdog` (any home dog, effect 0.0 flat).
7. **`public_claim_upset_letdown_fade`** -- "Teams off a straight-up upset
   win as underdog let down next week" -> FADE last week's upset winner.
   New column `prior_team_spread` (`team_spread.shift(1)` within
   `(team, season)`, strictly prior). Flag: `prior_team_spread < 0 AND
   prior_score_margin > 0` (last game, this team was the market underdog by
   any margin and won straight-up -- "upset" per the plain public meaning,
   not a magnitude threshold). No eligibility filter is declared (matching
   the parent's own treatment of `post_blowout_win_letdown`/
   `post_blowout_loss_bounce`): a missing prior game (`week` 1 of a team's
   season) makes the comparison `NaN < 0` / `NaN > 0`, which evaluates
   `False` in pandas and the row falls to the complement, not excluded.
   `sign = -1`.
8. **`public_claim_blowout_loss_bounce_21`** -- "Teams off a blowout loss
   (lost by 21+) bounce back" -> BACK post-blowout losers. Flag:
   `prior_score_margin <= -21` (same `prior_score_margin` column and the
   same no-eligibility-filter treatment as claim 7 and the parent's own
   17-point sibling). `sign = +1`. Related, not identical, to the
   already-recorded `bias_battery_post_blowout_loss_bounce` (>=17-point
   threshold, effect +0.0729), which already points the same direction at a
   looser threshold; this cell tests the public claim's own stated 21-point
   threshold.
9. **`public_claim_week1_dog`** -- "Week 1 underdogs cover (offseason
   overreaction)" -> BACK Week 1 dogs. Flag: `week == 1 AND team_spread <
   0`. `sign = +1`.
10. **`public_claim_eliminated_fade_wk17_18`** -- "Weeks 17-18, eliminated
    teams fail" -> FADE teams already eliminated from playoff contention
    against teams still alive. True mathematical elimination requires a full
    standings + tiebreaker reconstruction; `scripts/motivation_ladder_
    screen.py` already builds an approximate one (clinch/eliminate states,
    no tiebreakers) for a different lead, but wiring its states into this
    flag was judged out of scope for a single predeclared boolean cell in a
    12-claim battery, so **per this task's explicit fallback instruction,
    elimination is proxied by record**, stated plainly rather than presented
    as true elimination: eligible population is `week in {17, 18} AND
    prior_games >= 13 AND opp_prior_games >= 13` (a reliability floor -- 13
    of a team's 16-17 REG games played, so the win pct entering week 17/18
    means something); flag (within that eligible population) is
    `prior_win_pct <= 0.400 AND opp_prior_win_pct >= 0.600` -- "a team with a
    losing record (<=.400) in weeks 17-18 vs. a team >=.600", exactly the
    fallback definition this task specified. `sign = -1`.
12. **`public_claim_ats_streak_regress`** -- "Teams on a 3+ game ATS losing
    streak regress (cover next)" -> BACK teams on an ATS losing streak. New
    column `ats_streak_len`: for each `(team, season)` group sorted by
    gameday, a running count of consecutive `team_covered == 0` results
    immediately preceding this game (reset to 0 on any `team_covered == 1`,
    reset at the season boundary like every other prior-game feature in the
    parent template -- disclosed simplification: a streak that started in
    the previous season's final weeks is undercounted by design, not a
    leak). Because pushes (`team_covered` NaN) are already dropped from the
    long table before this column is built (same drop the parent battery
    performs), a push neither extends nor breaks a streak here -- a
    disclosed, defensible choice, not the only possible one. Flag:
    `ats_streak_len >= 3` (the streak length entering this game, using only
    strictly prior results). `sign = +1`.

## Reporting and recording

Every fresh cell (2, 3, 4, 5, 6, 7, 8, 9, 10, 12) is recorded to
`registry/weak_signals.json` via `nfl-ats weak-signals record` regardless of
sign or interval shape, `--classification unresolved_below_power` unless an
admissible closing ground applies (checked per cell in the results section
below), `--effect-units accuracy_points`, `--league nfl --season-start 2009
--season-end 2025`, `--family public_claim_battery`, and a `--plain-summary`
a fan can read. The two duplicate citations (1, 11) are also recorded under
`public_claim_*` names so the battery's own ledger is complete, citing the
existing entries' numbers in `--notes` rather than re-measuring.

## Results

**Measured**, one run of `scripts/public_claim_battery_screen.py` (seed
`20260905`, 20,000 bootstrap samples), artifact
`artifacts/public_claim_battery/20260905T031633Z/results.json`, experiment
row `registry/experiments/public-claim-battery-screen/20260905T031633Z.json`.
4,703 REG games (2009-2025) -> 8,634 team-game rows after dropping pushes.
Full-slate effect and week-blocked 95% interval are in `accuracy_points`
(percentage points of full-slate accuracy). All 12 cells (10 fresh + 2
duplicate citations) are recorded in `registry/weak_signals.json` under
`family=public_claim_battery`, all `unresolved_below_power` -- no cell in
this battery meets an admissible closing ground (no interval is entirely on
the wrong side of its predicted direction, and no positive control was run).

| # | Registry name | n_flag / n_total | Full-slate effect (pts) | Week-blocked 95% CI | P+ | 2009-2017 (pts, P+) | 2018-2025 (pts, P+) |
|---|---|---|---|---|---|---|---|
| 1 | `public_claim_dome_cold_fade` (duplicate cite) | 4,317 games population | +0.1052 | [-0.1180, +0.3264] | 0.8249 | n/a (cited cell not era-split in this doc) | n/a |
| 2 | `public_claim_primetime_dog` | 858 / 8,634 | -0.0257 | [-0.3781, +0.3283] | 0.4365 | -0.2836, P+0.1056 (n=429) | +0.2553, P+0.8256 (n=429) |
| 3 | `public_claim_post_bye_back` | 528 / 8,634 | -0.0493 | [-0.3016, +0.1982] | 0.3449 | +0.0713, P+0.6371 (n=278) | -0.1795, P+0.1401 (n=250) |
| 4 | `public_claim_division_dog` | 1,592 / 8,634 | +0.3692 | [-0.1574, +0.8948] | 0.9119 | +0.0823, P+0.5832 (n=838) | +0.6773, P+0.9658 (n=754) |
| 5 | `public_claim_road_fav_big_fade` | 338 / 8,634 | +0.2170 | [-0.0000, +0.4290] | 0.9736 | +0.1384, P+0.8220 (n=150) | +0.3029, P+0.9579 (n=188) |
| 6 | `public_claim_home_dog_3plus` | 1,149 / 8,634 | +0.0868 | [-0.3637, +0.5350] | 0.6409 | -0.0891, P+0.3739 (n=555) | +0.2812, P+0.7780 (n=594) |
| 7 | `public_claim_upset_letdown_fade` | 1,395 / 8,634 | +0.0345 | [-0.4595, +0.5189] | 0.5497 | +0.1727, P+0.6736 (n=721) | -0.1151, P+0.3598 (n=674) |
| 8 | `public_claim_blowout_loss_bounce_21` | 732 / 8,634 | +0.1392 | [-0.1703, +0.4519] | 0.8036 | +0.1471, P+0.7391 (n=404) | +0.1308, P+0.7070 (n=328) |
| 9 | `public_claim_week1_dog` | 266 / 8,634 | +0.1076 | [-0.0337, +0.2623] | 0.9219 | +0.1151, P+0.8761 (n=140) | +0.0994, P+0.7601 (n=126) |
| 10 | `public_claim_eliminated_fade_wk17_18` | 92 / 678 | +0.5119 | [-1.3210, +2.3345] | 0.6903 | +0.4167, P+0.5691 (n=38) | +0.5780, P+0.6746 (n=54) |
| 11 | `public_claim_thursday_road_fade` (duplicate cite) | 4,317 games population | +0.0442 | [-0.3116, +0.4041] | 0.5933 | n/a (cited cell not era-split in this doc) | n/a |
| 12 | `public_claim_ats_streak_regress` | 846 / 8,634 | +0.2440 | [-0.0883, +0.5746] | 0.9193 | +0.3084, P+0.9122 (n=431) | +0.1740, P+0.7437 (n=415) |

**Reading these per the taxonomy, not the "contains zero" reflex**: every
cell's interval crosses zero (claim 5's lower bound rounds to -0.0000,
closest of the batch) and per AGENTS.md that is the expected shape at this
sample size for a real small signal -- none is a negative result, none is
closed. The decay-not-replication prediction largely held for the
folklore-only claims (primetime dogs, post-bye, upset-letdown all land near
P+ 0.5, essentially coin flips), while the strongest leads in this batch
(`road_fav_big_fade` P+ 0.9736, `week1_dog` P+ 0.9219, `ats_streak_regress`
P+ 0.9193, `division_dog` P+ 0.9119, `eliminated_fade_wk17_18` P+ 0.6903 on
a thin n=92) point toward expected-value plays rather than settled facts --
probability_positive above 0.5 favours the candidate, per AGENTS.md, and a
promotion bar is not a decision bar. `public_claim_post_bye_back` (P+
0.3449) and `public_claim_primetime_dog` (P+ 0.4365) lean the other way but
are likewise not resolved (neither interval sits entirely on the wrong
side).

**Two claims (1, 11) duplicate already-recorded cells** and are cited, not
independently era-split in this table (their sources' own artifacts do not
report the same 2009-2017/2018-2025 split used here); their full citation
detail, including the opposite-signed forecast-based sibling for claim 1
and the plain-calendar sibling for claim 11, is in the "Claims 1 and 11"
section above and in each registry entry's `notes`.
