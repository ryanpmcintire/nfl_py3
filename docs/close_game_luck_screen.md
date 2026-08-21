# Close-game and turnover LUCK regression screen

Family: close-game and turnover LUCK regression. Status:
**predeclaration frozen before any cell was scored** (everything above the
"Measured results" heading was written before
`scripts/close_game_luck_screen.py` was run against any cover outcome);
measured results are appended at the bottom, tagged per the AGENTS.md
label-how-you-know-it rule.

## Prior-work overlap check

- `registry/weak_signals.json`: searched all 341 signal ids for
  `turn|fumb|luck|one_score|close_game|regress` (measured this session via a
  Python scan of the parsed JSON): zero hits. The closest string matches are
  `special_teams_return_*` and `travel_rest_return_trip_hangover`, which are
  unrelated families.
- Adjacent, DISCLOSED, not overlapping: `bias_battery_bad_team_late` and
  `bias_battery_great_team_late` (read, `registry/weak_signals.json`) are
  PER-GAME late-season situational contention flags (prior win pct thresholds
  within weeks 11-18 of the CURRENT season). This screen uses PRIOR-SEASON,
  full-season luck aggregates as persistent team traits joined one season
  forward — a different construct, population logic, and mechanism.
- Adjacent, DISCLOSED, not overlapping: `docs/offseason_retention.md`
  (read, lines 74-82) reports year-over-year persistence intervals for
  `off_turnover_rate` in a roster-retention context. That document measured
  whether retention predicts stat persistence; it never scored an ATS cell
  for prior-season turnover-differential extremes, which is what this screen
  does.
- `docs/pool_edge_plan.md` line 35 mentions "turnover bounces" and
  "one-score-game coin" flips as generic variance sources (read) — prose
  motivation, no screened cell anywhere.

Conclusion: the close-game-luck and turnover-regression cells below are
uncovered; built fresh.

## Mechanism

One-score-game record and turnover differential are famously noisy
small-N quantities (~5-9 one-score games, ~-13 to +13 turnover differential
per team-season). Teams that overperformed their point differential through
close-game wins and turnover bounces regress toward the mean, and markets may
under-adjust when pricing the following season's early openers. Prediction:
prior-season LUCK OVERperformers under-cover next season (fade),
LUCK UNDERperformers over-cover (rebound), with the fade strongest where
current-season information is thinnest (early weeks).

## Constructs (built from local schedules + PBP snapshot)

Sources: latest local schedules snapshot under `data/raw/*/schedules.parquet`
(REG, 2009-2025, franchise aliases applied, ATS outcomes via
`nfl_ats.features.add_ats_outcomes`, pushes dropped) and PBP snapshot
`20260817T184927Z` under `data/pbp/raw/`.

Per team-season:

- `one_score_luck`: win rate in one-score games (final margin of either kind
  <= 8 points) MINUS overall win rate. The expected-win-rate proxy is the
  team's own overall win rate; a positive value means the team won close
  games more often than its full record implies.
- `turnover_diff_per_game`: (takeaways − giveaways) / games played. The
  expectation proxy is the season's unweighted league mean (each trait is
  centered against its own season's league mean, PER-07/PBP-08 precedent).
- `takeaway_share`: takeaways / (takeaways + giveaways). DISCLOSED
  SUBSTITUTION: the task's construct (c) was fumble-recovery rate, but the
  local PBP snapshot carries only `interception` and `fumble_lost` — it has
  NO fumble-recovery attribution columns (measured this session: 45-column
  snapshot inspected), so pure recovery rate is not computable locally.
  Takeaway share is the computable combined-currency stand-in for the same
  recovery-luck mechanism (fumble recoveries are ~half of all takeaways and
  are the classic pure-noise component); it is NOT the same quantity and is
  labeled as such wherever reported.

Turnovers are counted per play as `interception + fumble_lost` attributed to
`posteam`; a team's takeaways are its opponents' giveaways in games it played.

## Reliability protocol (run BEFORE any cell was scored)

Year-over-year Pearson r and Spearman rho between a team's centered value in
season t and t+1, pooled across same-franchise consecutive-season pairs
2009-2025, with a 95% CI from 20,000 pair-level bootstrap resamples (seed
20260821). Exclusion rule (the ONE admissible input exclusion): a trait whose
YoY Pearson 95% CI sits entirely at or below 0 is excluded as a cell input on
`no_split_half_reliability` grounds. An interval crossing zero is NOT an
exclusion (binding taxonomy).

## Predeclared cells (6, frozen before scoring)

Population: NFL REG close-grade slate 2009-2025, team-perspective long table
(one row per team-game, `team_covered`). Flags use PRIOR-season traits joined
one season forward; 2009 games carry no prior trait and are reported as
missing, not dropped. Quartile thresholds are the 0.75 / 0.25 quantiles of the
pooled centered 2009-2025 team-season panel, computed once at run time.

| # | name | flag | value | sign |
|---|------|------|-------|------|
| L1 | `one_score_over_fade` | prior `one_score_luck_centered` >= Q75 | `team_covered` | -1 |
| L2 | `one_score_under_rebound` | prior `one_score_luck_centered` <= Q25 | `team_covered` | +1 |
| L3 | `turnover_over_fade` | prior `turnover_diff_per_game_centered` >= Q75 | `team_covered` | -1 |
| L4 | `turnover_under_rebound` | prior `turnover_diff_per_game_centered` <= Q25 | `team_covered` | +1 |
| L5 | `takeaway_share_extreme_fade` | prior `takeaway_share_centered` >= Q75 | `team_covered` | -1 |
| L6 | `early_season_luck_fade` | weeks 1-8 ONLY; prior `one_score_luck_centered` >= Q75 OR prior `turnover_diff_per_game_centered` >= Q75 | `team_covered` | -1 |

Sign convention: `sign` is the PREDICTED direction; `probability_positive` is
the bootstrap probability the prediction holds. Positive
`full_slate_effect_pts` = prediction confirmed, in accuracy points scaled to
the full slate (`nfl_ats.experiment_runner.scale_subset_effect`).

## Method

Week-blocked joint multinomial block bootstrap (primary), season-blocked
secondary, algorithm-identical to `scripts/redzone_reversion_screen.py` /
`scripts/team_style_screen.py`. 20,000 samples, seed 20260821, accuracy_points
full-slate units. Every cell is recorded regardless of sign or interval shape;
an interval crossing zero is never a closing ground (binding taxonomy).
Terminal classifications require an admissible `--closing-ground`; everything
else is `unresolved_below_power`. This script only measures; recording to the
weak-signal registry happens via explicit `nfl-ats weak-signals record`
commands returned separately.

## Measured results (2026-08-21 run)

All numbers below are **measured** this session: artifact
`artifacts/close_game_luck_screen/20260821T182234Z/results.json`, produced by
`scripts/close_game_luck_screen.py` against PBP snapshot `20260817T184927Z`
and schedules `data/raw/20260817T235649Z/schedules.parquet` (4,317 REG
close-graded games; 544 team-seasons in the panel).

### Reliability (measured, YoY on season-centered traits, 512 pairs each)

| trait | Pearson r | 95% CI | Spearman |
|-------|-----------|--------|----------|
| `one_score_luck` | +0.149 | [+0.063, +0.234] | +0.133 |
| `turnover_diff_per_game` | +0.132 | [+0.049, +0.213] | +0.135 |
| `takeaway_share` | +0.163 | [+0.082, +0.242] | +0.164 |

No trait's CI sits entirely at or below zero, so no cell input is excluded on
reliability grounds (measured). All three persistence figures are thin-but-
real, consistent with the luck-regression mechanism itself.

### Cell results (measured; week-blocked primary, accuracy_points full-slate units, P+ = probability_positive)

| # | cell | n_flag | effect pts | 95% CI | P+ |
|---|------|--------|-----------|--------|-----|
| L1 | one_score_over_fade | 2,021 | -0.008 | [-0.579, +0.550] | 0.482 |
| L2 | one_score_under_rebound | 2,012 | -0.015 | [-0.546, +0.517] | 0.473 |
| L3 | turnover_over_fade | 2,089 | +0.008 | [-0.569, +0.569] | 0.501 |
| L4 | turnover_under_rebound | 2,036 | +0.409 | [-0.153, +0.969] | 0.920 |
| L5 | takeaway_share_extreme_fade | 2,080 | -0.031 | [-0.612, +0.539] | 0.448 |
| L6 | early_season_luck_fade | 1,800 | -0.183 | [-1.465, +1.085] | 0.388 |

Season-blocked secondary intervals agree in sign for every cell; L4's
season-blocked secondary is [+0.030, +0.795], P+ 0.981 (measured, same
artifact). 496 team-game rows per full-slate cell carry no prior trait (2009
games and franchise-gap seasons), reported as missing, not dropped.

### Classification

Every cell is category 3, `unresolved_below_power`: no interval sits wholly on
the wrong side of zero, no reliability exclusion fired, and there is no
positive control. Per the binding taxonomy these are recorded, not closed.
The strongest read (**inferred**, my reasoning, not evidence): L4 — fading
NOTHING and instead backing prior-season NEGATIVE-turnover-differential teams
to over-cover — is the only cell with a directional lean (P+ 0.920 primary,
0.981 secondary), but a wholly-above-zero interval has no resolved-positive
terminal state in this taxonomy either, and L4 shares its underlying trait
with L3 (a correlated decomposition, not an independent confirmation). The
near-zero L1/L2 pair says (**inferred**) that close-game-luck regression, if
real, is not expressed in next-season ATS cover rates at this evaluator's
resolution.
