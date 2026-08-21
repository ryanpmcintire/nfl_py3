# Red-zone / third-down efficiency mean-reversion screen

Family: red-zone and third-down efficiency mean-reversion. Status:
**predeclaration frozen before any cell was scored** (this section was written
before `scripts/redzone_reversion_screen.py` was run against any cover
outcome); measured results are appended at the bottom, tagged per the
AGENTS.md label-how-you-know-it rule.

## Prior-work overlap check (measured this session)

- `docs/` grep for red-zone / third-down screens: no hit. The only "red zone"
  occurrences are `docs/era_events.md` using red zone as a drive-state bucket
  for a fourth-down-aggression construct (read, `docs/era_events.md` lines
  109, 155) — a different family.
- `registry/weak_signals.json` grep: the only "3rd" match is the unrelated
  "3rd+ consecutive true road game" schedule signal (read,
  `registry/weak_signals.json` line 1113).
- PBP-08 (`docs/team_style.md`) screened formation/pace *personality* traits
  (shotgun rate, air-yards shares, run-direction HHI, pace) — read, this
  session. It did not screen RZ TD rate or 3rd-down conversion.
- PBP-06 special teams (`docs/special_teams_battery.md`) is out of scope.

Conclusion: these two constructs are uncovered; built fresh. No cell below
reuses a PBP-08 dimension.

## Mechanism

RZ TD rate and 3rd-down conversion are noisy small-N rates (~50-70 RZ drives,
~180-220 third downs per team-season). Extreme values regress toward the
mean, and the market may anchor on last season's extremes when pricing the
next season's opener months. Prediction: prior-season OVERperformers under-
cover next season (fade), prior-season UNDERperformers over-cover (rebound),
with the effect strongest where current-season information is thinnest
(early weeks) and on the opponent side where a hot-looking offense meets a
stingy-looking defense.

## Constructs (built from the local PBP snapshot)

Source: latest local play-by-play snapshot under `data/pbp/raw/` (measured
present this session: seasons 2009-2025), filtered to REG plays through
`nfl_ats.pbp.analysis_plays` (the house v1 efficiency filter), franchise
aliases applied via `TEAM_ABBREVIATION_ALIASES`.

Per team-season, offense side:

- `rz_td_rate`: drives that reach the opponent red zone (minimum
  `yardline_100` within the drive <= 20) and end in a touchdown, divided by
  drives that reach the red zone.
- `third_down_conv_rate`: third-down plays converted (`first_down == 1`)
  divided by third-down plays.

Defense side (grouped by `defteam`, same formulas): `rz_td_rate_allowed`,
`third_down_conv_allowed`. Each trait is centered against its own season's
unweighted league mean (PER-07/PBP-08 precedent: raw rates conflate era-wide
drift with team identity).

## Reliability protocol (run BEFORE cells were scored)

Year-over-year Pearson r and Spearman rho between a team's centered value in
season t and t+1, pooled across same-franchise consecutive-season pairs
2009-2025, with a 95% CI from 20,000 pair-level bootstrap resamples (seed
20260821). Exclusion rule (the ONE admissible input exclusion): a trait whose
YoY Pearson 95% CI sits entirely at or below 0 is excluded as a cell input on
`no_split_half_reliability` grounds. An interval crossing zero is NOT an
exclusion (binding taxonomy).

## Predeclared cells (6, frozen before scoring)

Population: NFL REG close-grade slate 2009-2025, team-perspective long table
(one row per team-game, `team_covered`) for identity cells; game-level
(`home_cover`) for the matchup cell. Flags use PRIOR-season traits joined one
season forward; 2009 games carry no prior trait and are reported as missing,
not dropped. Quartile thresholds are the 0.75 / 0.25 quantiles of the pooled
centered 2009-2025 team-season panel.

| # | name | flag | value | sign |
|---|------|------|-------|------|
| C1 | `rz_over_fade` | prior-season `rz_td_rate_centered` >= Q75 | `team_covered` | -1 |
| C2 | `rz_under_rebound` | prior-season `rz_td_rate_centered` <= Q25 | `team_covered` | +1 |
| C3 | `third_down_over_fade` | prior-season `third_down_conv_rate_centered` >= Q75 | `team_covered` | -1 |
| C4 | `third_down_under_rebound` | prior-season `third_down_conv_rate_centered` <= Q25 | `team_covered` | +1 |
| C5 | `rz_hot_offense_vs_stingy_defense` | AWAY prior `rz_td_rate_centered` >= Q75 AND HOME prior `rz_td_rate_allowed_centered` <= Q25 (allowed fewest = stingiest) | `home_cover` | +1 |
| C6 | `early_season_extreme_fade` | weeks 1-8 ONLY; prior `rz_td_rate_centered` >= Q75 OR prior `third_down_conv_rate_centered` >= Q75 | `team_covered` | -1 |

Sign convention: `sign` is the PREDICTED direction; `probability_positive` is
the bootstrap probability the prediction holds. Positive `full_slate_effect_pts`
= prediction confirmed, in accuracy points scaled to the full slate
(`nfl_ats.experiment_runner.scale_subset_effect`).

## Method

Week-blocked joint multinomial block bootstrap (primary), season-blocked
secondary, algorithm-identical to `scripts/team_style_screen.py` /
`scripts/nfl_weather_battery_screen.py`. 20,000 samples, seed 20260821,
accuracy_points full-slate units. Every cell is recorded regardless of sign
or interval shape; an interval crossing zero is never a closing ground
(binding taxonomy). Terminal classifications require an admissible
`--closing-ground`; everything else is `unresolved_below_power`.

## Measured results (2026-08-21 run)

All numbers below are **measured** this session: artifact
`artifacts/redzone_reversion_screen/20260821T181025Z/results.json`, produced by
`scripts/redzone_reversion_screen.py` against PBP snapshot
`20260817T184927Z` and schedules `data/raw/20260817T235649Z/schedules.parquet`
(4,317 REG close-graded games; 544 team-seasons per panel).

### Reliability (measured, YoY on season-centered traits, 512 pairs each)

| trait | Pearson r | 95% CI | Spearman |
|-------|-----------|--------|----------|
| `rz_td_rate` | +0.141 | [+0.051, +0.227] | +0.128 |
| `third_down_conv_rate` | +0.407 | [+0.337, +0.473] | +0.394 |
| `rz_td_rate_allowed` | +0.201 | [+0.118, +0.282] | +0.188 |

No trait's CI sits entirely at or below zero, so no cell input is excluded on
reliability grounds (measured). RZ TD rate is the weakest trait — its
persistence is real but thin, which is consistent with the mean-reversion
mechanism itself.

### Cell results (measured; week-blocked primary, accuracy_points full-slate units, P+ = probability_positive)

| # | cell | n_flag | effect pts | 95% CI | P+ |
|---|------|--------|-----------|--------|-----|
| C1 | rz_over_fade | 2,012 | +0.030 | [-0.543, +0.596] | 0.538 |
| C2 | rz_under_rebound | 2,040 | -0.106 | [-0.699, +0.495] | 0.361 |
| C3 | third_down_over_fade | 2,086 | +0.367 | [-0.259, +0.999] | 0.872 |
| C4 | third_down_under_rebound | 2,041 | -0.356 | [-0.893, +0.181] | 0.099 |
| C5 | rz_hot_offense_vs_stingy_defense | 234 | +0.161 | [-0.207, +0.522] | 0.805 |
| C6 | early_season_extreme_fade | 1,431 | -0.450 | [-1.597, +0.683] | 0.209 |

Season-blocked secondary intervals agree in sign and width for every cell
(measured, same artifact).

### Classification

Every cell is category 3, `unresolved_below_power`: no interval sits wholly on
the wrong side of zero (C4's P+ 0.099 is a leaning, not a resolved wrong sign —
its upper bound is +0.181), no reliability exclusion fired, and there is no
positive control. Per the binding taxonomy these are recorded, not closed.
The strongest read (**inferred**, my reasoning, not evidence): C3/C4 form a
directionally coherent pair — fading prior-season 3rd-down overperformance at
P+ 0.872 while its mirror rebounds at P+ 0.099 — but each cell alone is
unresolved and the pair shares one underlying trait, so they are correlated
decompositions, not independent confirmations.
