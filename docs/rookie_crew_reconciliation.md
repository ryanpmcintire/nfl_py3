# Rookie-crew: reconciling two reads that disagree in sign

Two numbers for "rookie officiating crew, underdog side" sit in this
repository pointing opposite ways. This document reconciles them and then
predeclares ONE definitive measurement, written in full **before** that
measurement was scored.

## Part 1 -- the reconciliation (measured/read before anything new was run)

| | Read A (the ROADMAP line) | Read B (the 2026-09-08 battery) |
| --- | --- | --- |
| Headline | **-1.097 pts, [-2.691, +0.222], P+ 0.046** | **+0.732 pts, [+0.000, +1.483], P+ 0.972** |
| Registry name | `rookie_crew_underdog_on_production` | `officials_archive_rookie_archive_era_floor_proxyline_vs_openerline_opener_2020_2025` |
| Artifact | `artifacts/officials_flags_on_production/rookie_underdog/20260905T104625Z/results.json` (`mode=screen`) | `artifacts/research/laneAD/opener_proxy.json`; write-up `docs/officials_archive_battery.md:435` |
| Run date | 2026-09-05 | 2026-09-08 |
| Population / window | **2020-2021 only**, 456 non-push paired games, 35 week blocks | **2020-2025**, 1,503 non-push games, 107 week blocks |
| Grade | Tuesday opener, probability rule | Tuesday opener, probability rule |
| **Baseline** | **production** (`weak_stack`, 90 columns) | **its own opener-line twin** -- the SAME rule built from the Tuesday-opener store alone. Not production. |
| Rule definition | shipped `rookie_crew_underdog_flag`: referee `prior_seasons_experience <= 1`, `season >= ROOKIE_ELIGIBLE_SEASON_FLOOR = 2016`, referee identity from the 2015-2025 officials feed, signed +1 home-dog / -1 away-dog at the Tuesday opener | same flag with two constraints lifted: referee identity from the **archive-extended** table (2009-2025) and the season floor derived from that population's own first season (**2010**), plus the flag's line source filled with the archived nflverse spread before 2020 |
| Probability method | harness default **ECDF** -- `probability_method` was omitted and `opener_pick_evaluation` hard-coded `"ecdf"` | **`gaussian_median`**, read from `artifacts/active_ats_model.json` by `active_model()` |
| Home-side offset | **absent** (predates the 2026-09-07 serve) | served |
| Picks changed | 9 | 33 |

**The sign difference is not a disagreement about football.** The two cells
answer different questions:

1. **Different baseline, and this is the whole story.** Read B is an
   archive-vs-feed DELTA, not a vs-production effect. The same battery's
   vs-production rows are `officials_archive_rookie_feed_opener_2020_2025`
   **-0.599 [-1.340, +0.068] P+ 0.037** (the shipped build) and
   `officials_archive_rookie_archive_era_floor_proxyline_vs_production_opener_2020_2025`
   **+0.133 [-0.203, +0.528] P+ 0.700** (the rebuilt one). The arithmetic
   closes exactly: -0.599 + 0.732 = +0.133. So "+0.73" is what the archive
   rebuild BUYS, measured against the rule's own previous build; it was never
   a claim that the rule beats production by 0.73.
2. **Different rule.** Read A's flag is identically zero before 2016 and
   before 2015 has no referee identity at all; Read B's has real values on
   2009-2019 training games, which changes the coefficient the ridge carries
   into the graded window.
3. **Different window.** 2020-2021 (456 games) versus 2020-2025 (1,503). The
   battery re-scored its own harness on the 2020-2021 sub-window and got
   **-1.316 [-2.995, +0.214] P+ 0.027** against Read A's -1.097 -- same sign,
   same magnitude. The two agree on that window.
4. **The ECDF defect is real and is FIXED.** `opener_pick_evaluation`
   defaulted an absent `probability_method` to `"ecdf"` while production serves
   `gaussian_median`; `scripts/on_production_opener_confirmation.py:99` now
   calls `resolve_active_probability_method()`. Fixed in commit `013e99b`
   (2026-09-08 13:25 -0400). Read A was produced before that commit and is
   graded on the wrong mapping; Read B reads the active manifest and is not
   affected. This explains why the 2020-2021 replication is close but not
   bit-for-bit.

**Neither number is fabricated; the ROADMAP line is mislabelled, not wrong.**
It states a 2020-2021, pre-fix, ECDF-graded, vs-production figure as if it
were the family's verdict. The ROADMAP edit made alongside this document adds
the window, the baseline and the superseding 2020-2025 numbers.

## Part 2 -- the predeclaration (written before scoring)

### The rule

`rookie_archive_era_floor` on close-proxy lines, exactly as
`scripts/officials_archive_battery_eval.py` defines it after its wiring fix:

- **Crew identity**: the archive-extended referee table,
  `referee_game_table(include_archive=True)`, 2009-2025 regular season, head
  referee only.
- **Rookie**: `prior_seasons_experience <= 1`
  (`ROOKIE_PRIOR_EXPERIENCE_MAX`), restricted to
  `season >= ARCHIVE_ROOKIE_SEASON_FLOOR = 2010` -- the archive's own first
  season plus one, the same derivation that produced the shipped 2016 from the
  feed's 2015.
- **Side**: signed by the UNDERDOG side --- `+1` when the home team is the
  underdog at the line, `-1` when the away team is, `0` otherwise
  (`_signed_by_line(..., favourite=False)`).
- **Line source**: the Tuesday-opener consensus spread where it exists
  (2020-2025); the archived nflverse spread as a close proxy before 2020,
  where the opener store does not reach.
- **How it enters**: as a FEATURE column in the ridge, profile
  `weak_stack_rookie_crew_underdog` (= production `weak_stack` plus exactly
  this one column). It is not a hand-set pick flip; the ridge learns the
  coefficient and therefore the served direction.

### Grade, population, harness

- Graded at the **Tuesday opener**, probability rule, with the served
  home-side offset, against the served evaluation
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` (active
  model `c657058903f3232b`, `gaussian_median`, 1,537 games, **1,503 non-push**).
- A **replay gate** runs first: the harness must reproduce that artifact's
  opener probabilities and picks (max gap <= 1e-9, zero pick disagreements)
  before any candidate is scored. If it fails, nothing is scored.
- Window **2020-2025**; eras **2020-2021 / 2022-2023 / 2024-2025** reported as
  magnitudes, never as presence/absence.

### Arms

- **(a) standalone** -- candidate opener picks vs the raw model's opener picks.
- **(b) marginal on the played card** -- the served three-member card (coach
  fade + division revenge + player arrests) recomposed on top of the candidate
  picks, versus the same card on top of production, using
  `scripts/spread_regime_opener_eval.composed_picks`, which is
  `nfl_ats.overlay_composition`'s own combination rule.

### Uncertainty

Week-blocked bootstrap, **20,000 draws, seed 20260821**, whole season-week
blocks resampled; within-week correlation is ZERO by owner mandate and is
never estimated or padded. Pushes excluded. `probability_positive` reported
for every cell; the binary "contains zero" is never a verdict.

### Reliability

Split-half reliability of the crew trait itself, two pairings over 2009-2025:

- **season-to-season** (does a crew's tendency persist year to year) --
  `build_season_to_season_pairs`, the predeclared primary;
- **within-season odd/even weeks** -- `build_odd_even_halves`, minimum 3 games
  per half, Spearman-Brown corrected, reported alongside.

The trait is the referee's **underdog-cover rate**: over that referee's games
in a season, the fraction in which the underdog at the line covered. That is
the quantity the rule's mechanism requires to persist. Tenure itself is not
measured for reliability -- it increments by one every year by construction
and its persistence is arithmetic, not evidence.

### Cells recorded

Every cell below is written with `nfl-ats weak-signals record`, league `nfl`,
units `accuracy_points` (the reliability cells in `correlation`), family
`rookie_crew_reconciliation`, category `onfield`, classification
`unresolved_below_power` unless one of the two admissible closing grounds
applies:

| Name | Arm | Window |
| --- | --- | --- |
| `rookie_crew_reconciled_standalone_2020_2025` | standalone vs raw model | 2020-2025 |
| `rookie_crew_reconciled_standalone_2020_2021` | standalone vs raw model | 2020-2021 |
| `rookie_crew_reconciled_standalone_2022_2023` | standalone vs raw model | 2022-2023 |
| `rookie_crew_reconciled_standalone_2024_2025` | standalone vs raw model | 2024-2025 |
| `rookie_crew_reconciled_card_2020_2025` | marginal on the played card | 2020-2025 |
| `rookie_crew_reconciled_card_2020_2021` | marginal on the played card | 2020-2021 |
| `rookie_crew_reconciled_card_2022_2023` | marginal on the played card | 2022-2023 |
| `rookie_crew_reconciled_card_2024_2025` | marginal on the played card | 2024-2025 |
| `rookie_crew_reconciled_reliability_season_pairs_2009_2025` | trait reliability | 2009-2025 |
| `rookie_crew_reconciled_reliability_odd_even_2009_2025` | trait reliability | 2009-2025 |

### Discount, declared in advance

This is a **second look** at a window whose first look was already read twice
(2026-09-05 and 2026-09-08). It is descriptive, not independent confirmation.
The rule, the direction, the window, the seed and the cell list above were
fixed before the runner was executed.

## Part 3 -- results

Measured 2026-09-09.
Artifacts: `artifacts/rookie_crew_reconciliation/20260909T190000Z/results.json`
and `.../wiring_isolation/results.json`.
Runners: `scripts/rookie_crew_reconciliation_eval.py`,
`scripts/rookie_crew_wiring_isolation.py`.

**Replay gate passed** before anything was scored: 1,537 games, maximum
opener-probability gap **3.77e-15**, **0** pick disagreements against
`artifacts/opener_evaluation/20260909T183120Z`, feature digest `5c5d1944...`.

### The decision, first

**Adding the reconciled rookie-crew rule to the played card raises expected
accuracy by +0.133 points over 2020-2025, `probability_positive` 0.790.** The
pool is forced picks, so a rule that is 79% likely to be better is played; the
only reason it is not on the card today is that crew assignments are not known
at the Tuesday lock. Standalone against the raw model it is +0.200,
`probability_positive` 0.843.

### 2020-2025, 1,503 non-push games, 107 week blocks

| Arm | Effect (pts) | 95% CI | P+ | Picks changed |
| --- | ---: | --- | ---: | ---: |
| Standalone vs the raw model | **+0.200** | [-0.198, +0.602] | **0.843** | 11 |
| **Marginal on the played three-member card** | **+0.133** | [-0.198, +0.467] | **0.790** | 6 |
| Marginal on the four-member union (zone included) | +0.466 | [+0.135, +0.811] | 0.9998 | 7 |
| vs its own opener-line twin (the 2026-09-08 cell, re-measured) | +0.798 | [+0.067, +1.545] | 0.986 | 32 |

The last row reproduces the battery's recorded **+0.732 [+0.000, +1.483] P+
0.972** on a different active model and a different seed. Same sign, same
magnitude, and the arithmetic closes: the shipped feed-only arm is -0.599 vs
production, -0.599 + 0.798 = +0.200, the reconciled arm's standalone number.

### Per-era magnitudes (never absence)

| Era | Standalone | P+ | On the played card | P+ | n / weeks |
| --- | ---: | ---: | ---: | ---: | --- |
| 2020-2021 | **+0.439** [+0.000, +1.111] | **0.936** | **+0.439** [+0.000, +1.111] | **0.936** | 456 / 35 |
| 2022-2023 | 0.000 [-0.591, +0.575] | 0.504 | 0.000 [-0.591, +0.574] | 0.503 | 514 / 36 |
| 2024-2025 | +0.188 [-0.571, +0.963] | 0.666 | 0.000 [-0.565, +0.564] | 0.492 | 533 / 36 |

The whole six-season gain sits in 2020-2021, which is also where the flag
fires most: flagged games in the graded window run 51, 29, 16, 32, 16, 16 for
2020 through 2025. The 2022-2025 cells are dead heats on two to five changed
picks each, which is a magnitude statement about those eras and not an absence.

### The wiring fix is not what moved the sign

The shipped feed-only flag, re-scored against production under the FIXED
wiring (`gaussian_median` read from the active manifest, served home-side
offset):

| Window | Fixed wiring | P+ | Forced back to the retired `ecdf` default | P+ |
| --- | ---: | ---: | ---: | ---: |
| 2020-2025 | -0.599 [-1.338, +0.131] | 0.049 | -0.599 [-1.360, +0.132] | 0.055 |
| 2020-2021 | -1.316 [-2.921, +0.212] | 0.040 | -1.316 [-3.057, +0.222] | 0.055 |

The point estimate is **identical to the digit** under both mappings, because
baseline and candidate move together; only the interval and
`probability_positive` shift. The ECDF defect was real and is fixed, but it
never carried this sign. The -1.316 also reproduces the ROADMAP's recorded
-1.097 on the same window and rule, so that number is sound for what it
measured.

### Split-half reliability of the crew trait

Referee's underdog-cover rate, 4,280 graded games, 290 referee-seasons:

| Pairing | Pearson r | 95% CI | P+ | Units |
| --- | ---: | --- | ---: | ---: |
| **Season to season (primary)** | **+0.0756** | [-0.0201, +0.1453] | **0.951** | 255 pairs, 16 seasons |
| Within-season odd/even weeks | -0.0185 | [-0.1269, +0.0927] | 0.344 | 290 referee-seasons |

The label-shuffle null sits at mean r 0.0144, sd 0.0584, so the estimator is
trusted. **Reliability is small but positive, so
`no_split_half_reliability` is NOT available as a closing ground for this
family.** The Spearman-Brown corrected within-season figure is -0.038; half a
season is too few games to see a tendency this faint, which is a statement
about that instrument, not about the trait.

### What serving it would take (NOT wired here)

Crew assignments are published Wednesday-Thursday of game week, after the
Tuesday lock (`docs/referee_assignments_capture.md`), so this rule can never
reach the Tuesday card. The vehicle is the late-week refresh path, and the
live captures already exist (`data/players/referee_assignments/`, newest
`20260909T201816Z`; scheduler job `referee_assignments_wed`,
`scripts/capture_scheduler.py:873`). Three things stand between the measured
number and a served one:

1. **This rule is a model FEATURE, not an overlay tilt.**
   `nfl_ats.crew_tilt_refresh_overlay.crew_tilt_flags` is a probability tilt
   on two `penalty_crew_tendencies` cells; the rookie-crew flag changes the
   fitted ridge. Serving it means either a refresh-time refit on profile
   `weak_stack_rookie_crew_underdog` for games whose deadline is still open,
   or converting the learned coefficient into a third tilt cell in that
   module. The first preserves the measurement; the second does not and would
   need its own read.
2. **The season floor has to move from 2016 to the loaded population's own
   first season.** `ROOKIE_ELIGIBLE_SEASON_FLOOR` in
   `src/nfl_ats/officials_flag_features.py:78` is the constant this whole
   reconciliation turns on. That is item (b) of the officials battery's own
   three-change list.
3. **The flag's line source has to accept the close proxy before 2020.** The
   graded number above is built that way; a serve built on the Tuesday-opener
   store alone is the -0.599 arm, not the +0.200 one.

The pick-deadline rule already in `nfl_ats.pick_refresh.pick_deadline` --
`min(kickoff, Sunday 16:00 ET)` -- bounds which games a Wednesday snapshot may
touch, and on a normal week that is every game except any Wednesday or
Thursday kickoff.

### Cells recorded

Twelve rows, family `rookie_crew_reconciliation`, all
`unresolved_below_power`, all carrying `--reliability 0.0756`: the eight
predeclared accuracy cells, the two reliability cells, and two unpredeclared
reconciliation controls (`rookie_crew_reconciled_shipped_feed_arm_2020_2025`
and `..._2020_2021`) added so the ROADMAP correction is auditable from the
registry alone. Nothing is closed and no closing ground is claimed.
