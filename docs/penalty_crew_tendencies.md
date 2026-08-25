# Penalty-type crew tendencies: widening the referee battery to penalty types

**Session: 2026-08-20.** Builds `docs/archive/data_source_scout_v4.md`'s #1-ranked lead
("Penalty-type crew tendencies — an in-house schema-widening question, not a
new source"): widen the local PBP snapshot to retain `penalty_type`/
`penalty_team`, then extend `docs/referee_battery.md`'s already-built,
already-reliable (`mean_total` **+0.370**, read from that file) referee-crew
penalty-rate battery from a raw count to penalty-**type**-specific rates.

**Predeclared BEFORE any effect on this population was computed.** Family
name: `penalty_crew_tendencies` (registry entries prefixed `penalty_crew_`).
Units: `accuracy_points`. Blocking: week primary, season secondary. Grade:
mostly `close`, one cell `opener` (see cell A). Seed: `20260820` throughout.
Samples: 20,000. Every cell below is recorded to `registry/weak_signals.json`
via `nfl-ats experiment run`, **regardless of which way the sign comes out**
— per `AGENTS.md`'s binding rule, an interval crossing zero is never grounds
to reject, fail, or skip recording an experiment. Only two grounds ever close
a line of work: (1) a refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability, or (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`, reported with
`probability_positive`, never the binary "contains zero".

## 1. Widening the data

**Gap confirmed, read**: `nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS`
(`src/nfl_ats/pbp.py`) carries `penalty`/`penalty_yards` but not
`penalty_type`/`penalty_team`; the same is true of the project's separate,
wider `data/pbp/team_style/raw_pbp_narrow.parquet` snapshot (**read**, its own
manifest's column list has `air_yards`/`pass_length`/`shotgun`/etc. but no
`penalty_type`). The existing referee battery's own
`data/raw/officials/*/game_penalties.parquet` (built 2026-08-19) already
aggregates penalty **counts** from a fresh `nflreadpy.load_pbp()` pull but
never persisted the type breakdown either.

**Measured** (`nflreadpy.load_pbp(seasons=[2023])`): the upstream nflverse PBP
schema DOES carry `penalty_type`, `penalty_team`, `penalty_player_id`,
`penalty_player_name`, and the already-retained `penalty`/`penalty_yards` —
confirming the scout doc's framing: this is a re-pull, not a new source.

**Built this session**: `scripts/fetch_penalty_type_snapshot.py` re-fetches
`nflreadpy.load_pbp()` one season at a time (2015-2025, matching
`load_officials()`'s own documented floor and the existing
`game_penalties.parquet`'s coverage) and persists ONLY a small derived
long-format aggregate — one row per `(game_id, penalty_type)` with
`penalties_total`/`penalties_on_home`/`penalties_on_away`/
`penalty_yards_total` — to a NEW snapshot directory. The existing
`officials.parquet`/`game_penalties.parquet` files are untouched.

- **Measured**: new snapshot `data/raw/officials/20260820T112517Z/game_penalty_types.parquet`,
  23,962 rows, 3,028 distinct games, 64 distinct `penalty_type` values.
- **Row-alignment check (measured)**, run by the same script
  (`verify_against_existing_totals`): re-summing the new long table's type
  counts back up to one row per `game_id` and comparing against the existing
  `game_penalties.parquet` gives **0 games only in either table, 0 count
  mismatches across all 3,028 matched games**, and **every one of the 11
  seasons' (2015-2025) distinct game counts matches exactly** between the two
  tables. Home/away attribution (`penalty_team == home_team` →
  `penalties_on_home`, etc.) reproduces the existing table's own
  `2015_01_BAL_DEN` row (8 home / 3 away) bit-for-bit.
- Home-team pregame pass rate (used by cells B/C below) comes from the
  ALREADY-BUILT, already pregame-safe `data/processed/game_features_pbp.parquet`
  (`home_pbp_off_pass_rate`, an EWMA of games strictly before the one being
  scored — **read**, `nfl_ats.pbp.enrich_with_pbp_features`'s own docstring:
  "Attach PBP states from games strictly earlier than the game being
  scored"). No new leakage surface is introduced by reusing it.

## 2. Pregame-safety argument

Identical to `docs/referee_battery.md`'s existing argument: crew assignment
(who is head referee) is public before kickoff, and every trait below is the
referee's own PRIOR-season history — never this game's own penalties.
`tests/test_experiment_runner.py::test_referee_type_trait_does_not_use_this_games_own_penalty_type_count`
(new, this session) mutates a game's OWN penalty-type count to a value that
would flip its lagged quartile if it were (incorrectly) read directly, and
asserts the lagged quartile is unchanged — the type-trait construction only
ever reads `shift(1)` over `(official_name, season)`, exactly mirroring the
existing `mean_total`/`mean_diff` leakage test's mutation pattern.
`test_referee_type_trait_uses_the_prior_season_lag` additionally confirms the
new per-type trait reproduces the SAME quartile ranking as the already-verified
`mean_total` trait when fed identical counts.

## 3. Reliability table (predeclared selection rule, measured before any cell)

Selection rule, fixed before any correlation was computed: the **8 most
frequent penalty types by total 2015-2025 occurrence count** (an objective,
sign-blind frequency floor), which happens to include every type named in
this lead's own predeclared cells (Offensive Holding, Defensive Pass
Interference, False Start). Same construction as `mean_total`/`mean_diff`:
per-(official, season) mean count/game, year-over-year (season, season+1)
Pearson correlation, 158 referee-season pairs throughout (officials snapshot
`20260819T190537Z`, penalty-type snapshot `20260820T112517Z`, 29 distinct
referees).

| Penalty type | 2015-2025 occurrences | `mean_total`-style split-half r (158 pairs) | Read |
|---|---|---|---|
| Offensive Holding | 7,323 | **+0.3226** | real, moderate — similar magnitude to the existing overall `mean_total` (+0.370) |
| False Start | 6,634 | +0.0915 | weak |
| Defensive Pass Interference | 3,080 | **-0.0663** | near zero (slightly negative) |
| Defensive Holding | 2,301 | +0.2702 | real, moderate |
| Unnecessary Roughness | 2,068 | +0.2271 | real, moderate |
| Defensive Offside | 1,732 | +0.0836 | weak |
| Delay of Game | 1,657 | +0.1946 | weak-moderate |
| Neutral Zone Infraction | 1,375 | +0.1776 | weak-moderate |

Per `AGENTS.md`, none of these near-zero readings are grounds to discard the
type on their own (this is a point-estimate correlation, not a bootstrapped
interval, so it is reported descriptively, not run through the mechanical
zero-crossing classifier) — Offensive Holding and Defensive Holding both show
real, moderate persistence comparable to the existing battery's own
`mean_total`; Defensive Pass Interference does not, and that is reported
plainly in cell B below (mirrors `docs/referee_battery.md` cells 5/6's own
treatment of `mean_diff`'s near-zero reliability: still run, sign reported
either way).

## 4. Predeclared cells

Four cells, matching the task brief's own named examples exactly. Each is a
boolean AND of two top/bottom-quartile flags (`subset_bias` is a boolean-flag
framework project-wide, see `docs/experiment_pipeline.md` — no continuous
interaction terms). Population `[2016, 2025]` throughout (excludes the 2015
left-censoring artifact, same as the existing referee battery); cell A's
`grade="opener"` further trims to the paired-opener archive's own ~2020-2025
coverage.

Flag builders live in `src/nfl_ats/experiment_runner.py` (`FLAG_BUILDERS`
registry: `referee_high_flag_heavy_underdog`,
`referee_dpi_tilt_pass_heavy_favorite`, `referee_holding_tilt_run_heavy`,
`referee_flag_rate_high_total_line`). Specs:
`registry/experiment_specs/penalty_crew_*.json` (four files).

### Cell A — `penalty_crew_high_flag_heavy_underdog_opener`

Home team's referee's PRIOR-season `mean_total` (existing trait, reused
unchanged) top quartile AND home team getting >= 7 points at the **opener**
line (heavy underdog), graded at the opener per `AGENTS.md`'s binding "grade
the decision at the opener" rule. Mechanism: cell 1's hypothesized road-team
tempo-disruption edge (`docs/referee_battery.md`) is hypothesized to
concentrate when the home team most needs the extra stoppages' clock/tempo
control against a stronger opponent. Sign: +1.

**Measured**: n_flag=18 / n_total=2,860 (opener population trimmed to
1,537 paired games x 2 sides minus incomplete rows → 2,860 classifiable
team-games; fraction_of_slate=0.63%). Week-blocked (107 blocks): effect
**+0.1056** accuracy points, 95% **[-0.0351, +0.2320]**, se=0.0675,
**P+ = 0.9204**. Reliability: reused `mean_total` (+0.370, 158 pairs).
Classification: `unresolved_below_power`, `closing_ground: null` (interval
crosses zero — expected at this resolution, per `AGENTS.md`, not a rejection
ground).

### Cell B — `penalty_crew_dpi_tilt_pass_heavy_favorite`

Home team is the favorite AND top-quartile prior-rolling pregame pass rate
AND the game's referee's PRIOR-season Defensive Pass Interference rate in the
top quartile. Mechanism: a high-DPI-calling crew is hypothesized to
disproportionately extend a pass-heavy offense's drives, so a pass-heavy
favorite facing such a crew is hypothesized to cover MORE. Sign: +1.

**Measured**: n_flag=79 / n_total=4,786 (fraction_of_slate=1.65%).
Week-blocked (175 blocks): effect **-0.0744** accuracy points, 95%
**[-0.2518, +0.1087]**, se=0.0924, **P+ = 0.1926** (leans AGAINST the
hypothesized direction). Reliability: DPI rate's own **-0.0663** (158 pairs,
near zero, reported plainly per section 3). Classification:
`unresolved_below_power`, `closing_ground: null`.

### Cell C — `penalty_crew_holding_tilt_run_heavy`

Home team in the BOTTOM quartile of prior-rolling pregame pass rate
(run-heavy) AND the game's referee's PRIOR-season Offensive Holding rate in
the top quartile. Mechanism: a high-holding-calling crew is hypothesized to
disproportionately disrupt a run-heavy team's sustained run-blocking scheme,
hurting the home team's drive sustain — cover rate hypothesized SMALLER.
Sign: -1 (the reported effect below is already sign-oriented: positive =
supports the hypothesized direction, i.e. home cover WAS actually lower in
this subset).

**Measured**: n_flag=111 / n_total=4,786 (fraction_of_slate=2.32%).
Week-blocked (175 blocks): effect **+0.1390** accuracy points, 95%
**[-0.0595, +0.3415]**, se=0.1034, **P+ = 0.9015**. Reliability: Offensive
Holding rate's own **+0.3226** (158 pairs, real, moderate). Classification:
`unresolved_below_power`, `closing_ground: null`. **This is the standout
cell in the family** — highest P+, real underlying trait reliability, and
(section 5 below) the same direction holds in both eras.

### Cell D — `penalty_crew_flag_rate_high_total_line`

Home team's referee's PRIOR-season `mean_total` (existing trait, reused
unchanged) top quartile AND the game's own total (over/under) line in the top
quartile. Mechanism: a high-flag crew's extra stoppages are hypothesized to
matter most for tempo/possession control in a high-total (shootout-projected)
game; the home team, which controls the game plan at home, is hypothesized to
benefit more. Sign: +1. (Implemented as a boolean top-quartile x top-quartile
AND — `subset_bias` has no continuous z-score interaction term; see
`docs/experiment_pipeline.md`.)

**Measured**: n_flag=185 / n_total=4,786 (fraction_of_slate=3.87%).
Week-blocked (175 blocks): effect **+0.0326** accuracy points, 95%
**[-0.2586, +0.3263]**, se=0.1492, **P+ = 0.5698** (near coin flip).
Reliability: reused `mean_total` (+0.370, 158 pairs). Classification:
`unresolved_below_power`, `closing_ground: null`.

## 5. Per-era magnitudes (diagnostic, not separately recorded)

Per the project's "era magnitude, not presence" rule: a weaker-era reading is
never treated as absence. These are ad hoc season-window re-runs of the SAME
registered flag builders (not new named hypotheses, so not separately
recorded to the registry — the underlying named cell IS the registry entry;
this is a diagnostic breakdown of it), measured this session, week-blocked
primary interval:

| Cell | Era | n_flag | effect (pts) | 95% interval | P+ |
|---|---|---|---|---|---|
| A (opener, heavy underdog) | 2020-2022 | 9 | +0.1141 | [-0.1138, +0.3409] | 0.8364 |
| A (opener, heavy underdog) | 2023-2025 | 9 | +0.0982 | [-0.0982, +0.2941] | 0.8070 |
| C (holding tilt, run-heavy) | 2016-2020 | 74 | +0.2315 | [-0.1114, +0.5870] | 0.8953 |
| C (holding tilt, run-heavy) | 2021-2025 | 37 | +0.0596 | [-0.1659, +0.3017] | 0.6623 |
| D (flag rate, high total) | 2016-2020 | 162 | +0.1931 | [-0.4154, +0.8090] | 0.7268 |
| D (flag rate, high total) | 2021-2025 | 23 | -0.0989 | [-0.2720, +0.0827] | 0.1127 |
| B (DPI tilt, pass-heavy) | 2016-2020 | 33 | -0.0227 | [-0.2510, +0.2137] | 0.3804 |
| B (DPI tilt, pass-heavy) | 2021-2025 | 46 | -0.1197 | [-0.3928, +0.1572] | 0.1732 |

Read plainly, not summarized into a verdict: cell A holds a stable, positive
magnitude across both opener-era halves despite a thin n_flag of 9 per half
(genuinely underpowered, not evidence of absence). Cell C's direction is
consistent across both eras (P+ stays above 0.5 in both), though the later
era is measurably weaker (0.895 → 0.662) — a real magnitude change, not a
sign flip. Cell D's direction actually FLIPS between eras (P+ 0.727 in
2016-2020 vs 0.113 in 2021-2025); reported honestly rather than smoothed over
— this weakens confidence in cell D's own mechanism specifically, though per
`AGENTS.md` an era split flipping sign is a DIAGNOSTIC observation, not one of
the two admissible closing grounds (it is neither a whole-interval-below-zero
RESOLVED wrong sign at the full-window grade nor a positive-control bound),
so cell D stays `unresolved_below_power` at its full-window registered value,
not reclassified. Cell B opposes its own hypothesized direction in BOTH eras
(P+ 0.380 and 0.173) — consistent with its own near-zero DPI reliability
(section 3) — worth noting for any future re-visit of this cell specifically,
though a season-level warm-start of season-blocking (5 blocks per half,
flagged by the runner's own `BootstrapDegeneracyWarning`) means these
half-window intervals are themselves coarse and are reported as point
estimates + P+, not leaned on for interval precision.

## 6. Registry

All four cells recorded via `nfl-ats experiment run
registry/experiment_specs/penalty_crew_*.json` (four runs, non-dry-run).
**Measured**, confirmed via `nfl-ats weak-signals status` / direct registry
read after recording: all four names
(`penalty_crew_dpi_tilt_pass_heavy_favorite`,
`penalty_crew_holding_tilt_run_heavy`,
`penalty_crew_flag_rate_high_total_line`,
`penalty_crew_high_flag_heavy_underdog_opener`) present in
`registry/weak_signals.json` (301 total signals in the registry at time of
writing), each `classification: unresolved_below_power`,
`closing_ground: null` — no parallel-writer race observed (single-writer
filesystem lock, `experiment_runner._RegistryLock`).

## 7. Follow-up read

None of the four cells clears either admissible closing ground (no interval
sits entirely below zero at 1.099x-widened significance, and no positive
control was run against this family this session) — all four correctly stay
`unresolved_below_power`. Per `AGENTS.md`, a 0.90 P+ bar is a promotion
threshold for what the docs may CLAIM, never a decision bar for what gets
played; a candidate is worth a forward-looking EV read regardless. Two cells
are the standouts for that read: **cell C (holding-tilt/run-heavy, P+
0.9015, built on a real +0.3226 split-half trait, consistent-direction
across both eras)** and **cell A (opener heavy-underdog, P+ 0.9204, stable
magnitude across both eras despite n_flag=9/era)** — both are plausible
candidates for a future pooled read alongside the existing referee-battery
cells (same commensurable unit, `accuracy_points`, same population family)
rather than a standalone promotion decision. Cell D's cross-era sign flip and
cell B's own near-zero underlying reliability make them the weaker two of
the four, though neither is refuted (no whole-interval-below-zero result at
the full-window grade) and both remain correctly `unresolved_below_power`,
not closed.

## 8. Files

- `scripts/fetch_penalty_type_snapshot.py` — the data-widening fetch/verify script.
- `data/raw/officials/20260820T112517Z/game_penalty_types.parquet` +
  `manifest_penalty_types.json` — the new supplemental snapshot (gitignored,
  local only).
- `src/nfl_ats/experiment_runner.py` — four new flag builders
  (`_flag_referee_high_flag_heavy_underdog`,
  `_flag_referee_dpi_tilt_pass_heavy_favorite`,
  `_flag_referee_holding_tilt_run_heavy`,
  `_flag_referee_flag_rate_high_total_line`), the shared
  `_build_referee_type_trait_data`/`_RefereeTypeTraitData` per-type trait
  builder, and the `_merge_home_pass_rate_quartile`/`_merge_total_line_quartile`
  join helpers.
- `registry/experiment_specs/penalty_crew_*.json` — four predeclared specs.
- `tests/test_experiment_runner.py` —
  `test_referee_type_trait_uses_the_prior_season_lag` and
  `test_referee_type_trait_does_not_use_this_games_own_penalty_type_count`
  (the required leakage regression test for this new feature family), plus
  the updated `test_flag_builders_registry_has_the_documented_names`.
