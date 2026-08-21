# Combined weak-signal stacker: ONE predeclared confirmation look

Written 2026-08-21. **Design only.** Nothing has been declared, assigned, or
spent in `registry/rotation_registry.json`; neither registry JSON was modified
today. Every decision below is stated before any future run and is mechanically
derivable from `registry/weak_signals.json` and
`registry/rotation_registry.json` as they stood on 2026-08-21. If execution
deviates from any frozen choice below, the deviation voids the look.

This document discharges the obligation created by `docs/pool_edge_plan.md`
(guard 3): *"A pooled estimate is not a finding. It is grounds for building ONE
combined candidate and confirming it, predeclared, on a rotation window none of
the inputs touched."*

Provenance convention: numbers quoted below carry their source inline
(**measured** = computed this session from the named registry/command;
**read** = opened file cited; **reported** = doc claim not re-verified).

## 1. Input selection rule (frozen 2026-08-21, applied in this order)

**Step A — registry filter (all conditions required):**

1. `league == "nfl"`
2. `effect_units == "accuracy_points"`
3. `classification == "unresolved_below_power"` (only category 3 is poolable)
4. `probability_positive >= 0.85`
5. `recorded_at < 2026-08-21T00:00:00Z`. The registry field is `recorded_at`
   (date or ISO timestamp, no `_utc` variant exists — verified **measured**
   this session); comparison is string-wise against `"2026-08-21"`, which is
   equivalent for both formats present. Pre-today cutoff so nothing recorded
   on the day of freezing self-selects.

Applied **measured** this session via a read-only Python scan of
`registry/weak_signals.json` (341 signals): **55 entries pass Step A**.

**Step B — admissibility as a pregame model column (named exclusions,
frozen).** A selected entry is dropped if it is not expressible as a
point-in-time-safe, per-game feature column. Dropped, individually named:

- `era_weighting_nfl_half_life_8` — training sample-weight recipe, not a column.
- `mod08_smooth_cdf_mapping` — output calibration layer (rescale class), not a
  game-level column.
- `movement_attribution_pop_threshold_attributed_any`,
  `movement_attribution_pop_threshold_injury`,
  `movement_attribution_pop_unfiltered_attributed_any`,
  `movement_attribution_pop_unfiltered_injury` — attribute realized post-opener
  line movement; not available at the Tuesday-opener information cutoff this
  stacker operates under.
- `observed_movement_oracle_full_slate`, `observed_movement_threshold_0_5`,
  `observed_movement_threshold_1_0`,
  `observed_movement_threshold_1_0_sunday_am_realism` — perfect-foresight /
  realized-movement diagnostics (oracle constructions cannot be features).
- `odds_microstructure_H3_3_0a_full_week_oracle_2020_2025_sanity_check`,
  `odds_microstructure_H3_3_0b_full_week_oracle_2023_2025_baseline`,
  `odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025` — intraweek line
  movement oracles, same reason.
- `opener_error_mining_movement_agreement_agrees_corrected`,
  `opener_error_mining_movement_agreement_disagrees`,
  `opener_error_mining_movement_agreement_disagrees_overlay_paired_delta`,
  `opener_error_mining_movement_agreement_disagrees_overlay_paired_delta_move_ge_1_0`
  — condition on realized movement relative to the model pick (post-opener).
- `player_family_base_vs_continuity`, `player_family_base_vs_qb_continuity`,
  `player_family_base_vs_value` — ablation contrasts (block-removal deltas),
  not additive per-game signals.
- `sbr_opener_era_2020_2021` — an evaluator-population diagnostic (production
  model rescored against SBR openers), carries no feature.

After Step B: **34 entries remain** (**measured**).

**Step C — window-first freshness constraint.** The target window is chosen
FIRST (Section 4): `[2022, 2023]`, the earliest unspent opener-grade block in
`nfl-ats rotation status` as of 2026-08-21 (**measured**: unspent opener blocks
are `[2022,2023]` and `[2024,2025]`). An entry survives only if its `seasons`
interval is disjoint from `{2022, 2023}`. This makes pool_edge_plan's defining
guarantee — *a window none of the inputs touched* — exactly true for every
retained input. The honest cost, stated up front: the union of Step-A seasons
covers all of the opener pool's 2020–2025 range (many entries span 2009–2025),
so no strictly-fresh window coexists with the full 55-entry pile; the large
multi-season batteries are therefore excluded from THIS candidate, and their
only untouched historical test would be prospective 2026. After Step C: **8
entries remain** (**measured**): the five `injury_value_lost_*` entries
(2020–2021), the two `*_pre2020` weather reruns (2009–2019), and
`pick_conditioned_spread_gap_zone_pre2018` (2011–2017).

**Step D — correlated-decomposition dedup: one entry per family.** Families
are defined by leading name prefixes, frozen here:
`injury_value_lost`, `forecast_weather_kn_temp_gap_cold_visitor`,
`forecast_weather_kn_warm_team_cold_late`, `pick_conditioned_spread_gap_zone`.
(The `weather_battery_*` / `weather_followup_*` siblings of the two weather
families failed Step C and cannot represent them.) Within a family the
survivor is, in order: (1) largest `sample_games`; (2) lexicographically
smallest name. Applied to the 8 survivors (**measured**): the five injury
entries all sit on 456 games, so the lex tie-break selects
`injury_value_lost_gradient`; each weather family retains its sole eligible
member. Result: **4 families / 4 selected inputs**.

### Frozen appendix: the 8 Step-C survivors

All fields **measured** from `registry/weak_signals.json` this session.

| name | family | survivor | P+ | effect (pts) | 95% interval | games | blocks | seasons | recorded_at |
|---|---|---|---|---|---|---|---|---|---|
| `injury_value_lost_gradient` | `injury_value_lost` | **yes** | 0.8990 | +1.750 | [-0.69, +4.27] | 456 | 35 | 2020–2021 | 2026-08-18 |
| `injury_value_lost_narrowed` | `injury_value_lost` | no | 0.8875 | +1.316 | [-0.46, +3.25] | 456 | 35 | 2020–2021 | 2026-08-18 |
| `injury_value_lost_prior_week_absence_saturday_channel` | `injury_value_lost` | no | 0.8859 | +1.535 | [-0.87, +3.86] | 456 | 35 | 2020–2021 | 2026-08-20 |
| `injury_value_lost_tuesday_saturday_channel_official_only` | `injury_value_lost` | no | 0.9003 | +1.316 | [-0.45, +3.18] | 456 | 35 | 2020–2021 | 2026-08-19 |
| `injury_value_lost_tuesday_saturday_channel_pft_augmented` | `injury_value_lost` | no | 0.9173 | +1.535 | [-0.45, +3.67] | 456 | 35 | 2020–2021 | 2026-08-19 |
| `forecast_weather_kn_temp_gap_cold_visitor_pre2020` | `temp_gap_cold_visitor` | **yes** | 0.8707 | +0.300 | [-0.23, +0.82] | 2,735 | 187 | 2009–2019 | 2026-08-20 |
| `forecast_weather_kn_warm_team_cold_late_pre2020` | `warm_team_cold_late` | **yes** | 0.9848 | +0.229 | [+0.02, +0.42] | 2,735 | 187 | 2009–2019 | 2026-08-20 |
| `pick_conditioned_spread_gap_zone_pre2018` | `spread_gap_zone` | **yes** | 0.9136 | +4.756 | [-2.16, +11.70] | 1,743 | 119 | 2011–2017 | 2026-08-19 |

Split-half `reliability`, where present (**read**): `gradient` 0.9325,
`narrowed` 0.87; all other survivors report none.

**Selected-input count: 4** (`injury_value_lost` represented by `gradient`;
`temp_gap_cold_visitor` by `..._pre2020`; `warm_team_cold_late` by
`..._pre2020`; `spread_gap_zone` by `..._pre2018`).

Disclosure on the injury representative: the mechanical tie-break lands on
`injury_value_lost_gradient`, whose own description attributes the effect to
value-lost *magnitude* rather than availability rate, and whose availability
confound `injury_value_lost_narrowed` removes (**read**:
`docs/pool_edge_plan.md` lead 3). The column built in Section 2 therefore
implements the narrowed, confound-free magnitude construction; the registry
representative identifies the family, the build defines the column. Stated
now so nothing is chosen after results are seen.

## 2. Build recipe (MOD-07 weak_stack precedent, unchanged machinery)

Baseline arm = current production configuration (**read**:
`artifacts/active_ats_model.json`, `HANDOFF.md` this session):
`feature_profile=weak_stack`, `target=market_residual`, `regressor=ridge`,
`ridge_alpha=10.0`, calibration none, Gaussian probability method,
expanding walk-forward with `min_train_games=500`,
`distribution_fraction=0.20`, training strictly prior.

Candidate arm = identical in every respect except that four columns are
appended to the design matrix (weak_stack was 90 columns; it becomes 94):

1. `ivl_home_top_tertile`, `ivl_away_top_tertile` (two columns) — pregame
   injury-value-lost magnitude, narrowed fixed-prior-severity construction
   from `game_features_player_value.parquet`, Saturday-decision-cutoff
   (pre-kickoff) information only; each side flagged when its value lost sits
   in the top tercile, with tercile boundaries computed from training-period
   games only.
2. `kn_temp_gap_cold_visitor_pre2020` (one column) — binary: outdoor game,
   away team's climatological-normal outdoor home temperature minus the
   Tuesday-noon kickoff-nearest forecast temperature >= 25 F.
3. `warm_team_cold_late_pre2020` (one column) — binary: away team in the
   static warm-winter-metro list, outdoor, Tuesday-noon forecast temperature
   <= 35 F, week >= 13.
4. `spread_gap_zone` (one column) — binary: `7.0 < abs(spread_line) <= 10.0`.

Thresholds, lists, and cutoffs are inherited verbatim from the registry
entries' own descriptions and are never retuned. Every new column ships with a
leakage regression test proving it is computable from pre-prediction-timestamp
information only (AGENTS.md requirement); a failing leakage test aborts the
look (Section 7).

Forced picks use the production probability rule
`home_cover_probability >= 0.5` in both arms (**read**: `src/nfl_ats/pool.py:41`,
`src/nfl_ats/backtest.py:56`) — not a residual-sign rule.

## 3. Evaluation protocol

Paired candidate-vs-production-baseline comparison on every 2022–2023 REG
game with paired Tuesday-opener coverage (expected roughly 450–520 paired
games by analogy with the 456-game 2020–2021 opener windows; **inferred**, not
measured).

- **Primary:** paired forced-pick accuracy delta (candidate minus baseline) at
  the OPENER grade, week-blocked bootstrap (resampling whole weeks; harness
  settings identical to the MOD-07/SPEC-4 opener evaluations). Reports point
  estimate, 95% interval, and `probability_positive`.
- **Secondary (context only):** season-blocked bootstrap. Two blocks is
  degenerate by construction; it informs direction, never gates anything.
- **Direction-only secondaries:** Brier and log-loss improvements, reported
  with intervals and P+, no gate attached.
- Absolute accuracies for both arms are reported alongside every delta.

## 4. Window choice

**Measured** from `nfl-ats rotation status` this session: the opener-grade
pool spans 2020–2025 with unspent blocks `[2022, 2023]` and `[2024, 2025]`.
Strict subtraction of the selected inputs' seasons from the pool leaves
`[2022, 2023]` fully intact for all four inputs (seasons touched:
2020–2021, 2009–2019, 2009–2019, 2011–2017 — disjoint by construction, which
is what Step C enforced). **This predeclaration claims `[2022, 2023]`** — the
earliest eligible unspent opener block, per house convention — for the new
combined-stacker family. Execution begins by declaring that family and
assigning `[2022, 2023]` through the rotation registry; this document spends
nothing.

Disclosed discounts on the claimed window:

- It intersects the mined 2018–2025 ledger (~130–150 looks; ROADMAP RWB-16),
  so the family declares `acknowledges_mined_2018_2025=true`. The specific
  selection-effect contamination the freshness rule guards against is absent
  (no input was measured on these seasons — **verified** from registry
  `seasons` fields); the residual discount is generic ledger multiplicity.
- Cross-family reuse context: `[2020, 2021]` is heavily spent; `[2022, 2023]`
  currently carries no spent opener window in any family (**measured**).

## 5. Decision rule (both numbers explicit)

Per AGENTS.md: a promotion bar is not a decision bar; thresholds govern what
docs may CLAIM, never which card is PLAYED.

- **Play decision (expected value, graded at the opener):** the pool card uses
  whichever arm has the higher paired opener forced-pick accuracy point
  estimate on this window. Decision gate: `sign(delta_hat)` — play the
  candidate iff `delta_hat > 0`. Intervals, P+, and claim language never veto
  this; the opener is the grade the pool settles on.
- **Claim gate (what write-ups may say):** the candidate may be described as
  *confirmed at the predeclared bar* only if primary `probability_positive >=
  0.90` (house standard, matching SPEC-4/MOD-07). Below 0.90 the only
  admissible verdict is `unresolved_below_power`, reported with its P+.
- **Closing taxonomy (binding):** an interval crossing zero is never grounds
  to reject or close. The only terminal classifications available here are
  `refuted_mechanism` (whole week-blocked interval below zero ->
  `wrong_sign_resolved`) or a positive-control bound (none is planned, so none
  is available). Everything else records as `unresolved_below_power` through
  the rotation-spend / `weak-signals record` commands, never through prose.

## 6. Costs, multiplicity, and overlap disclosures

- **This is one look of a long program.** Selection drew on 55 qualifying
  entries accumulated across dozens of screens; the candidate gets exactly one
  fresh-window confirmation, and a negative here closes nothing else.
- **The pooled estimate is not evidence.** The pool's random-effects number
  motivates building this candidate; it says nothing about whether the
  combination works. That is what this look tests.
- **Overlap warnings among inputs.** The five injury entries are decompositions
  of the same 456 games — their P+ values are correlated reads, not
  independent votes (this is why Step D collapses them to one column). The two
  weather entries come from the same 2,735-game 2009–2019 rerun and describe
  related cold-weather cells; inside the model they will co-fire on
  cold late-season outdoor games, and ridge alpha 10 will shrink both rather
  than double-count — accepted, not tuned around.
- **Small-stack caveat:** only four inputs survive the freshness constraint.
  The multi-season attention/weather/bias/penalty batteries that dominate the
  pool's mass are absent because they touch every historical season; a null
  here reflects on this four-column candidate only, not on those families
  (all remain category 3, unresolved).
- **Era extrapolation:** two inputs are era-restricted constructions
  (pre-2020, pre-2018) being scored on 2022–2023 games; if the effects are
  era-bound, attenuation is expected. Regression of selected effects on fresh
  windows is the documented norm (MOD-07: +1.97 on 456 discovery games became
  the +0.33 promoted opener claim — **read**: `docs/pool_edge_plan.md`).
- **Selection inflation:** inputs were admitted at P+ >= 0.85 partly because
  they looked good on their own windows; winner's-curse shrinkage on the fresh
  window is expected and is the reason the claim gate (0.90) sits well above
  the selection floor (0.85).

## 7. What would make us NOT run this

Any of the following voids the look before it starts; none permits changing a
frozen choice mid-flight:

1. **Prospective 2026 results arrive first** for any input family. New
   evidence changes the selection basis; re-derive Steps A–D and re-freeze a
   new predeclaration rather than running this one stale.
2. **`[2022, 2023]` becomes unavailable** (assigned or spent by another family
   before this candidate declares). Fall back to `[2024, 2025]` only if the
   freshly re-derived input set is still disjoint from it; otherwise abort.
3. **A retained input is reclassified** (`refuted_mechanism` or
   `bounded_by_control`) before the run: drop that family, re-run Steps A–D
   mechanically, re-freeze before running.
4. **A leakage regression test fails** for any new column, or required source
   data for 2022–2023 proves unavailable or not point-in-time-safe: abort
   without scoring; repair and redeclare.
5. **Insufficient paired coverage:** fewer than 400 paired 2022–2023
   opener-graded games are constructible. Below that the look is underpowered
   even for its own purpose; defer rather than run a token confirmation.
