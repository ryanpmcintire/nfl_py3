# Unserved tilt adjustments, measured on top of the played card

## The question

The played card was chosen by `nfl_ats.overlay_composition`, which enumerates
only SEVEN flip members (`OVERLAY_NAMES` plus the player-arrest policy). Roughly
twenty other tilt overlays are registered as `ACTIVE_PROSPECTIVE` challengers in
`artifacts/prospective/challengers.json`, several with standalone
`probability_positive` above 0.8, and none of them had ever been scored as a
**marginal on top of the card that is actually played**. A standalone number on
the bare model says nothing about whether an adjustment still adds anything once
the three served rules have already moved the picks they move.

This measures all fifteen of them the same way, at the opener grade, on the
frozen 2020-2025 archive.

## How it was measured

- Baseline: the served three-member joint-OR card (`coach_fade` OR
  `division_revenge_tilt` OR `player_arrests_back_side_policy`) applied to the
  opener archive `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
  (active model `c657058903f3232b`, `gaussian_median`). The served flip set is
  279 games; it was verified to be byte-identical whether built through
  `overlay_composition.reconstruct_arrest_flip_set` or through the production
  `apply_player_arrests_back_side_overlay`.
- Population: 1,537 archive games, 34 opener pushes, **1,503 scored games**,
  107 week blocks, 6 season blocks.
- Combination rule: joint OR against the raw model card, complemented exactly
  once — identical to `nfl_ats.four_overlay_composition`. A member's marginal is
  therefore (played union OR that member) minus the played union: a flip the card
  already makes contributes nothing.
- Uncertainty: the same `build_delta_matrix` / `blocked_bootstrap_matrix`
  machinery the composition study uses, 20,000 samples, seed 20260821, both the
  week-blocked and season-blocked bootstrap. Within-week game correlation is zero
  by owner mandate; no ICC is estimated or padded.
- Raw model accuracy on the same 1,503 games: **54.558%**. Served card:
  **55.888%**.
- Runner: `scripts/unserved_tilt_marginals.py`; importable functions in
  `nfl_ats.unserved_tilt_marginals`. Artifact:
  `artifacts/unserved_tilt_marginals/20260909T210338Z/`.

## The table

Sorted by week-blocked `probability_positive`. "Flips" is the member's own flip
count on the archive; "Already on card" is how many of those the served card
already flips; "New flips" is what is left.

| Adjustment | Flips | Already on card | New flips | Card % | With it % | Delta (pts) | Week 95% | P+ (week) | P+ (season) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| interim_hc_first_game_tilt_overlay | 6 | 4 | 2 | 55.89 | 56.02 | +0.133 | [+0.000, +0.336] | 0.9326 | 0.9555 |
| bye_edge_fade_overlay | 73 | 9 | 64 | 55.89 | 56.42 | +0.532 | [-0.394, +1.453] | 0.8710 | 0.9097 |
| forecast_cold_visitor_tilt_overlay | 57 | 13 | 44 | 55.89 | 56.29 | +0.399 | [-0.466, +1.277] | 0.8185 | 0.9110 |
| pbp08_protection_mismatch_tilt_overlay | 111 | 23 | 88 | 55.89 | 56.29 | +0.399 | [-0.665, +1.457] | 0.7714 | 0.7411 |
| forecast_weather_kn_precip_high_total_tilt_overlay | 15 | 4 | 11 | 55.89 | 55.95 | +0.067 | [-0.332, +0.467] | 0.6319 | 0.7160 |
| tank_zone_fade_tilt_overlay | 17 | 3 | 14 | 55.89 | 55.95 | +0.067 | [-0.331, +0.468] | 0.6280 | 0.6190 |
| spread_gap_zone_fade_overlay | 195 | 42 | 153 | 55.89 | 55.49 | -0.399 | [-1.995, +1.276] | 0.3149 | 0.3317 |
| third_down_reversion_fade_overlay | 312 | 49 | 263 | 55.89 | 55.29 | -0.599 | [-2.908, +1.674] | 0.3018 | 0.2245 |
| pace_mismatch_dog_tilt_overlay | 177 | 39 | 138 | 55.89 | 55.29 | -0.599 | [-2.133, +0.941] | 0.2232 | 0.0665 |
| forecast_weather_kn_warm_team_cold_late_tilt_overlay | 2 | 1 | 1 | 55.89 | 55.82 | -0.067 | [-0.202, +0.000] | 0.1837 | 0.1698 |
| special_teams_return_tilt_overlay | 275 | 50 | 225 | 55.89 | 54.89 | -0.998 | [-2.908, +0.932] | 0.1535 | 0.0050 |
| surface_switch_tilt_overlay | 225 | 42 | 183 | 55.89 | 54.69 | -1.198 | [-2.846, +0.401] | 0.0748 | 0.0958 |
| injury_value_tilt_overlay | 711 | 128 | 583 | 55.89 | 52.36 | -3.526 | [-6.922, +0.000] | 0.0243 | 0.0000 |
| backup_qb_fade_overlay | 174 | 30 | 144 | 55.89 | 54.22 | -1.663 | [-3.362, -0.067] | 0.0208 | 0.0114 |
| turnover_luck_rebound_tilt_overlay | 305 | 63 | 242 | 55.89 | 53.09 | -2.794 | [-4.633, -0.870] | 0.0023 | 0.0003 |

Four of the fifteen (`injury_value_tilt_overlay`, `surface_switch_tilt_overlay`,
`backup_qb_fade_overlay`, `spread_gap_zone_fade_overlay`) were already inside the
seven-member enumeration; they are reported here on the same footing so the table
is complete. `third_down_reversion_fade_overlay` and the other ten were never
enumerated at all.

Every row is recorded to the weak-signal registry as
`unserved_tilt_on_played_card_<member>`, classification
`unresolved_below_power`, family `unserved_tilt_marginal_on_played_card`. None of
them closes: no wrong sign is RESOLVED (every interval that sits below zero does
so on a member whose own mechanism was measured positive on the bare model, which
is an interaction with the served card rather than a refuted mechanism), and no
positive control was proven able to detect an effect this size and found it
absent. An interval containing zero is not a reason to reject any of them.

## Coverage gaps worth naming

- `forecast_weather_kn_warm_team_cold_late_tilt_overlay` fires on **2 archive
  games total**, both in 2025; 2020-2024 are structurally NO-OP. Its conditions
  (warm-metro visitor AND outdoors AND kickoff-nearest forecast at or below 35F
  AND week 13+) simply almost never co-occur. Its -0.067 is one game.
- `forecast_weather_kn_precip_high_total_tilt_overlay` has no 2025 flip. Its
  `total_line` here comes from the nflverse schedule snapshot (a settled line),
  not the point-in-time Tuesday total the live overlay reads off the card — the
  one place this reconstruction is not decision-time faithful.
- `interim_hc_first_game_tilt_overlay` has no 2023 flip and only 6 flips overall,
  4 of which the served card already makes. Its narrow interval is narrow because
  it moves almost nothing, not because it is precise.
- Forecast archives: `forecast_cold_visitor_tilt_overlay` was scored against the
  tuesday-noon archive `data/raw/forecast_archive/full_2020_2025` (MOS model MEX,
  its own registered cutoff); both kickoff-nearest weather tilts against
  `data/raw/forecast_archive/kickoff_nearest_2009_2025` (MOS model GFS).
- `pbp08_protection_mismatch_tilt_overlay`'s flag table was built once over the
  full REG history from 2009 and filtered to the archive. Its quartile thresholds
  expand over strictly-earlier week blocks, so a one-shot build is identical to
  the per-week `flags_for_week_fail_open` path — verified on 2020 wk5, 2022 wk12,
  2024 wk10 and 2025 wk3, all four identical on `back_side`.

## Adding them together

Six members have week-blocked `probability_positive` above 0.5. Adding all six to
the played card at once:

| Combination | New flips | Card % | With them % | Delta (pts) | Week 95% | P+ (week) | P+ (season) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| all six positive members | 208 | 55.888 | 56.886 | +0.998 | [-0.662, +2.656] | 0.8827 | 0.9404 |

Recorded as `unserved_tilt_on_played_card_all_probability_positive_union`.

### Greedy forward ordering (attribution only)

Post-hoc attribution on the same games the members were scored on. It proposes an
ordering; it confirms nothing, and it is deliberately NOT recorded as registry
cells, because a selected maximum is not commensurable with the per-member rows.

| Step | Added | Cumulative delta (pts) | Cumulative % |
| ---: | --- | ---: | ---: |
| 1 | bye_edge_fade_overlay | +0.532 | 56.421 |
| 2 | forecast_cold_visitor_tilt_overlay | +0.798 | 56.687 |
| 3 | pbp08_protection_mismatch_tilt_overlay | +1.064 | 56.953 |
| 4 | interim_hc_first_game_tilt_overlay | +1.198 | 57.086 |
| 5 | forecast_weather_kn_warm_team_cold_late_tilt_overlay | +1.198 | 57.086 |
| 6 | forecast_weather_kn_precip_high_total_tilt_overlay | +1.131 | 57.019 |
| 7 | tank_zone_fade_tilt_overlay | +0.998 | 56.886 |
| 8 | spread_gap_zone_fade_overlay | +0.466 | 56.354 |

Adding anything past step 8 takes the card below the three-member baseline; by
step 15 the full stack is -8.98 points. Stacking every registered tilt is not the
same experiment as stacking the ones that earn it.

## The decision

The pool is forced picks: a card is submitted either way. All six members with
`probability_positive` above 0.5 are served. A predeclared threshold like 0.90
governs what this document may CLAIM, never which card is played, and declining a
member that is 88% likely to help is taking the other side of an 88/12 bet.

The played policy is now
`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`
(fingerprint `4bc0cc7724ac0f82f5ce2dfac44a2485b486e4ff15e63af593f2d16fce956b35`),
a nine-member joint OR. The previous three-member union is retained as a paired
prospective challenger, `overlay_three_member_union_retired_20260909`,
reconstructed each week from the primary ledger's own member flip columns —
exactly the way `overlay_four_member_union_retired_20260907` was kept when the
spread-gap fade was retired.

### Fail-open posture of the six added members

The three original members keep their existing contracts: the arrest policy fails
CLOSED and `card_view` still refuses to publish if any of the three is disabled.
Every one of the six added members documents in its own module that it must never
be able to block a publish, so each loads through its own fail-open helper and a
missing snapshot or a failed forecast fetch reads as zero flips, with the reason
recorded on that member's provenance row. `FAIL_CLOSED_MEMBERS` in
`nfl_ats.four_overlay_composition` is what draws that line.

Because only three of the nine members have their own column in the paper-decision
ledger, the ledger invariant for the new policy checks that the composed flip
matches the recorded side change and that every recorded member flip implies the
composed flip, rather than requiring equality with an OR of columns that do not
exist. Per-member provenance for the six added members lives on the composition
result and in the card lineage.

## Week 1 2026 effect

Regenerated with the same command `weekly-run` issues
(`nfl-ats margin-predict --season 2026 --week 1 --features
data/processed/game_features_weak_stack.parquet --feature-profile weak_stack
--probability-method gaussian_median`), then composed under both policies.
All nine members applied; none was disabled.

Three of sixteen picks change, all three from the protection-mismatch rule:

| Game | Spread | Model | Old card | New card |
| --- | ---: | --- | --- | --- |
| ATL at PIT | +3.5 | ATL | ATL | PIT |
| DEN at KC | +2.5 | KC | KC | DEN |
| NO at DET | +6.5 | DET | DET | NO |

The other five added members are structurally silent in Week 1: no team is off a
bye yet, the tank-zone rule is weeks 14-18 only, the warm-team cold-late rule is
week 13+, no game carries a 25F climate-to-forecast temperature gap in early
September, and no interim head coach is in his first game.
