# pooled-signal-model

## Goal
MOD-20: one hierarchical model with every situational family as a shrunk term, judged as a whole on line movement toward the pick and calibration at the opener; no cell resolved alone.

## State
- 2026-09-16: queued by owner; ROADMAP row written.
- Unit 1 inventory DONE 2026-09-16 (read-only grep/read): only 9 composition
  flag columns are reproducibly computable point-in-time at the opener, via
  `pick_probability.py:signed_composition_flags` (7 counted: coach, division,
  arrests, bye, cold_visitor, protection, tank_zone; interim-HC and precip
  hardcoded 0, owner-held). Registry has 6,582 signals / 198 family keys;
  ~189 families have no reproducible column. The first pooled model on the
  reproducible set is the four-term pick-probability fit, refit today on the
  active model: `artifacts/pick_probability/20260916T163620Z/`
  (pointer `artifacts/active_pick_probability.json`), LOSO 841-662 vs
  model-only 819-684 on the same 1,503 games. Nothing committed.

## Tried
- Subagent inventory (verified): margin base is `margin.py` weak_stack on
  `game_features.parquet` via `features.py:build_game_features`; the
  pick-probability fit uses opener `per_game.parquet` + flags, not the game
  features table.

## Next
- Unit 1 CLOSED 2026-09-16: paired interval computed
  (`scripts/pooled_signal_paired_eval.py`,
  `artifacts/pooled_signal/20260916T191645Z/`), +1.464 pts
  [+0.130, +2.660], P+ 0.9815, decisive 274-252 on 526, positive in 5
  of 6 seasons; recorded as the family's first cell
  (`pooled_signal_calibrated_vs_model_only_loso`,
   unresolved_below_power). Next: grow the reproducible feature set
   before any second pooled fit. Committed in 786a569.
- Unit 2 inventory DONE 2026-09-17 (subagent, read-only, 82 families
  inspected): 15 candidate families for reproducible point-in-time opener
  columns — 9 easy (spread_size_calibration, key_number_seven,
  ats_streak_regress_on_production, post_ot_fatigue_on_production,
  division_dog_on_production, low_total_div_home_dog_on_production,
  roof_state, sept_heat_home_on_production, altitude_fourth_quarter),
  5 medium (open_corner_wind_dog, snow_game_home_prep,
  ol_rush_continuity, rookie_qb_debut_fade, backup_tenure_gap),
  1 hard (officials_archive_battery — assignment timestamps missing).
  Never-at-opener families listed by class (post-open moves, T-90
  inactives, in-game, wrong population, process meta). No cells scored,
  no looks spent. Next: build the easy columns, then the second pooled
  fit.
- Unit 2 DONE 2026-09-18 (subagent, parent-verified by rerun;
  `scripts/pooled_signal_second_fit.py`, `tests/scratch/pooled_signal_second_fit.json`):
  8 of 9 easy columns assembled from existing builders (roof_state dropped,
  needs new snapshot history); joint ridge fit LOSO, same population and
  bootstrap as unit 1. Full vs model-only +1.33 [-2.00, +4.54], P+ 0.788,
  decisive 300-280 on 580; full vs four-term base -0.13 [-1.99, +1.74],
  P+ 0.429, decisive 101-103 on 204. In-sample vs OOS gap +2.06 for full
  vs +0.80 for base; OOS Brier/log loss favor base too. Reading: growing
  the pool this way adds nothing out of season — the extra terms overfit.
  Both cells recorded in family `pooled_signal_first_model_v1`,
  unresolved_below_power (registry now 6,710; SEs approximated from
  interval width, noted). 9 looks. Next: medium columns or a different
  pooling structure; the easy-column direction is spent.
- Unit 3 DONE 2026-09-18 (subagent, parent-verified by rerun;
  `scripts/pooled_signal_third_fit.py`, `tests/scratch/pooled_signal_third_fit.json`):
  only 1 of 5 mediums faithfully buildable (Tuesday wind-dog flag, frozen
  venue list, Tue wind ≥ 15, 112 games; snow needs Tuesday precip that does
  not exist; rookie/backup depth ungateable pre-Tuesday; continuity has no
  pick direction). Full vs four-term base -0.33 [-0.74, +0.07], P+ 0.039,
  decisive 4-9 on 13 — reaches above zero, so not a resolved wrong sign;
  full vs model-only +1.13 [-1.57, +3.68], P+ 0.800, decisive 272-255 on
  527. Both cells recorded in family `pooled_signal_first_model_v1`,
  unresolved_below_power (registry now 6,712). 5 looks. Standing read after
  three fits: the four-term pool is not improved by growing it; the next
  change of direction is structural (hierarchical shrinkage, interactions)
  or nothing.

## Open
- None yet.
