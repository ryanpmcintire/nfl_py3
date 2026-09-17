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

## Open
- None yet.
