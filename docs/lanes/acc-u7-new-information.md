# ACC U7: new information inventory (parent accuracy-ceiling-theory.md)

## Goal
Rank information the Tuesday opener lacks, untested, free, available by pick deadline.

## State
2026-10-01 inventory done (read-only, grep counts over ROADMAP.md, docs/*.md, registry/weak_signals.json families). No measurements yet.
Survivors: (1) availability-matched history: rolling team inputs re-weighted by own and opponent starter availability on each past game day (grep "opponent availability rolling" 0/0/0; near: expected_lineup_loss, mod23_unit3 opponent_adjusted, qb_exit_mixture - those fix the current game or season quality, not past-game lineups). (2) non-offensive scoring and starting-field-position luck stripped from margin/EPA history (grep pick-six/return TD 0 ROADMAP, 10 docs, 0 registry; field position 0/46/0; near: special_teams_battery, close_game_luck_turnover, sim04). (3) mid-season acquisitions re-weighting (traded 1/11/0; near transaction_wire_battery). (4) Wikipedia player pageviews Wed-Sat as lineup-news proxy (0 registry; docs hits are source scouts, verify; near gdelt_backfill, reddit, tv_attention).
Discards: clinch/eliminated -> docs/motivation_ladder_screen.md, tank_zone_fade_tilt; playcaller -> playcaller_change*, lead29, per07; snap counts/OL continuity -> players_on_field_rating, ol_crosswalk_fix, pbp08_protection_mismatch; kicker -> special_teams_battery, schedule_flag_battery, kicker_change_underdog; wind/weather -> forecast_weather, weak_stack_v4; short week/travel -> travel_rest_battery, schedule_flag_battery, LEAD-81; referee -> officials_archive_battery, penalty_crew_tendencies; practice/injury text -> transaction_wire_battery, injury_*, coach_speak; backup QB -> qb_exit_mixture, backup_tenure_gap; WP-band filter -> docs/modeling.md; contract year -> only pool_edge_plan mention, no mechanism.

## Tried
Nothing measured.

## Next
Unit 7a: scripts/ only, paired candidate-vs-base grade on the 2011-2025 population via the U1 harness; log with weak-signals record.

## Open
Does the 88-input set already carry availability-conditioned stats? Check feature manifest before 7a.

## Orchestrator check 2026-10-01
Survivor 4 (Wikipedia pageviews) is discarded: the team-level pageview attention
family was already screened (attention_battery_*, docs/attention_followup.md).
Units launched: 7a availability-matched history (acc-u7a), 7b luck-stripped
inputs (acc-u7b), 7c mid-season acquisitions (acc-u7c).
