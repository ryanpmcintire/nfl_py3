# Week 3 research recorder coverage

## Goal

Repair missing prospective challenger records without changing the completed served Week 3 decisions or weakening discrete probability and registry gates.

## State

Measured: `scripts/lockday_verify.py --season 2026 --week 3 --run-summary artifacts/scheduled_locks/2026-week-03/weekly_summary.json --json` exits 1: 62 active entries, 42 recorded, 11 skipped, nine missing, and no pending wiring. Evidence: `.tmp/backlog-resume/lockday-verify.json`. The served paper card has 16 unique game decisions and one Best Pick.

## Tried

Missing: `consensus_movement_1_0_off_incumbent`, `ecdf_mapping_incumbent`, `era_weighted_half_life_8`, `gaussian_mean_mapping_incumbent`, `late_week_follow_no_news_veto_off_incumbent`, `late_week_follow_no_sunday_blackout`, `late_week_leader_median_follow_0_5_off_incumbent`, `late_week_leader_median_follow_flat_1_0_off_incumbent`, and `tiebreaker_low_side_shade`. Three refit comparisons could not reproduce the served discrete probability; the others had no record or admissible skip gate.

## Next

Read the audit's reasons and owning recorder functions. Repair like-for-like discrete incumbent reconstruction first, then retry only affected prospective recorders where chronology still permits. Preserve decision timestamps and never backdate records. Re-run the named audit once after repair.

## Open

These are unresolved recording gaps, not negative research results. Zero crossing does not close a signal; one fitted probability must select the side. Any research verdict must use the existing weak-signals or rotation recorder.
