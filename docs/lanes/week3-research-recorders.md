# Week 3 research recorder coverage

## Goal

Repair missing prospective challenger records without changing the completed served Week 3 decisions or weakening discrete probability and registry gates.

## State

Measured 2026-09-23: the Week 3 card's `home_cover_probability` is the served discrete read (`discrete_conditional_non_push_v1`); the pre-lattice smooth read is `home_cover_probability_smooth`. A `gaussian_median` refit with metadata home-side offsets reproduces the smooth column to 5.6e-17 on all 16 games. The ECDF, Gaussian-mean and half-life-8 recorders checked against the discrete column and could never pass on a lattice card. `card_refit.smooth_reference_probability` now supplies the smooth column for the reproduction check (key-line overrides replay only on legacy cards; the incumbent side stays the served discrete column). All three then recorded 16 pre-kickoff rows at 14:2x UTC, 5 flips each against the served card. `tiebreaker_shade_prospective.skip` now carries `challenger_id`, so the audit can see its admissible gate (served tiebreaker already carries the shade). Audit re-run: 45 recorded, 11 skipped, 62 active.

## Tried

The five `refresh-picks` recorders (`consensus_movement_1_0_off_incumbent`, four `late_week_*`) are not broken: Week 2 recorded them on the midweek refresh, and Week 3's `refresh_wed` (18:15 ET) and `refresh_thu` (15:00 ET) jobs had not yet run when audited. The scheduler was alive (pids from 2026-09-22 21:33).

## Next

After Thursday 15:00 ET, re-run `scripts/lockday_verify.py --season 2026 --week 3 --run-summary artifacts/scheduled_locks/2026-week-03/weekly_summary.json --json` once and confirm the five late-week challengers recorded. Then move this lane to `done/`.

## Open

The Week 3 run summary predates the fixes, so the audit still lists the three refit recorders under `failed_recorders` and the tiebreaker skip as missing until next week's lock run. `spread_explorer.py:137-149` has the same discrete-vs-smooth reproduction check and will refuse lattice cards; the owner-valued spread explorer needs the discrete read, not a smooth comparison (separate lane). `tiebreaker_low_side_shade` stays ACTIVE_PROSPECTIVE but can never pair while the shade is served; its registry status is an owner decision. These are recording gaps, not research results.
