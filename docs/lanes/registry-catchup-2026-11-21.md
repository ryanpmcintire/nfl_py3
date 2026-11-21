# Registry catch-up 2026-11-21

## Goal
Record unrecorded MOD-52, MOD-23, LEAD-102, MOD-39 results.

## State (2026-11-21 09:54 local)
- MOD-52: already recorded 2026-11-14, family mod52_mod41_stability_u1 (17 cells); skipped.
- MOD-23: shard registry/weak_signals/mod23_ewma_span_wide.json already holds all 4 batch cells; skipped, no reproduction run needed.
- LEAD-102: already recorded 2026-11-14, family lead102_name_blinded_control_read; skipped.
- MOD-39: recorded 4 cells in family mod39_playoff_leverage (stakes_only and levdiff_plus_stakes, each accuracy_points and log_loss_improvement), batch artifacts/mod39/weak_signals_batch_stakes.json. Pool exit 0.

## Tried
Elimination and clinch differences are not fitted separately in artifacts/mod39/mod39_results.json (only jointly as stakes_only); leverage-diff slope cells (ats_points) already repaired 2026-11-14.

## Next
Separate elimination-only and clinch-only fits if wanted; forward read after Week 16.

## Open
Family now has 10 cells vs 7-8 looks declared.

ROADMAP MOD-52 update: already recorded 2026-11-14 (17 cells, mod52_mod41_stability_u1); drop the "none recorded" note.
ROADMAP MOD-23 update: ewma_span_wide 4 cells already in registry (mod23_ewma_span_wide); no further record owed.
ROADMAP LEAD-102 update: unit 1 already recorded 2026-11-14 (lead102_name_blinded_control_read, correlation cell).
ROADMAP MOD-39 update: 2026-11-21 recorded stakes_only (elim+clinch joint) and levdiff_plus_stakes cells, accuracy_points and log_loss_improvement, all unresolved_below_power (stakes_only acc +0.84 pp P+ 0.72, ll -0.0013 P+ 0.29; combined acc -0.68 pp P+ 0.29, ll -0.0032 P+ 0.10); elim and clinch not separable in artifact.
