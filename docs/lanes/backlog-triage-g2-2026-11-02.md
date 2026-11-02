# Backlog triage G2 (2026-11-02)

## Goal
Read-only verdicts for PER-03 PER-07 PER-09 PER-10 ENV-02 MOD-07 MOD-09 MOD-10 MOD-17 MOD-18.

## State
- PER-03 OWNER-PHYSICAL: SPORTRADAR_API_KEY (or NFL.com written consent) plus one live complete manifest. pfr_transactions_sat last OK 2026-10-31 (scheduler_log).
- PER-07 NEXT-UNIT: LEAD-29 first-game-under-new-playcaller look using dated OC changes in data/raw/coordinators/20260911T033007437805Z; record via weak-signals; light CPU.
- PER-09 CLOSE-NOW: the row's stated remaining item (ST season-lagged chronological ATS eval) ran and is recorded: registry/weak_signals/st_rating_ats_study.json, lane st-ratings-ats-study.md. Availability-gap arm is a new declared unit, not this row's DoD.
- PER-10 CLOSE-NOW: done/injury-scenario-producer.md; two cells unresolved_below_power in registry/weak_signals/_no_family.json. Producer unbuilt by design (kernel deleted b7ed31d); loose end: E501 in scripts/injury_scenario_producer.py.
- ENV-02 FORWARD-ONLY: surface term recorded (registry/weak_signals/surface_switch_fitted_term_v1.json) unresolved; needs 2026 season served data, earliest scoring after Week 18 (Jan 2027). Roof issuance values naive-timezone, blocked.
- MOD-07 FORWARD-ONLY: challenger_decisions.parquet has weeks 1-8; refresh-picks --record-decisions ran OK 2026-10-31. Score at season end (Jan 2027).
- MOD-09 / MOD-10: gated by PBP-05 / player-state reliability; not refuted, not closeable.
- MOD-17 NEXT-UNIT: lattice_centre_decisions.parquet has only 3 rows (weeks 1-3); enrollment skipped since 2026-09-24. Unit: decide retire/repoint ledger in src/nfl_ats/lattice_centre_challenger.py, then close.
- MOD-18 CLOSE-NOW: done/mod18-spread-regime.md, registry/weak_signals/mod18_spread_regime.json; paired-opener evaluation also complete (done/mod18-paired-opener-evaluation.md).

## Tried
Row tails, done lanes, registry shard greps, scheduler_log greps. No code run.

## Next
Orchestrator updates ROADMAP statuses for CLOSE-NOW rows; spawn PER-07 and MOD-17 units.

## Open
PBP-05 not found in ROADMAP.md by grep (MOD-09 dependency unverified).
