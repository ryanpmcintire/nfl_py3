# SIM-04/SIM-05 play-by-play simulator and counterfactuals

## Goal
Write the build plan for ROADMAP SIM-04 (full play-by-play simulator) and
SIM-05 (counterfactuals). Done for this lane's opening unit means the plan
document exists, is grounded in measured inventory and the MOD-09 audit, and
names a unit 1 a subagent can start immediately. No model training or `src/`
code in this planning unit.

## State
- `docs/sim04_play_simulator_plan.md` written 2026-09-25: architecture
  (game-state, per-play/clock/penalty/turnover/4th-down submodels, sampling),
  leakage rules, a predeclared chronological LOSO evaluation protocol against
  the served `DiscretePushReader` / `serve_discrete_three_way`
  (`src/nfl_ats/mass_preserving_lattice.py`), 10 build units, compute
  estimates, dashboard deliverable, risks, and a plain-English section.
- This lane file.
- ROADMAP.md **not edited** (out of scope for this task); proposed replacement
  row text was handed back to the caller for the primary orchestrator to
  apply.
- No commits made.

## Tried
- Measured (`tests/scratch` inventory script, this session): NFL pbp capture
  `data/pbp/raw/20260817T184927Z/` has 17 season partitions (2009–2025); 2025
  sample = 48,771 rows, 45 columns, matching the fixed allowlist
  `PBP_SNAPSHOT_COLUMNS` in `src/nfl_ats/pbp.py`. Missing from that allowlist:
  timeouts-remaining, penalty team/type, personnel/formation, non-passer
  player IDs — present upstream in the `nflreadpy` source already fetched
  elsewhere (`nflverse_current_season.py`), so widening is a config change
  (Unit 2), not a new data source.
- Measured: `data/processed/game_features_pbp.parquet` = 4,902 rows x 201
  columns, seasons 2009–2026 — this is the already-rejected MOD-09
  drive/summary bundle (50.36% ATS no edge, worsened Brier); the plan does not
  reuse it as a side-pick feature source.
- Measured: `data/cfb/pbp/raw/` has 44 parquet files (~1.3 GB) across capture
  runs — noted as an auxiliary/positive-control corpus only.
- Read: `docs/play_level_audit.md` (ICC 0.0131 finding) and ROADMAP MOD-09
  (line ~666) / Phase 7 simulations section (~672–683, including the existing
  "Simulation is accepted only if it improves held-out distribution
  calibration" acceptance line) to ground the plan's honest framing.
- Read: `src/nfl_ats/mass_preserving_lattice.py` (`DiscretePushReader`,
  `serve_discrete_three_way`, key numbers `(3, 7, 10, 14)`,
  `MIN_BAND_GAMES=200`) as the comparison baseline the plan's evaluation
  protocol targets.

## Next
Spawn a fresh subagent for **Unit 1** of the plan
(`docs/sim04_play_simulator_plan.md`, "Drive-outcome key-number baseline"):
build the unconditional drive-chain simulator from
`data/pbp/raw/20260817T184927Z/`, sample N=10,000 games, and report simulated
vs. historical key-number mass at 3/7/10/14/17. Script goes under
`tests/scratch/` or `scripts/`, not `src/`. This is the single go/no-go gate
for the rest of the plan (see plan's Risks section) — if it fails to reproduce
key-number mass, escalate to the owner before starting Units 2+.

## Open
- Whether the primary orchestrator wants Unit 2 (capture-column widen, touches
  `src/nfl_ats/pbp.py`) done before or in parallel with Unit 1 (Unit 1 does
  not need the widened columns).
- Whether SIM-04/05 should get a ROADMAP.md status change now (proposed row
  text was returned to the caller, not applied) or wait until Unit 1's result
  is in hand.
