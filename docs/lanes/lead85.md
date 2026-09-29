# LEAD-85: coefficient uncertainty adapter

## Goal
Execute the declared adapter unit; preserve its outputs for the separate paired nomination replay.

## State
**Measured:** unit 1 complete; one research run exited 0. The declaration was saved before outcomes; its unchanged snapshot is copied to [lead85_protocol.md](../lead85_protocol.md). Fixed 20-node integration; 713 study looks. Original declaration SHA-256: `cf45b05eb22564630d795565a0349dc674c159380300049890ee257310de9cce`.
**Measured:** 1,531 source-complete opener games (34 pushes); six missing-move exclusions. Outer 2023-2025: 799 non-push games, both arms 456-343; decisive record 0-0. Candidate Brier IS/OOS/gap: 0.244725/0.245552/+0.000828.
**Measured:** OOS Brier improvement +0.00005617 [-0.00003111,+0.00017559], probability_positive 0.8465; log loss +0.00012859 [-0.00006655,+0.00040250], 0.8487; RPS +0.00035534 [-0.00052524,+0.00168452], 0.6767. Brier improves in 2023/2024 and worsens in 2025. **Inferred:** unresolved_below_power, not a serving verdict.

## Tried
`.tools/uv.exe run --no-sync python scripts/lead85_unit1.py` once (exit 0); two threads. Source hashes, dated move parity, upstream dates, discrete PMF reconstruction (max error 2.13e-13), node-mixture identity and push preservation passed. Script Ruff and scoped whitespace checks passed. No new tests, registry writes, publication or Git mutation.

## Record commands
Prepared, not executed; one serial batch records the 16 fold/pooled held-out diagnostic cells, with player-readable text on every cell. Bash-compatible:
```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --batch tests/scratch/codex/lead85_unit1/registry_batch.json --plain-summary 'Allowing for uncertainty in the football estimates slightly improved the overall probability scores, but helped in two seasons and hurt in the third. The picks stayed the same; the weekly Best Pick comparison is still to come.'
```

## Next
Orchestrator records the batch, then assigns unit 2: identical weekly contender replay with model-only, dated market and Elo baselines, nominee Brier primary, weekly reward and existing push/tie rules. Script, report and full declaration are in scripts/lead85_unit1.py, docs/lead85_unit1.md and docs/lead85_protocol.md; predictions, PMFs and batch are under tests/scratch/codex/lead85_unit1/.

## Open
Weekly nomination effects and five-arm comparison remain unmeasured. Three outer seasons; previously examined archives. Reliability for research closure is unmeasured. No primary-endpoint verdict or serving change. Full-repository integration checks remain with the orchestrator.

**Recorded 2026-09-29 (root):** pooled Brier, log-loss and RPS cells only (per-season and zero-change accuracy cells not recorded), unresolved_below_power.
