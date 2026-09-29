# LEAD-69 CFB replication

## Goal

Replicate the NFL market-move horizon interaction independently in CFB; research only.

## State

**Measured:** complete, one completed scoring run; 2,378 decisive games, 45 week blocks, one CFB look, zero NFL looks. Horizon 1,265-1,113; base 1,267-1,111. LOSO accuracy gain -0.0841 pp [-0.3608,+0.2054], probability_positive 0.267. Log-loss gain -0.0001965 [-0.0006194,+0.0002286], probability_positive 0.1835; Brier gain -0.0001042 [-0.0002812,+0.0000720], probability_positive 0.126. Provisional unresolved_below_power; registry write awaits orchestrator.

## Tried

Protocol fixed before outcomes: FBS regular-season 2023-2025; frozen historical opener proxy; opener-to-kickoff hours; opener-implied probability + move base, adding move ? log(hours); training-only ridge/standardization, LOSO; accuracy, log loss, Brier, 4,000 week-block bootstraps. Full original predeclaration is preserved verbatim in the report. **Measured:** missing opener timestamps required a first-observed-quote proxy; 30 ambiguous event joins excluded before scores, after one aborted loader run. Horizon coefficients 2023 +0.000901, 2024 +0.011169, 2025 -0.003508: NFL sign agrees only in 2025. IS accuracy gain +0.0421 pp; IS-OOS gain gap +0.1262 pp. Artifact checks passed; no new tests. No protocol revision after outcomes.

## Next

Orchestrator reviews source limitations and runs the commands below serially. Next research requires actual opener timestamps or a separately declared study; do not retune this look.

## Open

Exact opener-time replication remains unavailable. Long horizons and one stale last quote remain as declared; no retrospective freshness cutoff. No closing ground established; one fitted probability selects each side. Files: scripts/lead69_cfb_replication.py, docs/lead69_cfb_replication.md; prediction rows, summary and log under tests/scratch/codex/lead69_cfb_replication/. No Git mutations, serving, publication, or registry writes.

## Record commands

Three correlated metrics of one look, one family; not independent votes. Commands prepared, not executed.

```bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name lead69_cfb_horizon_accuracy_v1 --description 'CFB first-observed opener horizon interaction; LOSO; accuracy' --source docs/lead69_cfb_replication.md --league cfb --season-start 2023 --season-end 2025 --effect -0.0841042893188 --effect-units accuracy_points --interval-low -0.360848444928 --interval-high 0.205362104486 --probability-positive 0.267 --standard-error 0.143495229879 --sample-games 2378 --sample-blocks 45 --classification unresolved_below_power --family lead69_cfb_horizon_v1 --category market --plain-summary 'We tested giving college line moves different weight based on how long the opening line had been up. It did not improve this sample; picks stay unchanged.'
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name lead69_cfb_horizon_log_loss_v1 --description 'CFB first-observed opener horizon interaction; LOSO; log_loss' --source docs/lead69_cfb_replication.md --league cfb --season-start 2023 --season-end 2025 --effect -0.000196512663956 --effect-units log_loss_improvement --interval-low -0.000619399182504 --interval-high 0.000228611628916 --probability-positive 0.1835 --standard-error 0.000213980556289 --sample-games 2378 --sample-blocks 45 --classification unresolved_below_power --family lead69_cfb_horizon_v1 --category market --plain-summary 'We tested giving college line moves different weight based on how long the opening line had been up. It did not improve this sample; picks stay unchanged.'
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name lead69_cfb_horizon_brier_v1 --description 'CFB first-observed opener horizon interaction; LOSO; brier' --source docs/lead69_cfb_replication.md --league cfb --season-start 2023 --season-end 2025 --effect -0.000104173685209 --effect-units brier_improvement --interval-low -0.000281190554013 --interval-high 7.20098001834e-05 --probability-positive 0.126 --standard-error 8.94880068494e-05 --sample-games 2378 --sample-blocks 45 --classification unresolved_below_power --family lead69_cfb_horizon_v1 --category market --plain-summary 'We tested giving college line moves different weight based on how long the opening line had been up. It did not improve this sample; picks stay unchanged.'
```
