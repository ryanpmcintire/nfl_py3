# LEAD-85: coefficient uncertainty and weekly Best Pick

## Goal
Complete the assigned candidate-versus-served weekly nomination replay.

## State
**Measured:** unit 2 exited 0; 54 held-out weeks, zero changed nominees.
Both arms 30-22-2; nonpush cover 57.69% [95% Wilson 44.19,70.13].
Reward difference 0 pp [0,0], probability_positive 0.5; exact paired p=1,
zero discordant weeks (no nomination-effect information).
Nominee Brier improvement +0.00021444 [-0.00032115,+0.00089586],
probability_positive 0.7375; candidate IS/OOS/gap 0.234435/0.246434/+0.011999.
**Read:** root recorded unit 1's three pooled probability-score cells.

## Protocol fixed before outcomes
Unchanged full declaration: `tests/scratch/codex/lead85_unit2/protocol_declaration.md`;
original study: `docs/lead85_protocol.md`. Historical 2020-2025 OPENER proxy;
outer 2023/2024/2025, fit through Y-3, reserve Y-2, calibrate Y-1. Reuse frozen
four-term/integrated fits; common post-Sunday-12:45 contenders and dispersion pool.
Rank calibrated unconditional cover + 0.5 push with LEAD-53 ties; one probability
selects each side. Conditional nominee Brier primary; reward secondary; IS/OOS/gap,
reliability, changed records, Wilson intervals and exact weekly arm-label-swap null.
10,000 hierarchical season/week draws, base seed 85, half-credit zero draws.
110 unit-2 cells; 713 reserved study looks + 28 supplemental = 741 charged.
No protocol revision, refit, search, or side flip. Zero crossing closes nothing.

## Tried
Ran `.tools/uv.exe run --no-sync python scripts/lead85_unit2.py` once (exit 0),
two threads. Scoped Ruff checks passed. PMF/chronology/side parity passed.
Report: `docs/lead85_unit2.md`; rows, frozen declaration, hashes and log:
`tests/scratch/codex/lead85_unit2/`. No registry, publication or Git mutation.

## Record commands
Prepared only; orchestrator executes serially. Bash-compatible, candidate versus served:
```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead85_unit2_nominee_brier_2023_2025 --description 'Uncertainty versus served weekly Best Pick' --source docs/lead85_unit2.md --effect-units brier_improvement --effect 0.000214435973170224 --standard-error 0.000317601855592181 --interval-low -0.000321153866561299 --interval-high 0.000895861378439068 --probability-positive 0.7375 --sample-blocks 54 --classification unresolved_below_power --league nfl --season-start 2023 --season-end 2025 --family lead85_weekly_nominee_brier --classification-evidence 'No admissible closing ground established' --plain-summary 'Accounting for uncertainty slightly improved the weekly Best Pick probability scores; this small replay does not settle the benefit.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead85_unit2_weekly_reward_2023_2025 --description 'Uncertainty versus served weekly Best Pick' --source docs/lead85_unit2.md --effect-units accuracy_points --effect 0 --interval-low 0 --interval-high 0 --probability-positive 0.5 --sample-blocks 54 --classification unresolved_below_power --league nfl --season-start 2023 --season-end 2025 --family lead85_weekly_weekly_reward --classification-evidence 'No admissible closing ground established' --plain-summary 'Both versions chose the same Best Pick in all 54 weeks: 30 wins, 22 losses and 2 ties. There were no changed picks to compare.'
```

## Next
Orchestrator reviews the report and runs the two prepared record commands.

## Open
**Inferred:** unresolved_below_power, pending registry entry; no closing ground.
Three reused outer seasons; no prospective serving conclusion. Original five-arm
model-only/dated-market/Elo comparison remains outside this two-arm packet.
