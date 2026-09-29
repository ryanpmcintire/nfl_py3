# LEAD-71 ` unit 2 complete; serial records pending

## Goal
Execute ROADMAP.md:854's two reserved market-implied-lattice contrasts.

## State
**Measured:** 1,480 opener games, 34 pushes; 1,446 decisive games.
Implied lattice: 727-719 (50.28%, Wilson 95% [47.70%, 52.85%]);
served lattice: 762-684 (52.70%, [50.12%, 55.26%]).
RPS improvement 0.034858, 95% [-0.000318, 0.085864], probability_positive 0.9741.
**Measured:** extra fitted term: 720-553 versus four-term 732-541, n=1,273;
log-loss improvement -0.001173, 95% [-0.001984, -0.000436], probability_positive 0.
**Inferred:** lattice mechanism stays open; proposed wrong-sign closure concerns
only the fixed extra-term extension, pending the orchestrator's serial record.
**Read:** the protocol was saved here before outcome reads; its unchanged full copy
is in [the report](../lead71_unit2.md#predeclared-protocol).
Two reserved contrasts; all diagnostic look counts, IS/OOS gaps, coefficient stability,
reliability bands and per-season scores are reported there.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead71_unit2.py` completed,
exit 0, threads <=2. Two earlier incomplete attempts stopped at timestamp and solver
checks; no family or parameter search. Final source checks exclude 46 timestamp-invalid
model rows and 11 missing quote matches; all 2,960 price fits pass KKT <=8.05e-8.
Max quote-probability error 2.50e-6. Ruff format/check, AST and whitespace checks pass.
Report-only regeneration used saved scores; no additional fits or predictive looks.
Owned changes: script, report and this lane. Rows/hashes: `tests/scratch/codex/lead71_unit2/`.
No test additions, registry writes, publication, Git mutation or served-card changes.

## Next
Orchestrator reviews and executes the two commands below serially.

## Open
Market lattice remains unresolved_below_power. Exact margin log score is infinite for
65 OOS served-lattice zero-mass outcomes. Combined inputs include archived prekick flags
and movement; only the new market feature freezes Tuesday. No serving decision.

## Record commands
Proposed only; Bash-compatible, with pool-player explanations. Read-only closure validation passed.

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look1 --description 'Reserved market-implied lattice look 1; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=0.0348581399733 --effect-units rps_improvement --standard-error=0.0226235062059 --interval-low=-0.000317990775327 --interval-high=0.0858635838172 --probability-positive=0.9741 --sample-games 1480 --sample-blocks 6 --season-start 2020 --season-end 2025 --family LEAD-71-unit2 --classification unresolved_below_power --classification-evidence 'The margin-lattice mechanism remains open; no admissible closure established' --plain-summary 'Moneyline, spread and total prices gave a slightly better picture of final margins. That result leaves this idea open; the pool picks are unchanged.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
```

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look2 --description 'Reserved market-implied lattice look 2; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=-0.00117333787234 --effect-units log_loss_improvement --standard-error=0.000400826685553 --interval-low=-0.00198424742371 --interval-high=-0.000436330749859 --probability-positive=0 --sample-games 1273 --sample-blocks 5 --season-start 2021 --season-end 2025 --family LEAD-71-unit2 --classification refuted_mechanism --closing-ground wrong_sign_resolved --classification-evidence 'Fixed fifth-term extension only: log-loss improvement interval entirely negative; the margin-lattice family remains open' --plain-summary 'Adding the new market chance to the pick formula made held-out chances worse. This finding concerns that extra term; the separate margin table idea remains open.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
```
