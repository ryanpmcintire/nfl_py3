# LEAD-71: unit 3 complete; serial records pending

## Goal
Compare market-implied and served push chances at the frozen pool opener.

## State
**Measured:** 1,480 games, 34 pushes, 103 week blocks; 645 integer-line games.
Decisive implied 727-719 (50.28%, Wilson 95% [47.70%, 52.85%]); served 762-684 (52.70%, [50.12%, 55.26%]).
Served-minus-implied log loss +0.00001127, 95% [-0.00090354, +0.00107979], probability_positive 0.4851;
Brier -0.0000218623, 95% [-0.000160194, +0.000127629], probability_positive 0.3718.
**Inferred:** better push calibration is not established; unresolved_below_power, no closure or serving decision.
**Read:** protocol was frozen in this lane before Unit 3 scoring; unchanged copy is in
[the report](../lead71_unit3.md#predeclared-protocol): two primary contrasts, 127 total estimands;
chronological LOSO reuse, IS/OOS gaps, all seasons and 3/7/10/14 reliability reported.

## Tried
**Measured:** .tools/uv.exe run --no-sync python scripts/lead71_unit3.py completed, exit 0, <=2 threads.
Used UV_CACHE_DIR=tests/scratch/codex/lead71_unit3/uv-cache and PYTHONDONTWRITEBYTECODE=1.
Two incomplete runs stopped at fractional-line validation and lost pandas block-level names;
both corrected without changing protocol/population. One completed scoring run; prose-only regeneration reused scores.
Ruff, AST, no-comments/docstrings, frozen-protocol/hash/look-budget checks and read-only record-payload validation passed.
Owned files: scripts/lead71_unit3.py, docs/lead71_unit3.md, this lane; rows/hashes/logs under tests/scratch/codex/lead71_unit3/.
No new tests, refits, registry writes, publication, Git mutation or served-card changes.

## Next
Orchestrator reviews report and runs the two Unit 3 commands serially; Unit 2 proposals retained below.

## Open
Key 10: 0/22 pushes, Wilson 95% [0%, 14.87%]; key 14: 1/11, [1.62%, 37.74%].
Week-bootstrap intervals condition on observed events and do not capture new seasons or refit uncertainty.
Same researched Unit 2 population; no independent outer test. Only primary contrasts proposed for recording.

## Record commands
Proposed only; Bash-compatible. Set UV_CACHE_DIR as above if the default cache is inaccessible.
Unit 3 (both unresolved; pool-player summaries; validated without registry writes):

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit3-logloss --description 'Frozen-opener push logloss; 2 primary and 125 descriptive looks' --source docs/lead71_unit3.md --league nfl --category market --effect=1.12700029517e-05 --effect-units log_loss_improvement --standard-error=0.000513855014418 --interval-low=-0.000903542138385 --interval-high=0.00107979328137 --probability-positive=0.4851 --sample-games 1480 --sample-blocks 103 --season-start 2020 --season-end 2025 --family LEAD-71-unit3 --classification unresolved_below_power --classification-evidence 'No closure or serving decision; paired week bootstrap with inherited LOSO folds' --plain-summary 'We checked how often a game lands exactly on the pool spread. The market prices and the current margin table give different chances. This check leaves the idea open and does not change any pool pick.' --notes '2 primary endpoints; 127 total estimands; unadjusted intervals; no new fits'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit3-brier --description 'Frozen-opener push brier; 2 primary and 125 descriptive looks' --source docs/lead71_unit3.md --league nfl --category market --effect=-2.1862293285e-05 --effect-units brier_improvement --standard-error=7.3328728204e-05 --interval-low=-0.000160194332112 --interval-high=0.000127629197674 --probability-positive=0.3718 --sample-games 1480 --sample-blocks 103 --season-start 2020 --season-end 2025 --family LEAD-71-unit3 --classification unresolved_below_power --classification-evidence 'No closure or serving decision; paired week bootstrap with inherited LOSO folds' --plain-summary 'We checked how often a game lands exactly on the pool spread. The market prices and the current margin table give different chances. This check leaves the idea open and does not change any pool pick.' --notes '2 primary endpoints; 127 total estimands; unadjusted intervals; no new fits'
```

Unit 2 retained pending the orchestrator's serial handling:

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look1 --description 'Reserved market-implied lattice look 1; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=0.0348581399733 --effect-units rps_improvement --standard-error=0.0226235062059 --interval-low=-0.000317990775327 --interval-high=0.0858635838172 --probability-positive=0.9741 --sample-games 1480 --sample-blocks 6 --season-start 2020 --season-end 2025 --family LEAD-71-unit2 --classification unresolved_below_power --classification-evidence 'The margin-lattice mechanism remains open; no admissible closure established' --plain-summary 'Moneyline, spread and total prices gave a slightly better picture of final margins. That result leaves this idea open; the pool picks are unchanged.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look2 --description 'Reserved market-implied lattice look 2; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=-0.00117333787234 --effect-units log_loss_improvement --standard-error=0.000400826685553 --interval-low=-0.00198424742371 --interval-high=-0.000436330749859 --probability-positive=0 --sample-games 1273 --sample-blocks 5 --season-start 2021 --season-end 2025 --family LEAD-71-unit2 --classification refuted_mechanism --closing-ground wrong_sign_resolved --classification-evidence 'Fixed fifth-term extension only: log-loss improvement interval entirely negative; the margin-lattice family remains open' --plain-summary 'Adding the new market chance to the pick formula made held-out chances worse. This finding concerns that extra term; the separate margin table idea remains open.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
```
