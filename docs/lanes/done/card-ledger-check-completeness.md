# Card ledger check completeness

## Goal
Make incomplete ledger comparisons fail closed and distinguish them from measured disagreements.

## State
Completed checker and CLI fix; frozen historical rows are preserved.
**Measured:** the current Week 3 command exits 1 with evaluated=true, ok=false and five side disagreements among 16 paper rows. The missing-source command exits 1 with evaluated=false, ok=false. Evidence: `.tmp/backlog-card-ledger-reviewed.log`, `.tmp/backlog-card-ledger-missing.log`.
**Read:** Tuesday rows name model `429b12a7106c7e7d`; the active model is `284a38bf00c29c53`. A comparison with a later active model does not authorize rewriting the Tuesday baseline.

## Tried
Served-load failures now return ok=false; board-content errors make evaluated comparisons incomplete and unsuccessful. Operational CLI exits 1 and publication diagnostics retain the reason. Reviewed source changes and ran the real commands once; no new tests.

## Next
Use the explicit incomplete/mismatch result when investigating later ledger checks.

## Open
The five disagreements remain visible: HOU/IND, KC/MIA, LA/DEN, MIN/TB, SEA/WAS. No historical rows were rewritten.
