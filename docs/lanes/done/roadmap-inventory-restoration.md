# Roadmap inventory restoration

## Goal

Restore the documented offline roadmap accounting command, including human-readable and JSON summaries of remaining, actionable, status, and phase totals.

## State

`scripts/roadmap_inventory.py` is restored locally for root integration. It parses tracked status rows, rejects duplicate IDs, and reports overall, status, and per-phase counts.

## Tried

The documented command initially exited 1 because the script was absent. Repository history showed that the repository cut removed the prior implementation while the command remained documented. The restored locked JSON command exited 0 against the current roadmap and measured 55 total items, 11 done, 44 remaining, 24 actionable, 10 phases, and no unassigned rows.

## Next

Root should review and integrate the script and this completed lane.

## Open

None.
