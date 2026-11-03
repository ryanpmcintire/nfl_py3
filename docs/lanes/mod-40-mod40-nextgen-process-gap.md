# MOD-40 Next Gen Stats process gap

## Goal
Next Gen Stats process gap: time to throw, separation and rushing yards over expected against results as an unluckiness term (ROADMAP MOD-40).

## State
Added 2026-11-03, unmeasured. If points were lost to finishing luck while the tracked process (separation, time to throw, rushing yards over expected) was strong, the team is better than its box score. This uses tracking-derived measures that result-based ratings may omit.

## Tried
Nothing yet. Novelty: mod23-unit12 FTN scheme lane (charting traits), team-quality-is-priced memory, PBP-05 adjusted stats. Grep for 'nextgen' returned no hit in ROADMAP, docs or registry.

## Next
First unit: Split-half reliability of team-weekly process residuals (process minus results); only if it clears the gate, add-one on the served model with the MOD-34 orthogonalised version against the MOD-36 noise floor.

## Open
Data: nflverse nextgen_stats weekly passing, rushing and receiving 2016-2025; reported available, schema to be read in the first unit. Zero crossing closes nothing; record unresolved results with a closing ground.
