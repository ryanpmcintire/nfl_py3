# XLG-10 Pooled add-one screen

## Goal
Use the measured 4-5x pooled power to screen features, not model settings. ROADMAP row XLG-10.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: mod24 showed NFL+CFB accuracy pooling is admissible on the 14 shared inputs (about 4-5x effective games). Each candidate that exists in both leagues (rest, bye, role, special teams, venue) is added one at a time to the shared harness and graded pooled, with heterogeneity z deciding whether pooling holds. Data: data/processed/cfb_game_features.parquet and the NFL per_game artifact, both local.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: mod24_commonfeat.py and mod24_pooled_levers.py (vary the ridge alpha and training levers on fixed features), cfb_*_replication.md (separate replication, never pooled). No row grades added features on the pooled harness.

## Next
Pick the add-one list from features present in both parquets and declare the family and look count. First unit: First unit: rebuild the commonfeat harness with one added shared candidate (rest and bye first), contrast against the 14-input base in both leagues, report heterogeneity z and the pooled effect with probability_positive.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Pooling is admissible only where heterogeneity z is below 1.96; report cells where it fails as unresolved.
