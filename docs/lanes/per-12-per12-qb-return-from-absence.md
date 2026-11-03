# PER-12 Quarterback return from absence

## Goal
Quarterback return from absence: first starts back after three or more missed games as a Tuesday term (ROADMAP PER-12).

## State
Added 2026-11-03, unmeasured. A returning starter is priced at his season-long name while rust, conditioning and lineup chemistry lag. The opener also cannot know how the backup performed, which feeds the return week. This is an information event the market sees but may price too fully or too little.

## Tried
Nothing yet. Novelty: PER-02 quarterback state (performance level), registry backup_tenure_gap_on_production (backup tenure), docs/qb_age_curve_screen.md (career stage). Grep for 'returning from injury' and 'return from injury' returned no hit.

## Next
First unit: List return events 2009-2025 (starter with at least 3 consecutive missed starts, then a start), compute the team's opener margin residual for returns 1 and 2 against the same QB's other starts and against matched healthy starters at the same spread, and report probability_positive with a QB-clustered bootstrap and a within-QB permutation null.

## Open
Data: nflverse weekly starters and injuries; QB state table in src/nfl_ats/quarterbacks.py. Zero crossing closes nothing; record unresolved results with a closing ground.
