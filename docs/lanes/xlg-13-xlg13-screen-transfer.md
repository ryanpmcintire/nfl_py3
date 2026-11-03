# XLG-13 Screen-to-served transfer audit

## Goal
Screen-to-served transfer audit: does a candidate gain on the pooled 14-input harness predict its gain on the 88-input served model ROADMAP row XLG-13.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: The 2026-11-03 pooled grade shows the ranking of shrinkage arms holds across NFL and CFB (accuracy-gain Spearman 0.88), but levers that only change shrinkage or calibration help the 14-feature model and not the 88-input served model. Before XLG-10 may gate any served promotion, measure whether pooled-harness add-one gains rank the same candidates the same way as the served model does. Data: Candidate add-one results already on disk from MOD-23 units 6-18 and the MOD-24 arms (artifacts under artifacts/ and the unit lanes); no new fits needed for the first unit.

## Tried
Novelty check only: Nearest: XLG-10 (builds the screen, does not validate it against the served model), mod24-more-games-2026-11-03.md (rank agreement across shrinkage arms on one model, not across model sizes), forward_pooling_declaration_2026-11-03.md (forward pooling rules). Differs: tests transfer of the screen itself across model size using finished candidates; grep for screen-to-served and transfer validity returned no hit.

## Next
First unit: Tabulate every completed candidate with its 14-input harness gain and its 88-input served gain (accuracy points and log loss, same opener population), then Spearman and Kendall with a bootstrap over candidates and a permutation null; report probability_positive and the count of candidates; decide only afterwards whether the pooled screen may rank lanes.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
