# MKT-22 Trait pricing audit

## Goal
Trait pricing audit: does the opener-to-close move already follow each reliable trait that failed to predict cover (ROADMAP MKT-22).

## State
Added 2026-11-03, unmeasured. MOD-37 venue false starts and MOD-32 coach aggression were reliable traits that did not carry into scoring. If the line moves toward a trait before kickoff, the market prices it late; if it does not move, the trait is absent or already in the opener. The move is a dense label on the same games with more resolution than a cover bit.

## Tried
Nothing yet. Novelty: MKT-20 (line move as the target, general), ENG-47 (move-toward-pick metric), MOD-37, MOD-32. Grep found no trait-by-trait pricing audit.

## Next
First unit: For each declared trait, regress the signed opener-to-close move on the trait difference with a within-week permutation null, and tabulate trait reliability against pricing so the transfer gap becomes explained or stated as unexplained, counted as one look family.

## Open
Data: Existing opener and close archive and the trait tables from the mod31, mod32, mod37 and mod38 scripts. Zero crossing closes nothing; record unresolved results with a closing ground.
