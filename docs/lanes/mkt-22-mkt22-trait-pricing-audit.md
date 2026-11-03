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

## Unit 1 in flight (2026-11-03)
Script scripts/mkt22_trait_pricing.py, output artifacts/mkt22/cells.json and run.log. Family mkt22_move_follows_traits, 12 cells: 3 traits (venue false-start premium, 4th-down coach aggression home-minus-away, 4th-quarter pass-rate-over-expected when ahead home-minus-away; SIM-14 is the same 4th-down trait) x 2 eras (Tue 2020-25, SBR 2011-19) x 2 outcomes (close-minus-open move; opener margin beyond move). Traits as-of (prior games only). OLS with open spread and season-week dummies, row bootstrap 2000, within-week permutation 2000.

## Unit 1 result (done, 12 cells recorded unresolved_below_power)
Move slope per SD, P+, perm p (move / cover-beyond-move): SBR 2011-19 aggression +0.144 [0.048,0.241] P+ 0.9995 perm 0.0025 / +0.04 P+ 0.56; venue false start +0.054 P+ 0.89 / -0.40 P+ 0.10; pass-ahead -0.003 / +0.27 P+ 0.80. Tue 2020-25: aggression -0.034 P+ 0.20 / +0.50 P+ 0.93 perm 0.15; venue -0.02 / -0.31; pass-ahead +0.009 / +0.79 [0.08,1.49] P+ 0.98 perm 0.02. Aggression move seen only in SBR era, absent in Tue era (not replicated). Caveats: raw expanding means, no shrinkage; cover look uses margin vs opener (ats points).
## Next
Pass-ahead cover-beyond-move in Tue era is the only perm-supported cover cell: needs LOSO check before serving. Aggression move: test other eras/closer-to-kickoff.
