# MOD-46

## Goal
Favourite lead-protection skew: how a favourite's OC trait changes the chance of winning by 3 versus 7 versus 14 (ROADMAP MOD-46).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: A favourite that protects leads burns clock and converts big leads into 7 to 10 point wins rather than 17, moving mass between key numbers without changing the mean. Fitted term: trait times spread bin in a multinomial over the discrete margin lattice conditional on the line, entering the cover and push probability served from the discrete read.

## Tried
Novelty check only. Nearest: MOD-18 (lattice), SIM-18 (trait), MOD-17 (lattice ledger); grep 'skew' found only MOD-18 lattice text.

## Next
First unit for one agent: Tabulate realised margin lattice mass at 3, 7, 10, 14, 17 for favourites by trait tercile within spread bins, with a permutation null and season-block bootstrap, then LOSO add-one on the discrete cover read. Data: Local margins 2009-2025 and SIM-18 coordinator traits (free).

## Open
Terciles are a look count of at least 15 cells; lopsided bins need an exact null. Zero crossing closes nothing; one fitted probability selects the side.

## Unit 1 progress (2026-11-03 ~17:15 local)
Script scripts/mod46_favourite_skew.py (ruff clean; trait build copied from sim18_unit2_multiplier.py lines 1-474, grading from mkt27_total_lattice.py). Traits: fav sec_adj, run_adj (prior-only OC spell). Shape: tercile within season, perm null within season x spread bin (2000), game bootstrap; primary opener 2020-25, secondary closing-line 2013-19. Graded: trait-restricted band_read pool vs served read, LOSO add-one. Output artifacts/mod46/{shape.csv,summary.json,run.log}. Run launched in background; read results next, then record cells.

## Unit 1 correction (2026-11-03 ~17:40 local)
First run was invalid: traits of exactly 0.0 (no prior OC spell games, tgt fillna) were ranked into terciles by row order (cover-margin tercile means of +-7 were the symptom), and the restricted-pool grade lacked a placebo (LOSO delta coef ~2 and +0.015 LL gain looked like pool-size recalibration, not trait info). Fixed: zero traits dropped, plus placebo arm (trait values shuffled within the prior pool, random tercile) per trait. Served-read replication diff vs file column ~0.028 mean (same as MKT-27: absolute level not served, paired valid). Rerun launched as background task bx7j0vnof; results in artifacts/mod46/{shape.csv,summary.json,run.log}. Next: read them, judge trait arms against placebo arms, record cells (units rate_pp for mass diffs, log_loss_improvement for graded), fill this lane.

## Unit 1 result (2026-11-03 ~18:10 local; measurement only; scripts/mod46_favourite_skew.py, artifacts/mod46/)
Looks: 32 shape cells (2 traits x 8 stats x 2 samples) + 4 graded arms + 2 placebo arms = 38. Recorded 4 unresolved cells (family mod46_favourite_lead_skew_unit1).
- Shape (opener 2020-25, n 991 with real traits; tercile within season; perm null within season x spread bin): no stat moves. Largest: run_adj mean cover margin high-low -1.30 [-2.91,.27] P+ .06 perm p .10; sec_adj mass at 7 +2.7pp P+ .91 perm p .21; just_under (0 to -3 under spread) sec_adj +3.0pp P+ .86 p .23. Key masses 3/10/14/17 all |diff|<2pp, perm p .2-1.0. Secondary 2013-19 closing: run_adj mass at 7 -5.3pp [-9.2,-1.4] perm p .006 but 0/7 seasons positive and 0.0pp in 2020-25 (1 of 32 looks, not replicated).
- Graded (1491 non-push opener games, own control read; served file not replicated exactly, paired valid): trait pool read vs control, decisive sec_adj 140 games control right 76 trait 64; run_adj 134, control 77 trait 57. LOSO add-one dLL sec -.0020 [-.0047,.0006] P+ .07; run +.0028 [-.0028,.0083] P+ .84; placebo (shuffled trait) +.0016 (P+ .78) and +.0028 (P+ .90): trait equals placebo. Intercept LL .69327, control .69568.
- Verdict: unresolved_below_power; no skew toward under-spread or specific key numbers distinguishable from noise; lattice shape not served. Not closed.
Next: unit 2 only if reliability-weighted trait on the whole favourite-minus-underdog gap (both teams) or SIM-18 engine multiplier is wanted; otherwise MOD-46 parked.
