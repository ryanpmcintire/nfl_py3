# fix-picks-u8-embedded-share-oos

## Goal
Out-of-season measurement of how much of the pool-hook push mass the model's stated probability at the pool line already embeds, and which stage-2 offset form to serve. Parent: fix-picks-2026-11-10.md; predecessor: fix-picks-u7-at-market-exact.md.

## State (2026-11-10, measured)
- Method: `opener_pick_evaluation` (clv.py) re-run twice with `opener_line_override` (whole-number Tuesday-open games shifted to M+0.5 / M-0.5, others unchanged), same weekly walk-forward, offsets and discrete reader as the artifact; base run reproduces per_game exactly (max diff 5e-17). Scripts: tests/scratch/fixpicks/u8_run.py (plus, minus, base -> u8_*.csv), u8_analysis.py. Stacked rows 1322 (661 games, 2020-2025); hook mass>0 on 938 (2020 has no prior-season mass, so 0 there). Overrides also shift the offset stream slightly (inferred, small).
- Step 2 (logit exact-minus-stated on conversion-minus-stated, through origin): slope 0.470 overall, season-block bootstrap 95% [0.407, 0.534], P(slope<1)=1.0. Per season 2021-25: .368 .424 .497 .539 .556 (drifting up). By band: |M|<=3 .567 (n476), 3-7 .222 (n362), >7 .231 (n100). So the model embeds about 53% of the mass out of season (U7 2026 read ~68%); the full conversion over-corrects, no conversion under-corrects.
- Step 3 LOSO coefficient (held season 2020..2025): A .805 .875 .790 .769 .812 .777 (full .805); B .608 .755 .654 .575 .660 .611 (full .643, sd .063); C 1.016 .991 .933 .982 .984 .966 (full .978); D .800 .872 .789 .763 .807 .773 (full .801; share .45-.50 from training seasons).
- Out-of-season log loss / Brier / accuracy (1322 rows): A .6786/.2427/.5711; B .6783/.2426/.5726; C .6784/.2427/.5703; D .6784/.2426/.5703; B at fixed .805 .6783/.2426/.5696; C at fixed .805 .6785/.2427/.5726. In-sample gap about +.0013 for every arm.
- Paired week-block bootstrap (107 blocks, 4000; ll gain, positive favours the arm): B vs A +.00027 [-.00011,+.00066] P+ .915 (40 differing sides, 21-19); C vs A +.00013 P+ .727 (20-21); D vs A +.00021 P+ .853 (20-21); B vs C +.00014 [-.00008,+.00036] P+ .899 (7 differ, 5-2); D vs C +.00008 P+ .912 (4 differ, 2-2); B_fixed vs C_fixed +.00017 [-.00031,+.00068] P+ .759 (58 differ, 27-31).

## Tried
- Looks: step 2 ten slopes (1 overall, 6 seasons, 3 bands), one family; step 3 six arms, seven paired comparisons. All arms differ by under .0003 log loss; nothing separable at this resolution.

## Next
- Serving (inferred, separate from closure): keep the no-conversion form. Best arm is B (stated, no conversion); with its own coefficient .643 (fold sd .063) versus fixed .805, ll .6783 either way (equal to 4 decimals), so refitting the coefficient is optional and gains nothing measurable; exact at-market recomputation (A) is not better than B (B vs A P+ .915), so it is not worth the extra refit. If a coefficient change is made, refit stage 2 on the B offset rather than reusing .805.
- Proposed record (not run): `nfl-ats weak-signals record` family fix_picks_2026_11_10, cell embedded_share_oos_no_conversion, effect log loss gain B vs C +.00014, P+ .90, classification unresolved_below_power (no closing ground; interval not wholly on wrong side).

## Open
- 2020 contributes no hook mass (no earlier season); share trend across seasons rising; band split (|M|<=3 .57 vs wider .22) not tested for a mechanism or permutation null.
