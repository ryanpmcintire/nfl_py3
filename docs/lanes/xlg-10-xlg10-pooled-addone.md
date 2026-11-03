# XLG-10 Pooled add-one screen

## Goal
Use the measured 4-5x pooled power to screen features, not model settings. ROADMAP row XLG-10.

## State
Unit 1 done 2026-11-03 (measured). Family xlg10_pooled_addone, 5 arms (home_rest, away_rest, bye_diff, short_week_diff, rest_bye_bundle) x NFL opener 2020-25, CFB opener-matched, x 3 metrics = 30 looks. Rest_diff is already in the 14, so the arms add per-side rest days, bye (no game prior week, same-season), short week (rest < 7 d), built identically from each frame's own schedule (scripts/xlg10_addone.py). 10 accuracy cells recorded, all unresolved_below_power. Pooled accuracy pts [95%] P+ (het z): home_rest +0.06 [-0.23,0.35] .66 (0.51); away_rest +0.14 [-0.08,0.36] .89 (1.56); bye_diff +0.21 [-0.04,0.46] .95 (0.05); short_week +0.12 [-0.18,0.41] .78 (-0.09); bundle +0.16 [-0.25,0.58] .78 (0.69). Pooled log loss P+ .03-.16 for all arms (worse), bundle NFL LL P+ .01. Artifacts artifacts/pooled_screen/xlg10_*.json.

## Tried
Unit 1 above. CFB rest/bye is noisy (non-FBS games absent; 21% bye rate, max rest 42 d).
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: mod24_commonfeat.py and mod24_pooled_levers.py (vary the ridge alpha and training levers on fixed features), cfb_*_replication.md (separate replication, never pooled). No row grades added features on the pooled harness.

## Next
Next candidates: role, special teams, venue (need a same-construction check first). bye_diff is the only accuracy P+ near .95 but LL is worse; ungraded on the served path.

## Open
Pooling is admissible only where heterogeneity z is below 1.96; report cells where it fails as unresolved.

## Unit 2 done 2026-11-03 (measured)
Family xlg10_pooled_addone_unit2: 3 arms x 2 leagues x 3 metrics = 18 looks (scripts/xlg10_addone2.py; within-season prior-game rates, NFL from data/pbp slim pbp, CFB from cfb pbp type.text). Skipped: venue travel (CFB has venue_id, no coordinates; neutral_site already in the 14); special teams (NFL 2018+ pbp lacks field_goal_result). Accuracy pts [95%] P+ (flips W-L): fumble_lost NFL -0.73 [-1.59,+0.07] .03 (20-31), CFB -0.02 .46 (63-64), pooled -0.19 .18 het 1.50; int_rate NFL -0.33 .11 (10-15), CFB -0.09 [-0.17,-0.02] .00 (0-4, four flips, not a resolved sign), pooled het ok; bundle NFL -1.06 [-2.18,-0.07] .02 (21-37), CFB -0.04 .42, pooled -0.21 .17 het 1.75. Pooled LL P+ .05-.16. 6 accuracy cells recorded unresolved_below_power; LL/Brier not recorded. Artifacts artifacts/pooled_screen/xlg10_{fumble_lost_diff,int_rate_diff,turnover_bundle}.json, xlg10u2_all.log.
Diagnostic (bye_diff, scripts/xlg10_spread_diag.py, xlg10u2_diag.log): edge sd 1.890 -> 1.891 NFL, 5.805 -> 5.804 CFB; edge corr .9993/.9997; sign agrees with base 99.4%/98.8%; calibration slope 1.30 -> 1.29 both leagues (base underconfident; BFGS slope SE unreliable). So log loss is not lost to spread or slope: the arm only nudges near-zero edges (accuracy from a few coin-flip flips) while the extra fitted term adds noise, about 6e-5 LL.
Next: role arm needs a same-construction check; accuracy gains in bye_diff remain ungraded on the served path.

## ROOT CORRECTION 2026-11-03 13:55
- The pooled harness used the closing spread_line as model input while grading at the opener (scripts/mod24_more_games.py:63-73), so these results include close-minus-open line movement unavailable on Tuesday; not evidence for the served model until the opener-input rerun (see docs/lanes/mod24-more-games-2026-11-03.md).

## RERUN 2026-11-03 17:05 (measured, opener-input guarded harness; supersedes every number above)
Pre-fix determination (read): artifacts xlg10_* written 13:05-13:12 (bye_diff rerun 13:59), registry cells 13:13 and 14:02; harness fix committed 14:05 (35f7f070); all unit 1 and unit 2 numbers above used the closing line as input, except the bye_diff open-input cells recorded 14:02. Reran all 8 arms (scripts/xlg10_addone.py, xlg10_addone2.py unchanged). Base is now 754-749 NFL opener (was 786-717).
Accuracy pts [95%] P+ (flips W-L), pooled het z. Old pre-fix values struck.
- home_rest: NFL -0.40 [-1.13,+0.32] .12 (14-20), CFB -0.30 .15; pooled -0.34 [-0.79,+0.11] .07. ~~pooled +0.06 .66~~ pre-fix
- away_rest: NFL +0.00 [-0.66,+0.72] .47 (18-18), CFB -0.15 .19; pooled -0.12 [-0.43,+0.20] .23. ~~pooled +0.14 .89~~ pre-fix
- bye_diff: NFL +0.07 .57 (6-5), CFB +0.09 .67; pooled +0.08 [-0.20,+0.36] .71. ~~pooled +0.21 .95~~ pre-fix
- short_week_diff: NFL +0.60 [-0.14,+1.39] .92 (23-14), CFB +0.43 [-0.02,+0.86] .97 (51-31); pooled +0.47 [+0.09,+0.84] P+ .99, het -0.39. ~~pooled +0.12 .78~~ pre-fix. Only arm that rose; LL pooled -0.00022 P+ .04 (worse).
- rest_bye_bundle: NFL +0.53 [-0.41,+1.52] .85 (37-29), CFB +0.19 .67; pooled +0.34 [-0.29,+0.96] .86. ~~pooled +0.16 .78~~ pre-fix
- fumble_lost_diff: NFL -0.40 [-1.20,+0.39] .14 (16-22), CFB -0.17 .30; pooled -0.27 [-0.77,+0.24] .15. ~~NFL -0.73 .03 (20-31), pooled -0.19 .18~~
- int_rate_diff: NFL -0.27 .24 (14-18), CFB +0.15 [-0.11,+0.44] .85 (25-18); pooled +0.10 [-0.16,+0.37] .78. ~~NFL -0.33 .11, CFB -0.09 .00 (0-4)~~
- turnover_bundle: NFL -0.40 [-1.54,+0.67] .22 (25-31), CFB -0.26 .23; pooled -0.29 [-0.88,+0.29] .16. ~~NFL -1.06 [-2.18,-0.07] .02 (21-37)~~
Pooled log loss P+ now: home .13, away .03, bye .06, short .04, bundle .04, fumble .22, int .02, turnover .06 (all arms worse or flat on LL; no arm improves it).
Intercept-only control (constant pick, scripts/pooled_screen.py with a scratch callable): vs the 14-input base NFL opener +0.20 [-2.65,+2.90] P+ .52, CFB +1.11 [-0.49,+2.68] .92, pooled +0.88 .89; LL gain pooled +0.00185 [+0.00017,+0.00352] P+ .98. So the base ridge itself is not distinguishable from a constant on accuracy and has worse LL than a constant at the opener; every arm effect (<=0.6 pts) is below what this control shows noise to be (interval half-width ~2.7 NFL).
Conclusions changed: short_week_diff goes from unremarkable to the only accuracy candidate (pooled P+ .99, CFB .97) but LL worse; bye_diff and away_rest lose their P+ (.95, .89 -> .71, .23); rest/bye bundle and all three turnover arms turn negative or flat (the NFL turnover cells at P+ .02-.03 were close-line artefacts). Nothing closes (no refuted mechanism, no positive control bound). 16 accuracy cells re-recorded with --replace (10 unit 1, 6 unit 2); bye_diff open-input cells from 14:02 kept. Next: short_week_diff needs a leave-one-season-out check and served-path grade before any use; LL/Brier cells still unrecorded.
