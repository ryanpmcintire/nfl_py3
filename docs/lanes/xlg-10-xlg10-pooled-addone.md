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
