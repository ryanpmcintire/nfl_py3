# XLG-14 College-first discovery

## Goal
Screen declared candidate families on college games first, carry only survivors to one pre-declared NFL confirmation each (ROADMAP row XLG-14). Zero crossing closes nothing; one fitted probability selects the side; every parameter leave-one-season-out; every cell a look; opener input and opener grade.

## State (2026-11-14 16:59 local)
Unit 3, luck family `xlg14_luck_family_u3`, is screened and recorded. One of five candidates passed the pre-declared rule: `fumble_recovery_luck`. Its one declared NFL confirmation ran at 16:57: opposite sign, not confirmed, unresolved (result block below). Nothing is served.

**Measured** (`scripts/xlg14_luck_family.py build` then `screen`; `artifacts/xlg14_luck_family/` build.json, summary.json, summary.txt, weak_signals_batch.json, team_values.parquet). 8,925 college games, 13 seasons; 200 within-season shuffles per candidate (seed 36, stream 3). Null log-loss mean -.00006 to -.00007 [-.00019, +.00013].

| Candidate (nonzero games) | Changed; record | Accuracy pts [95%] P+ (null pct) | Log-loss gain [95%] P+ (null pct) | In-sample; gap | Coef per SD; folds; halves 2012-18 / 2019-25 |
|---|---|---|---|---|---|
| fumble_recovery_luck (7,392) | 1,058; 541-517 | +0.27 [-0.40,+0.96] .78 (93.5) | +.00025 [-.00037,+.00085] .79 (99.5) | +.00042; +.00017 | -.058; 13/13; -.043 / -.078 |
| interception_luck (8,208) | 72; 32-40 | -0.09 [-0.28,+0.09] .16 (37.5) | -.00009 [-.00015,-.00002] .006 (40.5) | +.000001; +.00009 | +.003; 9/13; -.002 / +.014 |
| non_offensive_td (6,700) | 454; 214-240 | -0.29 [-0.74,+0.18] .12 (6.5) | -.00010 [-.00033,+.00015] .24 (32.5) | +.00006; +.00016 | +.022; 13/13; +.020 / +.025 |
| field_goal_luck (8,393) | 766; 400-366 | +0.38 [-0.20,+1.03] .89 (96.5) | +.00003 [-.00036,+.00041] .56 (89.0) | +.00016; +.00013 | -.035; 13/13; -.073 / +.010 |
| kick_block_muff (6,155) | 112; 53-59 | -0.07 [-0.29,+0.16] .30 (48.3) | -.00009 [-.00016,-.00003] .003 (47.0) | +.000001; +.00009 | +.003; 10/13; +.003 / +.002 |

Controls (**measured**): base 50.48% and log loss .69400; constant 50.72% and .69311; base plus fitted intercept 50.40% and .69403. The base is worse than a constant on log loss, so the survivor's +.00025 does not bring it to the constant.

What the survivor is and is not:
- It passed a rule fixed before outcomes: 99.5th percentile of its own null on log loss, negative coefficient in both season halves (the hypothesised over-reaction sign), 13 of 13 folds. That buys one NFL look, nothing more.
- **Inferred:** with five gating reads at a 95th-percentile bar, about 23% of no-information families pass at least one (1 - .95^5). Its size (.058 logit per SD) is under what this harness sees four times in five (.086-.109 per SD, +0.7 to +1.1 points; `docs/lanes/xlg-harness-power-2026-11-14.md`).
- The held-out gain is all in the later half (-.00005 earlier, +.00059 later). The term is zero in 2012 and thin in 2013 (own recoveries not labelled) and fumble labels are thin in 2021-24.
- The other four showed no effect this harness can see; none is closed. interception_luck and kick_block_muff have whole log-loss intervals below zero yet sit at the 40th-47th percentile of their nulls: that is the cost of one more fitted term, not a resolved wrong sign. field_goal_luck reached the 96.5th percentile on accuracy but 89th on log loss with opposite half signs, so it does not pass.

Recorded: 15 cells (`xlg14_luck_cfb_<candidate>_<units>`; accuracy_points, log_loss_improvement, brier_improvement; league cfb, category onfield, unresolved_below_power). `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Unit 3 declaration (written 16:40, before the script ran; amended about 16:45 from recording counts only, before any outcome was joined)
- History (**read**): NFL only and all unresolved: close-game/turnover luck and special-teams return terms on production, FTN turnover luck, luck-stripped base inputs (acc-u7b, -0.3 to -0.6 points), a season-lagged special-teams rating. Both leagues on the shared 14-input harness (XLG-10 opener rerun): raw fumbles-lost rate, interception rate, bundle (college -0.17, +0.15, -0.26 points); the NFL-only negative was a closing-line artefact. Nothing against an expected baseline, no non-offensive touchdowns, no special-teams term and no matched null existed for college.
- Candidates: each a team's per-game value over its games in strictly earlier weeks of the same season, home minus away, 0 with none. (1) opponent fumbles recovered minus own lost, each less the league lost share; (2) interceptions made minus thrown, each less the league share of defended passes (interceptions plus broken-up incompletions) intercepted; (3) return and defensive touchdowns for minus against, scorer read from the score columns; (4) field goals made minus expected from the same yard line in earlier seasons, own less opponents'; (5) blocked kicks plus muffed punts recovered by the punting team, for minus against. Hypothesis negative for all; rule two-sided.
- Amendment: own fumble recoveries are unlabelled in 2005-13 and broken-up passes thin in 2004-06 and 2021-24, so (1) and (2) take the league share from earlier weeks of the SAME season (a season that lacks the denominator gives share 1 and a zero term). Blocked kicks are labelled from 2014 only. Score columns are post-play (96.9% of 54,778 rushing touchdowns reproduce).
- Harness: unit 2 college frame; the instrument's add-one (one offset-logistic coefficient per SD, leave-one-season-out). Controls: intercept only; within-season shuffle. Survivor rule: log-loss gain above the 95th percentile of its own null AND equal coefficient sign in 2012-18 and 2019, 2021-25. Looks: 5 gating, 15 recorded cells, 10 half coefficients, 2 controls.
- As-of check (**measured**): five random season-weeks, later games removed and the cutoff week's own events blanked; 590 team-games, largest difference 2e-16.

## NFL confirmation: terms resolved 2026-11-14 16:57, before any outcome was joined
Declared terms unchanged (Next, first three bullets). Resolved the way the college build did (`scripts/xlg14_luck_family.py`), in `scripts/xlg14_fumble_luck_nfl_confirmation.py`:
- Fumble = run or pass play with a fumbling player, charged to the team in possession; lost = `fumble_lost`. The opponent's side is the same plays read by the defending team (college read the defending-team id). Special-teams fumbles stay out, as declared.
- Regular-season plays only (college kept its regular season type). A frame game whose two teams do not both have a regular-season row (any postseason game) carries 0, as the college join does.
- Baseline year: the SAME season (the amended college rule): league lost share = all teams' lost over fumbles in strictly earlier weeks; week 1 has none, so its term is 0. Value = (earlier-week opponent lost minus own lost, less share times opponent fumbles minus own fumbles) / earlier games; home minus away.
- Fit: one coefficient per SD of the frame column, no intercept, against the served probability (college `intercept: False`). Entered as MOD-36 template index 4 (after its four): null stream `[36, 4, 0]`, 200 within-season shuffles; 2,000-draw game bootstrap `[36, 900, 4]`.
- Outcome words: negative full-sample coefficient and log-loss percentile above 95 = confirmed; negative and at or below 95 = not confirmed; positive = opposite sign. The refutation clause's interval is the 95% game-bootstrap interval of the full-sample coefficient (same bootstrap weights); refuted only if wholly positive.
- As-of check: five random season-weeks (stream `[36, 4, 1]`), later games removed and the cutoff week's own events blanked, tolerance 1.5e-8.

## NFL confirmation result (2026-11-14 16:59; one look, run once as declared)
**Measured** (`scripts/xlg14_fumble_luck_nfl_confirmation.py build` then `confirm`; `artifacts/xlg14_luck_family/nfl_confirmation/` build.json, summary.json, summary.txt, null_draws.parquet, weak_signals_batch.json). Play-by-play snapshot 20261111T010519Z, 3,230 regular-season team-games, season lost share .43 to .49; base `artifacts/pick_probability/20261114T184614Z/per_game.parquet`, column `out_of_season_home_probability`; 1,503 games, all with both teams, 1,400 nonzero.
- As-of check: five season-weeks (2021 wk 9; 2023 wk 3, 15, 18; 2024 wk 17), 156 team-games, largest difference 0.
- Decisive record: 56 changed picks, 25-31.
- The one read: held-out log-loss gain -.00095 [-.00185, -.00015], probability_positive .009, 16th percentile of its matched null (null mean -.00045, 5th-95th [-.00129, +.00042]); bar 95th.
- Coefficient per SD +.013 [-.094, +.118], P+ .60, 54.5th percentile of the null coefficients (null SD .053). Folds 2020 -.009, 2021 +.032, 2022 +.058, 2023 +.025, 2024 -.021, 2025 +.001: the expected negative sign in 2 of 6.
- Outcome in the declared terms: **opposite sign**, so not confirmed. Not refuted: the coefficient interval spans zero and the log-loss cost sits inside the null's range, the price of one fitted term that carries no information here. `unresolved_below_power`.
- Context only, not part of the confirmation: accuracy -0.40 points [-1.33, +0.47] P+ .20 (40.5th percentile); Brier -.00047 [-.00091, -.00007] (15.5th); in-sample log loss +.00002, gap +.00098. Controls: served 55.16% and .68671; constant 49.57% and .69328; served plus fitted intercept 55.29% and .68675. Size .013 per SD beside the college -.058 and the roughly .26 per SD (+1.6 points) this NFL harness sees four times in five (**read**, power lane). **Inferred:** a college-sized effect of either sign would not show here, so this is a missing sign and percentile, not evidence of absence.
- Implication (**inferred**): the college pass did not carry over on the one look and earns no forward-log slot; at most the term could be logged forward as an unresolved candidate, and it is never a served term on this evidence. No second NFL construction, window or metric is run.
- Recorded: 3 cells `xlg14_luck_nfl_fumble_recovery_luck_<units>` (accuracy_points, log_loss_improvement, brier_improvement; league nfl, family `xlg14_luck_family_u3`, onfield, unresolved_below_power). `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
- Unit 1 (2026-11-03): MOD-41 workload replication on college, 6 cells `xlg14_mod41_cfb_u1`; residual coefficient +.0013 against NFL -.044, log loss -.00013 P+ .014; not commensurable, nothing refuted. `artifacts/xlg14/mod41_cfb.json`.
- Unit 2 (2026-11-03): schedule family (short week, bye, prior ATS, early carryover, road trip), 5 cells `xlg14_family_u2`; log-loss P+ .001 to .29; zero survivors, no NFL run. `artifacts/xlg14/family_cfb.json`. Full declarations in git history of this file.
- Unit 3 first build used earlier-season league rates; recording counts showed them wrong (.96 lost share in 2012-14), amended before the screen. `xlg_harness_power.py` ends in a bare `main()`, so its null machinery is loaded from source without that call (an import reruns 788 s and overwrites its artifacts).

## NFL confirmation as declared (2026-11-14 16:50; text unchanged; run once, result above)
- Build the same value from NFL play-by-play (`data/pbp/raw` latest snapshot per season; run or pass plays with a fumbling player = fumbles, `fumble_lost` = lost, by `posteam`, as `scripts/xlg10_addone2.py` `nfl_counts`): opponent fumbles recovered minus own lost, each less the league lost share in earlier weeks of the same season, per game over strictly earlier weeks, home minus away; same as-of check.
- Frame: MOD-36 `games()` (NFL 2020-25, opener grade, 1,503 games), base = served `out_of_season_home_probability`, never the raw `home_cover_probability_at_open`. Same offset add-one, leave-one-season-out; intercept-only control; 200 within-season shuffles (seed 36, a new stream).
- The single gating read: held-out log-loss gain as a percentile of its matched NFL null, expected coefficient sign negative. Passes only with a negative full-sample coefficient AND a percentile above the 95th. A positive coefficient with the whole interval on the wrong side is the only refutation; everything else stays `unresolved_below_power`. The NFL harness sees about .26 per SD (+1.6 points), so expect a sign and a percentile, not a resolved effect; a smaller NFL reading is magnitude, not absence. Accuracy and Brier secondary. Record league nfl in family `xlg14_luck_family_u3`. Passing is not serving: serving needs the term fitted inside the one calibrated probability.

## Next
- The one NFL look at `fumble_recovery_luck` is spent; do not re-cut it. Next unit: a third college family declared before outcomes, or split-half reliability of each luck trait (college `artifacts/xlg14_luck_family/team_values.parquet`, NFL `nfl_confirmation/team_values.parquet`) for the registry `reliability` field.

## Open
- `reliability` is unrecorded on all 18 cells (15 college, 3 NFL).
- The NFL script loads `mod36_addone_noise_floor.py` from source without its trailing `main()` (an import would rerun it and overwrite `artifacts/mod36`).
- College fumble and breakup labels vary by season; the fumble term's later-half concentration may be a recording effect.
- ROADMAP row XLG-14 and the handoff are the orchestrator's to update; nothing committed.
