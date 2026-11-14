# XLG early overreaction in college (2026-11-14)

## Goal
Test the NFL early-season overreaction term T (MOD-57 P2-Q and P2-R) on college football, where the add-one harness sees effects of about 0.7 to 1.1 accuracy points against the NFL's 1.5 or more. Says whether the named mechanism (the line over-reacts to a small early sample) shows outside the NFL. Serves nothing.

## Declaration (written 2026-11-14 16:32:54 local, before any college outcome was joined to the term)
- Family `xlg_early_overreaction_cfb`, league cfb; `scripts/xlg_early_overreaction_cfb.py`; `artifacts/xlg_early_overreaction_cfb/`.
- Term: `surprise_values` imported from P2-Q unchanged: each team's mean of margin minus closing line over its earlier lined games this season, home minus away; undefined when either team has no earlier game. Table `cfb_game_features` (FBS v FBS regular season), seasons 2012-19 and 2021-25 (the XLG-14 population; 2020 has no opener).
- Windows by count of earlier games, not calendar week: NFL Weeks 2-6 are the games where teams bring 1 to 5 earlier results, and the mechanism names sample size; college weeks carry byes and unlined lower-division games. A game's count is the larger of the two teams' earlier lined games. Early = 1-5. Placebos = the rest of a 12-game regular season (counts 6-11) halved, as the NFL's later thirds were: mid = 6-8, late = 9 and up. No fade expected there (NFL slopes +0.009 and +0.087).
- Direction: the NFL's (fade: take the side with the lower surprise), read from the sign of the NFL fitted coefficient in `artifacts/mod57_p2r/results.json`; no college game chooses it. Diagnostic only: the sign college's own earlier seasons would choose, test seasons from 2015.
- Arms: (a) one-term fade against the close and the opener, per window; (b) slope of result minus line on surprise, per window; (c) T (surprise inside the window, zero elsewhere) added to the XLG-14 college-first harness (opener the only line input; 13 folds), per window; (d) matched null: 200 within-season shuffles through the xlg-harness-power college add-one, and the smallest planted size seen for this term's shape.
- Controls: intercept only; T shuffled among early-window games within season. Looks: 6 fade + 2 learned-sign + 6 slope + 6 added-term = 20, plus 2 control cells. Primary: early fade against the opener; early added-term log loss.
- Reading rule: an interval crossing zero closes nothing. College and NFL are not pooled; a weaker college reading is a difference of magnitude, not of presence.

## State
Done 2026-11-14 (all **measured**, `artifacts/xlg_early_overreaction_cfb/summary.txt`, results.json, batch.json; 352 s, one thread). 9,093 lined games, 13 seasons; harness 8,925; truncation check 0 on 10 games.
- Early (3,667 games): fade against the close 1852-1728, +1.73 pts [-0.03, +3.41], P+ .972, 9 of 13 seasons above half; against the opener 1820-1763, +0.80 [-0.86, +2.32], P+ .814, 9 of 13. Slope: close -0.015 [-0.058, +0.022] P+ .23 (8 of 13 seasons negative); opener +0.007 [-0.038, +0.048] P+ .63. College's own walk-forward direction matched the NFL's in 3 of 10 test seasons; following it went 1377-1435 and 1381-1436.
- Early added term (3,583 nonzero): changed picks 76-77; accuracy -0.01 [-0.27, +0.25] P+ .44; log loss -0.00012 [-0.00020, -0.00006] P+ .000 (the cost of a useless parameter, not a wrong sign: 22.5th percentile of its shuffle null, null mean -0.00007); in-sample log loss 0.00000, gap 0.00012; coefficient -0.00011 per point against the NFL's -0.0171, folds -0.0013 to +0.0015, negative in 7 of 13. Controls: intercept only 50.7%, log loss .69311; base 50.3%, .69304; shuffled term 141-143, log loss P+ .09.
- Detectable for this term's shape: .104 logit per SD (+0.97 pts) by log loss, .082 (+0.72) by accuracy; the NFL per-point coefficient at college scale is .157 per SD (+1.63 pts) and was seen in 100% of 200 planted draws.
- Mid (2,509): fade close 1250-1226, +0.48 P+ .70; opener 1216-1250, -0.69 P+ .23; slope close +0.041 P+ .87, opener +0.071 [-0.005, +0.147] P+ .97; added term changed 459-430, accuracy +0.33 [-0.31, +0.98] P+ .84, log loss +0.00008 P+ .67, coefficient +0.0092 positive in 13 of 13 folds; null percentile 93.5 (log loss), 97.5 (coefficient).
- Late (1,871): fade close 915-921, -0.16 P+ .44; opener 899-934, -0.95 P+ .24; slope opener +0.074 P+ .88; added term changed 113-110, log loss -0.00021 P+ .001, null percentile 3.5.
- Implication for the NFL term (**inferred**): same sign and smaller on the sign-rule fade (about a quarter of the NFL's +6.0 against the close, a sixth of its +5.0 against the opener; the college opener interval tops out below the NFL's lower bound); no reading in the fitted linear form T would be served in, where college had full power to see an NFL-sized coefficient. Later windows lean the other way (follow), as the NFL's did. T stays a candidate, not closed (the NFL reads stand; college differs in magnitude), but there is no cross-league support for its size: expect the 2027 refit to shrink, forward-log it from Week 2 of 2027 before serving rather than entering it on the 2020-25 fit.
- Recorded: 22 cells, family `xlg_early_overreaction_cfb`, league cfb; 20 `unresolved_below_power`, 2 `bounded_by_control` with `positive_control_bound` (early added-term accuracy and log loss; bounds the college linear form only). `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
- History: no college test of this term existed. Nearest: `cfb_bias_battery_week2_overreaction_to_week1_blowout`, XLG-14 unit 2 `prior_ats` (116-115), graph-rating `ats_residual` screens.
- `xlg_harness_power.py` ends with a bare `main()`; a plain import reruns it and overwrites its artifacts, so it is loaded from source without that call. `report` argument rebuilds batch and summary from results.json.

## Next
Orchestrator verifies the two bounded cells. If pursued: declare a sign-rule or capped form of T before looking (the sign rule reads positive in college where the linear term reads zero; a new look in both leagues).

## Open
- The table has no lower-division games, so counts run about one low. 200 draws per planted size. The mid-window follow reading (93rd to 97th percentile) is a placebo that is not flat; recorded, not chased.
