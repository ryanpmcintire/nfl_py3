# Win totals as a preseason prior (2026-11-16)

## Goal
Test whether the preseason market view (season win total; second look Super Bowl odds) is a prior that early-season lines drift away from too far, in the MOD-57 hindsight-gap frame, and if it survives, one declared served look at the opener.

## Declaration (stamped 2026-11-16T21:02:42Z, before any outcome joined)
- Family `win_totals_drift_2026_11_16`, 2 looks (wt_drift, sb_drift) x harness cells (gap and margin labels, weeks 2-6 primary, 7-12 placebo).
- Per team, season s, week w: percentile rank among the 32 teams of the preseason number (win total, early-September close; or Super Bowl implied probability "prior to Wk 1") minus percentile rank of current market rating. Market rating = least-squares team ratings plus a home column (neutral = 0, minimum-norm solution) fitted to lines of season s games with week < w. Game term = home minus away. Week 1 = 0.
- Deviation from the brief, declared here: openers (tue_open_home_spread) exist only 2020+, while the harness trains on 2015+, so the market rating uses earlier-week closing lines (spread_line) for every season. All are public before the week-w Tuesday opener.
- Expected sign POSITIVE (home rated higher preseason than now covers more: fade the drift).
- Survivor rule: column model beats intercept-only control and real percentile above the within-season shuffle 95th on the primary gap or margin cell, placebo not stronger. Survivor gets ONE served look: opener 2020-25 weeks 2-6, served stage, pushes kept.
- Raw win total used without juice adjustment (converting juice needs a win-distribution SD not measured here).

## State
- Features built (measured): artifacts/win_totals_test/features.parquet, 2009-2026 REG games, wt_drift nonzero 4307, sb_drift 4313; truncation check 12 games week>=2 (future weeks removed, own-week lines blanked) max diff 0.0; corr wt/sb 0.875; SD week 2 0.39 falls to ~0.2 by week 5 (features_summary.json). Team codes mapped OAK/SD/STL to LV/LAC/LA; 2011 IND has no win total (term 0 for its games).
- Harness done, both exit 0 (artifacts/win_totals_test/<tag>_table.txt, _results.json). 2015-2025 test, 841 primary games, 939 placebo.
- Result (measured). wt_drift weeks 2-6: gap share -0.0034 vs intercept-only -0.0003, shuffle percentile 0.044, close 442-385 vs intercept 443-384, folds 0+/7-; margin share -0.0031 (pct 0.42), folds 1+/0-. sb_drift weeks 2-6: the elastic net zeroed the term in all 11 folds, so its results equal intercept-only (gap pct 0.64, margin pct 0.67). No survivor, so the declared served look at the opener was NOT run.
- Placebo weeks 7-12, gap label: wt share +0.038 (pct 1.000, folds 0+/11-, in-sample gap +0.006, EL-added +0.023 p+ 0.99); sb share +0.020 (pct 1.000, 0+/11-). Margin label placebo: wt pct 0.23, sb pct 0.97 (folds 0+/8-), mae_gain negative on both. Sign NEGATIVE: from week 7 on, the line stays anchored to the preseason view (under-reacts), the reverse of the declared fade. The label is hindsight strength, which correlates with how the in-season market rating moves, so this is not a playable result until a fresh look tests it on the real margin.
- Recorded (measured): 4 cells, `weak-signals record --batch` on wt_drift_batch.json and sb_drift_batch.json (2 each, primary gap and margin, unresolved_below_power, mae_improvement), family win_totals_drift_2026_11_16; `pool --league nfl --effect-units accuracy_points` exit 0. Placebo cells are in the results json, not the registry.

## Tried
- Opener-based market rating was not possible before 2020; used earlier-week closing lines (declared).

## Next
- Optional fresh declared look, not a re-read: negative-sign preseason-anchor term, weeks 7-12 primary, graded on the real margin at the opener 2020-25 through the served stage with pushes kept. The margin placebo (sb pct 0.97, wt 0.23) is the only real-outcome evidence so far and is mixed.

## Open

## Weeks 7-12 anchor look
### Declaration (stamped 2026-11-16T21:50:14Z, before any outcome joined to this look)
- One fresh look, family `win_totals_anchor_2026_11_16`. Term wt_drift as built in artifacts/win_totals_test/features.parquet (same as-of construction, truncation check max diff 0.0), weeks 7-12, REG 2020-25, home minus away, scaled per SD on that sample. Expected sign NEGATIVE (follow the drift: line stays anchored to the preseason view).
- Grade at the opener through the served stage: population `build_fit_population(keep_pushes=True)`, offset = eng62 `stage_one` out-of-season logit, pushes kept with label 0.5. One offset-logistic coefficient, no intercept, leave-one-season-out over 2020-25.
- Gating read: held-out log-loss gain against the served stage, season-block bootstrap interval (2000 draws over 6 seasons), percentile among 200 within-season shuffles of the term. Pass = null percentile above 95 AND full-sample coefficient negative AND negative coefficient fitted separately on 2020-22 and 2023-25. Context: accuracy P+, Brier, changed picks and record (weeks 7-12 and all games, pushes counted separately), fold coefficients, in-sample minus held-out, intercept-only control, correlation with served logit and with the MOD-57 surprise-to-date (T before its weeks 2-6 indicator).
- Looks: 1 gating read; 3 recorded cells (accuracy, log loss, Brier). A pass writes a 2026 Weeks 11-12 forward-log declaration here; nothing is scheduled or served.
### Result (measured; scripts/win_totals_anchor.py, artifacts/win_totals_anchor/results.json, exit 0, 47 s)
- 494 games weeks 7-12 2020-25 (11 pushes kept at 0.5), 481 nonzero; served-stage reproduction gap 0.0.
- Coefficient -0.079 per SD (expected negative), negative in 6/6 folds (-0.054 to -0.124); halves 2020-22 -0.043, 2023-25 -0.104 (both negative). Null coefficient SD 0.085.
- Held-out log-loss gain -0.000056 [-0.00295, +0.00199] season-block, P+ 0.51, null percentile 83.5 (bar 95; null 5-95 [-0.0045, +0.0016]). Brier -0.000030, P+ 0.50, pct 83.5. Accuracy -2.02 pts [-5.63, +1.42], P+ 0.13, pct 7.3.
- Changed picks 54: 22-29 against the term (3 pushes). Weeks 7-12 261-222 v served 268-215; all games 822-681 v 829-674.
- In-sample minus held-out: ll +0.00083, acc +1.42 pts. Intercept-only control ll +0.00030 (beats the term).
- Correlation: served logit -0.03; surprise to date weeks 7-12 -0.32; MOD-57 T weeks 2-6 -0.06.
- Declared pass FAILED (null pct 83.5 < 95), so no 2026 Weeks 11-12 forward-log declaration. Sign stable but unresolved_below_power, not closed (no wrong-sign resolution, no positive control).
- Recorded (measured): 3 cells, family win_totals_anchor_2026_11_16, `weak-signals record --batch` exit 0 (recorded 3); `pool --league nfl --effect-units accuracy_points` exit 0.
