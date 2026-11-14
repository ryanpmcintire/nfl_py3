# XLG-13 Screen-to-served transfer audit

## Goal
When a candidate gains on a cheap screen (the 14-input opener-input harness, or college games) or on a non-served NFL base, how much of that gain appears on the served NFL model, 2020-25 opener grade. Pairs of existing results only; nothing is fitted on outcomes, nothing is served.

## Declaration (stamped 2026-11-14 18:27 -05:00 by Get-Date; before any pair statistic, after each pair's two results had been read in the lanes, so every pair is observed after its results were known)
- Kind A, screen to served: cheap = the candidate on the 14-input harness (NFL 2020-25) or a college harness; served = the same candidate on the NFL served model (M: 88-input production evaluator, for an estimator swap; P: the served out-of-season probability, for an added term). Kind B, raw to served: same NFL games; cheap = added to a non-served base (raw model probability or the unit's own refit base), served = added to the served probability. Never pooled.
- Per pair: accuracy points and log-loss gain against each side's own base, with interval and probability_positive, from the artifact or its registry cell.
- Admission: both sides exist for the metric on the same construction (marked same or approximate); one row per candidate (NFL screen preferred, the college screen counted only in the college sub-count); a candidate that is a function of the base probability is out; one-sided candidates are listed as excluded.
- Statistics per kind and metric: sign agreement; cheap-side positives that stayed positive (coin-flip tail, pairs not independent); slope through the origin (two or more pairs); Spearman, Pearson, intercept slope, permutation tail (three or more), bootstrap over pairs. Sub-counts by screen and served type. 26 looks (family total 28). A description, not a test; an interval crossing zero closes nothing.

## State (2026-11-14 18:32 -05:00; **measured**: `scripts/xlg13_screen_transfer.py --pairs artifacts/xlg13_screen_transfer/pairs.json --out .../report_pairs.json`; sources per number are in pairs.json; all unresolved_below_power)
Kind A, accuracy points, cheap [95%] P+ -> served [95%] P+ (order; construction):
- ridgecv swap +0.93 [-0.26,+2.14] .93 -> -0.73 M; elastic net +0.60 [-1.47,+2.47] .71 -> -3.19 M; accuracy-chosen alpha -0.40 [-1.55,+0.66] .22 -> -1.80 M; Huber +0.73 [-1.41,+2.78] .76 -> -0.87 M (served first; same; served sides from MOD-23 units, no interval, not re-verified after the opener-input fix).
- college-centred prior +1.13 [-0.93,+3.09] .87 -> -1.20 [-4.05,+1.42] .19 M (screen first; approximate). Log loss -.00024 -> +.00146.
- short-week differential +0.60 [-0.14,+1.39] .92 -> -0.13 [-0.73,+0.53] .37 P (screen first; approximate). Log loss -.00025 -> -.00046. Its college screen read +0.43 [-0.02,+0.86] .97.
- fumble-recovery luck, college +0.27 [-0.40,+0.96] .78 -> -0.40 [-1.33,+0.47] .20 P (screen first, one declared NFL look; same). Log loss +.00025 -> -.00095.
- early overreaction, college -0.01 [-0.27,+0.25] .44 -> -0.05 [-1.47,+1.36] .46 P (served first; approximate). Log loss -.00012 -> +.00098.
A accuracy, 8 pairs: sign agreement 2 of 8; 6 positive screens, 0 stayed positive (tail .016); served negative in 8 of 8. Spearman -0.17 [-0.97,+0.60] P+ .36 (permutation share at or above .67); Pearson -0.01 [-0.91,+0.70] P+ .49; fitted line served = -1.04 - 0.01 x screen (slope [-1.60,+1.03] P+ .49); slope through origin -1.12 [-2.44,+0.03] P+ .03. Sub-counts: NFL 14-input screens 5 positive, 0 stayed; college screens 2 positive, 0 stayed; five swaps on M average -1.56; three added terms on P read -0.13, -0.40, -0.05.
A log loss, 4 pairs: sign agreement 1 of 4; 1 positive screen, 0 stayed; served positive in 2 (both where the screen was negative). Spearman -0.40 [-1,+1] P+ .25 (exact permutation .79); Pearson -0.62 [-1,+1] P+ .25; origin slope -2.99 [-6.54,+0.41] P+ .035.
Kind B, 2 pairs: college opener-error rule +0.67 [-0.32,+1.50] .92 on its own base -> -0.53 [-2.33,+1.20] .27 served (served log loss +.00011; no cheap-side log loss on file); playoff leverage +0.68 [-1.18,+2.53] .74 on the raw model (592 games) -> -0.87 [-3.82,+2.08] .27 served (576); log loss -.00094 -> -.00060. Accuracy sign agreement 0 of 2, 2 positive and 0 stayed; origin slope -1.05 (a two-pair bootstrap takes only three values, -1.28 to -0.80; no rank statistic). Log loss: one pair, same sign.
Excluded: Questionable-habit term (served side only, -0.60; `artifacts/per16/unit4_addone.json` has no raw-base arm); short-week confidence interaction (function of the base); four luck candidates that did not pass and seven other XLG-10 arms (screen side only); zero-centre heavier shrinkage (same lever as ridgecv, 768 wins both).

Working rule restated (**inferred**): stop converting a screen gain with the 0.53 slope; that line came from four swaps and reads -0.01 with intercept -1.04 on eight. For a screen gain of +0.69 to +1.09 (what the college harness detects) the record gives a served accuracy change near -1.05 by the fitted line, -0.77 [-1.68,+0.02] to -1.22 [-2.65,+0.04] through the origin; for an added term on the served probability, -0.05 to -0.40, the cost of one more fitted coefficient. A raw-base gain has not survived on the served base (2 of 2 reversed, by about its own size). A screen or raw-base pass buys one declared served look; it is not an expected served gain. College-first stays the more powerful filter; only the translation goes. Nothing closes: 8 and 2 pairs, intervals reach zero, pairs share 1,503 NFL games.
One sentence: of the 8 results that looked positive on a cheaper or non-served harness (6 screens, 2 raw-base), 0 stayed positive on the served card on accuracy.

Registry: 4 `correlation` cells, family `xlg13_screen_transfer`, `record --batch --replace` (`artifacts/xlg13_screen_transfer/weak_signals_batch_pairs.json`): Spearman and Pearson accuracy replace the four-pair +0.80 and +0.28; Spearman and Pearson log loss are new. `weak-signals pool --league nfl --effect-units accuracy_points` exit 0, 4,642 signals. Left in the lane: both slopes (a ratio of two gains has no registry unit), the sign counts (no honest interval on dependent pairs), all of kind B (two pairs).

## Tried
- 2026-11-03: four swap pairs. Pre-fix values (closing line as input; Spearman -0.40, Pearson -0.84) struck; post-fix +0.80 and +0.28, sign agreement 1 of 4 (`report_fixed.json`; the script's `--screens --served` mode still reproduces it exactly, checked 2026-11-14).
- 2026-11-14 15:17: registry catch-up recorded the post-fix four-pair cells; superseded 18:32 by the eight-pair cells.

## Next
1. Orchestrator corrects the transfer arithmetic in `docs/lanes/xlg-harness-power-2026-11-14.md` (the "+0.44 to +0.56 served" line) to the rule above.
2. Add a row to pairs.json and rerun for every NFL confirmation of a college survivor and every forward term with a raw-base history. One declared look would give B a third pair: a raw-base arm of the Questionable-habit term (`scripts/per16_unit4_availability_addone.py`, raw column `home_cover_probability_at_open`).
3. Re-measure the four swap served gains on today's base, with intervals and log loss.

## Open
- Candidates reach a served look because the screen was positive, so shrinkage toward zero is expected; the record cannot separate that from the served base already holding the information (market move, calibration).
- The opener-error done lane calls its +0.67 base raw; `summary.json` shows a 13-season four-term refit (`base_four_term_loso_fold_betas`).
- On log loss two served reads are slightly positive where accuracy is negative (college-centred prior +.00146, P+ .71, which is the penalty; opener-error +.00011, P+ .55).
