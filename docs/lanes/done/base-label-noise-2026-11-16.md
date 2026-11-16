# base-label-noise-2026-11-16

## Goal
Attack label noise in the served base (ridge alpha 10, 88 inputs, target ats_margin = result minus closing line): train on one untried denoised label, grade at the opener through the served pick-probability stage 2020-25. Code scripts/base_label_noise.py, outputs artifacts/base_label_noise/.

## State
2026-11-16 15:40 EST DECLARED before any outcome read. Label H (hindsight gap): for each training game, the same-season least-squares team-rating margin with the game's own result removed (15 or so other games; mod57_p2o hindsight(), home-field intercept) minus that game's closing spread_line. Mechanism: the realised margin carries about 13.5 points of single-game noise; the rating from the other games of the season averages it out while keeping the team-strength gap the line missed (its slope on real results 0.365 in P2-O vs served residual 0.05 in P2-N). Training target only, inputs unchanged. Leak guard: H replaces ats_margin only for seasons strictly before the test season (in-season weekly training rows keep the real label because H uses later games). Grade: base logit replaced by H-trained opener logit, served four-term logistic refit leave-one-season-out (lead98 loso, BASE terms) on the 1,503 served games (served fitter excludes pushes; push rows not in this grade). Primary look: log loss vs served out-of-season probability, season-block and week-block intervals; also accuracy P+, changed-pick record, residual calibration slope on the real opener margin vs the base residual's, raw model-alone accuracy, LOSO coefficients per fold, in-sample vs out-of-sample gap. Looks: 1 label x 1 era; 4 metric cells recorded.

## Tried (novelty check, read)
- acc-u5 (mod24_denoised_target): T1 total EPA diff (is the real margin, R2 .99), T2 scrimmage EPA+success with fumbles neutralised and wp 0.05-0.95 (SD -17%, flips 40% of picks, 2020-25 -0.73 P+ .28), T3 LOSO blend real/T2 (-1.00). All unresolved_below_power. So EPA, success-rate and realised/expected blend labels are taken.
- mod23 unit 8: label-reliability per-game weights (-1.13). unit 10/10r: winsorized and Huber targets, garbage-time label. unit 9b: two-sided home/away points (identical by algebra; per-side alpha/inputs).
- mod57 P2-P: hindsight-gap label, but elastic net/boosting on 295 columns graded on MAE and slope only, never the 88-input ridge through the served card. P4-A: later-market label = leak (slope 1.7). P2-O: hindsight ratings themselves (not servable).
- Untried and closest: H on the served ridge through the served stage (this unit).

RESULT (measured, artifacts/base_label_noise/report.json; served reproduction max abs 0.0, 1,503 games): label H pre-2020 SD 5.82 v 13.40, corr .17, slope of real on H 0.39. Served stage: 812-691 v 829-674, accuracy -1.13 [-2.61, +0.65] P+ .089 (season block); log loss 0.68794 v 0.68670, -0.00124 [-0.00216, -0.0003] P+ .005 season block, week block [-0.0032, +0.0007] P+ .12; changed picks 92-109 of 201; worse in 5 of 6 seasons; in-sample/OOS gap 0.27 pts (base 0.60). Model alone 761-742 v 802-701, -2.73 [-4.35, -1.34] P+ 0; log loss -0.0063 [-0.0085, -0.0040]. Residual calibration slope on real opener margin: H 0.13 [-0.10, 0.30] v base 0.32 [0.07, 0.55] (residual corr .18). Served model_logit fold coefficients H .13/.01/.00/.16/-.06/.09 v base .38/.22/.26/.37/.30/.33.
Recorded 4 cells family base_label_noise_hindsight: served accuracy and served log loss unresolved_below_power; model-alone accuracy and log loss refuted_mechanism wrong_sign_resolved. `weak-signals pool` loads (4649 signals).

## Next
Done; move to done/. Inference: denoising by season ratings throws away the base's within-game information (residual correlation .18) and the served stage nearly drops the term. The closest untried variant is a LOSO-chosen blend of real ats_margin and H (T3 analogue with hindsight instead of EPA); the acc-u5 T3 result (-1.00) argues against it.

## Open
Season-block (6 blocks) and week-block intervals disagree for served log loss; left unresolved.
