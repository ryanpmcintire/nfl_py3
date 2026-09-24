# Four-term nonlinear check (MOD-07 ensemble/stacking)

## Goal
Test whether nonlinear structure in the same four served terms
(`FIT_FEATURES` = model_logit, flag_sum, move_toward_home, move_available)
beats the served ridge logistic out of season, on the identical 1,503
opener-graded 2020-2025 games from `build_fit_population`.

## Predeclared arms (locked before any fit)
- Base: served ridge logistic, `FIT_FEATURES`, ridge=`FIT_RIDGE` (1e-3), LOSO by
  season (train on 5 seasons, test held season), identical to
  `pick_probability_fit.fit_pick_probability` out-of-season fold logic.
- Arm N1: `sklearn.ensemble.HistGradientBoostingClassifier(max_depth=2,
  max_iter=100, learning_rate=0.05, min_samples_leaf=50)`, no tuning, same
  four raw features (unstandardised, trees do not need scaling), same LOSO
  season folds.
- Arm N2: served logistic design plus the six pairwise products of the four
  standardised terms (model_logit*flag_sum, model_logit*move_toward_home,
  model_logit*move_available, flag_sum*move_toward_home,
  flag_sum*move_available, move_toward_home*move_available), ridge fixed at
  `FIT_RIDGE` (not retuned), same LOSO season folds.
- Comparison: paired season-block bootstrap (`week_blocked_bootstrap`,
  `block="season"`, reused from `scripts/four_term_extended_training.py` /
  `scripts/mod18_spread_regime.py` machinery) on accuracy-point effect vs
  base, plus Brier/log-loss effect, on the same scored games (LOSO-covered
  intersection). Decisive-game record (N1 vs base, N2 vs base) reported
  before any headline. Week 3 2026 pick-flip check: full-sample refit of
  base vs N1 vs N2, scored against `artifacts/waterfall_feed/20260924T161758Z`.
- Look count: base LOSO, N1 LOSO, N2 LOSO, 2 paired bootstraps (accuracy),
  2 paired bootstraps (brier/logloss), 2 decisive reports, 2 calibration
  tables, 1 week-3 full-sample refit/flip check = 11 predeclared looks.
- Zero-crossing rule applies: `probability_positive` reported, never
  "contains zero"; nothing closed on a crossing interval alone.

## State
Script `scripts/four_term_nonlinear_check.py` written and run once. Artifact:
`artifacts/four_term_nonlinear_check/20260924T202959Z/` (summary.json,
per_game.csv), 1503 scored games, matches served population's 1,503-game
LOSO count.

Results (measured from summary.json):
- Base served ridge logistic: 859-644, accuracy 0.5715, brier 0.24539,
  log_loss 0.68395.
- N1 (HGB trees, max_depth=2, max_iter=100, lr=0.05, min_leaf=50): 840-663,
  accuracy 0.5589, brier 0.24740, log_loss 0.68810. vs base accuracy effect
  -1.264 pts, interval [-2.527, -0.349], probability_positive 0.000125 (whole
  interval negative -> N1 loses this comparison, `wrong_sign_resolved`
  closing ground per AGENTS.md, refuted as an accuracy improvement). Brier/
  log-loss effect also positive (worse), probability_positive 0.9445/0.9385.
  Decisive games (201 where N1 disagrees with base): base 110-91, N1 91-110.
- N2 (logistic + 6 pairwise interactions, ridge fixed at FIT_RIDGE): 857-646,
  accuracy 0.5702, brier 0.24655, log_loss 0.68644. vs base accuracy effect
  -0.133 pts, interval [-0.944, 0.470], probability_positive 0.3925 (crosses
  zero -> `unresolved_below_power`, not closed). Brier/log-loss effect
  positive (worse) with probability_positive 0.929/0.926 but brier interval
  low end -0.000236 (also technically crosses zero) -> unresolved. Decisive
  games (70 where N2 disagrees with base): base 36-34, N2 34-36.
- Week 3 2026 (16 games, ungraded, full-sample refit): N1 flips 6 of 16 picks
  vs base (all razor-thin, base probabilities 0.499-0.519); N2 flips 1 of 16
  (NO/LV, 0.499 base vs 0.500 N2). Given N1's OOS result is resolved worse,
  none of its 6 flips should be adopted; N2's 1 flip sits inside the
  unresolved-below-power band.
- Look count actually taken: 14 (3 LOSO fits [base/N1/N2], 2 paired-accuracy
  bootstraps, 2 paired brier/log-loss bootstraps, 2 decisive reports, 3
  calibration tables, 1 full-sample refit, 1 week-3 flip check). The script's
  own `look_log`/`look_count` field only records 5 (LOSO + refit + week3
  check); the bootstrap/decisive/calibration helper calls were not wired to
  `record_look` in this script -- a gap vs. the reference scripts' pattern,
  noted here rather than fixed silently.

## Tried
- Read `src/nfl_ats/pick_probability_fit.py` for FIT_FEATURES/FIT_RIDGE/
  build_fit_population/LOSO fold logic.
- Read `scripts/four_term_extended_training.py` and
  `scripts/mod18_spread_regime.py` for reusable LOSO/bootstrap/decisive/
  calibration/week-3-flip machinery; sklearn 1.9.0 confirmed present in the
  uv env.

## Next
Answered: no, neither shallow-model arm beats the served ridge logistic OOS
on these four terms. N1 (trees) is resolved worse (wrong-sign, whole interval
negative). N2 (interactions) is unresolved_below_power, not closed. If a
write-up is going to call N2 settled, first run
`nfl-ats weak-signals record` for the N2-vs-base accuracy comparison
(unresolved_below_power) before that write-up ships; N1 can be recorded
`closed_negative` citing `wrong_sign_resolved` (whole interval on the wrong
side). Orchestrator decides whether MOD-07 stays open for a different
nonlinear family (these two fixed-hyperparameter arms do not clear the
served baseline) -- no ROADMAP/registry edits made here.

## Open
Fix `record_look` coverage in the script if it is reused again (bootstrap/
decisive/calibration helpers currently silent on the look log; true look
count for this run was 14, not the 5 the script printed). No src/ changes,
no registry writes, no commits made in this lane.
