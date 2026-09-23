# pooled-signal-model

## Goal
MOD-20: one hierarchical model with every situational family as a shrunk term, judged as a whole on line movement toward the pick and calibration at the opener; no cell resolved alone.

## State
- 2026-09-16: queued by owner; ROADMAP row written. Population/pipeline: opener
  `per_game.parquet` via `pick_probability_fit.build_fit_population`, LOSO
  2020-2025, 1,503 graded games; margin base (`margin.py`/`features.py`) is a
  separate pipeline, not used here.
- Units 1-3 (2026-09-16 to 09-18, all committed/recorded, family
  `pooled_signal_first_model_v1`): Unit 1 built the four-term base
  (model_logit + composition_flag_sum [7 reproducible flags: coach, division,
  arrests, bye, cold_visitor, protection, tank_zone] + market move + move
  available) vs model-only, +1.464 pts [+0.130,+2.660] P+ 0.9815, recorded
  unresolved_below_power. Units 2-3 grew the pool with 8 easy + 1 medium
  additional columns; both additive variants lost to the four-term base
  out-of-season (-0.13 and -0.33 pts, both intervals crossing/touching zero,
  both unresolved_below_power) with larger IS-OOS overfit gaps. Standing read
  after unit 3: growing the pool additively does not help; next direction is
  structural. Scripts: `scripts/pooled_signal_paired_eval.py`,
  `scripts/pooled_signal_second_fit.py`, `scripts/pooled_signal_third_fit.py`.

## Tried
- Unit 4 (2026-09-23, parent-run, predeclared before running): structural
  variant, not another additive column. Hierarchical/partial-pooling logistic
  keeping the four-term base's 3 non-composition terms plus
  composition_flag_sum as a shared weight mu (all ridge=FIT_RIDGE=1e-3), PLUS
  the 7 individual flags as deviation columns penalized at kappa*FIT_RIDGE
  (kappa->inf recovers exactly the four-term base). kappa grid
  {1,3,10,30,100,300,1000,10000} fixed before running; selected per outer LOSO
  fold by nested inner LOSO CV over the 5 training seasons, minimizing mean
  held-out log loss (criterion fixed before seeing any outer result). Script:
  `scripts/pooled_signal_fourth_fit_hierarchical.py`,
  `tests/scratch/pooled_signal_fourth_fit.json`.

## Next
- Unit 4 result: hierarchical **vs four-term base** -1.530 pts
  [-2.585, -0.598], P+ 0.0005, decisive 29-52 on 81 of 1503 -- interval
  entirely below zero. Recorded **refuted_mechanism** / closing ground
  `wrong_sign_resolved` (`pooled_signal_fourth_fit_vs_four_term`) — this one
  structural variant is closed, the pooled family stays open. Hierarchical vs
  model-only +2.262 [-0.274,+4.980], P+ 0.9545, decisive 271-237 on 508,
  recorded unresolved_below_power (`pooled_signal_fourth_fit_vs_model_only`,
  companion, not decision-relevant alone). Registry now 6,983. 2 looks.
  kappa saturated the predeclared grid max (10000) in all 6 outer folds and
  in-sample — inner CV wanted more shrinkage than offered; not extended
  post-hoc (no in-sample gates). OOS accuracy hier 55.62% vs base 57.15% vs
  model-only 53.36%; OOS Brier 0.2466/0.2454/0.2517; logloss
  0.6865/0.6840/0.6969; IS-OOS gap hier +0.86pt vs base +0.20pt. Shared mu
  stable 0.23-0.29 across folds (matches base coefficient); per-flag
  deviations nonzero but modest (largest |mean| ~0.12 cold_visitor).
  Reliability (5 quintile bins) reasonably calibrated: predicted
  0.397/0.460/0.488/0.529/0.603 vs observed 0.432/0.460/0.449/0.510/0.631.
  Standing read after 4 fits: the served four-term base beats both growing
  the pool (units 2-3) and hierarchically reweighting its own 7 flags (unit
  4) out of season — do not ship any of these variants. Next: either close
  "improve the base with these 7 flags" as exhausted, or try a variant that
  is not a reparameterization of the same 7 flags (interaction terms between
  composition and market-move, or a genuinely new reproducible family) before
  another attempt.

## Open
- None yet.
