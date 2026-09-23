# pooled-signal-model

## Goal
MOD-20: one hierarchical model with every situational family as a shrunk term, judged as a whole on line movement toward the pick and calibration at the opener; no cell resolved alone.

## State
- 2026-09-16: queued by owner; ROADMAP row written. Population/pipeline: opener
  `per_game.parquet` via `pick_probability_fit.build_fit_population`, LOSO
  2020-2025, 1,503 graded games; margin base (`margin.py`/`features.py`) is a
  separate pipeline, not used here. Pooling family for all units:
  `pooled_signal_first_model_v1` (`--category modeling`).
- Units 1-3 (2026-09-16 to 09-18, committed/recorded): Unit 1 built the
  four-term base (`model_logit` + `composition_flag_sum` [7 flags: coach,
  division, arrests, bye, cold_visitor, protection, tank_zone] + market move
  + move available) vs model-only, +1.464 pts [+0.130,+2.660] P+ 0.9815,
  unresolved_below_power. Units 2-3 grew the pool additively (8 easy + 1
  medium column); both lost to the four-term base OOS (-0.13, -0.33 pts,
  intervals crossing zero, both unresolved_below_power, larger overfit gaps).
  Standing read: additive growth does not help. Scripts:
  `scripts/pooled_signal_paired_eval.py`, `scripts/pooled_signal_second_fit.py`,
  `scripts/pooled_signal_third_fit.py`.
- Unit 4 (2026-09-23, committed/recorded): structural variant, hierarchical
  partial-pooling logit (shared flag_sum weight mu + 7 per-flag deviations,
  kappa shrinkage grid {1..10000} selected by nested inner-LOSO). **vs
  four-term base** -1.530 pts [-2.585,-0.598], P+ 0.0005, decisive 29-52 on
  81 -- interval entirely below zero, recorded **refuted_mechanism** /
  `wrong_sign_resolved` (`pooled_signal_fourth_fit_vs_four_term`). vs
  model-only +2.262 [-0.274,+4.980], P+ 0.9545, unresolved_below_power
  (companion, `pooled_signal_fourth_fit_vs_model_only`). kappa saturated the
  grid max in all folds (inner CV wanted more shrinkage than offered; not
  extended post-hoc). Script: `scripts/pooled_signal_fourth_fit_hierarchical.py`.
  Standing read after 4 fits: served four-term base beats both additive
  growth and hierarchical reweighting of its own 7 flags OOS.

## Tried
- Unit 5 (2026-09-23, run for real, NOT YET RECORDED by this session --
  next agent runs the commands below): predeclared before fitting, not a
  reparameterisation of the 7 flags. Two interaction terms added to the
  served four-term fit, same ridge (`FIT_RIDGE=1e-3`)/standardisation, LOSO
  2020-2025, 1,503 games, via `pooled_signal_second_fit.loso_fit`:
  (i) `flag_sum_x_move_toward_home` -- composition flags detect schedule-spot
  mismatches the market may already be pricing; tests whether flag_sum's
  marginal weight is conditional on the direction the market already moved
  rather than constant.
  (ii) `model_logit_x_move_available` -- `market_move_available` is 0/1 for
  whether an independent sharp-book move signal was observed; tests whether
  `model_logit`'s weight differs between games with vs without an
  independent corroborating/contradicting market read.
  Season-block bootstrap used as primary (predeclared) for the decision-
  relevant delta vs base; week-block `paired_summary` kept as secondary/
  vs-model-only for consistency with units 1-4. Line-move-toward-pick cell
  (interaction pick vs base pick) added, reusing
  `scripts/line_move_yardstick_paired_eval.py`'s cell pattern (inherits its
  contamination caveat: both variants include the move terms). Script:
  `scripts/pooled_signal_fifth_fit_interactions.py`. Ran once for real,
  artifact `artifacts/pooled_signal_fifth_fit/20260923T202505Z/results.json`.

## Next

- 2026-09-23 root: this unit's cells are recorded (registry 7,003); the commands below are history.
- Unit 5 measured result: interaction fit **vs four-term base**, season-block
  (primary) -0.399 pts [-1.037,+0.251] P+ 0.0645, decisive 24-30 on 54 of
  1503; week-block (secondary) -0.399 [-1.149,+0.334] P+ 0.13, same decisive
  record -- both intervals cross zero (not entirely below), so this is NOT a
  resolved wrong sign; classify **unresolved_below_power**, no closing
  ground. OOS accuracy inter 56.75% vs base 57.15% vs model-only 53.36%; OOS
  Brier 0.24568/0.24539; OOS logloss 0.68453/0.68395; IS-OOS gap smaller for
  interaction (+0.133pt) than base (+0.200pt) -- no added overfit. Per-fold
  coefficients: `model_logit_x_move_available` stable negative (mean -0.187,
  range -0.392 to +0.0001, consistent sign 5/6 folds); `flag_sum_x_move_
  toward_home` small and sign-unstable (mean +0.024, range -0.009 to +0.057).
  Reliability (5 quintile bins) reasonable: predicted 0.399/0.458/0.492/
  0.532/0.600 vs observed 0.425/0.457/0.445/0.530/0.625. Line-move-toward-
  pick cell: interaction vs base pick, 0.001 pts, season interval
  [-0.013,+0.021], P+ 0.496, decisive 20-21 on 41 -- indistinguishable from
  zero, secondary/contaminated, not decision-relevant alone. vs model-only
  (companion) +3.393 pts [+0.664,+6.206] week-block, P+ 0.993, decisive
  272-221 on 493.
- Standing read after 5 fits: the served four-term base still beats every
  variant tried OOS (additive growth, hierarchical reweighting, now these two
  interactions) -- none ship. The interaction fit is the closest to the base
  yet (smallest overfit gap of any variant, negative point estimate but
  wide-crossing interval, unlike unit 4's cleanly refuted hierarchical). Not
  closing the family: only 2 of many possible structural variants tried.
  Candidate unit 6 directions (not yet predeclared): a genuinely new
  reproducible family (not derived from the 7 flags or the 2 move terms), or
  a single interaction chosen by domain reasoning rather than a 2-term sweep.
- **Root: run these to record unit 5** (registry not touched this session):
  `nfl-ats weak-signals record --name pooled_signal_fifth_fit_vs_four_term --family pooled_signal_first_model_v1 --league nfl --season-start 2020 --season-end 2025 --effect -0.39920159680638717 --effect-units accuracy_points --interval-low -1.0365922024972396 --interval-high 0.2505450254080643 --probability-positive 0.0645 --sample-games 1503 --sample-blocks 6 --classification unresolved_below_power --category modeling --source "scripts/pooled_signal_fifth_fit_interactions.py; artifacts/pooled_signal_fifth_fit/20260923T202505Z/results.json; docs/lanes/pooled-signal-model.md" --classification-evidence "Season-block interval [-1.0366,+0.2505] and week-block interval [-1.1494,+0.3336] both cross zero (upper bound positive); not a resolved wrong sign; no positive control run for this family; unresolved_below_power is the only admissible classification" --description "MOD-20 unit 5: served four-term base plus two predeclared interaction terms (flag_sum x move_toward_home, model_logit x move_available), same ridge/standardisation, LOSO 2020-2025, vs the four-term base on the same 1,503 games; -0.399 accuracy points, decisive 24-30 on 54. OOS Brier 0.2457 vs 0.2454, logloss 0.6845 vs 0.6840; IS-OOS gap smaller for the interaction fit (+0.13pt) than base (+0.20pt). model_logit x move_available coefficient stable negative across folds; flag_sum x move_toward_home small and sign-unstable. Line-move-toward-pick cell (interaction pick vs base pick) 0.001 pts, P+ 0.496, indistinguishable from zero." --notes "Unit 5, predeclared in docs/lanes/pooled-signal-model.md before fitting: not a reparameterisation of the 7 composition flags (units 1-4 exhausted that). 2 looks this unit (this + companion vs-model-only). Season-block bootstrap used as primary per predeclaration; week-block secondary agrees in sign/magnitude."`
  `nfl-ats weak-signals record --name pooled_signal_fifth_fit_vs_model_only --family pooled_signal_first_model_v1 --league nfl --season-start 2020 --season-end 2025 --effect 3.3932135728542914 --effect-units accuracy_points --interval-low 0.6639776216538749 --interval-high 6.205825573472631 --probability-positive 0.993 --sample-games 1503 --sample-blocks 107 --classification unresolved_below_power --category modeling --source "scripts/pooled_signal_fifth_fit_interactions.py; artifacts/pooled_signal_fifth_fit/20260923T202505Z/results.json; docs/lanes/pooled-signal-model.md" --classification-evidence "Companion vs-model-only read, not the decision-relevant comparison (that is vs the served four-term base, which crossed zero); per lane convention (units 1-4) recorded unresolved_below_power regardless of this cell's own P+, since MOD-20 resolves no cell alone and the family stays open" --description "MOD-20 unit 5 companion: interaction fit (four-term base + flag_sum x move_toward_home + model_logit x move_available) vs model-only picks, LOSO 2020-2025, same 1,503 games; +3.393 accuracy points, week-block interval [+0.664,+6.206], P+ 0.993, decisive 272-221 on 493." --notes "Companion, not decision-relevant alone; see pooled_signal_fifth_fit_vs_four_term for the decision-relevant unit 5 read."`

## Open
- None yet.
