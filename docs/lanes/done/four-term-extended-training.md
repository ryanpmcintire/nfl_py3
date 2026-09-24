# Four-term fit: does extended (2011-2025) training beat served (2020-2025) training?

## Goal
Predeclared before fitting. Served pick probability is a four-term ridge
logit (`FIT_FEATURES` = `model_logit`, `composition_flag_sum`,
`market_move_toward_home`, `market_move_available`,
`src/nfl_ats/pick_probability_fit.py`) fit on 1,503 opener-graded 2020-2025
games. Question: does training the same four terms on the larger
2011-2025 population (`artifacts/extended_fit_population/20260923T205910Z/
population.parquet`, 3,734 games, discrete-read `model_logit`, pre-2020
`market_move_*` = 0/unavailable) produce better held-out predictions on the
identical 1,503 2020-2025 games than training on 2020-2025 alone?

**Arms (LOSO over held-out seasons 2020-2025, both scored on the identical
1,503 games)**:
- A = served: for held season h in 2020..2025, train on
  2020-2025-minus-h (same population `build_fit_population` returns), test
  on h.
- B = extended: for held season h in 2020..2025, train on
  2011-2025-minus-h (includes all of 2011-2019 plus 2020-2025-minus-h),
  test on h.
- C (predeclared now, before seeing A vs B): B's population plus
  `pre_2020_indicator` (1 if season<2020) and
  `pre_2020_indicator * model_logit`, same LOSO-over-2020-2025 protocol.
  Tests whether letting pre-2020 rows carry a different model_logit slope
  recovers any A-vs-B gap.

**Decisive-game rule** (matches `scripts/mod18_spread_regime.py`
convention): decisive games are those where two arms' picks (>=0.5)
disagree; report each arm's record on exactly those games before the
headline pooled-effect number.

**Bootstrap**: paired season-block bootstrap via
`nfl_ats.clv.week_blocked_bootstrap(block="season")`, reused unmodified,
matching `scripts/mod18_spread_regime.py`'s `paired_accuracy_effect`
pattern. seed=20260924, samples=4000 (same constants as mod18).

**Week 3 2026 check**: current card has no graded outcome yet, so this is a
pick-flip check only (which side each arm would serve), not an accuracy
comparison. `model_logit` for these games is read from
`artifacts/waterfall_feed/20260924T161758Z/feed.json`'s
`discrete_base_home_cover_probability` (the discrete-lattice read at the
currently tracked line) via logit transform -- **not** guaranteed identical
to the true Tuesday-open value used for 2020-2025 training rows; flagged as
a caveat, not fabricated precision. `composition_flag_sum` and
`market_move_toward_home`/`available` are computed by calling the actual
production functions `signed_composition_flags_fail_open` and
`market_move_toward_home` (`src/nfl_ats/pick_probability.py`) directly on a
minimal predictions frame built from the feed's 16 games, joined to the
latest schedule snapshot.

## State
DONE. `scripts/four_term_extended_training.py` (new, ruff format+check
clean) run once for real:
`artifacts/four_term_extended_training/20260924T195332Z/` (`summary.json` +
`per_game.csv`). n_scored=1503 (assert-checked). No commits, no registry
writes, no src/ changes -- this is a read-only research look for the primary
orchestrator to review and optionally draft `nfl-ats weak-signals record`
from.

**Result: B (extended 2011-2025 training) does not beat A (served
2020-2025-only training) on the identical 1,503 held-out games. Accuracy
effect crosses zero and Brier/log loss read measurably worse for B.**

- A (served): 859-644, accuracy 0.5715, Brier 0.2454, log loss 0.6840.
- B (extended): 855-648, accuracy 0.5689, Brier 0.2461, log loss 0.6854.
- C (extended + pre-2020 indicator x model_logit): 854-649, accuracy 0.5682,
  Brier 0.2461, log loss 0.6854.
- B vs A accuracy (season-block paired bootstrap, seed 20260924,
  samples=4000): **-0.266 pts**, 95% interval [-0.914, +0.575],
  probability_positive=0.251 -- crosses zero, `unresolved_below_power` per
  AGENTS.md (not a resolved loss; interval crossing zero is not grounds for
  closure).
- C vs A accuracy: -0.333 pts [-1.289, +0.517], probability_positive=0.228.
  C vs B accuracy: -0.067 pts [-1.006, +0.633], probability_positive=0.455
  (C adds nothing over B).
- B vs A Brier effect (B minus A, positive = B worse): **+0.000754**
  [-0.000402, +0.001736], probability_positive=0.9145 (B's Brier worse than
  A's in 91% of bootstrap draws). Log loss effect: +0.00146
  [-0.000795, +0.003476], probability_positive=0.9073. C vs A reads almost
  identically (Brier probability_positive=0.887, log loss=0.896).
- Decisive games (picks disagree): B vs A n=92, A record 48-44, B record
  44-48 on exactly those games. C vs A n=125, A record 65-60, C record
  60-65.
- Reliability: A's out-of-sample predictions land in wider bins (109 games
  in [0.2,0.4), 122 in [0.6,0.8)) than B's (27 and 44 respectively) --
  extended training pulls predictions toward 0.5, consistent with the
  higher Brier/log loss.
- Per-fold coefficients: `summary.json`
  `a_served_2020_2025_training.fold_betas` /
  `b_extended_2011_2025_training.fold_betas` /
  `c_extended_pre2020_interaction_training.fold_betas`, one dict per held
  season with intercept + each feature's natural-scale coefficient (not
  reproduced here for length; read the artifact for stability check across
  the 6 folds).
- Week 3 2026 pick check (ungraded, pick-flip only, not an accuracy claim):
  16 games, **1 flip** -- `2026_03_LV_NO` (NO home): A gives NO 0.4993
  (picks LV), B gives NO 0.5008 (picks NO) -- a coin-flip-level flip right
  at the 0.5 boundary, not a confident disagreement. `model_logit` for
  these games was read from `discrete_base_home_cover_probability` in
  `artifacts/waterfall_feed/20260924T161758Z/feed.json` (currently tracked
  line, not confirmed identical to the true Tuesday-open value used for
  2020-2025 training rows) -- flagged caveat, not fabricated precision.

**Verdict**: confirms and sharpens the prior sanity look in
`docs/lanes/opener-population-backfill.md` (-0.26 to -0.46 pts, wash). This
run adds the missing pieces that lane flagged as not yet done: paired
bootstrap with intervals/probability_positive, Brier/log loss (which DO
read a probable (~91%) degradation, not just a wash), decisive-game
records, per-fold coefficients, reliability tables, and the predeclared C
arm (no better than B). Recommended next action for the root: draft
`nfl-ats weak-signals record` for
`four_term_extended_training_2011_2025_vs_served` as
`unresolved_below_power` on accuracy (interval crosses zero) while noting
the Brier/log-loss degradation is not itself grounds for closure either
(no admissible closing ground met -- not a resolved wrong sign with the
whole interval on the wrong side, and no positive control run). Extending
`market_move` to pre-2020 (still 0/unavailable for all 2,231 sbr_proxy
rows) remains the open lever if this is revisited.

## Tried
Reused `build_fit_population` for arm A's population (exact served 1,503
games), `EXTENDED_POPULATION` parquet from
`docs/lanes/opener-population-backfill.md` Unit 3 for arm B/C's training
pool, `nfl_ats.clv.week_blocked_bootstrap(block="season")` unmodified for
both the accuracy-effect and a new paired Brier/log-loss draw, and
`signed_composition_flags_fail_open`/`market_move_toward_home` (the actual
production functions) directly on a minimal predictions frame built from
the waterfall feed + schedule snapshot for the week 3 2026 check.

## Next
No further action required unless the root wants the `weak-signals record`
command actually run (draft above) or wants market_move extended to
pre-2020 to retest B with a genuinely complete extended population.

## Open
- Week 3 2026 `model_logit` provenance caveat above (tracked-line read, not
  a confirmed Tuesday-open read) -- the single flip is coin-flip-level
  regardless, so this caveat does not change the practical conclusion.
- C's interaction term does not resolve the B vs A gap; not investigated
  further (out of scope, and the predeclaration only asked for a single
  interaction test).

## Drafted `weak-signals record` command (not run; root's call)
```
nfl-ats weak-signals record \
  --name four_term_extended_training_2011_2025_vs_served \
  --description "Training the served four-term pick-probability logit on the 2011-2025 extended population (vs 2020-2025 only) does not improve held-out accuracy on the identical 1503 2020-2025 opener-graded games; Brier/log loss read probably worse." \
  --source artifacts/four_term_extended_training/20260924T195332Z/summary.json \
  --effect -0.266 \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --league nfl \
  --season-start 2020 \
  --season-end 2025 \
  --interval-low -0.914 \
  --interval-high 0.575 \
  --probability-positive 0.251 \
  --sample-games 1503 \
  --sample-blocks 6 \
  --classification-evidence "Season-block paired bootstrap (seed 20260924, samples=4000) interval crosses zero; per AGENTS.md that is expected for a real small signal at this evaluator's resolution and is not grounds for a terminal classification. Brier effect +0.000754 (probability_positive=0.9145) and log loss effect +0.00146 (probability_positive=0.9073) suggest calibration is probably worse under extended training, but neither meets an admissible closing ground (no resolved wrong sign with the whole interval on the wrong side; no positive control run)." \
  --category modeling \
  --notes "C arm (extended + pre-2020 indicator x model_logit) predeclared before seeing A vs B; reads -0.333 pts vs A, -0.067 vs B -- adds nothing. Week 3 2026 pick-flip check: 1/16 games flips (2026_03_LV_NO), coin-flip-level (0.4993 vs 0.5008)."
```
