# fix-picks-u7-at-market-exact

## Goal
Replace the stated-to-at-market approximation (pick_probability.py:896-905) question with the exact quantity: the served model's non-push home cover probability when fed the Tuesday anchor line, and grade what serving it changes.

## State (2026-11-10)
- Line-dependent inputs of the served model (market_residual, weak_stack, 90 features): `spread_line` as a ridge feature (read: margin_feature_columns), point = spread + ridge raw + home-side offset bucketed by |line| (outcomes.py:_score_methods, home_side_location), discrete band conditioning on the line and the three-way grade at the line (mass_preserving_lattice.band_read). Nothing else in the feature list is derived from the current line.
- Weekly artifacts do NOT save the fitted ridge. Exact recompute = refit via `fit_margin_models_for_week` on data/processed/game_features_weak_stack.parquet, `apply_external_lines`, `model.predict(center_offset)`, `serve_discrete_three_way` with `fit_production_discrete_push_reader` (as pick_refresh.py:1080-1100). Script tests/scratch/fixpicks/u7_exact.py -> u7_exact.csv (W1-10, 2026). Key-line pick read is not applied (inapplicable at half-point pool lines; a whole-number anchor on 3/7 not modelled).
- Reproduction check (measured): recomputed stated at pool vs deadline stated, 136 games, mean abs diff .0072, max .092, corr .970.
- 80 hook games (71 in W1-9, 9 in W10), mean mass .067. Aligned move stated->at-market: approximation +.0359, exact +.0114 (exact moves ~32% of the approximation; embedded share ~68% by ratio of means). Exact minus approximated in logit: mean -.025, sd .129, 5-95% [-.19,+.19]. Slope of exact on approximated .76 (logit) and .77 (probability).
- Replay W1-9 (136 games, served coefficients 20261111T011859Z, flag-free, weeks block bootstrap 4000): approximation as served 81-55 ll .6815; exact 85-51 ll .6788 (gain +.0027, 95% [-.0001,+.0053], P+ .97, 6 sides differ, exact 5-1); no conversion 85-51 ll .6787 (+.0028 [+.0003,+.0052], P+ .99, differs 6, 5-1). Controls on recomputed stated: approx 81-55 (+.0001, P+ .61), exact 85-51 (+.0021, P+ .96). On the 71 hook games only: 40-31 v 44-27.
- Week 10 (rounded served probabilities, stage-2 shift 0.308 x delta logit): no side changes. Home prob moves: NYG 50.6->52.4 (pick holds), NYJ 52.3->51.7, CLE 49.1->48.4, ATL 51.5->52.5, GB 55.4->56.1, DET 48.7->49.8 (NE +2.5 still, 50.2), LV 48.2->47.8, DAL 50.1->50.2, BAL 48.6->47.9.

## Tried
- Count of looks: 3 arms vs approximation plus 2 controls = 5 comparisons, one family; one regression pair.

## Next
- Implication (inferred, separate from closure): the approximation over-corrects; the fit's stage-1 term was at-market exact, so serving the exact at-market read (or the stated one) matches the fit. Not served by this unit; src is pinned by v8 validation. Orchestrator decides after v8 or as a new cohort.
- Proposed record (not run): `nfl-ats weak-signals record` family fix_picks_2026_11_10, cell at_market_exact_vs_approx, effect accuracy/log loss, gain +.0027 log loss, P+ .97, classification unresolved_below_power (no closing ground; whole interval not on the wrong side).
- Out-of-season fit of the embedded share (or direct exact serving) is the remaining unit; hook games' conversion could be replaced by a per-game exact recompute at refresh using the same refit already done there.

## Open
- Single-season (2026) replay, 71 hook games; no LOSO.
