# per-flag-weights

## Goal
Test whether each situational flag in the four-term pick probability should carry its own LOSO weight instead of the single pooled `flag_sum` weight, and whether the Tuesday card should use Tuesday-fit coefficients. Done = verdicts recorded, served formula changed only if a variant wins out of season.

## State
- 2026-10-14: owner challenged Week 6 picks LAC +8.5 (bye flag only) and HOU +2.5 (protection flag). Bye inside the card: leave-one-out +0.133 pts, P+ 0.618; standalone +0.60, P+ 0.85 (`docs/bye_edge_fade_overlay.md`).
- Per-flag variant measured and REJECTED (recorded `per_flag_weights_vs_pooled_{tuesday,full}_view_2020_2025`, refuted_mechanism / wrong_sign_resolved, closes the weighting variant only, no flag). Served formula unchanged. No Week 6 side changes. The script's Week 6 rows omit the protection and cold-visitor flags (HOU@JAX .493, SEA@DEN .501 there vs card 57.3% HOU, 53.7% DEN), so only LAC (.434 vs .460) and PHI (.561 vs .530) are valid; HOU would strengthen under per-flag (protection 0.324 > 0.266, inferred).

## Tried
`scripts/per_flag_weights_eval.py` -> `tests/scratch/per_flag_weights/results.{json,txt}`, LOSO 2020-2025, 1503 games, 20 looks. Measured OOS:
- Tuesday view (move zeroed in fit and score): pooled 836-667 (55.62%) Brier .24652; per-flag 815-688 (54.22%) .24782.
- Full view: pooled 863-640 (57.42%) .24484; per-flag 841-662 (55.95%) .24611.
- Per-flag minus pooled: Tuesday -1.40 pts [-2.68, -0.20] P+ 0.0125; full -1.46 [-2.48, -0.52] P+ 0.0006; Brier worse in both. Decisive: 44-65 and 27-49.
- Per-flag fold means (Tuesday): bye 0.151 (0.042-0.219, positive every fold), protection 0.324, coach 0.317, cold 0.377, arrests 0.249, tank 0.199, division 0.174; pooled 0.266.

## Next
1. Tuesday coefficients: score the served full-fit coefficients applied with move=0 against Tuesday-fit pooled coefficients, LOSO, on Tuesday information; serve Tuesday-fit at the lock only if it wins on Brier/log loss. Note the honest Tuesday OOS accuracy is 55.6%, not 57.4%.
2. Card copy: the confidence band text quotes full-view landing rates; check whether Tuesday-view bands should be shown at the lock.

## Declared before scoring (2026-10-16, unit Tuesday-fit)
Arms: A served full-fit coefficients with move and availability zeroed at score (src/nfl_ats/pick_probability.py:194-200 `home_probability` fillna(0) on both move terms, so the lock card is arm A); B coefficients fitted with both zeroed in training. LOSO 2020-2025, 1503 games, Tuesday information. Looks: 2 arms x 3 metrics (accuracy, Brier, log loss) + 2 paired deltas (accuracy, Brier) = 8; band landing rates descriptive. Script: scratchpad tue_fit.py.

## Result (measured, tests/scratch/tue_fit_out.json; recorded `tuesday_fit_vs_served_move0_brier_2020_2025`, unresolved_below_power)
- A served-full-fit move=0: 837-666 (55.69%) Brier .246605 LL .686360. B Tuesday-fit: 836-667 (55.62%) Brier .246517 LL .686185. 121 changed picks, B 60 - A 61.
- B minus A: Brier -0.000089 [-0.000495, +0.000313] P(B better) 0.664; accuracy -0.06 pts [-1.40, +1.32] P+ 0.446. B better Brier and LL in 5 of 6 seasons (2020 worse). Season-blocked bootstrap, 107 week blocks.
- Reliability A/B (bins <.40,.40,.45,.50,.55,.60+): n 81/276/588/315/191/52 vs 73/265/486/371/233/75; B top bin pred .625 obs .613, A .622 obs .673.
- Coefficients per fold: A model_logit .18-.34 (mean .276), flag .24-.30, move .19-.25, avail .00-.05, intercept -.04..-.06. B model_logit .25-.41 (mean .343), flag .25-.30, intercept -.008..-.024. Gain is B's higher model weight and neutral intercept; pick set nearly identical.
- Verdict: B wins on Brier and LL out of season but by an unresolved margin; side effect none on record. Serve-worthy by the stated rule.
- Serving proposal (not applied): fit a second coefficient set with move terms zeroed in `fit_pick_probability` (src/nfl_ats/pick_probability_fit.py:362; FIT_FEATURES line 46 and design loop lines 83-99), store it in the artifact beside the full-fit set (to_dict/from_payload src/nfl_ats/pick_probability.py:238-321, expected_features check line 483), and in `home_probability` (line 182-202) use the Tuesday set when the move series is all unavailable (lock card), full set once move is available.
- Card copy (Next 2): landing_rate_sentence (pick_probability.py:962-979) quotes full-view bands: strong 59.5%, slight 54.8% (n 504/484, cuts .530/.567). Same served cuts applied to Tuesday-card probabilities (A): strong 58.7% (433), lean 57.7%, slight 51.8% (581). Tuesday-fit own terciles (B, cuts .524/.562): strong 57.9%, slight 53.3%. Card says 60%/55%, so Tuesday would land 59%/52% now or 58%/53% under B. Proposal: at the lock, compute bands from the Tuesday view (B own terciles) in `_strength_bands` (pick_probability_fit.py:314-343) and quote 58%/53%; revert to full-view bands once move is available.

## Open
None.

## Card copy unit (2026-10-16, done in code, not wired)
`StrengthBand` now stores `tuesday_games`/`tuesday_accuracy` (same served cuts .530/.567, held-out rows, move zeroed); `landing_rate_sentence(model, move_used=...)`. Refit deterministic: coefficients bit-identical to prior artifact; active artifact pick_probability/20261016T221402Z (strong 58.7% n433, slight 51.8% n581 Tuesday). NOT wired: board_content.py:4003 and publishing.py:147 still call with default (full view); lock card must pass `move_used=False` when no game's move is available. Not run: existing tests.
