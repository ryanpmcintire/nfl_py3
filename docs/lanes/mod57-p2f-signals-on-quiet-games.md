# mod57-p2f signals on quiet games

## Goal
Which built signals carry cover information on quiet games (abs open_move <= 0.5, 837 games) versus moved (>= 1.0, 653), 2020-25, judged together against a within-season label-permutation null; deliver pooling candidates. Nothing fitted.

## State (2026-11-11 09:38)
- MOD-19 atlas splits only on week_in_season and spread_band; market state is not a split, so the table was built from disk. 94 signals included (read from per_game.parquet 20261111T011859Z: 7 flags, composition_flag_sum, leader_median_net, model_logit, served-minus-model logit; apm_unit_opener_eval features.parquet diff_* (75, home-minus-away; mechanism-negative ones sign-flipped, suffix _NEG), apm rating diffs, expected_lineup_loss_cx5 diffs, friday_designation_momentum diff).
- Left out (no per-game values found on disk by scan of artifacts/ for game_id tables): officials/crew, forecast weather beyond cold_visitor, news/attention, lead106 injury language (only population/scored parquet, not scanned into table), mod10 graph terms, handle/pool-field, FTN scheme, QB-specific luck. flag_interim_hc and flag_precip are constant 0. market_move_toward_home excluded (it is the state itself).
- Measured: null max95 of max |diff z| = 3.39; real max 2.98 (0 beyond). Count beyond 2 SE: real 8, null mean 4.44. Looks 282 (94x2 states + 94 differences).
- Stand-outs (opposite signs by state, same sign in both season halves): leader_median_net quiet -2.10 (p_pos 0.08) vs moved +2.15; served_minus_model_logit quiet -0.61 vs moved +1.35. Recorded in registry/weak_signals/mod57_signals_on_quiet_games.json.
- Files: tests/scratch/mod57/p2f/{engine.py,results.csv,ranked_candidates.csv,corr_quiet.csv,summary.json}.

## Tried
- Full engine run (2000 bootstraps, 2000 permutations); recorded family summary plus 3 lowest-permutation-p signals.

## Next
- Pool unit: declare family from ranked_candidates.csv (27 signals, quiet slope >0 and p_pos >= all-games p_pos); flag_coach (130 nonzero, 80-50 style cover 0.615) is least correlated with the rest; continuity and success-rate groups correlate 0.8+; fit leave-one-season-out.
- Scan lead106, mod10, officials parquet tables not covered here.

## Open
- Weak-signal record of flag_coach and other positives not done; sign of apm_def_rating_diff unverified.
