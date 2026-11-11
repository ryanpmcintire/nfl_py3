# fix-picks-u4-crowd

## Goal
Crowd information (pool field shares, public ticket and money shares) at the pool line through Week 9 2026: tallies, forward-chained fitted terms on the served logit, and whether the crowd effect survives the line-value gap. Parent: fix-picks-2026-11-10.md.

## State
Scripts tests/scratch/fixpicks/u4_build.py, u4_analysis.py, u4_hist.py; data u4_games.csv (136 games W1-9, pool line = data/splash, finals = data/raw_schedules/20261110T140026Z, served = last pre-deadline row of artifacts/clv_ledger/published_picks.parquet, displayed_score = p of pick_team).
Inventory (read): field pick counts exist only after lock (W1-8 in data/splash/field/*best_picks_w01-08.parquet; W9 not captured); ticket+money (Action Network) in data/raw/public_betting_live, 2026 only, one capture every ~2 days, latest pre-cutoff capture is 0.6-3.7 h before kick/deadline so it is NOT a Tuesday-time input; ticket coverage W1 12/16, W2 7/16, W3 8/16, W4-9 all. History ticket: 278 games 2020-25 (public_betting 20260820 snapshot), money 2023-25 only. No consensus-pick feed.
Card measured 70-66 W1-9, 47-57 W3-9 (parent audit says 46-52 on W3-9; denominator differs, unreconciled). Published market_spread equals pool line on all games, so the gap must come from nflverse spread_line (close, post-decision: a diagnostic, not an input).
Tallies (crowd majority covers; exact binomial vs 0.5; 45 looks): field >=50% 53-67 p .235, >=60% 38-43, >=70% 13-14; ticket >=50% 51-61 p .395, >=70% 10-21 p .071; money >=50% 53-59, >=70% 17-19; money-minus-ticket sign 52-54. Card against field 33-22 p .177, against ticket>=70% 11-5 p .21; card with crowd 31-34 field, 25-31 ticket. None separable from chance. Crowd is on the favourite 86/120 (field), favourite covers 67/136.
Forward-chained term (fit weeks < w, score W3-9, week-block bootstrap, positive = term better): ticket coef -1.3 to -3.0, differ 33 games new side right 21-12 p .163, acc .458 to .552, but log loss -0.0118 P+ 0.38; field -1.4 to -1.7 (10-7 on 17 differ), ll P+ 0.39; money ll P+ 0.36; money-minus-ticket ll P+ 0.05 (worse, 9-8). History ticket coefficient (n=278, LOSO per season 0.29..-0.52, sign unstable, all-fit -0.03; crowd side 137-135) applied to 2026: ll +0.0002 P+ 0.76, differs on 2 games.
Gap: gap-only term differs 15 games, 12-3 p .035, ll P+ 0.45, Brier P+ 0.51. Crowd coefficient unchanged by controlling gap (ticket -2.5 to -2.6; field -1.7 to -2.2; field+gap ll +0.005 P+ 0.64, Brier P+ 0.76). Field-gap corr +0.27 (field leans toward the side the market later moves to), ticket-gap corr 0.00. Field against the market-move side 11-20 (p .15); field with it 23-21.

## Tried
Fitted with unpenalised one-parameter logistic offset (no ridge, no magic constant); hooked pool lines, no pushes.

## Next
Proposed records (family fix_picks_u4_crowd_2026, all unresolved_below_power, no closing ground): field_forward_term, ticket_forward_term, money_forward_term, money_minus_ticket_forward_term, ticket_history_applied_2026, crowd_vs_gap_control. Forward-log ticket at the Tuesday capture time if a Tuesday scrape can be added.

## Open
Week 10 ticket is the only live crowd input; fade coefficient is unstable and over-confident out of sample.
