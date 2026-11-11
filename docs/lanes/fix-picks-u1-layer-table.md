# fix-picks-u1-layer-table

## Goal
Grade every layer of the pick chain on finished 2026 games W1-W9 at the pool Tuesday line; save one per-game table for reuse. Parent: fix-picks-2026-11-10.md.

## State
Script tests/scratch/fixpicks/u1.py (measured 2026-11-10). Outputs: tests/scratch/fixpicks/u1_games.csv (136 games), u1_layers.txt. Records are W-L at the pool line, pushes dropped, p = exact binomial two-sided vs 50%.
- Data: pool line = latest data/splash file per week (home_spread positive = home favored); card = last published_picks row <= deadline, Tuesday = first row; finals from data/raw_schedules/20261110T140026Z (W9 present, 15 games); leader line = median of last bovada/williamhill_us/mybookieag spread quote <= deadline (all-book fallback, n_leader_books column); Tuesday-noon market = median of last quote <= tuesday_noon_et_cutoff_utc. gap = leader line minus pool home_spread (positive = market makes home a bigger favourite than the pool, so HOME is the value side). better-number side: pool_minus_tue_market > 0 -> AWAY.
- W1-9 (n=136): raw model 74-62 54.4% p.35; Tuesday card 70-66 51.5%; final card 70-66 51.5% p.80; card!=model n=46 card 21-25 (model side 25-21); Tue->final flips n=30 flipped-to 15-15; value |gap|>=0.5 n=100 value side 55-45, card with 30-25, against 20-25; >=1.0 n=65 value side 40-25 61.5% p.08, card with 19-14, against 11-21 p.11; better-number n=104 side 56-48, card with 24-23, against 25-32.
- W3-9 (n=104): raw 53-51; Tue card 49-55; final card 47-57 45.2% p.38; card!=model n=34 card 14-20; flips n=24 flipped-to 11-13; value >=0.5 n=83 value side 48-35 57.8% p.19, card with 23-19, against 16-25 p.21; >=1.0 n=62 value side 37-25 59.7% p.16, card with 16-14, against 11-21 34.4% p.11; better-number n=92 side 50-42, card with 19-20, against 22-31.
- Final card W3-9 by split: home picks 22-35 (p.11), away 25-22; fav 22-29, dog 25-28; spread<=3 22-14, 3.5-6.5 19-29, >=7 6-14 (p.12); primetime 15-6 (p.08), non-primetime 32-51 (p.048). By week card (raw): W3 6-10 (10-6), W4 9-7 (8-8), W5 8-7 (7-8), W6 6-8 (7-7), W7 7-7 (11-3), W8 6-8 (3-11), W9 5-10 (7-8); W1 9-7, W2 14-2.
- Look count: about 20 splits per window, 2 windows; nothing is a finding on its own.

## Tried
- Stack term decomposition: served home prob read from each artifact's explanations.json model_probability; model_term = coef*logit(raw), other_terms_sum = remainder (flag+move+hook). Per-term flag/hook values are not stored per game for 2026; move_delta/move_pick_side come from the pick_revisions ledger only where a revision row exists (partial). deciding_term column is coarse (move if ledger movement side equals the card, else hook_or_other).

## Next
- Reconcile n=136 and card 70-66 against the 11-08 audit (69-61, 130 games): likely W5-W8 bye-week game sets and W1-2 early artifacts without baseline_forecast_artifact (resolved by timestamp). Not checked.
- Raw side uses the artifact's own home_cover_probability at that artifact's market line, not re-evaluated at the pool line.

## Open
- served_home_prob_final missing for 1 game (explanations absent in an early artifact).
