# MOD-57 U3 - deadline fair line

## Goal
Test whether deadline spread juice (T1) and deadline moneyline divergence (T2) carry cover information beyond the half-point-quantized move, as single added terms to the served four terms; primary population = games with abs(close - Tuesday open) <= 0.5.

## State (2026-11-11 08:31)
Novelty (read): LEAD-98 = Tuesday static price skew at pool line (not deadline). LEAD-74 = within-spread price change, same-book, 2024-25 only, historical opener proxy, own pressure term (not a deadline-vs-Tuesday cross-book juice term; adjacent family, shares its look count). ml_spread_divergence = Tuesday-opener moneyline, 2020-21, on weak_stack. No deadline version of either T1 or T2 scored, so unit proceeds. Both terms are variants of the price-skew/LEAD-74/ml-divergence families and share those look counts.
Deadline snapshot (read): src/nfl_ats/sharp_book_movement_features.py:56-64 and :117-119 with pick_probability.py:47,767 (Sunday version: last quote with observed_at before kickoff, bookmaker_last_update <= observed_at). Sign: home_spread_line positive when home favoured (read from quotes: line -3.0 -> 3.0).
Declaration (before outcomes): T1 = logit(median across books no-vig home cover prob at book's last pre-kick spread/price) - logit(same at Tuesday open, lead98 skew_table). T2 = median over books of logit(no-vig ML home win) - logit(median ML win prob of earlier-season book quotes at the same deadline home spread value); seasons before row season only, so 2020 masked to 0. Missing -> 0 (design fillna). Ridge FIT_RIDGE 1e-3 (metadata ridge 0.001). LOSO 2020-25 on artifact 20261111T011859Z. Populations all and primary. 12 looks, family mod57_deadline_fair_line.
Script: scripts/mod57_u3_deadline_fair_line.py -> tests/scratch/mod57/u3/results.json.

## Tried (2026-11-11 08:35)
Ran scripts/mod57_u3_deadline_fair_line.py once (exit 0); results tests/scratch/mod57/u3/results.json; 12 cells recorded (family mod57_deadline_fair_line, unresolved_below_power).
All 1503 games: four-term 829-674 ll .6867; +juice 820-683 (acc -0.60 [-2.32,1.07] P+ .23; ll P+ .48; 145 changed 68-77); +ML divergence 825-678 (acc -0.27 [-1.34,0.79] P+ .29; Brier/ll P(improve) .03; 66 changed 31-35).
Primary (837 games, move <= 0.5): four-term 430-407; +juice 437-400 (acc +0.84 [-1.67,3.34] P+ .73; Brier P+ .77; ll -0.0014 [-0.0050,0.0022] P+ .78; 101 changed 54-47); +ML 434-403 (acc +0.47 [-0.98,1.97] P+ .70; ll P(improve) .31; 44 changed 24-20).
Juice coef per fold 1.43-3.57 (6/6 positive, term std 0.038); ML coef -0.19..1.20 (5/6). In-sample minus OOS accuracy gap: four-term 1.19 pts primary / 0.60 all; juice 1.43 / 0.80.
Live 2026: captures carry no moneyline at all and only 3 books; 0 of 133 games have two books with spread plus ML.

## Next
Juice is the only lead (primary P+ ~.73-.78, all-games ~.23-.48): sign is stable but small; do not serve. ML needs live moneyline capture first.

## Open
T2 masked all of 2020 (no earlier season). 2020-22 quotes are sparse.
