# MKT-23

## Goal
Spread and total move decomposition: implied team-total changes from Tuesday to the deadline as fitted side terms (ROADMAP MKT-23).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: A spread move with a flat total is a view on the sides; a move with a total drop is a view on one offence (a QB or weather). Implied team totals move by half the total change plus or minus half the spread change; which offence lost points identifies the news, and the four-term probability, which uses spread move only, cannot see it.

## Tried
Novelty check only. Nearest: MKT-16..19 (spread move only), LEAD-71 (moneyline lattice), LEAD-88 (total news for the tiebreaker). Grep for implied team total and spread-total decomposition found only totals_program.md noting team totals are absent from nflverse. Differs: derives them from the two published numbers.

## Next
First unit: Tabulate Tuesday-to-deadline change in spread and total, derive implied home and away total changes, and report how often the total moves with the spread and in which direction; then fit one extra term leave-one-season-out, with the intercept-only control.

## Open
Changes: Post-opener market information, split by side. Data: Archived hourly quotes (2023-25, 799 games) and earlier Tuesday-to-close pairs. Delta spread, delta total, implied delta team totals. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03 16:03 EST)
Script scripts/mkt23_move_decomposition.py; artifacts/mkt23/{decomposition.json,move_table.parquet,record_batch.json}. Data: data/market/raw hourly spread+totals quotes 2023-25, 799 games (266/266/267), consensus = median over books of last quote at/before Tue noon ET and at/before min(kickoff, Sun 4PM ET). Sign verified: home_spread_line = -(HOME outcome line), positive = home favoured, so home TT = T/2 + hs/2, away TT = T/2 - hs/2. Measured quote step of consensus moves 0.25.
Spread moved 598 games (+158 total-only moves). Total moved same direction as spread (home-favoured up with total up) 37.3%, opposite 39.3%, flat 23.4%. Categories (share, follow rate = final margin past Tuesday opener in move direction): pure side 23.4% 54.3%; one-team 18.2% 57.8% (P+ vs 50 0.952, CI -2.3..17.0); two-team same 33.6% 53.7% (P+ 0.85); two-team opposite 24.7% 56.8% (P+ 0.94). One-team minus two-team same +4.1 pts (CI -7.3..15.9, P+ 0.76); by season -7.7, +5.6, +15.4 (unstable). 5 looks, recorded unresolved_below_power (registry/weak_signals, family mkt23_move_decomposition). Nothing closed.
## Next
Unit 2: fit one signed term (one_team indicator x move direction) on out_of_season_home_probability leave-one-season-out with intercept-only control; it enters the one calibrated probability as a fitted logit term, never a side rule. Extend to 2026 captures.

## Unit 2 result (2026-11-03 16:10 EST)
Script scripts/mkt23_unit2_term.py; artifacts/mkt23/{unit2_term.json,unit2_record_batch.json}. Base out_of_season_home_probability, REG 2023-25, pushes excluded, n 799 (598 with a spread move; 109 one-team); unmoved games carry direction 0. Logistic on base logit, LOSO (fit on two seasons, score third). Direction = sign of Tuesday-to-deadline home spread move: a deadline-view term, unknown at the Tuesday pick. 9 looks (4 arms x 2 baselines + 1).
Out-of-sample ll / Brier / acc: intercept 0.6931/.2500/.508; served base 0.6818/.2443/.573; refit base 0.6818/.2444/.575; c direction 0.6833/.2451/.570; d one-team x dir 0.6833/.2452/.561; e both 0.6840/.2455/.559.
Improvement vs refit base (ll): c -0.0015 [-.0036,.0007] P+ .078; d -0.0016 [-.0047,.0017] P+ .16; e -0.0022 [-.0056,.0013] P+ .10. Decisive flips vs served base (arm right/base right): c 30/33 of 63; d 15/25 of 40; e 28/39 of 67. Gap OOS-in: c .0019, d .0024, e .0031. Coefs unstable by fold (direction: +.14, -.02, +.01; one-team x dir: +.42, +.06, +.10), 2023 drives it. Recorded 3 cells unresolved_below_power (family mkt23_unit2_term). Not wrong_sign_resolved: intervals straddle 0.
## Next
Extend to 2026 captures; one-team term is positive in coefficient but ll worse OOS; re-test when n grows.
