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
