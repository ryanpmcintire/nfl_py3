# MOD-44

## Goal
Head-coach tenure: games under the current head coach, first-year and interim, as a margin term (ROADMAP MOD-44).

## State
Unit 1 DONE Tue, Nov  3, 2026  4:31:10 PM. scripts/mod44_coach_tenure.py; outputs artifacts/mod44_unit1/ (cells.csv, loso.json, batch.json). 16 looks (8 residual cells, 8 LOSO arms), family mod44_coach_tenure, all recorded unresolved_below_power. States from schedule coach columns: first_year (new HC stint starting wk1), interim (stint starting mid-season), established; 2009 starters censored. Rows: 2011-19 SBR-open proxy n 2231, 2020-25 opener n 1503, not pooled. Team-games first_year/interim: 949/65 and 622/73.
Shrunk residual pp (P+): first_year -1.15 (.24) proxy, -2.61 (.11) opener; interim +6.97 (.86), +6.16 (.85); established +.19, +.49; games slope -.32 (.18), +.68 (.96) per game.
LOSO term vs served (flips won-lost first): first_year 40-42 / 95-96; interim 26-13 / 25-19; games_into 65-62 / 49-47. dLL (+ better) -.0003..-.0004 proxy, ~0 to -.0004 opener; all P+ <= .53 except none above. No term earns its place.

## Tried
Novelty check only. Nearest: PER-07 (coordinator changes, done), SIM-14 (aggression latent); grep 'tenure' found none. Differs: Head-coach tenure rather than coordinator change.

## Next
Unit 2 candidate: interim cell pooled (n 138 team-games, same sign both eras, P+ .85) needs more seasons; track forward. Original Next: First unit: tabulate opener residual by tenure bucket 2009-2025 with season-block bootstrap and permutation null, then LOSO add-one.

## Open
Data: nflverse schedule coach columns (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
