# ENG-60

## Goal
Neutral-site and international orientation: home flag and home-field term for games without a true home team (ROADMAP ENG-60).

## State
Unit 1 audit done 2026-11-03T23:05Z. Script scripts/eng60_neutral_audit.py (ruff clean), output artifacts/eng60/audit.log and artifacts/eng60/neutral_games.csv. Nothing recorded in the registry; no src edits.
- measured: 83 neutral games 2011-2026 (67 REG, 15 SB, 1 WC; 80 graded; 2026 has 9 incl. wk10/11 ungraded, no line yet). 28 are in the opener-graded 2020-25 artifact.
- Home field in features: features.py:481-482 zeroes the 55 Elo HFA when location is Neutral; features.py:700 builds neutral_site; constants.py:317/351 feed it to the model; target ats_margin = result - spread_line (features.py:49), so HFA is carried only by the market line, there is no model HFA constant. travel_geometry.py:276 and experiment_runner.py:823/929 also branch on neutral. Full HFA at neutral is NOT given.
- Orientation: schedule home == game_id home for all games (the 209 home_team != id mismatches are relocated-franchise abbreviations); result == home_score - away_score, 0 mismatches. Splash 2026: 136 rows, 0 orientation mismatches, 7 neutral games spread identical to schedule. Provider archive (272 rows, 2025): 0 home mismatches. Grader: margin_vs_open == result - open for 28/28, correct_at_open independent recompute 0 mismatches. Served ledger: 129 graded rows, 0 outcome/settle mismatches; only 2026_04_IND_WAS (neutral) is in it (pick flipped HOME to AWAY, won).
- Open question: opener archive vs nflverse sign disagreement at 2 neutral games: 2020_13_BUF_SF (open -2.5 vs nflverse +1.5; SBR open +1.0/close +1.5 agrees with nflverse) and 2024_06_JAX_CHI (open +2.0 vs nflverse -1.5). Original quotes not loadable locally (pairing table empty from data/market) so unresolved. Grades do not flip under either sign (both games pick/outcome same); the model pick could change since spread_line is an input.
- SBR (2009-21) flags 31 neutral vs nflverse 51; none the reverse.
- Neutral opener record 14-14 (probability rule 11-17) vs non-neutral 780-695; model picks nominal home 9 of 28 (0.321) vs 0.470, mean residual -0.97 vs -0.10: leans away, not toward home (inferred; p about 0.1, not tested).

## Tried
Novelty check; unit 1 audit above. Nearest: MKT-24, ENG-55.

## Next
Unit 2: resolve BUF_SF / JAX_CHI sign from a third source (odds_tape, scores), then re-score the two picks with the right sign. Also check 2026 neutral games weeks 1,3,5,6,7 against the served board archive (ledger only covers IND_WAS).

## Open
Rules: audit only, no research verdict. Whether nflverse labels the nominal home arbitrarily at SB/international games does not affect grades since lines and results share the label.
