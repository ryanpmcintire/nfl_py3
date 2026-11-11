# MOD-57 P4-A - later-market label for early-season line error (2026-11-11 13:56)

## Goal
Build label = (home minus away market rating fitted on closing lines of weeks w+3..w+8 of the same season, plus fitted home term) minus this game's close; alternative w+1..w+4. Measure its quality against the hindsight label, weeks 1-6. No predictor fitted here.

## State
- Novelty (read): no prior later-market LABEL. Nearest: MKT-20 (close minus open as target, line_move_target_v1), MOD-21, MKT-25, P2-Q (past drift as feature).
- Script scripts/mod57_p4a_future_market_label.py (ruff clean); artifacts/mod57_p4a/{future_market_label.parquet,results.json,batch.json,run.out}. Primary 3,085 games (176/yr, 193-195 from 2021), alternative 3,616; skips games with under 3 later weeks.
- Weeks 1-6 (1,560 games): primary slope +1.709, alternative +2.448, hindsight +0.399 (same games); |label| 2.88 / 2.50 v 5.53; picks v close 66.9% / 72.1% v hindsight 57.4%; v opener 65.3% / 69.8% v 58.6%; corr with hindsight gap 0.595 / 0.352.
- Controls (measured): shuffled teams slope +0.049 / +0.095, picks ~51-53%; past-weeks placebo (w-8..w-3) slope +0.081 overall, picks 50.7%.
- 4 cells recorded (family mod57_future_market_label, unresolved); pool exit 0.
- Reading (inferred): slope above 1 and rising as the window nears the game means the label contains the game's own result via the market re-rating the teams after it. Not a clean target; the 67-72% picks are own-result leakage, not edge.

## Tried
One primary and one alternative window, declared before results; no tuning.

## Next
Do not fit a predictor on this label. A clean version needs the own-game effect removed (e.g. later-week lines with this game's market reaction netted out, or ratings from lines that precede any result-driven move), or return to the hindsight label.

## Open
Whether any later-line construction can exclude own-game reaction; unknown.
