# Tiebreaker total study: protocol (declared before any outcome was examined)

Question: does any pregame adjustment to the market total, or any rounding rule, reduce out-of-season
error in the pool tiebreaker guess (combined score of the last game), and is the served guess better
or worse than entering the market total?

Data (read): `artifacts/totals_backtest/20260901T184010Z/predictions.parquet` (walk-forward
`predicted_residual`, `market_total`, `actual_total`, 2010-2025) joined to the newest
`data/raw/*/schedules.parquet` for `gametime`. The market total is the schedule file's single
line (closing-grade; no opener exists in this file), so results are a closing-line study.
Population A: all REG games, evaluated seasons 2013-2025. Population B: last REG game of each week
(same rule as `last_game_of_week`), same seasons.

Arms (looks counted per population; 9 arms A, 9 arms B = 18 looks; family = tiebreaker guess rule):
- M0 market total (baseline; entered as floor(m + 0.5))
- M1 market + w * predicted_residual, w in {0, .1, .25, .5, .75, 1} chosen leave-one-season-out
- M2 market + s, s in {-3,-2.5,...,+1} chosen leave-one-season-out
- M3 joint (w, s) on the same grids chosen leave-one-season-out
- M4 served proxy: market + 0.1 * predicted_residual - 1.0 (constants from `src/nfl_ats/tiebreaker.py`)
- R1 floor(market), R2 ceil(market)
- R3 mode-window rounding of the market: integer in [round(m)-2, round(m)+2] with the most
  training-season games whose actual total equals it, among games with |market - m| <= 3
- R4 mode-window rounding applied to the M3 adjusted total
Held-out season is never in the training set for any weight, shade, or count table.

Metrics: MAE of the integer guess vs actual total (primary); closest-guess win share against a
simulated field of 30 entrants whose guesses are round(market + N(1.5, 6)) (constants from
`scripts/tiebreaker_low_side_shading.py`, sensitivity sd 4 and 8; 200 draws per game, seed fixed,
ties split); win-share baseline is 1/31 for an average entrant. Reported: pooled in-sample
(fit on all seasons) vs out-of-season MAE and the gap, per-season paired MAE difference vs M0,
seasons won of 13, season-blocked bootstrap probability_positive of the MAE reduction, LOSO
chosen (w, s) per fold and stability.

Decision rule (declared): recommend a served change only if an arm beats M0 out of season in at
least 9 of 13 seasons on Population A MAE and does not lose win share on Population B. A
non-winning arm is unresolved_below_power, not closed.
