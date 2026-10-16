# LEAD-98 - Tuesday spread-price skew

## Goal
Measure whether the opener's two-sided spread price (e.g. -120/+100 at an unchanged line) adds to the served four-term pick probability. Done = recorded verdict via `weak-signals record` or a serving proposal.

## State
Queued 2026-10-14 from an owner question about Week 6 HOU +2.5 at -120. Nothing run.
Prior work (read): MKT-03 `docs/novig_diagnostics.md` calibration only (price-favoured side 50.3% vs 52.1% implied, n=288, unresolved); LEAD-74 `docs/lanes/lead74.md` price changes at fixed spread (+0.83 pts, P+ 0.80, unresolved). Neither fits the static skew as a term.

## Declared before outcomes (2026-10-16)
Term `price_skew` = logit of the median across books of the no-vig home-cover probability (`odds.no_vig_probabilities` on the same book's home and away prices) using only `tue_open` quotes whose home spread equals the pool line (`tue_open_home_spread`). Games with no two-sided book at the pool line are masked: standardised term set to 0 (no fabricated price); no missing indicator. Fifth term fitted jointly with model_logit, flag sum, move, move-available, LOSO 2020-2025, Tuesday view (move zeroed) and full view. Baselines: four-term, market (0.5), model-only. Looks: 2 views x (accuracy, Brier, log loss) x 1 paired comparison = 6, plus 2 reliability tables, plus 2 price_skew coefficient-sign reads = 10; quote-covered subset and per-fold coefficients are descriptive. Script `scripts/lead98_price_skew.py`, output `tests/scratch/lead98_price_skew/`.

## Tried
2026-10-16 `scripts/lead98_price_skew.py` -> `tests/scratch/lead98_price_skew/results.json`. Coverage 1465 of 1503 games priced (2020 216/220, 2021 232/236, 2022 247/248, 2023 258/266, 2024 249/266, 2025 263/267); median 10 books; skew std 0.029 logit (almost all -110/-110). `novig.py` no longer exists; used `odds.no_vig_probabilities` directly.
LOSO OOS (four-term -> five-term): Tuesday 836-667 .24652 -> 842-661 .24679 (acc +0.41 pts [-0.13, 0.97] P+ 0.91; Brier worse, P(better) 0.04; 24 changed picks 15-9); full 863-640 .24484 -> 860-643 .24515 (acc -0.20 [-0.93, 0.53] P+ 0.27; Brier worse, P(better) 0.03; 33 changed 15-18). Model-only 800-703 .24950, market 746-757 .25. price_skew coefficient per fold -1.1 to +1.2, positive in 2 of 6 folds (unstable sign).
Recorded 4 cells `lead98_price_skew_{tuesday,full}_{accuracy,brier}_2020_2025`, unresolved_below_power. 10 looks.


## Next
Not served: loses on Brier and log loss out of season. Possible re-test: skew conditioned on key-number lines (not declared, needs its own declaration).

## Open
None for the owner.
