# MOD-57 P2-Q market reaction speed (line drift and surprise v hindsight gap)

## Goal
Measure whether the closing line reacts too slowly or too fast to what a season has shown, using P2-O's hindsight gap as a descriptive label only, and whether following it is worth anything on real results. Family mod57_how_wrong_is_the_line (shared). Script scripts/mod57_p2q_market_reaction_speed.py, outputs artifacts/mod57_p2q/.

## State
2026-11-11 10:38:26 DECLARED before any outcome read. Novelty (read): MOD-21 (market_derived_ratings_v1) fitted a decay-0.9 ridge-3.0 walk-forward rating from closing spreads, added opener-minus-implied residual to the served logit, 2020-25 LOSO: -1.93 accuracy points [-3.32, -0.29], P+ .009 (wrong sign), Brier +0.00175 P+ .976. MKT-25 fitted a trailing 17-week closing-spread rating, opener minus implied, 9 looks: log-loss gain -0.0001 to -0.0003, P+ .29-.48; slope on prior-line move +0.047 P+ .84, on injury news -0.04 P+ .17. Both grade rating-minus-opener as a served-model input; neither grades a team's market-rating drift or surprise-to-date against a hindsight label. Not a duplicate. Prior count in shard mod57_how_wrong_is_the_line: 24 cells (read).
- M(g): least squares of the close (home spread positive = home favoured) on team indicators + home term, games of that season strictly before g's date, min-norm; a team with fewer than 4 prior games (P2-O's MIN_PRIOR_GAMES) uses the previous season's final fitted rating (NaN for 2009, those games dropped from D1).
- D1 = home-minus-away of (M before g minus the team's start-of-season market rating = previous season's final). D2 = home-minus-away of the team's mean (margin minus close) over games before g's date this season (needs a prior game both sides). D3 = D2 where week 2-6, else 0.
- Label H-gap = R_H - close (P2-O, future games, LEAK, outcome only in descriptive slopes). Real = margin - close.
- Read 1: OLS slope of label on D, season-block bootstrap (P2-O slope_boot), overall and weeks 2-6/7-12/13-18; D3 has no variance outside weeks 2-6 so only overall and 2-6 cells exist. Read 2: slope on real residual; walk-forward sign from seasons before the test season, test 2015-25, pick = home if sign*D>0, vs close and vs Tuesday opener (2020-25), overall and weeks 2-6, decisive games only (margin != line), record first.
- Truncation check: ten games, drop rows from g's date on (same-day margin/line zeroed), D1 and D2 must not change.

## Tried
2026-11-11 10:41:24 RUN DONE, 32 cells recorded (shard 31 KB to 73 KB). Measured, artifacts/mod57_p2q/run.out: truncation diff 0.0 on 10 games; 2,048 of 8,862 team-games use prior-season rating. Slopes on hindsight gap (mechanical for D2: the label contains the same earlier games): D1 +0.32 [+0.23,+0.41], thirds +0.26/+0.30/+0.35; D2 +0.21, thirds +0.12/+0.34/+0.50 (all P+ 1.00). On real residual: D1 +0.15 [+0.04,+0.28], thirds -0.02/+0.26/+0.10; D2 -0.02 [-0.07,+0.02], thirds -0.057 [-0.117,-0.001] P+ .022 / +0.009 / +0.087. Walk-forward fade-the-surprise (D2 sign negative every season): close 1376-1259 52.2% (wk 2-6 457-359 56.0%, +6.0 [+2.9,+8.9]); opener 725-681 51.6% (wk 2-6 236-193 55.0%, +5.0 [+2.5,+7.8]). D1 follow-drift: close 50.7%, opener 52.2%.

## Next
Raw per-game surprise (D2) in weeks 2-6 is the only lead; test it as a fitted term through the stack with leave-one-season-out, not a flip. Mechanism check: shrinkage of early noisy means.

## Open
D3 overall equals weeks 2-6 by construction (zero elsewhere); 20 slope cells and 12 accuracy cells recorded, D3 thirds 7-12/13-18 undefined.
