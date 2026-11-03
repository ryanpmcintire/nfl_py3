# ENG-59

## Goal
Moneyline vs spread favourite consistency across free on-disk sources, 2011-2026 (ROADMAP ENG-59).

## State
Unit 1 done Tue, Nov  3, 2026  6:18:53 PM. Script scripts/eng59_ml_spread_audit.py; artifacts/eng59/audit.csv (644 flag rows, 490 games), scorecard_2026_crosscheck.csv, run.log. Games 4368; sources nflverse (spread+ML 4232/4231), SBR (2957), Tuesday opener tue_open/close (1537), hourly captures first/last (1931 spread, 1930 ML), Kaggle 2025 close (272). Thresholds derived: logit(ML p) vs spread slope ~0.138/pt (all sources agree); flag |clipped-sd z| > normal quantile at 1-0.5/n (2.9-3.7); sign conflicts judged within open or close stage.
Flags: favourite_disagrees nflv 13 (12 near-pickem, |spread| <=1.5, ML p within 0.51), sbr 2, cap_last 27, cap_first 27; ML non-monotone nflv 10, sbr 11, cap_last 63, cap_first 71, kaggle 3; spread magnitude outlier vs peers nflv 75, tue_close 50, tue_open 19; within-stage sign conflicts open 246 close 27.
Impact: tue_open dissents from the majority at 47 games, all inside the capture book range (genuine moves), so 0 opener grades change and no served side changes. 2020_13_BUF_SF: captures -2.75..-3.0 Nov 26-30 then +1.5 by Dec 7; opener -2.5 correct, nflverse +1.5 is the close. 2024_06_JAX_CHI: captures show +2.0..2.5 CHI favoured Oct 7-9 (ML home -136), flipped to -1.5 by Oct 12; opener +2.0 correct. Both are line moves, not sign errors; ENG-60 open question closed. cap_sp_first is a stale placeholder at times (JAX_CHI Aug -1.0), not an opener; sbr_open often a late opener (197 dissents).
2026 scorecard: 129 rows, 3 games with dissent (LA_DEN nflv +1.5 vs decision -2.5, genuine move; IND_WAS neutral; NO_NYG pick-em). No grade change found.

## Tried
Majority vote over 8 spread and 5 ML sources; captured-range support test for opener.

## Next
Optional: 2021_15_NE_IND close spread -1.0 vs capture last +2.5 (close-graded only, never gates). Odds_tape spreads not used (no ML).

## Open
Look count: 5 ML-spread pairs, 3 peer targets, 2 stage conflicts. Audit only, no research verdict.
