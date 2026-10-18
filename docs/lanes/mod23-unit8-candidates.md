# MOD-23 unit 8: untested base-model families (inventory, no fits)

## Goal
Name up to three families not yet tested that attack a named base limit; declare the top one for implementation. Base 802-701 (53.36%) 2020-25; 2011-19 graded at SBR open (base 1142-1089).

## State (2026-10-18, read from lanes/ROADMAP; pool summary run: eligible 4,068 cells, accuracy-points pool excludes zero only via ~2,600 effective signals, guidance = confirm once predeclared)
Tested on the base alone, all unresolved_below_power unless noted: u1 nested family/alpha (-1.06; alpha-only pooled -1.02 [-1.73,-0.29] P+0), u2 six QB/MOD-22 arms (-0.13..-0.93), u3 opponent-adjusted/compact net (-0.8..-1.46), u4 blends/shrink (-1.13; proper-score gains are calibration), u5 man/zone (+0.47 P+.77) and 5b repair, u6 availability-matched history (arm B pooled +0.40 P+.93, best of 15 looks), 6b depth, u7 target=opener (-0.07), u7 RB group B5 (+0.08), acc-u3 NGS (worse), acc-u5 denoised PBP target (3 arms), acc-u7b luck-stripped inputs (-0.3..-0.6), acc-u7a injury/OL history, acc-u8 prediction markets and acc-u9 beat reporters (acquisition only), pageviews (discarded), XLG-09 opener-error pooling (-1.0, refuted line-move mechanism), recency flat, trees worse, 2011+ training no better. Limit: information (team-quality ceiling 0.013 pts), SE ~1.3 pts.

## Candidates (ranked)
1. Label-reliability weighted training (declared below). Limit: label noise per game. Closest: acc-u5 (changes target values via EPA, R2 .66) and recency weights (time only). Differs: target untouched, per-game weights from measured outcome variance.
2. Per-input EWMA span/retention chosen LOSO (base fixed span 8, retention .67). Limit: history signal-to-noise. Closest: recency-row weighting (flat), acc-u7b. Needs builder monkeypatch (like mod24_u7b); prior-work grep for "span" in registry still needs a closer check before launch.
3. Two-sided points labels (home and away points fitted separately, margin by difference; 2 labels per game). Closest: mod17 joint margin/total residual (graded as cover on a different pipeline). Highest overlap, lowest rank.

## Declaration, candidate 1 (scripts/mod23_unit8.py, scripts only)
- Weights: w_g = 1/s2(stratum_g). Strata from completed-game facts: QB changed mid-game (snap_counts top QB vs starter), Q4-start margin band (0-7, 8-16, 17+), count of non-offensive TDs (pbp, 2009+). s2 = out-of-fold residual variance of the base per stratum, estimated only on seasons strictly before the scored season (no constant tuned). Weighted ridge alpha 10 same 88 inputs.
- Leak check: weights use training games only; zeroing outcomes after a cutoff leaves earlier weights/features identical; assert train seasons < test season.
- Data on disk: data/processed/game_features_weak_stack.parquet, game_features_pbp.parquet, data/players/raw/*/snap_counts.parquet (2013+; before 2013 strata use pbp only), pbp.
- Era split: 2011-19 (SBR open) and 2020-25 (Tuesday open), base reproduced 802-701 first.
- Grade: decisive accuracy paired vs base, season-block bootstrap with probability_positive; log loss/Brier vs base and market-only alongside; per-season d_acc; flips record.
- Looks: 2 (two eras) plus 3 single-stratum ablations reported uncounted; family mod23_unit8_label_weights; state count.
- Closing grounds: only whole interval below zero (wrong_sign_resolved) or proven positive control; else unresolved_below_power via weak-signals record.

## Next
Implement scripts/mod23_unit8.py per declaration (single-threaded, BelowNormal), then record.

## Open
Does weighting move picks at all (expect tens of flips)? Candidate 2 prior-work grep.
