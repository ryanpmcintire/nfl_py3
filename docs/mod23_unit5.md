# MOD-23 unit 5 - coverage matchup

Script `scripts/mod23_unit5.py`; outputs `artifacts/mod23_unit5/` (report.json, terms.parquet, per-arm per_game parquet, weak_signals_batch.json). Numbers **measured** this session. Baseline reproduced through the same machinery: 802-701 on 1,503 decisive games, identical to the artifact.

## Step 1: coverage availability (participation snapshot 20260813T131635Z)
Non-null share of plays (blank strings counted as null; man/zone valid = MAN_COVERAGE or ZONE_COVERAGE; coverage type valid = COVER_0/1/2/3/4/6):

| season | man/zone | coverage type |
|---|---|---|
| 2016, 2017 | 0% | 0% |
| 2018-2022 | 38-39% | 37-38% |
| 2023-2025 | 49-50% | 43-45% |

Usable: 2018 onward (pass plays only). Columns are zero-filled for games before 2018 (2011-2017 training rows), so 2020 predictions train on only about two seasons of nonzero terms.

## Step 2: arms (declared before scoring, 3 looks)
Enforcement: per team, each game's state is the sum over that team's previous 16 games with data (`shift(1).rolling(16)`); league means are cumulative over gamedays strictly before the game day. Nothing from the game itself or later enters.

Term for offense X vs defense Y: sum over classes c of `rate_Y[c] * (epa_X[c] - league_epa[c])`. `rate_Y` is Y's share of covered snaps in class c and `epa_X[c]` is X's EPA per play against class c, both shrunk toward league values with 100 plays. Columns home, away, diff (home minus away).
- (a) man vs zone (2 classes); (b) coverage type (6 buckets); (c) both. Added to the 90 served inputs, ridge alpha 10.

## Results (2020-2025 at the opener; base 802-701, log loss 0.6969, Brier 0.2517; market-even 0.6931, 0.25)
| arm | record | diff pts [95% block interval] | P(diff>0) | log loss | Brier |
|---|---|---|---|---|---|
| a man/zone | 809-694 | +0.47 [-0.66, 1.66] | 0.771 | 0.6964 | 0.2515 |
| b coverage type | 805-698 | +0.20 [-1.06, 1.48] | 0.600 | 0.7377 | 0.2531 |
| c both | 801-702 | -0.07 [-1.75, 1.57] | 0.454 | 0.7392 | 0.2539 |

Blocks: 107 season-weeks. Per-season diff points (2020..2025): a 0.45, 0.42, -0.40, 1.88, 0.38, 0.00; b 1.82, 0.00, 0.81, -0.75, -1.13, 0.75; c -0.91, 0.85, -2.02, 1.13, -1.88, 2.25. Arms b and c log loss is dominated by 2020 (+0.277 and +0.279 vs base); other seasons are within 0.004 of base (inferred: an extreme prediction from terms fit on two nonzero seasons; not investigated).

## Does the matchup term predict opener-to-cover margin alone (no fitting, 1,537 games, 107 blocks)
| term | correlation | interval | P(>0) |
|---|---|---|---|
| man/zone diff | +0.004 | [-0.055, 0.061] | 0.543 |
| coverage-type diff | +0.001 | [-0.056, 0.059] | 0.509 |
| both | +0.003 | [-0.056, 0.060] | 0.527 |

The term carries no detectable standalone information. Arm a's +0.47 is consistent with noise around zero.

## Looks and records
Looks: 3 outer arms plus 3 correlation reads = 6, one family. Recorded `unresolved_below_power` x6 (`mod23_unit5_*`, family `mod23_unit5_coverage_matchup`). No closing ground applies. Decision implication: nothing to serve; arm a is best on record, log loss and Brier but with a near-zero correlate, so it is not promoted.
