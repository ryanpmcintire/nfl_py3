# Futures anchor (outside-leads candidate 1), 2026-11-16

## Goal
Preseason championship futures as an anchor term (Fodor, Patterson and Shank 2025: bettors keep backing preseason favourites). College-first screen on the XLG-14 harness, then one declared NFL confirmation for a survivor.

## State
- Part 1 done (measured): scripts/futures_anchor.py pull/rejoin -> data/external/preseason_futures/ (futures_long.parquet, preseason_close.parquet, inventory.json, local .gitignore). Policy entry sportsoddshistory_championship_futures in config/source_policies.json. robots.txt allows (only wp-admin disallowed).
- NFL: covers.com/sportsoddshistory/nfl-reg/?y=Y&a=sb, 2009-2026, 32 teams every season, publication point "prior to Wk 1" (preseason close; weekly in-season prices through the year also stored); join 100% of repo team-seasons. No preseason opening number on this page.
- College: cfb-main/?y=Y&a=nc, 2012-2025 (14 seasons), 42-130 listed teams per season, preseason close = last August as-of column, opening = January column; names joined 100% after aliases (FIELD row dropped). Result column never stored.

## Declaration (stamped 2026-11-16T20:59:35Z, before any outcome joined)
Term futures_gap per team at season s, week w: percentile rank of preseason-close implied championship probability among listed teams minus percentile rank of opener-implied strength among the same teams, where opener strength is a least-squares rating (team columns plus home-field column, neutral = 0) fitted to openers of season s weeks < w only. Game term = home minus away; zero when either team is unlisted or unrated (week 1). Expected sign NEGATIVE (team rated higher by preseason futures than by current openers is overpriced, covers less). Second look futures_gap_resid_ats: futures_gap residualised by pooled least squares on home-minus-away share of strictly earlier same-season games covered against the opener. Harness: xlg14_luck_family.read_candidate via xlg_harness_power (LOSO ridge college base at opener, 200 within-season shuffles, stream 11); survivor = held-out log-loss percentile above the null 95th and same coefficient sign in both season halves. Family futures_anchor_family_2026_11_16: 2 candidates x 3 metrics = 6 cells, 2 gating reads. Declared NFL confirmation (not run): same term on nfl_frame 2020-2025 (opener = tue_open_home_spread, futures at "prior to Wk 1"), one read, survivor rule identical, only for a college survivor.

## Result (measured 2026-11-16, artifacts/futures_anchor/summary.txt)
- College 13 seasons (2012-2019, 2021-2025; harness drops 2020), 8,925 games, 3,994 with both teams listed and rated.
- futures_gap: changed 369, record 188-181; acc +0.078 [-0.336,+0.493] pp 0.652 (pct 81.0); ll -0.000034 [-0.000299,+0.000217] pp 0.395 (pct 73.0); coef +0.0245/SD, folds 13/13 same sign, halves +0.0098/+0.0329. Not a survivor. Sign opposite to declared (positive), interval not wholly wrong-side: unresolved_below_power.
- futures_gap_resid_ats: changed 399, record 212-187; acc +0.280 [-0.146,+0.717] pp 0.894 (pct 94.2); ll +0.000002 pp 0.503 (pct 83.0); coef +0.0263, folds 13/13, halves +0.0124/+0.0342. Not a survivor (gating ll pct 83 < 95).
- Recorded 6 cells (weak-signals record --batch, family futures_anchor_family_2026_11_16); pool --league nfl --effect-units accuracy_points ran clean.

## Tried
- First pull left 5 unmatched college names; fixed with aliases via rejoin (no refetch).

## Next
- No survivor, so the declared NFL confirmation is not run. Possible follow-up (new declaration, new family look): the consistent positive sign (13/13 folds) suggests futures carry information the openers lag, the reverse of the anchor mechanism; a declared reverse-sign term would be a fresh look, not a re-read.

## Open
- NFL page has no preseason opening price; college has January opening price (unused).
