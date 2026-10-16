# MOD-23 unit 6b: availability depth (dose-response and non-QB absences)

## Goal
Extend unit 6 arm B (own non-established-QB flag; 808-695 on 2020-25, +0.40, P+ 0.85) on the fair opener test (base 802-701, 1,503 decisive 2020-25; 2011-19 proxy graded at SBR open). Zero leakage; no src edits.

## State
Declared 2026-10-16 BEFORE scoring. Script scripts/mod23_unit6b.py (copy of unit 6, outputs artifacts/mod23_unit6b/). Family mod23_unit6b_availability_depth.
- B2: arm B with the flag replaced by dose = 1 - (pre-game established QB's offense snaps / team offense plays that game); same within-season-demeaned OLS on seasons strictly before.
- B3: arm B flag plus own-side group absence shares, each its own coefficient (prior seasons only). Absence share of a group = sum over its top-k established players of max(0, expected - actual snaps) / sum expected, expected = player's prior snap share x team plays that game; established = cumulative snaps earlier in the season (prior season if none). Groups (k = players on field): ol top 5 OL positions; sk top 3 WR/TE; df top 11 by defense snaps. Mapping: ol -> off epa/pass epa/rush epa/yards/sack rate/point_diff/ats_residual; sk -> off epa/pass epa/cpoe/yards/point_diff/ats_residual; df -> 6 def metrics + point_diff/ats_residual. All 15 metrics keep the QB flag.
- B4 (pooled vs season-local coefficients): skipped; arm B coefficients are already pooled over all prior seasons and a one-season local fit is noisier by construction.
- Looks: 2 arms x 2 eras = 4 accuracy cells; LL/Brier/pooled/changed-pick reported alongside. Arm B unit 6 is the reference (not a look).
- Leak check: coefficient seasons < game season asserted; zeroing all flags after 2022 wk10 leaves earlier features identical.

## Tried
Done 2026-10-16. Leak check passed (3494 early games max diff 0.0; 48956 later cells differ). Results (acc diff pts, season-block CI, P+): 2020-25 B2 801-702 -0.07 [-1.10,0.98] 0.45; B3 805-698 +0.20 [-0.51,1.06] 0.66; ref B 808-695 +0.40 [-0.25,1.27] 0.85. 2011-19 B2 +0.36 [-0.40,1.34] 0.76; B3 -0.09 [-0.94,0.63] 0.41; ref B +0.40 0.78. Pooled B2 1951-1783 +0.19 [-0.45,0.90] 0.68; B3 1945-1789 +0.03 [-0.56,0.57] 0.53; ref B +0.40 0.93. Changed picks pooled B2 72-65, B3 89-88. LL/Brier worse than base by ~0.0004/0.0002 in all arms (same as B). 4 cells recorded unresolved_below_power. Neither arm beats the binary flag.

## Next
Grade, record cells with weak-signals record.

## Open
Raw rebuild differs from features parquet on ~5% games (inherited from unit 6).
