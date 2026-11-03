# ENG-55

## Goal
Pool-line inference audit: is the pool line always the Tuesday market plus or minus half a point (ROADMAP ENG-55).

## State
Unit 1 done 2026-11-03 17:19 EST. scripts/eng55_pool_line_audit.py -> artifacts/eng55_pool_line_audit.csv (136 games Wk1-9). Pool minus market, absolute: 0 = 66, 0.25 = 2 (Wk9), 0.5 = 53, 1.0 = 13, 2.5 = 2. Market whole in 55 games: 53 are exactly 0.5 (hook), 2 are 2.5 (ARI_NYG Wk4, LV_NYJ Wk8; first post-noon snapshot is a thin transient, later Tuesday consensus = pool 1.5, not hooks). Hook set Wk1-8 = 44 of 121, identical to POOL-08; no rerun, no registry change. Field parquet line vs capture: 0 disagreements. Wk9 hooks 9 of 15. Served path (pool_hook.served_hook_frame, board_content pool_edge) reads captured pool; hook_rows (training) infers market+-0.5 on history only (no captures). 13 games with |diff| 1.0 are half-point markets that moved between first snapshot and capture.

## Tried
Join captures to tuesday_market_lines; compared field parquet.

## Unit 2 (2026-11-03 17:23 EST) anchors
scripts/eng55_anchor_compare.py -> artifacts/eng55_anchor_compare.csv. Anchors: first post-noon snapshot (A), last consensus at or before pool capture (B), median of snapshots in noon-6pm (C). 136 games. Pool==anchor / hook(+-0.5 off whole) / clean(either) / unclean: A 66/53/119/17; B 41/30/71/65 (undefined in Wk4,5,6,9: no snapshot before capture); C 52/42/94/42 (undefined Wk5,6 etc). On the 76 games where all three exist: A and B identical (41 equal, 30 hook, 71 clean), C slightly worse (40/28/68). So A equals B whenever B exists; A is the only anchor defined everywhere. No anchor gives a clean +-0.5 everywhere. The 13 one-point games are not an anchor artifact: for PIT_NE, KC_MIA, PHI_CHI, IND_JAX A, B and C all agree (market 1.0 from pool); the pool simply sits a whole point off the consensus. Wk5 (8 of the 13) has no consensus snapshot before capture and one snapshot for TB_DAL; A is a later snapshot than the capture, so those are unverifiable by B. The 2 transients (ARI_NYG 1.5 vs A -1.0, LV_NYJ 1.5 vs A -1.0): A is a thin first snapshot; C gives -1.25 and 0.0, B for LV_NYJ is 1.0 (pool 1.5 = a hook on a whole market). Not resolved by any anchor; flagged unresolved.
Anchor use (read): served Tuesday line and hook input = A: pool_hook.py:85-108 (tuesday_market_lines, noon cutoff from capture, first median-consensus snapshot), used by served_hook_frame pool_hook.py:111-149 and board_content.py:2692-2705 (badge/pool edge). Training and POOL-08: hook_rows pool_hook.py:61-82, called pick_probability_fit.py:520, uses tue_open_home_spread (historical tue_open consensus label via build_pairing_table, clv.py ~975-992), not A; POOL-08 (scripts/pool08_hook_field_share.py:61) uses A. Only history (no pool captures) feeds training; 2026 captures do not.
Hook-set change by anchor vs A (28 games differ in hook flag or side): B drops 26 games only because B is undefined (no snapshot); on defined games B differs in IND_MIN Wk7 (A none, B hook +1) and LV_NYJ Wk8 (A none, B hook +1), and drops MIA_NYJ Wk7 and PHI_WAS Wk8 (A hook -1, B not hook, moved to 4.5/5.5). C adds GB_NE Wk9 (hook -1, 3.0) and drops MIA_NYJ, PHI_WAS, 5 Wk5-6 undefined. Side the half point favours flips on no game; changes are inclusion/exclusion only. Wk1-8 hook count: A 44, B 30 (undefined), C 32. Wk9: A 9, B 0 (undefined), C 10.
Effect: under A, served rows are unchanged by construction. A-vs-B swap would add 2 hooks (IND_MIN Wk7, LV_NYJ Wk8, both played-week cards) and remove 2 (MIA_NYJ, PHI_WAS); under C, GB_NE Wk9 would get a hook badge/term. Whether those rows changed a rendered badge was not inspected (code reading only).

## Recommendation
Keep A for the served card and board: it is the only anchor defined for every game, equals B wherever B exists, and mirrors what the pool line is offset from. Do not switch to C. Training rows: hook_rows reads the historical tue_open label, consistent with A in kind (first Tuesday consensus); re-deriving is not warranted from this audit because history has no pool captures to compare. The 2026-only discrepancy (17 unclean games) is pool lines not offset by 0.5 from any consensus; for those, served_hook_frame correctly zeroes the term (pool_hook.py:141). Proposed pinned change (describe only): none required; optionally log the 17 unclean games so the term never fires on them (already true).

## Next
Unit 3: inspect rendered Wk7-9 pages for IND_MIN, LV_NYJ, MIA_NYJ, PHI_WAS, GB_NE badges; find why Wk5 has no pre-capture snapshot.

## Open
Wk9 badge check by code reading only (no rendered page inspected).
