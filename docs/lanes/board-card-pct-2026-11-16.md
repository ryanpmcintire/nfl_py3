# Board vs card cover-chance mismatch (2026-11-16)

## Goal
Every surface shows the same served cover chance for Week 10 picks; find which artifact each reads.

## State
- Board (docs/index.html) reads `frozen_picks` (src/nfl_ats/published_picks.py:187): last `site_publish` row of artifacts/clv_ledger/published_picks.parquet before each game's deadline (board_content.py:4423-4427).
- Card (CURRENT_PREDICTIONS.md) for locked games reads the pick-revision ledger `new_home_cover_probability` (pick_refresh.py:1873-1875 via `_served_side_rows`, synced into the main table by `_sync_card_table_rows`). The ledger only gets a row when the side flips, so it carries the flip-time chance, not the last pre-lock chance, while the card text says "latest information available before the pick locked" (pick_refresh.py:1932).
- Measured 4 of 14 disagree (board/card): CLE 53.9/52.0, MIN 53.8/50.9, NE 53.5/52.6, DAL 51.4/50.1. Other 10 equal. Sides all agree.
- Defect, not design. pick_refresh.py is pinned (PINNED_NAMES), so fix is a patch: tests/scratch/board_pct/pick_refresh_locked_score.patch (patched copy pick_refresh.py beside it). For locked games it uses the frozen published score when its team matches the ledger side.

## Tried
- verify.py with patched module: `_locked_scores` gives all 14 board values exactly. ruff clean on patched copy. pytest -k "board or card or ledger": 452 passed (source unchanged).

## Next
1. Root applies the patch to src/nfl_ats/pick_refresh.py (pinned: re-enroll vN same session), reruns the refresh card write, confirms card = board for all 14.
2. Consider extending card-ledger-check to compare percentages (card_ledger_check.py compares sides only).

## Open
- The ledger itself never records same-side probability updates; whether to log them is a policy choice.
