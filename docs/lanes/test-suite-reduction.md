# Test suite reduction

## Goal
Produce a concrete deletion proposal targeting ~10x reduction of tests/
(233 files / ~3,870 functions, excluding tests/scratch), keeping only durable
contracts: prediction safety/leakage, chronology, evaluator arithmetic, board
rendering (small set), scheduler argv, registry closure. Write proposal to
docs/test_suite_reduction_proposal.md. Do not delete/edit any test.

## State
Done. Proposal written to `docs/test_suite_reduction_proposal.md`. No test
file touched.

## Tried
- Measured: `.tools\uv.exe run --no-sync pytest -q --durations=40
  -p no:cacheprovider` -> 4,529 passed, 9 skipped, 162 warnings, 119.04s
  (xdist). Log: `tests/scratch/pytest_baseline.log`.
- Confirmed 233 test files / 3,872 `def test_` functions via grep (matches
  owner's figure; 4 non-test support files excluded: conftest.py,
  _overlay_test_kit.py, _card_refit_test_kit.py, _board_content_fixtures.py).
- Per-file function counts: `tests/scratch/per_file_counts.txt`.
- Classified all 233 files into 15 categories by filename/import grep (not
  full reads); wrote table with KEEP/TRIM/DELETE-FILE verdict, counts, and
  runtime rationale (board-rendering setup costs are the biggest single-file
  duration outliers: 35s/32s/24s/23s/23s/21s in board_improvements/humanised/
  site/terminal) to `docs/test_suite_reduction_proposal.md`.
- Proposed kept total: ~37 files / ~423 functions (~9.2x reduction). Long-tail
  71-file list needing owner spot-check is in
  `tests/scratch/longtail_files.txt`.

## Next
- Owner reviews `docs/test_suite_reduction_proposal.md`, vetoes/adjusts
  category verdicts (especially CFB category 9 — confirm still-active vs.
  deprecated — and the category-15 long-tail default DELETE-FILE list).
- Once approved, a separate execution task deletes/trims the named files and
  reruns pytest to confirm the kept set still passes.

## Open
- CFB (`test_cfb*.py`) status unconfirmed: still-served secondary league vs.
  deprecated. Proposal defaults to TRIM-to-smoke pending owner word.
- TRIM keep-counts (categories 2, 5, 8, 10, 11, 12) are estimates; exact
  function-level cuts need owner or a follow-up agent to actually open each
  file (this pass used grep/glob classification only, per the 50-tool-call
  budget).
