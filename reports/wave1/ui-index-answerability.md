# Wave-1 improvement 1 — headline historical-accuracy figure on the index landing page

**Task:** `ui-index-answerability` · branch `swarm/ui-index-answerability`
**Source instruction:** `scripts/swarm/tasks/uix-index.md` improvement 1 — *surface a
headline historical-accuracy figure (with grade basis and link to track record) on the
index landing page, using the synchronized active-model manifest values already available
to the generator. Must not imply profit/stable edge; keep the existing disclaimer
architecture. Add/extend tests.* Estimated **+2–3 weighted points on rubric dimension 1**.

## What changed

### `src/nfl_ats/public_board.py`
- Added `_historical_accuracy_headline(active)` — builds the ONE owner-sanctioned headline
  figure from the synchronized active-model manifest's `historical_evaluation` block.
  [measured] Rendered this session; output inspected directly (see "Verification" below).
- Added an `active_model: Mapping[str, Any] | None = None` parameter to `render_picks_page`.
  When an active model is linked, the headline is prepended to the page body — both on the
  normal card page and on the empty-card shell — so the landing page answers "how good has
  the model been" immediately (rubric D1, answerability).
- `build_public_site` now passes `artifacts.active` (the same synchronized manifest the
  track-record page already uses) into `render_picks_page`. [measured] confirmed the call
  site and that `artifacts.active` is the `load_active_ats_model` result.
- Updated the two consolidation-law comments (in `render_picks_page` and `build_public_site`)
  to record that the headline is the deliberate, single-figure exception — not a return of
  the percentage firehose the consolidation law closed.

### `tests/test_public_board.py`
- Added `_active_history_fixture()` and five new tests (see "Tests").

## Design decisions (and why each protects the project's invariant)

1. **Reads the manifest, not a hand-typed number.** The figure, the games/correct counts,
   and the season-blocked interval all come from `historical_evaluation` via `_mapping`/`_number`.
   No accuracy literal exists in the function body. [inferred] This is the cleanest way to
   satisfy "using the synchronized active-model manifest values already available to the
   generator" — the generator already had `artifacts.active`; it just wasn't threaded into
   `render_picks_page`.

2. **Grade basis is explicit, in the visible copy.** The block states the figure is *"the
   active model's own out-of-sample record: a forced side pick for every game, graded
   against the opening line on games it never trained on."* [measured] This answers rubric
   D3 ("every number carries its grade/opener-vs-close") without conflating it with each
   game's model probability (a separate sentence says so).

3. **Two-decimal rendering (`52.05%`), deliberately.** [inferred] The manifest value is
   `0.5205`. One-decimal rounding would render `52.1%`, which collides with the *close-grade*
   figure that is homed on `track_record.html` under the canonical-figure-home law
   (`tests/test_public_board.py`'s "52.1" home guard). Rendering the precise `52.05%` (a)
   avoids that token collision entirely and (b) is more honest than a rounded digit that
   would duplicate a *different* metric's digits. The season interval still uses one decimal
   (`50.2%–54.1%`) matching the manifest's own presentation.

4. **Does not imply profit or a stable edge (binding).** [measured] The copy states
   *"A single sample, not a promise of future or profitable results"* and the existing
   two-tier disclaimer architecture is preserved untouched: `DISCLAIMER_SHORT` still renders
   once near the top (`"…A small historical edge is not proof of a profitable one."`) and
   `DISCLAIMER_FULL` once in the footer. This directly honors AGENTS.md's standing order
   never to describe the historical forced-pick accuracy as proof of a profitable or stable
   edge.

5. **Fail-open, like every other optional artifact.** [measured] With no manifest, no
   `historical_evaluation`, or no usable `accuracy`, the function returns `""` — it never
   invents a figure. The index page degrades to its prior shape.

6. **One dominant number per page is preserved.** [measured] The headline uses
   `font-size:17px`; the single inline `font-size:24px` (Panel 1's crowned stat) is
   untouched, so `test_index_has_exactly_one_24px_number_the_crowned_stat` still holds. The
   headline is a distinct element *above* the grid, separate from Panel 1's two stats, so the
   consolidation law's "exactly two stats in Panel 1" shape is retained.

## Tests added (all [measured] passing this session)
- `test_historical_accuracy_headline_reads_only_the_active_manifest` — block built from the
  manifest: `52.05%`, `1,080 of 2,075 games correct`, `50.2%`/`54.1%`, track-record link,
  explicit grade basis, no-profit wording, and crucially **no `52.1%`** (collision guard).
- `test_historical_accuracy_headline_fails_open_without_a_manifest` — `None`, `{}`, and a
  manifest missing `accuracy` all return `""`.
- `test_render_picks_page_surfaces_the_headline_historical_accuracy` — full page carries the
  figure, grade basis, link, no-profit language, and still passes `assert_public_safe`.
- `test_render_picks_page_headline_absent_without_active_model` — omitting `active_model`
  omits the headline (no synthesis) and stays safe.
- `test_render_picks_page_headline_stays_a_single_figure_not_a_firehose` — enumerates every
  default-visible percentage on the headline page; the only new ones are the headline's own
  three (`52.05%`, `50.2%`, `54.1%`) on top of hero / measured-chain / cover-chances; and
  exactly one inline `24px` number remains.

## Verification (all [measured] this session)
- `ruff format --check .` → 636 files formatted, 0 issues.
- `ruff check .` → All checks passed.
- `mypy src` → Success: no issues found in 105 source files.
- `pytest` (basetemp outside repo, per worker instructions) → **1860 passed, 5 skipped**
  (the 5 skips are pre-existing, caused by absent local nflverse/PBP/feature data and live
  artifacts in this checkout — unrelated to this change).
  - Note: an initial full run reported 1 setup `ERROR` (`FileExistsError` on the shared
    `basetemp` dir left by my earlier scoped runs). Removing the stale `basetemp` and
    re-running gave a clean 1860-passed. The error is a re-run artifact, not a code defect.
- Direct render inspection: `_historical_accuracy_headline(active)` emits the block above;
  `render_picks_page(empty_frame, active_model=active)` emits the headline plus both
  disclaimers; `render_picks_page(empty_frame)` (no active model) emits no headline.

## Rubric impact (estimated, per task brief)
- **Dimension 1 (Answerability, weight 12):** a first-time visitor now sees how good the
  model has been — the active model's own out-of-sample accuracy, graded, with a link to the
  full track record — within ~30 seconds of landing. Estimated **+2–3 weighted points**.
- **Dimension 3 (Provenance & honesty, weight 14):** explicitly preserved — grade basis
  stated, season interval shown, no profit/stable-edge implication, historical accuracy kept
  distinct from per-game probabilities, two-tier disclaimers intact. No regression.

## Files touched
- `src/nfl_ats/public_board.py` (feature + wiring)
- `tests/test_public_board.py` (5 new tests + import)
- `reports/wave1/ui-index-answerability.md` (this report)
