# Conditional signal atlas (MOD-19)

## Goal
Match the approved Findings mockup with an interactive explorer backed by measured, paired signal comparisons. Every pick stays inside one calibrated probability.

## State
- 2026-09-22 owner rejection after deployment: the result still does not meet expectations and is not useful. Passing checks and publishing did not complete the requested feature. Do not describe this lane as accepted or finished.
- The assistant paused implementation, then emitted HOLD, creating contradictory directions. Clear verdicts describe whether conversation context is saved, not feature acceptance. The owner must not have to recover our mockup or reconstruct this failure.
- 2026-09-22 owner correction: the first UI did not match the mockup. Fixing labels or filters alone did not satisfy the request. The original image was recoverable; asking the owner to supply our own mockup was wrong.
- Read: original PNG and complete generation prompt are saved at `docs/design/mockups/conditional-signal-atlas.png` and `.md`. Use these assets for future work; never ask the owner to recover them again.
- Implemented the reference layout: signal rail, context workspace, selectable interval chart, selected-context explanation and season action. Clicking a row or using arrow/Home/End keys updates the explanation and season view. Removed the inert finding-filter buttons. Mobile layout stacks without horizontal overflow.
- The available family remains the combined game-situation term, split into weeks 1–4, 5–12 and 13–18. Illustrative offensive-line/rest/injury families and other dimensions from the mockup are not measured data and are not presented as working selectors.
- Existing research source: `registry/conditional_signal_atlas.json`, `scripts/conditional_signal_atlas.py`, and `artifacts/signal_atlas/20260921T231249832601Z`. Paired full/reduced fits and three evaluation views are unchanged; published picks are unchanged. Prior research measurements and look counts remain in ROADMAP MOD-19.
- Measured 2026-09-22: `ruff format --check .` (1119 files), `ruff check .`, `mypy src` (236 files), and full `pytest -q --basetemp <unique temporary directory>` (4529 passed, 9 skipped). Temporary-directory permissions required an isolated pytest base directory. Existing mobile CSS contract was adjusted to inspect all matching media blocks rather than only the last one; no new tests were added.
- Measured: local Edge checks passed chart click/keyboard selection, selected season content, no inert filters, desktop/mobile width and no script errors. Desktop/mobile screenshots were inspected. `nfl-ats publish-board` regenerated Findings and shared assets on all four pages.

## Tried
The earlier card/filter implementation was the wrong interpretation of the reference. The legacy independent-flip atlas is not the source of this comparison. This repair changes presentation and interaction only.

## Next
2026-09-24 decided from the saved mockup and the splits rule (memory: signals are tested in splits): the gap is data, not direction. Next unit fills the mockup's signal rail with families that already have measured paired full/reduced splits in `registry/conditional_signal_atlas.json`, adds no unmeasured families, and keeps the approved layout. The mockup at `docs/design/mockups/conditional-signal-atlas.png` is the spec.

History:
Resolve the gap between the original intended use, the saved mockup and the available data before further implementation. Do not automatically extend signal families, redesign the page, remove the tab or treat the prior options menu as an owner decision. No new direction has been agreed.

## Open
Individual signals and other split dimensions remain future work. Existing intervals exclude refitting/selection uncertainty. Fitted-source and model/declaration freshness guards remain in place. Active master and deployment state are recorded by Git and HANDOFF.md.
