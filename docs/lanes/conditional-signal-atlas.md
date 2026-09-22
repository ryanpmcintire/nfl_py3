# Conditional signal atlas (MOD-19)

## Goal
Match the approved Findings mockup with an interactive explorer backed by measured, paired signal comparisons. Every pick stays inside one calibrated probability.

## State
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
Extend one predeclared fitted family at a time, with paired out-of-season fits, chronological checks, proper scores, saved rows and counted looks. Build timestamp-certified inputs before using this diagnostic as new forward evidence. Preserve the saved visual reference when adding selectors.

## Open
Individual signals and other split dimensions remain future work. Existing intervals exclude refitting/selection uncertainty. Fitted-source and model/declaration freshness guards remain in place. Active master and deployment state are recorded by Git and HANDOFF.md.
