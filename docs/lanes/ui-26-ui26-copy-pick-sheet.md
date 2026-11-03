# UI-26 Pick sheet to copy

## Goal
Pick sheet to copy: our picks in the pool's entry order with the Best Pick and tiebreaker guess, one tap (ROADMAP UI-26).

## State
Added 2026-11-03, unmeasured. Entering picks by hand across the slate is the weekly pool task; a plain list in pool order saves it and removes transcription errors.

## Tried
Nothing yet. Novelty: UI-20 (standing lane). Grep of src/nfl_ats for 'clipboard' returned no hit.

## Next
First unit: Add a copy button on the Board that places a plain-text sheet on the clipboard, and check it at phone width.

## Open
Data: Board pick data already on the card. Zero crossing closes nothing; record unresolved results with a closing ground.
