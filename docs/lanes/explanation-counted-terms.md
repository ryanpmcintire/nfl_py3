# Explanation counted terms

## Goal
Per-game explanation must not claim overlays chose the pick when a pick-probability model serves it; it must name the served terms (line move, hook, counted flags) and their size.

## State
Done, uncommitted. card_explanation.py: `served_terms_sentence`, `_term_effect_points`, `served_terms` arg in render_pick_text, `pick_probability` arg in explain_pick/explain_card. publishing.py passes `pick_probability=pick_probability` to explain_card. Overlay firings still stored in explanations.json overlays for lineage.
Gates: ruff format/check, mypy clean; 189 tests pass, 1 fails (test_played_card_expectation caveat, findings_content, unrelated to this change).

## Tried
Scratch regen: tests/scratch/regen_expl.py (CLE at NO now: "Pushing the cover chance up: the line has moved toward NO since it opened (about 14 points).").

## Next
Orchestrator: regenerate/publish explanations; check 14-point move effect is plausible; investigate the unrelated failing test.

## Open
Waterfall "what tips it" sentence not exercised in scratch run (waterfall feed not passed).
