# UI-31

## Goal
Half-point chip: show the pool's half-point line against the market whole number on each row and which side it helps (ROADMAP UI-31).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: The pool posts half points over whole-number market lines; players see a free half point but cannot tell who it favours. Fixes the existing row in place; the number shown is the MOD-26 fitted term output.

## Tried
Novelty check only. Nearest: MOD-26 (pool hook fitted term), UI-27; grep 'hook chip' found none. Differs: Display of an existing fitted term.

## Next
First unit: Add a chip when pool and market lines differ, naming the side the hook helps; render at 1440 and 390.

## Open
Data: MOD-26 pool-line data (local). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 (2026-11-03T21:34Z)
State: shipped in working tree, not committed or published. Old badge showed only when no live quote (never on Week 9). Now shows on every unplayed row with a Tuesday gap, with the served hook term's cover-chance change (pick side, with minus without the term via the served model on stored row fields: board_content.py `_pool_hook_effect`, GameRow `pool_line_badge`/`pool_line_note`); the same note is appended to the game inspector's Why this pick text. Rendered Week 9: "1/2 pt better than Tuesday, +4.2% to cover" (CIN), "1/2 pt worse than Tuesday, -2.0% to cover" (CHI), no badge (HOU). 309 tests pass.
Next: check 390px render, then regenerate and publish site (orchestrator).
