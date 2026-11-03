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
