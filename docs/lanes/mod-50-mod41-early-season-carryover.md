# MOD-50

## Goal
MOD-41 early-season coverage: carry the prior season's final games into weeks 1-4 so the term exists for all games (ROADMAP MOD-50).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: MOD-41 is zero for 361 of 1,503 openers (weeks 1-4) because it needs four prior games, so a quarter of the card never sees the term. Prior-season games across an offseason gap are a different load population. Fitted term: carry-over rows enter with their own coefficient, not the in-season one.

## Tried
Novelty check only. Nearest: MOD-41 unit 2 (unusable rows listed, not filled), PER-07 early-season handling. Differs: fills the unusable rows with a separately fitted carry-over, not the in-season term extended.

## Next
First unit for one agent: Build the window from the last prior-season games with a flag for weeks 1-4, fit a separate carry-over coefficient leave-one-season-out on the served base, and report the weeks 1-4 gain, the in-season coefficient stability and the look count.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
