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

## Unit 1 in flight 2026-11-03 17:45 ET
scripts/mod50_early_carryover.py running in background (bg id bcvk8edox), log in scratchpad m50.log, output artifacts/mod50/unit1_early_carryover.json. Declared first: variants (a) prior-season carry window, (b) shorter window min(3, n-1), (c) no term; each early-only LOSO + merged whole-season + interaction; 6 cells under family mod50_mod41_early.

## Unit 1 result 2026-11-03 22:55Z
Source artifacts/mod50/unit1_early_carryover.json (361 early rows of 1,503; opener grade, served base, LOSO 2020-25). Decisive records are term-right vs base-right.
- (a) prior-season carry, weeks 1-4 only: decisive 18-23 of 41; log-loss gain +0.0000028 [-0.0116, +0.0088], P+ 0.554; accuracy -1.39 pts, P+ 0.224. Merged whole season: decisive 98-101 of 199; gain +0.00087 [-0.0020, +0.0035], P+ 0.735 (main term alone: decisive 157-131 of 288, +0.00404 [-0.0012, +0.0094], P+ 0.94). Interaction +0.065 [+0.022, +0.112], P+ 0.999: early coef sign flips relative to main -0.044 (early total +0.021).
- (b) window min(3, n-1), 181 rows with value: weeks 1-4 decisive 4-7 of 11; gain -0.00325 [-0.0087, +0.0006], P+ 0.055. Merged: decisive 161-153 of 314; +0.00366 [-0.0009, +0.0085], P+ 0.932 (below main's 0.00404). Interaction +0.032 [-0.040, +0.100], P+ 0.84.
- (c) no term = current main.
Looks: 6 recorded cells (family mod50_mod41_early). Verdict (inferred): neither variant fills the early rows usefully; (a) early gain is zero and its merged gain is smaller than main, the interaction says the early-season effect is not the in-season one; (b) early is slightly negative, merged is no better than main. No variant beats (c). Not served. All 6 recorded unresolved_below_power; no closing ground applies. Batch: artifacts/mod50/unit1_weak_signals_batch.json. ruff clean.

## Next (updated)
Forward-only re-check at week 5+ no earlier; nothing further for this lane unless a pre-declared averaged construction is tested.
