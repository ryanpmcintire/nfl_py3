# POOL-02 Entrant-skill panel

## Goal
Test persistent entrant skill with a proper null before any entrant-weighted term. ROADMAP row POOL-02.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Entry picks are revealed at the Sunday deadline for every entrant. If any entrants are persistently sharp, early-week records predict later-week picks beyond the line; the null is 250 coin-flippers following the market. Data: Splash standings and Picks by Week, read view-only in Chrome (data/splash/field).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: pool-leader-romana.md (one named entrant, in-sample), POOL-01 (field Best Pick shares and standings), POL-04 (pick-popularity, blocked). Panel persistence across all entrants has no prior lane.

## Next
Capture all-entrant per-game picks for weeks 1-8 from Splash and run the split-half test. First unit: First unit: split-half correlation of entrant accuracy (weeks 1-4 versus 5-8) with a permutation null over entrants and a market-following null; report probability_positive.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Only one season is observable; skill persistence cannot be checked out of season until 2027.
