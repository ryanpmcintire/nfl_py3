# MKT-25

## Goal
Opener revision size: this week's opener minus the team strength implied by last week's close, adjusted for known injuries (ROADMAP MKT-25).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: When the opener moves a lot from the prior close without an injury reason, the market is acting on information or overreacting; either way the size of the revision may carry signal. Fitted term: unexplained revision per side in the four-term probability.

## Tried
Novelty check only. Nearest: MOD-21 (strength from closing lines), MKT-22, MKT-23; grep 'revision' found only pick-revision ledger text. Differs: Revision of the opener versus the previous close.

## Next
First unit: Build revision size for 2013-2025, test its relation to opener-to-deadline move and cover with an intercept-only control, split-half by season parity.

## Open
Data: Local lines archive and injury captures (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
