# MOD-42

## Goal
Offensive-line continuity: consecutive games with the same five starters, interacted with the opponent pass-rush strength (ROADMAP MOD-42).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: A line that has kept its five starters together protects better than one reshuffled by injury; the market sees injuries but not the length of the streak. Fitted term: continuity-streak difference (home minus away) times opposing pressure rate enters the opener-margin ridge and then the four-term probability as one more fitted term.

## Tried
Novelty check only. Nearest: SIM-12 (OL exits inside a game), docs/ol_crosswalk_defect.md (crosswalk defect), MOD-22 (players on the field), PER-13; grep 'line continuity' found none. Differs: A pregame streak-length term, not an in-game exit hazard.

## Next
First unit: Build the streak per team-week from snap counts, check split-half reliability across seasons with a team permutation null, then LOSO add-one on the opener residual.

## Open
Data: nflverse snap counts and participation, local play-by-play pressure fields (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
