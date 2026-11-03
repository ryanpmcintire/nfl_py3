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

## Unit 1 result (2026-11-03 ~17:00 local, scripts/mod42_ol_continuity.py, log tests/scratch/mod42/run.txt)
Features: starters = 5 OL (roster position) on first scrimmage play per team-game, participation 2020-25 (1503/1503 non-push opener games covered). Expectation = prior game's five (current starters NOT used). Streak = trailing consecutive prior in-season games with that five; new = starters in prior game not in the game before. Pressure = opponent defense prior-game sack+qb_hit per dropback (>=2 games, else mean-filled).
LOSO 2020-25, 5 looks (family mod42_ol_continuity). Served LL .69216 W-L 800-703. cont: LL .69282, dLL -.00067 CI[-.00126,-.00007] P+ .014, flips 79 model 38-41 vs served 41-38. new: -.00015 P+ .44, flips 210 105-105. cont+press -.00093 P+ .14, 106-131. cont x pressure -.00106 P+ .18, 122-152. all -.00109 P+ .27. In-sample gaps .001-.003 (overfit). Coefs stable-sign for new (+.03..+.10, right sign), cont ~0. Split-half team-season corr .056, SB .107, perm P .21 (weak reliability, not zero-proved).
Recorded 3 cells via weak-signals record, unresolved_below_power. Nothing closed; no pick term earned.
Next: stronger starter definition (snap-share top-5 instead of first play) and QB/WR-less line-change magnitude by position (C/LT), or play-level pressure from was_pressure in participation.
- 2026-11-03 16:34 ET (root): the cont arm's whole log-loss interval is negative, but its coefficient is near 0 in every fold, so this is the estimation cost of a null term, not a resolved wrong-sign mechanism; unresolved_below_power stands.
