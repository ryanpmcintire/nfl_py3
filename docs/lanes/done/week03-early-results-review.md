# Week 3 early-results review

## Goal

Verify the owner's reported 3-7 start against the published picks and public final scores. Retrieve public results directly instead of asking the owner to supply them.

## State

- **Measured:** fetched [nflverse final scores](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) at 2026-09-27 21:17 UTC. Published picks are **3-7**, including Thursday; Sunday early games are **2-7**. Covers: ATL +6.5 (35-14), IND +2.5 (19-17), JAX -2.5 (35-6).
- **Measured:** losses are CAR -2.5 (18-21), CIN -3.5 (27-30), MIA +10.5 (10-24), LAC +7.5 (16-24), NYJ +6.5 (24-31), SEA -6.5 (31-33), TEN +3.5 (7-12). All scores here list the picked team first; grading uses published original lines.
- **Measured:** the saved raw forecast card is 6-4 on the same ten games, or 5-4 on Sunday's nine. Its three disagreements with publication all covered: PIT +3.5 versus published CIN -3.5; WAS +6.5 versus SEA -6.5; NYG -3.5 versus TEN +3.5. This is a snapshot comparison, not proof of a better future policy.
- **Measured:** the three early-game changes listed in `CURRENT_PREDICTIONS.md:47-51` added one win (HOU to IND) and lost two (KC to MIA, WAS to SEA). Reconstructing Tuesday from that log gives 4-6. The four published estimates at least 55% went 2-2; the six below 53% went 1-5.
- **Read:** `src/nfl_ats/publishing.py:186-225` loads the active recommendations, resolves the card view, and applies frozen picks. The raw card is not the final served card.

## Tried

Read the published card, active forecast, and cached schedule; all cached final scores were stale. ESPN returned HTTP 403; the public nflverse CSV succeeded. Saved source CSV, fetch summary, grading CSV, and grading summary under `.tmp/week03-*`. Used the locked environment. No model edits, operational jobs, publication, or research closure.

## Next

The score verification is complete. A separate mechanism review can trace the three raw-to-published side differences through the fitted probability and frozen-pick records, keeping pregame evidence intact.

## Open

The component trace is now complete in `week03-pick-change-trace.md`: coaching term for CIN, market movement for SEA, baseline calibration offset for TEN. These ten outcomes establish this week's arithmetic, not systematic failure or a reason to flip or disable a signal. One fitted probability must continue to select the side.
