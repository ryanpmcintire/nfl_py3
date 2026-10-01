# Why accuracy is stuck, and how to get past it

## Goal
Name what keeps the model's own record against the opener stuck near 53.4%
(802-701, 2020-2025), and pick the work most likely to raise it.

## State
2026-10-01: theory written (in the conversation, summarized here). Four limits:
1. Information: the inputs are public team data the opener already prices.
   Even perfect knowledge of team strength moved error by only 0.013 points.
2. Resolution: 1,503 games give a standard error of about 1.3 points. A real
   gain of about 0.5-1 point can't be seen by win-loss grading. MOD-23 unit 1
   improved log loss and margin error in 5 of 6 seasons and still lost.
3. Noise: the margin varies by about 13 points and the edge is about 1 point.
   88 ridge inputs fit noise, and every blend or shrink chose "add nothing".
4. Reuse: about 7,700 tests on the same 2020-2025 games.

## Next (ranked)
1. Extend the model-alone opener test to 2011-2025 using sbr_odds openers.
   That is about 2.5 times the games and a standard error near 0.8. Check
   first that the SBR open matches the Tuesday pool line.
2. Promote on proper scores (log loss and margin score) with picks as a
   secondary. Re-grade MOD-23 unit 1 under that rule.
3. Add data the opener doesn't use: tracking and charting (Next Gen Stats,
   FTN charting). Unit 5 coverage charting was the only positive model-alone
   arm (+0.47, P+ 0.77). No ROADMAP row for NGS or FTN was found.
4. Grade all the above on the v2 forward cohort, untouched.

## Tried
See docs/lanes/mod23-base-model.md units 1-5, plus the team-quality ceiling
and MKT-20 (move as target, closed).

## Open
Does the SBR open match the Tuesday Splash line closely enough to grade on?
