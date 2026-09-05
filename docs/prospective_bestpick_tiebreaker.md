# Prospective Best Pick refresh and tiebreaker shade

Predeclared 2026-09-05, before implementation or prospective outcomes.
Read (`ROADMAP.md:980-981`): LEAD-53 predicts Sunday re-nomination beats
Tuesday nomination; LEAD-54 specifies a one-point low-side market shade.
Both remain open, prospective-only, with no historical window spent.

The observation unit is one regular-season (season, week), starting 2026
Week 1 at the 2026-09-08 publish with `--record-decisions`. First writes
freeze decisions; repeat publishes/refreshes never replace an arm.

LEAD-53 freezes the served Tuesday nominee and its pick-side probability
from the publication, joined to the paper ledger's frozen nominee. On the first Sunday-morning refresh (Eastern time,
before noon), select from games still playable at min(kickoff, Sunday
16:00 ET). If Tuesday's nominee has already locked, retain it in both arms:
a finished Thursday Best Pick cannot be exchanged using its known outcome.
Otherwise rank refreshed probabilities by distance from 0.5, using the
existing v2 below-median opener-dispersion eligibility and tie ordering;
freeze that eligibility/dispersion at Tuesday recording so later captures
cannot rewrite the pool. Record nominee IDs, pick sides, probabilities,
original lines and whether IDs differ. Grade each frozen side using the
existing prospective ATS settlement at the recorded Tuesday line; retain
pushes separately. Compare Sunday minus Tuesday cover outcome within week.
Missing Tuesday rows or incomplete playable-game probabilities skip the pair.

LEAD-54 freezes the published last-game score from `tiebreaker.json`.
The alternative lattice centre is market total minus exactly 1.0, retaining
the production lattice's continuous margin and pick-side constraint. Select the closest
feasible pick-consistent cell using the existing lattice selector, including
its one-then-two-point proximity tolerance and deterministic mass tie-break.
Integer scores cannot always sum to a half-point target: store both the
continuous target and the selected integer total and their difference.
No cell means a logged skip, not an inconsistent guess. Historical lattice
support uses completed games from earlier NFL weeks (including earlier
seasons), scheduled before the publication day, avoiding same-week result
or future-final leakage while retaining earlier 2026 history. Record served/shaded scores, market total, game,
deadline and capture time. Backfill final total on subsequent recording
passes using local schedules, as the served-total challenger does. Grade
absolute error for each arm; smaller error wins and equal error is a tie.
Best Pick settlement also runs on each publication so the final Sunday pair
is graded even if no later refresh is necessary. Missing inputs skip with a
reason. Neither ledger changes a served card.

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability, or a positive control proven able to detect an effect that size. Everything else is unresolved_below_power.
