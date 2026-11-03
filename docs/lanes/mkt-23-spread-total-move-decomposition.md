# MKT-23

## Goal
Spread and total move decomposition: implied team-total changes from Tuesday to the deadline as fitted side terms (ROADMAP MKT-23).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: A spread move with a flat total is a view on the sides; a move with a total drop is a view on one offence (a QB or weather). Implied team totals move by half the total change plus or minus half the spread change; which offence lost points identifies the news, and the four-term probability, which uses spread move only, cannot see it.

## Tried
Novelty check only. Nearest: MKT-16..19 (spread move only), LEAD-71 (moneyline lattice), LEAD-88 (total news for the tiebreaker). Grep for implied team total and spread-total decomposition found only totals_program.md noting team totals are absent from nflverse. Differs: derives them from the two published numbers.

## Next
First unit: Tabulate Tuesday-to-deadline change in spread and total, derive implied home and away total changes, and report how often the total moves with the spread and in which direction; then fit one extra term leave-one-season-out, with the intercept-only control.

## Open
Changes: Post-opener market information, split by side. Data: Archived hourly quotes (2023-25, 799 games) and earlier Tuesday-to-close pairs. Delta spread, delta total, implied delta team totals. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
