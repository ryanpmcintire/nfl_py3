# POOL-05

## Goal
Ownership proxy v2: field share of the served side from Tuesday covariates (line, home, primetime, team brand effect) validated out of week (ROADMAP POOL-05).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: POOL-03 found the line-only proxy is the limit (within-week Spearman 0.50). The field loves certain teams and slots beyond the line; a shrunk team effect plus kickoff slot may raise the held-out fit, and the pool-objective term is only as good as the proxy.

## Tried
Novelty check only. Nearest: POOL-03 unit 1 (line-only proxy), POOL-01 (field model), POL-04 (closed: shares only appear at the deadline). Grep for team-effect ownership or brand covariates found none. Differs: covariates and team effects, validated before any leverage replay.

## Next
First unit: Fit leave-one-week-out logistic of served-side share on logit(line proxy), home, slot and a shrunk team effect; report held-out log loss against the POOL-03 line-only 0.6796 and the team-effect split-half correlation across odd and even weeks.

## Open
Changes: Better ownership proxy for the pool objective. Data: data/splash/field per-game counts (121 games W1-8, growing 15 per week). Out-of-week fit of ownership on Tuesday-known covariates. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
