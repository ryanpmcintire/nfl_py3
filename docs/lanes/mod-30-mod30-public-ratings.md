# MOD-30 Public rating disagreement

## Goal
Start a forward log of independent public model win probabilities and grade their gap to the opener. ROADMAP row MOD-30.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Published models (ESPN FPI game win probabilities, Massey composite ratings) are built from information and weights the opening line does not use; their gap to the line is a candidate fitted term. Forward-only because the archive is not free. Data: ESPN public summary JSON predictor block (same family as the existing pickcenter capture) and Massey composite page, free and no key.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: acc-u8 prediction markets (market prices, not model ratings), free-odds-sources.md, market-derived-ratings (ratings from lines). Grep for FPI, DVOA, Massey, SP+ and composite rating finds zero hits.

## Next
Inspect capture_espn_pickcenter output for a predictor field and add the capture to the scheduler list if absent. First unit: First unit: probe whether the existing ESPN pickcenter capture already stores the predictor block; if not, write the capture job spec and snapshot the Week 10 slate Tuesday; define the disagreement term and its frozen reading before any 2026 result.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Likely already priced; a forward log is the only honest test since history is not free. FPI methodology changes are unversioned.
