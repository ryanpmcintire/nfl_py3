# MOD-34 Orthogonalised add-one

## Goal
Orthogonalised add-one: add only the part of a candidate feature that lies outside the span of the 88 served inputs ROADMAP row MOD-34.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Add-one features mostly flip coin-flip games because most of a candidate is already in the served inputs and only reweights them. Residualising each candidate on the served inputs (fit leave-one-season-out, season by season) leaves only genuinely new information, which can be graded without the reweighting noise. Data: Served-model input matrix and the candidate columns from MOD-23 units 6-18; no new data.

## Tried
Novelty check only: Nearest: mod23-unit10r-robust-wide.md and mod23-unit10-candidates.md (wide candidate sets added raw), acc-u7-new-information.md (new-information screen by source, not by span). Differs: the added column is the span-residual and the diagnostic is the flip pattern.

## Next
First unit: For the finished MOD-23 candidates compute the share of variance outside the served-input span, and compare raw versus residualised add-one on decisive games and on flips of games inside the model indistinguishable band; report the record on decisive games before any headline.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
