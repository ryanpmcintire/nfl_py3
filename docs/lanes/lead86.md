# LEAD-86: observed-market training regularization

## Goal
Execute the fixed LEAD-86 protocol without changing the served card.

## State
**Measured:** complete: 262 captures / 316,530 quotes; 1,309 eligible games,
697 outer-season games. Candidate 399-298; current four-term 398-299.
Decisive record 5-4, 55.56% [26.67%, 81.12%] Wilson interval.
Brier improvement +0.00051945 [-0.00106546, +0.00216971],
probability_positive 0.7320. Candidate IS/OOS Brier 0.243733/0.243387;
OOS-minus-IS gap -0.000346 [-0.008289, +0.007592].
Penalties 2023/2024/2025: 1/0/0.1; 505 declared looks.
**Inferred:** provisional, unresolved_below_power; no closing ground or promotion.
Protocol copied here before outcomes; exact pre-run snapshot retained in
[lead86_protocol.md](../lead86_protocol.md). Full fold coefficients, stability,
IS/OOS/gaps, baseline comparisons and reliability: [report](../lead86_unit1.md).

## Tried
**Measured:** one successful full replay:
.tools/uv.exe run --no-sync --no-cache python scripts/lead86_unit1.py.
An earlier inventory attempt stopped on a college-football manifest before
outcome access; fixed the NFL source filter. Report/record batch finalized
from cached predictions, without refitting. Scoped Ruff format/check pass.
Scratch: tests/scratch/codex/lead86_unit1/; log: lead86_run.log beside it.

## Record commands
Prepared, not executed; four paired current-model comparisons (Brier primary).
~~~bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead86_unit1/registry_batch.json \
  --plain-summary 'Training on sportsbook prices gave a small, uncertain gain. One conflicting price pair needs checking; keep the current picks.'
~~~

## Next
Orchestrator reviews the source discrepancy and records four cells serially;
fresh thread for source reconciliation. No additional replay for this unit.

## Open
**Measured:** one inconsistent 2021 LV-DEN bookmaker pair drives a maximum
16.99-point quoted-price error in 3/3,231 market fits; no exclusion/refit.
Retrospective upstream selection; nine decisive games; varying penalty and
model-logit sign. Pushes retained in lattice training; scored rows exclude pushes.
