# SKY-05 research gate

## Goal
Auditable propose / fixed look budget / approval-reference record for research families, without blocking `weak-signals record`.

## State
Built `src/nfl_ats/research_proposals.py`, registered from `cli_commands/registry.py`.
- `nfl-ats research propose --family F --row ID --look-budget N --declaration PATH --approval-reference TEXT` writes `registry/proposals/F.json`; refuses unknown/closed (done or dropped) ROADMAP rows, missing declaration, repeat family.
- `nfl-ats research status --family F`: looks in weak-signals family v budget; exit 1 over budget, 2 no proposal.
- Proposal written for sim08_market_anchored (SIM-08, budget 34, docs/sim08_market_anchored_declaration.md); status: 0/34. lead89_option_policy_contrasts: no proposal, exit 2.
- ruff, format, mypy clean; pytest -k "weak_signal or registry": 156 passed. Pinned set unchanged (82 sources, no pinned file edited).

## Tried
Enforcement is advisory by design (record unchanged).

## Next
Orchestrator: update ROADMAP SKY-05 row, commit registry/proposals; propose other declared families; optionally have scheduler/handoff run status.

## Open
Whether `weak-signals record` should refuse un-proposed families (owner decision; would block existing recording).
