# Small maintenance tasks

## Goal

Complete small documentation repairs and verify lane links within the remaining
session budget.

## State

The roadmap execution order now defers to the conditional workflow and limits
scheduler startup to operational sessions. All 42 lane-index links and all 10
local README links resolve. The every-metric lane remains active because its
research-look accounting question is still open.

## Tried

Compared the roadmap execution order with `AGENTS.md` and
`docs/agent_workflow.md`; the roadmap still required scheduler runs every session.
The working tree was clean before this task. `git diff --check` passed.
The default uv cache was inaccessible in the sandbox; a repository-local
`UV_CACHE_DIR` allowed the existing locked environment to run successfully.

## Next

None for this documentation repair.

## Open

None.
