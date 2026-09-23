# Agent harness hygiene

## Goal

Reduce repeated context and conflicting workflow instructions while retaining research, prediction, publishing, and repository safeguards.

## State

Shared `AGENTS.md` cleanup and conditional `docs/agent_workflow.md` are reviewed and integrated. `CLAUDE.md` imports the shared instructions. No tracked skills or memory files were found in the scoped harness discovery.

## Tried

Read official OpenAI AGENTS, skills and prompting guidance and Anthropic memory, best-practices and current prompting guidance. Keep always-loaded instructions concise; load task-specific workflow details when needed. Parallelize independent work within actual tool and resource limits.

Removed the duplicate constant `UserPromptSubmit` injection from ignored local `.claude/settings.json`; its Git and comment guards remain unchanged. Backup: `.tmp/agent-hygiene/claude-settings.before.json`. JSON validation and preservation assertions passed.

Made explicit backlog sessions durable: keep available subagents on concrete bounded useful work while root progresses, replenish only from the authorized backlog, and persist through review, applicable verification, lane and handoff refresh, commit, and push. Status notes do not end authorized work. Operational jobs remain conditional, and automatic or open-ended goals remain prohibited.

## Next

Use `AGENTS.md` as the normative policy and load `docs/agent_workflow.md` for the current task's conditional procedure.

## Open

None. The local Claude settings change is intentionally untracked.
