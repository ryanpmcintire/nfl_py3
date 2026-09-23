# Agent workflow

`AGENTS.md` contains normative repository policy. This document contains the
commands and conditional procedures used to carry it out. Run only the parts
that apply to the current task.

## Operational session startup

This section applies to the primary orchestrator in an operational research,
forecast, scheduler, dashboard, or release session. Skip it for read-only
questions, harness maintenance, instruction hygiene, and bounded subagent work.

In a checkout shared between Windows and Linux or WSL, Windows uses `.venv`.
Before every Linux `uv run` or `uv sync`, run from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-linux"
uv sync --locked
```

Use the Linux uv binary for Linux commands and never let it replace the Windows
`.venv`. Recovery and setup details are in `docs/windows_linux_environment.md`.

Read `HANDOFF.md`, `docs/lanes/README.md`, and the named lane, or the most
recently modified lane when asked to continue. Read only `## Recommended
execution order` in `ROADMAP.md` when choosing new work. Search
`docs/roadmap_archive.md`, `docs/research_history.md`, and
`docs/agents_history.md` for specific history; do not read them whole.

Inspect live repository state:

```powershell
git status --short
git log -3 --oneline --decorate
git config --get core.hooksPath
```

Live Git state overrides handoff snapshots. Inspect
`artifacts/active_ats_model.json` before quoting the active model or its
historical result. `CURRENT_PREDICTIONS.md` is the last deliberately published
forecast, not necessarily the newest local forecast.

If the hooks path is not `.githooks`, set it without asking:

```powershell
git config --local core.hooksPath .githooks
```

Check local agent settings for prompt-time injections that duplicate
`AGENTS.md`, stale repository counts, or unconditional job instructions. Keep
tool guards and status integrations that enforce repository contracts.

## Explicit backlog sessions

When the owner explicitly requests backlog work, the primary orchestrator keeps
available subagents assigned to concrete bounded useful tasks while advancing
its own root-owned work. Give each subagent the required context packet, review
its return, and assign another independent authorized item within the current
bounded backlog batch when one is available. Stop replenishing assignments when
the batch reaches a clear stopping point or remaining work is blocked, dependent,
destructive, outside scope, or requires new authority. Complete the batch and
save a short handoff before starting a fresh thread. Do not create an automatic
queue or an open-ended goal.

A commentary or status update does not complete backlog work. Carry each ready
unit through review, applicable verification, lane and handoff refresh, commit,
and push at a verified clear stopping point. Operational jobs and publication
remain conditional on the work that changed; backlog mode does not trigger them
for harness maintenance or unrelated tasks.

## Capture scheduler

In an operational session, run the scheduler once and inspect brief status:

```powershell
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status --brief
```

A `MISSED` row means a capture is overdue; it does not prove the daemon stopped.
Inspect running status before starting another process. If no scheduler is
running, start it and report the restart:

```powershell
scripts\start_capture_scheduler.cmd
```

Exercise every scheduler job added or edited in the session through the actual
daemon argv:

```powershell
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --run-job NAME
```

Use `--dry` when the job writes a ledger or prediction card. Include the
`MANUAL-RUN OK` line in the session report. Treat a `NEVER RUN` job whose first
window falls within seven days like an overdue job and exercise it now. Calling
the underlying Python function does not verify the daemon argv.

## Reading and command output

Use `cr.ps1 PATH START COUNT` to read at most 100 lines and `cs.ps1 PATTERN
SUBDIR_OR_FILE` to search. Discover files with `cs.ps1 --files GLOB SUBDIR`.
Start with the named file or owning module and widen only after a scoped miss.
Read unchanged passages once. Save long output to a temporary log and report
the exit status with a short relevant excerpt. Prefer brief and summary command
forms, including `weak-signals pool` without `--full`, scheduler `--status
--brief`, `pytest -q`, and the tails shown below for ruff and mypy.

## Verification after code changes

Run all four commands after code changes:

```powershell
.\.tools\uv.exe run --no-sync ruff format --check . 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync ruff check . 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync mypy src 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync pytest -q 2>&1 | Select-Object -Last 15
```

The test moratorium in `AGENTS.md` still applies when a command exposes a
failure: fix production code or adjust an existing test; add no test file or
test function.

## Dashboard and forecast publication

Run publication only when the task changed dashboard output or the forecast.
For dashboard changes, regenerate the board, review the rendered-page diff, and
fix any fail-closed publication error in the same session:

```powershell
.\.tools\uv.exe run --no-sync nfl-ats publish-board
```

When the active weekly forecast changed, also publish predictions:

```powershell
.\.tools\uv.exe run --no-sync nfl-ats publish-predictions
```

The primary orchestrator reports the rendered diff and pushes the completed
change so GitHub Pages rebuilds. Subagents do not publish or push unless the
delegated task explicitly assigns that action.

## Handoff, commit, and push

At each verified clear stopping point, the primary orchestrator refreshes the
task lane and `HANDOFF.md`, then commits and pushes completed work under the
owner's standing authorization. Do not ask again. Preserve unrelated changes
and never rewrite history.

Before every commit or push intended for `master`, refresh the handoff with the
repository CLI and run:

```powershell
.\.tools\uv.exe run --no-sync nfl-ats handoff --check
```

If the refresh changes `HANDOFF.md` after a commit, make a follow-up commit
before pushing. Report remaining Git changes and the exact verification commands
run. Publication, handoff, commit, and push remain the primary orchestrator's
responsibility unless explicitly delegated.

## Instruction design references

Repository instructions follow the current vendor guidance to keep always-on
policy concise, put conditional detail in referenced files, state goals and
boundaries clearly, and avoid legacy over-insistence that can cause unnecessary
tool use in newer models:

- [OpenAI: Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
- [OpenAI: Agent Skills](https://developers.openai.com/codex/skills)
- [OpenAI: Prompting](https://developers.openai.com/api/docs/guides/prompting)
- [Anthropic: Claude Code best practices](https://docs.anthropic.com/en/docs/claude-code/best-practices)
- [Anthropic: Manage Claude's memory](https://docs.anthropic.com/en/docs/claude-code/memory)
- [Anthropic: Extend Claude with skills](https://docs.anthropic.com/en/docs/claude-code/skills)
- [Anthropic: Claude 4 prompting best practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
