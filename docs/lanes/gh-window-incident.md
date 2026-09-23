# Repeated GitHub CLI windows

## Goal

Identify and prevent the repeated `gh.exe` windows reported on September 22, 2026.

## State

Launcher unresolved. **Reported:** the owner saw more than 1,000 windows and killed VS Code to stop them; another agent may have caused it. **Measured:** `@(Get-Process -Name gh -ErrorAction SilentlyContinue).Count` returned 0 during this investigation. No fix is claimed.

## Tried

- **Measured:** `Get-Command gh -All` resolved only `C:\Program Files\GitHub CLI\gh.exe`. Git configuration selected `manager-core` and `wincred` credential helpers.
- **Read:** the Codex `hooks.json` PreToolUse hook calls `lean_ctx.py`; the helper runs the bounded search tool and does not invoke GitHub CLI.
- **Read:** a recent dashboard session launched a hidden local preview server and Edge. This does not identify the GitHub CLI launcher. The narrow extracted command record is saved in `.tmp/gh-window-incident/prior-command.txt`.
- Windows denied the CIM process-details query. No `gh` command, project job, publication, or settings change was performed.

## Next

If the launcher becomes identifiable, inspect that agent's command or hook and repair the launch behavior. Capture parent process and command-line evidence before stopping a recurrence when practical; do not reproduce the window flood.

## Open

Which agent or application invoked GitHub CLI, its arguments and parent process, and why launches repeated remain unknown.
