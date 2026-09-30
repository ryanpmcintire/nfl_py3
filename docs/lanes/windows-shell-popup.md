# Windows shell popup

## Goal

Stop repeated popup windows during agent work and prevent recurrence across sessions.

## State

**Measured:** ordinary `exec_command` calls trigger a visible console owned by
`codex-windows-sandbox-setup.exe`, launched by Codex app-server PID 29780.
The Win32 window trace captured helper PIDs 13420 and 7152 at
2026-09-27 00:17:23Z and 00:17:59Z, with class `ConsoleWindowClass`.
**Reported (owner):** launching Codex with `codex --no-daemon` stopped the
popups. Use this owner-confirmed workaround for repository shell work; stop
the affected launch path immediately if popups return. The daemon launcher
is identified but not repaired.

## Tried

- **Reported:** popups recur with helper scripts and native PowerShell. Prior wrapper edits set `UseShellExecute = false` and `CreateNoWindow = true`; their runtime effect remains unverified.
- **Measured:** `.tmp/popup-trace/windows.jsonl` records visible window ownership; `.tmp/popup-trace/Capture-Popup.ps1` is the recorder. The helper is in `C:/Users/Ryan/.codex/packages/app-server-daemon/releases/0.157.1-x86_64-pc-windows-msvc/codex-resources/`. Its PE subsystem is console and its Authenticode signature is valid.
- **Measured:** the reset left zero hook events in `C:/Users/Ryan/.codex/hooks.json` and changed only `hooks = true` to `hooks = false` in the global config. Configuration readback matched the intended edit.
- Backups: `C:/Users/Ryan/.codex/hooks.before-reset-20260927T000954750Z.json` and `C:/Users/Ryan/.codex/config.before-hook-reset-20260927T001114066Z.toml`.
- **Read:** global config line 9 defines a separate turn-ended notification executable. Automatic approval review rejected clearing that setting without separate approval; the hook-only reset succeeded. **Reported:** the owner says the notification command is not the popup source. It remains unchanged.
- No repository jobs, tests, publication, commit, or push were run for this reset. The prepared `.tmp/hook-popup-fix/AGENTS.after.md` policy replacement remains unapplied.

## Next

Keep using `codex --no-daemon`. If daemon repair is pursued, verify window
events during an ordinary sandboxed command before declaring that launcher
fixed. Do not repeat hook or notification changes as a remedy.

## Open

The installed signed daemon app-server/helper launch path remains unresolved.
Window ownership is measured; the exact creation flags in the installed
launcher and a supported fix remain unverified. Hook reset did not establish
a launcher fix. Do not disable sandboxing or alter signed binaries to hide it.
