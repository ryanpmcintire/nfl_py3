# Windows and Linux in one checkout

Windows uses `.venv`. Linux/WSL must use `.venv-linux` when it opens that same
checkout. Set `UV_PROJECT_ENVIRONMENT` before invoking uv, including when
resuming an agent session or starting a new terminal. A separate Linux-only
checkout, such as the backup server, can keep its existing `.venv` setup.

From the repository root in Linux/WSL:

```sh
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-linux"
uv sync --locked
uv run --no-sync python -c 'import sys; import nfl_ats; print(sys.executable)'
```

Use `./.tools/uv` in place of `uv` if that is the installed Linux binary.
Keep the exported variable set for all later uv commands and child processes.
Do not activate `.venv` or invoke `.venv/Scripts` from Linux. Both environment
directories are ignored by Git; neither belongs in a commit or a backup copy
that is restored onto a different operating system.

To restore the Windows environment, run these commands from the repository
root in PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv'
.\.tools\uv.exe sync --locked
.\.tools\uv.exe run --no-sync python -c "import sys; import nfl_ats; import tzdata; print(sys.executable)"
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status --brief
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --is-running
```

If `--is-running` reports no live daemon, restart it headlessly:

```powershell
Start-Process -WindowStyle Hidden -WorkingDirectory (Get-Location).Path -FilePath "$env:ComSpec" -ArgumentList '/c', 'scripts\start_capture_scheduler.cmd' -Wait
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status --brief
```

Read the scheduler status and follow the startup requirements in `AGENTS.md`.
The separate Python environments do not isolate capture state: the scheduler
uses `data/scheduler_state.json` and `data/scheduler_heartbeat.json` in the
checkout (read: `scripts/capture_scheduler.py`, `STATE_PATH` and
`HEARTBEAT_PATH`). Keep the shared checkout's existing scheduler host; use a
separate checkout for a separate capture host.
