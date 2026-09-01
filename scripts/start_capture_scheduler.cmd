@echo off
REM Launch the in-repo capture scheduler, headless, and return immediately.
REM
REM This is the whole persistence story: a shortcut to this file in the user's
REM Startup folder starts the scheduler at login. No Task Scheduler object, no
REM admin rights, no service registration -- and the schedule itself lives in
REM scripts/capture_scheduler.py under version control rather than in per-machine
REM config you cannot diff.
REM
REM Headless on purpose (owner request, 2026-09-01): Start-Process
REM -WindowStyle Hidden launches uv with its console created hidden (SW_HIDE),
REM so nothing appears in the taskbar for the life of the daemon. The obvious
REM alternative -- the venv's pythonw.exe -- does NOT deliver this: uv's
REM pythonw trampoline is a console-subsystem binary and allocates a visible
REM console anyway (observed 2026-09-01). The hidden console still exists, so
REM stdout works and child captures inherit it without flashing windows of
REM their own; data/scheduler_log.txt remains the record either way.
REM
REM Safe to run twice: the scheduler records each job occurrence in
REM data/scheduler_state.json before running it, so a second copy cannot
REM double-fire a capture. It is still tidier to have one.
REM
REM Check on it:   .tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status
REM Stop it:       scripts\stop_capture_scheduler.cmd
REM                (kills by command line; headless means no window title to match)

powershell -NoProfile -Command ^
  "Start-Process -WindowStyle Hidden -WorkingDirectory 'F:\Repos\nfl_py3' -FilePath 'F:\Repos\nfl_py3\.tools\uv.exe' -ArgumentList 'run','--no-sync','python','F:\Repos\nfl_py3\scripts\capture_scheduler.py'"
