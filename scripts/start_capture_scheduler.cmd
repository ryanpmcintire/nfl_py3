@echo off
REM Launch the in-repo capture scheduler, hidden, and return immediately.
REM
REM This is the whole persistence story: a shortcut to this file in the user's
REM Startup folder starts the scheduler at login. No Task Scheduler object, no
REM admin rights, no service registration -- and the schedule itself lives in
REM scripts/capture_scheduler.py under version control rather than in per-machine
REM config you cannot diff.
REM
REM Safe to run twice: the scheduler records each job occurrence in
REM data/scheduler_state.json before running it, so a second copy cannot
REM double-fire a capture. It is still tidier to have one.
REM
REM Check on it:   .tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status
REM Stop it:       taskkill /FI "WINDOWTITLE eq nfl-ats capture scheduler" /F
REM                (or end the pythonw.exe process)

start "nfl-ats capture scheduler" /MIN /D "F:\Repos\nfl_py3" ^
  "F:\Repos\nfl_py3\.tools\uv.exe" run --no-sync python "F:\Repos\nfl_py3\scripts\capture_scheduler.py"
