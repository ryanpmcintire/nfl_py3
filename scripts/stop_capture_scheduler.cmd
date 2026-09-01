@echo off
REM Stop the headless capture scheduler started by start_capture_scheduler.cmd.
REM
REM The daemon runs under pythonw with no console window, so the old
REM taskkill-by-window-title trick cannot find it. This matches on the command
REM line instead, and only among python/pythonw/uv processes so it can never
REM touch anything else. It also takes down the venv trampoline alongside the
REM real interpreter. Harmless when nothing is running, and it will also stop a
REM concurrent --once/--status invocation -- acceptable for a stop button.

powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe' OR Name='uv.exe'\" | Where-Object { $_.CommandLine -match 'capture_scheduler' } | ForEach-Object { Write-Host ('stopping ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
