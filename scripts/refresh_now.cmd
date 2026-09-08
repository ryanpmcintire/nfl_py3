@echo off
REM One-click refresh (owner, 2026-09-08: "a button to refresh the
REM lineups/injuryreports/spreads all at once"). Double-click this file, or
REM pin a shortcut to it. It runs scripts\refresh_now.py in a visible window
REM and keeps the window open at the end so the outcome can be read.
REM
REM What it does, in order (details in scripts\refresh_now.py):
REM   1. capture the current spreads (skipped on a Tuesday before 12:05 ET,
REM      so a button press can never become the week's opener line)
REM   2. refresh depth charts + the player snapshot (injury reports) and
REM      rebuild the forecast and the board  (about fifteen minutes)
REM   3. apply the late-week rule against the frozen Tuesday line and label
REM      any changed picks on the card
REM   4. regenerate the site pages
REM
REM Dry run (prints the commands, runs nothing):  refresh_now.cmd --dry
setlocal
cd /d "%~dp0.."
echo Refreshing spreads, lineups, injury reports, picks and the board...
echo.
".tools\uv.exe" run --no-sync python scripts\refresh_now.py %*
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (echo Done.) else (echo Finished with problems, exit code %CODE%.)
pause
exit /b %CODE%
