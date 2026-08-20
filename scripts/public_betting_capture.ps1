# Weekly live public-betting-percentage capture (Task Scheduler wrapper).
#
# Companion to scripts/odds_capture.ps1, same shape, different source: hits
# actionnetwork.com/nfl/public-betting directly (never through Wayback) via
# scripts/public_betting_live_capture.py. Contains NO secrets -- the live
# page needs no API key, unlike odds_capture.ps1's THE_ODDS_API_KEY lookup.
# Appends a one-line result per run to data\raw\public_betting_live\capture_log.txt.
#
# Intended Task Scheduler cadence (see docs/public_betting_sourcing.md
# section 7 item 1 and this session's report): TWO runs per week, Saturday
# 12:00 and Sunday 12:00, bracketing the slate. This machine's own local
# timezone is already America/New_York (Eastern, DST-aware; measured via
# `Get-TimeZone` this session), so a Task Scheduler local-time trigger of
# 12:00 already IS noon ET with no manual UTC conversion needed -- verify
# this holds if the task is ever registered on a different machine.

$ErrorActionPreference = 'Stop'
$repo = 'F:\Repos\nfl_py3'
Set-Location $repo

$stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$logDir = Join-Path $repo 'data\raw\public_betting_live'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$log = Join-Path $logDir 'capture_log.txt'

try {
    # Same two deliberate choices as scripts/odds_capture.ps1, learned there
    # first (2026-08-16/17 capture_log.txt incidents) and reused verbatim:
    # 1. `--no-sync` -- a capture must not depend on the venv not having
    #    changed since it was last built.
    # 2. stderr goes to its own file, never `2>&1` into the pipeline --
    #    under Windows PowerShell 5.1, $ErrorActionPreference = 'Stop'
    #    promotes a native command's stderr line to a terminating error even
    #    on a zero-exit-code success. $LASTEXITCODE is the only trustworthy
    #    success signal for a native exe.
    $errFile = Join-Path ([System.IO.Path]::GetTempPath()) "public_betting_capture_$PID.err"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & (Join-Path $repo '.tools\uv.exe') run --no-sync python (Join-Path $repo 'scripts\public_betting_live_capture.py') 2> $errFile | Out-String
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    $err = ''
    if (Test-Path $errFile) {
        $content = Get-Content -Path $errFile -Raw -ErrorAction SilentlyContinue
        if ($null -ne $content) { $err = [string]$content }
        Remove-Item -Path $errFile -Force -ErrorAction SilentlyContinue
    }
    $out = [string]$out
    if ($code -ne 0) { throw "public_betting_live_capture exit $code : $out $err" }

    $snap = if ($out -match '"snapshot_id":\s*"([^"]+)"') { $Matches[1] } else { 'unknown' }
    $rows = if ($out -match '"rows":\s*(\d+)') { $Matches[1] } else { '?' }
    $withData = if ($out -match '"rows_with_public_data":\s*(\d+)') { $Matches[1] } else { '?' }
    $era = if ($out -match '"era":\s*"([^"]+)"') { $Matches[1] } else { 'unknown' }
    Add-Content -Path $log -Encoding utf8 -Value "$stamp OK snapshot=$snap era=$era rows=$rows rows_with_data=$withData"
    exit 0
} catch {
    $msg = ($_ | Out-String) -replace '\s+', ' '
    if ($msg.Length -gt 400) { $msg = $msg.Substring(0, 400) }
    try { Add-Content -Path $log -Encoding utf8 -Value "$stamp FAIL $msg" } catch {}
    exit 1
}
