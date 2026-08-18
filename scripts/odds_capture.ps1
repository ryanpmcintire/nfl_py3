# Weekly point-in-time NFL odds capture (Task Scheduler wrapper).
#
# Contains NO secrets: THE_ODDS_API_KEY is read from the user registry
# environment at runtime and is never echoed, logged, or written to disk.
# Appends a one-line result per run to data\market\capture_log.txt.

$ErrorActionPreference = 'Stop'
$repo = 'F:\Repos\nfl_py3'
Set-Location $repo

$stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$logDir = Join-Path $repo 'data\market'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$log = Join-Path $logDir 'capture_log.txt'

function Get-OddsApiKey {
    # Process env first (covers interactive runs), then HKCU\Environment,
    # then the explicit HKEY_USERS\<sid>\Environment hive (S4U tasks may not
    # map HKCU to the real user profile).
    if ($env:THE_ODDS_API_KEY) { return $env:THE_ODDS_API_KEY }
    $k = [Environment]::GetEnvironmentVariable('THE_ODDS_API_KEY', 'User')
    if ($k) { return $k }
    try {
        $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $p = Get-ItemProperty -Path "Registry::HKEY_USERS\$sid\Environment" -Name 'THE_ODDS_API_KEY' -ErrorAction Stop
        return $p.THE_ODDS_API_KEY
    } catch { return $null }
}

try {
    $key = Get-OddsApiKey
    if (-not $key) { throw 'THE_ODDS_API_KEY not found in user environment' }
    $env:THE_ODDS_API_KEY = $key

    # Two deliberate choices here, both learned from a real failed capture
    # (2026-08-16T20:15:00Z, recorded in capture_log.txt):
    #
    # 1. `--no-sync`. Plain `uv run` re-syncs when src/ has changed since the
    #    venv was last built, and prints "Building nfl-ats @ file:///..." to
    #    stderr. That is routine in an actively developed repo, so a capture
    #    must not depend on it not happening.
    # 2. stderr goes to its own file, never `2>&1` into the pipeline. Under
    #    Windows PowerShell 5.1 a native command's stderr line becomes a
    #    NativeCommandError record, which $ErrorActionPreference = 'Stop'
    #    promotes to a terminating error — aborting the run before
    #    odds-ingest is even reached, on a message that was not an error.
    #    $LASTEXITCODE is the only trustworthy success signal for a native exe.
    $errFile = Join-Path ([System.IO.Path]::GetTempPath()) "odds_capture_$PID.err"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & (Join-Path $repo '.tools\uv.exe') run --no-sync nfl-ats odds-ingest --markets spreads,h2h,totals 2> $errFile | Out-String
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    # `Get-Content -Raw` on a zero-byte file emits NO pipeline object at all
    # (not even $null-the-value), so `[string](Get-Content ...)` still leaves
    # $err as $null rather than ''. That is the common case: a healthy
    # `--no-sync` run against an unchanged venv writes nothing to stderr, so
    # the redirect file exists but is empty. Discovered 2026-08-17T23:00:00Z:
    # a fully successful capture (snapshot 20260817T230004Z, 4,580 quotes)
    # still crashed here on `$err.Replace(...)` and logged FAIL. Guard the
    # assignment itself so an empty/absent stderr file always leaves $err as
    # a real string.
    $err = ''
    if (Test-Path $errFile) {
        $content = Get-Content -Path $errFile -Raw -ErrorAction SilentlyContinue
        if ($null -ne $content) { $err = [string]$content }
        Remove-Item -Path $errFile -Force -ErrorAction SilentlyContinue
    }
    # Belt and braces: never allow the key value into the log even if a
    # downstream tool misbehaves.
    $out = [string]$out
    if ($key) {
        $out = $out.Replace($key, '***')
        $err = $err.Replace($key, '***')
    }
    if ($code -ne 0) { throw "odds-ingest exit $code : $out $err" }

    $snap = if ($out -match '"snapshot_id":\s*"([^"]+)"') { $Matches[1] } else { 'unknown' }
    $rows = if ($out -match '"quotes":\s*(\d+)') { $Matches[1] } else { '?' }
    $left = if ($out -match '"requests_remaining":\s*"?(\d+)"?') { $Matches[1] } else { '?' }
    Add-Content -Path $log -Encoding utf8 -Value "$stamp OK snapshot=$snap rows=$rows quota_remaining=$left"
    exit 0
} catch {
    $msg = ($_ | Out-String) -replace '\s+', ' '
    if ($msg.Length -gt 400) { $msg = $msg.Substring(0, 400) }
    try { Add-Content -Path $log -Encoding utf8 -Value "$stamp FAIL $msg" } catch {}
    exit 1
}
