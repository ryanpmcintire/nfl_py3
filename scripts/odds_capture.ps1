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

    $out = & (Join-Path $repo '.tools\uv.exe') run nfl-ats odds-ingest --markets spreads,h2h,totals 2>&1 | Out-String
    $code = $LASTEXITCODE
    # Belt and braces: never allow the key value into the log even if a
    # downstream tool misbehaves.
    $out = $out.Replace($key, '***')
    if ($code -ne 0) { throw "odds-ingest exit $code : $out" }

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
