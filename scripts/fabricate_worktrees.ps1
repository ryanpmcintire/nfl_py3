[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$Name,

    [ValidateRange(1, 64)]
    [int]$Count = 1,

    [switch]$Remove
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..')).Path
$fleetParent = Split-Path -Path $repoRoot -Parent
$namePrefix = 'nfl_py3_wt_' + $Name

$junctionRootDirs = @('data', '.venv', '.tools')
$junctionArtifactSubdirs = @('opener_evaluation', 'market_decomposition', 'prospective')

function Test-JunctionPath {
    param([string]$LiteralPath)
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Container)) { return $false }
    return ((Get-Item -LiteralPath $LiteralPath -Force).LinkType -eq 'Junction')
}

function Get-NormalizedTarget {
    param([string]$LiteralPath)
    $target = [string](Get-Item -LiteralPath $LiteralPath -Force).Target
    return $target.TrimEnd('\').ToLowerInvariant()
}

function Remove-JunctionSafely {
    param([string]$LiteralPath)
    $item = Get-Item -LiteralPath $LiteralPath -Force
    if ($item.LinkType -ne 'Junction') {
        throw "Refusing to remove '$LiteralPath': it is not a junction."
    }
    $target = [string]$item.Target
    [System.IO.Directory]::Delete($LiteralPath)
    if (-not (Test-Path -LiteralPath $target)) {
        throw "Junction '$LiteralPath' was removed but its target '$target' is now missing; inspect for damage."
    }
    Write-Verbose "Removed junction '$LiteralPath' -> '$target'; target verified intact."
}

function Assert-ReplaceableByJunction {
    param([string]$WorktreeRoot, [string]$GitRelativePath)
    $destination = Join-Path $WorktreeRoot ($GitRelativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $destination)) { return }
    if (Test-JunctionPath $destination) {
        throw "'$destination' is already a junction; unexpected during fabrication."
    }
    $status = @()
    $raw = git -C $WorktreeRoot status --porcelain=v1 --ignored=matching -- $GitRelativePath 2>$null
    if ($null -ne $raw) { $status = @($raw | Where-Object { "$_".Trim().Length -gt 0 }) }
    if ($status.Count -gt 0) {
        $detail = ($status -join '; ')
        throw "'$destination' holds untracked, ignored or modified state and will not be replaced by a junction: $detail"
    }
    Remove-Item -LiteralPath $destination -Recurse -Force
}

function Install-Junction {
    param([string]$Destination, [string]$Source)
    if (Test-JunctionPath $Destination) {
        if ((Get-NormalizedTarget $Destination) -eq ((Resolve-Path -LiteralPath $Source).Path.TrimEnd('\').ToLowerInvariant())) {
            Write-Verbose "'$Destination' already junctioned to '$Source'; skipping."
            return
        }
        throw "'$Destination' is a junction to '$(Get-NormalizedTarget $Destination)', expected '$Source'."
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "'$Destination' already exists and is not a junction; refusing to overwrite."
    }
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Junction source '$Source' does not exist in the main repo."
    }
    New-Item -ItemType Junction -Path $Destination -Value $Source | Out-Null
    Write-Verbose "Created junction '$Destination' -> '$Source'"
}

function Get-FleetMembers {
    $listing = @(git -C $repoRoot worktree list --porcelain)
    $paths = @()
    foreach ($line in $listing) {
        if ($line -like 'worktree *') {
            $candidate = $line.Substring(9)
            if ((Split-Path -Path $candidate -Leaf) -like ($namePrefix + '_*')) { $paths += $candidate }
        }
    }
    return , $paths
}

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot '.git'))) {
    throw "Repo root '$repoRoot' does not look like a git repository."
}

if ($Remove) {
    $members = Get-FleetMembers
    if ($members.Count -eq 0) {
        Write-Warning "No fleet worktrees matching '${namePrefix}_*' found."
        exit 0
    }
    Write-Host "Removing $($members.Count) fleet worktree(s) matching '${namePrefix}_*':"
    foreach ($member in $members) {
        Write-Host "  $member"

        foreach ($dir in $junctionRootDirs) {
            $dest = Join-Path $member $dir
            if (Test-JunctionPath $dest) {
                if ($PSCmdlet.ShouldProcess($dest, "remove junction (target '$(Get-NormalizedTarget $dest)' untouched)")) {
                    Remove-JunctionSafely -LiteralPath $dest
                }
            }
        }

        foreach ($sub in $junctionArtifactSubdirs) {
            $dest = Join-Path $member ('artifacts\' + $sub)
            if (Test-JunctionPath $dest) {
                if ($PSCmdlet.ShouldProcess($dest, "remove junction (target untouched)")) {
                    Remove-JunctionSafely -LiteralPath $dest
                }
            }
        }

        $artifactsDest = Join-Path $member 'artifacts'
        if ((Test-Path -LiteralPath $artifactsDest) -and -not (Test-JunctionPath $artifactsDest)) {
            if ($PSCmdlet.ShouldProcess($artifactsDest, 'remove per-worktree artifacts directory')) {
                Remove-Item -LiteralPath $artifactsDest -Recurse -Force
            }
        }

        $strayLinks = @(Get-ChildItem -LiteralPath $member -Recurse -Force -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint })
        if ($strayLinks.Count -gt 0) {
            $names = (($strayLinks | ForEach-Object { $_.FullName }) -join '; ')
            throw "'$member' still contains reparse points after cleanup; refusing 'git worktree remove': $names"
        }

        if ($PSCmdlet.ShouldProcess($member, 'git worktree remove --force')) {
            git -C $repoRoot worktree remove --force $member 2>$null
            if ($LASTEXITCODE -ne 0) {
                git -C $repoRoot worktree remove --force --force $member 2>$null
                if ($LASTEXITCODE -ne 0) { throw "git worktree remove failed for '$member'." }
            }
        }
    }
    if ($PSCmdlet.ShouldProcess($repoRoot, 'git worktree prune')) {
        git -C $repoRoot worktree prune
        if ($LASTEXITCODE -ne 0) { throw 'git worktree prune failed.' }
    }
    Write-Host 'Fleet removal complete.'
    exit 0
}

foreach ($dir in $junctionRootDirs) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $dir) -PathType Container)) {
        throw "Heavy directory '$dir' missing in main repo '$repoRoot'; nothing to junction."
    }
}

for ($i = 1; $i -le $Count; $i++) {
    $memberPath = Join-Path $fleetParent ("${namePrefix}_$i")
    Write-Host "== Fabricating $memberPath"

    if (Test-Path -LiteralPath $memberPath) {
        throw "Refusing to reuse existing path '$memberPath'; remove it first or pick another -Name."
    }

    if ($PSCmdlet.ShouldProcess($memberPath, 'git worktree add --detach (current HEAD)')) {
        git -C $repoRoot worktree add --detach $memberPath HEAD | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for '$memberPath'." }
    }

    foreach ($dir in $junctionRootDirs) {
        $dest = Join-Path $memberPath $dir
        $src = Join-Path $repoRoot $dir
        if ($PSCmdlet.ShouldProcess($dest, "junction -> $src")) {
            Assert-ReplaceableByJunction -WorktreeRoot $memberPath -GitRelativePath $dir
            Install-Junction -Destination $dest -Source $src
        }
    }

    $artifactsDest = Join-Path $memberPath 'artifacts'
    if ($PSCmdlet.ShouldProcess($artifactsDest, 'create writable per-worktree artifacts directory')) {
        if (-not (Test-Path -LiteralPath $artifactsDest)) {
            New-Item -ItemType Directory -Path $artifactsDest -Force | Out-Null
        }
    }

    foreach ($sub in $junctionArtifactSubdirs) {
        $src = Join-Path $repoRoot ('artifacts\' + $sub)
        if (-not (Test-Path -LiteralPath $src -PathType Container)) {
            Write-Verbose "Skipping artifacts\${sub}: absent in main repo."
            continue
        }
        $dest = Join-Path $artifactsDest $sub
        if ($PSCmdlet.ShouldProcess($dest, "junction -> $src")) {
            Assert-ReplaceableByJunction -WorktreeRoot $memberPath -GitRelativePath ('artifacts/' + $sub)
            Install-Junction -Destination $dest -Source $src
        }
    }

    Write-Host "   ready: run tools from $memberPath via .\.tools\uv.exe (read-shared), artifacts\ is private"
}

Write-Host "Fabricated $Count worktree(s) named ${namePrefix}_1..${namePrefix}_$Count."
