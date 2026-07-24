# Bridges this repo's committed memory into the harness's per-machine memory path.
#
# The repo is the source of truth: .claude/memory/*.md + .claude/MEMORY.md.
# The Claude Code harness, however, reads memory from a fixed machine-local path:
#   ~/.claude/projects/<repo-slug>/{memory/, MEMORY.md}
# That path is not configurable, so we point it at the repo with a directory
# junction (for memory/) and a hard link (for MEMORY.md). Both are local
# filesystem objects that git does not track, which is why this must be run once
# per machine after cloning.
#
# Safe to re-run. Backs up any pre-existing real memory files before linking.
#
# Usage (from anywhere):  pwsh -File .claude/link-memory.ps1

$ErrorActionPreference = 'Stop'

# Repo root = parent of this script's .claude directory.
$repoRoot   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$repoMemDir = Join-Path $repoRoot '.claude\memory'
$repoIndex  = Join-Path $repoRoot '.claude\MEMORY.md'

# Derive the harness slug from the repo path: drive/colons/backslashes -> hyphens,
# lowercased. e.g. C:\decoder-only-transformer -> c--decoder-only-transformer
$slug = ($repoRoot -replace '[:\\/]', '-').ToLower()
$harnessRoot = Join-Path $env:USERPROFILE ".claude\projects\$slug"
$harnessMem  = Join-Path $harnessRoot 'memory'
$harnessIdx  = Join-Path $harnessRoot 'MEMORY.md'

Write-Host "repo memory : $repoMemDir"
Write-Host "harness path: $harnessRoot"

New-Item -ItemType Directory -Force -Path $harnessRoot | Out-Null

function Backup-IfRealFile($path) {
    if (Test-Path $path) {
        $item = Get-Item $path -Force
        $isLink = $item.Attributes -band [IO.FileAttributes]::ReparsePoint
        if (-not $isLink) {
            $bak = "$path.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
            Write-Host "backing up existing $path -> $bak"
            Move-Item $path $bak
        } else {
            Remove-Item $path -Force -Recurse
        }
    }
}

Backup-IfRealFile $harnessMem
Backup-IfRealFile $harnessIdx

New-Item -ItemType Junction -Path $harnessMem -Target $repoMemDir | Out-Null
New-Item -ItemType HardLink -Path $harnessIdx -Target $repoIndex  | Out-Null

Write-Host "linked. harness memory now resolves to the committed repo files."
