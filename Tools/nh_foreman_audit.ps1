param(
  [Parameter(Mandatory=$true)]
  [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"

function SizeBytes($p) {
  if (!(Test-Path $p)) { return 0 }
  (Get-ChildItem -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
}

function FindUProjects($root) {
  Get-ChildItem -LiteralPath $root -Recurse -Force -Filter *.uproject -ErrorAction SilentlyContinue |
    Select-Object FullName, DirectoryName, LastWriteTime
}

$uprojects = FindUProjects $WorkspaceRoot

Write-Host "=== AUDIT RESULTS ==="
Write-Host ("Workspace: {0}" -f $WorkspaceRoot)
Write-Host ("Found .uproject files: {0}" -f ($uprojects.Count))

$scored = @()

foreach ($u in $uprojects) {
  $projDir = $u.DirectoryName
  $content = Join-Path $projDir "Content"
  $source  = Join-Path $projDir "Source"

  $contentBytes = SizeBytes $content
  $hasSource    = Test-Path $source
  $isNyghtshade = ($u.FullName -match "NYGHTSHADE|Nyghtshade|THENYGHTSHADEHOLLOW")

  # Score: prefer Nyghtshade-named, has Source, more Content size, more recent.
  $score = 0
  if ($isNyghtshade) { $score += 1000 }
  if ($hasSource)    { $score += 200 }
  $score += [math]::Min(500, [int]($contentBytes / 1GB * 50))
  $score += [int]([math]::Min(200, ((Get-Date) - $u.LastWriteTime).TotalDays * -2)) # newer = higher

  $scored += [pscustomobject]@{
    UProject     = $u.FullName
    ProjectDir   = $projDir
    LastWrite    = $u.LastWriteTime
    HasSource    = $hasSource
    ContentGB    = [math]::Round($contentBytes / 1GB, 2)
    IsNyghtshade = $isNyghtshade
    Score        = $score
  }
}

$scored = $scored | Sort-Object Score -Descending

$scored | Format-Table -AutoSize

if ($scored.Count -eq 0) {
  Write-Host "No Unreal projects found. Either wrong folder or the universe is messing with you." -ForegroundColor Red
  exit 1
}

$authoritative = $scored[0]
Write-Host ""
Write-Host "AUTHORITATIVE PROJECT (heuristic pick):" -ForegroundColor Green
Write-Host $authoritative.UProject
Write-Host ""
Write-Host "Next: run cleanup + ingestion against that path."
