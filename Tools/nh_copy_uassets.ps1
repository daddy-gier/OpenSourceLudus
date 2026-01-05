param(
  [Parameter(Mandatory=$true)]
  [string]$VaultPath,
  [Parameter(Mandatory=$true)]
  [string]$ProjectDir,
  [string]$DestSubdir = "PrisonCore"
)

$ErrorActionPreference = "Stop"

$dest = Join-Path $ProjectDir "Content\$DestSubdir"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Write-Host "Copying .uasset/.umap into: $dest"

$items = Get-ChildItem -LiteralPath $VaultPath -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in ".uasset", ".umap" }

foreach ($it in $items) {
  $rel = $it.FullName.Substring($VaultPath.Length).TrimStart('\')
  $target = Join-Path $dest $rel
  $targetDir = Split-Path -Parent $target
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  Copy-Item -LiteralPath $it.FullName -Destination $target -Force
}

Write-Host ("✅ Copied {0} assets." -f $items.Count) -ForegroundColor Green
Write-Host "Next: launch Unreal once, then run fixup redirectors script."
