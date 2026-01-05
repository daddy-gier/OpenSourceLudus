param(
  [Parameter(Mandatory=$true)]
  [string]$ProjectDir,
  [switch]$NukeDerivedDataCache
)

$ErrorActionPreference = "Stop"

function Zap($p) {
  if (Test-Path -LiteralPath $p) {
    Write-Host "Deleting: $p"
    Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Zap (Join-Path $ProjectDir "Binaries")
Zap (Join-Path $ProjectDir "Intermediate")
Zap (Join-Path $ProjectDir "Saved")

if ($NukeDerivedDataCache) {
  Zap (Join-Path $ProjectDir "DerivedDataCache")
  $globalDDC = Join-Path $env:LOCALAPPDATA "UnrealEngine\Common\DerivedDataCache"
  Zap $globalDDC
}

Write-Host "✅ Cleanup done." -ForegroundColor Green
