param(
  [Parameter(Mandatory=$true)]
  [string]$UProjectPath
)

$ErrorActionPreference = "Stop"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (!(Test-Path -LiteralPath $UProjectPath)) { Fail "uproject not found: $UProjectPath" }
$projDir = Split-Path -Parent $UProjectPath
$repoRoot = Split-Path -Parent $PSScriptRoot

$srcPlugin = Join-Path $repoRoot "Plugins\NyghtshadeHollowCore"
if (!(Test-Path -LiteralPath $srcPlugin)) {
  Fail "Plugin folder not found: $srcPlugin`nRun the Codex prompt first so the repo contains Plugins\NyghtshadeHollowCore."
}

$dstPluginsDir = Join-Path $projDir "Plugins"
$dstPlugin = Join-Path $dstPluginsDir "NyghtshadeHollowCore"

New-Item -ItemType Directory -Force -Path $dstPluginsDir | Out-Null

if (Test-Path -LiteralPath $dstPlugin) {
  Write-Host "Removing existing plugin: $dstPlugin"
  Remove-Item -LiteralPath $dstPlugin -Recurse -Force
}

Write-Host "Copying plugin into project..."
Copy-Item -LiteralPath $srcPlugin -Destination $dstPlugin -Recurse -Force

Write-Host ""
Write-Host "✅ NyghtshadeHollowCore installed into:" -ForegroundColor Green
Write-Host "   $dstPlugin"

Write-Host ""
Write-Host "NEXT STEPS (pick one):" -ForegroundColor Cyan
Write-Host "1) In Windows Explorer: Right-click the .uproject -> 'Generate project files' (fast + reliable)."
Write-Host "2) Command line (engine install):"
Write-Host '   "%UE57_ROOT%\Engine\Build\BatchFiles\GenerateProjectFiles.bat" -Project="%UProjectPath%" -Game -Engine'
Write-Host ""
Write-Host "Then build the project from VS Code / Visual Studio / Unreal Editor."
Write-Host ""
Write-Host "UE docs for generating project files + VS Code setup (if you need it):"
Write-Host "- Generate project files: https://dev.epicgames.com/documentation/en-us/unreal-engine/how-to-generate-unreal-engine-project-files-for-your-ide"
Write-Host "- VS Code setup (UE 5.7): https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-your-development-environment-for-cplusplus-in-unreal-engine?application_version=5.7"
