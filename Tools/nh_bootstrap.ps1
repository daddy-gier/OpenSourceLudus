param(
    [Parameter(Mandatory = $true)]
    [string]$UProjectPath
)

$pluginSource = Join-Path $PSScriptRoot "..\Plugins\NyghtshadeHollowCore"
if (-not (Test-Path $pluginSource)) {
    Write-Error "NyghtshadeHollowCore plugin not found at $pluginSource"
    exit 1
}

if (-not (Test-Path $UProjectPath)) {
    Write-Error "UProject not found at $UProjectPath"
    exit 1
}

$projectDir = Split-Path -Parent $UProjectPath
$pluginsDir = Join-Path $projectDir "Plugins"
$targetDir = Join-Path $pluginsDir "NyghtshadeHollowCore"

if (-not (Test-Path $pluginsDir)) {
    New-Item -ItemType Directory -Path $pluginsDir | Out-Null
}

Copy-Item -Path $pluginSource -Destination $targetDir -Recurse -Force

$readmePath = Join-Path $projectDir "NyghtshadeHollowCore_README.txt"
$readmeContent = @(
    "NyghtshadeHollowCore installed.",
    "Plugin location: $targetDir",
    "Next steps:",
    "1) Generate project files: \"%UE57_ROOT%\\Engine\\Build\\BatchFiles\\GenerateProjectFiles.bat\" -project=\"$UProjectPath\" -game -engine",
    "2) Build: \"%UE57_ROOT%\\Engine\\Build\\BatchFiles\\Build.bat\" <ProjectName>Editor Win64 Development -project=\"$UProjectPath\" -waitmutex",
    "3) Launch: \"%UE57_ROOT%\\Engine\\Binaries\\Win64\\UnrealEditor.exe\" \"$UProjectPath\""
)
$readmeContent | Set-Content -Path $readmePath

Write-Host "NyghtshadeHollowCore copied to $targetDir" -ForegroundColor Green
Write-Host "README note written to $readmePath" -ForegroundColor Green
Write-Host "Set UE57_ROOT to your Unreal 5.7 install root before running build commands." -ForegroundColor Yellow
