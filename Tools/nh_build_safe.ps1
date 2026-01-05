param(
  [Parameter(Mandatory=$true)]
  [string]$UProjectPath,
  [Parameter(Mandatory=$true)]
  [string]$UE57Root,
  [int]$MaxParallelActions = 6
)

$ErrorActionPreference = "Stop"

Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$buildBat = Join-Path $UE57Root "Engine\Build\BatchFiles\Build.bat"
if (!(Test-Path $buildBat)) { throw "Build.bat not found at $buildBat" }

$projName = [IO.Path]::GetFileNameWithoutExtension($UProjectPath)
$target = "${projName}Editor"

Write-Host "Building $target with MaxParallelActions=$MaxParallelActions"
& $buildBat $target Win64 Development "-Project=$UProjectPath" -WaitMutex "-MaxParallelActions=$MaxParallelActions"
