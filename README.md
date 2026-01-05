# Nyghtshade Hollow Core Plugin

This repository provides a drop-in Unreal Engine 5.7 runtime plugin with starter gameplay systems for **Nyghtshade Hollow** plus a Windows "Foreman Pack" of helper scripts.

## Install

1. Copy `Plugins/NyghtshadeHollowCore` into your Unreal project root under `Plugins/`.
2. Ensure the plugin is enabled in the Unreal Editor (Edit → Plugins → Nyghtshade Hollow Core).

You can also use the bootstrap helper:

```powershell
./Tools/nh_bootstrap.ps1 -UProjectPath "C:\Path\To\YourProject.uproject"
```

## UE57_ROOT

Set the `UE57_ROOT` environment variable to your Unreal Engine 5.7 install directory, for example:

```powershell
setx UE57_ROOT "C:\Program Files\Epic Games\UE_5.7"
```

## Generate Project Files

```powershell
"%UE57_ROOT%\Engine\Build\BatchFiles\GenerateProjectFiles.bat" -Project="C:\Path\To\YourProject.uproject" -Game -Engine
```

## Build

```powershell
"%UE57_ROOT%\Engine\Build\BatchFiles\Build.bat" YourProjectEditor Win64 Development -Project="C:\Path\To\YourProject.uproject" -WaitMutex
```

## Foreman Pack (Windows)

The scripts below are intended to be run from PowerShell 7+:

### Audit for authoritative project

```powershell
pwsh Tools/nh_foreman_audit.ps1 -WorkspaceRoot "C:\Users\FRANK\Desktop\ltvall\shiny-happiness"
```

### Clean build artifacts

```powershell
pwsh Tools/nh_clean_project.ps1 -ProjectDir "C:\Users\FRANK\Desktop\ltvall\shiny-happiness\THENYGHTSHADEHOLLOW" -NukeDerivedDataCache
```

### Copy .uasset/.umap from the vault

```powershell
pwsh Tools/nh_copy_uassets.ps1 -VaultPath "C:\Nyghtshade_Assets_Vault" -ProjectDir "C:\Users\FRANK\Desktop\ltvall\shiny-happiness\THENYGHTSHADEHOLLOW" -DestSubdir "PrisonCore"
```

### Import FBX/PNG via Unreal Python

Copy `Tools/vault_to_unreal.py` into your project `Content/Python/` folder, then in the Unreal editor run:

```
py "<ProjectRoot>\Content\Python\vault_to_unreal.py"
```

### Safe build with capped parallel actions

```powershell
pwsh Tools/nh_build_safe.ps1 -UProjectPath "C:\Users\FRANK\Desktop\ltvall\shiny-happiness\THENYGHTSHADEHOLLOW\THENYGHTSHADEHOLLOW.uproject" -UE57Root "C:\Program Files\Epic Games\UE_5.7" -MaxParallelActions 6
```

## Quick In-Editor Test

1. Set your GameState class to `ANHTimeGameState` (Project Settings → Maps & Modes).
2. Create a DataTable from `FNHActivityRow` and assign it to an actor with `UNHScheduleComponent`.
3. Place `TargetPoint` actors with matching `Actor Tag` values from the schedule rows.
4. Add `UNHWalletComponent` to the same actor to track DC.
5. Create a UMG widget based on `UNHDebugWidget`, add it to the viewport, and call `SetObservedActor`.

## VS Code Tasks

Copy the files in `Tools/nh_ue_tasks_vscode/.vscode` into your project `.vscode` folder to get build and launch tasks.

## Systems Included

- Time system (`ANHTimeGameState`)
- Schedule system (`UNHScheduleComponent` + `FNHActivityRow` DataTable)
- Wallet system (`UNHWalletComponent`)
- Faction reputation (`UNHFactionSubsystem`)
- Contract/Hit Market gameplay-only subsystem (`UNHContractSubsystem`)
- Inmate AI controller hook (`ANHInmateAIController`)
- Debug widget (`UNHDebugWidget`)
