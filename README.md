# Nyghtshade Hollow Core Plugin

This repository provides a drop-in Unreal Engine 5.7 runtime plugin with starter gameplay systems for **Nyghtshade Hollow**.

## Contents

```
Plugins/NyghtshadeHollowCore
Tools/
```

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
"%UE57_ROOT%\Engine\Build\BatchFiles\GenerateProjectFiles.bat" -project="C:\Path\To\YourProject.uproject" -game -engine
```

## Build

```powershell
"%UE57_ROOT%\Engine\Build\BatchFiles\Build.bat" YourProjectEditor Win64 Development -project="C:\Path\To\YourProject.uproject" -waitmutex
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
