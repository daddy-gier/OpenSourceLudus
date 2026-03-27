#!/usr/bin/env python3
"""OpenSourceLudus: AI Copilot Scripts for Unity and Unreal Engine.

This single-file tool bundles prompt templates, code skeletons, and task
checklists to help drive AI-assisted scripting workflows for both engines.
It also implements an offline-first registry mirror with deterministic,
verifiable, and audit-logged synchronization.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Dict, Any

yaml = None
if importlib.util.find_spec("yaml"):
    import yaml  # type: ignore


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    engine: str
    summary: str
    checklist: List[str]
    copilot_prompt: str
    code_skeleton: str


UNITY_PROMPT = dedent(
    """
    You are an AI copilot assisting a Unity developer.
    - Use C# and Unity API conventions.
    - Favor readable, well-commented MonoBehaviour scripts.
    - Include serialized fields for inspector configuration.
    - Provide safe defaults and guard clauses.
    - Avoid editor-only APIs unless explicitly requested.
    """
).strip()

UNREAL_PROMPT = dedent(
    """
    You are an AI copilot assisting an Unreal Engine developer.
    - Use C++ with Unreal coding conventions.
    - Include UCLASS/USTRUCT/UFUNCTION macros where appropriate.
    - Favor components and sensible defaults.
    - Avoid editor-only modules unless requested.
    - Keep code compatible with Unreal 5.
    """
).strip()


MIRROR_ROOT_DEFAULT = Path("~/.osl/registry").expanduser()
INDEX_FILENAME = "index.yaml"
INTEGRITY_FILENAME = "integrity/mirror.hash"
AUDIT_LOG = "audit/mirror.log.jsonl"
SYNC_LOG = "sync/sync.log.jsonl"
LAST_SYNC = "sync/last_sync.json"


class MirrorError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unity_task_templates() -> List[TaskTemplate]:
    return [
        TaskTemplate(
            name="player-movement",
            engine="unity",
            summary="3D character movement with camera-relative controls and jump.",
            checklist=[
                "Create a CharacterController-based movement script.",
                "Expose speed, jump height, and gravity in the inspector.",
                "Align movement to camera forward/right vectors.",
                "Use ground checks to reset vertical velocity.",
            ],
            copilot_prompt=UNITY_PROMPT
            + "\n\nTask: Implement a third-person player movement script.",
            code_skeleton=dedent(
                """
                using UnityEngine;

                [RequireComponent(typeof(CharacterController))]
                public class PlayerMovement : MonoBehaviour
                {
                    [Header("Movement")]
                    [SerializeField] private float moveSpeed = 5f;
                    [SerializeField] private float jumpHeight = 1.5f;
                    [SerializeField] private float gravity = -9.81f;
                    [SerializeField] private Transform cameraTransform;

                    private CharacterController controller;
                    private Vector3 velocity;

                    private void Awake()
                    {
                        controller = GetComponent<CharacterController>();
                    }

                    private void Update()
                    {
                        if (cameraTransform == null)
                        {
                            Debug.LogWarning("PlayerMovement: Missing cameraTransform reference.");
                            return;
                        }

                        bool isGrounded = controller.isGrounded;
                        if (isGrounded && velocity.y < 0f)
                        {
                            velocity.y = -2f; // keep grounded
                        }

                        float horizontal = Input.GetAxis("Horizontal");
                        float vertical = Input.GetAxis("Vertical");

                        Vector3 cameraForward = cameraTransform.forward;
                        cameraForward.y = 0f;
                        cameraForward.Normalize();
                        Vector3 cameraRight = cameraTransform.right;
                        cameraRight.y = 0f;
                        cameraRight.Normalize();

                        Vector3 move = (cameraForward * vertical + cameraRight * horizontal).normalized;
                        controller.Move(move * moveSpeed * Time.deltaTime);

                        if (Input.GetButtonDown("Jump") && isGrounded)
                        {
                            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
                        }

                        velocity.y += gravity * Time.deltaTime;
                        controller.Move(velocity * Time.deltaTime);
                    }
                }
                """
            ).strip(),
        ),
        TaskTemplate(
            name="interaction-raycast",
            engine="unity",
            summary="Camera-based raycast interaction with UI prompt.",
            checklist=[
                "Cast a ray from the camera each frame.",
                "Detect objects with an IInteractable interface.",
                "Expose interact range and prompt UI text.",
                "Trigger interaction on input.",
            ],
            copilot_prompt=UNITY_PROMPT
            + "\n\nTask: Implement a raycast interaction system.",
            code_skeleton=dedent(
                """
                using UnityEngine;
                using UnityEngine.UI;

                public interface IInteractable
                {
                    string Prompt { get; }
                    void Interact();
                }

                public class InteractionRaycast : MonoBehaviour
                {
                    [SerializeField] private Camera playerCamera;
                    [SerializeField] private float interactRange = 3f;
                    [SerializeField] private Text promptText;

                    private IInteractable currentTarget;

                    private void Update()
                    {
                        if (playerCamera == null || promptText == null)
                        {
                            return;
                        }

                        currentTarget = null;
                        promptText.enabled = false;

                        Ray ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
                        if (Physics.Raycast(ray, out RaycastHit hit, interactRange))
                        {
                            if (hit.collider.TryGetComponent(out IInteractable interactable))
                            {
                                currentTarget = interactable;
                                promptText.text = interactable.Prompt;
                                promptText.enabled = true;
                            }
                        }

                        if (currentTarget != null && Input.GetButtonDown("Fire1"))
                        {
                            currentTarget.Interact();
                        }
                    }
                }
                """
            ).strip(),
        ),
    ]


def unreal_task_templates() -> List[TaskTemplate]:
    return [
        TaskTemplate(
            name="third-person-character",
            engine="unreal",
            summary="C++ third-person character with camera boom and movement bindings.",
            checklist=[
                "Create a Character subclass with a SpringArm + Camera component.",
                "Bind movement and look input axes.",
                "Expose movement settings in defaults.",
                "Provide jump input action binding.",
            ],
            copilot_prompt=UNREAL_PROMPT
            + "\n\nTask: Implement a third-person character class.",
            code_skeleton=dedent(
                """
                #pragma once

                #include "CoreMinimal.h"
                #include "GameFramework/Character.h"
                #include "ThirdPersonHero.generated.h"

                UCLASS()
                class AThirdPersonHero : public ACharacter
                {
                    GENERATED_BODY()

                public:
                    AThirdPersonHero();

                protected:
                    virtual void BeginPlay() override;
                    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

                    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
                    class USpringArmComponent* CameraBoom;

                    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
                    class UCameraComponent* FollowCamera;

                    void MoveForward(float Value);
                    void MoveRight(float Value);
                    void Turn(float Value);
                    void LookUp(float Value);
                };
                """
            ).strip(),
        ),
        TaskTemplate(
            name="interaction-component",
            engine="unreal",
            summary="An interaction component with line trace and interface trigger.",
            checklist=[
                "Create a UActorComponent to manage tracing.",
                "Define an interaction interface for targets.",
                "Expose trace distance and debug options.",
                "Trigger interaction on input binding.",
            ],
            copilot_prompt=UNREAL_PROMPT
            + "\n\nTask: Implement an interaction component with line trace.",
            code_skeleton=dedent(
                """
                #pragma once

                #include "CoreMinimal.h"
                #include "Components/ActorComponent.h"
                #include "InteractionComponent.generated.h"

                UINTERFACE(BlueprintType)
                class UInteractableInterface : public UInterface
                {
                    GENERATED_BODY()
                };

                class IInteractableInterface
                {
                    GENERATED_BODY()

                public:
                    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Interaction")
                    void Interact(AActor* InstigatorActor);
                };

                UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
                class UInteractionComponent : public UActorComponent
                {
                    GENERATED_BODY()

                public:
                    UInteractionComponent();

                    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction")
                    float TraceDistance = 250.f;

                    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction")
                    bool bDebugTrace = false;

                    void TryInteract();
                };
                """
            ).strip(),
        ),
        TaskTemplate(
            name="foreman-pack-ue57",
            engine="unreal",
            summary="Foreman Pack scripts + integration shims for Nyghtshade on UE 5.7.",
            checklist=[
                "Add Bridge folders under Source/Nyghtshade/Private and Public.",
                "Create optional Crown renderer and IntegratedDynamics shims with __has_include guards.",
                "Update Nyghtshade.Build.cs to set WITH_CROWN and WITH_INTEGRATED_DYNAMICS defines.",
                "Add PowerShell automation scripts for audit, clean, copy, and safe build.",
                "Add Unreal Python import script for FBX/PNG + material setup.",
                "Write README with exact commands to run on Windows + UE 5.7.",
            ],
            copilot_prompt=UNREAL_PROMPT
            + "\n\nTask: Generate a Foreman Pack for Nyghtshade (Windows + UE 5.7) with scripts, shims, and README.",
            code_skeleton=dedent(
                """
                File: Source/Nyghtshade/Public/Bridge/CrownRenderShim.h
                ```cpp
                #pragma once

                #include "CoreMinimal.h"

                #ifndef WITH_CROWN
                #define WITH_CROWN 0
                #endif

                namespace Nyghtshade
                {
                    class FCrownRenderShim
                    {
                    public:
                        static bool IsAvailable();
                        static void InitializeCrownRenderer();
                        static void RenderPrisonPass();
                    };
                }
                ```

                File: Source/Nyghtshade/Private/Bridge/CrownRenderShim.cpp
                ```cpp
                #include "Bridge/CrownRenderShim.h"

                #if WITH_CROWN && __has_include("../../../../crown-master/Source/Crown/Public/CrownRenderer.h")
                #include "../../../../crown-master/Source/Crown/Public/CrownRenderer.h"
                #define WITH_CROWN_RUNTIME 1
                #else
                #define WITH_CROWN_RUNTIME 0
                #endif

                namespace Nyghtshade
                {
                    bool FCrownRenderShim::IsAvailable()
                    {
                        return WITH_CROWN_RUNTIME != 0;
                    }

                    void FCrownRenderShim::InitializeCrownRenderer()
                    {
                    #if WITH_CROWN_RUNTIME
                        Crown::Initialize();
                    #endif
                    }

                    void FCrownRenderShim::RenderPrisonPass()
                    {
                    #if WITH_CROWN_RUNTIME
                        Crown::RenderPrisonPass();
                    #endif
                    }
                }
                ```

                File: Source/Nyghtshade/Public/Bridge/IntegratedDynamicsDataShim.h
                ```cpp
                #pragma once

                #include "CoreMinimal.h"

                #ifndef WITH_INTEGRATED_DYNAMICS
                #define WITH_INTEGRATED_DYNAMICS 0
                #endif

                namespace Nyghtshade
                {
                    class FIntegratedDynamicsDataShim
                    {
                    public:
                        static bool IsAvailable();
                        static bool LoadPrisonDataset(const FString& DatasetPath);
                        static bool SavePrisonDataset(const FString& DatasetPath);
                    };
                }
                ```

                File: Source/Nyghtshade/Private/Bridge/IntegratedDynamicsDataShim.cpp
                ```cpp
                #include "Bridge/IntegratedDynamicsDataShim.h"

                #if WITH_INTEGRATED_DYNAMICS && __has_include("../../../../IntegratedDynamics/Source/IntegratedDynamics/Public/IntegratedDynamicsAPI.h")
                #include "../../../../IntegratedDynamics/Source/IntegratedDynamics/Public/IntegratedDynamicsAPI.h"
                #define WITH_INTEGRATED_DYNAMICS_RUNTIME 1
                #else
                #define WITH_INTEGRATED_DYNAMICS_RUNTIME 0
                #endif

                namespace Nyghtshade
                {
                    bool FIntegratedDynamicsDataShim::IsAvailable()
                    {
                        return WITH_INTEGRATED_DYNAMICS_RUNTIME != 0;
                    }

                    bool FIntegratedDynamicsDataShim::LoadPrisonDataset(const FString& DatasetPath)
                    {
                    #if WITH_INTEGRATED_DYNAMICS_RUNTIME
                        return IntegratedDynamics::LoadPrisonDataset(TCHAR_TO_UTF8(*DatasetPath));
                    #else
                        return false;
                    #endif
                    }

                    bool FIntegratedDynamicsDataShim::SavePrisonDataset(const FString& DatasetPath)
                    {
                    #if WITH_INTEGRATED_DYNAMICS_RUNTIME
                        return IntegratedDynamics::SavePrisonDataset(TCHAR_TO_UTF8(*DatasetPath));
                    #else
                        return false;
                    #endif
                    }
                }
                ```

                File: Source/Nyghtshade/Nyghtshade.Build.cs (additions)
                ```csharp
                using System.IO;

                public class Nyghtshade : ModuleRules
                {
                    public Nyghtshade(ReadOnlyTargetRules Target) : base(Target)
                    {
                        string ProjectRoot = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", ".."));
                        string CrownRoot = Path.GetFullPath(Path.Combine(ProjectRoot, "..", "crown-master", "Source"));
                        string IntegratedDynamicsRoot = Path.GetFullPath(Path.Combine(ProjectRoot, "..", "IntegratedDynamics", "Source"));

                        bool HasCrown = Directory.Exists(CrownRoot);
                        bool HasIntegratedDynamics = Directory.Exists(IntegratedDynamicsRoot);

                        PublicDefinitions.Add($"WITH_CROWN={(HasCrown ? 1 : 0)}");
                        PublicDefinitions.Add($"WITH_INTEGRATED_DYNAMICS={(HasIntegratedDynamics ? 1 : 0)}");
                    }
                }
                ```

                File: Tools/nh_foreman_audit.ps1
                ```powershell
                param(
                  [Parameter(Mandatory=$true)]
                  [string]$WorkspaceRoot
                )

                $ErrorActionPreference = "Stop"

                function Get-ProjectScore {
                  param([string]$ProjectPath)
                  $score = 0
                  $projName = [IO.Path]::GetFileNameWithoutExtension($ProjectPath)
                  $projDir = Split-Path -Parent $ProjectPath
                  if ($projName -match "Nyghtshade") { $score += 40 }
                  if ($projDir -match "Nyghtshade") { $score += 10 }
                  if (Test-Path (Join-Path $projDir "Source\\Nyghtshade")) { $score += 25 }
                  if (Test-Path (Join-Path $projDir "Config\\DefaultGame.ini")) { $score += 5 }
                  if (Test-Path (Join-Path $projDir "Content\\PrisonCore")) { $score += 5 }
                  if (Test-Path (Join-Path $projDir "Plugins")) { $score += 5 }
                  return $score
                }

                $projects = Get-ChildItem -Path $WorkspaceRoot -Recurse -Filter *.uproject -File -ErrorAction SilentlyContinue
                if (-not $projects) {
                  Write-Host "No Unreal projects found in $WorkspaceRoot." -ForegroundColor Yellow
                  exit 1
                }

                $scored = $projects | ForEach-Object {
                  [pscustomobject]@{
                    Path = $_.FullName
                    Score = Get-ProjectScore $_.FullName
                  }
                } | Sort-Object Score -Descending, Path

                $selected = $scored | Select-Object -First 1
                Write-Host "Selected Nyghtshade project:" -ForegroundColor Cyan
                Write-Host $selected.Path
                Write-Host ""
                Write-Host "All candidates:" -ForegroundColor Cyan
                $scored | Format-Table -AutoSize
                ```

                File: Tools/nh_clean_project.ps1
                ```powershell
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
                  $globalDDC = Join-Path $env:LOCALAPPDATA "UnrealEngine\\Common\\DerivedDataCache"
                  Zap $globalDDC
                }

                Write-Host "Cleanup done."
                ```

                File: Tools/nh_copy_uassets.ps1
                ```powershell
                param(
                  [Parameter(Mandatory=$true)]
                  [string]$VaultPath,
                  [Parameter(Mandatory=$true)]
                  [string]$ProjectDir,
                  [string]$DestSubdir = "PrisonCore"
                )

                $ErrorActionPreference = "Stop"

                $dest = Join-Path $ProjectDir "Content\\$DestSubdir"
                New-Item -ItemType Directory -Force -Path $dest | Out-Null

                Write-Host "Copying .uasset/.umap into: $dest"
                $items = Get-ChildItem -LiteralPath $VaultPath -Recurse -Force -File -ErrorAction SilentlyContinue |
                  Where-Object { $_.Extension -in ".uasset",".umap" }

                foreach ($it in $items) {
                  $rel = $it.FullName.Substring($VaultPath.Length).TrimStart("\\","/")
                  $target = Join-Path $dest $rel
                  $targetDir = Split-Path -Parent $target
                  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
                  Copy-Item -LiteralPath $it.FullName -Destination $target -Force
                }

                Write-Host ("Copied {0} assets." -f $items.Count)
                ```

                File: Tools/nh_build_safe.ps1
                ```powershell
                param(
                  [Parameter(Mandatory=$true)]
                  [string]$UProjectPath,
                  [Parameter(Mandatory=$true)]
                  [string]$UE57Root,
                  [int]$MaxParallelActions = 6
                )

                $ErrorActionPreference = "Stop"

                Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

                $buildBat = Join-Path $UE57Root "Engine\\Build\\BatchFiles\\Build.bat"
                if (!(Test-Path $buildBat)) { throw "Build.bat not found at $buildBat" }

                $projName = [IO.Path]::GetFileNameWithoutExtension($UProjectPath)
                $target = "${projName}Editor"

                Write-Host "Building $target with MaxParallelActions=$MaxParallelActions"
                & $buildBat $target Win64 Development "-Project=$UProjectPath" -WaitMutex "-MaxParallelActions=$MaxParallelActions"
                ```

                File: Content/Python/vault_to_unreal.py
                ```python
                import os
                import re
                import unreal

                VAULT_ROOT = r"C:\\Nyghtshade_Assets_Vault"
                DEST_ROOT = "/Game/Nyghtshade/Final_Prison_Assets"
                TEXTURES_DIR = f"{DEST_ROOT}/Textures"
                MESHES_DIR = f"{DEST_ROOT}/Meshes"
                MATS_DIR = f"{DEST_ROOT}/Materials"

                CONCRETE_HINTS = ("concrete", "cement", "wall", "floor", "stone")
                METAL_HINTS = ("metal", "steel", "iron", "pipe", "grate", "bar")


                def log(msg):
                    unreal.log(msg)


                def ensure_dir(path):
                    if not unreal.EditorAssetLibrary.does_directory_exist(path):
                        unreal.EditorAssetLibrary.make_directory(path)


                def find_files(root, exts):
                    out = []
                    for base, _, files in os.walk(root):
                        for f in files:
                            if os.path.splitext(f)[1].lower() in exts:
                                out.append(os.path.join(base, f))
                    return out


                def safe_asset_name(s):
                    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
                    s = re.sub(r"_+", "_", s).strip("_")
                    return s[:120] if len(s) > 120 else s


                def import_file(src_path, dest_path, automated=True, replace=True):
                    task = unreal.AssetImportTask()
                    task.filename = src_path
                    task.destination_path = dest_path
                    task.automated = automated
                    task.replace_existing = replace
                    task.save = True
                    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
                    return task.imported_object_paths


                def make_master_material(name, dest_path):
                    ensure_dir(dest_path)
                    pkg = f"{dest_path}/{name}"
                    if unreal.EditorAssetLibrary.does_asset_exist(pkg):
                        return unreal.load_asset(pkg)

                    mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                        name, dest_path, unreal.Material, unreal.MaterialFactoryNew()
                    )

                    mel = unreal.MaterialEditingLibrary
                    base_param = mel.create_material_expression(
                        mat, unreal.MaterialExpressionTextureSampleParameter2D, -600, 0
                    )
                    base_param.parameter_name = "BaseColorTex"
                    norm_param = mel.create_material_expression(
                        mat, unreal.MaterialExpressionTextureSampleParameter2D, -600, 220
                    )
                    norm_param.parameter_name = "NormalTex"
                    norm_param.sampler_type = unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL

                    mel.connect_material_property(base_param, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
                    mel.connect_material_property(norm_param, "RGB", unreal.MaterialProperty.MP_NORMAL)
                    mel.recompile_material(mat)
                    unreal.EditorAssetLibrary.save_asset(pkg)
                    return mat


                def make_material_instance(mi_name, parent_mat, dest_path):
                    ensure_dir(dest_path)
                    pkg = f"{dest_path}/{mi_name}"
                    if unreal.EditorAssetLibrary.does_asset_exist(pkg):
                        return unreal.load_asset(pkg)

                    mi = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                        mi_name,
                        dest_path,
                        unreal.MaterialInstanceConstant,
                        unreal.MaterialInstanceConstantFactoryNew(),
                    )
                    mi.set_editor_property("parent", parent_mat)
                    unreal.EditorAssetLibrary.save_asset(pkg)
                    return mi


                def set_mi_texture(mi, param_name, tex_asset):
                    unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                        mi, param_name, tex_asset
                    )
                    unreal.EditorAssetLibrary.save_loaded_asset(mi)


                def choose_master_for_name(asset_name_lower):
                    if any(h in asset_name_lower for h in METAL_HINTS):
                        return "Master_Metal"
                    if any(h in asset_name_lower for h in CONCRETE_HINTS):
                        return "Master_Concrete"
                    return "Master_Prison"


                def guess_texture_role(filename_lower):
                    if "normal" in filename_lower or filename_lower.endswith("_n.png") or "_n_" in filename_lower:
                        return "NormalTex"
                    return "BaseColorTex"


                def assign_materials_to_mesh(mesh_asset, material_asset):
                    if not isinstance(mesh_asset, unreal.StaticMesh):
                        return
                    slots = mesh_asset.get_editor_property("static_materials")
                    for i in range(len(slots)):
                        slots[i].material_interface = material_asset
                    mesh_asset.set_editor_property("static_materials", slots)
                    mesh_asset.mark_package_dirty()
                    unreal.EditorAssetLibrary.save_loaded_asset(mesh_asset)


                def run():
                    ensure_dir(DEST_ROOT)
                    ensure_dir(TEXTURES_DIR)
                    ensure_dir(MESHES_DIR)
                    ensure_dir(MATS_DIR)

                    master_prison = make_master_material("Master_Prison_Material", MATS_DIR)
                    master_conc = make_master_material("Master_Concrete", MATS_DIR)
                    master_metal = make_master_material("Master_Metal", MATS_DIR)
                    masters = {
                        "Master_Prison": master_prison,
                        "Master_Prison_Material": master_prison,
                        "Master_Concrete": master_conc,
                        "Master_Metal": master_metal,
                    }

                    fbx_files = find_files(VAULT_ROOT, {".fbx"})
                    png_files = find_files(VAULT_ROOT, {".png"})
                    log(f"Found FBX: {len(fbx_files)} | PNG: {len(png_files)}")

                    texture_map = {}
                    for p in png_files:
                        imported = import_file(p, TEXTURES_DIR)
                        for obj_path in imported:
                            tex = unreal.load_asset(obj_path)
                            key = (os.path.dirname(p).lower(), os.path.splitext(os.path.basename(p))[0].lower())
                            texture_map[key] = tex

                    for fbx in fbx_files:
                        base_name = safe_asset_name(os.path.splitext(os.path.basename(fbx))[0])
                        imported = import_file(fbx, MESHES_DIR)

                        mesh = None
                        for obj_path in imported:
                            obj = unreal.load_asset(obj_path)
                            if isinstance(obj, unreal.StaticMesh):
                                mesh = obj
                                break
                        if not mesh:
                            log(f"Skipping (no StaticMesh found): {fbx}")
                            continue

                        master_key = choose_master_for_name(base_name.lower())
                        parent = masters.get(master_key, master_prison)
                        mi = make_material_instance(f"MI_{base_name}", parent, MATS_DIR)

                        folder = os.path.dirname(fbx).lower()
                        candidates = []
                        for (fld, bn), tex in texture_map.items():
                            if fld == folder and (bn.startswith(base_name.lower()) or base_name.lower() in bn):
                                candidates.append((bn, tex))

                        for bn, tex in candidates:
                            role = guess_texture_role(bn)
                            set_mi_texture(mi, role, tex)

                        assign_materials_to_mesh(mesh, mi)

                    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
                    redir_objs = []
                    for a in unreal.EditorAssetLibrary.list_assets(DEST_ROOT, recursive=True, include_folder=False):
                        obj = unreal.load_asset(a)
                        if isinstance(obj, unreal.ObjectRedirector):
                            redir_objs.append(obj)
                    if redir_objs:
                        asset_tools.fixup_redirectors(redir_objs)

                    log("Vault import + material assignment + redirector fixup complete.")


                if __name__ == "__main__":
                    run()
                ```

                File: README_ForemanPack.md
                ```markdown
                # Foreman Pack (Windows + UE 5.7)

                ## 1) Audit workspace for Nyghtshade project
                ```powershell
                pwsh -File Tools/nh_foreman_audit.ps1 -WorkspaceRoot "C:\\Users\\FRANK\\Desktop\\ltvall\\shiny-happiness"
                ```

                ## 2) Clean project artifacts
                ```powershell
                pwsh -File Tools/nh_clean_project.ps1 -ProjectDir "C:\\Path\\To\\NyghtshadeProject" -NukeDerivedDataCache
                ```

                ## 3) Copy Vault assets into the project
                ```powershell
                pwsh -File Tools/nh_copy_uassets.ps1 -VaultPath "C:\\Nyghtshade_Assets_Vault" -ProjectDir "C:\\Path\\To\\NyghtshadeProject" -DestSubdir "PrisonCore"
                ```

                ## 4) Run the Unreal Python import
                - Copy `Content/Python/vault_to_unreal.py` into your project.
                - Launch the project once, then execute the script in Unreal:
                ```python
                import vault_to_unreal
                vault_to_unreal.run()
                ```

                ## 5) Build safely with capped parallel actions
                ```powershell
                pwsh -File Tools/nh_build_safe.ps1 -UProjectPath "C:\\Path\\To\\NyghtshadeProject\\Nyghtshade.uproject" -UE57Root "C:\\Program Files\\Epic Games\\UE_5.7" -MaxParallelActions 6
                ```
                ```
                """
            ).strip(),
        ),
    ]


def all_templates() -> List[TaskTemplate]:
    return unity_task_templates() + unreal_task_templates()


def normalize_filter(text: str) -> str:
    return "".join(char for char in text.lower().strip() if char.isalnum())


def filter_templates(engine: str | None, name: str | None) -> List[TaskTemplate]:
    templates = all_templates()
    if engine:
        normalized_engine = engine.strip().lower()
        templates = [
            template
            for template in templates
            if template.engine.lower() == normalized_engine
        ]
    if name:
        normalized_name = normalize_filter(name)
        templates = [
            template
            for template in templates
            if normalized_name in normalize_filter(template.name)
        ]
    return templates


def format_template(template: TaskTemplate) -> str:
    checklist = "\n".join(f"- {item}" for item in template.checklist)
    return dedent(
        f"""
        Name: {template.name}
        Engine: {template.engine}
        Summary: {template.summary}

        Checklist:
        {checklist}

        Copilot Prompt:
        {template.copilot_prompt}

        Code Skeleton:
        {template.code_skeleton}
        """
    ).strip()


def write_output(content: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
        return
    print(content)


def build_templates_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenSourceLudus: AI Copilot templates for Unity and Unreal Engine",
    )
    parser.add_argument(
        "--engine",
        choices=["unity", "unreal"],
        help="Filter templates by engine.",
    )
    parser.add_argument(
        "--task",
        help="Filter by task name.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available templates.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for templates.",
    )
    parser.add_argument(
        "--output",
        help="Write output to a file.",
    )
    parser.add_argument(
        "--all-in-one",
        action="store_true",
        help="Emit all templates concatenated into a single file.",
    )
    return parser


def parse_simple_yaml(raw: str) -> Dict[str, Any]:
    data_stack: list[Dict[str, Any] | list[Any]] = [{}]
    indent_stack = [0]

    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_value = line.strip()

        while indent < indent_stack[-1]:
            data_stack.pop()
            indent_stack.pop()

        if key_value.startswith("- "):
            value = key_value[2:].strip()
            if not isinstance(data_stack[-1], list):
                raise MirrorError("Invalid YAML list structure.")
            data_stack[-1].append(coerce_scalar(value))
            continue

        if ":" in key_value:
            key, value = key_value.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                container: Dict[str, Any] | list[Any]
                if lookahead_is_list(raw, line):
                    container = []
                else:
                    container = {}
                data_stack[-1][key] = container
                data_stack.append(container)
                indent_stack.append(indent + 2)
            else:
                data_stack[-1][key] = coerce_scalar(value)
            continue

    if not isinstance(data_stack[0], dict):
        raise MirrorError("Invalid YAML document.")
    return data_stack[0]


def lookahead_is_list(raw: str, current_line: str) -> bool:
    lines = raw.splitlines()
    try:
        index = lines.index(current_line)
    except ValueError:
        return False
    for next_line in lines[index + 1 :]:
        if not next_line.strip() or next_line.strip().startswith("#"):
            continue
        return next_line.lstrip().startswith("-")
    return False


def coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    return value


def dump_simple_yaml(data: Dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(dump_simple_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def load_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if yaml:
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise MirrorError(f"YAML document at {path} is not a mapping.")
        return loaded
    return parse_simple_yaml(raw)


def dump_yaml(data: Dict[str, Any]) -> str:
    if yaml:
        return yaml.safe_dump(data, sort_keys=False)
    return dump_simple_yaml(data)


def ensure_layout(root: Path) -> None:
    (root / "packs").mkdir(parents=True, exist_ok=True)
    (root / "keys" / "trusted").mkdir(parents=True, exist_ok=True)
    (root / "keys" / "revoked").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(parents=True, exist_ok=True)
    (root / "sync").mkdir(parents=True, exist_ok=True)
    (root / "integrity").mkdir(parents=True, exist_ok=True)


def index_path(root: Path) -> Path:
    return root / INDEX_FILENAME


def integrity_path(root: Path) -> Path:
    return root / INTEGRITY_FILENAME


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_integrity_hash(root: Path) -> None:
    index_file = index_path(root)
    if not index_file.exists():
        raise MirrorError("Cannot write integrity hash without index.yaml.")
    integrity = integrity_path(root)
    integrity.parent.mkdir(parents=True, exist_ok=True)
    integrity.write_text(compute_sha256(index_file), encoding="utf-8")


def verify_integrity_hash(root: Path) -> None:
    integrity = integrity_path(root)
    index_file = index_path(root)
    if not integrity.exists():
        raise MirrorError("Missing integrity hash file.")
    if not index_file.exists():
        raise MirrorError("Missing index.yaml.")
    expected = integrity.read_text(encoding="utf-8").strip()
    actual = compute_sha256(index_file)
    if expected != actual:
        raise MirrorError("Index integrity hash mismatch.")


def build_index() -> Dict[str, Any]:
    return {
        "registry": {
            "schema_version": 1,
            "mirror_created": utc_now(),
            "last_verified": utc_now(),
        },
        "packs": {},
    }


def write_index_atomic(root: Path, index_data: Dict[str, Any]) -> None:
    serialized = dump_yaml(index_data)
    tmp_dir = root / "integrity"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding="utf-8") as handle:
        handle.write(serialized)
        tmp_name = handle.name
    target = index_path(root)
    Path(tmp_name).replace(target)
    write_integrity_hash(root)


def load_index(root: Path) -> Dict[str, Any]:
    index_file = index_path(root)
    if not index_file.exists():
        raise MirrorError("Missing index.yaml.")
    return load_yaml(index_file)


def resolve_pack_dir(root: Path, pack_id: str, version: str) -> Path:
    return root / "packs" / pack_id / version


def parse_semver(version: str) -> tuple:
    parts = version.split(".")
    if all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return tuple(parts)


def ensure_monotonic_versions(versions: List[str]) -> None:
    sorted_versions = sorted(versions, key=parse_semver)
    if versions != sorted_versions:
        raise MirrorError("Pack versions are not monotonic in index.yaml.")


def validate_migration_chain(pack_dir: Path) -> None:
    migrations_dir = pack_dir / "migrations"
    if not migrations_dir.exists() or not any(migrations_dir.iterdir()):
        raise MirrorError("Missing migration files.")


def read_signature_bytes(signature_path: Path) -> bytes:
    raw = signature_path.read_bytes()
    try:
        decoded = base64.b64decode(raw, validate=True)
        if decoded:
            return decoded
    except (ValueError, binascii.Error):
        pass
    return raw


def load_public_key(key_path: Path):
    if not importlib.util.find_spec("cryptography"):
        raise MirrorError("cryptography is required to load public keys.")
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_public_key(key_path.read_bytes())


def verify_signature(key_path: Path, data: bytes, signature: bytes) -> None:
    if not importlib.util.find_spec("cryptography"):
        verify_signature_with_openssl(key_path, data, signature)
        return

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    public_key = load_public_key(key_path)
    if hasattr(public_key, "verify"):
        if public_key.__class__.__name__.startswith("_RSAPublicKey"):
            public_key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
            return
        if public_key.__class__.__name__.startswith("_EllipticCurvePublicKey"):
            try:
                public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            except ValueError:
                r, s = decode_dss_signature(signature)
                public_key.verify((r, s), data, ec.ECDSA(hashes.SHA256()))
            return
    raise MirrorError("Unsupported public key type.")


def verify_signature_with_openssl(key_path: Path, data: bytes, signature: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir) / "data.bin"
        sig_path = Path(tmp_dir) / "sig.bin"
        data_path.write_bytes(data)
        sig_path.write_bytes(signature)
        openssl_path = shutil.which("openssl")
        if not openssl_path:
            raise MirrorError("OpenSSL not available for signature verification.")
        command = [
            openssl_path,
            "dgst",
            "-sha256",
            "-verify",
            str(key_path),
            "-signature",
            str(sig_path),
            str(data_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise MirrorError("Signature verification failed via OpenSSL.")


def resolve_key_path(root: Path, pack_meta: Dict[str, Any], signature_dir: Path) -> Path:
    signature_info = pack_meta.get("signature", {})
    key_id = signature_info.get("key_id")
    if not key_id:
        key_id_path = signature_dir / "key_id.txt"
        if key_id_path.exists():
            key_id = key_id_path.read_text(encoding="utf-8").strip()
    if not key_id:
        trusted_keys = sorted((root / "keys" / "trusted").glob("*.pem"))
        if len(trusted_keys) == 1:
            return trusted_keys[0]
        raise MirrorError("Unable to resolve signing key for pack.")
    key_path = root / "keys" / "trusted" / key_id
    if not key_path.exists():
        raise MirrorError("Signing key not found in trusted keys.")
    revoked_path = root / "keys" / "revoked" / key_id
    if revoked_path.exists():
        raise MirrorError("Signing key has been revoked.")
    return key_path


def validate_pack_hashes(pack_dir: Path, pack_meta: Dict[str, Any]) -> None:
    hashes = pack_meta.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise MirrorError("pack.yaml missing hashes mapping.")
    for rel_path, expected_hash in hashes.items():
        file_path = pack_dir / rel_path
        if not file_path.exists():
            raise MirrorError(f"Missing file referenced in pack.yaml: {rel_path}")
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            raise MirrorError(f"Hash mismatch for {rel_path}.")


def compute_pack_digest(pack_meta: Dict[str, Any]) -> str:
    hashes = pack_meta.get("hashes")
    if not isinstance(hashes, dict):
        raise MirrorError("pack.yaml missing hashes mapping for digest.")
    digest = hashlib.sha256()
    for path_key in sorted(hashes.keys()):
        digest.update(path_key.encode("utf-8"))
        digest.update(hashes[path_key].encode("utf-8"))
    return digest.hexdigest()


def load_pack_meta(pack_dir: Path) -> Dict[str, Any]:
    pack_yaml = pack_dir / "pack.yaml"
    if not pack_yaml.exists():
        raise MirrorError("Missing pack.yaml.")
    return load_yaml(pack_yaml)


def verify_pack(root: Path, pack_id: str, version: str) -> str:
    pack_dir = resolve_pack_dir(root, pack_id, version)
    if not pack_dir.exists():
        raise MirrorError(f"Missing pack directory for {pack_id}@{version}.")
    pack_meta = load_pack_meta(pack_dir)
    if pack_meta.get("pack") != pack_id or str(pack_meta.get("version")) != version:
        raise MirrorError("pack.yaml metadata does not match directory name.")

    validate_pack_hashes(pack_dir, pack_meta)

    signature_dir = pack_dir / "signature"
    signature_file = signature_dir / "pack.sig"
    if not signature_file.exists():
        raise MirrorError("Missing signature file.")
    key_path = resolve_key_path(root, pack_meta, signature_dir)
    signature_bytes = read_signature_bytes(signature_file)
    pack_yaml = (pack_dir / "pack.yaml").read_bytes()
    verify_signature(key_path, pack_yaml, signature_bytes)

    validate_migration_chain(pack_dir)

    return compute_pack_digest(pack_meta)


def verify_index_and_packs(root: Path) -> None:
    verify_integrity_hash(root)
    index = load_index(root)
    packs = index.get("packs", {})
    if not isinstance(packs, dict):
        raise MirrorError("index.yaml packs section missing or invalid.")
    for pack_id, pack_info in packs.items():
        versions = pack_info.get("versions")
        if not isinstance(versions, list) or not versions:
            raise MirrorError(f"Pack {pack_id} has no versions.")
        versions = [str(v) for v in versions]
        ensure_monotonic_versions(versions)
        for idx, version in enumerate(versions):
            verify_pack(root, pack_id, str(version))
            if idx > 0:
                validate_migration_chain(resolve_pack_dir(root, pack_id, version))


def append_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_audit_payload(event: str, session_id: str, command_id: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {
        "event": event,
        "session_id": session_id,
        "command_id": command_id,
        "timestamp": utc_now(),
    }
    payload.update(kwargs)
    return payload


def init_registry(root: Path) -> None:
    ensure_layout(root)
    if index_path(root).exists():
        raise MirrorError("Mirror already initialized.")
    index_data = build_index()
    write_index_atomic(root, index_data)


def registry_status(root: Path) -> Dict[str, Any]:
    if not index_path(root).exists():
        return {"status": "uninitialized"}
    try:
        verify_index_and_packs(root)
        status = "verified"
    except MirrorError as exc:
        status = f"invalid: {exc}"
    index = load_index(root)
    pack_count = len(index.get("packs", {}))
    return {"status": status, "packs": pack_count}


def sync_registry(root: Path, source: str | None, url: str | None, actor: str, surface: str) -> None:
    if not source and not url:
        raise MirrorError("Sync requires --source or --url.")

    session_id = str(uuid.uuid4())
    command_id = str(uuid.uuid4())
    added_packs: list[str] = []
    rejected_packs: list[str] = []

    ensure_layout(root)
    if not index_path(root).exists():
        write_index_atomic(root, build_index())

    local_index = load_index(root)
    local_packs = local_index.get("packs", {})
    if source:
        upstream_root = Path(source).expanduser()
        upstream_index = load_yaml(upstream_root / INDEX_FILENAME)
    else:
        upstream_index = fetch_remote_index(url)

    updates = False
    for pack_id, pack_info in upstream_index.get("packs", {}).items():
        versions = [str(v) for v in pack_info.get("versions", [])]
        if not versions:
            continue
        local_versions = [str(v) for v in local_packs.get(pack_id, {}).get("versions", [])]
        for version in versions:
            if version in local_versions:
                compare_existing_pack(root, pack_id, version, source, url)
        missing = [version for version in versions if version not in local_versions]
        for version in missing:
            try:
                pack_digest = fetch_and_verify_pack(root, pack_id, version, source, url)
                added_packs.append(f"{pack_id}@{version}")
                local_versions.append(version)
                local_versions = sorted(local_versions, key=parse_semver)
                local_packs.setdefault(pack_id, {})["versions"] = local_versions
                local_packs[pack_id]["latest"] = local_versions[-1]
                updates = True
            except MirrorError as exc:
                rejected_packs.append(f"{pack_id}@{version}: {exc}")

        if local_versions:
            ensure_monotonic_versions(local_versions)

    if updates:
        local_index["packs"] = local_packs
        local_index["registry"]["last_verified"] = utc_now()
        write_index_atomic(root, local_index)

    append_log(
        root / AUDIT_LOG,
        build_audit_payload(
            "registry_sync_completed",
            session_id,
            command_id,
            actor=actor,
            surface=surface,
            result="success" if not rejected_packs else "partial",
            added_packs=added_packs,
            rejected_packs=rejected_packs,
        ),
    )
    append_log(
        root / SYNC_LOG,
        build_audit_payload(
            "registry_sync_details",
            session_id,
            command_id,
            added_packs=added_packs,
            rejected_packs=rejected_packs,
        ),
    )
    (root / LAST_SYNC).write_text(
        json.dumps(
            {
                "last_sync": utc_now(),
                "added_packs": added_packs,
                "rejected_packs": rejected_packs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if rejected_packs:
        raise MirrorError("Some packs failed verification during sync.")


def fetch_remote_index(url: str | None) -> Dict[str, Any]:
    if not url:
        raise MirrorError("Missing upstream URL.")
    import urllib.request

    index_url = url.rstrip("/") + f"/{INDEX_FILENAME}"
    with urllib.request.urlopen(index_url) as response:  # noqa: S310
        raw = response.read().decode("utf-8")
    if yaml:
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise MirrorError("Remote index.yaml invalid.")
        return loaded
    return parse_simple_yaml(raw)


def compare_existing_pack(root: Path, pack_id: str, version: str, source: str | None, url: str | None) -> None:
    local_pack_meta = load_pack_meta(resolve_pack_dir(root, pack_id, version))
    local_digest = compute_pack_digest(local_pack_meta)
    if source:
        upstream_pack_meta = load_pack_meta(resolve_pack_dir(Path(source).expanduser(), pack_id, version))
    else:
        upstream_pack_meta = fetch_remote_pack_meta(url, pack_id, version)
    upstream_digest = compute_pack_digest(upstream_pack_meta)
    if local_digest != upstream_digest:
        raise MirrorError("Conflict: same pack version has different hash.")


def fetch_and_verify_pack(root: Path, pack_id: str, version: str, source: str | None, url: str | None) -> str:
    if resolve_pack_dir(root, pack_id, version).exists():
        raise MirrorError("Pack already exists locally.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        pack_dir = temp_root / "packs" / pack_id / version
        pack_dir.mkdir(parents=True, exist_ok=True)

        if source:
            upstream_root = Path(source).expanduser()
            source_pack_dir = resolve_pack_dir(upstream_root, pack_id, version)
            if not source_pack_dir.exists():
                raise MirrorError("Pack not found in source mirror.")
            shutil.copytree(source_pack_dir, pack_dir, dirs_exist_ok=True)
        else:
            download_pack_from_url(url, pack_id, version, pack_dir)

        digest = verify_pack_in_temp(root, pack_dir)

        target_dir = resolve_pack_dir(root, pack_id, version)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(pack_dir, target_dir, dirs_exist_ok=False)

    return digest


def verify_pack_in_temp(root: Path, pack_dir: Path) -> str:
    pack_meta = load_yaml(pack_dir / "pack.yaml")
    pack_id = str(pack_meta.get("pack"))
    version = str(pack_meta.get("version"))
    if not pack_id or not version:
        raise MirrorError("pack.yaml missing pack or version.")
    validate_pack_hashes(pack_dir, pack_meta)
    signature_dir = pack_dir / "signature"
    signature_file = signature_dir / "pack.sig"
    if not signature_file.exists():
        raise MirrorError("Missing signature file.")
    key_path = resolve_key_path(root, pack_meta, signature_dir)
    signature_bytes = read_signature_bytes(signature_file)
    pack_yaml = (pack_dir / "pack.yaml").read_bytes()
    verify_signature(key_path, pack_yaml, signature_bytes)
    validate_migration_chain(pack_dir)
    return compute_pack_digest(pack_meta)


def fetch_remote_pack_meta(url: str | None, pack_id: str, version: str) -> Dict[str, Any]:
    if not url:
        raise MirrorError("Missing upstream URL.")
    import urllib.request

    pack_yaml_url = url.rstrip("/") + f"/packs/{pack_id}/{version}/pack.yaml"
    with urllib.request.urlopen(pack_yaml_url) as response:  # noqa: S310
        raw = response.read().decode("utf-8")
    if yaml:
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise MirrorError("Remote pack.yaml invalid.")
        return loaded
    return parse_simple_yaml(raw)


def download_pack_from_url(url: str | None, pack_id: str, version: str, dest_dir: Path) -> None:
    if not url:
        raise MirrorError("Missing upstream URL.")
    import urllib.request

    base_url = url.rstrip("/") + f"/packs/{pack_id}/{version}"
    pack_yaml_url = base_url + "/pack.yaml"
    with urllib.request.urlopen(pack_yaml_url) as response:  # noqa: S310
        pack_yaml = response.read().decode("utf-8")
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "pack.yaml").write_text(pack_yaml, encoding="utf-8")
    pack_meta = load_yaml(dest_dir / "pack.yaml")

    hashes = pack_meta.get("hashes", {})
    if not isinstance(hashes, dict):
        raise MirrorError("Remote pack.yaml missing hashes.")
    for rel_path in hashes.keys():
        file_url = base_url + f"/{rel_path}"
        file_path = dest_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(file_url) as response:  # noqa: S310
            file_path.write_bytes(response.read())

    signature_dir = dest_dir / "signature"
    signature_dir.mkdir(parents=True, exist_ok=True)
    signature_url = base_url + "/signature/pack.sig"
    with urllib.request.urlopen(signature_url) as response:  # noqa: S310
        (signature_dir / "pack.sig").write_bytes(response.read())

    key_id_url = base_url + "/signature/key_id.txt"
    try:
        with urllib.request.urlopen(key_id_url) as response:  # noqa: S310
            (signature_dir / "key_id.txt").write_bytes(response.read())
    except Exception:
        pass


def run_templates_cli(args: Iterable[str] | None) -> int:
    parser = build_templates_parser()
    args = parser.parse_args(args)

    templates = filter_templates(args.engine, args.task)
    if not templates:
        parser.error("No templates matched the provided filters.")

    if args.list:
        listing = "\n".join(
            f"{template.engine}:{template.name} - {template.summary}" for template in templates
        )
        write_output(listing, args.output)
        return 0

    if args.format == "json":
        payload = [asdict(template) for template in templates]
        write_output(json.dumps(payload, indent=2), args.output)
        return 0

    if args.all_in_one or len(templates) > 1:
        content = "\n\n".join(format_template(template) for template in templates)
        write_output(content, args.output)
        return 0

    content = format_template(templates[0])
    write_output(content, args.output)
    return 0


def build_registry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline-first registry mirror")
    parser.add_argument(
        "--mirror-root",
        default=str(MIRROR_ROOT_DEFAULT),
        help="Registry mirror root path (default: ~/.osl/registry).",
    )
    parser.add_argument(
        "--actor",
        default=os.getenv("USER", "unknown"),
        help="Actor name for audit logs.",
    )
    parser.add_argument(
        "--surface",
        default="cli",
        help="Surface identifier for audit logs (cli, unity, unreal, ci).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Initialize a new mirror layout.")
    subparsers.add_parser("status", help="Report mirror verification status.")
    subparsers.add_parser("verify", help="Verify mirror integrity and packs.")

    sync_parser = subparsers.add_parser("sync", help="Sync mirror from upstream source.")
    sync_parser.add_argument("--source", help="Local path to upstream mirror.")
    sync_parser.add_argument("--url", help="Upstream mirror base URL.")

    return parser


def run_registry_cli(args: Iterable[str] | None) -> int:
    parser = build_registry_parser()
    parsed = parser.parse_args(args)
    root = Path(parsed.mirror_root).expanduser()

    try:
        if parsed.command == "init":
            init_registry(root)
            print(f"Mirror initialized at {root}")
            return 0
        if parsed.command == "status":
            status = registry_status(root)
            print(json.dumps(status, indent=2))
            return 0
        if parsed.command == "verify":
            verify_index_and_packs(root)
            index = load_index(root)
            index["registry"]["last_verified"] = utc_now()
            write_index_atomic(root, index)
            print("Mirror verified")
            return 0
        if parsed.command == "sync":
            sync_registry(root, parsed.source, parsed.url, parsed.actor, parsed.surface)
            print("Mirror sync complete")
            return 0
    except MirrorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 1


def main(argv: Iterable[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] == "registry":
        return run_registry_cli(argv_list[1:])
    if argv_list and argv_list[0] == "templates":
        return run_templates_cli(argv_list[1:])
    return run_templates_cli(argv_list)


if __name__ == "__main__":
    raise SystemExit(main())
