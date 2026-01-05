# Unreal Engine Python script: vault_to_unreal.py
import os
import re
from pathlib import Path

import unreal

VAULT_ROOT = r"C:\Nyghtshade_Assets_Vault"
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
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                out.append(os.path.join(base, fname))
    return out


def safe_asset_name(value):
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:120] if len(value) > 120 else value


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
    if any(hint in asset_name_lower for hint in METAL_HINTS):
        return "Master_Metal"
    if any(hint in asset_name_lower for hint in CONCRETE_HINTS):
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
    for png in png_files:
        imported = import_file(png, TEXTURES_DIR)
        for obj_path in imported:
            tex = unreal.load_asset(obj_path)
            key = (os.path.dirname(png).lower(), os.path.splitext(os.path.basename(png))[0].lower())
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
    for asset_path in unreal.EditorAssetLibrary.list_assets(DEST_ROOT, recursive=True, include_folder=False):
        obj = unreal.load_asset(asset_path)
        if isinstance(obj, unreal.ObjectRedirector):
            redir_objs.append(obj)
    if redir_objs:
        asset_tools.fixup_redirectors(redir_objs)

    log("✅ Vault import + material assignment + redirector fixup complete.")


if __name__ == "__main__":
    run()
