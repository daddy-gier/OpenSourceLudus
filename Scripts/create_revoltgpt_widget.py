"""Create an Editor Utility Widget for RevoltGPT inside Unreal Editor.

Run this inside the Unreal Editor Python console (Editor Scripting Utilities enabled).
"""
import unreal

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
editor_asset_lib = unreal.EditorAssetLibrary

# Destination path and asset name
package_path = "/Game/RevoltGPT/UI"
asset_name = "EUW_RevoltGPT_Panel"

# Factory for Editor Utility Widget Blueprints
factory = unreal.EditorUtilityWidgetBlueprintFactory()

# Create the folder if it doesn't exist
if not editor_asset_lib.does_directory_exist(package_path):
    editor_asset_lib.make_directory(package_path)
    unreal.log("Created directory: {}".format(package_path))

# Create or reuse the asset
asset_path = "{}/{}".format(package_path, asset_name)
existing = editor_asset_lib.find_asset_data(asset_path)
if existing.is_valid():
    unreal.log_warning("Asset already exists: {}".format(asset_path))
    new_asset = editor_asset_lib.load_asset(asset_path)
else:
    new_asset = asset_tools.create_asset(
        asset_name,
        package_path,
        unreal.EditorUtilityWidgetBlueprint,
        factory,
    )
    unreal.log("Created Editor Utility Widget: {}".format(asset_path))

# Open the widget blueprint for editing
unreal.AssetEditorSubsystem().open_editor_for_assets([new_asset])

# Create a basic widget tree with named widgets
widget_bp = editor_asset_lib.load_asset(asset_path)
widget_tree = widget_bp.get_widget_tree()

canvas = widget_tree.construct_widget(unreal.CanvasPanel, "RootCanvas")
widget_tree.set_root_widget(canvas)

prompt_box = widget_tree.construct_widget(unreal.EditableTextBox, "PromptTextBox")
canvas.add_child(prompt_box)
prompt_box.set_editor_property("hint_text", unreal.Text("Enter prompt..."))
prompt_box.set_editor_property("minimum_desired_width", 600)

send_btn = widget_tree.construct_widget(unreal.Button, "SendButton")
canvas.add_child(send_btn)

btn_text = widget_tree.construct_widget(unreal.TextBlock, "SendLabel")
btn_text.set_editor_property("text", unreal.Text("Send to RevoltGPT"))
send_btn.add_child(btn_text)

resp = widget_tree.construct_widget(unreal.TextBlock, "ResponseText")
resp.set_editor_property("auto_wrap_text", True)
resp.set_editor_property("text", unreal.Text("Awaiting response..."))
canvas.add_child(resp)

# Save asset
editor_asset_lib.save_asset(asset_path)

unreal.log("Created Editor Utility Widget with named children:")
unreal.log("PromptTextBox, SendButton, ResponseText")
