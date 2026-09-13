"""Run using Blender --background --factory-startup --python <this file>."""

import importlib
import sys
from pathlib import Path

import addon_utils
import bpy

addon_directory = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(addon_directory))

creator_recon = importlib.import_module("creator_recon")

initial_objects = tuple(bpy.data.objects)
for cycle in range(2):
    creator_recon.register()
    assert hasattr(bpy.types.Scene, "creator_recon")
    assert (
        bpy.context.scene.creator_recon.configuration_status
        == "Configuration has not been checked."
    )
    for registered_class in creator_recon._CLASSES:
        assert registered_class.is_registered
    creator_recon.unregister()
    assert not hasattr(bpy.types.Scene, "creator_recon")
    for registered_class in creator_recon._CLASSES:
        assert not registered_class.is_registered

addon_utils.enable("creator_recon", default_set=True)
try:
    preferences = bpy.context.preferences.addons["creator_recon"].preferences
    assert preferences.core_python == ""
    assert preferences.project_root == ""
    assert bpy.ops.creator.check_configuration() == {"CANCELLED"}
    assert "not configured" in bpy.context.scene.creator_recon.configuration_status
finally:
    addon_utils.disable("creator_recon", default_set=True)

assert "creator_recon" not in bpy.context.preferences.addons
assert not hasattr(bpy.types.Scene, "creator_recon")
assert tuple(bpy.data.objects) == initial_objects
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert "pydantic" not in sys.modules
print(
    "Creator add-on smoke check passed: registration, preferences, path-check operator, clean unload."
)
