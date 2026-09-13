"""Configure the Creator add-on inside the dedicated D/project Blender profile."""
from pathlib import Path

import addon_utils
import bpy

root = Path(__file__).resolve().parents[1]
expected_config = (root / '.local/blender-profile/config').resolve()
actual_config = Path(bpy.utils.user_resource('CONFIG')).resolve()
if actual_config != expected_config:
    raise RuntimeError('Use Start-CreatorBlender.ps1 to select the dedicated profile first.')
addon_utils.enable('creator_recon', default_set=True, persistent=True)
addon = bpy.context.preferences.addons.get('creator_recon')
if addon is None:
    raise RuntimeError('Creator add-on registration failed')
addon.preferences.project_root = str(root)
addon.preferences.core_python = str(root / 'reconstruction/.venv/Scripts/python.exe')
from creator_recon.bridge.configuration import validate_configuration  # noqa: E402

issues = validate_configuration(addon.preferences.core_python, addon.preferences.project_root)
if issues:
    raise RuntimeError('; '.join(issues))
bpy.context.preferences.filepaths.temporary_directory = str(root / '.local/tmp')
bpy.ops.wm.save_userpref()
print('Creator Blender configuration PASS: ' + str(actual_config))