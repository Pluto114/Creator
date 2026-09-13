"""Blender host for Creator's future thin-structure reconstruction workflow.

This legacy add-on currently provides configuration and a status panel only.
The external core must never be imported into Blender's Python environment.
"""

import bpy
from bpy.props import PointerProperty

from .operators import CREATOR_OT_check_configuration
from .properties import CreatorPreferences, CreatorSceneProperties
from .ui import CREATOR_PT_reconstruction

bl_info = {
    "name": "Creator: Thin Structure Reconstruction",
    "author": "Creator contributors",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar > Creator",
    "description": "Configuration scaffold for thin-structure reconstruction research",
    "category": "3D View",
}

_CLASSES = (
    CreatorPreferences,
    CreatorSceneProperties,
    CREATOR_OT_check_configuration,
    CREATOR_PT_reconstruction,
)


def register():
    """Register host UI only; do not start processes or load model packages."""
    registered = []
    try:
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
            registered.append(cls)
        bpy.types.Scene.creator_recon = PointerProperty(type=CreatorSceneProperties)
    except Exception:
        for cls in reversed(registered):
            bpy.utils.unregister_class(cls)
        raise


def unregister():
    """Release the properties and classes owned by this add-on."""
    if hasattr(bpy.types.Scene, "creator_recon"):
        del bpy.types.Scene.creator_recon
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
