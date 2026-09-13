"""Small host settings, with no model arrays or process handles."""

import bpy
from bpy.props import StringProperty

ADDON_MODULE = __package__.split(".")[0]


class CreatorPreferences(bpy.types.AddonPreferences):
    """Machine-local locations for the independent reconstruction core."""

    bl_idname = ADDON_MODULE

    core_python: StringProperty(
        name="Core Python",
        description="Absolute path to the external core environment's Python executable",
        subtype="FILE_PATH",
        default="",
    )
    project_root: StringProperty(
        name="Project root",
        description="Absolute path to the Creator project containing reconstruction/pyproject.toml",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "core_python")
        layout.prop(self, "project_root")
        layout.operator("creator.check_configuration", icon="CHECKMARK")
        layout.label(text="Configuration scaffold only; reconstruction is not implemented.")


class CreatorSceneProperties(bpy.types.PropertyGroup):
    """Transient display feedback, not an authoritative core RunStatus."""

    configuration_status: StringProperty(
        name="Configuration status",
        default="Configuration has not been checked.",
        options={"SKIP_SAVE"},
    )
