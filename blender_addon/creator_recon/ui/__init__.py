"""Blender panels only; photo ROI interaction is not implemented yet."""

import bpy

from ..properties import ADDON_MODULE


class CREATOR_PT_reconstruction(bpy.types.Panel):
    """Expose the current scaffold's actual capabilities without mock results."""

    bl_label = "Creator Reconstruction"
    bl_idname = "CREATOR_PT_reconstruction"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Creator"

    def draw(self, context):
        layout = self.layout
        status = layout.box()
        status.label(text="Scaffold: configuration only", icon="INFO")
        status.label(text="Target: recover missing or broken thin rods.")
        status.label(text="Reconstruction and refinement are not implemented.")

        addon = context.preferences.addons.get(ADDON_MODULE)
        if addon is None:
            layout.label(text="Enable the add-on in Preferences to configure it.")
            return

        layout.prop(addon.preferences, "core_python")
        layout.prop(addon.preferences, "project_root")
        layout.operator("creator.check_configuration", icon="CHECKMARK")
        layout.label(text="Checking paths does not start a process.")
        if context.scene is not None:
            layout.label(text=context.scene.creator_recon.configuration_status)
