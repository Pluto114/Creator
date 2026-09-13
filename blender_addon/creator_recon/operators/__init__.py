"""Explicit host actions; research operators will be added after the core works."""

import bpy

from ..bridge.configuration import validate_configuration
from ..properties import ADDON_MODULE


class CREATOR_OT_check_configuration(bpy.types.Operator):
    """Check configured paths without executing Python or changing project files."""

    bl_idname = "creator.check_configuration"
    bl_label = "Check configured paths"
    bl_description = (
        "Check file locations only; this does not execute or validate the core environment"
    )

    @classmethod
    def poll(cls, context):
        return ADDON_MODULE in context.preferences.addons

    def execute(self, context):
        preferences = context.preferences.addons[ADDON_MODULE].preferences
        issues = validate_configuration(preferences.core_python, preferences.project_root)
        if issues:
            message = " | ".join(issues)
            if context.scene is not None:
                context.scene.creator_recon.configuration_status = message
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        message = "Paths exist; execution and installed packages have not been checked."
        if context.scene is not None:
            context.scene.creator_recon.configuration_status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}
