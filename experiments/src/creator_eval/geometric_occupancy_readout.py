"""Read the actual occupancy of the SAME foreground-selected 3D axis.

RGB selects admissible actual coordinates and the axis. Its uncertain raster
rows do not clip that axis a second time. Output is exactly the selected
actual-point +/-1v interval union, bounded by observed projections. Thus a
positive geometric gap above 2v remains a gap; smaller image-only holes are
not claimed to be resolved by this geometric readout.
"""

import numpy as np

from . import occupied_axis_readout as previous


def readout(values, voxel_size, origin, context):
    result = previous.readout(values, voxel_size, origin, context)
    result.update(output_support_policy="selected_actual_3d_occupancy_without_second_rgb_clip",
        rgb_role="coordinate_membership_and_unchanged_axis_selection",
        minimum_resolvable_geometric_gap_voxels=2.,
        continuous_rgb_support_required=False, image_only_gaps_preserved=False,
        image_constructed_geometry=False, axis_refitted=False,
        previous_rgb_clipped_components=result["components"],
        previous_rgb_clipped_length_m=float(np.linalg.norm(
            result["segments"][:, 1]-result["segments"][:, 0], axis=1).sum()))
    selected = result["selected_candidate"]
    if result["state"] != "complete" or selected is None:
        return result
    graph = result["readouts"][selected]["graph"]
    segments = graph["segments"].copy()
    length = float(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1).sum())
    result.update(segments=segments, components=len(segments),
        actual_occupancy_length_m=length,
        retained_geometry_previously_rgb_clipped_m=length-result["previous_rgb_clipped_length_m"])
    return result
