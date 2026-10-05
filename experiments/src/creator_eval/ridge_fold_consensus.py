"""Finite-segment agreement, without averaging, extending, or completing gaps."""

import numpy as np


def endpoint_distance(first, second):
    direct = np.linalg.norm(first-second, axis=1).max()
    reverse = np.linalg.norm(first-second[::-1], axis=1).max()
    return float(min(direct, reverse))


def agreed_segments(first, second, existing, config):
    arrays = [np.asarray(value, float) for value in (first, second, existing)]
    if any(value.ndim != 3 or value.shape[1:] != (2, 3) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Finite Nx2x3 segment arrays required")
    tolerance = config["voxel_size"]*config["merge_distance_voxels"]
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite endpoint agreement tolerance required")
    accepted = []
    for segment in arrays[0]:
        if not any(endpoint_distance(segment, other) <= tolerance for other in arrays[1]):
            continue
        if any(endpoint_distance(segment, other) <= tolerance for other in [*arrays[2], *accepted]):
            continue
        accepted.append(segment.copy())
    return np.asarray(accepted, float).reshape(-1, 2, 3)


def overlaps_existing(segment, existing, config):
    """Do not append another finite fit of the same already-emitted member."""
    tolerance = config["voxel_size"]*config["merge_distance_voxels"]
    for other in existing:
        delta = other[1]-other[0]
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        axis = delta/length
        projection = (segment-other[0]) @ axis
        radial = np.linalg.norm(segment-other[0]-projection[:, None]*axis, axis=1)
        overlap = min(length, projection.max())-max(0., projection.min())
        if radial.max() <= tolerance and overlap > .5*min(length, np.ptp(projection)):
            return True
    return False
