"""Camera/depth convention checks for fresh fixture-conditioned predictions."""

from __future__ import annotations

import copy

import numpy as np


def camera_inputs(cameras):
    if len(cameras) != 5 or len({c["view_id"] for c in cameras}) != 5:
        raise ValueError("Five ordered unique cameras required")
    if any(c["state"] != "validated" for c in cameras):
        raise ValueError("Every fixture camera must validate before conditioning depth")
    edge = np.asarray([c["K_index"] for c in cameras], float).copy()
    extrinsics = np.asarray([c["world_to_camera_cv"] for c in cameras], float)
    if edge.shape != (5, 3, 3) or extrinsics.shape != (5, 3, 4):
        raise ValueError("Explicit pinhole K and 3x4 world-to-camera matrices required")
    if not np.isfinite(edge).all() or not np.isfinite(extrinsics).all():
        raise ValueError("Finite cameras required")
    if not np.allclose(edge[:, 2], [0, 0, 1]) or np.any(edge[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Positive focal lengths and pinhole homogeneous row required")
    if not np.allclose(edge[:, [0, 1], [1, 0]], 0, atol=1e-10):
        raise ValueError("Skew is not encoded by the upstream model")
    rotations = extrinsics[:, :, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
        raise ValueError("Proper rigid cameras required")
    edge[:, :2, 2] += .5
    full = np.repeat(np.eye(4)[None], 5, axis=0)
    full[:, :3] = extrinsics
    return edge, full


def exported_cameras(native, cameras, source_wh):
    """Validate exported identity, then use those exact cameras for points/rods."""
    edge, expected_e = camera_inputs(cameras)
    depth = np.asarray(native["depth"])
    k, e = np.asarray(native["intrinsics"], float), np.asarray(native["extrinsics"], float)
    if depth.ndim != 3 or depth.shape[0] != 5 or min(depth.shape) <= 0 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Five finite positive depth maps required")
    if k.shape != (5, 3, 3) or e.shape != (5, 3, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
        raise ValueError("One finite exported camera per depth frame required")
    size = np.asarray(source_wh, float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Positive source image dimensions required")
    h, w = depth.shape[1:]
    scale = np.array([w, h]) / size
    expected_k = np.diag([*scale, 1.]) @ edge
    if not np.allclose(k, expected_k, atol=2e-4, rtol=0) or not np.allclose(e, expected_e[:, :3], atol=1e-6, rtol=0):
        raise ValueError("Prediction does not belong to the supplied fixture cameras")
    index = k.copy()
    index[:, :2, 2] -= .5
    lifted_edge = np.diag([*(1/scale), 1.]) @ k
    lifted = lifted_edge.copy()
    lifted[:, :2, 2] -= .5
    output = copy.deepcopy(cameras)
    for row, calibration, extrinsic in zip(output, lifted, e):
        row.update(K_index=calibration.tolist(), world_to_camera_cv=extrinsic.tolist(),
                   camera_source="exact_conditioned_prediction_export_lifted_to_original_RGB")
    return index, e, output


def source_roundtrip(points, ids, depth, k, extrinsics, per_view=1024):
    maximum_pixel, maximum_depth, checked = 0., 0., 0
    for view in range(len(depth)):
        available = np.flatnonzero(ids[:, 0] == view)
        if not len(available):
            raise ValueError("Snapshot lost an entire source frame")
        selected = available[np.linspace(0, len(available)-1, min(per_view, len(available)), dtype=int)]
        source = ids[selected]
        camera = points[selected] @ extrinsics[view, :, :3].T + extrinsics[view, :, 3]
        pixel = camera @ k[view].T
        maximum_pixel = max(maximum_pixel, float(np.max(np.abs(pixel[:, :2]/pixel[:, 2, None]-source[:, [2, 1]]))))
        maximum_depth = max(maximum_depth, float(np.max(np.abs(camera[:, 2]-depth[view, source[:, 1], source[:, 2]]))))
        checked += len(selected)
    if maximum_pixel > 1e-6 or maximum_depth > 1e-6:
        raise ValueError("Source-pixel roundtrip failed")
    return dict(pixel_checks=checked, maximum_pixel_error=maximum_pixel, maximum_depth_error=maximum_depth)
