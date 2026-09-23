"""Materialize a new point version from explicit depth and camera identities.

This is an integration artifact, not a depth-repair algorithm. Depth samples are
retained exactly; changing cameras alone does not certify their physical accuracy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .point_patch import content_hash, require_hash, validate_frames, write_snapshot


def native_intrinsics(original, source_wh, native_wh):
    source, native = np.asarray(source_wh, float), np.asarray(native_wh, float)
    if source.shape != (2,) or native.shape != (2,) or not np.isfinite([source, native]).all() or np.any(source <= 0) or np.any(native <= 0):
        raise ValueError('Positive source and native raster sizes required')
    sx, sy = native / source
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return resize @ np.asarray(original, float)


def materialize(depths, intrinsics, extrinsics):
    depth, k, e = np.asarray(depths), np.asarray(intrinsics, float), np.asarray(extrinsics, float)
    if depth.ndim != 3 or depth.dtype.kind != 'f' or min(depth.shape) <= 0:
        raise ValueError('Nonempty floating point depth raster stack required')
    if k.shape != (len(depth), 3, 3) or e.shape != (len(depth), 3, 4):
        raise ValueError('One intrinsic and world-to-camera matrix per depth frame')
    if not np.isfinite(k).all() or not np.isfinite(e).all() or np.any(k[:, [0, 1], [0, 1]] <= 0):
        raise ValueError('Finite pinhole cameras with positive focal length required')
    if not np.allclose(k[:, 2], [0, 0, 1]) or not np.allclose(e[:, :, :3] @ e[:, :, :3].transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(e[:, :, :3]), 1, atol=1e-4):
        raise ValueError('Rigid camera rotations and pinhole K required')
    _, height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    points, ids = [], []
    for view, (z, calibration, camera) in enumerate(zip(depth, k, e)):
        valid = np.isfinite(z) & (z > 0)
        inverse = np.linalg.inv(np.vstack([camera, [0, 0, 0, 1]]))
        xyz = (pixels[valid] @ np.linalg.inv(calibration).T) * z[valid, None]
        points.append(xyz @ inverse[:3, :3].T + inverse[:3, 3])
        ids.append(np.c_[np.full(valid.sum(), view), yy[valid], xx[valid]].astype('<u4'))
    return np.concatenate(points), np.concatenate(ids)


def write_camera_snapshot(destination, depths, intrinsics, extrinsics, *, frames, prediction_sha256, camera_source_sha256):
    validate_frames(frames)
    for value in (prediction_sha256, camera_source_sha256):
        require_hash(value)
    depth = np.ascontiguousarray(depths)
    points, ids = materialize(depth, intrinsics, extrinsics)
    if len(frames) != len(depth) or any(f['prediction_size_wh'] != [depth.shape[2], depth.shape[1]] for f in frames):
        raise ValueError('Frame identities must match native depth dimensions')
    depth_hash = hashlib.sha256(depth.dtype.str.encode() + str(depth.shape).encode() + depth.tobytes()).hexdigest()
    source = dict(version='1', source_prediction_sha256=prediction_sha256, camera_source_sha256=camera_source_sha256,
        depth_content_sha256=depth_hash, depth_shape=list(depth.shape), depth_dtype=depth.dtype.str,
        native_intrinsics=np.asarray(intrinsics, float).tolist(), world_to_camera_cv=np.asarray(extrinsics, float).tolist(),
        frames=frames, depth_policy='unchanged finite positive source depth samples; no confidence pruning or metric depth correction')
    identity = content_hash(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'camera-depth-source.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(geometry_source_id=identity, source=source), stream, indent=2, allow_nan=False)
        stream.write('\n')
    snapshot_id = write_snapshot(destination / 'base', points, ids, frames=frames,
        world_frame_id='camera-depth:' + identity, length_unit='reconstruction_unit', source_prediction_sha256=prediction_sha256,
        point_policy='New reprojection version; explicit native camera/depth identity ' + identity + '; unchanged depths are not certified geometry')
    return dict(snapshot_id=snapshot_id, geometry_source_id=identity, point_count=len(points), depth_content_sha256=depth_hash)
