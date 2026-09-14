"""Display-only transforms and calibrated point splats; raw geometry stays unchanged."""
from __future__ import annotations

import numpy as np


def align_for_display(points: np.ndarray, extrinsics: np.ndarray):
    """CV world -> first-camera glTF frame. Rigid transform, not a geometry repair."""
    first = np.eye(4)
    first[:3] = extrinsics[0]
    transform = np.diag([1., -1., -1., 1.]) @ first
    displayed = points @ transform[:3, :3].T + transform[:3, 3]
    center = np.median(displayed, axis=0)
    transform[:3, 3] -= center
    displayed -= center
    cameras = []
    for extrinsic in extrinsics:
        matrix = np.eye(4)
        matrix[:3] = extrinsic
        camera = np.linalg.inv(matrix)[:3, 3]
        cameras.append(transform[:3, :3] @ camera + transform[:3, 3])
    return displayed, np.asarray(cameras), transform


def camera_raster(points, colors, intrinsic, extrinsic, depth, yaw_degrees=0.0):
    """Render with camera 1's K; optional physical orbit about its central depth ray."""
    height, width = depth.shape
    target = np.linalg.inv(intrinsic) @ np.array([width / 2, height / 2, 1.])
    target *= float(depth[height // 2, width // 2])
    angle = np.deg2rad(yaw_degrees)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    eye = target - rotation @ target
    first_camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    camera_points = (first_camera - eye) @ rotation
    projection = camera_points @ intrinsic.T
    front = projection[:, 2] > 1e-8
    camera_points, projection, colors = camera_points[front], projection[front], colors[front]
    uv = np.rint(projection[:, :2] / projection[:, 2, None]).astype(np.int64)
    pixels, depths, rgb = [], [], []
    # A 3x3 pixel splat makes the discrete samples visible; no surface is inferred.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = uv[:, 0] + dx, uv[:, 1] + dy
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            pixels.append(y[inside] * width + x[inside])
            depths.append(camera_points[inside, 2])
            rgb.append(colors[inside])
    pixel, z, color = np.concatenate(pixels), np.concatenate(depths), np.concatenate(rgb)
    order = np.lexsort((z, pixel))
    _, unique = np.unique(pixel[order], return_index=True)
    selected = order[unique]
    image = np.full((height * width, 3), (12, 18, 32), dtype=np.uint8)
    image[pixel[selected]] = color[selected]
    return image.reshape(height, width, 3)