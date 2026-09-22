"""Analytic textured planes for detector/track controls, separate from real scenes."""
from __future__ import annotations

import numpy as np


def control_camera(angle, size_wh):
    theta = np.radians(angle)
    center = np.array([5.8 * np.sin(theta), -5.8 * np.cos(theta), .12])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    w, h = size_wh
    return dict(K_index=[[w * 49 / 36, 0, (w - 1) / 2], [0, w * 49 / 36, (h - 1) / 2], [0, 0, 1]],
                world_to_camera_cv=np.c_[rotation, -rotation @ center], size_wh=size_wh)


def project(points, camera):
    e, k = np.asarray(camera["world_to_camera_cv"]), np.asarray(camera["K_index"])
    xyz = np.asarray(points) @ e[:, :3].T + e[:, 3]
    pixel = xyz @ k.T
    return pixel[:, :2] / pixel[:, 2, None], xyz[:, 2]


def cast_planes(xy, camera, planes):
    e, k = np.asarray(camera["world_to_camera_cv"]), np.asarray(camera["K_index"])
    center = -e[:, :3].T @ e[:, 3]
    ray = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(k).T @ e[:, :3]
    depth, label = np.full(len(xy), np.inf), np.full(len(xy), -1, int)
    for i, p in enumerate(planes):
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (p["y"] - center[1]) / ray[:, 1]
            world = center + distance[:, None] * ray
        valid = np.isfinite(distance) & (distance > 0) & (distance < depth)
        valid &= (world[:, 0] >= p["x0"]) & (world[:, 0] <= p["x1"]) & (world[:, 2] >= p["z0"]) & (world[:, 2] <= p["z1"])
        depth[valid], label[valid] = distance[valid], i
    world = np.full((len(xy), 3), np.nan)
    hit = label >= 0
    world[hit] = center + depth[hit, None] * ray[hit]
    return world, depth, label


def texture(seed, repeat, size=1024):
    import cv2
    rng = np.random.default_rng(seed)
    # Independent values per cell give a positive correspondence control. This
    # is a deliberately helpful artificial target, not an improved rod method.
    coarse = rng.integers(35, 225, (8 if repeat else 64, 8 if repeat else 64), dtype=np.uint8)
    if repeat:
        coarse = np.tile(coarse, (8, 8))
    pixels = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(pixels, (3, 3), .6)


def render_planes(camera, planes, repeat, seed):
    import cv2
    width, height = camera["size_wh"]
    image = np.full((height, width, 3), 28, np.uint8)
    yy, xx = np.mgrid[:height, :width]
    _, _, labels = cast_planes(np.c_[xx.ravel(), yy.ravel()], camera, planes)
    for i, p in enumerate(planes):
        source = texture(seed + i * 37, repeat)
        world = [[p["x0"], p["y"], p["z1"]], [p["x1"], p["y"], p["z1"]],
                 [p["x1"], p["y"], p["z0"]], [p["x0"], p["y"], p["z0"]]]
        dst, _ = project(world, camera)
        last = source.shape[0] - 1
        transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [last, 0], [last, last], [0, last]]), dst.astype(np.float32))
        warped = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR, borderValue=28)
        mask = labels.reshape(height, width) == i
        image[mask] = warped[mask, None]
    return image


def physical_pair(first_xy, second_xy, first, second, planes):
    """Both directions must hit and reproject within 2px; >5px is wrong."""
    observations = []
    for a, b, ca, cb in ((first_xy, second_xy, first, second), (second_xy, first_xy, second, first)):
        world, _, label = cast_planes(a, ca, planes)
        uv, depth = project(world, cb)
        _, hit_depth, hit_label = cast_planes(uv, cb, planes)
        w, h = cb["size_wh"]
        visible = (label >= 0) & (hit_label >= 0) & np.isfinite(uv).all(axis=1) & (depth > 0)
        visible &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (abs(hit_depth - depth) <= .002)
        observations.append((visible, np.linalg.norm(uv - b, axis=1)))
    (av, ae), (bv, be) = observations
    correct = av & bv & (ae <= 2) & (be <= 2)
    wrong = (av & (ae > 5)) | (bv & (be > 5))
    return np.where(correct, "correct", np.where(wrong, "wrong", "indeterminate"))
