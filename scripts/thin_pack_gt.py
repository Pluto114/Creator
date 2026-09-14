"""Geometry-only GT and explicit DA3 image sampling for the paired pilot dataset.

This independent experiment helper does not implement the formal reconstruction DTOs.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resize_plan(width, height, process_res):
    scale = process_res / max(width, height)
    middle = [round(width * scale), round(height * scale)]

    def nearest(n):
        down = n // 14 * 14
        return max(1, down + 14 if (down + 14 - n) <= n - down else down)

    target = [nearest(n) for n in middle]
    if any(b > a for a, b in zip([width, height], middle)) or any(
        b > a for a, b in zip(middle, target)
    ):
        raise ValueError("Only the verified two-stage INTER_AREA downsampling is supported")
    sx, sy = target[0] / width, target[1] / height
    return {
        "method": "upper_bound_resize",
        "source_size_wh": [width, height],
        "intermediate_size_wh": middle,
        "target_size_wh": target,
        "steps": [
            {
                "input_size_wh": [width, height],
                "output_size_wh": middle,
                "interpolation": "cv2.INTER_AREA",
            },
            {"input_size_wh": middle, "output_size_wh": target, "interpolation": "cv2.INTER_AREA"},
        ],
        "source_index_to_target_index": [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]],
        "source_edge_to_target_edge": [[sx, 0, 0], [0, sy, 0], [0, 0, 1]],
        "sampling": "half_pixel; target edge center maps to ((j+.5)/sx,(i+.5)/sy)",
        "upstream_intrinsics_note": "DA3 scales K rows without a half-pixel correction. Feed K_edge only in an explicitly labeled oracle adapter; GT stores both edge and index K. Its arange-based unprojection needs K_index.",
    }


def preprocess_rgb(rgb, plan):
    out = rgb
    for step in plan["steps"]:
        if list(out.shape[1::-1]) != step["output_size_wh"]:
            out = cv2.resize(out, tuple(step["output_size_wh"]), interpolation=cv2.INTER_AREA)
    return out


def support_bounds(source, target):
    """All positive-area contributing source cells, using exact integer arithmetic."""
    j = np.arange(target, dtype=np.int64)
    low = j * source // target
    high = ((j + 1) * source + target - 1) // target - 1
    return low, high


def reduce_support(array, size, minimum):
    out = array
    op = np.minimum if minimum else np.maximum
    for axis, count in ((1, size[0]), (0, size[1])):
        low, high = support_bounds(out.shape[axis], count)
        value = np.take(out, low, axis=axis)
        for offset in range(1, int(np.max(high - low)) + 1):
            value = op(value, np.take(out, np.minimum(low + offset, high), axis=axis))
        out = value
    return out


def pure_resize_support(native_ids, native_safe, plan):
    lo = np.where(native_safe, native_ids, 0).astype(np.uint32)
    hi = lo.copy()
    for step in plan["steps"]:
        lo = reduce_support(lo, step["output_size_wh"], True)
        hi = reduce_support(hi, step["output_size_wh"], False)
    return (lo == hi) & (lo != 0), lo


class MeshRays:
    def __init__(self, mesh_path):
        import open3d as o3d

        mesh = np.load(mesh_path, allow_pickle=False)
        self.vertices = mesh["vertices"]
        self.triangles = mesh["triangles"]
        self.labels = mesh["surface_ids"]
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=8)
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.triangles))

    def cast(self, camera, uv):
        uv = np.asarray(uv)
        inverse = np.linalg.inv(np.array(camera["world_to_camera_cv"]))
        direction = (
            np.c_[uv.reshape(-1, 2), np.ones(uv.size // 2)]
            @ np.linalg.inv(camera["K_edge"]).T
            @ inverse[:3, :3].T
        )
        near = camera["clip_start"]
        origins = inverse[:3, 3] + direction * near
        rays = np.concatenate([origins, direction], axis=1).astype(np.float32)
        hits = self.scene.cast_rays(self.o3d.core.Tensor(rays), nthreads=8)
        z = hits["t_hit"].numpy() + near
        primitive = hits["primitive_ids"].numpy()
        valid = np.isfinite(z) & (z <= camera["clip_end"])
        sid = np.zeros(len(z), np.uint32)
        sid[valid] = self.labels[primitive[valid]]
        z[~valid] = np.nan
        ranges = z * np.linalg.norm(direction, axis=1)
        return z.astype(np.float32), sid, ranges.astype(np.float32)

    def raster(self, camera, samples=4):
        w, h = camera["size_wh"]
        depth = np.empty((h, w), np.float32)
        distance = np.empty_like(depth)
        depth_min = np.empty_like(depth)
        depth_max = np.empty_like(depth)
        surface = np.empty((h, w), np.uint32)
        counts = np.zeros((7, h, w), np.uint8)
        mixed = np.zeros((h, w), bool)
        # Tiles bound peak memory while preserving exact ray positions.
        for top in range(0, h, 48):
            end = min(h, top + 48)
            y, x = np.mgrid[top:end, :w]
            uv = np.stack([x + 0.5, y + 0.5], axis=-1)
            z, sid, ranges = self.cast(camera, uv)
            shape = x.shape
            center = sid.reshape(shape)
            depth[top:end] = z.reshape(shape)
            distance[top:end] = ranges.reshape(shape)
            surface[top:end] = center
            depth_min[top:end] = z.reshape(shape)
            depth_max[top:end] = z.reshape(shape)
            for sy in range(samples):
                for sx in range(samples):
                    uv = np.stack([x + (sx + 0.5) / samples, y + (sy + 0.5) / samples], axis=-1)
                    sample_z, sid, _ = self.cast(camera, uv)
                    depth_min[top:end] = np.fmin(depth_min[top:end], sample_z.reshape(shape))
                    depth_max[top:end] = np.fmax(depth_max[top:end], sample_z.reshape(shape))
                    sid = sid.reshape(shape)
                    mixed[top:end] |= sid != center
                    for segment in range(1, 8):
                        counts[segment - 1, top:end] += (sid == segment).astype(np.uint8)
        rod = np.where((surface >= 1) & (surface <= 7), np.minimum(surface, 6), 0).astype(np.uint32)
        return {
            "depth_z": depth,
            "ray_distance": distance,
            "pixel_sample_depth_min": depth_min,
            "pixel_sample_depth_max": depth_max,
            "surface_id": surface,
            "rod_id": rod,
            "rod_visible_center_masks": np.stack([rod == i for i in range(1, 7)]),
            "segment_coverage_counts": counts,
            "sample_mixed": mixed,
            "hit_valid": surface != 0,
        }


def safe_native(arrays, guard):
    ids = arrays["surface_id"].astype(np.float32)
    kernel = np.ones((2 * guard + 1, 2 * guard + 1), np.uint8)
    stable = cv2.erode(ids, kernel) == cv2.dilate(ids, kernel)
    valid_samples = arrays["hit_valid"] & ~arrays["sample_mixed"]
    return stable & cv2.erode(
        valid_samples.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)


def check_blender_rays(rays, camera, probes, tolerance):
    z, ids, _ = rays.cast(camera, np.array([p["uv_edge"] for p in probes]))
    expected = np.array([p["surface_id"] for p in probes])
    target = np.array([p["z"] if p["z"] is not None else np.nan for p in probes])
    valid = expected != 0
    error = float(np.max(np.abs(z[valid] - target[valid]))) if valid.any() else 0.0
    mismatch = int(np.count_nonzero(ids != expected))
    if error > tolerance or mismatch:
        raise AssertionError(f"Blender ray disagreement: {error=} {mismatch=}")
    return {
        "samples": len(z),
        "surface_id_mismatches": mismatch,
        "max_z_error_m": error,
        "rod_samples": int(np.count_nonzero((expected >= 1) & (expected <= 7))),
        "no_hit_samples": int(np.count_nonzero(~valid)),
    }


def read_blender_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.float32:
        raise ValueError(f"Cannot read float32 EXR: {path}")
    if raw.ndim == 3:
        assert np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 1], raw[..., 2])
        raw = raw[..., 0]
    return raw


def depth_hypotheses(raw, arrays, safe):
    selected = safe & np.isfinite(raw) & (raw < 1e9)
    if not np.any(selected):
        raise AssertionError("No interior depth calibration samples")
    result = {"samples": int(selected.sum())}
    for name, key in [("camera_z", "depth_z"), ("ray_distance", "ray_distance")]:
        delta = np.abs(raw[selected] - arrays[key][selected])
        result[name] = {
            "max_abs_m": float(delta.max()),
            "mean_abs_m": float(delta.mean()),
            "p99_abs_m": float(np.quantile(delta, 0.99)),
        }
    return result


def verify_projection_roundtrip(camera, depth, stride=31):
    h, w = depth.shape
    y, x = np.mgrid[:h:stride, :w:stride]
    d = depth[y, x]
    valid = np.isfinite(d)
    uv = np.c_[x[valid] + 0.5, y[valid] + 0.5, np.ones(valid.sum())]
    xyz = (uv @ np.linalg.inv(camera["K_edge"]).T) * d[valid, None]
    inverse = np.linalg.inv(camera["world_to_camera_cv"])
    world = np.c_[xyz, np.ones(len(xyz))] @ inverse.T
    back = world @ np.array(camera["world_to_camera_cv"]).T
    projected = back[:, :3] @ np.array(camera["K_edge"]).T
    err = float(np.max(np.abs(projected[:, :2] / projected[:, 2:] - uv[:, :2])))
    assert err < 1e-6
    return err


def save_arrays(folder, arrays):
    folder.mkdir(exist_ok=True)
    specs = {}
    for key, array in arrays.items():
        path = folder / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        specs[key] = {
            "path": path.name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": sha256(path),
            "byte_size": path.stat().st_size,
            "unit": "meter"
            if key
            in ("depth_z", "ray_distance", "pixel_sample_depth_min", "pixel_sample_depth_max")
            else "dimensionless",
            "sampling": "pixel_center"
            if key
            in (
                "depth_z",
                "ray_distance",
                "surface_id",
                "rod_id",
                "rod_visible_center_masks",
                "hit_valid",
            )
            else "see DATASET.md",
        }
    write_json(folder / "arrays.json", specs)


def stats(arrays):
    n = arrays["surface_id"].size
    counts = arrays["segment_coverage_counts"]
    return {
        "pixel_count": n,
        "no_hit_pixels": int((~arrays["hit_valid"]).sum()),
        "sample_mixed_pixels": int(arrays["sample_mixed"].sum()),
        "strict_depth_eval_pixels": int(arrays["strict_depth_eval_valid"].sum()),
        "visible_center_pixels_per_segment": {
            str(i): int((arrays["surface_id"] == i).sum()) for i in range(1, 8)
        },
        "strict_depth_eval_pixels_per_segment": {
            str(i): int(((arrays["surface_id"] == i) & arrays["strict_depth_eval_valid"]).sum())
            for i in range(1, 8)
        },
        "subpixel_rod_pixels_without_center_hit": int(
            ((counts.sum(axis=0) > 0) & (arrays["rod_id"] == 0)).sum()
        ),
    }
