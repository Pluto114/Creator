"""Evaluation-only axis, true mesh silhouette, native visibility and RGB row audit.

Nothing in this file is imported by a detector or fitter. The NumPy helpers also
support synthetic known-answer tests without Blender, Open3D, or image packages.
"""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project_world(points, camera):
    points = np.asarray(points, dtype=float)
    transform = np.asarray(camera["world_to_camera_cv"], dtype=float)
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    if np.any(camera_points[..., 2] <= camera["clip_start"]) or np.any(
        camera_points[..., 2] >= camera["clip_end"]
    ):
        raise ValueError("Audit requires complete rod triangles inside the depth clip range")
    image = camera_points @ np.asarray(camera["K_index"], dtype=float).T
    return image[..., :2] / image[..., 2, None]


def axis_at_row(endpoints, camera, y_index):
    """Finite 3D axis intersection with a native RGB integer-index scanline."""
    a, b = np.asarray(endpoints, dtype=float)
    matrix = np.asarray(camera["K_index"]) @ np.asarray(camera["world_to_camera_cv"])[:3]
    equation = matrix[1] - y_index * matrix[2]
    denominator = equation[:3] @ (b - a)
    if abs(denominator) < 1e-12:
        if abs(equation @ np.r_[a, 1.0]) < 1e-10:
            raise ValueError("Horizontal projected axis has no unique point at this row")
        return None
    fraction = -float(equation @ np.r_[a, 1.0]) / denominator
    if not 0 <= fraction <= 1:
        return None
    world = a + fraction * (b - a)
    uv = project_world(world, camera)
    if abs(uv[1] - y_index) > 1e-8:
        raise AssertionError("Axis scanline projection did not round-trip")
    return {
        "u_index": float(uv[0]),
        "y_index": float(uv[1]),
        "world_fraction": fraction,
        "world_point": world.tolist(),
    }


def merge_intervals(intervals, tolerance=1e-9):
    merged = []
    for low, high in sorted(intervals):
        if not merged or low > merged[-1][1] + tolerance:
            merged.append([float(low), float(high)])
        else:
            merged[-1][1] = max(merged[-1][1], float(high))
    return merged


def mesh_scanline_intervals(vertices, triangles, camera, y_index):
    """Continuous unoccluded silhouette of the actual finite triangle mesh.

    Intersect every projected triangle with the horizontal scanline, then union
    the finite intervals. This is geometry, not a threshold on rendered light.
    Occlusion by other objects is handled separately by first-hit ray labels.
    """
    triangles = np.asarray(triangles, dtype=int)
    if not len(triangles):
        return []
    projected = project_world(np.asarray(vertices)[triangles], camera)
    intervals = []
    for triangle in projected:
        if y_index < triangle[:, 1].min() - 1e-10 or y_index > triangle[:, 1].max() + 1e-10:
            continue
        intersections = []
        for first, last in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[first], triangle[last]
            if abs(b[1] - a[1]) < 1e-12:
                if abs(y_index - a[1]) < 1e-10:
                    intersections.extend([float(a[0]), float(b[0])])
                continue
            fraction = (y_index - a[1]) / (b[1] - a[1])
            if -1e-10 <= fraction <= 1 + 1e-10:
                intersections.append(float(a[0] + np.clip(fraction, 0, 1) * (b[0] - a[0])))
        if intersections:
            intervals.append([min(intersections), max(intersections)])
    return merge_intervals(intervals)


def native_pixel_runs(mask_row, rod_id):
    """Inclusive integer center-hit ranges; their outer cells are not exact outlines."""
    hits = np.flatnonzero(np.asarray(mask_row) == rod_id)
    if not len(hits):
        return []
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
    return [
        {
            "u_min_index": int(group[0]),
            "u_max_index": int(group[-1]),
            "center_midpoint_index": float((group[0] + group[-1]) / 2),
            "pixel_count": len(group),
            "pixel_cell_footprint": [float(group[0] - 0.5), float(group[-1] + 0.5)],
        }
        for group in groups
    ]


def audit_segment_axis(vertices, triangles, endpoints):
    """Check whether the exported local-Z axis is centered in both actual mesh caps."""
    selected = np.unique(np.asarray(triangles, int))
    points = np.asarray(vertices, float)[selected]
    a, b = np.asarray(endpoints, float)
    direction = (b - a) / np.linalg.norm(b - a)
    axial = (points - a) @ direction
    radial = points - a - axial[:, None] * direction
    cap_masks = [np.abs(axial - end) < 1e-6 for end in (axial.min(), axial.max())]
    caps = [points[mask] for mask in cap_masks]
    radial_cap = radial[cap_masks[0]]
    eigenvalues = np.linalg.eigvalsh(radial_cap.T @ radial_cap / len(radial_cap))
    central_symmetry = np.min(
        np.linalg.norm(radial_cap[:, None] + radial_cap[None], axis=-1), axis=1
    )
    return {
        "vertices": len(points),
        "triangles": len(triangles),
        "cap_vertex_counts": [len(cap) for cap in caps],
        "cap_center_to_axis_endpoint_m": [
            float(np.linalg.norm(cap.mean(0) - endpoint)) for cap, endpoint in zip(caps, (a, b))
        ],
        "radial_centroid_offset_m": float(np.linalg.norm(radial.mean(0))),
        "opposite_vertex_symmetry_max_residual_m": float(central_symmetry.max()),
        "radial_principal_rms_ratio": float(np.sqrt(eigenvalues[-1] / eigenvalues[-2])),
        "radius_min_m": float(np.linalg.norm(radial, axis=1).min()),
        "radius_max_m": float(np.linalg.norm(radial, axis=1).max()),
        "scope": "actual_exported_mesh_caps_and_radial_symmetry; no_appearance_assumption",
    }


def ray_visible_intervals(rays, camera, y_index, surface_id, projected_intervals, spacing=1 / 64):
    """Dense first-hit visibility with boundary bisection, not a claim of exact raster AA."""
    intervals = []
    probes = 0
    for low, high in projected_intervals:
        samples = np.linspace(low - 1, high + 1, int(np.ceil((high - low + 2) / spacing)) + 1)
        uv = np.c_[samples + 0.5, np.full(len(samples), y_index + 0.5)]
        _, ids, _ = rays.cast(camera, uv)
        visible = ids == surface_id
        probes += len(samples)
        transitions = np.flatnonzero(visible[1:] != visible[:-1])
        bounds = []
        for index in transitions:
            left, right, initial = samples[index], samples[index + 1], visible[index]
            for _ in range(16):
                middle = (left + right) / 2
                _, label, _ = rays.cast(camera, [[middle + 0.5, y_index + 0.5]])
                probes += 1
                if bool(label[0] == surface_id) == bool(initial):
                    left = middle
                else:
                    right = middle
            bounds.append((left + right) / 2)
        if visible[0] or visible[-1] or len(bounds) % 2:
            raise ValueError("Visible interval extends beyond the projected segment silhouette")
        intervals.extend([[a, b] for a, b in zip(bounds[::2], bounds[1::2])])
    return {
        "intervals_index": intervals,
        "ray_probes": probes,
        "initial_sampling_max_spacing_px": spacing,
        "boundary_note": "float32_mesh_rays_with_bisection; sub-grid_visibility_islands_below_1/64px_may_be_missed",
    }


def audit_scanline(
    mesh, geometry, camera, native_rod_row, y_index, rod_id, *, rays=None, rgb_row=None
):
    """Evaluation interface for one row: integer GT spans, axis, mesh outline, RGB."""
    entries = []
    for obj in geometry["objects"]:
        if obj["rod_id"] != rod_id or obj["duplicate_geometry_of"] is not None:
            continue
        sid = obj["surface_id"]
        faces = mesh["triangles"][mesh["surface_ids"] == sid]
        intervals = mesh_scanline_intervals(mesh["vertices"], faces, camera, y_index)
        axis = axis_at_row(obj["world_centerline_endpoints"], camera, y_index)
        intervals_report = [
            {
                "u_left_index": a,
                "u_right_index": b,
                "width_px": b - a,
                "midpoint_index": (a + b) / 2,
                "midpoint_minus_physical_axis_px": (a + b) / 2 - axis["u_index"] if axis else None,
            }
            for a, b in intervals
        ]
        entry = {
            "surface_id": sid,
            "object": obj["object"],
            "physical_axis": axis,
            "unoccluded_mesh_silhouette": intervals_report,
        }
        if rays is not None:
            entry["first_hit_visibility"] = ray_visible_intervals(
                rays, camera, y_index, sid, intervals
            )
        if rgb_row is not None and intervals:
            left = max(0, int(np.floor(intervals[0][0])) - 12)
            right = min(len(rgb_row), int(np.ceil(intervals[-1][1])) + 13)
            x = np.arange(left, right)
            luminance = np.asarray(rgb_row)[left:right].astype(float) @ [0.2126, 0.7152, 0.0722]
            delta = np.diff(luminance)
            ranked = np.argsort(-np.abs(delta), kind="stable")[:6]
            inside = (x >= intervals[0][0]) & (x <= intervals[-1][1])
            brightest = int(x[inside][np.argmax(luminance[inside])]) if inside.any() else None
            entry["rgb_row_profile"] = {
                "u_index": x.tolist(),
                "display_luma_255": luminance.tolist(),
                "largest_absolute_gradients": [
                    {
                        "u_index_between_centers": float(x[index] + 0.5),
                        "signed_delta_luma": float(delta[index]),
                    }
                    for index in ranked
                ],
                "brightest_inside_mesh_u_index": brightest,
                "brightest_minus_physical_axis_px": brightest - axis["u_index"]
                if brightest is not None and axis
                else None,
                "scope": "GT_selected_display_RGB_profile_only_not_detector_input; AgX_png_luma_is_not_linear_irradiance",
            }
        entries.append(entry)
    return {
        "y_index": y_index,
        "rod_id": rod_id,
        "native_center_hit_runs": native_pixel_runs(native_rod_row, rod_id),
        "segments": entries,
        "pixel_convention": "RGB integer index y; rays use edge coordinates (x+0.5,y+0.5)",
    }


def main():
    from environment_paths import require_project_environment
    from PIL import Image
    from thin_pack_gt import MeshRays, read_json, sha256, write_json

    require_project_environment(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", default="rod-geometry-audit-20260917")
    parser.add_argument(
        "--share-json", type=Path, help="Optional compact report for the repository"
    )
    args = parser.parse_args()
    if not args.output_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in args.output_id
    ):
        parser.error("Invalid output ID")
    producers = [
        Path(__file__),
        ROOT / "scripts/thin_pack_gt.py",
        ROOT / "scripts/blender_export_thin_pack.py",
    ]
    producer_hashes = {path: sha256(path) for path in producers}
    source = ROOT / "data/eval_gt/thin-pack-v2-20260914r1"
    status = read_json(source / "status.json")
    if (
        status["state"] != "complete"
        or sha256(source / "artifact_hashes.json") != status["artifact_hashes_sha256"]
    ):
        raise ValueError("Original GT identity gate failed")
    hashes = read_json(source / "artifact_hashes.json")
    inputs = ROOT / "data/inputs/thin-pack-v2-20260914r1"
    input_manifest = read_json(inputs / "manifest.json")
    consumed = {}

    def checked(relative):
        path = source / relative
        if path not in consumed:
            digest = sha256(path)
            if digest != hashes[Path(relative).as_posix()]["sha256"]:
                raise ValueError("GT changed: " + str(relative))
            consumed[path] = digest
        return path

    summary = read_json(checked("validation_summary.json"))
    if summary["input_manifest_sha256"] != sha256(inputs / "manifest.json"):
        raise ValueError("Original GT and RGB identities differ")
    cases = ["brick-texture-thinner", "brick-texture-origin", "brick-texture-thicker"]
    rows, shape_checks = [], []
    native_ray_disagreements, native_ray_samples = 0, 0
    for case in cases:
        mesh_path = checked(Path(case) / "mesh.npz")
        mesh = dict(np.load(mesh_path, allow_pickle=False))
        geometry = read_json(checked(Path(case) / "geometry.json"))
        rays = MeshRays(mesh_path)
        for obj in geometry["objects"]:
            if obj["rod_id"] in (5, 6) and obj["duplicate_geometry_of"] is None:
                faces = mesh["triangles"][mesh["surface_ids"] == obj["surface_id"]]
                shape_checks.append(
                    {
                        "case_id": case,
                        "object": obj["object"],
                        "surface_id": obj["surface_id"],
                        **audit_segment_axis(
                            mesh["vertices"], faces, obj["world_centerline_endpoints"]
                        ),
                    }
                )
        group = next(g for g in input_manifest["groups"] if g["case_id"] == case)
        for frame in group["frames"]:
            prefix = Path(case) / frame["frame_id"]
            camera = read_json(checked(prefix / "camera.json"))
            native = np.load(
                checked(prefix / "native/rod_id.npy"), allow_pickle=False, mmap_mode="r"
            )
            rgb_path = inputs / frame["rgb"]
            if sha256(rgb_path) != frame["sha256"]:
                raise ValueError("Original RGB changed")
            consumed[rgb_path] = frame["sha256"]
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            for y in (300, 500, 700):
                x = np.arange(camera["size_wh"][0])
                _, sid, _ = rays.cast(camera, np.c_[x + 0.5, np.full(len(x), y + 0.5)])
                first_hit_rod = np.where((sid >= 1) & (sid <= 7), np.minimum(sid, 6), 0)
                native_ray_disagreements += int(np.count_nonzero(first_hit_rod != native[y]))
                native_ray_samples += len(x)
                for rod in (5, 6):
                    rows.append(
                        {
                            "case_id": case,
                            "frame_id": frame["frame_id"],
                            **audit_scanline(
                                mesh, geometry, camera, native[y], y, rod, rays=rays, rgb_row=rgb[y]
                            ),
                        }
                    )
    if native_ray_disagreements:
        raise AssertionError("Native masks disagree with same mesh/camera center rays")
    output = ROOT / "data/evaluation" / args.output_id
    output.mkdir(exist_ok=False)
    frozen = output / "producer_sources"
    frozen.mkdir()
    for path, digest in producer_hashes.items():
        if sha256(path) != digest:
            raise ValueError("Audit producer changed during evaluation")
        shutil.copy2(path, frozen / path.name)
    statistics = {}
    for case in cases:
        offsets = [
            part["midpoint_minus_physical_axis_px"]
            for row in rows
            if row["case_id"] == case
            for segment in row["segments"]
            for part in segment["unoccluded_mesh_silhouette"]
            if part["midpoint_minus_physical_axis_px"] is not None
        ]
        statistics[case] = {
            "valid_axis_silhouette_rows": len(offsets),
            "signed_min_px": min(offsets),
            "signed_max_px": max(offsets),
            "max_abs_px": max(abs(value) for value in offsets),
        }
    source_config = read_json(ROOT / "configs/thin_pack_export.json")
    if sha256(ROOT / source_config["source_scene"]) != source_config["source_sha256"]:
        raise ValueError("Verified source scene changed")
    inventory_path = ROOT / ".runtime/scene-inspection/20260914T071326Z/scene_inventory.json"
    inventory = read_json(inventory_path)
    consumed[inventory_path] = sha256(inventory_path)
    objects = [
        obj
        for obj in inventory["objects"]
        if obj["name"] in ("Cylinder.005", "Cylinder.006", "Cylinder.008")
    ]
    target_materials = {name for obj in objects for name in obj["materials"]}
    appearance = []
    for material in inventory["materials"]:
        if material["name"] in target_materials:
            node = next(n for n in material["nodes"] if n["type"] == "ShaderNodeBsdfPrincipled")
            appearance.append(
                {
                    "material": material["name"],
                    "principled_inputs": {
                        key: node["inputs"][key]
                        for key in (
                            "Base Color",
                            "Metallic",
                            "Roughness",
                            "IOR",
                            "Specular IOR Level",
                            "Alpha",
                            "Transmission Weight",
                        )
                    },
                }
            )
    result = {
        "state": "complete",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "evaluation_only_geometry_and_RGB_appearance_audit; no_detector_or_candidate_results_read",
        "source_bundle": "thin-pack-v2-20260914r1",
        "source_scene_sha256": source_config["source_sha256"],
        "fixed_rows": [300, 500, 700],
        "rod_ids": [5, 6],
        "cases": cases,
        "shape_checks": shape_checks,
        "axis_vs_mesh_silhouette": statistics,
        "native_ray_samples": native_ray_samples,
        "native_ray_disagreements": native_ray_disagreements,
        "rows": rows,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): digest for path, digest in consumed.items()
        },
        "producer_sha256": sha256(Path(__file__)),
        "frozen_producer_sha256": {path.name: digest for path, digest in producer_hashes.items()},
        "original_target_materials": appearance,
        "original_scene_object_transforms_before_variant_scale": [
            {
                key: obj[key]
                for key in (
                    "name",
                    "scale",
                    "rotation_euler_degrees",
                    "parent",
                    "modifiers",
                    "materials",
                )
            }
            for obj in objects
        ],
        "interpretation": "Mesh silhouette includes actual finite triangles, independent of shading. Native center-hit pixel bounds are quantized. RGB gradients/highlights are appearance evidence and need not be silhouette edges. Never feed this audit to a detector.",
    }
    for path, digest in consumed.items():
        if sha256(path) != digest:
            raise ValueError("Read-only source changed during audit")
    write_json(output / "summary.json", result)
    if args.share_json:
        share_path = args.share_json if args.share_json.is_absolute() else ROOT / args.share_json
        compact = {
            key: value for key, value in result.items() if key not in ("rows", "source_sha256")
        }
        compact["full_local_report"] = (output / "summary.json").relative_to(ROOT).as_posix()
        compact["full_report_sha256"] = sha256(output / "summary.json")
        compact["rows"] = []
        for row in rows:
            reduced = {key: value for key, value in row.items() if key != "segments"}
            reduced["segments"] = []
            for segment in row["segments"]:
                part = {
                    key: value
                    for key, value in segment.items()
                    if key not in ("first_hit_visibility", "rgb_row_profile")
                }
                part["visible_intervals_index"] = segment["first_hit_visibility"]["intervals_index"]
                if "rgb_row_profile" in segment:
                    part["brightest_minus_physical_axis_px"] = segment["rgb_row_profile"][
                        "brightest_minus_physical_axis_px"
                    ]
                reduced["segments"].append(part)
            compact["rows"].append(reduced)
        share_path.parent.mkdir(parents=True, exist_ok=True)
        with share_path.open("x", encoding="utf-8") as stream:
            json.dump(compact, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "axis_vs_mesh_silhouette": statistics,
                "native_ray_samples": native_ray_samples,
                "native_ray_disagreements": native_ray_disagreements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
