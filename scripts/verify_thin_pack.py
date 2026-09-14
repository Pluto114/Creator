"""Read-only audit of a completed paired bundle, including serialized ray/pixel correspondence."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import MeshRays, read_json, sha256


def verify(root, bundle_id):
    gt = root / "data/eval_gt" / bundle_id
    inputs = root / "data/inputs" / bundle_id
    if gt.resolve().parent != (root / "data/eval_gt").resolve():
        raise ValueError("Bundle ID must be a direct child of eval_gt")
    status = read_json(gt / "status.json")
    assert status["state"] == "complete"
    assert sha256(gt / "artifact_hashes.json") == status["artifact_hashes_sha256"]
    hashes = read_json(gt / "artifact_hashes.json")
    actual_files = {p.relative_to(gt).as_posix() for p in gt.rglob("*") if p.is_file()}
    assert actual_files == set(hashes) | {"artifact_hashes.json", "status.json"}
    total_bytes = 0
    for relative, identity in hashes.items():
        path = gt / relative
        assert path.resolve().is_relative_to(gt.resolve())
        assert path.stat().st_size == identity["byte_size"], relative
        assert sha256(path) == identity["sha256"], relative
        total_bytes += path.stat().st_size
    provenance = read_json(gt / "provenance.json")
    assert sha256(root / provenance["source_scene"]) == provenance["source_sha256"]
    inputs_manifest = read_json(inputs / "manifest.json")
    assert inputs_manifest["known_cameras"] == []
    expected_files = {"manifest.json"}
    count = 0
    array_count = 0
    max_depth_error = 0.0
    for group in inputs_manifest["groups"]:
        assert group["frame_order"] == [f["frame_id"] for f in group["frames"]]
        rays = MeshRays(gt / group["case_id"] / "mesh.npz")
        for frame in group["frames"]:
            count += 1
            expected_files.add(frame["rgb"])
            assert sha256(inputs / frame["rgb"]) == frame["sha256"]
            with Image.open(inputs / frame["rgb"]) as im:
                assert im.mode == "RGB" and list(im.size) == frame["size_wh"]
            folder = gt / group["case_id"] / frame["frame_id"]
            for size in ("native", "da3_504", "da3_756"):
                camera = read_json(
                    folder / "camera.json" if size == "native" else folder / size / "camera.json"
                )
                arrays = {}
                for name, spec in read_json(folder / size / "arrays.json").items():
                    a = np.load(folder / size / spec["path"], mmap_mode="r", allow_pickle=False)
                    assert list(a.shape) == spec["shape"] and str(a.dtype) == spec["dtype"]
                    arrays[name] = a
                    array_count += 1
                w, h = camera["size_wh"]
                assert arrays["depth_z"].shape == (h, w)
                hit = arrays["hit_valid"]
                assert np.array_equal(np.isfinite(arrays["depth_z"]), hit)
                assert np.array_equal(np.isnan(arrays["depth_z"]), ~hit)
                assert np.all(arrays["depth_z"][hit] > 0)
                assert np.array_equal(hit, arrays["surface_id"] != 0)
                for rod_id in range(1, 7):
                    assert np.array_equal(
                        arrays["rod_visible_center_masks"][rod_id - 1], arrays["rod_id"] == rod_id
                    )
                assert arrays["segment_coverage_counts"].shape == (7, h, w)
                assert np.all(arrays["segment_coverage_counts"].sum(0) <= 16)
                assert not np.any(
                    arrays["strict_depth_eval_valid"] & (~hit | arrays["sample_mixed"])
                )
                if size == "native":
                    assert np.all(
                        arrays["blender_depth_within_sample_footprint"][
                            arrays["strict_depth_eval_valid"]
                        ]
                    )
                ki, ke = np.array(camera["K_index"]), np.array(camera["K_edge"])
                corrected = ke.copy()
                corrected[:2, 2] -= 0.5
                np.testing.assert_allclose(ki, corrected, atol=1e-10, rtol=0)
                rotation = np.array(camera["world_to_camera_cv"])[:3, :3]
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
                assert abs(np.linalg.det(rotation) - 1) < 1e-6
                y, x = np.mgrid[3:h:19, 3:w:23]
                z, ids, distance = rays.cast(camera, np.stack([x + 0.5, y + 0.5], -1))
                np.testing.assert_array_equal(ids.reshape(x.shape), arrays["surface_id"][y, x])
                valid = np.isfinite(z)
                delta = np.abs(z[valid] - arrays["depth_z"][y, x].ravel()[valid])
                error = float(delta.max()) if delta.size else 0.0
                assert error < 1e-4
                max_depth_error = max(max_depth_error, error)
                np.testing.assert_allclose(
                    distance.reshape(x.shape),
                    arrays["ray_distance"][y, x],
                    atol=1e-4,
                    rtol=0,
                    equal_nan=True,
                )
    assert {
        p.relative_to(inputs).as_posix() for p in inputs.rglob("*") if p.is_file()
    } == expected_files
    assert count == status["view_count"]
    # Validate old input identities against the earlier frozen v1 source record.
    historical = read_json(root / "data/eval_gt/thin-pack-v1/geometry_cameras.json")
    old_count = 0
    for case in historical["cases"]:
        for view in case["views"]:
            assert sha256(view["input_file"]) == view["input_sha256"]
            old_count += 1
    return {
        "status": "passed",
        "bundle_id": bundle_id,
        "paired_view_count": count,
        "gt_file_count": len(hashes),
        "gt_bytes": total_bytes,
        "arrays_validated": array_count,
        "max_serialized_depth_raycast_error_m": max_depth_error,
        "historical_inputs_hash_verified": old_count,
        "source_scene_hash_verified": True,
        "rgb_input_isolation_verified": True,
        "audit_script_sha256": sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.bundle_id), indent=2))
