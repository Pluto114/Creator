"""Explicit oracle-camera diagnostic; never use this as the RGB-only model baseline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from smoke_da3 import run
from thin_pack_gt import read_json, sha256, write_json

CONVENTION = "opencv_world_to_camera_meters_K_edge"


def normalized_extrinsics_known_answer(extrinsics):
    """DA3 anchors to camera zero, then divides translations by a median baseline."""
    normalized = extrinsics @ np.linalg.inv(extrinsics[0])
    centers = np.linalg.inv(normalized)[:, :3, 3]
    distances = np.linalg.norm(centers, axis=1)
    # torch.median picks the lower middle value for an even view count; np.median does not.
    median = max(float(np.sort(distances)[(len(distances) - 1) // 2]), 0.1)
    normalized[:, :3, 3] /= median
    return normalized, median


def scaled_edge_intrinsics(cameras, output_height, output_width):
    result = np.array([frame["K_edge"] for frame in cameras["frames"]], dtype=np.float64)
    for k, frame in zip(result, cameras["frames"]):
        k[0] *= output_width / frame["width"]
        k[1] *= output_height / frame["height"]
    return result


def validate_request(request, root):
    from PIL import Image

    if request["model"] != "large" or request["process_res"] not in (504, 756):
        raise ValueError("This frozen oracle diagnostic uses large at 504 or 756 only")
    camera_path = Path(request["camera_file"])
    if not camera_path.is_absolute() or not camera_path.resolve().is_relative_to(root):
        raise ValueError("Oracle camera file must be an absolute path within the actual project")
    if camera_path.resolve().is_relative_to(root / "data/inputs"):
        raise ValueError("Oracle cameras must stay outside the RGB-only input tree")
    camera_hash = sha256(camera_path)
    if request.get("camera_sha256", camera_hash) != camera_hash:
        raise ValueError("Oracle camera file identity mismatch")
    cameras = read_json(camera_path)
    if cameras.get("coordinate_convention") != CONVENTION:
        raise ValueError("Expected explicitly declared OpenCV w2c meters and K_edge")
    frames = request["frames"]
    if len(frames) != 5 or len(cameras["frames"]) != 5:
        raise ValueError("This diagnostic is paired to the five-view pilot; no RANSAC branch")
    images = []
    sizes = []
    for frame, camera in zip(frames, cameras["frames"]):
        path = Path(frame["rgb"])
        if not path.is_absolute() or not path.resolve().is_relative_to(root / "data/inputs"):
            raise ValueError("Oracle RGB must be an absolute path in the paired RGB input tree")
        if sha256(path) != frame["sha256"] or camera["rgb_sha256"] != frame["sha256"]:
            raise ValueError("RGB order or oracle camera identity mismatch")
        with Image.open(path) as image:
            size = image.size
        if size != (camera["width"], camera["height"]):
            raise ValueError("Camera calibration dimensions differ from actual input RGB")
        k = np.asarray(camera["K_edge"], dtype=np.float64)
        ext = np.asarray(camera["world_to_camera_cv"], dtype=np.float64)
        if k.shape != (3, 3) or ext.shape != (4, 4):
            raise ValueError("Expected one 3x3 K and one 4x4 world-to-camera matrix per frame")
        if not np.isfinite(k).all() or not np.isfinite(ext).all():
            raise ValueError("Non-finite camera calibration")
        if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsic matrix")
        # The installed camera encoder ignores principal point and skew. An off-center
        # crop would pretend to provide more calibration than the network actually sees.
        if not np.allclose([k[0, 1], k[1, 0]], 0, atol=1e-8):
            raise ValueError("Skew is unsupported by this full-frame oracle protocol")
        if not np.allclose(k[:2, 2], np.asarray(size) / 2, atol=1e-4):
            raise ValueError("Only centered full-frame oracle cameras are supported here")
        if not np.allclose(ext[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError("Extrinsics must be affine world-to-camera matrices")
        rotation = ext[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
            raise ValueError("Camera rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Camera rotation must be proper, with no axis reflection")
        images.append(path)
        sizes.append(size)
    if len(set(sizes)) != 1 or len(set(images)) != 5:
        raise ValueError("Require five distinct frames with a shared full-frame image size")
    ext = np.array([frame["world_to_camera_cv"] for frame in cameras["frames"]])
    centers = np.linalg.inv(ext)[:, :3, 3]
    singular_values = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    if singular_values[0] <= 0 or singular_values[1] / singular_values[0] < 0.001:
        raise ValueError("Camera path is degenerate for a scale diagnostic")
    return images, cameras, camera_hash


@contextmanager
def observe_oracle_inference(model, audit, raw_arrays, cameras):
    """Observe the real API calls, preserve its output, then put the instance back."""
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    original_normalize = model._normalize_extrinsics
    original_align = model._align_to_input_extrinsics_intrinsics

    def normalize(extrinsics):
        incoming = extrinsics[0].detach().cpu().numpy().copy()
        expected, baseline = normalized_extrinsics_known_answer(incoming)
        result = original_normalize(extrinsics)
        actual = result[0].detach().cpu().numpy()
        residual = float(np.max(np.abs(actual - expected)))
        if residual > 2e-5:
            raise ValueError("Actual upstream camera normalization failed its known answer")
        audit.update(
            input_baseline_normalizer_m=baseline,
            normalized_extrinsics_max_abs_residual=residual,
            normalized_extrinsics=actual.tolist(),
        )
        return result

    def align(extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, **kwargs):
        if not align_to_input_ext_scale or len(extrinsics) != 5:
            raise ValueError("Frozen oracle protocol requires input-scale alignment and five views")
        ext = extrinsics.numpy()
        raw_arrays.update(
            depth=prediction.depth.copy(),
            extrinsics=prediction.extrinsics.copy(),
            intrinsics=prediction.intrinsics.copy(),
        )
        # This is the actual API's direction: INPUT camera path -> PREDICTED path.
        # Its scale is the reciprocal of the scale often used by our normal evaluator.
        _, _, scale, _ = align_poses_umeyama(
            raw_arrays["extrinsics"], ext, ransac=False, return_aligned=True, random_state=42
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid input-to-prediction camera scale")
        result = original_align(extrinsics, intrinsics, prediction, True, **kwargs)
        h, w = result.depth.shape[-2:]
        expected_k = scaled_edge_intrinsics(cameras, h, w)
        k_error = float(np.max(np.abs(result.intrinsics - expected_k)))
        ext_error = float(np.max(np.abs(result.extrinsics - ext[:, :3])))
        depth_error = float(np.max(np.abs(result.depth - raw_arrays["depth"] / scale)))
        if k_error > 2e-4 or ext_error > 1e-7 or depth_error > 1e-5:
            raise ValueError("Returned oracle calibration or scale does not match upstream semantics")
        audit.update(
            raw_input_to_predicted_camera_scale=float(scale),
            raw_depth_to_input_meter_multiplier=float(1 / scale),
            api_replaced_returned_extrinsics_with_input=True,
            returned_intrinsics_convention="K_edge_scaled_by_upstream; preserved unchanged",
            returned_intrinsics_vs_expected_scaled_K_edge_max_abs_px=k_error,
            returned_extrinsics_vs_input_w2c_max_abs=ext_error,
            returned_depth_vs_raw_divide_scale_max_abs=depth_error,
            backprojection_diagnostic={
                "array_grid": "integer arange(u), arange(v)",
                "intrinsics": "copy returned K; subtract 0.5 from cx and cy only in diagnostic",
                "depth": "returned camera Z already scaled to input-camera meters",
                "world": "inverse returned OpenCV world-to-camera; no extra axis flip or Sim3",
                "note": "Zero returned pose error is an API replacement, not camera accuracy.",
            },
        )
        return result

    model._normalize_extrinsics = normalize
    model._align_to_input_extrinsics_intrinsics = align
    try:
        yield
    finally:
        model._normalize_extrinsics = original_normalize
        model._align_to_input_extrinsics_intrinsics = original_align


def known_answer_checks():
    """CPU only: exercise installed upstream code without constructing or loading a model."""
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
    from depth_anything_3.specs import Prediction
    from depth_anything_3.utils.io.input_processor import InputProcessor

    processor = InputProcessor()
    k = np.array([[1600.0, 0, 960], [0, 1600.0, 540], [0, 0, 1]])
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    checks = {}
    for resolution, expected_shape in ((504, (280, 504)), (756, (420, 756))):
        tensor, shape, native_k, _ = processor._process_one(
            image, intrinsic=k, process_res=resolution, process_res_method="upper_bound_resize"
        )
        if shape != expected_shape or tuple(tensor.shape[-2:]) != expected_shape:
            raise AssertionError("Unexpected installed upstream resize dimensions")
        h, w = shape
        expected_k = np.diag([w / 1920, h / 1080, 1]) @ k
        np.testing.assert_allclose(native_k, expected_k, atol=1e-10)
        index_k = native_k.copy()
        index_k[:2, 2] -= 0.5
        uv = np.array([[0, 0, 1], [w - 1, h - 1, 1], [w // 2, h // 2, 1]])
        edge_uv = uv.astype(float)
        edge_uv[:, :2] += 0.5
        np.testing.assert_allclose(
            uv @ np.linalg.inv(index_k).T, edge_uv @ np.linalg.inv(native_k).T, atol=1e-12
        )
        checks[f"resize_{resolution}_and_edge_index_rays"] = True

    centers = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0], [-2, 1, 0], [-1, -2, 0]])
    ext = np.repeat(np.eye(4)[None], 5, axis=0)
    ext[:, :3, 3] = -centers
    ext[:, :3, 3] += [4, -3, 2]
    expected, baseline = normalized_extrinsics_known_answer(ext)
    tensor_ext = torch.tensor(ext, dtype=torch.float64)
    actual = DepthAnything3._normalize_extrinsics(None, tensor_ext[None].clone())[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual[0], np.eye(4), atol=1e-12)
    checks["normalization_first_view_and_median_baseline"] = True
    checks["known_median_baseline"] = baseline

    raw_ext = ext.copy()
    raw_ext[:, :3, 3] *= 0.25
    intrinsic = torch.tensor(np.repeat(k[None], 5, axis=0), dtype=torch.float64)
    prediction = Prediction(
        depth=np.full((5, 2, 3), 2.0), is_metric=0,
        extrinsics=raw_ext[:, :3].copy(), intrinsics=np.repeat(k[None], 5, axis=0),
    )
    output = DepthAnything3._align_to_input_extrinsics_intrinsics(
        None, tensor_ext, intrinsic, prediction, True
    )
    np.testing.assert_allclose(output.depth, 8.0, atol=1e-12)
    np.testing.assert_array_equal(output.extrinsics, ext[:, :3])
    np.testing.assert_array_equal(output.intrinsics, intrinsic.numpy())
    checks["actual_api_depth_divides_scale_and_overwrites_extrinsics"] = True

    shifted = intrinsic.clone()
    shifted[:, :2, 2] += 23
    encoding = extri_intri_to_pose_encoding(tensor_ext[None], intrinsic[None], (1080, 1920))
    shifted_encoding = extri_intri_to_pose_encoding(tensor_ext[None], shifted[None], (1080, 1920))
    torch.testing.assert_close(encoding, shifted_encoding, atol=0, rtol=0)
    checks["actual_camera_encoder_ignores_principal_point"] = True

    class CpuApi:
        _normalize_extrinsics = DepthAnything3._normalize_extrinsics
        _align_to_input_extrinsics_intrinsics = DepthAnything3._align_to_input_extrinsics_intrinsics

    cpu_api = CpuApi()
    original_normalize = cpu_api._normalize_extrinsics
    original_align = cpu_api._align_to_input_extrinsics_intrinsics
    camera_fixture = {"frames": [
        {"K_edge": k.tolist(), "width": 1920, "height": 1080} for _ in range(5)
    ]}
    small_intrinsic = torch.tensor(scaled_edge_intrinsics(camera_fixture, 2, 3))
    observed_prediction = Prediction(
        depth=np.full((5, 2, 3), 2.0), is_metric=0,
        extrinsics=raw_ext[:, :3].copy(), intrinsics=np.repeat(k[None], 5, axis=0),
    )
    audit, raw_arrays = {}, {}
    with observe_oracle_inference(cpu_api, audit, raw_arrays, camera_fixture):
        cpu_api._normalize_extrinsics(tensor_ext[None].clone())
        observed_output = cpu_api._align_to_input_extrinsics_intrinsics(
            tensor_ext, small_intrinsic, observed_prediction, True
        )
    np.testing.assert_allclose(observed_output.depth, 8.0, atol=1e-12)
    np.testing.assert_allclose(raw_arrays["depth"], 2.0, atol=1e-12)
    np.testing.assert_allclose(audit["raw_depth_to_input_meter_multiplier"], 4.0, atol=1e-12)
    if cpu_api._normalize_extrinsics != original_normalize or cpu_api._align_to_input_extrinsics_intrinsics != original_align:
        raise AssertionError("Observer failed to restore original methods")
    checks["observer_preserves_raw_depth_and_restores_actual_api_methods"] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from environment_paths import require_project_environment

    require_project_environment(root)
    if args.self_test:
        import json

        print(json.dumps({"ok": True, "checks": known_answer_checks()}, indent=2))
        return
    destination = args.request.resolve().parent
    if not destination.is_relative_to(root / ".runtime/experiments"):
        raise ValueError("Oracle output must stay in the project experiment runtime directory")
    if (destination / "prediction.npz").exists():
        raise FileExistsError("Refusing to overwrite an existing oracle prediction")
    audit = {}
    write_json(destination / "report.json", {"ok": False, "state": "running", "camera_mode": "oracle_camera"})
    try:
        request = read_json(args.request)
        images, cameras, camera_hash = validate_request(request, root)
        checks = known_answer_checks()
        raw_arrays = {}
        inference_args = argparse.Namespace(
            images=images, model=request["model"], process_res=request["process_res"],
            use_ray_pose=False,
        )
        report = run(
            inference_args, root, destination,
            inference_options={
                "extrinsics": np.array([f["world_to_camera_cv"] for f in cameras["frames"]]),
                "intrinsics": np.array([f["K_edge"] for f in cameras["frames"]]),
                "align_to_input_ext_scale": True,
            },
            inference_context=lambda model: observe_oracle_inference(model, audit, raw_arrays, cameras),
        )
        if not audit.get("api_replaced_returned_extrinsics_with_input"):
            raise ValueError("Oracle observer did not observe the expected upstream camera path")
        raw_path = destination / "pre_alignment_prediction.npz"
        np.savez_compressed(raw_path, **raw_arrays)
        report.update(
            state="succeeded",
            scope="oracle_camera diagnostic; known cameras supplied; not ordinary RGB-only recovery",
            camera_mode="oracle_camera",
            camera_file=str(Path(request["camera_file"]).resolve()),
            camera_sha256=camera_hash,
            request_sha256=sha256(args.request),
            prediction_sha256=sha256(destination / "prediction.npz"),
            pre_alignment_prediction_sha256=sha256(raw_path),
            known_answer_checks=checks,
            oracle_audit=audit,
        )
        write_json(destination / "report.json", report)
    except Exception as error:
        write_json(destination / "report.json", {
            "ok": False, "state": "failed", "camera_mode": "oracle_camera",
            "error": str(error), "oracle_audit": audit,
        })
        raise


if __name__ == "__main__":
    main()
