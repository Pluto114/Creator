"""Small, explicit diagnostics before we introduce a curve-scoring pipeline."""

import numpy as np


class AlignmentDegenerate(ValueError):
    """Camera positions do not determine a stable similarity transform."""


def homogeneous(extrinsic):
    value = np.asarray(extrinsic, dtype=np.float64)
    if value.shape == (4, 4):
        return value
    if value.shape != (3, 4):
        raise ValueError("Expected a 3x4 or 4x4 world-to-camera matrix")
    out = np.eye(4)
    out[:3] = value
    return out


def camera_centers(extrinsics):
    return np.array([np.linalg.inv(homogeneous(e))[:3, 3] for e in extrinsics])


def apply_similarity(points, scale, rotation, translation):
    return scale * (np.asarray(points) @ rotation.T) + translation


def fit_sim3(source, target, minimum_ratio=0.001):
    """Equal-weight least squares; no target geometry, ICP, or outlier shopping."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Expected matching Nx3 point arrays")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise AlignmentDegenerate("Need at least three finite camera correspondences")
    mx, my = source.mean(0), target.mean(0)
    x, y = source - mx, target - my
    spectra = [np.linalg.svd(a, compute_uv=False) for a in (x, y)]
    # 拍摄轨迹在一个平面里完全正常。真正麻烦的是几台相机挤成一条线：
    # 这时绕那条线怎么转都差不多，硬算出来的三维分数会非常唬人。
    ratios = [float(s[1] / s[0]) if s[0] > 1e-12 else 0.0 for s in spectra]
    if min(ratios) < minimum_ratio:
        raise AlignmentDegenerate(f"Nearly collinear camera centers: {ratios}")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = (u * correction) @ vt
    scale = float((singular * correction).sum() / np.mean(np.sum(x * x, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise AlignmentDegenerate("No positive similarity scale")
    translation = my - scale * (rotation @ mx)
    residual = np.linalg.norm(
        apply_similarity(source, scale, rotation, translation) - target, axis=1
    )
    return (
        scale,
        rotation,
        translation,
        {
            "source_singular_values": spectra[0].tolist(),
            "target_singular_values": spectra[1].tolist(),
            "second_to_first_singular_ratios": ratios,
            "camera_residuals_m": residual.tolist(),
            "camera_rmse_m": float(np.sqrt(np.mean(residual**2))),
            "camera_span_m": float(
                np.max(np.linalg.norm(target[:, None] - target[None, :], axis=-1))
            ),
        },
    )


def align_cameras(predicted, truth, config):
    source, target = camera_centers(predicted), camera_centers(truth)
    scale, rotation, translation, report = fit_sim3(
        source, target, config["minimum_second_to_first_singular_ratio"]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    report.update(
        scale=scale,
        prediction_world_to_gt_world=matrix.tolist(),
        fit_inputs="camera_centers_only",
        camera_rmse_over_span=report["camera_rmse_m"] / report["camera_span_m"],
    )
    report["poor_camera_fit"] = (
        report["camera_rmse_over_span"] > config["poor_fit_rmse_over_camera_span"]
    )
    angles = []
    for p, g in zip(predicted, truth):
        a = (
            np.linalg.inv(homogeneous(g))[:3, :3].T
            @ rotation
            @ np.linalg.inv(homogeneous(p))[:3, :3]
        )
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(a) - 1) / 2, -1, 1)))))
    report["orientation_errors_degrees"] = angles
    leave_out = []
    if config["leave_one_camera_out_diagnostic"]:
        for index in range(len(source)):
            keep = np.arange(len(source)) != index
            try:
                s, r, t, _ = fit_sim3(
                    source[keep], target[keep], config["minimum_second_to_first_singular_ratio"]
                )
                leave_out.append(
                    {
                        "omitted_view": index,
                        "scale": s,
                        "held_out_camera_error_m": float(
                            np.linalg.norm(apply_similarity(source[index], s, r, t) - target[index])
                        ),
                    }
                )
            except AlignmentDegenerate as error:
                leave_out.append({"omitted_view": index, "reason": str(error)})
    report["leave_one_camera_out"] = leave_out
    return scale, rotation, translation, report


def unproject(depth, intrinsic, extrinsic):
    """Native DA3 uses integer pixel coordinates. GT callers supply K_index."""
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[..., None]
    inverse = np.linalg.inv(homogeneous(extrinsic))
    return camera @ inverse[:3, :3].T + inverse[:3, 3]


def confidence_masks(depth, confidence, config):
    if not np.isfinite(confidence).all():
        raise ValueError("Non-finite confidence values")
    low, high = np.percentile(
        confidence, [config["default_percentile"], config["default_upper_percentile"]]
    )
    threshold = float(min(max(config["fixed_confidence"], low), high))
    valid = np.isfinite(depth) & (depth > 0)
    return {
        "unfiltered": valid,
        "default_p40": valid & (confidence >= threshold),
        "fixed_1_05": valid & (confidence >= config["fixed_confidence"]),
    }, threshold


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        # 空集合就是没法算。这里填0，最后的表格会把“没有证据”画成“完美”。
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "mean": None,
            "reason": "no_finite_samples",
        }
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
    }


def region_report(region, kept, z_error=None, point_error=None):
    total = int(region.sum())
    selected = region & kept
    count = int(selected.sum())
    result = {
        "available_pixels": total,
        "retained_pixels": count,
        "retained_fraction": count / total if total else None,
    }
    if not total:
        result["reason"] = "empty_region"
    if z_error is not None:
        result["scaled_camera_z_abs_error_m"] = distribution(z_error[selected])
    if point_error is not None:
        result["paired_pixel_world_point_displacement_m"] = distribution(point_error[selected])
    return result
