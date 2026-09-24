"""Native-gauge sensitivity diagnostics; no truth, alignment, or pass/fail gate.

A fold is compared with the fixed full-training result, never promoted over it.
The inherited camera scale is arbitrary, so distances are divided by its baseline.
"""
from __future__ import annotations

import numpy as np

from creator_eval.camera_bundle import centers, validate_cameras
from creator_eval.line_controls import curve_metrics

NORMALIZED_TOLERANCE = 0.001
NORMALIZED_SPACING = 0.001
# DA3 starts with float32 rotations. The old gauge decodes t=-RC and reads
# centers with -R.T@t; its inherited R.T@R error survives both optimizer stages.
# Two zero-parameter decodes already show about 1e-7 baseline drift. This is
# numerical comparison slack, not an optimizer bound or a fitted scale change.
GAUGE_TOLERANCE = 1e-6


def _extrinsics(value):
    e = np.asarray(value, float)
    if e.ndim != 3 or e.shape[1:] != (3, 4):
        raise ValueError("Expected native Nx3x4 extrinsics")
    _, e = validate_cameras(np.tile(np.eye(3), (len(e), 1, 1)), e)
    return e


def _baseline(initial_e):
    e = _extrinsics(initial_e)
    c = centers(e)
    baseline = float(np.linalg.norm(c[-1] - c[0]))
    if baseline < 1e-8:
        raise ValueError("Initial first-last camera baseline is degenerate")
    return e, baseline


def check_native_gauge(initial_e, e):
    """Check the original first pose and first-last distance without realigning."""
    initial, baseline = _baseline(initial_e)
    result = dict(comparable=False, reason=None, baseline=baseline,
                  first_rotation_max_abs_delta=None, first_translation_max_abs_delta=None,
                  baseline_relative_delta=None, alignment_performed=False,
                  first_rotation_tolerance=GAUGE_TOLERANCE,
                  first_translation_tolerance=GAUGE_TOLERANCE * max(1., baseline),
                  baseline_relative_tolerance=GAUGE_TOLERANCE)
    if e is None:
        return {**result, "reason": "missing_camera"}
    current = _extrinsics(e)
    if current.shape != initial.shape:
        raise ValueError("Camera count or view order contract changed")
    rotation_delta = float(np.max(abs(current[0, :, :3] - initial[0, :, :3])))
    translation_delta = float(np.max(abs(current[0, :, 3] - initial[0, :, 3])))
    c = centers(current)
    relative_delta = float(abs(np.linalg.norm(c[-1] - c[0]) / baseline - 1))
    reasons = []
    if rotation_delta > GAUGE_TOLERANCE or translation_delta > GAUGE_TOLERANCE * max(1., baseline):
        reasons.append("first_camera_pose_changed")
    if relative_delta > GAUGE_TOLERANCE:
        reasons.append("first_last_baseline_changed")
    return {**result, "comparable": not reasons, "reason": ";".join(reasons) or None,
            "first_rotation_max_abs_delta": rotation_delta,
            "first_translation_max_abs_delta": translation_delta,
            "baseline_relative_delta": relative_delta}


def camera_difference(ref_k, ref_e, k, e, initial_e):
    """Compare all available numeric cameras, independently of their RGB gate.

    Center and rotation summaries exclude the fixed first pose. Focal summaries
    include every camera, because its first focal length is still a free value.
    """
    initial, baseline = _baseline(initial_e)
    output = dict(
        comparable=False, reason=None, baseline=baseline, view_count=len(initial),
        compared_pose_view_count=len(initial) - 1, reference_gauge=None, candidate_gauge=None,
        center_delta_over_baseline_median=None, center_delta_over_baseline_max=None,
        rotation_degrees_median=None, rotation_degrees_max=None,
        absolute_log_focal_ratio_median=None, absolute_log_focal_ratio_max=None,
        per_view=[], alignment_performed=False,
        scope="native_camera_sensitivity_not_accuracy_or_acceptance_gate",
    )
    missing = []
    if ref_k is None or ref_e is None:
        missing.append("missing_reference_camera")
    if k is None or e is None:
        missing.append("missing_candidate_camera")
    if missing:
        return {**output, "reason": ";".join(missing)}
    reference_k, reference_e = validate_cameras(ref_k, ref_e)
    candidate_k, candidate_e = validate_cameras(k, e)
    reference_gauge = check_native_gauge(initial, reference_e)
    candidate_gauge = check_native_gauge(initial, candidate_e)
    output.update(reference_gauge=reference_gauge, candidate_gauge=candidate_gauge)
    failures = [name + ":" + check["reason"] for name, check in
                (("reference", reference_gauge), ("candidate", candidate_gauge))
                if not check["comparable"]]
    if failures:
        return {**output, "reason": ";".join(failures)}
    center_delta = np.linalg.norm(centers(candidate_e) - centers(reference_e), axis=1) / baseline
    relative_rotations = candidate_e[:, :, :3] @ reference_e[:, :, :3].transpose(0, 2, 1)
    cosine = np.clip((np.trace(relative_rotations, axis1=1, axis2=2) - 1) / 2, -1., 1.)
    # atan2 retains tiny rotations that acos(trace) would round down to zero.
    skew = np.stack((relative_rotations[:, 2, 1] - relative_rotations[:, 1, 2],
                     relative_rotations[:, 0, 2] - relative_rotations[:, 2, 0],
                     relative_rotations[:, 1, 0] - relative_rotations[:, 0, 1]), axis=1)
    angles = np.degrees(np.arctan2(np.linalg.norm(skew, axis=1) / 2, cosine))
    log_focal = np.log(candidate_k[:, [0, 1], [0, 1]] / reference_k[:, [0, 1], [0, 1]])
    output.update(
        comparable=True,
        center_delta_over_baseline_median=float(np.median(center_delta[1:])),
        center_delta_over_baseline_max=float(np.max(center_delta[1:])),
        rotation_degrees_median=float(np.median(angles[1:])),
        rotation_degrees_max=float(np.max(angles[1:])),
        absolute_log_focal_ratio_median=float(np.median(abs(log_focal))),
        absolute_log_focal_ratio_max=float(np.max(abs(log_focal))),
        per_view=[dict(view=i, pose_in_summary=i != 0,
                       center_delta_over_baseline=float(center_delta[i]),
                       rotation_degrees=float(angles[i]), log_focal_ratio_xy=log_focal[i].tolist())
                  for i in range(len(initial))],
    )
    return output


def _accepted_segments(identity, name):
    if identity is None:
        return None, name + "_missing_identity"
    if identity.get("state") != "accepted":
        return None, name + "_identity_not_accepted"
    value = identity.get("segments")
    if value is None:
        return None, name + "_missing_segments"
    segments = np.asarray(value, float)
    if not segments.size:
        return np.empty((0, 2, 3)), name + "_empty_segments"
    if segments.shape == (2, 3):
        segments = segments[None]
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Expected finite Nx2x3 accepted segments")
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1e-12):
        raise ValueError("Accepted segments must have positive length")
    return segments, None


def segment_difference(ref_identity, identity, baseline):
    """Compare finite accepted geometry already verified to share a native gauge.

    The caller must first establish gauge comparability. The fixed 0.001 baseline
    tolerance describes proximity only; it neither accepts a fold nor certifies
    correctness. Empty/empty is absence of measurable geometry, never zero error.
    """
    baseline = float(baseline)
    if not np.isfinite(baseline) or baseline < 1e-8:
        raise ValueError("Finite nondegenerate native baseline required")
    reference, reference_reason = _accepted_segments(ref_identity, "reference")
    candidate, candidate_reason = _accepted_segments(identity, "candidate")
    output = dict(
        comparable=False, reason=None, baseline=baseline,
        reference_state=None if ref_identity is None else ref_identity.get("state"),
        candidate_state=None if identity is None else identity.get("state"),
        reference_segment_count=None if reference is None else len(reference),
        candidate_segment_count=None if candidate is None else len(candidate),
        segment_count_delta=None, reference_length_over_baseline=None,
        candidate_length_over_baseline=None, length_ratio=None,
        reference_to_candidate_p95_over_baseline=None,
        candidate_to_reference_p95_over_baseline=None,
        symmetric_p95_over_baseline=None,
        reference_coverage_fraction=None, candidate_coverage_fraction=None,
        normalized_tolerance=NORMALIZED_TOLERANCE,
        normalized_sampling_spacing=NORMALIZED_SPACING,
        alignment_performed=False,
        scope="finite_curve_sensitivity_not_accuracy_topology_or_acceptance_gate",
    )
    reasons = [reason for reason in (reference_reason, candidate_reason) if reason]
    if reasons:
        return {**output, "reason": ";".join(reasons)}
    # 只借用双向有限段距离计算。full 是参照结果，不是偷偷换个名字的真值。
    metrics = curve_metrics(candidate, reference, tolerance=baseline * NORMALIZED_TOLERANCE,
                            spacing=baseline * NORMALIZED_SPACING)
    forward, backward = metrics["truth_to_prediction"], metrics["prediction_to_truth"]
    a, b = forward["distance_p95"] / baseline, backward["distance_p95"] / baseline
    return {**output, "comparable": True,
            "segment_count_delta": len(candidate) - len(reference),
            "reference_length_over_baseline": forward["source_length"] / baseline,
            "candidate_length_over_baseline": backward["source_length"] / baseline,
            "length_ratio": backward["source_length"] / forward["source_length"],
            "reference_to_candidate_p95_over_baseline": a,
            "candidate_to_reference_p95_over_baseline": b,
            "symmetric_p95_over_baseline": max(a, b),
            "reference_coverage_fraction": forward["covered_fraction"],
            "candidate_coverage_fraction": backward["covered_fraction"]}
