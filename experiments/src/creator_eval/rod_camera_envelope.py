"""Finite-rod ranges under a fixed set of camera perturbations.

This describes the supplied five conditions. It never selects a replacement
control, estimates coverage probability, or certifies physical correctness.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from creator_eval.camera_bundle import validate_cameras
from creator_eval.camera_native_sensitivity import check_native_gauge

DEFAULT_CONDITIONS = tuple("leave_group_" + str(i) for i in range(4))
STRUCTURE_EPS_OVER_BASELINE = 1e-6
_SUPPORT_HASHES = ("assignment_sha256", "source_rows_sha256", "pool_sha256")


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(value):
    data = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _support(value):
    if not isinstance(value, dict):
        raise ValueError("Accepted identity requires its nonempty frozen support")
    for name in _SUPPORT_HASHES:
        sha = value.get(name)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Support requires lowercase SHA256 values")
    count = value.get("source_row_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Accepted identity requires a positive source row count")
    return {name: value[name] for name in (*_SUPPORT_HASHES, "source_row_count")}


def _geometry(value, eps):
    segments = np.asarray(value, float)
    if not segments.size:
        return None, "empty_accepted_segments"
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Accepted geometry must be finite Nx2x3 segments")
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if np.any(lengths <= eps):
        return None, "unsupported_numerically_short_segments"
    origin = segments[0, 0]
    axis = (segments[0, 1] - origin) / lengths[0]
    offsets = segments.reshape(-1, 3) - origin
    lateral = offsets - (offsets @ axis)[:, None] * axis
    if np.max(np.linalg.norm(lateral, axis=1)) > eps:
        return None, "unsupported_non_collinear_segments"
    coordinates = (segments - origin) @ axis
    intervals = np.sort(coordinates, axis=1)
    order = np.argsort(intervals[:, 0], kind="stable")
    gaps = intervals[order[1:], 0] - intervals[order[:-1], 1]
    if np.any(gaps < -eps):
        return None, "unsupported_overlapping_segments"
    if np.any(gaps <= eps):
        return None, "unsupported_touching_or_indistinct_segments"
    return dict(segments=segments, origin=origin, axis=axis, intervals=intervals, order=order), None


def _slot(packet, condition_id, initial, eps):
    slot = dict(condition_id=condition_id, packet_sha256=None, state="unresolved", reasons=[],
                camera_decision_state=None, identity_state=None, support=None,
                gauge=None, segment_count=None, matched_segments=None,
                matched_source_segment_indices=None)
    if packet is None:
        slot.update(state="missing", reasons=["missing_condition"])
        return slot, None
    slot["packet_sha256"] = _digest(packet)
    slot["camera_decision_state"] = packet.get("camera_decision", {}).get("state")
    identity = packet.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Each present packet requires an identity record")
    slot["identity_state"] = identity.get("state")
    if slot["camera_decision_state"] != "candidate_camera_correction":
        slot["reasons"].append("camera_not_accepted")
    if packet.get("intrinsics") is None or packet.get("extrinsics") is None:
        slot["reasons"].append("missing_numeric_camera")
    else:
        _, e = validate_cameras(packet["intrinsics"], packet["extrinsics"])
        slot["gauge"] = check_native_gauge(initial, e)
        if not slot["gauge"]["comparable"]:
            slot["reasons"].append("native_gauge_changed:" + slot["gauge"]["reason"])
    geometry = None
    if slot["identity_state"] != "accepted":
        # 拒绝就是拒绝，不用伪造空像素哈希来凑一个“完整”输入。
        slot["reasons"].append("identity_not_accepted")
    else:
        slot["support"] = _support(packet.get("support"))
        if identity.get("segments") is None:
            slot["reasons"].append("missing_accepted_segments")
        else:
            geometry, reason = _geometry(identity["segments"], eps)
            if reason:
                slot["reasons"].append(reason)
            else:
                slot["segment_count"] = len(geometry["segments"])
    return slot, geometry


def _match(control, candidate, eps):
    reference, supplied = control["segments"], candidate["segments"]
    if len(reference) != len(supplied):
        return None, None, "segment_count_changed"
    coordinates = (supplied - control["origin"]) @ control["axis"]
    intervals = np.sort(coordinates, axis=1)
    if np.any(intervals[:, 1] - intervals[:, 0] <= eps):
        return None, None, "unsupported_zero_projected_segment_span"
    candidate_order = np.argsort(intervals[:, 0], kind="stable")
    overlap = (np.minimum(control["intervals"][:, None, 1], intervals[None, :, 1])
               - np.maximum(control["intervals"][:, None, 0], intervals[None, :, 0])) > eps
    if np.any(overlap.sum(axis=1) != 1) or np.any(overlap.sum(axis=0) != 1):
        return None, None, "non_one_to_one_positive_overlap_correspondence"
    assignment = np.argmax(overlap, axis=1)
    if not np.array_equal(assignment[control["order"]], candidate_order):
        return None, None, "segment_order_changed"
    gaps = intervals[candidate_order[1:], 0] - intervals[candidate_order[:-1], 1]
    if np.any(gaps <= eps):
        return None, None, "control_gap_not_preserved"
    matched = []
    for i, source in enumerate(assignment):
        low_first = coordinates[source, 0] <= coordinates[source, 1]
        current = supplied[source] if low_first else supplied[source, ::-1]
        reference_forward = float((reference[i, 1] - reference[i, 0]) @ control["axis"]) > 0
        matched.append(current if reference_forward else current[::-1])
    return np.asarray(matched), assignment.tolist(), None


def _frame(axis):
    helper = np.eye(3)[int(np.argmin(abs(axis)))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return np.stack((axis, first, np.cross(axis, first)))


def _angles(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                np.sum(a * b, axis=-1)))


def _segment_envelopes(control, conditions, frame, baseline, count):
    fractions = np.linspace(0., 1., count)
    result = []
    for index, reference in enumerate(control["segments"]):
        supplied = np.asarray([slot["matched_segments"][index] for slot in conditions])
        delta = supplied - reference
        centers = supplied.mean(axis=1)
        lengths = np.linalg.norm(supplied[:, 1] - supplied[:, 0], axis=1)
        directions = (supplied[:, 1] - supplied[:, 0]) / lengths[:, None]
        reference_direction = (reference[1] - reference[0]) / np.linalg.norm(reference[1] - reference[0])
        angles = _angles(directions, reference_direction)
        pair_angles = _angles(directions[:, None, :], directions[None, :, :])
        points = supplied[:, 0, None, :] + fractions[None, :, None] * (
            supplied[:, 1, None, :] - supplied[:, 0, None, :])
        control_points = reference[0] + fractions[:, None] * (reference[1] - reference[0])
        local_delta = (points - control_points) @ frame.T
        endpoint_local = delta @ frame.T
        center_local = (centers - reference.mean(axis=0)) @ frame.T
        observations = [dict(condition_id=slot["condition_id"], endpoints_native=supplied[i],
                             center_native=centers[i], length_native=float(lengths[i]),
                             length_over_baseline=float(lengths[i] / baseline),
                             direction_unit=directions[i], angle_from_control_degrees=float(angles[i]),
                             endpoint_delta_local=endpoint_local[i], center_delta_local=center_local[i])
                        for i, slot in enumerate(conditions)]
        result.append(dict(
            control_segment_index=index, local_frame=frame, fractions=fractions,
            control_points=control_points, min_delta_local=local_delta.min(axis=0),
            max_delta_local=local_delta.max(axis=0),
            endpoint_native_min=supplied.min(axis=0), endpoint_native_max=supplied.max(axis=0),
            center_native_min=centers.min(axis=0), center_native_max=centers.max(axis=0),
            endpoint_delta_local_min=endpoint_local.min(axis=0),
            endpoint_delta_local_max=endpoint_local.max(axis=0),
            center_delta_local_min=center_local.min(axis=0), center_delta_local_max=center_local.max(axis=0),
            length_native_min=float(lengths.min()), length_native_max=float(lengths.max()),
            length_over_baseline_min=float(lengths.min() / baseline),
            length_over_baseline_max=float(lengths.max() / baseline),
            angle_from_control_degrees_min=float(angles.min()),
            angle_from_control_degrees_max=float(angles.max()),
            maximum_pairwise_direction_angle_degrees=float(pair_angles.max()),
            maximum_fractional_displacement_over_baseline=float(np.linalg.norm(delta, axis=2).max() / baseline),
            observations=observations,
            box_semantics="Observed same-fraction positions in a fixed control frame; not a distance tube or confidence region",
        ))
    return result


def _gap_envelopes(control, conditions, baseline):
    result = []
    for left, right in zip(control["order"][:-1], control["order"][1:]):
        observations = []
        for slot in conditions:
            segments = np.asarray(slot["matched_segments"])
            coordinates = (segments - control["origin"]) @ control["axis"]
            left_end = segments[left, int(np.argmax(coordinates[left]))]
            right_start = segments[right, int(np.argmin(coordinates[right]))]
            gap = float((right_start - left_end) @ control["axis"])
            observations.append(dict(condition_id=slot["condition_id"], left_endpoint_native=left_end,
                                     right_endpoint_native=right_start, gap_along_control_axis_native=gap,
                                     gap_along_control_axis_over_baseline=gap / baseline,
                                     endpoint_distance_native=float(np.linalg.norm(right_start - left_end))))
        values = [row["gap_along_control_axis_native"] for row in observations]
        result.append(dict(left_control_segment_index=int(left), right_control_segment_index=int(right),
                           gap_along_control_axis_native_min=min(values),
                           gap_along_control_axis_native_max=max(values),
                           gap_along_control_axis_over_baseline_min=min(values) / baseline,
                           gap_along_control_axis_over_baseline_max=max(values) / baseline,
                           observations=observations,
                           scope="Predicted finite gap only; no proof of physical absence"))
    return result


def build_envelope(control, perturbations, *, initial_extrinsics,
                   expected_condition_ids=DEFAULT_CONDITIONS, samples_per_segment=33):
    """Describe control plus exactly four planned slots; missing slots remain visible.

    Structure epsilon is 1e-6 * the original native baseline. It only identifies
    numerically degenerate geometry and never declares a narrow range reliable.
    The control's original segment order and endpoint orientation are preserved.
    """
    expected = tuple(expected_condition_ids)
    if (len(expected) != 4 or len(set(expected)) != 4 or "control" in expected
            or any(not isinstance(name, str) or not name for name in expected)):
        raise ValueError("Exactly four distinct non-control condition IDs required")
    if isinstance(samples_per_segment, bool) or not isinstance(samples_per_segment, int) or samples_per_segment < 2:
        raise ValueError("At least two integer fractional samples required")
    control, perturbations = _plain(control), _plain(perturbations)
    if not isinstance(control, dict) or control.get("condition_id") != "control":
        raise ValueError("One explicitly named control packet required")
    by_id = {}
    for packet in perturbations:
        if not isinstance(packet, dict) or packet.get("condition_id") not in expected:
            raise ValueError("Unexpected perturbation condition")
        name = packet["condition_id"]
        if name in by_id:
            raise ValueError("Duplicate perturbation condition")
        by_id[name] = packet
    initial = np.asarray(initial_extrinsics, float)
    baseline = check_native_gauge(initial, initial)["baseline"]
    eps = STRUCTURE_EPS_OVER_BASELINE * baseline
    packets = [control] + [by_id.get(name) for name in expected]
    names = ["control", *expected]
    prepared = [_slot(packet, name, initial, eps) for packet, name in zip(packets, names)]
    slots, geometries = [p[0] for p in prepared], [p[1] for p in prepared]
    reference = geometries[0]
    control_available = not slots[0]["reasons"]
    if control_available:
        slots[0].update(state="comparable", matched_segments=reference["segments"],
                        matched_source_segment_indices=list(range(len(reference["segments"]))))
        for slot, geometry in zip(slots[1:], geometries[1:]):
            if slot["support"] is not None:
                for key in (*_SUPPORT_HASHES, "source_row_count"):
                    if slot["support"][key] != slots[0]["support"][key]:
                        slot["reasons"].append("support_changed:" + key)
            if not slot["reasons"]:
                matched, indices, reason = _match(reference, geometry, eps)
                if reason:
                    slot["reasons"].append(reason)
                else:
                    slot.update(state="comparable", matched_segments=matched,
                                matched_source_segment_indices=indices)
    else:
        slots[0]["state"] = "unavailable"
        for slot in slots[1:]:
            slot["reasons"].append("control_unavailable")
    complete = all(slot["state"] == "comparable" for slot in slots)
    frame = _frame(reference["axis"]) if reference is not None else None
    output = dict(
        state="complete_empirical_envelope" if complete else "unresolved" if control_available else "unavailable",
        reasons=[slot["condition_id"] + ":" + reason for slot in slots for reason in slot["reasons"]],
        baseline=baseline, structure_epsilon_native=eps,
        expected_condition_ids=list(expected), expected_condition_count=5,
        comparable_condition_count=sum(slot["state"] == "comparable" for slot in slots),
        input_sha256=_digest(dict(packets=packets, initial_extrinsics=initial,
                                 expected_condition_ids=expected, samples_per_segment=samples_per_segment)),
        control_segments=control["identity"].get("segments"), local_frame=frame,
        local_frame_convention="Rows are control axis, first perpendicular, second perpendicular; delta_local=delta_native @ local_frame.T",
        conditions=slots, samples_per_segment=samples_per_segment,
        segment_envelopes=_segment_envelopes(reference, slots, frame, baseline, samples_per_segment) if complete else None,
        gap_envelopes=_gap_envelopes(reference, slots, baseline) if complete else None,
        control_modified=False, alignment_performed=False, perturbation_selected=False,
        scope="Complete means all planned perturbations are comparable, not reliable or accurate. Empirical ranges have no confidence or coverage guarantee.",
    )
    plain = _plain(output)
    json.dumps(plain, allow_nan=False)
    return plain
