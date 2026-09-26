"""Locate a new RGB view in a fixed estimated map, then score disjoint pixels.

No target clicks, depth truth, model inference, or old-map optimization occurs
here. Provenance receipts are supplied by the caller: the module validates their
contract, but cannot prove how the caller obtained the input pixels or map.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .background_correspondences import residual_summary
from .camera_bundle import project

DEFAULTS = dict(minimum_training_tracks=24, minimum_inlier_tracks=24,
    minimum_validation_tracks=8, reprojection_threshold_px=2.0,
    ransac_iterations=5000, ransac_confidence=.99, seed=113,
    minimum_3d_singular_ratio=.001, refine_max_iterations=50, refine_epsilon=1e-10,
    maximum_validation_p95_px=2.0, minimum_validation_fraction_within_2px=.8)
SOURCE_FIELDS = ('source_kind', 'rgb_sha256', 'map_sha256', 'intrinsics_sha256', 'size_wh', 'observation_role')


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value):
    """Same numerical JSON convention as the experiment runners."""
    return hashlib.sha256(json.dumps(_clean(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _policy(config):
    policy = {**DEFAULTS, **(config or {})}
    if policy != DEFAULTS:
        raise ValueError('Use the frozen held-out pose policy; do not tune it on new outcomes')
    return policy


def _array(value, tail_shape):
    try:
        result = np.asarray(value, float)
    except (TypeError, ValueError):
        return None
    if result.ndim != len(tail_shape) + 1 or result.shape[1:] != tail_shape or not np.isfinite(result).all():
        return None
    return np.ascontiguousarray(result)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _source(source, role, k):
    reasons, clean = [], {}
    source = source if isinstance(source, dict) else {}
    if source.get('source_kind') != 'rgb':
        reasons.append('source_must_be_rgb')
    clean['source_kind'] = source.get('source_kind') if isinstance(source.get('source_kind'), str) else None
    for key in ('rgb_sha256', 'map_sha256', 'intrinsics_sha256'):
        value = source.get(key)
        clean[key] = value if _sha(value) else None
        if not _sha(value):
            reasons.append('missing_or_invalid_' + key)
    size = _array([source.get('size_wh')], (2,))
    if size is None or np.any(size <= 0) or np.any(size != np.floor(size)):
        reasons.append('invalid_image_size')
        clean['size_wh'] = None
    else:
        clean['size_wh'] = size[0].astype(int).tolist()
    clean['observation_role'] = source.get('observation_role') if isinstance(source.get('observation_role'), str) else None
    if clean['observation_role'] != role:
        reasons.append('wrong_observation_role')
    if k is not None and clean['intrinsics_sha256'] is not None and canonical_hash(k) != clean['intrinsics_sha256']:
        reasons.append('intrinsics_receipt_mismatch')
    return clean, reasons


def _ids(track_ids):
    if not isinstance(track_ids, (list, tuple, np.ndarray)):
        return None, ['invalid_track_ids']
    values = []
    for value in track_ids:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            values.append(int(value))
        elif isinstance(value, str) and value:
            values.append(value)
        else:
            return None, ['invalid_track_ids']
    # An integer ID and its textual spelling should not count as two tracks.
    if len({str(value) for value in values}) != len(values):
        return values, ['duplicate_track_ids']
    return values, []


def _inputs(points3d, xy, K, track_ids, source, role):
    xyz, pixels = _array(points3d, (3,)), _array(xy, (2,))
    matrices = _array([K], (3, 3))
    k = matrices[0] if matrices is not None else None
    ids, reasons = _ids(track_ids)
    if xyz is None:
        reasons.append('invalid_or_nonfinite_map_points')
    if pixels is None:
        reasons.append('invalid_or_nonfinite_pixels')
    if k is None:
        reasons.append('missing_or_nonfinite_intrinsics')
    elif (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
          or abs(np.linalg.det(k)) <= 1e-12):
        reasons.append('invalid_pinhole_intrinsics')
    receipt, source_reasons = _source(source, role, k)
    reasons.extend(source_reasons)
    if xyz is not None and pixels is not None and ids is not None:
        if len(xyz) != len(pixels) or len(xyz) != len(ids):
            reasons.append('observation_count_mismatch')
        if receipt['size_wh'] is not None:
            size = np.asarray(receipt['size_wh'])
            if np.any(pixels < -.5) or np.any(pixels > size - .5):
                reasons.append('pixel_outside_image')
    return xyz, pixels, k, ids, receipt, list(dict.fromkeys(reasons))


def _rank(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    cutoff = np.finfo(float).eps * max(points.shape) * singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    ratio = float(singular[-1] / singular[0]) if len(singular) and singular[0] > 0 else None
    return dict(rank=rank, singular_values=singular.tolist(), last_to_first_ratio=ratio)


def _nonplanar(diagnostic, policy):
    return diagnostic['rank'] == 3 and diagnostic['last_to_first_ratio'] > policy['minimum_3d_singular_ratio']


def _scores(points, xy, k, e, track_ids):
    uv, depth = project(points, k, e)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
    errors = np.full(len(xy), np.nan)
    errors[valid] = np.linalg.norm(uv[valid] - xy[valid], axis=1)
    rows = [dict(track_id=tid, observed_xy=pixel.tolist(),
        projected_xy=prediction.tolist() if ok else None,
        depth_native=float(z) if np.isfinite(z) else None,
        error_px=float(error) if ok and np.isfinite(error) else None,
        reason=None if ok and np.isfinite(error) else 'nonfinite_projection_or_nonpositive_depth')
        for tid, pixel, prediction, z, ok, error in zip(track_ids, xy, uv, depth, valid, errors)]
    summary = residual_summary(errors)
    summary.update(maximum_px=float(np.nanmax(errors)) if np.isfinite(errors).any() else None,
                   negative_or_zero_depth_count=int(np.count_nonzero(depth <= 0)))
    return dict(summary=summary, rows=rows), valid & np.isfinite(errors), errors


def _seal(result):
    result['proposal_sha256'] = canonical_hash(result)
    return result


def propose_pose(points3d, xy, K, *, track_ids, source, config=None):
    """PnP RANSAC then inlier-only LM; fixed 3D points and K are never optimized.

    source fields: source_kind='rgb', rgb_sha256, map_sha256,
    intrinsics_sha256=canonical_hash(float K), size_wh, and
    observation_role='background_pose_training'. Numeric pose availability is
    separate from validation acceptance; planar/weak-depth support is unresolved.
    """
    import cv2

    policy = _policy(config)
    xyz, pixels, k, ids, receipt, reasons = _inputs(points3d, xy, K, track_ids, source, 'background_pose_training')
    result = dict(state='unresolved', reasons=reasons, E=None, K=k.tolist() if k is not None else None,
        source=receipt, config=policy, training_track_ids=ids, input_sha256=None,
        mapped_training_count=len(xyz) if xyz is not None else None, map_geometry=None, image_geometry=None,
        ransac_inlier_indices=None, ransac_inlier_track_ids=None, ransac_inlier_count=None,
        ransac_inlier_geometry=None, final_support_geometry=None, final_support_count=None,
        all_training=None, refined_ransac_inliers=None, refinement_returned=False, refinement_converged=None,
        map_modified=False, intrinsics_modified=False, target_observations_used=False,
        opencv_version=cv2.__version__,
        scope='Numerical pose in a fixed estimated RGB map; source contract is caller supplied, not independent metric accuracy')
    if reasons:
        return _seal(result)
    result['input_sha256'] = canonical_hash(dict(points3d=xyz, xy=pixels, K=k, track_ids=ids, source=receipt))
    if len(xyz) < policy['minimum_training_tracks']:
        reasons.append('insufficient_mapped_training_tracks')
        return _seal(result)
    try:
        result['map_geometry'], result['image_geometry'] = _rank(xyz), _rank(pixels)
        if not _nonplanar(result['map_geometry'], policy):
            reasons.append('planar_or_weak_depth_map_support')
        if result['image_geometry']['rank'] < 2:
            reasons.append('collinear_image_support')
        if reasons:
            return _seal(result)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(policy['seed'])
        success, rotation, translation, inliers = cv2.solvePnPRansac(xyz, pixels, k, None,
            iterationsCount=policy['ransac_iterations'], reprojectionError=policy['reprojection_threshold_px'],
            confidence=policy['ransac_confidence'], flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None:
            reasons.append('ransac_returned_no_pose')
            return _seal(result)
        selected = np.asarray(inliers).reshape(-1)
        if (not np.issubdtype(selected.dtype, np.integer) or len(set(selected.tolist())) != len(selected)
                or np.any(selected < 0) or np.any(selected >= len(xyz))):
            reasons.append('invalid_ransac_inlier_indices')
            return _seal(result)
        selected = np.sort(selected)
        result.update(ransac_inlier_indices=selected.tolist(), ransac_inlier_track_ids=[ids[i] for i in selected],
                      ransac_inlier_count=len(selected))
        if len(selected) < policy['minimum_inlier_tracks']:
            reasons.append('insufficient_ransac_inliers')
            return _seal(result)
        result['ransac_inlier_geometry'] = _rank(xyz[selected])
        if not _nonplanar(result['ransac_inlier_geometry'], policy):
            reasons.append('planar_or_weak_depth_ransac_inliers')
            return _seal(result)
        rotation, translation = cv2.solvePnPRefineLM(xyz[selected], pixels[selected], k, None,
            rotation, translation, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                policy['refine_max_iterations'], policy['refine_epsilon']))
        result['refinement_returned'] = True
        # OpenCV gives no convergence/iteration receipt for this routine. A
        # returned matrix is not permission to invent a convergence claim.
        r = cv2.Rodrigues(rotation)[0]
        e = np.c_[r, np.asarray(translation).reshape(3)]
        if (not np.isfinite(e).all() or not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)):
            reasons.append('nonfinite_or_improper_pose')
            return _seal(result)
        all_scores, _, _ = _scores(xyz, pixels, k, e, ids)
        inlier_scores, forward, errors = _scores(xyz[selected], pixels[selected], k, e, [ids[i] for i in selected])
        result.update(all_training=all_scores, refined_ransac_inliers=inlier_scores)
        if not forward.all():
            reasons.append('refined_inlier_cheirality_or_projection_failure')
        support = forward & (errors <= policy['reprojection_threshold_px'])
        result['final_support_count'] = int(support.sum())
        if support.sum() < policy['minimum_inlier_tracks']:
            reasons.append('insufficient_support_after_refinement')
        else:
            result['final_support_geometry'] = _rank(xyz[selected[support]])
            if not _nonplanar(result['final_support_geometry'], policy):
                reasons.append('planar_or_weak_depth_refined_support')
        if not reasons:
            result.update(state='fitted', E=e.tolist())
    except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        reasons.append('pose_numerical_failure:' + type(error).__name__)
    return _seal(result)


def verify_pose(proposal, points3d, xy, *, track_ids, source, config=None):
    """Only score unused background tracks; validation cannot alter numeric pose.

    source uses the same RGB/map/K receipts and size, with observation_role set
    to background_pose_validation. Track IDs must be disjoint from training.
    """
    policy = _policy(config)
    reasons = []
    output = dict(state='unresolved', reasons=reasons, proposal_sha256=None,
        input_sha256=None, source=None, validation_track_ids=None, validation=None,
        pose_modified=False, map_modified=False, threshold_px=policy['reprojection_threshold_px'],
        scope='Validation decision only; no new pose, map, focal fit, or model selection')
    try:
        seal = proposal.get('proposal_sha256')
        content = {key: value for key, value in proposal.items() if key != 'proposal_sha256'}
        if not _sha(seal) or canonical_hash(content) != seal:
            reasons.append('proposal_receipt_mismatch')
            return output
    except (ValueError, TypeError, AttributeError):
        reasons.append('invalid_proposal')
        return output
    output['proposal_sha256'] = seal
    if proposal.get('config') != policy:
        reasons.append('proposal_policy_mismatch')
        return output
    xyz, pixels, k, ids, receipt, errors = _inputs(points3d, xy, proposal.get('K'), track_ids, source,
                                                  'background_pose_validation')
    output.update(source=receipt, validation_track_ids=ids)
    reasons.extend(errors)
    for key in SOURCE_FIELDS:
        if key != 'observation_role' and receipt.get(key) != proposal['source'].get(key):
            reasons.append('training_validation_source_mismatch:' + key)
    if ids is not None and set(map(str, ids)) & set(map(str, proposal.get('training_track_ids') or [])):
        reasons.append('validation_tracks_overlap_training')
    if xyz is not None and len(xyz) < policy['minimum_validation_tracks']:
        reasons.append('insufficient_validation_tracks')
    e = _array([proposal.get('E')], (3, 4))
    if proposal.get('state') != 'fitted' or e is None:
        reasons.append('proposal_not_fitted')
    if reasons:
        output['reasons'] = list(dict.fromkeys(reasons))
        return output
    output['input_sha256'] = canonical_hash(dict(points3d=xyz, xy=pixels, track_ids=ids, source=receipt))
    scores, valid, _ = _scores(xyz, pixels, k, e[0], ids)
    output['validation'] = scores
    if not valid.all():
        reasons.append('validation_cheirality_or_projection_failure')
    stats = scores['summary']
    if (stats['p95_px'] is None or stats['p95_px'] > policy['maximum_validation_p95_px']
            or (stats['fraction_within_2px'] or 0.) < policy['minimum_validation_fraction_within_2px']):
        reasons.append('validation_pixel_budget_exceeded')
    output['state'] = 'validated' if not reasons else 'unresolved'
    # Validation may withhold the numerical pose, never repair it. The caller
    # keeps this decision alongside the sealed proposal and its original bytes.
    return copy.deepcopy(output)
