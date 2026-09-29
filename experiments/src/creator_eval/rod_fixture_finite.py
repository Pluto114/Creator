"""Reuse frozen RGB finite-line rules with separately measured fixture cameras.

The strip is a declared task input, not an automatically identified target.
Missing rows stay unknown. This produces finite curves, never a repaired cloud.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import numpy as np

from .rod_candidate_association import associate_multiview_lines_cached
from .rod_candidate_extent import bound_selected_candidate
from .rod_candidate_pool import enumerate_image_line_pool
from .rod_cylinder_gate import recheck_finite_cylinder
from .rod_cylinder_support import select_supported_cylinders
from .rod_multiview_candidates import _validated_views
from .rod_observations import extract_rod_observations

FROZEN_METHOD_SHA256 = '25951e76d832313d74e961b98c3c930cad91270b1751862b0438488bf9cada58'
NARROW_METHOD_SHA256 = 'b33ce471de3fbe6b7387d78494c7675108a68793fa9d68711a5eb6a1f4c3072e'
METHODS = ('baseline', 'cylinder_support')


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def method_policy(method):
    if canonical_hash(method) != FROZEN_METHOD_SHA256:
        raise ValueError('Use the predeclared unchanged reference finite method')
    return copy.deepcopy(method)


def narrow_method_policy(method):
    if canonical_hash(method) != NARROW_METHOD_SHA256:
        raise ValueError('Use the predeclared unresolved-narrow finite method')
    return copy.deepcopy(method)


def _frame_contract(frames):
    if len(frames) != 5 or len({f['view_id'] for f in frames}) != 5:
        raise ValueError('Exactly five distinct ordered views are required')
    for frame in frames:
        size = np.asarray(frame['size_wh'], float)
        guide = np.asarray(frame['guide_xyxy'], float)
        sha = frame['rgb_sha256']
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != np.floor(size)):
            raise ValueError('Finite positive integer size_wh required')
        if guide.shape != (2, 2) or not np.isfinite(guide).all() or guide[1, 1] <= guide[0, 1]:
            raise ValueError('Finite increasing-y search guide required')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Exact RGB source SHA required')


def _extract_fixture_evidence(frames, images, method, policy, narrow):
    """Scan all guide rows before looking at cameras; keep the entire raw pool."""
    method = policy(method)
    _frame_contract(frames)
    if len(images) != len(frames):
        raise ValueError('RGB images must follow the frozen view order')
    output = []
    for frame, image in zip(frames, images):
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or list(rgb.shape[1::-1]) != frame['size_wh']:
            raise ValueError('RGB shape differs from its declared frame')
        observations = extract_rod_observations(
            rgb,
            frame['guide_xyxy'],
            method['observation'],
            narrow_config=method['narrow_observation'] if narrow else None,
        )
        pool = enumerate_image_line_pool(observations, method['image_hypotheses'], require_refit_support=True)
        for candidate in pool['candidates']:
            matches = candidate['row_matches']
            ys = [observations['rows'][i]['y'] for i, _, _ in matches]
            if (len(matches) != candidate['support_rows'] or len({m[0] for m in matches}) != len(matches)
                    or len(matches) < method['image_hypotheses']['minimum_rows']
                    or np.ptp(ys) < method['image_hypotheses']['minimum_y_span']):
                raise ValueError('Final candidate support no longer matches raw rows')
        output.append(dict(**copy.deepcopy(frame), observations=observations, pool=pool,
            observation_sha256=canonical_hash(observations), pool_sha256=canonical_hash(pool),
            row_status_counts=dict(Counter(row['status'] for row in observations['rows'])),
            raw_row_count=len(observations['rows']), full_pool_count=len(pool['candidates']),
            retained_pool_count=min(len(pool['candidates']), method['candidate_cap'])))
    return json_ready(output)


def extract_fixture_evidence(frames, images, method):
    return _extract_fixture_evidence(frames, images, method, method_policy, False)


def extract_fixture_narrow_evidence(frames, images, method):
    return _extract_fixture_evidence(frames, images, method, narrow_method_policy, True)


def _export_diagnostics(result):
    # A point behind an unused view may have no finite projection. Keep its
    # slot and the exact field path; JSON null must never be mistaken for 0 px.
    missing = []
    def visit(value, path):
        if isinstance(value, dict):
            return {key: visit(item, path + '.' + key) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, path + '[' + str(i) + ']') for i, item in enumerate(value)]
        if isinstance(value, np.ndarray):
            return visit(value.tolist(), path)
        if isinstance(value, np.generic):
            return visit(value.item(), path)
        if isinstance(value, float) and not np.isfinite(value):
            missing.append(path)
            return None
        return value
    clean = visit(result, '$')
    clean['nonfinite_diagnostic_fields'] = missing
    return clean


def _empty(method, reasons):
    return dict(method=method, state='withheld_camera', segments=[], segment_count=0,
        total_length_m=0., endpoints=[], rejection_reasons=reasons, proposal=None, finite=None)


def _reconstruct_fixture_finite(evidence_frames, cameras, method, policy, schema_version):
    """Fixed cameras and original selected row assignments, with two old methods.

    Both outputs use the same pool and association search. A rejected/ambiguous
    association is not forced into a single winning axis for this easier scene.
    """
    method = policy(method)
    _frame_contract(evidence_frames)
    if len(cameras) != 5 or [c['view_id'] for c in cameras] != [f['view_id'] for f in evidence_frames]:
        raise ValueError('Camera/frame view order differs')
    for frame in evidence_frames:
        if (canonical_hash(frame['observations']) != frame['observation_sha256']
                or canonical_hash(frame['pool']) != frame['pool_sha256']):
            raise ValueError('Raw RGB evidence or candidate pool changed')
    camera_gate = dict(passed=all(c.get('state') == 'validated' for c in cameras),
        required_view_count=5, views=[dict(view_id=c['view_id'], state=c.get('state')) for c in cameras],
        reason='all_five_fixture_cameras_validated' if all(c.get('state') == 'validated' for c in cameras)
        else 'one_or_more_fixture_cameras_not_validated')
    result = dict(schema_version=schema_version, camera_gate=camera_gate,
        camera_input_sha256=canonical_hash(cameras), evidence_input_sha256=canonical_hash(evidence_frames),
        method_sha256=canonical_hash(method), camera_modified=False, target_geometry_used=False,
        pool_budget=dict(candidate_cap=method['candidate_cap'],
            full_counts=[len(f['pool']['candidates']) for f in evidence_frames],
            retained_counts=[min(len(f['pool']['candidates']), method['candidate_cap']) for f in evidence_frames],
            cap_reached_views=[f['view_id'] for f in evidence_frames if len(f['pool']['candidates']) >= method['candidate_cap']],
            truncated_views=[f['view_id'] for f in evidence_frames if len(f['pool']['candidates']) > method['candidate_cap']],
            search_scope='Complete only over retained cap8 pools from the fixed 1024-sample image search; not all image explanations'),
        identity_scope='declared_single_target_strip_not_automatic_foreground_identity',
        coordinate_system='declared_metric_fixture_world', association=None, methods=[])
    if not camera_gate['passed']:
        reasons = [c['view_id'] + ':' + str(c.get('state')) for c in cameras if c.get('state') != 'validated']
        result['methods'] = [_empty(name, reasons) for name in METHODS]
        return _export_diagnostics(result)
    views = [dict(view_id=f['view_id'], rgb_sha256=f['rgb_sha256'], size_wh=f['size_wh'],
        K_index=c['K_index'], world_to_camera_cv=c['world_to_camera_cv'],
        y_range=[f['guide_xyxy'][0][1], f['guide_xyxy'][1][1]],
        candidates=f['pool']['candidates'][:method['candidate_cap']]) for f, c in zip(evidence_frames, cameras)]
    _validated_views(views)
    raw = [f['observations'] for f in evidence_frames]
    association = associate_multiview_lines_cached(views, method['association_common'])
    # cap=8 over five views needs at most 10*8^3=5120 seeds; the old 50000
    # budget covers this. A future budget mismatch must not masquerade as unique.
    if not association['search_complete']:
        raise ValueError('Frozen retained-pool association search did not complete')
    before = canonical_hash((views, raw, association))
    cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
    for name, proposal in (('baseline', association), ('cylinder_support', cylinder)):
        finite = bound_selected_candidate(proposal, views, raw, method['extent'])
        if name == 'cylinder_support':
            finite = recheck_finite_cylinder(finite, proposal.get('selected'), views, raw, method['cylinder_screen'])
        segments = np.asarray(finite['segments'], float).reshape(-1, 2, 3)
        if (finite['state'] != 'accepted' and len(segments)) or not np.isfinite(segments).all():
            raise ValueError('Nonaccepted/nonfinite geometry must not be emitted')
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        if np.any(lengths <= 0):
            raise ValueError('Finite output contains a nonpositive segment')
        result['methods'].append(dict(method=name, state=finite['state'], segments=segments,
            segment_count=len(segments), total_length_m=float(lengths.sum()),
            endpoints=segments.reshape(-1, 3), rejection_reasons=finite['rejection_reasons'],
            proposal=proposal, finite=finite))
    if canonical_hash((views, raw, association)) != before:
        raise ValueError('A method mutated the common camera/pixel/association input')
    result['association'] = association
    result = _export_diagnostics(result)
    json.dumps(result, allow_nan=False)
    return result


def reconstruct_fixture_finite(evidence_frames, cameras, method):
    return _reconstruct_fixture_finite(
        evidence_frames, cameras, method, method_policy, 'fixture-finite-rgb-v1'
    )


def reconstruct_fixture_narrow(evidence_frames, cameras, method):
    return _reconstruct_fixture_finite(
        evidence_frames,
        cameras,
        method,
        narrow_method_policy,
        'fixture-finite-narrow-rgb-v1',
    )
