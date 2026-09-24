"""Replay fixed camera families and test empirical rod envelopes; no accuracy gate."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from audit_camera_training_sensitivity import describe_identity
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import ROOT, camera_score, digest, read_json, write_json
from run_camera_training_sensitivity import checked as checked_parent
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open
from scipy.optimize import linear_sum_assignment

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import (
    correction_decision,
    project,
    triangulate,
    validate_heldout,
    validation_plan,
)
from creator_eval.camera_bundle_shared import fit_shared
from creator_eval.camera_envelope_challenges import generate_challenges
from creator_eval.camera_training_groups import build_training_group_plan, select_training_tracks
from creator_eval.line_controls import curve_metrics, fit_line_tls
from creator_eval.rod_camera_envelope import build_envelope

RUNS = {'replay': 'rod-camera-envelope-v1-20260924', 'analytic': 'camera-envelope-challenges-v1-20260924'}
NEW_SOURCES = ['scripts/run_rod_camera_envelope.py', 'scripts/audit_camera_training_sensitivity.py',
    'experiments/src/creator_eval/rod_camera_envelope.py',
    'experiments/src/creator_eval/camera_envelope_challenges.py',
    'configs/camera_envelope_challenges_v1.json']
EVALUATION = dict(alignment=dict(minimum_second_to_first_singular_ratio=.001,
    poor_fit_rmse_over_camera_span=.05, leave_one_camera_out_diagnostic=True),
    curve_tolerance_m=.025, curve_spacing_m=.002,
    alignment_policy='One camera-centers-only Sim3 from the full control, reused unchanged for all folds',
    box_policy='Same-fraction three-coordinate containment in the control local frame; no physical tube or confidence claim')


def locations(mode):
    rid = RUNS[mode]
    return ROOT / '.runtime/experiments' / rid, ROOT / 'data/inputs' / rid


def checked(mode):
    run, inputs = locations(mode)
    config = read_json(run / 'method_config.json')
    assert config['run_id'] == RUNS[mode] and not config['gt_read_during_inference']
    assert digest(inputs / 'manifest.json') == config['input_sha256']
    assert digest(run / 'source_freeze.json') == config['source_freeze_sha256']
    for name, sha in config['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(run / 'source_snapshot' / name), name
    for name, sha in config['receipts'].items():
        assert digest(ROOT / name) == sha, name
    return run, config, read_json(inputs / 'manifest.json')


def prepare(mode):
    run, inputs = locations(mode)
    truth_root = ROOT / 'data/eval_gt' / RUNS[mode]
    if run.exists() or inputs.exists() or truth_root.exists():
        raise FileExistsError('Keep old attempts; use a new run id')
    parent, parent_config, old_manifest = checked_parent()
    index = read_json(parent / 'inference.json')
    assert index['state'] == 'complete' and not index['gt_read_during_inference']
    assert index['config_sha256'] == digest(parent / 'method_config.json')
    assert index['input_sha256'] == parent_config['input_sha256']
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    (run / 'records').mkdir()
    # Freeze the complete implementation before producing the analytical input.
    sources = source_snapshot(run, set(parent_config['source_sha256']) | set(NEW_SOURCES))
    write_json(run / 'source_freeze.json', dict(source_sha256=sources, evaluation=EVALUATION,
        scope='No fitted results existed at source freeze time'))
    receipts = {p.relative_to(ROOT).as_posix(): digest(p) for p in
                [parent / 'method_config.json', parent / 'inference.json']}
    parent_input = ROOT / 'data/inputs' / parent_config['run_id'] / 'manifest.json'
    receipts[parent_input.relative_to(ROOT).as_posix()] = parent_config['input_sha256']
    for ancestor, names in parent_config['parents'].items():
        for name, sha in names.items():
            path = ROOT / '.runtime/experiments' / ancestor / name
            assert digest(path) == sha
            receipts[path.relative_to(ROOT).as_posix()] = sha
    tasks = []
    if mode == 'replay':
        sys.addaudithook(reject_truth_open)
        expected = {(t['parent'], t['case_id'], c['condition_id']) for t in old_manifest['tasks']
                    for c in t['group_plan']['conditions']}
        actual = [(r['parent'], r['case_id'], r['condition']) for r in index['records']]
        assert len(actual) == len(expected) and set(actual) == expected
        for task in old_manifest['tasks']:
            pool = ROOT / task['pool_path']
            assert digest(pool) == task['pool_sha256']
            receipts[task['pool_path']] = task['pool_sha256']
            frames = read_json(pool)['frames']
            packets = {name: [] for name in ('baseline', 'cylinder_support')}
            for entry in index['records']:
                if (entry['parent'], entry['case_id']) != (task['parent'], task['case_id']):
                    continue
                path = parent / entry['path']
                assert digest(path) == entry['sha256']
                receipts[path.relative_to(ROOT).as_posix()] = entry['sha256']
                record = read_json(path)
                assert record['task_sha256'] == canonical_hash(task)
                assert record['condition']['condition_id'] == entry['condition']
                assert not record['gt_read_during_inference']
                for name in packets:
                    rod = next((r for r in record['rods'] if r['method'] == name), None)
                    selected = describe_identity(rod, task, frames)['selected'] if rod else None
                    support = None if selected is None else dict(
                        assignment_sha256=canonical_hash({k: selected[k] for k in ('hypothesis_assignments', 'finite_assignments')}),
                        source_rows_sha256=selected['source_rows_sha256'], source_row_count=selected['source_row_count'],
                        pool_sha256=task['pool_sha256'])
                    packets[name].append(dict(condition_id=entry['condition'],
                        intrinsics=record['result'].get('intrinsics'), extrinsics=record['result'].get('extrinsics'),
                        camera_decision=record['decision'], identity=rod['identity'] if rod else dict(state='missing', segments=[]),
                        support=support, parent_record_sha256=entry['sha256']))
            for name, family in packets.items():
                tasks.append(dict(task_id=task['parent'] + '-' + task['case_id'] + '-' + name,
                    parent=task['parent'], case_id=task['case_id'], method=name,
                    initial_extrinsics=task['initial_extrinsics'], packets=family))
        manifest = dict(tasks=tasks, scope='Fourteen families of already known DA3-initialized rods; no new images or fits')
    else:
        protocol = read_json(ROOT / 'configs/camera_envelope_challenges_v1.json')
        generated, truth = generate_challenges(protocol)
        method = old_manifest['tasks'][0]['method']
        variant = old_manifest['tasks'][0]['variant']
        for case in generated['cases']:
            case['group_plan'] = build_training_group_plan(case['training'],
                validation_track_ids=[t['track_id'] for t in case['validation']], view_count=len(case['view_ids']),
                minimum_tracks=method['optimizer']['minimum_initial_tracks'],
                minimum_tracks_per_view=method['optimizer']['minimum_trimmed_tracks_per_view'])
            case['original_plan'] = validation_plan(case['validation'], case['initial_extrinsics'])
            case['all_view_plan'] = all_view_plan(case['validation'], case['initial_extrinsics'],
                training_track_ids=[t['track_id'] for t in case['training']])
        manifest = dict(**generated, method=method, variant=variant)
        truth_root.mkdir(parents=True)
        write_json(truth_root / 'manifest.json', truth)
    write_json(inputs / 'manifest.json', manifest)
    if mode == 'analytic':
        write_json(truth_root / 'frozen.json', dict(truth_sha256=digest(truth_root / 'manifest.json'),
            input_sha256=digest(inputs / 'manifest.json'), source_freeze_sha256=digest(run / 'source_freeze.json')))
    write_json(run / 'method_config.json', dict(run_id=RUNS[mode], mode=mode,
        source_sha256=sources, input_sha256=digest(inputs / 'manifest.json'), receipts=receipts,
        source_freeze_sha256=digest(run / 'source_freeze.json'), evaluation=EVALUATION,
        gt_read_during_inference=False, no_automatic_promotion=True))
    print('PREPARED', mode, len(tasks) if mode == 'replay' else len(manifest['cases']), flush=True)


def analytical_rod(case, k, e):
    points, pixel_residuals = [], []
    try:
        for segment in case['rod_tracks']:
            for endpoint in segment['endpoint_tracks']:
                obs = endpoint['observations']
                if len({o['view'] for o in obs}) < 2:
                    raise ValueError('fewer_than_two_endpoint_views')
                point = triangulate(obs, k, e)
                if not np.isfinite(point).all():
                    raise ValueError('nonfinite_triangulation')
                for observation in obs:
                    view = observation['view']
                    uv, z = project(point[None], k[view], e[view])
                    if z[0] <= 0 or not np.isfinite(uv).all():
                        raise ValueError('endpoint_behind_camera')
                    pixel_residuals.append(float(np.linalg.norm(uv[0] - observation['xy'])))
                points.append(point)
        tls = fit_line_tls(points, min_points=2)
        # This component test supplies endpoint identities and a common straight
        # axis. It tests camera propagation, not discovery of rods in photographs.
        points = np.asarray(points)
        fitted = tls['centroid'] + ((points - tls['centroid']) @ tls['direction'])[:, None] * tls['direction']
        segments = fitted.reshape(-1, 2, 3)
        if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1e-12):
            raise ValueError('zero_length_segment')
        return dict(state='accepted', reason='provided_endpoint_correspondences_common_axis', segments=segments,
            triangulated_endpoints=points, endpoint_reprojection_errors_px=pixel_residuals,
            line_fit_perpendicular_rmse=tls['perpendicular_rmse'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return dict(state='unavailable', reason=str(error), segments=[], endpoint_reprojection_errors_px=pixel_residuals)


def analytical_worker(case, condition, method, variant):
    sys.addaudithook(reject_truth_open)
    run, _, _ = checked('analytic')
    k, e = np.asarray(case['initial_intrinsics']), np.asarray(case['initial_extrinsics'])
    before = validate_heldout(k, e, case['original_plan'])
    if condition['state'] == 'eligible':
        train = select_training_tracks(case['training'], case['group_plan'], condition['condition_id'])
        try:
            result = fit_shared(k, e, train, method['optimizer'], variant)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            result = dict(state='numerical_failure', success=False, error=repr(error))
    else:
        result = dict(state='skipped_group_coverage', success=False)
    after, all_views = None, None
    identity = dict(state='unavailable', reason='no_camera', segments=[])
    if 'extrinsics' in result:
        ck, ce = np.asarray(result['intrinsics']), np.asarray(result['extrinsics'])
        after = validate_heldout(ck, ce, case['original_plan'])
        all_views = score_all_views(ck, ce, case['all_view_plan'])
        identity = analytical_rod(case, ck, ce)
        kept, discarded = set(result['kept_track_ids']), set(result['discarded_track_ids'])
        assert not kept & discarded and kept | discarded == set(condition['kept_track_ids'])
    decision = correction_decision(result, before, after or before, method['decision'])
    support = None
    if identity['state'] == 'accepted':
        sha = canonical_hash(case['rod_tracks'])
        assert sha == case['rod_support_sha256']
        support = dict(assignment_sha256=sha, source_rows_sha256=sha, pool_sha256=sha,
            source_row_count=sum(len(t['observations']) for s in case['rod_tracks'] for t in s['endpoint_tracks']))
    packet = dict(condition_id=condition['condition_id'], intrinsics=result.get('intrinsics'),
        extrinsics=result.get('extrinsics'), camera_decision=decision, identity=identity, support=support)
    path = run / 'records' / (case['case_id'] + '-' + condition['condition_id'] + '.json')
    write_json(path, dict(case_id=case['case_id'], condition=condition, case_sha256=canonical_hash(case),
        packet=packet, result=result, before=before, after=after, all_views=all_views, gt_read_during_inference=False))
    print('FIT', path.stem, result['state'], decision['state'], identity['state'], flush=True)
    return dict(case_id=case['case_id'], condition=condition['condition_id'], path=path.relative_to(run).as_posix(), sha256=digest(path))


def infer(mode):
    sys.addaudithook(reject_truth_open)
    run, config, manifest = checked(mode)
    if (run / 'inference.json').exists():
        raise FileExistsError('Keep frozen inference')
    started = time.perf_counter()
    records, families = [], []
    if mode == 'analytic':
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(analytical_worker, c, condition, manifest['method'], manifest['variant'])
                for c in manifest['cases'] for condition in c['group_plan']['conditions']]
            records = [future.result() for future in as_completed(futures)]
        tasks = []
        for case in manifest['cases']:
            packets = []
            for record in sorted(records, key=lambda r: r['condition']):
                if record['case_id'] == case['case_id']:
                    assert digest(run / record['path']) == record['sha256']
                    packets.append(read_json(run / record['path'])['packet'])
            tasks.append(dict(task_id=case['case_id'], case_id=case['case_id'], method='provided_endpoints',
                initial_extrinsics=case['initial_extrinsics'], packets=packets))
    else:
        tasks = manifest['tasks']
    for task in tasks:
        control = next(p for p in task['packets'] if p['condition_id'] == 'control')
        perturbations = [p for p in task['packets'] if p['condition_id'] != 'control']
        envelope = build_envelope(control, perturbations, initial_extrinsics=task['initial_extrinsics'])
        path = run / 'records' / (task['task_id'] + '-envelope.json')
        write_json(path, dict(task=task, envelope=envelope, gt_read_during_inference=False))
        families.append(dict(task_id=task['task_id'], case_id=task['case_id'], path=path.relative_to(run).as_posix(), sha256=digest(path)))
    checked(mode)
    write_json(run / 'inference.json', dict(state='complete', records=sorted(records, key=lambda r: (r['case_id'], r['condition'])),
        families=families, config_sha256=digest(run / 'method_config.json'), input_sha256=config['input_sha256'],
        elapsed_seconds=time.perf_counter() - started, gt_read_during_inference=False))
    print('INFERRED', mode, len(families), 'families', flush=True)


def score_family(envelope, packets, truth_segments, truth_cameras, policy):
    """Evaluation only: one control camera alignment for the complete family."""
    control = next(p for p in packets if p['condition_id'] == 'control')
    by_id = {p['condition_id']: p for p in packets}
    expected_ids = [slot['condition_id'] for slot in envelope['conditions']]
    assert len(by_id) == len(packets) and set(by_id) <= set(expected_ids)
    def unscored_rows(reason):
        return [dict(condition_id=cid, metrics=None, reason=reason if cid in by_id else 'missing_condition')
                for cid in expected_ids]
    if control['extrinsics'] is None:
        return dict(state='no_control_camera', alignment=None, rows=unscored_rows('no_control_camera'), coverage=None)
    try:
        score = camera_score(np.asarray(control['intrinsics']), np.asarray(control['extrinsics']), truth_cameras, policy['alignment'])
    except ValueError as error:
        return dict(state='alignment_unavailable', reason=str(error), alignment=None,
                    rows=unscored_rows('alignment_unavailable'), coverage=None)
    matrix = np.asarray(score['prediction_world_to_gt_world'])
    truth = np.asarray(truth_segments).reshape(-1, 2, 3)
    rows = []
    for cid in expected_ids:
        packet = by_id.get(cid)
        if packet is None:
            rows.append(dict(condition_id=cid, metrics=None, reason='missing_condition'))
            continue
        segments = np.asarray(packet['identity'].get('segments', [])).reshape(-1, 2, 3)
        aligned = segments @ matrix[:3, :3].T + matrix[:3, 3]
        rows.append(dict(condition_id=packet['condition_id'], identity_state=packet['identity']['state'],
            camera_decision=packet['camera_decision']['state'],
            metrics=curve_metrics(aligned, truth, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m'])))
    coverage = None
    if envelope['state'] == 'complete_empirical_envelope':
        ref = np.asarray(envelope['control_segments'])
        # GT matching is only for this diagnostic, never fed to inference. Report
        # topology mismatch instead of squeezing two truth segments into one box.
        if len(ref) != len(truth):
            coverage = dict(state='segment_count_mismatch', fraction=None)
        else:
            native_truth = (truth - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T
            cost = np.empty((len(ref), len(truth)))
            for i, a in enumerate(ref):
                for j, b in enumerate(native_truth):
                    cost[i, j] = min(np.linalg.norm(a - b), np.linalg.norm(a - b[::-1]))
            indices, assignment = linear_sum_assignment(cost)
            assert np.array_equal(indices, np.arange(len(ref)))
            parts = []
            scale = float(score['scale'])
            for box, j in zip(envelope['segment_envelopes'], assignment):
                i = box['control_segment_index']
                b = native_truth[j]
                if np.linalg.norm(ref[i] - b[::-1]) < np.linalg.norm(ref[i] - b):
                    b = b[::-1]
                fractions = np.asarray(box['fractions'])
                target = b[0] + fractions[:, None] * (b[1] - b[0])
                delta = (target - np.asarray(box['control_points'])) @ np.asarray(box['local_frame']).T
                low, high = np.asarray(box['min_delta_local']), np.asarray(box['max_delta_local'])
                epsilon = 64 * np.finfo(float).eps * max(1., envelope['baseline'], float(abs(delta).max()))
                inside = ((delta >= low - epsilon) & (delta <= high + epsilon)).all(axis=1)
                outside = np.maximum(low - delta, 0) + np.maximum(delta - high, 0)
                parts.append(dict(control_segment_index=i, truth_segment_index=int(j), covered_count=int(inside.sum()),
                    sample_count=len(inside), fraction=float(inside.mean()), max_outside_box_m=float(np.linalg.norm(outside, axis=1).max() * scale),
                    maximum_box_diagonal_m=float(np.linalg.norm(high - low, axis=1).max() * scale),
                    maximum_pairwise_direction_angle_degrees=box['maximum_pairwise_direction_angle_degrees']))
            coverage = dict(state='scored', fraction=sum(p['covered_count'] for p in parts) / sum(p['sample_count'] for p in parts),
                segments=parts, matching='Evaluation-only minimum endpoint cost bijection; orientation chosen against control',
                scope='33 same-fraction samples per segment, equal segment weight (not arc length); correlated deterministic conditions, no probabilistic coverage guarantee')
    return dict(state='scored', alignment=score, alignment_reused_for_all_conditions=True, rows=rows, coverage=coverage)



def validate_analytic_record(record, receipt, case, method):
    """Reject a relabelled fold or a packet detached from its numerical fit."""
    assert record['case_id'] == receipt['case_id'] == case['case_id']
    assert record['case_sha256'] == canonical_hash(case) and not record['gt_read_during_inference']
    condition = next(c for c in case['group_plan']['conditions'] if c['condition_id'] == receipt['condition'])
    assert record['condition'] == condition
    packet, result = record['packet'], record['result']
    assert packet['condition_id'] == condition['condition_id']
    assert packet['intrinsics'] == result.get('intrinsics') and packet['extrinsics'] == result.get('extrinsics')
    before = validate_heldout(case['initial_intrinsics'], case['initial_extrinsics'], case['original_plan'])
    assert record['before'] == before
    after = validate_heldout(packet['intrinsics'], packet['extrinsics'], case['original_plan']) if 'extrinsics' in result else None
    assert record['after'] == after
    decision = correction_decision(result, before, after or before, method['decision'])
    assert packet['camera_decision'] == decision
    if 'extrinsics' in result:
        assert record['all_views'] == score_all_views(packet['intrinsics'], packet['extrinsics'], case['all_view_plan'])
        kept, discarded = set(result['kept_track_ids']), set(result['discarded_track_ids'])
        assert not kept & discarded and kept | discarded == set(condition['kept_track_ids'])
        identity = analytical_rod(case, np.asarray(packet['intrinsics']), np.asarray(packet['extrinsics']))
        assert canonical_hash(identity) == canonical_hash(packet['identity'])
    else:
        assert packet['identity'] == dict(state='unavailable', reason='no_camera', segments=[])
    if packet['identity']['state'] == 'accepted':
        sha = canonical_hash(case['rod_tracks'])
        assert sha == case['rod_support_sha256']
        assert packet['support'] == dict(assignment_sha256=sha, source_rows_sha256=sha, pool_sha256=sha,
            source_row_count=sum(len(t['observations']) for s in case['rod_tracks'] for t in s['endpoint_tracks']))
    else:
        assert packet['support'] is None


def evaluate(mode):
    run, config, manifest = checked(mode)
    index = read_json(run / 'inference.json')
    assert index['state'] == 'complete' and not index['gt_read_during_inference']
    assert index['config_sha256'] == digest(run / 'method_config.json') and index['input_sha256'] == config['input_sha256']
    expected_tasks = [t['task_id'] for t in manifest['tasks']] if mode == 'replay' else [c['case_id'] for c in manifest['cases']]
    assert sorted(r['task_id'] for r in index['families']) == sorted(expected_tasks)
    truth_receipts, truths = {}, {}
    if mode == 'analytic':
        truth_root = ROOT / 'data/eval_gt' / RUNS[mode]
        frozen = read_json(truth_root / 'frozen.json')
        assert digest(truth_root / 'manifest.json') == frozen['truth_sha256']
        assert digest(run / 'source_freeze.json') == frozen['source_freeze_sha256']
        assert frozen['input_sha256'] == config['input_sha256']
        truth_receipts[str(truth_root.relative_to(ROOT))] = frozen
        truth_manifest = read_json(truth_root / 'manifest.json')
        assert truth_manifest['dataset_id'] == manifest['dataset_id']
        assert truth_manifest['schema_version'] == manifest['schema_version']
        truths = {c['case_id']: c for c in truth_manifest['cases']}
        assert len(truths) == len(truth_manifest['cases']) == len(expected_tasks) == len(set(expected_tasks))
        assert set(truths) == set(expected_tasks)
        for case in manifest['cases']:
            assert case['view_ids'] == [camera['view_id'] for camera in truths[case['case_id']]['cameras']]
            assert all(camera['size_wh'] == case['size_wh'] for camera in truths[case['case_id']]['cameras'])
        expected = {(c['case_id'], condition['condition_id']) for c in manifest['cases'] for condition in c['group_plan']['conditions']}
        actual = [(r['case_id'], r['condition']) for r in index['records']]
        assert len(actual) == len(expected) and set(actual) == expected
        for receipt in index['records']:
            assert digest(run / receipt['path']) == receipt['sha256']
            record = read_json(run / receipt['path'])
            case = next(c for c in manifest['cases'] if c['case_id'] == receipt['case_id'])
            validate_analytic_record(record, receipt, case, manifest['method'])
    rows = []
    for receipt in index['families']:
        assert digest(run / receipt['path']) == receipt['sha256']
        family = read_json(run / receipt['path'])
        task, envelope = family['task'], family['envelope']
        assert task['task_id'] == receipt['task_id'] and task['case_id'] == receipt['case_id']
        assert not family['gt_read_during_inference']
        # Recompute the reader under the same frozen inputs before consulting GT.
        control = next(p for p in task['packets'] if p['condition_id'] == 'control')
        assert envelope == build_envelope(control, [p for p in task['packets'] if p['condition_id'] != 'control'],
            initial_extrinsics=task['initial_extrinsics'])
        if mode == 'replay':
            original_task = next(t for t in manifest['tasks'] if t['task_id'] == task['task_id'])
            assert task == original_task
            parent = ROOT / '.runtime/experiments' / task['parent']
            parent_frozen = read_json(parent / 'prepared.json')
            assert digest(parent / 'protocol.json') == parent_frozen['protocol_sha256']
            parent_protocol = read_json(parent / 'protocol.json')
            scene_run, _, truth_root, scene, _ = checked_scene(parent_protocol['run_id'])
            assert digest(scene_run / 'prepared.json') == parent_frozen['scene_prepared_sha256']
            assert digest(truth_root / 'manifest.json') == scene['truth_sha256']
            truth_manifest = read_json(truth_root / 'manifest.json')
            assert truth_manifest['input_sha256'] == scene['input_sha256'] == parent_frozen['scene_input_sha256']
            truth = next(c for c in truth_manifest['cases'] if c['case_id'] == task['case_id'])
            target = truth['declared']['target']
            segments = target.get('segments', [target['endpoints']])
            truth_receipts[task['parent']] = dict(scene_run_id=parent_protocol['run_id'], scene_prepared_sha256=digest(scene_run / 'prepared.json'),
                truth_sha256=scene['truth_sha256'], scene_input_sha256=scene['input_sha256'])
        else:
            assert task['task_id'] == task['case_id'] == receipt['task_id'] == receipt['case_id']
            assert task['method'] == 'provided_endpoints'
            truth = truths[task['case_id']]
            segments = truth['rod_segments']
            case = next(c for c in manifest['cases'] if c['case_id'] == task['case_id'])
            assert task['initial_extrinsics'] == case['initial_extrinsics']
            assert sorted(p['condition_id'] for p in task['packets']) == sorted(c['condition_id'] for c in case['group_plan']['conditions'])
            for packet in task['packets']:
                source = next(r for r in index['records'] if (r['case_id'], r['condition']) == (task['case_id'], packet['condition_id']))
                assert packet == read_json(run / source['path'])['packet']
        physical = score_family(envelope, task['packets'], segments, truth['cameras'], config['evaluation'])
        rows.append(dict(task_id=task['task_id'], case_id=task['case_id'], parent=task.get('parent'), method=task['method'],
            family=truth.get('family'), intended_role=truth.get('intended_role'), expected_limit=truth.get('expected_limit'),
            envelope=envelope, physical=physical))
    checked(mode)
    result = dict(state='complete', run_id=RUNS[mode], mode=mode, rows=rows, config=config,
        truth_receipts=truth_receipts, inference_sha256=digest(run / 'inference.json'),
        elapsed_seconds=index['elapsed_seconds'], gt_read_during_inference=False, no_automatic_promotion=True,
        scope='Empirical camera training perturbation ranges. Complete range is not a confidence interval or a physical success claim; synthetic endpoints are supplied.')
    destination = ROOT / 'data/evaluation' / RUNS[mode]
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', result)
    write_json(ROOT / 'docs/experiments/results' / ('2026-09-24-' + mode + '-rod-camera-envelope.json'), result)
    print('EVALUATED', mode, len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer', 'evaluate'])
    parser.add_argument('mode', choices=list(RUNS))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[args.stage](args.mode)
