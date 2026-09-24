"""Frozen RGB-only training-group sensitivity; physical evaluation is a separate stage."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import (
    ROOT,
    camera_score,
    checked_run,
    digest,
    load_case,
    read_json,
    write_json,
)
from run_rod_candidate_ablation import canonical_hash
from run_rod_foreground_identity import compact_bundle, make_views
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import (
    centers,
    correction_decision,
    observations_from_record,
    validate_heldout,
    validation_plan,
)
from creator_eval.camera_bundle_shared import fit_shared
from creator_eval.camera_native_sensitivity import (
    camera_difference,
    check_native_gauge,
    segment_difference,
)
from creator_eval.camera_training_groups import build_training_group_plan
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.rod_candidate_association import associate_multiview_lines_cached
from creator_eval.rod_cylinder_support import select_supported_cylinders
from creator_eval.rod_foreground_identity import build_anchor_proposals, select_foreground_identity

RUN_ID = 'camera-training-sensitivity-v1-20260924r2'
PARENTS = ['rod-reference-bundle-v1-20260923', 'rod-height-bundle-v1-20260923']
SHARE = ROOT / 'docs/experiments/results/2026-09-24-camera-training-sensitivity-r2.json'
NEW_SOURCES = ['scripts/run_camera_training_sensitivity.py',
               'experiments/src/creator_eval/camera_training_groups.py',
               'experiments/src/creator_eval/camera_native_sensitivity.py',
               'experiments/src/creator_eval/camera_all_view_validation.py']


def prepare():
    sys.addaudithook(reject_truth_open)
    run = ROOT / '.runtime/experiments' / RUN_ID
    inputs = ROOT / 'data/inputs' / RUN_ID
    if run.exists() or inputs.exists():
        raise FileExistsError('Keep old attempts; choose another run ID')
    names, tasks, receipts = set(NEW_SOURCES), [], {}
    for parent_id in PARENTS:
        parent, _, frozen, manifest = checked_run(parent_id)
        names.update(frozen['source_sha256'])
        rod_index = read_json(parent / 'rod-inference.json')
        assert rod_index['state'] == 'complete' and not rod_index['gt_read_during_inference']
        assert rod_index['input_sha256'] == frozen['input_sha256'] and rod_index['source_sha256'] == frozen['source_sha256']
        assert rod_index['camera_inference_sha256'] == digest(parent / 'inference.json')
        assert rod_index['annotations_sha256'] == digest(parent / 'annotations.json')
        assert rod_index['pools_sha256'] == digest(parent / 'rod-pools.json')
        receipts[parent_id] = {n: digest(parent / n) for n in
                               ['prepared.json', 'predictions.json', 'inference.json', 'rod-inference.json', 'annotations.json', 'rod-pools.json']}
        for case in manifest['cases']:
            cid = case['case_id']
            _, _, tracks, method, k, e = load_case(parent_id, cid)
            train = observations_from_record(tracks, 'train')
            validation = observations_from_record(tracks, 'validation')
            plan = build_training_group_plan(train, validation_track_ids=[t['track_id'] for t in validation],
                view_count=len(k), minimum_tracks=method['optimizer']['minimum_initial_tracks'],
                minimum_tracks_per_view=method['optimizer']['minimum_trimmed_tracks_per_view'])
            old_path = parent / 'rod-results' / (cid + '-common_K_focal_trimmed.json')
            old_index = read_json(parent / 'rod-inference.json')
            old_receipt = next(r for r in old_index['records'] if r['case_id'] == cid and r['camera_variant'] == 'common_K_focal_trimmed')
            assert digest(old_path) == old_receipt['sha256']
            old = read_json(old_path)
            pool_path = parent / 'rod-pools' / (cid + '.json')
            assert digest(pool_path) == old['pool_sha256']
            variant = next(v for v in method['variants'] if v['id'] == 'common_K_focal_trimmed')
            tasks.append(dict(parent=parent_id, case_id=cid, training=train, group_plan=plan,
                original_plan=validation_plan(validation, e), all_view_plan=all_view_plan(validation, e,
                    training_track_ids=[t['track_id'] for t in train]),
                initial_intrinsics=k, initial_extrinsics=e, method=method, variant=variant,
                rod_method=manifest['rod_method'], pool_path=pool_path.relative_to(ROOT).as_posix(),
                pool_sha256=old['pool_sha256'], query=old['query'], old_rod_path=old_path.relative_to(ROOT).as_posix(),
                old_rod_sha256=digest(old_path), old_camera_receipt=old['camera_receipt'],
                view_ids=[f['view_id'] for f in case['frames']]))
    inputs.mkdir(parents=True)
    run.mkdir(parents=True)
    (run / 'records').mkdir()
    write_json(inputs / 'manifest.json', dict(tasks=tasks, parents=receipts))
    sources = source_snapshot(run, names)
    write_json(run / 'method_config.json', dict(run_id=RUN_ID, input_sha256=digest(inputs / 'manifest.json'),
        source_sha256=sources, parents=receipts,
        policy='Full training is the fixed control. Four leave-group-out fits are diagnostics, never selected or averaged into the control.',
        scope='Known seven cases, not independent objects or a confidence interval. No dense base or patch changed.',
        gt_read_during_inference=False))
    print('PREPARED', len(tasks), 'cases; all plans frozen before fitting', flush=True)


def checked():
    run = ROOT / '.runtime/experiments' / RUN_ID
    inputs = ROOT / 'data/inputs' / RUN_ID
    config = read_json(run / 'method_config.json')
    assert digest(inputs / 'manifest.json') == config['input_sha256']
    for name, sha in config['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(run / 'source_snapshot' / name), name
    for parent, receipts in config['parents'].items():
        for name, sha in receipts.items():
            assert digest(ROOT / '.runtime/experiments' / parent / name) == sha
    return run, config, read_json(inputs / 'manifest.json')


def rod_records(task, k, e):
    path = ROOT / task['pool_path']
    assert digest(path) == task['pool_sha256']
    frames = [{**f, 'K_index': ck, 'world_to_camera_cv': ce} for f, ck, ce in
              zip(read_json(path)['frames'], k, e)]
    assert [f['view_id'] for f in frames] == task['view_ids']
    method = task['rod_method']
    views = make_views(frames, method['candidate_cap'])
    raw = [f['observations'] for f in frames]
    before = canonical_hash(dict(views=views, raw=raw, query=task['query']))
    association = associate_multiview_lines_cached(views, method['association_common'])
    assert association['search_complete']
    cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
    records = []
    for name, proposal in [('baseline', association), ('cylinder_support', cylinder)]:
        bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method['extent'],
            method['cylinder_screen'] if name == 'cylinder_support' else None))
        bundle_hash = canonical_hash(bundle)
        identity = select_foreground_identity(bundle, views, raw, task['query']['anchors'], method['identity_policy'])
        assert canonical_hash(bundle) == bundle_hash
        records.append(dict(method=name, identity=identity, bundle=bundle))
    assert before == canonical_hash(dict(views=views, raw=raw, query=task['query']))
    return records


def worker(task, condition):
    sys.addaudithook(reject_truth_open)
    run, _, _ = checked()
    stem = task['parent'].replace('-bundle-v1-20260923', '') + '-' + task['case_id'] + '-' + condition['condition_id']
    path = run / 'records' / (stem + '.json')
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    # Preserve the original observation order. Changing a fold must not quietly
    # change the full-control optimizer's numerical path as well.
    kept = set(condition['kept_track_ids'])
    training = [t for t in task['training'] if t['track_id'] in kept]
    assert len(training) == len(kept)
    k, e = np.asarray(task['initial_intrinsics']), np.asarray(task['initial_extrinsics'])
    before = validate_heldout(k, e, task['original_plan'])
    if condition['state'] == 'eligible':
        try:
            fit = fit_shared(k, e, training, task['method']['optimizer'], task['variant'])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            fit = dict(state='numerical_failure', success=False, error=repr(error))
    else:
        fit = dict(state='skipped_group_coverage', success=False)
    after, all_views, gauge, rods = None, None, None, []
    if 'extrinsics' in fit:
        ck, ce = np.asarray(fit['intrinsics']), np.asarray(fit['extrinsics'])
        gauge = check_native_gauge(e, ce)
        assert gauge['comparable'], gauge
        actual_kept, discarded = set(fit['kept_track_ids']), set(fit['discarded_track_ids'])
        assert not actual_kept & discarded and actual_kept | discarded == kept
        after = validate_heldout(ck, ce, task['original_plan'])
        all_views = score_all_views(ck, ce, task['all_view_plan'])
        rods = rod_records(task, ck, ce)
    decision = correction_decision(fit, before, after if after is not None else before, task['method']['decision'])
    result = dict(parent=task['parent'], case_id=task['case_id'], condition=condition, result=fit,
        before=before, after=after, all_views=all_views, decision=decision, gauge=gauge, rods=rods,
        task_sha256=canonical_hash(task), pool_sha256=task['pool_sha256'], gt_read_during_inference=False,
        elapsed_seconds=time.perf_counter() - started)
    write_json(path, result)
    print('SENSITIVITY', stem, fit['state'], decision['state'], [r['identity']['state'] for r in rods], flush=True)
    return dict(parent=task['parent'], case_id=task['case_id'], condition=condition['condition_id'],
                path=path.relative_to(run).as_posix(), sha256=digest(path))


def infer():
    sys.addaudithook(reject_truth_open)
    run, config, manifest = checked()
    if (run / 'inference.json').exists():
        raise FileExistsError('Keep frozen inference')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, task, condition) for task in manifest['tasks']
                   for condition in task['group_plan']['conditions']]
        records = [f.result() for f in as_completed(futures)]
    checked()
    write_json(run / 'inference.json', dict(state='complete', records=sorted(records, key=lambda r: (r['parent'], r['case_id'], r['condition'])),
        config_sha256=digest(run / 'method_config.json'), input_sha256=config['input_sha256'],
        elapsed_seconds=time.perf_counter() - started, gt_read_during_inference=False))


def evaluate():
    run, config, manifest = checked()
    index = read_json(run / 'inference.json')
    assert index['config_sha256'] == digest(run / 'method_config.json')
    expected = {(t['parent'], t['case_id'], c['condition_id']) for t in manifest['tasks'] for c in t['group_plan']['conditions']}
    actual = [(r['parent'], r['case_id'], r['condition']) for r in index['records']]
    assert len(actual) == len(expected) and set(actual) == expected
    rows, comparisons, replays, camera_rows = [], [], [], []
    for task in manifest['tasks']:
        parent = ROOT / '.runtime/experiments' / task['parent']
        parent_frozen = read_json(parent / 'prepared.json')
        assert digest(parent / 'protocol.json') == parent_frozen['protocol_sha256']
        parent_config = read_json(parent / 'protocol.json')
        _, _, truth_root, scene, _ = checked_scene(parent_config['run_id'])
        assert digest(truth_root / 'manifest.json') == scene['truth_sha256']
        assert digest(truth_root / 'artifact_hashes.json') == scene['truth_artifacts_sha256']
        truth = next(c for c in read_json(truth_root / 'manifest.json')['cases'] if c['case_id'] == task['case_id'])
        records = []
        for receipt in index['records']:
            if (receipt['parent'], receipt['case_id']) != (task['parent'], task['case_id']):
                continue
            assert digest(run / receipt['path']) == receipt['sha256']
            record = read_json(run / receipt['path'])
            assert record['task_sha256'] == canonical_hash(task)
            assert (record['parent'], record['case_id']) == (task['parent'], task['case_id'])
            expected_condition = next(c for c in task['group_plan']['conditions'] if c['condition_id'] == receipt['condition'])
            assert record['condition'] == expected_condition
            assert not record['gt_read_during_inference']
            records.append(record)
            fit = record['result']
            numerical = 'extrinsics' in fit
            cameras = truth['cameras']
            score = camera_score(np.asarray(fit['intrinsics']), np.asarray(fit['extrinsics']), cameras,
                                 parent_config['evaluation']['alignment']) if numerical else None
            camera_rows.append(dict(parent=task['parent'], case_id=task['case_id'], condition=record['condition']['condition_id'],
                state=fit['state'], decision=record['decision'], camera=score, group=record['condition'], gauge=record['gauge'],
                original_residual=record['after']['summary'] if numerical else None,
                all_view_residual=record['all_views']['summary'] if numerical else None))
            matrix = np.asarray(score['prediction_world_to_gt_world']) if score else None
            target = truth['declared']['target']
            true_segments = target.get('segments', [target['endpoints']])
            policy = parent_config['evaluation']
            if numerical:
                assert len(record['rods']) == 2 and {r['method'] for r in record['rods']} == {'baseline', 'cylinder_support'}
            else:
                assert not record['rods']
                for method in ['baseline', 'cylinder_support']:
                    rows.append(dict(parent=task['parent'], case_id=task['case_id'], condition=record['condition']['condition_id'],
                        method=method, state='missing_camera', reason=fit['state'], camera_decision=record['decision'],
                        camera=None, original_residual=None, all_view_residual=None, metrics=None, gap=None))
            for rod in record['rods']:
                native = np.asarray(rod['identity']['segments']).reshape(-1, 2, 3)
                aligned = native @ matrix[:3, :3].T + matrix[:3, 3]
                gap = truth['declared'].get('gap_segment')
                rows.append(dict(parent=task['parent'], case_id=task['case_id'], condition=record['condition']['condition_id'],
                    method=rod['method'], state=rod['identity']['state'], reason=rod['identity']['reason'],
                    camera_decision=record['decision'], camera=score,
                    original_residual=record['after']['summary'], all_view_residual=record['all_views']['summary'],
                    metrics=curve_metrics(aligned, true_segments, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m']),
                    gap=gap_coverage(aligned, gap, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m']) if gap else None))
        control = next(r for r in records if r['condition']['condition_id'] == 'control')
        old = read_json(ROOT / task['old_rod_path'])
        assert digest(ROOT / task['old_rod_path']) == task['old_rod_sha256']
        old_cam_path = parent / task['old_camera_receipt']['path']
        assert digest(old_cam_path) == task['old_camera_receipt']['sha256']
        old_camera = read_json(old_cam_path)
        replay = dict(parent=task['parent'], case_id=task['case_id'],
            camera_result_exact=control['result'] == old_camera['result'],
            rod_identity_exact=[r['identity'] for r in control['rods']] == [r['identity'] for r in old['records']])
        # Numerical solvers may follow slightly different paths after JSON materialization.
        # Report that drift; never erase a condition just because a replay differs.
        replay['old_to_control_camera'] = camera_difference(old_camera['result'].get('intrinsics'), old_camera['result'].get('extrinsics'),
            control['result'].get('intrinsics'), control['result'].get('extrinsics'), task['initial_extrinsics'])
        replay['old_camera_decision'] = old_camera['decision']
        replay['control_camera_decision'] = control['decision']
        replay['old_identity_states'] = [r['identity']['state'] for r in old['records']]
        replay['control_identity_states'] = [r['identity']['state'] for r in control['rods']]
        assert old_camera['initial_intrinsics'] == task['initial_intrinsics']
        assert old_camera['initial_extrinsics'] == task['initial_extrinsics']
        assert old_camera['validation_plan'] == task['original_plan']
        replays.append(replay)
        initial_e = np.asarray(task['initial_extrinsics'])
        for left, right in combinations(records, 2):
            numerical = all('extrinsics' in r['result'] for r in (left, right))
            camera = camera_difference(left['result']['intrinsics'], left['result']['extrinsics'],
                right['result']['intrinsics'], right['result']['extrinsics'], initial_e) if numerical else dict(comparable=False, reason='missing_camera')
            rods = []
            for method in ['baseline', 'cylinder_support']:
                a = next((r['identity'] for r in left['rods'] if r['method'] == method), dict(state='missing', segments=[]))
                b = next((r['identity'] for r in right['rods'] if r['method'] == method), dict(state='missing', segments=[]))
                difference = segment_difference(a, b, float(np.linalg.norm(centers(initial_e)[-1] - centers(initial_e)[0]))) if camera['comparable'] else dict(comparable=False, reason='incomparable_camera_gauge')
                rods.append(dict(method=method, **difference))
            comparisons.append(dict(parent=task['parent'], case_id=task['case_id'],
                left=left['condition']['condition_id'], right=right['condition']['condition_id'], camera=camera, rods=rods))
    summary = dict(state='complete', run_id=RUN_ID, config=config, rows=rows, cameras=camera_rows, comparisons=comparisons,
        full_replays=replays, inference_sha256=digest(run / 'inference.json'),
        condition_count=len(index['records']), elapsed_seconds=index['elapsed_seconds'],
        no_automatic_promotion=True, gt_read_during_inference=False,
        scope='Sensitivity of seven known cases. Correlated folds, no best-fold selection; stability does not establish physical correctness.')
    destination = ROOT / 'data/evaluation' / RUN_ID
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', summary)
    write_json(SHARE, summary)
    print('EVALUATED', len(index['records']), len(rows), len(comparisons), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer', 'evaluate'])
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[args.stage]()
