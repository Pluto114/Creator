"""Independent pre/post evaluation audit of frozen training-group sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import ROOT, checked_run, digest, load_case, read_json, write_json
from run_rod_candidate_ablation import canonical_hash
from run_rod_reference import checked_annotations

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import (
    correction_decision,
    observations_from_record,
    validate_heldout,
    validation_plan,
)
from creator_eval.camera_native_sensitivity import camera_difference
from creator_eval.camera_training_groups import build_training_group_plan, select_training_tracks

DEFAULT_RUN = 'camera-training-sensitivity-v1-20260924r2'
DEFAULT_AUDIT = 'camera-training-audit-v1-20260924'
DEFAULT_EVALUATION = ROOT / 'docs/experiments/results/2026-09-24-camera-training-sensitivity-r2.json'
DEFAULT_OUTPUT = ROOT / 'docs/experiments/results/2026-09-24-camera-training-audit.json'
METHODS = ('baseline', 'cylinder_support')


def _one(values, predicate, message):
    found = [value for value in values if predicate(value)]
    if len(found) != 1:
        raise ValueError(message)
    return found[0]


def _receipted(path, receipts):
    path = Path(path)
    key, sha = path.relative_to(ROOT).as_posix(), digest(path)
    if key in receipts and receipts[key] != sha:
        raise ValueError('Artifact changed during audit: ' + key)
    receipts[key] = sha
    return read_json(path)


def _sources(run, sources):
    for name, sha in sources.items():
        if digest(ROOT / name) != sha or digest(run / 'source_snapshot' / name) != sha:
            raise ValueError('Frozen source changed: ' + name)


def proposal_assignment(proposal, view_ids, frames, pool_sha256, cap):
    """Resolve hypothesis and finite row ids against one exact frozen pixel pool."""
    hypothesis, finite = proposal['hypothesis'], proposal['finite']
    matches = hypothesis['matches']
    support = set(hypothesis['supporting_views'])
    if len(matches) != len(view_ids) or not support <= set(range(len(view_ids))):
        raise ValueError('Candidate view identity is malformed')
    assignments = []
    for view, view_id in enumerate(view_ids):
        candidate = matches[view]['candidate_index'] if view in support else None
        if candidate is not None and not 0 <= candidate < min(cap, len(frames[view]['pool']['candidates'])):
            raise ValueError('Hypothesis candidate outside the retained pool')
        assignments.append(dict(view_id=view_id, candidate_index=candidate, pool_sha256=pool_sha256))
    selections = finite.get('candidate_selection', [])
    source_rows, finite_assignments = [], []
    if selections:
        if [s['view_id'] for s in selections] != view_ids:
            raise ValueError('Finite candidate view order changed')
        for view, selection in enumerate(selections):
            candidate = selection['candidate_index']
            finite_assignments.append(dict(view_id=view_ids[view], candidate_index=candidate, pool_sha256=pool_sha256))
            if candidate != assignments[view]['candidate_index']:
                raise ValueError('Finite rows no longer belong to the declared hypothesis')
            seen = set()
            allowed = set() if candidate is None else {tuple(m[:2]) for m in frames[view]['pool']['candidates'][candidate]['row_matches']}
            for row_index, raw_index in selection['row_matches']:
                if row_index in seen or (row_index, raw_index) not in allowed:
                    raise ValueError('Finite support row is repeated or detached from its image candidate')
                seen.add(row_index)
                raw = frames[view]['observations']['rows']
                if not 0 <= row_index < len(raw) or not 0 <= raw_index < len(raw[row_index]['candidates']):
                    raise ValueError('Finite row outside the original observation pool')
                source_rows.append((view, row_index, raw_index))
    source_rows.sort()
    return dict(ordinal=proposal['ordinal'], finite_state=finite['state'], hypothesis_assignments=assignments,
                finite_assignments=finite_assignments, source_rows=source_rows,
                source_rows_sha256=canonical_hash(source_rows), source_row_count=len(source_rows),
                per_view_row_count=[sum(row[0] == view for row in source_rows) for view in range(len(view_ids))])


def describe_identity(rod, task, frames):
    identity, bundle = rod['identity'], rod['bundle']
    proposals = [proposal_assignment(p, task['view_ids'], frames, task['pool_sha256'], task['rod_method']['candidate_cap'])
                 for p in bundle['proposals']]
    selected = None
    if identity['state'] == 'accepted':
        selected = _one(proposals, lambda p: p['ordinal'] == identity['selected_ordinal'], 'Accepted identity needs one selected proposal')
        source = _one(bundle['proposals'], lambda p: p['ordinal'] == identity['selected_ordinal'], 'Selected proposal missing')
        if source['finite']['state'] != 'accepted' or source['finite']['segments'] != identity['segments']:
            raise ValueError('Accepted identity is detached from its finite geometry')
    return dict(state=identity['state'], reason=identity['reason'], selected_ordinal=identity.get('selected_ordinal'),
                selected=selected, proposals=[{k: v for k, v in p.items() if k != 'source_rows'} for p in proposals])


def compare_assignments(control, fold):
    a, b = control['selected'], fold['selected']
    output = dict(comparable=False, control_state=control['state'], fold_state=fold['state'],
                  hypothesis_assignments_equal=None, finite_assignments_equal=None, support_row_jaccard=None,
                  control_row_count=None, fold_row_count=None, intersection_count=None, union_count=None)
    if a is None or b is None:
        return {**output, 'reason': 'control_or_fold_identity_not_accepted'}
    if {x['pool_sha256'] for x in a['hypothesis_assignments']} != {x['pool_sha256'] for x in b['hypothesis_assignments']}:
        return {**output, 'reason': 'different_source_pixel_pools'}
    left, right = {tuple(row) for row in a['source_rows']}, {tuple(row) for row in b['source_rows']}
    union, intersection = left | right, left & right
    # 两个空集合不能被算成“完美一致”。没有支持像素，就是没有可比证据。
    if not left or not right:
        return {**output, 'reason': 'empty_selected_pixel_support'}
    return {**output, 'comparable': True, 'reason': None,
            'hypothesis_assignments_equal': a['hypothesis_assignments'] == b['hypothesis_assignments'],
            'finite_assignments_equal': a['finite_assignments'] == b['finite_assignments'],
            'support_row_jaccard': len(intersection) / len(union), 'control_row_count': len(left),
            'fold_row_count': len(right), 'intersection_count': len(intersection), 'union_count': len(union)}


def audit_inference(run_id):
    run, inputs = ROOT / '.runtime/experiments' / run_id, ROOT / 'data/inputs' / run_id
    receipts = {}
    config = _receipted(run / 'method_config.json', receipts)
    if config['run_id'] != run_id or config['gt_read_during_inference']:
        raise ValueError('Wrong or privileged sensitivity configuration')
    _sources(run, config['source_sha256'])
    manifest = _receipted(inputs / 'manifest.json', receipts)
    if digest(inputs / 'manifest.json') != config['input_sha256'] or manifest['parents'] != config['parents']:
        raise ValueError('Detached sensitivity input')
    index = _receipted(run / 'inference.json', receipts)
    if index['state'] != 'complete' or index['gt_read_during_inference'] or index['config_sha256'] != digest(run / 'method_config.json') or index['input_sha256'] != config['input_sha256']:
        raise ValueError('Incomplete or detached sensitivity inference')
    parents, expected_cases, chains = {}, set(), []
    for parent_id, previous in config['parents'].items():
        parent, parent_inputs, frozen, parent_manifest = checked_run(parent_id)
        for name, sha in previous.items():
            _receipted(parent / name, receipts)
            if digest(parent / name) != sha:
                raise ValueError('Frozen parent receipt changed')
        for name in ('inference.json', 'rod-inference.json'):
            parent_index = read_json(parent / name)
            if parent_index['state'] != 'complete' or parent_index['input_sha256'] != frozen['input_sha256'] or parent_index['source_sha256'] != frozen['source_sha256'] or parent_index['gt_read_during_inference']:
                raise ValueError('Parent inference detached')
        camera_index, rod_index = read_json(parent / 'inference.json'), read_json(parent / 'rod-inference.json')
        if camera_index['predictions_sha256'] != digest(parent / 'predictions.json') or rod_index['camera_inference_sha256'] != digest(parent / 'inference.json') or rod_index['annotations_sha256'] != digest(parent / 'annotations.json') or rod_index['pools_sha256'] != digest(parent / 'rod-pools.json'):
            raise ValueError('Parent prediction/rod/annotation/pool chain detached')
        annotations = checked_annotations(parent, frozen)
        protocol = _receipted(parent / 'protocol.json', receipts)
        if digest(parent / 'protocol.json') != frozen['protocol_sha256']:
            raise ValueError('Parent evaluation protocol changed')
        scene_run, scene_inputs, truth_root, scene, scene_manifest = checked_scene(protocol['run_id'])
        for path, expected_sha in ((scene_run / 'prepared.json', frozen['scene_prepared_sha256']),
                                   (scene_run / 'inference.json', frozen['scene_inference_sha256']),
                                   (scene_inputs / 'manifest.json', frozen['scene_input_sha256'])):
            _receipted(path, receipts)
            if digest(path) != expected_sha:
                raise ValueError('Parent bundle no longer names this frozen scene')
        truth = _receipted(truth_root / 'manifest.json', receipts)
        _receipted(truth_root / 'artifact_hashes.json', receipts)
        if digest(truth_root / 'manifest.json') != scene['truth_sha256'] or digest(truth_root / 'artifact_hashes.json') != scene['truth_artifacts_sha256'] or truth['input_sha256'] != scene['input_sha256']:
            raise ValueError('Truth manifest detached from the exact scene RGB input')
        if {c['case_id'] for c in truth['cases']} != {c['case_id'] for c in scene_manifest['cases']}:
            raise ValueError('Scene truth/RGB case identities disagree')
        chains.append(dict(parent=parent_id, scene_run_id=protocol['run_id'], bundle_prepared_sha256=digest(parent / 'prepared.json'),
                           scene_prepared_sha256=digest(scene_run / 'prepared.json'), scene_input_sha256=scene['input_sha256'],
                           scene_inference_sha256=digest(scene_run / 'inference.json'), truth_manifest_sha256=scene['truth_sha256'],
                           truth_usage='hash and input/case identity audit only; no numerical truth used in this audit'))
        parents[parent_id] = (parent, parent_inputs, frozen, parent_manifest, annotations)
        expected_cases.update((parent_id, c['case_id']) for c in parent_manifest['cases'])
    actual_cases = [(t['parent'], t['case_id']) for t in manifest['tasks']]
    if len(actual_cases) != len(expected_cases) or set(actual_cases) != expected_cases:
        raise ValueError('Missing or duplicate sensitivity case')
    expected = {(t['parent'], t['case_id'], c['condition_id']) for t in manifest['tasks'] for c in t['group_plan']['conditions']}
    actual = [(r['parent'], r['case_id'], r['condition']) for r in index['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicate sensitivity condition')
    identities, replays, condition_checks = [], [], []
    for task in manifest['tasks']:
        parent, parent_inputs, _, parent_manifest, annotations = parents[task['parent']]
        _, case, track_record, method, k, e = load_case(task['parent'], task['case_id'])
        _receipted(parent_inputs / case['tracks_path'], receipts)
        train, validation = observations_from_record(track_record, 'train'), observations_from_record(track_record, 'validation')
        group = build_training_group_plan(train, validation_track_ids=[t['track_id'] for t in validation], view_count=len(k),
            minimum_tracks=method['optimizer']['minimum_initial_tracks'], minimum_tracks_per_view=method['optimizer']['minimum_trimmed_tracks_per_view'])
        if task['training'] != train or task['group_plan'] != group or task['method'] != method or task['rod_method'] != parent_manifest['rod_method']:
            raise ValueError('Training groups or algorithms changed from declared parent')
        np.testing.assert_array_equal(task['initial_intrinsics'], k)
        np.testing.assert_array_equal(task['initial_extrinsics'], e)
        if task['original_plan'] != validation_plan(validation, e) or task['all_view_plan'] != all_view_plan(validation, e, training_track_ids=[t['track_id'] for t in train]):
            raise ValueError('Validation observations or initial-camera pairing changed')
        query = _one(annotations['queries'], lambda q: q['case_id'] == task['case_id'], 'Parent query missing')
        if query != task['query']:
            raise ValueError('Foreground query changed')
        pool = _receipted(ROOT / task['pool_path'], receipts)
        if digest(ROOT / task['pool_path']) != task['pool_sha256'] or [f['view_id'] for f in pool['frames']] != task['view_ids']:
            raise ValueError('Frozen pixel pool changed')
        old_rod = _receipted(ROOT / task['old_rod_path'], receipts)
        old_camera = _receipted(parent / task['old_camera_receipt']['path'], receipts)
        if pool['case_id'] != task['case_id'] or old_rod['case_id'] != task['case_id'] or old_camera['case_id'] != task['case_id'] or old_rod['pool_sha256'] != task['pool_sha256'] or old_rod['camera_receipt'] != task['old_camera_receipt']:
            raise ValueError('Old control, camera receipt, and pixel pool name different cases')
        if digest(ROOT / task['old_rod_path']) != task['old_rod_sha256'] or digest(parent / task['old_camera_receipt']['path']) != task['old_camera_receipt']['sha256']:
            raise ValueError('Historical full-control artifacts changed')
        np.testing.assert_array_equal(old_camera['initial_intrinsics'], k)
        np.testing.assert_array_equal(old_camera['initial_extrinsics'], e)
        if old_camera['validation_plan'] != task['original_plan'] or old_camera['training_count'] != len(train) or old_camera['variant'] != task['variant'] or old_camera['condition']['corrupt_fraction'] != 0 or old_camera['corruption']:
            raise ValueError('Historical full control used different initial inputs or training protocol')
        old_kept = set(old_camera['result']['kept_track_ids'])
        old_discarded = set(old_camera['result']['discarded_track_ids'])
        if old_kept & old_discarded or old_kept | old_discarded != set(group['training_track_ids']):
            raise ValueError('Historical full control did not account for exactly the same training tracks')
        for condition in group['conditions']:
            cid = condition['condition_id']
            receipt = _one(index['records'], lambda r: (r['parent'], r['case_id'], r['condition']) == (task['parent'], task['case_id'], cid), 'Condition receipt missing')
            path = run / receipt['path']
            record = _receipted(path, receipts)
            if digest(path) != receipt['sha256'] or (record['parent'], record['case_id']) != (task['parent'], task['case_id']) or record['condition'] != condition or record['task_sha256'] != canonical_hash(task) or record['pool_sha256'] != task['pool_sha256'] or record['gt_read_during_inference']:
                raise ValueError('Condition record identity detached')
            if condition['state'] == 'eligible':
                selected = select_training_tracks(train, group, cid)
                if [t['track_id'] for t in selected] != condition['kept_track_ids']:
                    raise ValueError('Whole-track selection changed')
            fit = record['result']
            before = validate_heldout(k, e, task['original_plan'])
            if record['before'] != before:
                raise ValueError('Initial validation score changed')
            numerical = 'extrinsics' in fit
            if condition['state'] != 'eligible' and (numerical or fit['state'] != 'skipped_group_coverage'):
                raise ValueError('Coverage-skipped condition was still fitted')
            if numerical:
                kept, discarded = set(fit['kept_track_ids']), set(fit['discarded_track_ids'])
                if kept & discarded or kept | discarded != set(condition['kept_track_ids']) or kept & set(group['validation_track_ids']):
                    raise ValueError('Fit used undeclared or validation tracks')
                after = validate_heldout(fit['intrinsics'], fit['extrinsics'], task['original_plan'])
                if record['after'] != after or record['all_views'] != score_all_views(fit['intrinsics'], fit['extrinsics'], task['all_view_plan']):
                    raise ValueError('Frozen held-out residuals do not replay')
                if len(record['rods']) != 2 or {r['method'] for r in record['rods']} != set(METHODS):
                    raise ValueError('A rod method is missing or duplicated')
            else:
                after = before
                if record['rods'] or record['after'] is not None or record['all_views'] is not None:
                    raise ValueError('Missing camera must retain explicit absence')
            if record['decision'] != correction_decision(fit, before, after, method['decision']):
                raise ValueError('Original candidate gate changed')
            condition_checks.append(dict(parent=task['parent'], case_id=task['case_id'], condition=cid,
                numerical_camera=numerical, plan_state=condition['state'], requested_tracks=len(condition['kept_track_ids']),
                fit_state=fit['state'], validation_track_count=len(validation), whole_track_selection=True,
                training_validation_disjoint=True, unchanged_validation_plans=True, residuals_replayed=True))
            for rod_method in METHODS:
                rod = next((r for r in record['rods'] if r['method'] == rod_method), None)
                description = describe_identity(rod, task, pool['frames']) if rod is not None else dict(state='missing_camera', reason=fit['state'], selected=None, proposals=[])
                identities.append(dict(parent=task['parent'], case_id=task['case_id'], condition=cid, method=rod_method,
                                       record_sha256=receipt['sha256'], **description))
            if cid == 'control':
                same_camera = fit == old_camera['result']
                same_identity = [r['identity'] for r in record['rods']] == [r['identity'] for r in old_rod['records']]
                same_bundles = {r['method']: r['bundle'] for r in record['rods']} == old_rod['bundles']
                # 同值数组的内存布局也可能改变优化器的数值路径。来源必须一样，
                # 结果则照实报告，不能因为重放略有漂移就把整组删掉。
                old_fit = old_camera['result']
                difference = camera_difference(old_fit.get('intrinsics'), old_fit.get('extrinsics'),
                    fit.get('intrinsics'), fit.get('extrinsics'), e)
                replays.append(dict(parent=task['parent'], case_id=task['case_id'],
                    same_initial_cameras=True, same_validation_plan=True, same_full_training_tracks=True,
                    camera_exact=same_camera, identities_exact=same_identity, bundles_exact=same_bundles,
                    old_state=old_fit['state'], control_state=fit['state'], old_decision=old_camera['decision'],
                    control_decision=record['decision'], old_to_control_camera_difference=difference))
    comparisons = []
    for task in manifest['tasks']:
        for method in METHODS:
            rows = [r for r in identities if (r['parent'], r['case_id'], r['method']) == (task['parent'], task['case_id'], method)]
            control = _one(rows, lambda r: r['condition'] == 'control', 'Control identity missing')
            for fold in rows:
                if fold['condition'] == 'control':
                    continue
                comparisons.append(dict(parent=task['parent'], case_id=task['case_id'], method=method,
                    condition=fold['condition'], **compare_assignments(control, fold)))
    for name, sha in receipts.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Artifact changed before audit finished: ' + name)
    return dict(run_id=run_id, chains=chains, conditions=condition_checks, identities=identities, comparisons=comparisons,
                full_replays=replays, receipts=receipts, source_file_count=len(config['source_sha256']),
                inference_sha256=digest(run / 'inference.json'), config_sha256=digest(run / 'method_config.json'))


def pre_evaluate(args):
    run = ROOT / '.runtime/experiments' / args.run_id
    audit = ROOT / '.runtime/experiments' / args.audit_run_id
    if not (run / 'inference.json').exists():
        raise ValueError('Wait for complete inference before freezing the audit')
    if (ROOT / 'data/evaluation' / args.run_id / 'summary.json').exists() or args.evaluation_output.exists():
        raise ValueError('Pre-evaluation evidence must be recorded before evaluation')
    if audit.exists():
        raise FileExistsError('Preserve earlier audit attempts; choose a new --audit-run-id')
    config = read_json(run / 'method_config.json')
    audit.mkdir(parents=True)
    names = set(config['source_sha256']) | {'scripts/audit_camera_training_sensitivity.py'}
    sources = source_snapshot(audit, names)
    write_json(audit / 'prepared.json', dict(run_id=args.run_id, audit_run_id=args.audit_run_id, source_sha256=sources,
        config_sha256=digest(run / 'method_config.json'), evaluation_output=args.evaluation_output.relative_to(ROOT).as_posix(),
        scope='Independent audit. Truth is read only to verify the scene identity chain; no truth-based condition selection.'))
    result = audit_inference(args.run_id)
    _sources(audit, sources)
    write_json(audit / 'before-evaluation.json', result)
    print('PRE_EVALUATION_AUDIT', len(result['conditions']), 'conditions;', len(result['identities']), 'method identities;', len(result['chains']), 'parent chains', flush=True)


def post_evaluate(args):
    audit = ROOT / '.runtime/experiments' / args.audit_run_id
    frozen = read_json(audit / 'prepared.json')
    if frozen['run_id'] != args.run_id or frozen['audit_run_id'] != args.audit_run_id or frozen['evaluation_output'] != args.evaluation_output.relative_to(ROOT).as_posix():
        raise ValueError('Audit target changed')
    _sources(audit, frozen['source_sha256'])
    before = read_json(audit / 'before-evaluation.json')
    after = audit_inference(args.run_id)
    if before != after:
        raise ValueError('Inference, plans, assignments, or parent artifacts changed across evaluation')
    result_path = ROOT / 'data/evaluation' / args.run_id / 'summary.json'
    summary = read_json(result_path)
    if digest(result_path) != digest(args.evaluation_output) or summary['state'] != 'complete' or summary['inference_sha256'] != before['inference_sha256'] or summary['gt_read_during_inference']:
        raise ValueError('Published evaluation detached from the unchanged inference')
    expected_cameras = {(r['parent'], r['case_id'], r['condition']) for r in before['conditions']}
    actual_cameras = [(r['parent'], r['case_id'], r['condition']) for r in summary['cameras']]
    expected_rods = {(*key, method) for key in expected_cameras for method in METHODS}
    actual_rods = [(r['parent'], r['case_id'], r['condition'], r['method']) for r in summary['rows']]
    if len(actual_cameras) != len(expected_cameras) or set(actual_cameras) != expected_cameras or len(actual_rods) != len(expected_rods) or set(actual_rods) != expected_rods:
        raise ValueError('Evaluation dropped or duplicated a condition or rod method')
    identities = []
    for identity in before['identities']:
        item = dict(identity)
        if item['selected'] is not None:
            item['selected'] = {k: v for k, v in item['selected'].items() if k != 'source_rows'}
        identities.append(item)
    result = dict(state='complete', run_id=args.run_id, audit_run_id=args.audit_run_id,
        audit_prepared_sha256=digest(audit / 'prepared.json'), audit_source_file_count=len(frozen['source_sha256']),
        before_evaluation_sha256=digest(audit / 'before-evaluation.json'), evaluation_sha256=digest(result_path),
        inference_sha256=before['inference_sha256'], inference_and_parent_files_unchanged_across_evaluation=True,
        condition_count=len(expected_cameras), rod_method_count=len(expected_rods), chains=before['chains'],
        conditions=before['conditions'], full_replays=before['full_replays'], identities=identities,
        comparisons=before['comparisons'],
        scope='Identity means retained image candidate and its source rows, not physical rod identity or accuracy. Rejected/empty support is not perfect agreement.')
    _sources(audit, frozen['source_sha256'])
    write_json(audit / 'after-evaluation.json', result)
    write_json(args.output, result)
    print('POST_EVALUATION_AUDIT', len(expected_cameras), len(expected_rods), 'unchanged inference and verified identity assignments', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pre-evaluate', 'post-evaluate'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--audit-run-id', default=DEFAULT_AUDIT)
    parser.add_argument('--evaluation-output', type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.evaluation_output = (ROOT / args.evaluation_output).resolve()
    args.output = (ROOT / args.output).resolve()
    for value in (args.run_id, args.audit_run_id):
        if Path(value).name != value or ':' in value or value in ('.', '..'):
            raise ValueError('Single-component run ids required')
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {'pre-evaluate': pre_evaluate, 'post-evaluate': post_evaluate}[args.stage](args)
