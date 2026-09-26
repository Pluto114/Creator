"""Paired multi-seed evidence checks, with shared cameras and separate truth scoring."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, camera_score, digest, read_json, write_json
from run_rod_camera_envelope import EVALUATION, analytical_rod
from run_rod_camera_envelope import checked as checked_envelope
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import correction_decision, validate_heldout, validation_plan
from creator_eval.camera_bundle_shared import fit_shared
from creator_eval.camera_evidence_challenges import _frame_banks, generate_paired_challenges
from creator_eval.camera_native_sensitivity import check_native_gauge
from creator_eval.camera_training_groups import build_training_group_plan, select_training_tracks
from creator_eval.line_controls import curve_metrics
from creator_eval.rod_validation_evidence import (
    MINIMUM_ENDPOINT_VIEWS,
    MINIMUM_TARGET_VIEWS,
    THRESHOLD_PX,
    check_endpoint_reprojection,
    check_target_anchors,
)

RUN_ID = 'rod-validation-evidence-v1-20260926'
RUN = ROOT / '.runtime/experiments' / RUN_ID
INPUTS = ROOT / 'data/inputs' / RUN_ID
TRUTH = ROOT / 'data/eval_gt' / RUN_ID
PUBLIC = ROOT / 'docs/experiments/results/2026-09-26-rod-validation-evidence.json'
SOURCES = ['scripts/run_rod_validation_evidence.py',
    'experiments/src/creator_eval/camera_evidence_challenges.py',
    'experiments/src/creator_eval/rod_validation_evidence.py',
    'configs/camera_evidence_challenges_v1.json']
ARMS = ('baseline', 'endpoints', 'target_claims', 'joint')
SLOTS = ('control', 'leave_group_0', 'leave_group_1', 'leave_group_2', 'leave_group_3')


def checked():
    config = read_json(RUN / 'method_config.json')
    assert config['run_id'] == RUN_ID and not config['gt_read_during_inference']
    validate_screen_policy(config['screen_policy'])
    assert digest(INPUTS / 'manifest.json') == config['input_sha256']
    assert digest(RUN / 'source_freeze.json') == config['source_freeze_sha256']
    for name, sha in config['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(RUN / 'source_snapshot' / name), name
    for name, sha in config['receipts'].items():
        assert digest(ROOT / name) == sha, name
    return config, read_json(INPUTS / 'manifest.json')



def validate_screen_policy(policy):
    assert policy['endpoint']['threshold_px'] == THRESHOLD_PX
    assert policy['endpoint']['minimum_views'] == MINIMUM_ENDPOINT_VIEWS
    assert policy['target']['threshold_px'] == THRESHOLD_PX
    assert policy['target']['minimum_views'] == MINIMUM_TARGET_VIEWS


def emitted_metrics(metrics, accepted):
    # An explicitly withheld result is empty even if cameras cannot be aligned.
    # Every target in this positive-target suite exists; empty means R=0/P=null.
    if not accepted:
        return dict(emitted_recovery=0., emitted_precision=None)
    return dict(emitted_recovery=metrics['recovery_fraction'] if metrics else None,
                emitted_precision=metrics['precision_fraction'] if metrics else None)


def unique(items, key):
    indexed = {key(item): item for item in items}
    assert len(indexed) == len(items), 'Repeated identity'
    return indexed


def prepare():
    if any(p.exists() for p in (RUN, INPUTS, TRUTH)):
        raise FileExistsError('Preserve prior attempts; choose another run id')
    parent, frozen, old = checked_envelope('analytic')
    for directory in (RUN, INPUTS, TRUTH):
        directory.mkdir(parents=True)
    (RUN / 'records').mkdir()
    sources = source_snapshot(RUN, set(frozen['source_sha256']) | set(SOURCES))
    write_json(RUN / 'source_freeze.json', dict(source_sha256=sources, evaluation=EVALUATION,
        arms=ARMS, scope='Frozen before generating paired pixel measurements or running any fits'))
    protocol = read_json(ROOT / 'configs/camera_evidence_challenges_v1.json')
    validate_screen_policy(protocol['policy'])
    inputs, truth = generate_paired_challenges(protocol)
    method, variant = old['method'], old['variant']
    for group in inputs['camera_groups']:
        group['group_plan'] = build_training_group_plan(group['training'],
            validation_track_ids=[r['track_id'] for r in group['validation']], view_count=len(group['view_ids']),
            minimum_tracks=method['optimizer']['minimum_initial_tracks'],
            minimum_tracks_per_view=method['optimizer']['minimum_trimmed_tracks_per_view'])
        group['original_plan'] = validation_plan(group['validation'], group['initial_extrinsics'])
        group['all_view_plan'] = all_view_plan(group['validation'], group['initial_extrinsics'],
            training_track_ids=[r['track_id'] for r in group['training']])
    manifest = dict(**inputs, method=method, variant=variant)
    validate_inputs(manifest)
    write_json(INPUTS / 'manifest.json', manifest)
    write_json(TRUTH / 'manifest.json', truth)
    write_json(TRUTH / 'frozen.json', dict(truth_sha256=digest(TRUTH / 'manifest.json'),
        input_sha256=digest(INPUTS / 'manifest.json'), source_freeze_sha256=digest(RUN / 'source_freeze.json')))
    receipts = {p.relative_to(ROOT).as_posix(): digest(p) for p in [parent / 'method_config.json',
        ROOT / 'data/inputs' / frozen['run_id'] / 'manifest.json']}
    write_json(RUN / 'method_config.json', dict(run_id=RUN_ID, source_sha256=sources,
        input_sha256=digest(INPUTS / 'manifest.json'), source_freeze_sha256=digest(RUN / 'source_freeze.json'),
        receipts=receipts, screen_policy=protocol['policy'], evaluation=EVALUATION, arms=ARMS,
        gt_read_during_inference=False, no_automatic_promotion=True,
        scope='Twenty paired seeds; four rod correspondence modes share the same five camera fits. Supplied synthetic target claims are an additional input assumption, not automatic recognition.'))
    print('PREPARED', len(inputs['camera_groups']), 'camera groups;', len(inputs['cases']), 'rod cases', flush=True)


def validate_inputs(manifest):
    groups = unique(manifest['camera_groups'], lambda r: r['camera_group_id'])
    cases = unique(manifest['cases'], lambda r: r['case_id'])
    assert len(groups) == 20 and len(cases) == 80
    for group_id, group in groups.items():
        siblings = [c for c in cases.values() if c['camera_group_id'] == group_id]
        assert len(siblings) == 4
        assert group['frames'] == _frame_banks(group, siblings), 'Detached measurement bank'
        assert group['view_ids'] == [f['view_id'] for f in group['frames']]
        assert len(set(group['view_ids'])) == 5
        for frame in group['frames']:
            assert frame['source_kind'] == 'synthetic_pixel_measurements'
            assert frame['source_sha256'] == canonical_hash(frame['measurement_bank'])
        if 'group_plan' in group:
            plan = build_training_group_plan(group['training'], validation_track_ids=[r['track_id'] for r in group['validation']],
                view_count=5, minimum_tracks=manifest['method']['optimizer']['minimum_initial_tracks'],
                minimum_tracks_per_view=manifest['method']['optimizer']['minimum_trimmed_tracks_per_view'])
            assert group['group_plan'] == plan
            assert group['original_plan'] == validation_plan(group['validation'], group['initial_extrinsics'])
            assert group['all_view_plan'] == all_view_plan(group['validation'], group['initial_extrinsics'],
                training_track_ids=[r['track_id'] for r in group['training']])
    for case in cases.values():
        assert case['camera_group_id'] in groups and case['rod_support_sha256'] == canonical_hash(case['rod_tracks'])
        evidence = unique(case['evidence_sets'], lambda r: r['evidence_id'])
        assert len(evidence) == 3
        frames = {f['view_id']: f for f in groups[case['camera_group_id']]['frames']}
        for item in evidence.values():
            assert item['evidence_sha256'] == canonical_hash({k: item[k] for k in ('source_kind', 'anchors')})
            for anchor in item['anchors']:
                assert anchor['source_sha256'] == frames[anchor['view_id']]['source_sha256']
    return groups, cases


def frames_for(group, k, e):
    return [dict(view_id=f['view_id'], size_wh=f['size_wh'], source_sha256=f['source_sha256'],
        source_kind=f['source_kind'], K_index=ck, world_to_camera_cv=ce)
        for f, ck, ce in zip(group['frames'], k, e)]


def endpoint_input(case, frames):
    return [dict(endpoint_id=t['track_id'], observations=[dict(view_id=frames[o['view']]['view_id'],
        xy=o['xy'], source_sha256=frames[o['view']]['source_sha256']) for o in t['observations']])
        for s in case['rod_tracks'] for t in s['endpoint_tracks']]


def decision(camera, identity, endpoint, target, arm):
    if arm not in ARMS:
        raise ValueError('Unknown evidence arm')
    if camera['state'] != 'candidate_camera_correction':
        return dict(state='withheld_camera', reasons=['camera_not_accepted'])
    if identity['state'] != 'accepted':
        return dict(state='unresolved', reasons=['missing_candidate_geometry'])
    checks = []
    if arm in ('endpoints', 'joint'):
        checks.append(('endpoints', endpoint))
    if arm in ('target_claims', 'joint'):
        checks.append(('target_claims', target))
    contradicted = [name for name, check in checks if check['state'] == 'contradicted']
    unresolved = [name for name, check in checks if check['state'] != 'supported']
    if contradicted:
        return dict(state='rejected', reasons=contradicted)
    if unresolved:
        return dict(state='unresolved', reasons=unresolved)
    return dict(state='accepted', reasons=[], scope='Conditional research output; not physical correctness')


def check_rods(group, cases, fit, camera):
    numerical = 'extrinsics' in fit
    k = np.asarray(fit['intrinsics']) if numerical else None
    e = np.asarray(fit['extrinsics']) if numerical else None
    frames = frames_for(group, k, e) if numerical else None
    rows = []
    for case in cases:
        identity = analytical_rod(case, k, e) if numerical else dict(state='unavailable', reason='missing_camera', segments=[])
        endpoint = check_endpoint_reprojection(frames, endpoint_input(case, frames)) if numerical else dict(state='unresolved', reason='missing_camera')
        evidence = []
        for item in case['evidence_sets']:
            target = check_target_anchors(frames, identity['segments'], item['anchors']) if numerical else dict(state='unresolved', reason='missing_camera')
            evidence.append(dict(evidence_id=item['evidence_id'], evidence_sha256=canonical_hash(item), target=target,
                decisions={arm: decision(camera, identity, endpoint, target, arm) for arm in ARMS}))
        rows.append(dict(case_id=case['case_id'], case_sha256=canonical_hash(case), identity=identity,
            endpoint=endpoint, evidence=evidence, geometry_changed_by_checks=False))
    return rows


def worker(group, cases, method, variant):
    sys.addaudithook(reject_truth_open)
    checked()
    k, e = np.asarray(group['initial_intrinsics']), np.asarray(group['initial_extrinsics'])
    before = validate_heldout(k, e, group['original_plan'])
    receipts = []
    for condition in group['group_plan']['conditions']:
        path = RUN / 'records' / (group['camera_group_id'] + '-' + condition['condition_id'] + '.json')
        if path.exists():
            raise FileExistsError(path)
        if condition['state'] == 'eligible':
            training = select_training_tracks(group['training'], group['group_plan'], condition['condition_id'])
            try:
                fit = fit_shared(k, e, training, method['optimizer'], variant)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                fit = dict(state='numerical_failure', success=False, error=repr(error))
        else:
            fit = dict(state='skipped_group_coverage', success=False)
        after, all_views, gauge = None, None, None
        if 'extrinsics' in fit:
            ck, ce = np.asarray(fit['intrinsics']), np.asarray(fit['extrinsics'])
            gauge = check_native_gauge(e, ce)
            assert gauge['comparable'], gauge
            after = validate_heldout(ck, ce, group['original_plan'])
            all_views = score_all_views(ck, ce, group['all_view_plan'])
            kept, discarded = set(fit['kept_track_ids']), set(fit['discarded_track_ids'])
            assert not kept & discarded and kept | discarded == set(condition['kept_track_ids'])
        camera = correction_decision(fit, before, after or before, method['decision'])
        rods = check_rods(group, cases, fit, camera)
        record = dict(camera_group_id=group['camera_group_id'], group_sha256=canonical_hash(group),
            condition=condition, fit=fit, before=before, after=after, all_views=all_views, gauge=gauge,
            camera=camera, rods=rods, gt_read_during_inference=False)
        write_json(path, record)
        receipts.append(dict(camera_group_id=group['camera_group_id'], condition=condition['condition_id'],
            path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
        print('EVIDENCE_FIT', path.stem, fit['state'], camera['state'], flush=True)
    return receipts


def infer():
    sys.addaudithook(reject_truth_open)
    config, manifest = checked()
    groups, cases = validate_inputs(manifest)
    if (RUN / 'inference.json').exists():
        raise FileExistsError('Keep existing inference')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, group, [c for c in cases.values() if c['camera_group_id'] == gid],
            manifest['method'], manifest['variant']) for gid, group in groups.items()]
        records = [record for future in as_completed(futures) for record in future.result()]
    checked()
    write_json(RUN / 'inference.json', dict(state='complete', input_sha256=config['input_sha256'],
        config_sha256=digest(RUN / 'method_config.json'), gt_read_during_inference=False,
        records=sorted(records, key=lambda r: (r['camera_group_id'], r['condition'])), elapsed_seconds=time.perf_counter() - started))
    print('INFERRED', len(records), 'camera fits shared across 400 rod conditions', flush=True)


def audit_records():
    config, manifest = checked()
    groups, cases = validate_inputs(manifest)
    index = read_json(RUN / 'inference.json')
    assert index['state'] == 'complete' and not index['gt_read_during_inference']
    assert index['config_sha256'] == digest(RUN / 'method_config.json') and index['input_sha256'] == config['input_sha256']
    records = unique(index['records'], lambda r: (r['camera_group_id'], r['condition']))
    assert set(records) == {(gid, slot) for gid in groups for slot in SLOTS}
    receipts = {p.relative_to(ROOT).as_posix(): digest(p) for p in
        [RUN / 'method_config.json', RUN / 'source_freeze.json', RUN / 'inference.json', INPUTS / 'manifest.json']}
    receipts.update(config['receipts'])
    for name, sha in config['source_sha256'].items():
        receipts[name] = sha
        receipts[(RUN / 'source_snapshot' / name).relative_to(ROOT).as_posix()] = sha
    for (gid, cid), receipt in records.items():
        path = RUN / receipt['path']
        assert digest(path) == receipt['sha256']
        receipts[path.relative_to(ROOT).as_posix()] = receipt['sha256']
        record = read_json(path)
        group = groups[gid]
        assert record['camera_group_id'] == gid and record['group_sha256'] == canonical_hash(group)
        assert record['condition'] == next(c for c in group['group_plan']['conditions'] if c['condition_id'] == cid)
        assert not record['gt_read_during_inference']
        before = validate_heldout(group['initial_intrinsics'], group['initial_extrinsics'], group['original_plan'])
        fit = record['fit']
        after = validate_heldout(fit['intrinsics'], fit['extrinsics'], group['original_plan']) if 'extrinsics' in fit else None
        assert record['before'] == before and record['after'] == after
        camera = correction_decision(fit, before, after or before, manifest['method']['decision'])
        assert record['camera'] == camera
        expected_cases = [c for c in cases.values() if c['camera_group_id'] == gid]
        assert canonical_hash(record['rods']) == canonical_hash(check_rods(group, expected_cases, fit, camera))
        if 'extrinsics' in fit:
            assert record['all_views'] == score_all_views(fit['intrinsics'], fit['extrinsics'], group['all_view_plan'])
            assert record['gauge'] == check_native_gauge(group['initial_extrinsics'], fit['extrinsics'])
            kept, discarded = set(fit['kept_track_ids']), set(fit['discarded_track_ids'])
            assert not kept & discarded and kept | discarded == set(record['condition']['kept_track_ids'])
    return config, manifest, index, records, receipts


def pre():
    if PUBLIC.exists() or (ROOT / 'data/evaluation' / RUN_ID).exists():
        raise FileExistsError('Pre audit must actually precede evaluation')
    sys.addaudithook(reject_truth_open)
    _, _, _, records, receipts = audit_records()
    write_json(RUN / 'before_evaluation.json', dict(state='passed', receipts=receipts,
        camera_count=len(records), rod_condition_count=len(records)*4, decision_count=len(records)*4*3*4,
        gt_read=False, physical_scores_read=False))
    print('PRE_PASSED', len(receipts), 'unchanged-source receipts;', len(records), 'camera records', flush=True)


def evaluate():
    config, manifest, index, records, receipts = audit_records()
    before = read_json(RUN / 'before_evaluation.json')
    assert before['state'] == 'passed' and before['receipts'] == receipts
    for name, sha in before['receipts'].items():
        assert digest(ROOT / name) == sha
    frozen = read_json(TRUTH / 'frozen.json')
    assert frozen['input_sha256'] == config['input_sha256']
    assert frozen['source_freeze_sha256'] == config['source_freeze_sha256']
    assert frozen['truth_sha256'] == digest(TRUTH / 'manifest.json')
    truth = read_json(TRUTH / 'manifest.json')
    assert truth['dataset_id'] == manifest['dataset_id'] and truth['schema_version'] == manifest['schema_version']
    gt_groups = unique(truth['camera_groups'], lambda r: r['camera_group_id'])
    gt_cases = unique(truth['cases'], lambda r: r['case_id'])
    groups, cases = validate_inputs(manifest)
    assert set(groups) == set(gt_groups) and set(cases) == set(gt_cases)
    rows, cameras = [], []
    for gid, group in groups.items():
        gt_group = gt_groups[gid]
        assert [c['view_id'] for c in gt_group['cameras']] == group['view_ids']
        assert all(c['size_wh'] == group['size_wh'] for c in gt_group['cameras'])
        control = read_json(RUN / records[(gid, 'control')]['path'])
        fit = control['fit']
        alignment, alignment_reason = None, None
        if 'extrinsics' in fit:
            try:
                alignment = camera_score(np.asarray(fit['intrinsics']), np.asarray(fit['extrinsics']), gt_group['cameras'], config['evaluation']['alignment'])
            except ValueError as error:
                alignment_reason = str(error)
        else:
            alignment_reason = 'missing_control_camera'
        matrix = np.asarray(alignment['prediction_world_to_gt_world']) if alignment else None
        cameras.append(dict(camera_group_id=gid, alignment=alignment, reason=alignment_reason,
            scope='One full-control camera-only Sim3 reused for all four paired modes, five conditions and all claims'))
        for slot in SLOTS:
            record = read_json(RUN / records[(gid, slot)]['path'])
            for rod in record['rods']:
                case, gt_case = cases[rod['case_id']], gt_cases[rod['case_id']]
                assert case['camera_group_id'] == gt_case['camera_group_id'] == gid
                assert {i['evidence_id'] for i in gt_case['evidence_labels']} == {i['evidence_id'] for i in case['evidence_sets']}
                assert len(gt_case['rod_segments']) > 0, 'This evaluation declares positive target geometry'
                native = np.asarray(rod['identity']['segments']).reshape(-1, 2, 3)
                metrics = None
                if matrix is not None:
                    aligned = native @ matrix[:3, :3].T + matrix[:3, 3]
                    metrics = curve_metrics(aligned, gt_case['rod_segments'], tolerance=config['evaluation']['curve_tolerance_m'], spacing=config['evaluation']['curve_spacing_m'])
                decisions = []
                for item in rod['evidence']:
                    for arm, result in item['decisions'].items():
                        accepted = result['state'] == 'accepted'
                        decisions.append(dict(evidence_id=item['evidence_id'], arm=arm, state=result['state'], reasons=result['reasons'],
                            **emitted_metrics(metrics, accepted)))
                rows.append(dict(camera_group_id=gid, case_id=case['case_id'], condition=slot,
                    family=gt_case['mode'], evidence_labels=gt_case['evidence_labels'], camera_decision=record['camera'],
                    camera_fit_state=record['fit']['state'], image_residual=record['all_views']['summary'] if record['all_views'] else None,
                    identity_state=rod['identity']['state'], geometry_metrics=metrics, endpoint_check=rod['endpoint'],
                    target_checks=[dict(evidence_id=i['evidence_id'], check=i['target']) for i in rod['evidence']],
                    decisions=decisions, shared_alignment_camera_group=gid))
    assert len(rows) == 400 and sum(len(r['decisions']) for r in rows) == 4800
    for name, sha in before['receipts'].items():
        assert digest(ROOT / name) == sha
    result = dict(state='complete', run_id=RUN_ID, config=config, rows=rows, cameras=cameras,
        inference_sha256=digest(RUN / 'inference.json'), before_sha256=digest(RUN / 'before_evaluation.json'),
        truth_frozen=frozen, elapsed_seconds=index['elapsed_seconds'], gt_read_during_inference=False,
        camera_fit_count=100, paired_seed_count=20, rod_condition_count=len(rows), decision_count=4800,
        all_before_hashes_unchanged=True, no_automatic_promotion=True,
        scope='Paired component study with supplied synthetic endpoint correspondences and 2D target claims. Twenty random seeds, not 400 independent objects or 4800 independent trials. Conditional gates do not establish target correctness.')
    destination = ROOT / 'data/evaluation' / RUN_ID
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', result)
    write_json(PUBLIC, result)
    print('EVALUATED', len(rows), 'geometry rows; 4800 fixed evidence decisions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'infer', 'pre', 'evaluate'))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {'prepare': prepare, 'infer': infer, 'pre': pre, 'evaluate': evaluate}[args.stage]()
