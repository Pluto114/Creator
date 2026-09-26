"""Test frozen rod candidates in new RGB views, with background-only map localization.

A new picture is extra evidence, but PnP still inherits the old map and intrinsics.
Nothing here turns a small reprojection residual into absolute 3D accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, checked_run, digest, read_json, write_json
from run_camera_training_sensitivity import checked as checked_training
from run_rod_camera_envelope import checked as checked_envelope
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.background_correspondences import (
    detect_features,
    exclusion_mask,
    mutual_ratio_matches,
)
from creator_eval.camera_bundle import project, triangulate
from creator_eval.heldout_view_pose import propose_pose, verify_pose
from creator_eval.rod_validation_evidence import check_target_anchors

RUN_ID = 'rod-heldout-view-validation-v1-20260926'
SCENE_ID = 'rod-heldout-view-scenes-v1-20260926'
RUN = ROOT / '.runtime/experiments' / RUN_ID
INPUTS = ROOT / 'data/inputs' / RUN_ID
SCENE = ROOT / '.runtime/experiments' / SCENE_ID
SCENE_INPUTS = ROOT / 'data/inputs' / SCENE_ID
ANNOTATIONS = ROOT / 'configs/rod_heldout_views_annotations_v1.json'
PUBLIC = ROOT / 'docs/experiments/results/2026-09-26-heldout-view-validation.json'
PHYSICAL = ROOT / 'docs/experiments/results/2026-09-24-replay-rod-camera-envelope.json'
PHYSICAL_SHA = '42a2f23dfd91d46b2a2ce94741b2f76e5a6f0e6ebe946b1dbdddcce878c44012'
CONDITIONS = ['control', *['leave_group_' + str(i) for i in range(4)]]
SIFT = dict(nfeatures=2000, nOctaveLayers=3, contrastThreshold=.04, edgeThreshold=10, sigma=1.6)
SOURCES = ['scripts/run_rod_heldout_views.py', 'experiments/src/creator_eval/heldout_view_pose.py',
    'tests/test_heldout_view_pose.py', 'tests/test_rod_heldout_views.py',
    'experiments/src/creator_eval/rod_validation_evidence.py', 'configs/rod_heldout_views_annotations_v1.json']
POLICY = dict(sift=SIFT, ratio=.75, minimum_old_views=2, target_exclusion_half_width_px=64,
    new_training_min_y=192., new_validation_max_y_exclusive=128., target_threshold_px=2.,
    pose_intrinsics='Geometric mean of old focal entries; mean principal point; same fixed lens assumption',
    map='Old fitted training points remain fixed; validation points triangulated only from old validation observations',
    identity='New assistant-read RGB membership clicks; never world projections or cross-view point identities',
    independence='New images and target clicks; localization remains conditional on old map and K. SIFT descriptor support can cross pixel masks.',
    no_geometry_refit=True, no_old_rejection_promotion=True)


def block_truth(event, arguments):
    reject_truth_open(event, arguments)
    if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0]).replace('\\', '/').lower()
        if '/docs/experiments/results/' in path:
            raise PermissionError('Physical result files are evaluation-only')


def consensus_matches(graph, pair_ids, new_xy):
    """Two old views must name the same track. Conflicts do not get a best-match vote."""
    owner = {tuple(node): tid for tid, track in enumerate(graph) for node in track}
    claims = defaultdict(list)
    for view, pairs in enumerate(pair_ids):
        for old_id, new_id in pairs:
            tid = owner.get((view, int(old_id)))
            if tid is not None:
                claims[int(new_id)].append((view, tid))
    rows = []
    for feature, values in sorted(claims.items()):
        tids, views = {t for _, t in values}, {v for v, _ in values}
        state = 'matched' if len(tids) == 1 and len(views) >= 2 else 'excluded'
        rows.append(dict(new_feature_id=feature, xy=new_xy[feature], old_track_ids=sorted(tids),
            old_view_ids=sorted(views), state=state,
            reason=None if state == 'matched' else 'conflict_or_fewer_than_two_source_views'))
    counts = Counter(r['old_track_ids'][0] for r in rows if r['state'] == 'matched')
    for row in rows:
        if row['state'] == 'matched' and counts[row['old_track_ids'][0]] != 1:
            row.update(state='excluded', reason='multiple_new_features_claim_one_old_track')
    return rows


def split_role(old_label, xy):
    if old_label == 'train' and xy[1] >= 192:
        return 'train'
    if old_label == 'validation' and xy[1] < 128:
        return 'validation'
    return 'excluded'


def shared_k(values):
    values = np.asarray(values, float)
    focal = float(np.exp(np.mean(np.log(values[:, [0, 1], [0, 1]]))))
    k = np.eye(3)
    k[0, 0] = k[1, 1] = focal
    k[:2, 2] = np.mean(values[:, :2, 2], axis=0)
    return k


def final_state(identity, camera, pose_checks, target):
    if identity != 'accepted':
        return 'original_identity_not_accepted'
    if camera != 'candidate_camera_correction':
        return 'withheld_original_camera'
    if not all(pose_checks):
        return 'unresolved_new_pose'
    return {'supported': 'retained_candidate', 'contradicted': 'rejected_new_target'}.get(target, 'unresolved_new_target')


def checked():
    config = read_json(RUN / 'method_config.json')
    assert config['run_id'] == RUN_ID and config['policy'] == POLICY
    assert digest(INPUTS / 'manifest.json') == config['input_sha256']
    for name, sha in config['source_sha256'].items():
        assert digest(ROOT / name) == digest(RUN / 'source_snapshot' / name) == sha, name
    for name, sha in config['receipts'].items():
        assert digest(ROOT / name) == sha, name
    return config, read_json(INPUTS / 'manifest.json')


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError('Preserve previous attempts; choose a new run')
    scene_frozen = read_json(SCENE / 'prepared.json')
    scene_manifest = read_json(SCENE_INPUTS / 'manifest.json')
    assert digest(SCENE_INPUTS / 'manifest.json') == scene_frozen['input_sha256']
    annotations = read_json(ANNOTATIONS)
    seal = read_json(SCENE / 'annotations-frozen.json')
    assert seal['annotations_sha256'] == digest(ANNOTATIONS) and seal['before_any_new_pose_or_projection']
    assert seal['scene_prepared_sha256'] == digest(SCENE / 'prepared.json')
    assert annotations['source_input_sha256'] == scene_frozen['input_sha256']
    assert annotations['source'] == 'assistant_visual_rgb_only_before_new_pose_or_projection'
    assert not annotations['ground_truth_used'] and not annotations['predicted_geometry_used']
    assert {q['case_id'] for q in annotations['queries']} == {'r01', 'r02', 'r03'}
    assert len(annotations['queries']) == 3
    parent, parent_config, training_manifest = checked_training()
    env_run, env_config, envelope = checked_envelope('replay')
    index = read_json(parent / 'inference.json')
    assert index['config_sha256'] == digest(parent / 'method_config.json')
    receipts = dict(env_config['receipts'])
    for path in (parent / 'inference.json', env_run / 'method_config.json',
                 ROOT / 'data/inputs' / env_config['run_id'] / 'manifest.json',
                 SCENE / 'prepared.json', SCENE / 'annotations-frozen.json', SCENE_INPUTS / 'manifest.json', ANNOTATIONS):
        receipts[path.relative_to(ROOT).as_posix()] = digest(path)
    tasks = []
    for old in training_manifest['tasks']:
        if old['case_id'] == 'r04':
            continue
        cid, parent_id = old['case_id'], old['parent']
        old_run, old_inputs, _, old_manifest = checked_run(parent_id)
        case = next(c for c in old_manifest['cases'] if c['case_id'] == cid)
        tracks = old_inputs / case['tracks_path']
        assert digest(tracks) == case['tracks_sha256']
        receipts[tracks.relative_to(ROOT).as_posix()] = digest(tracks)
        new_case = next(c for c in scene_manifest['cases'] if c['case_id'] == cid)
        query = next(q for q in annotations['queries'] if q['case_id'] == cid)
        assert len(new_case['frames']) == len(query['anchors']) == 2
        assert {a['view_id'] for a in query['anchors']} == {f['view_id'] for f in new_case['frames']}
        for frame in [*case['frames'], *new_case['frames']]:
            assert digest(ROOT / frame['rgb']) == frame['rgb_sha256']
            receipts[frame['rgb']] = frame['rgb_sha256']
        assert not {f['rgb_sha256'] for f in case['frames']} & {f['rgb_sha256'] for f in new_case['frames']}
        for anchor in query['anchors']:
            f = next(f for f in new_case['frames'] if f['view_id'] == anchor['view_id'])
            assert anchor['source_sha256'] == f['rgb_sha256']
            assert anchor['uncertainty_xy_px'] == [.5, .5]
            assert 0 <= anchor['xy'][0] < f['size_wh'][0] and 0 <= anchor['xy'][1] < f['size_wh'][1]
        entries = [r for r in index['records'] if (r['parent'], r['case_id']) == (parent_id, cid)]
        assert sorted(r['condition'] for r in entries) == sorted(CONDITIONS)
        maps = []
        for entry in entries:
            path = parent / entry['path']
            assert digest(path) == entry['sha256']
            record = read_json(path)
            assert record['task_sha256'] == canonical_hash(old)
            receipts[path.relative_to(ROOT).as_posix()] = entry['sha256']
            maps.append(dict(condition=entry['condition'], path=path.relative_to(ROOT).as_posix(), sha256=entry['sha256']))
        families = [t for t in envelope['tasks'] if (t['parent'], t['case_id']) == (parent_id, cid)]
        assert {f['method'] for f in families} == {'baseline', 'cylinder_support'} and len(families) == 2
        tasks.append(dict(task_id=parent_id + '-' + cid, parent=parent_id, case_id=cid,
            old_frames=case['frames'], new_frames=new_case['frames'], tracks_path=tracks.relative_to(ROOT).as_posix(),
            maps=maps, families=families, query=query))
    assert len(tasks) == 6
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    (RUN / 'matches').mkdir()
    (RUN / 'records').mkdir()
    sources = source_snapshot(RUN, set(parent_config['source_sha256']) | set(env_config['source_sha256']) |
        set(scene_frozen['source_sha256']) | set(SOURCES))
    write_json(INPUTS / 'manifest.json', dict(tasks=tasks, scene_id=SCENE_ID))
    write_json(RUN / 'method_config.json', dict(run_id=RUN_ID, source_sha256=sources, receipts=receipts,
        input_sha256=digest(INPUTS / 'manifest.json'), policy=POLICY,
        physical_source_sha256=PHYSICAL_SHA, annotations_frozen_before_inference=True,
        scope='Six known camera tasks on three old layouts, two new RGB views; no independent object claim'))
    print('HELDOUT_PREPARED 6 tasks, 30 fixed maps, 6 new RGB, 60 poses, 60 joint rod decisions', flush=True)


def features(frame, whole_height=False):
    assert digest(ROOT / frame['rgb']) == frame['rgb_sha256']
    with Image.open(ROOT / frame['rgb']) as im:
        rgb = np.asarray(im.convert('RGB'))
    assert list(rgb.shape[1::-1]) == frame['size_wh']
    (x0, y0), (x1, y1) = frame['guide_xyxy']
    guide_y = [0, frame['size_wh'][1]-1] if whole_height else [y0, y1]
    mask = exclusion_mask(frame['size_wh'], {'selected': [x0, x1]}, guide_y, 64)
    return detect_features(rgb, mask, 'SIFT', SIFT)


def match_task(task):
    tracks = read_json(ROOT / task['tracks_path'])
    old = [features(f) for f in task['old_frames']]
    for v, feature in enumerate(old):
        assert np.array_equal(feature['xy'], tracks['feature_xy'][v]), 'Old SIFT feature IDs changed'
    records = []
    for frame in task['new_frames']:
        new = features(frame, True)
        pairs = [mutual_ratio_matches(f, new, 'SIFT', .75)[0] for f in old]
        rows = consensus_matches(tracks['graph']['tracks'], pairs, new['xy'])
        for row in rows:
            row['role'] = split_role(tracks['band_labels'][row['old_track_ids'][0]], row['xy']) if row['state'] == 'matched' else 'excluded'
        records.append(dict(view_id=frame['view_id'], rgb_sha256=frame['rgb_sha256'], feature_xy=new['xy'],
            pair_ids=pairs, matches=rows, role_counts=dict(Counter(r['role'] for r in rows))))
    return dict(task_id=task['task_id'], old_feature_ids_exact=True, frames=records)


def map_observations(task, camera, matched):
    tracks = read_json(ROOT / task['tracks_path'])
    fit = camera['result']
    k, e = np.asarray(fit['intrinsics']), np.asarray(fit['extrinsics'])
    training = dict(zip(fit['kept_track_ids'], fit['points']))
    rows, missing = {'train': [], 'validation': []}, []
    for match in matched['matches']:
        role = match['role']
        if role == 'excluded':
            continue
        tid = match['old_track_ids'][0]
        if role == 'train':
            point = training.get(tid)
            if point is None:
                missing.append(dict(track_id=tid, role=role, reason='not_retained_in_this_fixed_training_map'))
                continue
        else:
            obs = [dict(view=v, xy=tracks['feature_xy'][v][f]) for v, f in tracks['graph']['tracks'][tid]]
            point = triangulate(obs, k, e)
            if not np.isfinite(point).all() or any(project(point[None], k[o['view']], e[o['view']])[1][0] <= 0 for o in obs):
                missing.append(dict(track_id=tid, role=role, reason='old_validation_triangulation_unavailable'))
                continue
        rows[role].append(dict(track_id=tid, xyz=point, xy=match['xy'], new_feature_id=match['new_feature_id']))
    assert not {r['track_id'] for r in rows['train']} & {r['track_id'] for r in rows['validation']}
    return rows, missing


def pose_for(task, camera, frame, matched, map_sha):
    rows, missing = map_observations(task, camera, matched)
    k = shared_k(camera['result']['intrinsics'])
    source = dict(source_kind='rgb', rgb_sha256=frame['rgb_sha256'], map_sha256=map_sha,
        intrinsics_sha256=canonical_hash(k), size_wh=frame['size_wh'])
    def arrays(role):
        r = rows[role]
        return np.asarray([x['xyz'] for x in r], float).reshape(-1, 3), np.asarray([x['xy'] for x in r], float).reshape(-1, 2), [x['track_id'] for x in r]
    xyz, xy, ids = arrays('train')
    proposal = propose_pose(xyz, xy, k, track_ids=ids,
        source={**source, 'observation_role': 'background_pose_training'})
    xyz, xy, ids = arrays('validation')
    verification = verify_pose(proposal, xyz, xy, track_ids=ids,
        source={**source, 'observation_role': 'background_pose_validation'})
    return dict(view_id=frame['view_id'], source=source, observations=rows, missing=missing,
                proposal=proposal, verification=verification)


def infer():
    import cv2
    sys.addaudithook(block_truth)
    config, manifest = checked()
    if (RUN / 'inference.json').exists():
        raise FileExistsError('Preserve inference')
    cv2.setNumThreads(1)
    started, receipts = time.perf_counter(), []
    for task in manifest['tasks']:
        matched = match_task(task)
        match_path = RUN / 'matches' / (task['task_id'] + '.json')
        write_json(match_path, matched)
        for entry in task['maps']:
            camera = read_json(ROOT / entry['path'])
            poses = [pose_for(task, camera, f, m, entry['sha256']) for f, m in zip(task['new_frames'], matched['frames'])]
            # A held-out target never touches localization or its model selection.
            # If pose validation is weak, the target check is diagnostic only.
            accepted_poses = [p['verification']['state'] == 'validated' and not any(x['role'] == 'validation' for x in p['missing']) for p in poses]
            frames = [dict(view_id=f['view_id'], size_wh=f['size_wh'], source_sha256=f['rgb_sha256'], source_kind='rgb',
                K_index=p['proposal']['K'], world_to_camera_cv=p['proposal']['E']) for f, p in zip(task['new_frames'], poses)]
            rods = []
            for family in task['families']:
                packet = next(p for p in family['packets'] if p['condition_id'] == entry['condition'])
                assert packet['parent_record_sha256'] == entry['sha256']
                identity = packet['identity'] or {'state': 'unavailable', 'segments': []}
                target = check_target_anchors(frames, identity['segments'], task['query']['anchors'], threshold_px=2.)
                state = final_state(identity['state'], packet['camera_decision']['state'], accepted_poses, target['state'])
                rods.append(dict(task_id=family['task_id'], method=family['method'], old_identity_state=identity['state'],
                    old_camera_state=packet['camera_decision']['state'], old_packet_sha256=canonical_hash(packet),
                    geometry_sha256=canonical_hash(identity['segments']), target_check=target, state=state))
            path = RUN / 'records' / (task['task_id'] + '-' + entry['condition'] + '.json')
            write_json(path, dict(task_id=task['task_id'], parent=task['parent'], case_id=task['case_id'], condition=entry['condition'],
                task_sha256=canonical_hash(task), fixed_map_sha256=entry['sha256'], match_sha256=digest(match_path),
                poses=poses, rods=rods, gt_read_during_inference=False, old_geometry_modified=False))
            receipts.append(dict(path=path.relative_to(RUN).as_posix(), sha256=digest(path), task_id=task['task_id'], condition=entry['condition']))
            print('NEW_VIEW', task['task_id'], entry['condition'], [p['verification']['state'] for p in poses], [r['state'] for r in rods], flush=True)
    assert len(receipts) == 30
    write_json(RUN / 'inference.json', dict(state='complete', records=receipts,
        source_sha256=config['source_sha256'], input_sha256=config['input_sha256'], config_sha256=digest(RUN / 'method_config.json'),
        elapsed_seconds=time.perf_counter()-started, gt_read_during_inference=False))


def audit_normal():
    config, manifest = checked()
    index = read_json(RUN / 'inference.json')
    assert index['state'] == 'complete' and index['source_sha256'] == config['source_sha256']
    assert index['input_sha256'] == config['input_sha256'] and index['config_sha256'] == digest(RUN / 'method_config.json')
    expected = {(t['task_id'], m['condition']) for t in manifest['tasks'] for m in t['maps']}
    assert len(index['records']) == 30 and {(r['task_id'], r['condition']) for r in index['records']} == expected
    receipts = {p.relative_to(ROOT).as_posix(): digest(p) for p in [RUN / 'method_config.json', RUN / 'inference.json', INPUTS / 'manifest.json']}
    records = []
    for item in index['records']:
        path = RUN / item['path']
        assert digest(path) == item['sha256']
        receipts[path.relative_to(ROOT).as_posix()] = digest(path)
        row = read_json(path)
        task = next(t for t in manifest['tasks'] if t['task_id'] == row['task_id'])
        assert row['task_sha256'] == canonical_hash(task)
        assert row['condition'] == item['condition'] and row['task_id'] == item['task_id']
        fixed = next(m for m in task['maps'] if m['condition'] == row['condition'])
        assert row['fixed_map_sha256'] == fixed['sha256'] == digest(ROOT / fixed['path'])
        match_path = RUN / 'matches' / (task['task_id'] + '.json')
        assert digest(match_path) == row['match_sha256']
        receipts[match_path.relative_to(ROOT).as_posix()] = digest(match_path)
        assert len(row['poses']) == len(row['rods']) == 2 and not row['gt_read_during_inference']
        assert [p['view_id'] for p in row['poses']] == [f['view_id'] for f in task['new_frames']]
        for pose, frame in zip(row['poses'], task['new_frames']):
            assert pose['source']['rgb_sha256'] == frame['rgb_sha256']
            assert pose['source']['map_sha256'] == fixed['sha256'] and pose['source']['size_wh'] == frame['size_wh']
            train, val = pose['observations']['train'], pose['observations']['validation']
            assert not {r['track_id'] for r in train} & {r['track_id'] for r in val}
            assert all(r['xy'][1] >= 192 for r in train) and all(r['xy'][1] < 128 for r in val)
        for rod in row['rods']:
            fam = next(f for f in task['families'] if f['task_id'] == rod['task_id'])
            packet = next(p for p in fam['packets'] if p['condition_id'] == row['condition'])
            assert canonical_hash(packet) == rod['old_packet_sha256']
            assert [r['view_id'] for r in rod['target_check']['per_anchor']] == [a['view_id'] for a in task['query']['anchors']]
            assert canonical_hash((packet['identity'] or {'segments': []})['segments']) == rod['geometry_sha256']
        records.append(row)
    return config, index, records, receipts


def pre():
    sys.addaudithook(block_truth)
    assert not PUBLIC.exists() and not (RUN / 'evaluation.json').exists()
    _, _, records, receipts = audit_normal()
    write_json(RUN / 'before-evaluation.json', dict(receipts=receipts, camera_count=60,
        rod_count=sum(len(r['rods']) for r in records), no_truth_read=True))
    print('NEW_VIEW_PRE', len(receipts), 'receipts; 60 poses; 60 joint rods', flush=True)


def evaluate():
    config, index, records, receipts = audit_normal()
    assert read_json(RUN / 'before-evaluation.json')['receipts'] == receipts
    assert digest(PHYSICAL) == config['physical_source_sha256']
    physical = read_json(PHYSICAL)
    env_run, env_config, _ = checked_envelope('replay')
    assert physical['run_id'] == env_config['run_id'] and physical['config'] == env_config
    assert physical['inference_sha256'] == digest(env_run / 'inference.json')
    scene_frozen = read_json(SCENE / 'prepared.json')
    truth_root = ROOT / 'data/eval_gt' / SCENE_ID
    assert digest(truth_root / 'manifest.json') == scene_frozen['truth_sha256']
    truth = read_json(truth_root / 'manifest.json')
    assert truth['input_sha256'] == scene_frozen['input_sha256'] == digest(SCENE_INPUTS / 'manifest.json')
    scene_cases = {c['case_id']: c for c in read_json(SCENE_INPUTS / 'manifest.json')['cases']}
    rows, pose_rows = [], []
    for record in records:
        gt = next(c for c in truth['cases'] if c['case_id'] == record['case_id'])
        family = next(f for f in physical['rows'] if (f['parent'], f['case_id'], f['method']) == (record['parent'], record['case_id'], 'baseline'))
        transform = np.asarray(family['physical']['alignment']['prediction_world_to_gt_world'])
        scale = np.cbrt(np.linalg.det(transform[:3, :3]))
        rotation = transform[:3, :3] / scale
        for i, pose in enumerate(record['poses']):
            e = pose['proposal'].get('E')
            camera = gt['cameras'][i]
            source_frame = scene_cases[record['case_id']]['frames'][i]
            assert pose['view_id'] == camera['view_id'] == gt['frames'][i]['view_id'] == source_frame['view_id']
            assert pose['source']['rgb_sha256'] == source_frame['rgb_sha256'] == gt['frames'][i]['rgb_sha256']
            location_error, rotation_error = None, None
            if e is not None:
                e = np.asarray(e)
                center = -e[:, :3].T @ e[:, 3]
                aligned = transform[:3, :3] @ center + transform[:3, 3]
                true_e = np.asarray(camera['world_to_camera_cv'])[:3]
                true_center = -true_e[:, :3].T @ true_e[:, 3]
                location_error = float(np.linalg.norm(aligned-true_center))
                relative = (e[:, :3] @ rotation.T) @ true_e[:, :3].T
                rotation_error = float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
            pose_rows.append(dict(task_id=record['task_id'], condition=record['condition'], view_id=pose['view_id'],
                proposal=pose['proposal'], verification=pose['verification'], missing=pose['missing'],
                center_error_m=location_error, rotation_error_degrees=rotation_error,
                alignment='Unchanged old full-control camera-only Sim3; no new-view realignment'))
        for rod in record['rods']:
            source = next(f for f in physical['rows'] if f['task_id'] == rod['task_id'])
            old = next(r for r in source['physical']['rows'] if r['condition_id'] == record['condition'])
            assert old['identity_state'] == rod['old_identity_state'] and old['camera_decision'] == rod['old_camera_state']
            retained = rod['state'] == 'retained_candidate'
            metrics = old.get('curve_metrics') or old.get('metrics')
            rows.append(dict(**rod, parent=record['parent'], case_id=record['case_id'], condition=record['condition'],
                physical=old, emitted_recovery=metrics['recovery_fraction'] if retained and metrics else (None if retained else 0.),
                emitted_precision=metrics['precision_fraction'] if retained and metrics else None))
    result = dict(state='complete', run_id=RUN_ID, config=config, input_sha256=config['input_sha256'],
        inference_sha256=digest(RUN / 'inference.json'), before_sha256=digest(RUN / 'before-evaluation.json'),
        truth_sha256=digest(truth_root / 'manifest.json'), physical_source_sha256=digest(PHYSICAL),
        pose_rows=pose_rows, rows=rows, states=dict(Counter(r['state'] for r in rows)),
        elapsed_seconds=index['elapsed_seconds'], no_automatic_promotion=True,
        scope='Three known Blender layouts; six new RGB, six old camera tasks, 30 correlated maps, 60 joint decisions. New-view PnP inherits old map/K; no independent absolute camera or real-photo claim.')
    assert len(rows) == len(pose_rows) == 60
    write_json(RUN / 'evaluation.json', result)
    write_json(PUBLIC, result)
    print('NEW_VIEW_EVALUATED', result['states'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'infer', 'pre', 'evaluate'))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    globals()[args.stage]()
