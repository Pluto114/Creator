"""Freeze and reconstruct finite RGB curves in an explicitly measured fixture frame.

Normal stages never read target truth or physical scores. An independent audit
must write its pre receipt before a different process evaluates physical errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/src'))
from creator_eval.rod_fixture_finite import (  # noqa: E402
    canonical_hash,
    extract_fixture_evidence,
    json_ready,
    method_policy,
    reconstruct_fixture_finite,
)

RUN_ID = 'rod-fixture-finite-v1-20260927'
SCENE_ID = 'rod-fixture-scenes-v1-20260927'
CAMERA_ID = 'rod-fixture-calibration-v1-20260927'
RUN = ROOT / '.runtime/experiments' / RUN_ID
INPUTS = ROOT / 'data/inputs' / RUN_ID
SCENE = ROOT / '.runtime/experiments' / SCENE_ID
SCENE_INPUTS = ROOT / 'data/inputs' / SCENE_ID
CAMERA = ROOT / '.runtime/experiments' / CAMERA_ID
CONFIG = ROOT / 'configs/rod_fixture_finite_v1.json'
SOURCES = ['scripts/run_rod_fixture_finite.py', 'experiments/src/creator_eval/rod_fixture_finite.py',
    'configs/rod_fixture_finite_v1.json', 'tests/test_rod_fixture_finite.py',
    'experiments/src/creator_eval/__init__.py', 'experiments/src/creator_eval/rod_candidate_association.py',
    'experiments/src/creator_eval/rod_candidate_pool.py', 'experiments/src/creator_eval/rod_candidate_extent.py',
    'experiments/src/creator_eval/rod_cylinder_gate.py', 'experiments/src/creator_eval/rod_cylinder_support.py',
    'experiments/src/creator_eval/rod_multiview_candidates.py', 'experiments/src/creator_eval/rod_observations.py',
    'experiments/src/creator_eval/rod_evidence.py', 'experiments/src/creator_eval/line_controls.py',
    'experiments/src/creator_eval/native_diagnostics.py', 'scripts/environment_paths.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes().decode('utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def reject_truth_open(event, arguments):
    if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes)):
        return
    item = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
    path = Path(item).resolve()
    for forbidden in (ROOT / 'data/eval_gt', ROOT / 'data/evaluation', ROOT / 'docs/experiments/results'):
        if path == forbidden or forbidden in path.parents:
            raise PermissionError('Normal finite reconstruction cannot read truth or physical results')
    if (path.name in {'protocol.json', 'generation-checks.json'} or path.name.endswith('render_request.json')
            or 'rendered-rgb' in path.parts):
        raise PermissionError('Normal finite reconstruction cannot read generation protocol')


def checked_sources(folder, prepared):
    for name, sha in prepared['source_sha256'].items():
        if digest(ROOT / name) != sha or digest(folder / 'source_snapshot' / name) != sha:
            raise ValueError('Frozen source changed: ' + name)


def checked():
    prepared = read_json(RUN / 'prepared.json')
    if prepared['run_id'] != RUN_ID:
        raise ValueError('Wrong finite run identity')
    checked_sources(RUN, prepared)
    if (digest(INPUTS / 'manifest.json') != prepared['input_sha256']
            or digest(RUN / 'method_config.json') != prepared['method_config_sha256']
            or digest(RUN / 'source_freeze.json') != prepared['source_freeze_sha256']):
        raise ValueError('Frozen finite input/method/source receipts changed')
    freeze = read_json(RUN / 'source_freeze.json')
    if freeze['source_sha256'] != prepared['source_sha256']:
        raise ValueError('Source freeze/prepared closure differs')
    for name, sha in prepared['receipts'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Parent/RGB receipt changed: ' + name)
    manifest, config = read_json(INPUTS / 'manifest.json'), read_json(RUN / 'method_config.json')
    method_policy(config['method'])
    if config['method_sha256'] != canonical_hash(config['method']):
        raise ValueError('Method digest differs from declared policy')
    return prepared, manifest, config


def prepare():
    sys.addaudithook(reject_truth_open)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError('Preserve every earlier finite attempt; choose a new run')
    from run_fixture_calibration import checked as checked_camera
    camera_prepared, camera_manifest = checked_camera()
    scene_prepared, scene_manifest = read_json(SCENE / 'prepared.json'), read_json(SCENE_INPUTS / 'manifest.json')
    checked_sources(CAMERA, camera_prepared)
    checked_sources(SCENE, scene_prepared)
    config, camera_index = read_json(CONFIG), read_json(CAMERA / 'inference.json')
    method_policy(config['method'])
    if (config['run_id'] != RUN_ID or config['scene_run_id'] != SCENE_ID or config['camera_run_id'] != CAMERA_ID
            or camera_prepared['run_id'] != CAMERA_ID or camera_prepared['scene_run_id'] != SCENE_ID
            or camera_prepared['scene_prepared_sha256'] != digest(SCENE / 'prepared.json')
            or camera_index['run_id'] != CAMERA_ID or camera_index['input_sha256'] != camera_prepared['input_sha256']
            or camera_index['source_sha256'] != camera_prepared['source_sha256'] or camera_index['gt_read'] is not False
            or camera_index['prepared_sha256'] != digest(CAMERA / 'prepared.json')
            or camera_index['cad_sha256'] != camera_prepared['cad_sha256']
            or digest(SCENE_INPUTS / 'manifest.json') != scene_prepared['input_sha256']):
        raise ValueError('Fixture camera/scene/config provenance mismatch')
    if config['method_sha256'] != canonical_hash(config['method']):
        raise ValueError('Predeclared finite method hash changed')
    reference = config['legacy_method_source']
    if digest(ROOT / reference['path']) != reference['sha256']:
        raise ValueError('Historical reference parameter source changed')
    case_ids, view_ids = config['case_ids'], config['view_ids']
    if case_ids != ['r01', 'r02', 'r03'] or view_ids != ['view_'+str(i).zfill(2) for i in range(5)]:
        raise ValueError('Keep all three cases and five fixed views')
    for records in (scene_manifest['cases'], camera_manifest['cases'], camera_index['cases']):
        if [c['case_id'] for c in records] != case_ids:
            raise ValueError('Case identities/order differ from the complete plan')
    cad = scene_manifest['declared_cad']
    if digest(ROOT / cad['path']) != cad['sha256'] or cad['sha256'] != camera_prepared['cad_sha256']:
        raise ValueError('Declared CAD receipt differs from calibration')
    receipts = {cad['path']: cad['sha256']}
    for path in (SCENE / 'prepared.json', SCENE_INPUTS / 'manifest.json', CAMERA / 'prepared.json',
                 CAMERA / 'inference.json', CONFIG, ROOT / reference['path']):
        receipts[path.relative_to(ROOT).as_posix()] = digest(path)
    # Parent receipts are kept byte-for-byte, rather than reducing their source
    # identity to a vague label such as "calibrated cameras".
    for parent in (camera_prepared, scene_prepared):
        for name, sha in parent.get('receipts', {}).items():
            if digest(ROOT / name) != sha:
                raise ValueError('Parent declared receipt changed: ' + name)
            receipts[name] = sha
    cases = []
    allowed_frame_fields = {'view_id', 'rgb', 'rgb_sha256', 'size_wh', 'guide_xyxy', 'guide_source'}
    for case, camera_case, camera_input in zip(scene_manifest['cases'], camera_index['cases'], camera_manifest['cases']):
        frames = case['frames']
        if any(set(f) != allowed_frame_fields for f in frames):
            raise ValueError('Unexpected normal RGB frame fields')
        if frames != camera_input['frames']:
            raise ValueError('Calibration and finite task did not consume identical RGB frames')
        if [f['view_id'] for f in frames] != view_ids or [c['view_id'] for c in camera_case['cameras']] != view_ids:
            raise ValueError('Frame/camera view IDs differ')
        if [f['view_id'] for f in camera_case['frames']] != view_ids:
            raise ValueError('Camera detector view order differs')
        for frame, detected in zip(frames, camera_case['frames']):
            if any(frame[k] != detected[k] for k in ('view_id', 'rgb_sha256', 'size_wh')):
                raise ValueError('Camera observations detach from the finite RGB')
            if frame['guide_xyxy'] != [[319.5, 20], [319.5, 459]]:
                raise ValueError('Do not adapt the declared strip to a predicted axis')
            if digest(ROOT / frame['rgb']) != frame['rgb_sha256']:
                raise ValueError('RGB changed: ' + frame['rgb'])
            receipts[frame['rgb']] = frame['rgb_sha256']
        cases.append(dict(case_id=case['case_id'], frames=frames, cameras=camera_case['cameras'],
            camera_case_sha256=canonical_hash(camera_case), camera_case_state=camera_case['state']))
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    (RUN / 'records').mkdir()
    sources = {}
    names = set(SOURCES) | set(camera_prepared['source_sha256']) | set(scene_prepared['source_sha256'])
    for name in sorted(names):
        path = ROOT / name
        target = RUN / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[name] = digest(path)
        if digest(target) != sources[name]:
            raise ValueError('Source copy mismatch')
    write_json(INPUTS / 'manifest.json', dict(run_id=RUN_ID, cases=cases, truth_excluded=True,
        coordinate_system='declared_metric_fixture_world', normal_camera_inference_sha256=digest(CAMERA / 'inference.json')))
    write_json(RUN / 'method_config.json', config)
    write_json(RUN / 'source_freeze.json', dict(source_sha256=sources, input_sha256=digest(INPUTS / 'manifest.json'),
        method_config_sha256=digest(RUN / 'method_config.json'), before_finite_inference=True))
    write_json(RUN / 'prepared.json', dict(run_id=RUN_ID, scene_run_id=SCENE_ID, camera_run_id=CAMERA_ID,
        source_sha256=sources, input_sha256=digest(INPUTS / 'manifest.json'),
        method_config_sha256=digest(RUN / 'method_config.json'), source_freeze_sha256=digest(RUN / 'source_freeze.json'),
        receipts=receipts, gt_read_during_prepare=False))
    print('FIXTURE_FINITE_PREPARED 3 cases x 2 methods, 15 RGB', flush=True)


def infer():
    sys.addaudithook(reject_truth_open)
    prepared, manifest, config = checked()
    if (RUN / 'inference.json').exists() or any((RUN / 'records').iterdir()):
        raise FileExistsError('Preserve old finite inference and partial attempts')
    started, records = time.perf_counter(), []
    for case in manifest['cases']:
        images = []
        for frame in case['frames']:
            if digest(ROOT / frame['rgb']) != frame['rgb_sha256']:
                raise ValueError('RGB changed')
            with Image.open(ROOT / frame['rgb']) as image:
                images.append(np.asarray(image.convert('RGB')))
        frames = extract_fixture_evidence(case['frames'], images, config['method'])
        result = reconstruct_fixture_finite(frames, case['cameras'], config['method'])
        path = RUN / 'records' / (case['case_id'] + '.json')
        write_json(path, dict(case_id=case['case_id'], input_case_sha256=canonical_hash(case),
            frames=frames, cameras=case['cameras'], result=result,
            gt_read_during_inference=False, target_identity_verified=False,
            output_scope='Finite curve in declared single-target strip, not dense/cloud/mesh repair'))
        records.append(dict(case_id=case['case_id'], path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
        print('FIXTURE_FINITE', case['case_id'], [(r['method'], r['state'], r['segment_count']) for r in result['methods']], flush=True)
    checked()
    if len(records) != 3:
        raise ValueError('Incomplete case plan')
    write_json(RUN / 'inference.json', dict(state='complete', run_id=RUN_ID, records=records,
        input_sha256=prepared['input_sha256'], source_sha256=prepared['source_sha256'],
        config_sha256=prepared['method_config_sha256'], gt_read=False, elapsed_seconds=time.perf_counter()-started,
        evaluation_status='Await independent before-evaluation receipt; no GT scoring in this process'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'infer'))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    globals()[args.stage]()
