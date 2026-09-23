"""Paired RGB-only camera and rod experiments on frozen reference-board scenes."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import (
    ROOT,
    checked_run,
    digest,
    load_case,
    locations,
    read_json,
    write_json,
)
from run_rod_candidate_ablation import canonical_hash
from run_rod_foreground_identity import compact_bundle, make_views
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.rod_candidate_association import associate_multiview_lines_cached
from creator_eval.rod_candidate_pool import enumerate_image_line_pool
from creator_eval.rod_cylinder_support import select_supported_cylinders
from creator_eval.rod_foreground_identity import (
    build_anchor_proposals,
    select_foreground_identity,
    validate_anchors,
)
from creator_eval.rod_observations import extract_rod_observations

CONFIG = ROOT / 'configs/rod_reference_v1.json'
RUN_ID = 'rod-reference-bundle-v1-20260923'


def prepare(path):
    config = read_json(path)
    scene, scene_inputs, _, old, manifest = checked_scene(config['run_id'])
    index = read_json(scene / 'inference.json')
    if index['state'] != 'complete' or index['input_sha256'] != old['input_sha256'] or index['source_sha256'] != old['source_sha256']:
        raise ValueError('Detached RGB tracks')
    run, inputs = locations(config['bundle_run_id'])
    if run.exists() or inputs.exists():
        raise FileExistsError('Preserve earlier calibration attempts')
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    cases = []
    for case in manifest['cases']:
        cid = case['case_id']
        receipt = next(r for r in index['records'] if r['case_id'] == cid and r['detector'] == 'SIFT')
        if digest(scene / receipt['path']) != receipt['sha256']:
            raise ValueError('RGB tracks changed')
        record = read_json(scene / receipt['path'])
        dest = inputs / (cid + '-tracks.json')
        write_json(dest, {key: record[key] for key in ('case_id', 'feature_xy', 'graph', 'band_labels', 'availability', 'coverage', 'split', 'nonplanar_validated_pairs')})
        cases.append(dict(case_id=cid, frames=case['frames'], source_run_id=config['run_id'],
                          tracks_path=dest.name, tracks_sha256=digest(dest), reuse=None))
    write_json(inputs / 'manifest.json', dict(cases=cases, method=config['calibration'], rod_method=config['rod_method']))
    write_json(run / 'protocol.json', config)
    hashes = source_snapshot(run, old['source_sha256'])
    write_json(run / 'prepared.json', dict(run_id=config['bundle_run_id'], source_sha256=hashes,
        input_sha256=digest(inputs / 'manifest.json'), protocol_sha256=digest(run / 'protocol.json'),
        scene_prepared_sha256=digest(scene / 'prepared.json'), scene_inference_sha256=digest(scene / 'inference.json'),
        scene_input_sha256=digest(scene_inputs / 'manifest.json')))
    print('REFERENCE_BUNDLE_PREPARED', len(cases), flush=True)


def seal_annotations(run_id, path):
    run, _, frozen, manifest = checked_run(run_id)
    if (run / 'predictions.json').exists() or any(run.glob('*/prediction')):
        raise ValueError('Record RGB-only annotations before model prediction')
    annotations = read_json(path)
    if annotations['ground_truth_used_for_coordinates'] or annotations['predicted_geometry_used_for_coordinates']:
        raise ValueError('No projected truth or model clicks')
    expected = {c['case_id'] for c in manifest['cases']}
    if {q['case_id'] for q in annotations['queries']} != expected or len(annotations['queries']) != len(expected):
        raise ValueError('Exactly one declared target query per scene')
    for case in manifest['cases']:
        query = next(q for q in annotations['queries'] if q['case_id'] == case['case_id'])
        frames = {f['view_id']: f for f in case['frames']}
        if len(query['anchors']) != 2 or len({a['view_id'] for a in query['anchors']}) != 2:
            raise ValueError('Exactly two distinct annotated views')
        for anchor in query['anchors']:
            frame = frames[anchor['view_id']]
            if anchor['rgb_sha256'] != frame['rgb_sha256'] or digest(ROOT / frame['rgb']) != frame['rgb_sha256']:
                raise ValueError('Anchor image identity mismatch')
            x, y = anchor['xy']
            if not (0 <= x < frame['size_wh'][0] and 0 <= y < frame['size_wh'][1]):
                raise ValueError('Anchor outside image')
    write_json(run / 'annotations.json', annotations)
    write_json(run / 'annotations-frozen.json', dict(sha256=digest(run / 'annotations.json'), input_sha256=frozen['input_sha256'], before_prediction=True))


def checked_annotations(run, frozen):
    receipt = read_json(run / 'annotations-frozen.json')
    if receipt['input_sha256'] != frozen['input_sha256'] or digest(run / 'annotations.json') != receipt['sha256'] or not receipt['before_prediction']:
        raise ValueError('Annotation freeze changed')
    return read_json(run / 'annotations.json')


def pools(run_id):
    run, _, frozen, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    checked_annotations(run, frozen)
    method, records = manifest['rod_method'], []
    folder = run / 'rod-pools'
    folder.mkdir(exist_ok=False)
    for case in manifest['cases']:
        frames = []
        for frame in case['frames']:
            if digest(ROOT / frame['rgb']) != frame['rgb_sha256']:
                raise ValueError('RGB changed')
            with Image.open(ROOT / frame['rgb']) as image:
                rgb = np.asarray(image.convert('RGB'))
            raw = extract_rod_observations(rgb, frame['guide_xyxy'], method['observation'])
            pool = enumerate_image_line_pool(raw, method['image_hypotheses'], require_refit_support=True)
            frames.append({**frame, 'observations': raw, 'pool': pool})
        path = folder / (case['case_id'] + '.json')
        write_json(path, dict(case_id=case['case_id'], frames=frames))
        records.append(dict(case_id=case['case_id'], path=path.relative_to(run).as_posix(), sha256=digest(path)))
        print('REFERENCE_POOL', case['case_id'], [len(f['pool']['candidates']) for f in frames], flush=True)
    write_json(run / 'rod-pools.json', dict(records=records, input_sha256=frozen['input_sha256'], source_sha256=frozen['source_sha256']))


def rod_variant(run_id, cid, camera_receipt):
    run, _, frozen, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    annotations = checked_annotations(run, frozen)
    _, _, _, _, k, e = load_case(run_id, cid)
    camera_name, decision = 'initial', None
    if camera_receipt is not None:
        if digest(run / camera_receipt['path']) != camera_receipt['sha256']:
            raise ValueError('Corrected cameras changed')
        camera = read_json(run / camera_receipt['path'])
        camera_name, decision = camera_receipt['variant'], camera['decision']
        if 'extrinsics' not in camera['result']:
            return dict(case_id=cid, camera_variant=camera_name, state='no_candidate_cameras')
        k, e = np.asarray(camera['result']['intrinsics']), np.asarray(camera['result']['extrinsics'])
    index = read_json(run / 'rod-pools.json')
    if index['input_sha256'] != frozen['input_sha256'] or index['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Detached rod observations')
    entry = next(r for r in index['records'] if r['case_id'] == cid)
    if digest(run / entry['path']) != entry['sha256']:
        raise ValueError('Rod pool changed')
    pool = read_json(run / entry['path'])
    frames = [{**f, 'K_index': ck, 'world_to_camera_cv': ce} for f, ck, ce in zip(pool['frames'], k, e)]
    method = manifest['rod_method']
    views, raw = make_views(frames, method['candidate_cap']), [f['observations'] for f in frames]
    query = next(q for q in annotations['queries'] if q['case_id'] == cid)
    validate_anchors(query['anchors'], views, method['identity_policy'])
    started = time.perf_counter()
    association = associate_multiview_lines_cached(views, method['association_common'])
    if not association['search_complete']:
        raise ValueError('Incomplete retained-pool search')
    cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
    records, bundles = [], {}
    for name, proposal in (('baseline', association), ('cylinder_support', cylinder)):
        bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method['extent'], method['cylinder_screen'] if name == 'cylinder_support' else None))
        before = canonical_hash(bundle)
        identity = select_foreground_identity(bundle, views, raw, query['anchors'], method['identity_policy'])
        if canonical_hash(bundle) != before:
            raise ValueError('Click selection mutated geometry')
        records.append(dict(method=name, identity=identity))
        bundles[name] = bundle
    path = run / 'rod-results' / (cid + '-' + camera_name + '.json')
    write_json(path, dict(case_id=cid, camera_variant=camera_name, camera_decision=decision, camera_receipt=camera_receipt,
        intrinsics=k, extrinsics=e, pool_sha256=entry['sha256'], query=query, records=records, bundles=bundles,
        elapsed_seconds=time.perf_counter() - started, gt_read_during_inference=False,
        scope='Research counterfactual for every numerical camera candidate, including withheld ones; no dense patch applied'))
    print('REFERENCE_RODS', cid, camera_name, [r['identity']['state'] for r in records], flush=True)
    return dict(case_id=cid, camera_variant=camera_name, path=path.relative_to(run).as_posix(), sha256=digest(path))


def rods(run_id):
    run, _, frozen, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    index = read_json(run / 'inference.json')
    if index['state'] != 'complete' or index['input_sha256'] != frozen['input_sha256'] or index['source_sha256'] != frozen['source_sha256'] or index['predictions_sha256'] != digest(run / 'predictions.json'):
        raise ValueError('Camera inference is detached')
    tasks = [(c['case_id'], None) for c in manifest['cases']]
    tasks.extend((r['case_id'], r) for r in index['records'])
    (run / 'rod-results').mkdir(exist_ok=False)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=manifest['method']['parallel_workers']) as executor:
        futures = [executor.submit(rod_variant, run_id, *task) for task in tasks]
        records = [f.result() for f in as_completed(futures)]
    write_json(run / 'rod-inference.json', dict(state='complete', records=sorted(records, key=lambda r: (r['case_id'], r['camera_variant'])),
        input_sha256=frozen['input_sha256'], source_sha256=frozen['source_sha256'], camera_inference_sha256=digest(run / 'inference.json'),
        annotations_sha256=digest(run / 'annotations.json'), pools_sha256=digest(run / 'rod-pools.json'),
        elapsed_seconds=time.perf_counter() - started, gt_read_during_inference=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'annotations', 'predict', 'bundle', 'pools', 'rods'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--annotations', type=Path, default=ROOT / 'configs/rod_reference_annotations_v1.json')
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == 'prepare':
        prepare(args.config)
    elif args.stage == 'annotations':
        seal_annotations(args.run_id, args.annotations)
    elif args.stage == 'predict':
        import run_camera_bundle_pilot as base
        root, _, frozen, _ = checked_run(args.run_id)
        checked_annotations(root, frozen)
        base.predict(args.run_id)
    elif args.stage == 'bundle':
        import run_camera_bundle_pilot as base
        from run_camera_bundle_shared import infer_variant
        base.infer_variant = infer_variant
        base.infer(args.run_id)
    elif args.stage == 'pools':
        pools(args.run_id)
    else:
        rods(args.run_id)
