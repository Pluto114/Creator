"""Independent mesh, camera-gauge, heldout-point and finite-rod scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_rod_reference import checked_scene
from run_camera_bundle_pilot import camera_score, checked_run, load_case
from run_rod_reference import ROOT, RUN_ID, checked_annotations, digest, read_json, write_json

# isort: split
from creator_eval.background_correspondences import mesh_direction
from creator_eval.camera_bundle import (
    centers,
    observations_from_record,
    triangulate,
    validate_heldout,
    validation_plan,
)
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.native_diagnostics import distribution


def require_plan(actual, expected):
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Incomplete, duplicate, or unexpected experimental condition')


def point_errors(plan, k, e, truth, mesh, alignment):
    matrix = np.array(alignment['prediction_world_to_gt_world'])
    errors = []
    for track in plan:
        first = track['triangulation'][0]
        camera = truth['cameras'][first['view']]
        uv = np.asarray(first['xy'])[None]
        depth, sid, _ = mesh.cast(camera, uv + .5)
        inv = np.linalg.inv(np.asarray(camera['world_to_camera_cv']))
        world = (np.c_[uv, np.ones(1)] @ np.linalg.inv(camera['K_index']).T * depth[:, None]) @ inv[:3, :3].T + inv[:3, 3]
        point = triangulate(track['triangulation'], k, e)
        aligned = point @ matrix[:3, :3].T + matrix[:3, 3]
        error = float(np.linalg.norm(aligned - world[0])) if sid[0] and np.isfinite(aligned).all() else None
        errors.append(dict(track_id=track['track_id'], error_m=error))
    values = np.array([r['error_m'] if r['error_m'] is not None else np.nan for r in errors])
    return dict(total=len(values), unknown=int((~np.isfinite(values)).sum()), summary=distribution(values), rows=errors)


def physical_tracks(record, cameras, mesh):
    checks = [[] for _ in record['graph']['tracks']]
    for pair in record['pairs']:
        a, b = pair['first_view'], pair['second_view']
        xy_a, xy_b = np.asarray(pair['first_xy']).reshape(-1, 2), np.asarray(pair['second_xy']).reshape(-1, 2)
        if not len(xy_a):
            continue
        forward = mesh_direction(xy_a, xy_b, cameras[a], cameras[b], mesh.cast, .002)
        reverse = mesh_direction(xy_b, xy_a, cameras[b], cameras[a], mesh.cast, .002)
        av, bv = forward['source_point_visible_in_target'], reverse['source_point_visible_in_target']
        error = np.fmax(forward['transfer_error_px'], reverse['transfer_error_px'])
        correct, wrong = av & bv & (error <= 2), (av | bv) & (error > 5)
        for i, tid in enumerate(pair['track_ids']):
            if tid >= 0:
                checks[tid].append('correct' if correct[i] else ('wrong' if wrong[i] else 'indeterminate'))
    labels = ['wrong' if 'wrong' in values else ('correct' if values and all(v == 'correct' for v in values) else 'indeterminate') for values in checks]
    return dict(all=dict(Counter(labels)), by_split={name: dict(Counter(label for label, split in zip(labels, record['band_labels']) if split == name)) for name in ('train', 'validation', 'excluded')})


def evaluate(run_id, output):
    from thin_pack_gt import MeshRays
    run, _, frozen, manifest = checked_run(run_id)
    if digest(run / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Evaluation protocol changed')
    config = read_json(run / 'protocol.json')
    scene, _, truth_root, old, _ = checked_scene(config['run_id'])
    if digest(scene / 'prepared.json') != frozen['scene_prepared_sha256'] or digest(scene / 'inference.json') != frozen['scene_inference_sha256']:
        raise ValueError('Scene parent changed')
    if digest(truth_root / 'manifest.json') != old['truth_sha256'] or digest(truth_root / 'artifact_hashes.json') != old['truth_artifacts_sha256']:
        raise ValueError('Truth index changed')
    for name, sha in read_json(truth_root / 'artifact_hashes.json').items():
        if digest(truth_root / name) != sha:
            raise ValueError('Truth artifact changed: ' + name)
    truth_manifest = read_json(truth_root / 'manifest.json')
    assert truth_manifest['input_sha256'] == old['input_sha256']
    truths = {c['case_id']: c for c in truth_manifest['cases']}
    meshes = {cid: MeshRays(truth_root / c['mesh_path']) for cid, c in truths.items()}
    index, rod_index = read_json(run / 'inference.json'), read_json(run / 'rod-inference.json')
    for result in (index, rod_index):
        if result['state'] != 'complete' or result['input_sha256'] != frozen['input_sha256'] or result['source_sha256'] != frozen['source_sha256'] or result['gt_read_during_inference']:
            raise ValueError('Detached or privileged inference')
    assert index['predictions_sha256'] == digest(run / 'predictions.json')
    assert rod_index['camera_inference_sha256'] == digest(run / 'inference.json')
    assert rod_index['annotations_sha256'] == digest(run / 'annotations.json')
    assert rod_index['pools_sha256'] == digest(run / 'rod-pools.json')
    checked_annotations(run, frozen)
    raw_tracks, context, track_rows = {}, {}, []
    for case in manifest['cases']:
        cid = case['case_id']
        _, _, tracks, method, k, e = load_case(run_id, cid)
        raw = read_json(scene / (cid + '-SIFT.json'))
        track_receipt = next(r for r in read_json(scene / 'inference.json')['records'] if r['case_id'] == cid)
        assert digest(scene / track_receipt['path']) == track_receipt['sha256']
        raw_tracks[cid] = raw
        train = observations_from_record(tracks, 'train')
        val = observations_from_record(tracks, 'validation')
        assert not {t['track_id'] for t in train} & {t['track_id'] for t in val}
        plan = validation_plan(val, e)
        context[cid] = (k, e, plan, train)
        track_rows.append(dict(case_id=cid, availability=tracks['availability'], split=tracks['split'], coverage=tracks['coverage'],
                               physical=physical_tracks(raw, truths[cid]['cameras'], meshes[cid])))
    eligible = {cid for cid, raw in raw_tracks.items() if raw['availability'] == 'eligible_for_further_validation'}
    expected = {(cid, c['id'], v['id']) for cid in eligible for c in manifest['method']['conditions'] for v in manifest['method']['variants']}
    require_plan([(r['case_id'], r['condition'], r['variant']) for r in index['records']], expected)
    require_plan([r['case_id'] for r in index['skipped']], set(truths) - eligible)
    camera_rows, scores = [], {}
    entries = [dict(case_id=cid, variant='initial', path=None) for cid in truths] + index['records']
    for entry in entries:
        cid, name = entry['case_id'], entry['variant']
        initial_k, initial_e, plan, train = context[cid]
        k, e, decision, audit, solver = initial_k, initial_e, None, {}, None
        if entry['path']:
            assert digest(run / entry['path']) == entry['sha256']
            record = read_json(run / entry['path'])
            assert record['validation_plan'] == plan
            assert record['before'] == validate_heldout(initial_k, initial_e, plan)
            assert not record['gt_read_during_inference'] and not record['dense_base_changed']
            result = record['result']
            decision = record['decision']
            if 'extrinsics' not in result:
                camera_rows.append(dict(case_id=cid, variant=name, decision=decision, state='numerical_failure'))
                continue
            k, e = np.asarray(result['intrinsics']), np.asarray(result['extrinsics'])
            assert record['after'] == validate_heldout(k, e, plan)
            first_error = float(abs(e[0] - initial_e[0]).max())
            baseline_error = float(abs(np.linalg.norm(centers(e)[-1] - centers(e)[0]) - np.linalg.norm(centers(initial_e)[-1] - centers(initial_e)[0])))
            assert first_error < 1e-6 and baseline_error < 1e-6
            kept, discarded = set(result['kept_track_ids']), set(result['discarded_track_ids'])
            assert not kept & discarded and kept | discarded == {t['track_id'] for t in train}
            audit = dict(first_pose_max_error=first_error, baseline_distance_error=baseline_error, train_validation_disjoint=True,
                         kept_tracks=len(kept), discarded_tracks=len(discarded), unchanged_holdout=True)
            solver = {key: result.get(key) for key in ('state', 'success', 'nfev', 'focal_scale', 'at_parameter_bound', 'warm_start')}
        score = camera_score(k, e, truths[cid]['cameras'], config['evaluation']['alignment'])
        scores[(cid, name)] = score
        camera_rows.append(dict(case_id=cid, variant=name, decision=decision, camera=score, solver=solver, audit=audit,
            heldout=validate_heldout(k, e, plan)['summary'], points=point_errors(plan, k, e, truths[cid], meshes[cid], score)))
    expected_rods = {(cid, 'initial') for cid in truths} | {(r['case_id'], r['variant']) for r in index['records']}
    require_plan([(r['case_id'], r['camera_variant']) for r in rod_index['records']], expected_rods)
    pools = {r['case_id']: r for r in read_json(run / 'rod-pools.json')['records']}
    rod_rows = []
    for entry in rod_index['records']:
        cid, name = entry['case_id'], entry['camera_variant']
        if 'path' not in entry:
            rod_rows.append(entry)
            continue
        assert digest(run / entry['path']) == entry['sha256']
        record = read_json(run / entry['path'])
        assert record['pool_sha256'] == pools[cid]['sha256'] == digest(run / pools[cid]['path'])
        assert not record['gt_read_during_inference']
        camera_reference = next(r for r in camera_rows if r['case_id'] == cid and r['variant'] == name)
        score = camera_score(np.asarray(record['intrinsics']), np.asarray(record['extrinsics']), truths[cid]['cameras'], config['evaluation']['alignment'])
        assert score == scores[(cid, name)]
        matrix = np.asarray(score['prediction_world_to_gt_world'])
        target = truths[cid]['declared']['target']
        truth_segments = target.get('segments', [target['endpoints']])
        require_plan([r['method'] for r in record['records']], {'baseline', 'cylinder_support'})
        for row in record['records']:
            segments = np.asarray(row['identity']['segments']).reshape(-1, 2, 3)
            aligned = segments @ matrix[:3, :3].T + matrix[:3, 3]
            metrics = curve_metrics(aligned, truth_segments, tolerance=config['evaluation']['curve_tolerance_m'], spacing=config['evaluation']['curve_spacing_m'])
            gap = truths[cid]['declared'].get('gap_segment')
            rod_rows.append(dict(case_id=cid, camera_variant=name, method=row['method'], state=row['identity']['state'],
                reason=row['identity']['reason'], segments=aligned, metrics=metrics, camera_decision=camera_reference['decision'],
                gap=gap_coverage(aligned, gap, tolerance=config['evaluation']['curve_tolerance_m'], spacing=config['evaluation']['curve_spacing_m']) if gap else None))
    report = dict(state='complete', run_id=run_id, scope=config['scope'], tracks=track_rows, cameras=camera_rows, rods=rod_rows,
        source_sha256=frozen['source_sha256'], evaluation_source_sha256=digest(Path(__file__)), protocol_sha256=frozen['protocol_sha256'],
        camera_inference_sha256=digest(run / 'inference.json'), rod_inference_sha256=digest(run / 'rod-inference.json'),
        truth_artifacts_sha256=old['truth_artifacts_sha256'], input_sha256=frozen['input_sha256'], annotations_sha256=digest(run / 'annotations.json'),
        independent_ray_samples=old['independent_ray_samples'], surface_id_mismatches=old['surface_id_mismatches'],
        max_blender_projection_error_px=max(f['projection_max_px'] for c in truths.values() for f in c['frames']),
        max_blender_ray_depth_error_m=max(f['ray_check']['max_z_error_m'] for c in truths.values() for f in c['frames']),
        target_visible_pixels={cid: [f['target_visible_pixels'] for f in c['frames']] for cid, c in truths.items()},
        dense_bases_and_patches_unchanged=True, no_independent_object_claim=True,
        timing_seconds=dict(camera=index['elapsed_seconds'], rods=rod_index['elapsed_seconds']))
    destination = ROOT / 'data/evaluation' / run_id
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', report)
    write_json(output, report)
    print('REFERENCE_EVALUATED', len(camera_rows), 'cameras;', len(rod_rows), 'rod conditions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/experiments/results/2026-09-23-rod-reference.json')
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    evaluate(args.run_id, args.output)
