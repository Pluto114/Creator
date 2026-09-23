"""Evaluation-only K/pose interventions. Never produce ordinary inputs or patches."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import camera_score, checked_run
from run_rod_foreground_identity import compact_bundle, make_views
from run_rod_reference import ROOT, RUN_ID, digest, read_json, write_json

# isort: split
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.rod_candidate_association import associate_multiview_lines_cached
from creator_eval.rod_cylinder_support import select_supported_cylinders
from creator_eval.rod_foreground_identity import build_anchor_proposals, select_foreground_identity

DIAGNOSTIC = 'rod-reference-camera-counterfactual-v1-20260923'


def intervene(cid, name):
    run, _, frozen, manifest = checked_run(RUN_ID)
    summary = read_json(ROOT / 'data/evaluation' / RUN_ID / 'summary.json')
    assert digest(run / 'rod-inference.json') == summary['rod_inference_sha256']
    config = read_json(run / 'protocol.json')
    _, _, truth_root, old, _ = checked_scene(config['run_id'])
    assert digest(truth_root / 'manifest.json') == old['truth_sha256']
    truth = next(c for c in read_json(truth_root / 'manifest.json')['cases'] if c['case_id'] == cid)
    original_path = run / 'rod-results' / (cid + '-common_K_focal_trimmed.json')
    receipt = next(r for r in read_json(run / 'rod-inference.json')['records'] if r['path'] == original_path.relative_to(run).as_posix())
    assert digest(original_path) == receipt['sha256']
    original = read_json(original_path)
    k, e = np.asarray(original['intrinsics']), np.asarray(original['extrinsics'])
    if name in ('true_K', 'true_K_and_pose'):
        k = np.asarray([c['K_index'] for c in truth['cameras']])
    if name in ('true_pose', 'true_K_and_pose'):
        e = np.asarray([c['world_to_camera_cv'] for c in truth['cameras']])[:, :3]
    pool_path = run / 'rod-pools' / (cid + '.json')
    assert digest(pool_path) == original['pool_sha256']
    frames = [{**f, 'K_index': ck, 'world_to_camera_cv': ce} for f, ck, ce in zip(read_json(pool_path)['frames'], k, e)]
    method = manifest['rod_method']
    views, raw = make_views(frames, method['candidate_cap']), [f['observations'] for f in frames]
    started = time.perf_counter()
    association = associate_multiview_lines_cached(views, method['association_common'])
    assert association['search_complete']
    cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
    score = camera_score(k, e, truth['cameras'], config['evaluation']['alignment'])
    matrix = np.asarray(score['prediction_world_to_gt_world'])
    target = truth['declared']['target']
    true_segments = target.get('segments', [target['endpoints']])
    rows = []
    for geometry, proposal in (('baseline', association), ('cylinder_support', cylinder)):
        bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method['extent'], method['cylinder_screen'] if geometry == 'cylinder_support' else None))
        result = select_foreground_identity(bundle, views, raw, original['query']['anchors'], method['identity_policy'])
        native = np.asarray(result['segments']).reshape(-1, 2, 3)
        aligned = native @ matrix[:3, :3].T + matrix[:3, 3]
        policy = config['evaluation']
        metrics = curve_metrics(aligned, true_segments, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m'])
        gap = truth['declared'].get('gap_segment')
        rows.append(dict(method=geometry, state=result['state'], reason=result['reason'], segments=aligned, metrics=metrics,
                         gap=gap_coverage(aligned, gap, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m']) if gap else None))
    output = ROOT / 'data/evaluation' / DIAGNOSTIC / (cid + '-' + name + '.json')
    write_json(output, dict(case_id=cid, intervention=name, rows=rows, camera=score, gt_used=True, ordinary_inference=False,
        original_sha256=digest(original_path), pool_sha256=original['pool_sha256'], input_sha256=frozen['input_sha256'], elapsed_seconds=time.perf_counter() - started))
    print('ORACLE_DIAGNOSTIC', cid, name, [r['state'] for r in rows], flush=True)
    return dict(case_id=cid, intervention=name, path=output.name, sha256=digest(output))


def run():
    parent, _, frozen, manifest = checked_run(RUN_ID)
    diagnostic = ROOT / '.runtime/experiments' / DIAGNOSTIC
    output = ROOT / 'data/evaluation' / DIAGNOSTIC
    diagnostic.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    tasks = [(c['case_id'], name) for c in manifest['cases'] for name in ('true_K', 'true_pose', 'true_K_and_pose')]
    sources = source_snapshot(diagnostic, [*frozen['source_sha256'], 'scripts/diagnose_rod_reference_cameras.py'])
    protocol = dict(run_id=DIAGNOSTIC, source_run_id=RUN_ID, tasks=tasks, source_sha256=sources,
                    parent_rod_sha256=digest(parent / 'rod-inference.json'), scope='GT-only diagnostic of frozen two-stage candidates; not model performance')
    write_json(diagnostic / 'protocol.json', protocol)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(intervene, *task) for task in tasks]
        records = [f.result() for f in as_completed(futures)]
    assert {(r['case_id'], r['intervention']) for r in records} == set(tasks) and len(records) == len(tasks)
    rows = []
    for receipt in sorted(records, key=lambda r: r['path']):
        assert digest(output / receipt['path']) == receipt['sha256']
        record = read_json(output / receipt['path'])
        rows.extend(dict(case_id=receipt['case_id'], intervention=receipt['intervention'], **row) for row in record['rows'])
    for name, sha in sources.items():
        assert digest(ROOT / name) == sha == digest(diagnostic / 'source_snapshot' / name)
    result = dict(state='complete', **protocol, records=records, rows=rows, elapsed_seconds=time.perf_counter() - started,
                  parent_unchanged=digest(parent / 'rod-inference.json') == protocol['parent_rod_sha256'])
    write_json(output / 'summary.json', result)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-reference-camera-counterfactual.json', result)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
