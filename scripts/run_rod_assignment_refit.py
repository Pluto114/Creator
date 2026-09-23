"""Frozen no-GT candidate-refit comparison; evaluation runs in a separate process."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import checked_run
from run_rod_foreground_identity import compact_bundle, make_views
from run_rod_identity_stress import reject_truth_open
from run_rod_reference import ROOT, canonical_hash, digest, read_json, write_json

# isort: split
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.rod_assignment_refit import associate_refitted_variants
from creator_eval.rod_cylinder_support import select_supported_cylinders
from creator_eval.rod_foreground_identity import build_anchor_proposals, select_foreground_identity

RUN_ID = 'rod-assignment-refit-v1-20260923r3'
PARENTS = ['rod-reference-bundle-v1-20260923', 'rod-height-bundle-v1-20260923']
VARIANTS = ['refit_residual', 'refit_coverage', 'retained_assignments']
SHARE = ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-refit-r3.json'


def prepare():
    run = ROOT / '.runtime/experiments' / RUN_ID
    run.mkdir(exist_ok=False)
    names, tasks = set(), []
    for parent in PARENTS:
        folder, _, frozen, manifest = checked_run(parent)
        names.update(frozen['source_sha256'])
        index = read_json(folder / 'rod-inference.json')
        assert index['state'] == 'complete'
        selected = [r for r in index['records'] if r['camera_variant'] in ('initial', 'common_K_focal_trimmed')]
        assert len(selected) == 2 * len(manifest['cases'])
        for r in selected:
            assert digest(folder / r['path']) == r['sha256']
            tasks.append(dict(parent=parent, **r))
    names.update(['experiments/src/creator_eval/rod_assignment_refit.py', 'scripts/run_rod_assignment_refit.py'])
    sources = source_snapshot(run, names)
    write_json(run / 'method_config.json', dict(run_id=RUN_ID, tasks=tasks, source_sha256=sources,
        primary='refit_residual', exploratory=['refit_coverage', 'retained_assignments'], variants=VARIANTS,
        policy='same RGB pool, anchors, cameras and finite gates; refit fixed assignments before ranking; no automatic promotion',
        scope='known development replay; no new object or blind-test claim'))
    (run / 'records').mkdir()
    print('PREPARED', len(tasks), 'camera tasks', flush=True)


def checked():
    run = ROOT / '.runtime/experiments' / RUN_ID
    protocol = read_json(run / 'method_config.json')
    for name, sha in protocol['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(run / 'source_snapshot' / name), name
    return run, protocol


def worker(task):
    sys.addaudithook(reject_truth_open)
    run, protocol = checked()
    parent, _, _, manifest = checked_run(task['parent'])
    path = parent / task['path']
    assert digest(path) == task['sha256']
    original = read_json(path)
    pool_path = parent / 'rod-pools' / (task['case_id'] + '.json')
    assert digest(pool_path) == original['pool_sha256']
    frames = [{**f, 'K_index': k, 'world_to_camera_cv': e} for f, k, e in zip(
        read_json(pool_path)['frames'], original['intrinsics'], original['extrinsics'])]
    method = manifest['rod_method']
    views, raw = make_views(frames, method['candidate_cap']), [f['observations'] for f in frames]
    before = canonical_hash(dict(views=views, raw=raw, anchors=original['query']['anchors']))
    started = time.perf_counter()
    variants = associate_refitted_variants(views, method['association_common'])
    records, audits = [], {}
    for name in protocol['variants']:
        association = variants[name]
        cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
        audits[name] = dict(association=association, cylinder_support=cylinder['cylinder_support'],
                           cylinder_screen=cylinder['cylinder_screen'])
        for geometry, proposal in [('baseline', association), ('cylinder_support', cylinder)]:
            bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method['extent'],
                method['cylinder_screen'] if geometry == 'cylinder_support' else None))
            identity = select_foreground_identity(bundle, views, raw, original['query']['anchors'], method['identity_policy'])
            records.append(dict(variant=name, method=geometry, identity=identity, bundle=bundle))
    assert before == canonical_hash(dict(views=views, raw=raw, anchors=original['query']['anchors']))
    stem = task['parent'].replace('-bundle-v1-20260923', '') + '-' + task['case_id'] + '-' + task['camera_variant']
    destination = run / 'records' / (stem + '.json')
    write_json(destination, dict(task=task, pool_sha256=original['pool_sha256'], camera_decision=original['camera_decision'],
        intrinsics=original['intrinsics'], extrinsics=original['extrinsics'], query=original['query'],
        old_records=original['records'], records=records, audits=audits, source_sha256=protocol['source_sha256'],
        gt_read=False, elapsed_seconds=time.perf_counter() - started))
    print('REFIT', stem, [(r['variant'], r['method'], r['identity']['state']) for r in records], flush=True)
    return dict(**task, output_path=destination.relative_to(run).as_posix(), output_sha256=digest(destination))


def infer():
    run, protocol = checked()
    if (run / 'inference.json').exists():
        raise FileExistsError('Keep the frozen result')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, task) for task in protocol['tasks']]
        records = [f.result() for f in as_completed(futures)]
    assert len(records) == len(protocol['tasks'])
    write_json(run / 'inference.json', dict(state='complete', records=records, gt_read=False,
        protocol_sha256=digest(run / 'method_config.json'), source_sha256=protocol['source_sha256'],
        elapsed_seconds=time.perf_counter() - started))


def evaluate():
    run, protocol = checked()
    inference = read_json(run / 'inference.json')
    assert inference['protocol_sha256'] == digest(run / 'method_config.json')
    rows = []
    for receipt in inference['records']:
        path = run / receipt['output_path']
        assert digest(path) == receipt['output_sha256']
        result = read_json(path)
        parent, _, _, manifest = checked_run(receipt['parent'])
        config = read_json(parent / 'protocol.json')
        _, _, truth_root, frozen_scene, _ = checked_scene(config['run_id'])
        assert digest(truth_root / 'manifest.json') == frozen_scene['truth_sha256']
        truth = next(c for c in read_json(truth_root / 'manifest.json')['cases'] if c['case_id'] == receipt['case_id'])
        prior = read_json(ROOT / 'data/evaluation' / receipt['parent'] / 'summary.json')
        camera = next(r for r in prior['cameras'] if r['case_id'] == receipt['case_id'] and r['variant'] == receipt['camera_variant'])
        matrix = np.asarray(camera['camera']['prediction_world_to_gt_world'])
        target = truth['declared']['target']
        true_segments = target.get('segments', [target['endpoints']])
        old = [dict(variant='legacy', **r) for r in result['old_records']]
        for record in old + result['records']:
            identity = record['identity']
            native = np.asarray(identity['segments']).reshape(-1, 2, 3)
            aligned = native @ matrix[:3, :3].T + matrix[:3, 3]
            policy = config['evaluation']
            gap = truth['declared'].get('gap_segment')
            metrics = curve_metrics(aligned, true_segments, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m'])
            rows.append(dict(parent=receipt['parent'], case_id=receipt['case_id'], camera_variant=receipt['camera_variant'],
                variant=record['variant'], method=record['method'], state=identity['state'], reason=identity['reason'],
                camera_decision=result['camera_decision'], metrics=metrics, segment_count=len(aligned),
                gap=gap_coverage(aligned, gap, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m']) if gap else None,
                candidate_count=len(record.get('bundle', {}).get('proposals', [])) if record['variant'] != 'legacy' else None,
                identity_states=[a['state'] for a in identity['proposal_audit']]))
    summary = dict(state='complete', run_id=RUN_ID, protocol=protocol, rows=rows,
        inference_sha256=digest(run / 'inference.json'), elapsed_seconds=inference['elapsed_seconds'],
        gt_read_during_inference=False, no_automatic_promotion=True)
    destination = ROOT / 'data/evaluation' / RUN_ID
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', summary)
    write_json(SHARE, summary)
    print('EVALUATED', len(rows), 'rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer', 'evaluate'])
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[args.stage]()
