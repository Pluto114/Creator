"""Evaluation-only replay explaining rejected cylinders and missing anchor rows."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from prepare_rod_reference import checked_scene, source_snapshot
from run_camera_bundle_pilot import checked_run
from run_rod_assignment_refit import PARENTS, VARIANTS
from run_rod_foreground_identity import compact_bundle, make_views
from run_rod_reference import ROOT, digest, read_json, write_json

# isort: split
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.rod_assignment_refit import associate_refitted_variants
from creator_eval.rod_candidate_association import associate_multiview_lines_cached
from creator_eval.rod_cylinder_support import select_supported_cylinders
from creator_eval.rod_foreground_identity import build_anchor_proposals, select_foreground_identity

RUN_ID = 'rod-assignment-causes-v1-20260923r2'


def task(parent_id, cid):
    parent, _, frozen, manifest = checked_run(parent_id)
    config = read_json(parent / 'protocol.json')
    _, _, truth_dir, scene, _ = checked_scene(config['run_id'])
    assert digest(truth_dir / 'manifest.json') == scene['truth_sha256']
    gt = next(c for c in read_json(truth_dir / 'manifest.json')['cases'] if c['case_id'] == cid)
    old_path = parent / 'rod-results' / (cid + '-common_K_focal_trimmed.json')
    old = read_json(old_path)
    pool_path = parent / 'rod-pools' / (cid + '.json')
    assert digest(pool_path) == old['pool_sha256']
    frames = [{**f, 'K_index': c['K_index'], 'world_to_camera_cv': c['world_to_camera_cv']}
        for f, c in zip(read_json(pool_path)['frames'], gt['cameras'])]
    method = manifest['rod_method']
    views, raw = make_views(frames, method['candidate_cap']), [f['observations'] for f in frames]
    variants = {'legacy': associate_multiview_lines_cached(views, method['association_common']),
                **associate_refitted_variants(views, method['association_common'])}
    rows, audits = [], {}
    true_target = gt['declared']['target']
    target = true_target.get('segments', [true_target['endpoints']])
    policy = config['evaluation']
    for name in ['legacy', *VARIANTS]:
        association = variants[name]
        cylinder = select_supported_cylinders(association, views, raw, method['extent'], method['cylinder_screen'])
        audits[name] = dict(association=association, cylinder_support=cylinder['cylinder_support'],
            cylinder_screen=cylinder['cylinder_screen'])
        for geometry, proposal in [('baseline', association), ('cylinder_support', cylinder)]:
            bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method['extent'],
                method['cylinder_screen'] if geometry == 'cylinder_support' else None))
            result = select_foreground_identity(bundle, views, raw, old['query']['anchors'], method['identity_policy'])
            metrics = curve_metrics(result['segments'], target, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m'])
            gap = gt['declared'].get('gap_segment')
            rows.append(dict(parent=parent_id, case_id=cid, variant=name, method=geometry, state=result['state'],
                reason=result['reason'], metrics=metrics, proposal_audit=result['proposal_audit'],
                gap=gap_coverage(result['segments'], gap, tolerance=policy['curve_tolerance_m'], spacing=policy['curve_spacing_m']) if gap else None,
                candidate_count=len(bundle['proposals']), segments=result['segments']))
    output = ROOT / 'data/evaluation' / RUN_ID / (parent_id.replace('-bundle-v1-20260923', '') + '-' + cid + '.json')
    write_json(output, dict(gt_used=True, ordinary_inference=False, parent=parent_id, case_id=cid,
        old_record_sha256=digest(old_path), truth_manifest_sha256=scene['truth_sha256'], pool_sha256=old['pool_sha256'],
        rows=rows, audits=audits, source_input_sha256=frozen['input_sha256']))
    print('CAUSE', parent_id, cid, [(r['variant'], r['method'], r['state']) for r in rows], flush=True)
    return dict(path=output.name, sha256=digest(output))


def run():
    run_dir, output = ROOT / '.runtime/experiments' / RUN_ID, ROOT / 'data/evaluation' / RUN_ID
    run_dir.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    names = {'experiments/src/creator_eval/rod_assignment_refit.py', 'scripts/run_rod_assignment_refit.py', 'scripts/audit_rod_assignment_causes.py'}
    for parent in PARENTS:
        _, _, frozen, _ = checked_run(parent)
        names.update(frozen['source_sha256'])
    sources = source_snapshot(run_dir, names)
    tasks = [(parent, cid) for parent in PARENTS for cid in ('r01', 'r02', 'r03')]
    protocol = dict(run_id=RUN_ID, tasks=tasks, source_sha256=sources,
        scope='evaluation-only all-true-camera cause audit; frozen known data and methods; never normal inputs or patches')
    write_json(run_dir / 'protocol.json', protocol)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(task, *t) for t in tasks]
        records = [f.result() for f in as_completed(futures)]
    rows = []
    for record in records:
        assert digest(output / record['path']) == record['sha256']
        rows.extend(read_json(output / record['path'])['rows'])
    for name, sha in sources.items():
        assert digest(ROOT / name) == sha == digest(run_dir / 'source_snapshot' / name)
    summary = dict(state='complete', **protocol, records=records, rows=rows, gt_used=True,
        ordinary_inference=False, elapsed_seconds=time.perf_counter() - started)
    write_json(output / 'summary.json', summary)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-causes-r2.json', summary)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
