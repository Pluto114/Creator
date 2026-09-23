"""Check frozen candidate experiments and exact legacy-score preservation."""
from __future__ import annotations

import ast

from run_rod_assignment_refit import PARENTS, ROOT, SHARE, VARIANTS, checked
from run_rod_reference import canonical_hash, digest, read_json, write_json


def close_tree(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((close_tree(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, list):
        assert len(left) == len(right)
        return max((close_tree(a, b) for a, b in zip(left, right)), default=0.)
    if isinstance(left, float):
        delta = abs(left - right)
        assert delta <= 1e-12
        return delta
    assert left == right
    return 0.


def main():
    run, protocol = checked()
    inference = read_json(run / 'inference.json')
    summary = read_json(SHARE)
    assert inference['state'] == summary['state'] == 'complete'
    assert not inference['gt_read'] and not summary['gt_read_during_inference']
    assert summary['inference_sha256'] == digest(run / 'inference.json')
    assert summary['protocol'] == protocol
    task_keys = {(t['parent'], t['case_id'], t['camera_variant']) for t in protocol['tasks']}
    assert len(task_keys) == len(protocol['tasks']) == len(inference['records']) == 14
    normal_rows = {}
    for row in summary['rows']:
        key = (row['parent'], row['case_id'], row['camera_variant'], row['variant'], row['method'])
        assert key not in normal_rows
        normal_rows[key] = row
    assert len(normal_rows) == 14 * 4 * 2
    seen, frozen_outputs = set(), 0
    for receipt in inference['records']:
        key = (receipt['parent'], receipt['case_id'], receipt['camera_variant'])
        assert key in task_keys and key not in seen
        seen.add(key)
        parent = ROOT / '.runtime/experiments' / receipt['parent']
        assert digest(parent / receipt['path']) == receipt['sha256']
        assert digest(run / receipt['output_path']) == receipt['output_sha256']
        output = read_json(run / receipt['output_path'])
        original = read_json(parent / receipt['path'])
        assert output['task'] in protocol['tasks'] and not output['gt_read']
        assert output['source_sha256'] == protocol['source_sha256']
        assert output['old_records'] == original['records']
        assert output['query'] == original['query']
        assert output['intrinsics'] == original['intrinsics'] and output['extrinsics'] == original['extrinsics']
        assert {(r['variant'], r['method']) for r in output['records']} == {(v, m) for v in VARIANTS for m in ('baseline', 'cylinder_support')}
        frozen_outputs += 1
    legacy_count = 0
    for parent_id in PARENTS:
        previous = read_json(ROOT / 'data/evaluation' / parent_id / 'summary.json')
        for row in previous['rods']:
            if row['camera_variant'] not in ('initial', 'common_K_focal_trimmed'):
                continue
            new = normal_rows[(parent_id, row['case_id'], row['camera_variant'], 'legacy', row['method'])]
            assert canonical_hash(new['metrics']) == canonical_hash(row['metrics'])
            assert new['state'] == row['state'] and new['gap'] == row['gap']
            legacy_count += 1
    diagnostic = read_json(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-causes-r2.json')
    assert diagnostic['gt_used'] and not diagnostic['ordinary_inference']
    d_run = ROOT / '.runtime/experiments' / diagnostic['run_id']
    for name, sha in diagnostic['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(d_run / 'source_snapshot' / name)
    for r in diagnostic['records']:
        assert digest(ROOT / 'data/evaluation' / diagnostic['run_id'] / r['path']) == r['sha256']
    assert len(diagnostic['rows']) == 6 * 4 * 2
    oracle_count, max_oracle_delta = 0, 0.
    for parent_id in PARENTS:
        kind = 'height' if 'height' in parent_id else 'reference'
        old = read_json(ROOT / ('docs/experiments/results/2026-09-23-rod-' + kind + '-camera-counterfactual.json'))
        for row in old['rows']:
            if row['case_id'] == 'r04' or row['intervention'] != 'true_K_and_pose':
                continue
            new = next(r for r in diagnostic['rows'] if r['parent'] == parent_id and r['case_id'] == row['case_id']
                and r['method'] == row['method'] and r['variant'] == 'legacy')
            # Old oracle scores applied a camera-only Sim3 that is numerically identity.
            # Keep exact discrete decisions, allowing only roundoff in floating metrics.
            max_oracle_delta = max(max_oracle_delta, close_tree(new['metrics'], row['metrics']), close_tree(new['gap'], row['gap']))
            assert new['state'] == row['state']
            oracle_count += 1
    # The LF round was required by Git normalization, not a second algorithm trial.
    predecessor = read_json(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-refit.json')
    def key(r):
        return tuple(r[k] for k in ('parent', 'case_id', 'camera_variant', 'variant', 'method'))
    assert {key(r): r for r in predecessor['rows']} == normal_rows
    old_diagnostic = read_json(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-causes.json')
    def dkey(r):
        return tuple(r[k] for k in ('parent', 'case_id', 'variant', 'method'))
    assert {dkey(r): r for r in old_diagnostic['rows']} == {dkey(r): r for r in diagnostic['rows']}
    module = 'experiments/src/creator_eval/rod_assignment_refit.py'
    previous_source = ROOT / '.runtime/experiments/rod-assignment-refit-v1-20260923r2/source_snapshot' / module
    assert ast.dump(ast.parse(previous_source.read_text(encoding='utf-8'))) == ast.dump(ast.parse((ROOT / module).read_text(encoding='utf-8')))
    result = dict(state='complete', source_sha256=digest(__file__), ordinary_run_id=protocol['run_id'], lf_replay_rows_exact=160, method_ast_unchanged=True,
        summary_sha256=digest(SHARE), diagnostic_summary_sha256=digest(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-causes-r2.json'),
        verified_camera_tasks=frozen_outputs, normal_rows=len(normal_rows), legacy_rows_exact=legacy_count,
        diagnostic_rows=len(diagnostic['rows']), diagnostic_legacy_rows_close=oracle_count, diagnostic_alignment_abs_tolerance=1e-12, maximum_abs_metric_difference=max_oracle_delta)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-assignment-audit-r3.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
