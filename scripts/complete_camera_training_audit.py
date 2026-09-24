"""Complete the preserved pre-audit with JSON-normalized comparison.

The first post-audit compared tuples rebuilt in memory with lists loaded from
JSON. Its failure and frozen source stay intact; this version records that fix
as a separate source freeze and reuses the real, earlier pre-evaluation evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from audit_camera_training_sensitivity import (
    DEFAULT_AUDIT,
    DEFAULT_EVALUATION,
    DEFAULT_OUTPUT,
    DEFAULT_RUN,
    METHODS,
    ROOT,
    _sources,
    audit_inference,
    canonical_hash,
    digest,
    read_json,
    source_snapshot,
    write_json,
)


def require_same_json_audit(before, after):
    """Match serialized meaning; list/tuple representation is not new evidence."""
    if canonical_hash(before) != canonical_hash(after):
        raise ValueError('Inference, plans, assignments, or parent artifacts changed across evaluation')


def complete(args):
    original = ROOT / '.runtime/experiments' / args.audit_run_id
    completion = ROOT / '.runtime/experiments' / args.completion_run_id
    if completion.exists() or args.output.exists():
        raise FileExistsError('Preserve existing completion attempts and outputs')
    frozen = read_json(original / 'prepared.json')
    if frozen['run_id'] != args.run_id or frozen['audit_run_id'] != args.audit_run_id or frozen['evaluation_output'] != args.evaluation_output.relative_to(ROOT).as_posix():
        raise ValueError('Audit target changed')
    _sources(original, frozen['source_sha256'])
    before_path = original / 'before-evaluation.json'
    before = read_json(before_path)
    completion.mkdir(parents=True)
    names = set(frozen['source_sha256']) | {'scripts/complete_camera_training_audit.py'}
    sources = source_snapshot(completion, names)
    failure = ROOT / '.runtime/camera-training-audit-post-20260924.log'
    receipt = dict(run_id=args.run_id, audit_run_id=args.audit_run_id, completion_run_id=args.completion_run_id,
                   source_sha256=sources, original_prepared_sha256=digest(original / 'prepared.json'),
                   before_evaluation_sha256=digest(before_path), original_post_failure_sha256=digest(failure),
                   correction='Compare canonical JSON meaning so tuple source-row triples equal the lists saved by JSON. No original evidence rewritten.')
    write_json(completion / 'prepared.json', receipt)
    after = audit_inference(args.run_id)
    require_same_json_audit(before, after)
    # 评价前留下的哈希才是证据。修完审计不能倒回去伪造一份“之前”的记录。
    for name, sha in before['receipts'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Pre-evaluation artifact changed: ' + name)
    result_path = ROOT / 'data/evaluation' / args.run_id / 'summary.json'
    summary = read_json(result_path)
    if digest(result_path) != digest(args.evaluation_output) or summary['state'] != 'complete' or summary['inference_sha256'] != before['inference_sha256'] or summary['gt_read_during_inference']:
        raise ValueError('Published evaluation detached from unchanged inference')
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
        completion_run_id=args.completion_run_id, completion_prepared_sha256=digest(completion / 'prepared.json'),
        completion_source_file_count=len(sources), original_post_failure_preserved=True,
        audit_prepared_sha256=digest(original / 'prepared.json'), audit_source_file_count=len(frozen['source_sha256']),
        before_evaluation_sha256=digest(before_path), evaluation_sha256=digest(result_path),
        inference_sha256=before['inference_sha256'], inference_and_parent_files_unchanged_across_evaluation=True,
        unchanged_receipted_file_count=len(before['receipts']), canonical_before_after_equal=True,
        correction=receipt['correction'], condition_count=len(expected_cameras), rod_method_count=len(expected_rods),
        chains=before['chains'], conditions=before['conditions'], full_replays=before['full_replays'],
        identities=identities, comparisons=before['comparisons'],
        scope='Identity means retained image candidate and source rows, not physical rod identity or accuracy. Rejected/empty support is not perfect agreement.')
    _sources(original, frozen['source_sha256'])
    _sources(completion, sources)
    if digest(before_path) != receipt['before_evaluation_sha256'] or digest(original / 'prepared.json') != receipt['original_prepared_sha256']:
        raise ValueError('Original pre-evaluation evidence changed during completion')
    write_json(completion / 'summary.json', result)
    write_json(args.output, result)
    print('COMPLETED_TRAINING_AUDIT', len(expected_cameras), len(expected_rods), len(before['receipts']), 'unchanged files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--audit-run-id', default=DEFAULT_AUDIT)
    parser.add_argument('--completion-run-id', default='camera-training-audit-completion-v1-20260924')
    parser.add_argument('--evaluation-output', type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.evaluation_output = (ROOT / args.evaluation_output).resolve()
    args.output = (ROOT / args.output).resolve()
    for value in (args.run_id, args.audit_run_id, args.completion_run_id):
        if Path(value).name != value or ':' in value or value in ('.', '..'):
            raise ValueError('Single-component run ids required')
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    complete(args)
