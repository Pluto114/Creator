"""Audit new held-out RGB and train-only section sources without rewriting history.

Prepared source identity is separate from experiment completion. Old CRLF bytes,
the failed camera attempt, and the first section test fixture stay preserved.
The latter two are snapshot-only records, never silently re-labelled as r2/r1.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ('rod-heldout-view-oracle-v1-20260926', 'prepared.json'),
    ('rod-heldout-view-scenes-v1-20260926', 'prepared.json'),
    ('rod-heldout-view-validation-v1-20260926', 'method_config.json'),
    ('rod-heldout-view-audit-v1-20260926', 'prepared.json'),
    ('section-model-trainonly-v1-20260926r1', 'prepared.json'),
    ('rod-validation-evidence-v1-20260926', 'method_config.json'),
    ('rod-validation-evidence-audit-v1-20260926', 'prepared.json'),
    ('rod-target-projection-replay-v1-20260926', 'method_config.json'),
    ('section-model-diagnostic-v1-20260926', 'prepared.json'),
    ('rod-camera-envelope-v1-20260924', 'method_config.json'),
    ('camera-envelope-challenges-v1-20260924', 'method_config.json'),
    ('readout-abstention-development-v1-20260924', 'prepared.json'),
    ('g1-point-patch-abstention-v1-20260924', 'prepared.json'),
    ('camera-training-sensitivity-v1-20260924r2', 'method_config.json'),
    ('camera-training-audit-v1-20260924', 'prepared.json'),
    ('camera-training-audit-completion-v1-20260924', 'prepared.json'),
    ('readout-background-challenges-v1-20260924', 'prepared.json'),
    ('readout-valley-development-v1-20260924', 'prepared.json'),
    ('g1-point-patch-valley-v1-20260924', 'prepared.json'),
    ('readout-valley-closed-development-v1-20260924', 'prepared.json'),
    ('g1-point-patch-valley-closed-v1-20260924', 'prepared.json'),
)
FAILED = 'camera-training-sensitivity-v1-20260924'
EXPECTED_REVISED = {'experiments/src/creator_eval/camera_native_sensitivity.py',
                    'scripts/run_camera_training_sensitivity.py'}
PRIOR = ROOT / 'docs/experiments/results/2026-09-23-sprint-git-freeze-audit.json'
OUTPUT = ROOT / 'docs/experiments/results/2026-09-26-heldout-git-freeze-audit.json'
PREVIOUS_AUDIT = ROOT / 'docs/experiments/results/2026-09-26-evidence-git-freeze-audit.json'
SECTION_HISTORY = 'section-model-trainonly-v1-20260926'
SECTION_CURRENT = 'section-model-trainonly-v1-20260926r1'
SECTION_REVISED = {'scripts/run_section_model_trainonly.py',
    'scripts/check_section_model_trainonly.py', 'configs/section_model_trainonly_v1.json',
    'tests/test_section_model_trainonly.py'}
NEW_RUNS = {run_id for run_id, _ in RUNS[:5]}


# Keep these explicit: once committed, --others will no longer find them.
# Reporting/checking code is still part of the evidence chain, even if it was
# written after inference and therefore cannot be retroactively frozen in a run.
ADDITIONAL_SOURCES = {
    '.github/workflows/ci.yml',
    'experiments/src/creator_eval/heldout_view_oracle.py',
    'scripts/diagnose_rod_heldout_views.py',
    'tests/test_heldout_view_oracle.py',
    'configs/rod_heldout_views_annotations_v1.json',
    'configs/rod_heldout_views_v1.json',
    'configs/section_model_trainonly_v1.json',
    'experiments/src/creator_eval/heldout_view_pose.py',
    'experiments/src/creator_eval/section_model_trainonly.py',
    'scripts/audit_heldout_source_identity_20260926.py',
    'scripts/audit_rod_heldout_views.py',
    'scripts/check_section_model_trainonly.py',
    'scripts/prepare_rod_heldout_views.py',
    'scripts/report_rod_heldout_views.py',
    'scripts/run_rod_heldout_views.py',
    'scripts/run_section_model_trainonly.py',
    'tests/test_heldout_view_pose.py',
    'tests/test_rod_heldout_views.py',
    'tests/test_section_model_trainonly.py',
    'configs/camera_evidence_challenges_v1.json',
    'configs/section_model_diagnostic_v1.json',
    'experiments/src/creator_eval/camera_evidence_challenges.py',
    'experiments/src/creator_eval/rod_validation_evidence.py',
    'experiments/src/creator_eval/section_model_controls.py',
    'experiments/src/creator_eval/section_model_diagnostic.py',
    'scripts/audit_evidence_source_identity_20260926.py',
    'scripts/audit_rod_validation_evidence.py',
    'scripts/check_section_model_diagnostic.py',
    'scripts/report_rod_validation_evidence.py',
    'scripts/run_rod_target_projection_replay.py',
    'scripts/run_rod_validation_evidence.py',
    'scripts/run_section_model_diagnostic.py',
    'tests/test_camera_evidence_challenges.py',
    'tests/test_rod_validation_evidence.py',
    'tests/test_rod_validation_evidence_runner.py',
    'tests/test_section_model_diagnostic.py',
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes().decode('utf-8-sig'))


def git(*arguments, data=None):
    return subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *arguments],
                          cwd=ROOT, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def paths(output):
    return {part.decode('utf-8') for part in output.split(b'\0') if part}


def technical_source(name):
    return name.startswith(('scripts/', 'experiments/', 'reconstruction/', 'tests/', 'configs/', 'backends/')) and Path(name).suffix.lower() in {'.py', '.json', '.toml', '.yml', '.yaml', '.ps1'}


def historical_section(metadata_bytes):
    """Keep v1 snapshots intact; substantiate the narrow fixture/identity revision."""
    folder = ROOT / '.runtime/experiments' / SECTION_HISTORY
    current = ROOT / '.runtime/experiments' / SECTION_CURRENT
    path = folder / 'prepared.json'
    metadata_bytes[path] = path.read_bytes()
    old = read(path)
    new = read(current / 'prepared.json')
    if old['run_id'] != SECTION_HISTORY or new['engineering_revision']['previous_run_id'] != SECTION_HISTORY:
        raise ValueError('Historical section revision identity mismatch')
    freeze_path, protocol_path = folder / 'source_freeze.json', folder / 'protocol.json'
    for item, key in ((freeze_path, 'source_freeze_sha256'), (protocol_path, 'protocol_sha256')):
        metadata_bytes[item] = item.read_bytes()
        if sha256(metadata_bytes[item]) != old[key]:
            raise ValueError('Historical section metadata receipt changed: ' + key)
    freeze = read(freeze_path)
    if freeze['source_sha256'] != old['source_sha256'] or freeze['protocol_sha256'] != old['protocol_sha256']:
        raise ValueError('Historical section freeze does not match prepared metadata')
    sources, revised = [], set()
    for name, expected in sorted(old['source_sha256'].items()):
        saved_path = folder / 'source_snapshot' / name
        metadata_bytes[saved_path] = saved_path.read_bytes()
        if sha256(metadata_bytes[saved_path]) != expected:
            raise ValueError('Historical section snapshot changed: ' + name)
        live_sha = sha256((ROOT / name).read_bytes())
        if live_sha != expected:
            revised.add(name)
        sources.append(dict(path=name, frozen_sha256=expected, snapshot_sha256=expected,
                            live_sha256=live_sha, live_matches_historical_attempt=live_sha == expected))
    if len(sources) != 6 or revised != SECTION_REVISED:
        raise ValueError('Unexpected historical section live revisions: ' + repr(sorted(revised)))
    current_protocol_path = current / 'protocol.json'
    metadata_bytes[current_protocol_path] = current_protocol_path.read_bytes()
    if sha256(metadata_bytes[current_protocol_path]) != new['protocol_sha256']:
        raise ValueError('Current section protocol receipt changed')
    previous_protocol, current_protocol = read(protocol_path), read(current_protocol_path)
    old_content = {k: v for k, v in previous_protocol.items() if k != 'run_id'}
    new_content = {k: v for k, v in current_protocol.items() if k not in {'run_id', 'engineering_revision'}}
    if old_content != new_content or current_protocol['engineering_revision'] != new['engineering_revision']:
        raise ValueError('Section protocol changed beyond run identity and declared engineering revision')
    stable = ('experiments/src/creator_eval/section_model_trainonly.py',
              'experiments/src/creator_eval/section_model_diagnostic.py')
    if any(old['source_sha256'][name] != new['source_sha256'][name] for name in stable):
        raise ValueError('Section numerical method/helper changed during fixture revision')
    runner_name = 'scripts/run_section_model_trainonly.py'
    before = ast.parse((folder / 'source_snapshot' / runner_name).read_text(encoding='utf-8'))
    after = ast.parse((ROOT / runner_name).read_text(encoding='utf-8'))
    functions = ('infer', 'evaluate')
    for name in functions:
        old_fn = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new_fn = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
        if ast.dump(old_fn, include_attributes=False) != ast.dump(new_fn, include_attributes=False):
            raise ValueError('Section inference/evaluation computation changed: ' + name)
    return dict(run_id=SECTION_HISTORY, successor_run_id=SECTION_CURRENT,
        metadata_sha256=sha256(metadata_bytes[path]), source_reference_count=len(sources), sources=sources,
        live_revised_files=sorted(revised), engineering_revision=new['engineering_revision'],
        protocol_unchanged_except_run_id_and_revision=True, numerical_source_files_unchanged=list(stable),
        runner_functions_ast_unchanged=list(functions),
        policy='Snapshot-only historical attempt. Six frozen snapshots retained; four exact live differences allowed for the tracked fixture and r1 identity/archive checks. This audit does not recompute or certify section scores.')


def oracle_links(metadata, metadata_bytes):
    """Only bind bytes and declared scope; do not turn an oracle into normal evidence."""
    normal_id = 'rod-heldout-view-validation-v1-20260926'
    if metadata['normal_run_id'] != normal_id or metadata['new_truth_read_during_prepare'] is not False:
        raise ValueError('Oracle normal-run identity or declared preparation boundary changed')
    receipts = metadata['normal_receipts']
    if len(receipts) != 47:
        raise ValueError('Oracle frozen normal receipt set changed')
    for name, expected in receipts.items():
        path = ROOT / name
        metadata_bytes[path] = path.read_bytes()
        if sha256(metadata_bytes[path]) != expected:
            raise ValueError('Oracle normal receipt changed: ' + name)
    base = '.runtime/experiments/' + normal_id + '/'
    linked = {
        base + 'method_config.json': 'normal_config_sha256',
        base + 'inference.json': 'normal_inference_sha256',
        base + 'evaluation.json': 'normal_evaluation_sha256',
        '.runtime/experiments/rod-heldout-view-audit-v1-20260926/before-evaluation.json': 'independent_before_sha256',
        '.runtime/experiments/rod-heldout-view-scenes-v1-20260926/prepared.json': 'scene_prepared_sha256',
    }
    for name, field in linked.items():
        if receipts.get(name) != metadata[field]:
            raise ValueError('Oracle redundant receipt fields disagree: ' + field)
    normal = read(ROOT / base / 'method_config.json')
    if any(metadata['source_sha256'].get(name) != sha for name, sha in normal['source_sha256'].items()):
        raise ValueError('Oracle must retain the full unchanged normal source closure')
    folder = ROOT / '.runtime/experiments' / metadata['run_id']
    freeze = read(folder / 'source_freeze.json')
    if freeze['policy'] != metadata['policy'] or freeze['before_new_truth_read'] is not True:
        raise ValueError('Oracle source freeze policy/boundary mismatch')
    scene = read(ROOT / '.runtime/experiments/rod-heldout-view-scenes-v1-20260926/prepared.json')
    if scene['truth_sha256'] != metadata['expected_truth_sha256']:
        raise ValueError('Oracle declared future truth receipt differs from scene preparation')
    return dict(normal_run_id=normal_id, frozen_normal_receipt_count=len(receipts),
        normal_receipts=receipts, declared_policy=metadata['policy'],
        scope='Post-hoc privileged evaluation. Byte links and declared preparation boundary only; this audit does not independently prove timing or numerical oracle outcomes.')


def audit(output):
    if output.exists():
        raise FileExistsError('Preserve prior source audit; choose another --output')
    head = git('rev-parse', 'HEAD').decode().strip()
    head_paths = paths(git('ls-tree', '-r', '--name-only', '-z', head))
    legacy = read(PRIOR)['legacy_newline_mappings']
    if len(legacy) != 8:
        raise ValueError('The historical newline allowlist changed')
    metadata_bytes, frozen_sources, runs, reference_count = {}, {}, [], 0
    metadata_bytes[PREVIOUS_AUDIT] = PREVIOUS_AUDIT.read_bytes()
    previous_runs = {r['run_id']: r for r in read(PREVIOUS_AUDIT)['runs']}
    if set(previous_runs) != {run_id for run_id, _ in RUNS[5:]} or len(RUNS) != 21:
        raise ValueError('Keep precisely the previous sixteen runs and five new active freezes')
    for run_id, metadata_name in RUNS:
        folder = ROOT / '.runtime/experiments' / run_id
        metadata_path = folder / metadata_name
        metadata_bytes[metadata_path] = metadata_path.read_bytes()
        metadata = read(metadata_path)
        if (run_id.endswith('20260926') or run_id in NEW_RUNS) and metadata.get('run_id', metadata.get('audit_run_id')) != run_id:
            raise ValueError('Metadata run identity mismatch: ' + run_id)
        if run_id in previous_runs and sha256(metadata_bytes[metadata_path]) != previous_runs[run_id]['metadata_sha256']:
            raise ValueError('Previous active freeze metadata changed: ' + run_id)
        if (run_id.endswith('20260926') or run_id in NEW_RUNS) and 'source_freeze_sha256' in metadata:
            freeze_path = folder / 'source_freeze.json'
            metadata_bytes[freeze_path] = freeze_path.read_bytes()
            if sha256(metadata_bytes[freeze_path]) != metadata['source_freeze_sha256']:
                raise ValueError('Source freeze receipt mismatch: ' + run_id)
            if read(freeze_path)['source_sha256'] != metadata['source_sha256']:
                raise ValueError('Source freeze/metadata source sets differ: ' + run_id)
        if run_id in {'rod-validation-evidence-audit-v1-20260926', 'rod-heldout-view-audit-v1-20260926'}:
            target_path = ROOT / '.runtime/experiments' / metadata['target_run_id'] / 'method_config.json'
            metadata_bytes[target_path] = target_path.read_bytes()
            if sha256(metadata_bytes[target_path]) != metadata['target_config_sha256']:
                raise ValueError('Independent audit target config changed')
            if run_id == 'rod-heldout-view-audit-v1-20260926':
                if metadata['target_run_id'] != 'rod-heldout-view-validation-v1-20260926':
                    raise ValueError('Independent held-out audit points to the wrong normal run')
                target = read(target_path)
                if any(metadata['source_sha256'].get(name) != sha for name, sha in target['source_sha256'].items()):
                    raise ValueError('Independent held-out audit source closure changed')
        supplementary_links = oracle_links(metadata, metadata_bytes) if run_id == 'rod-heldout-view-oracle-v1-20260926' else None
        per_run = []
        for name, expected in sorted(metadata['source_sha256'].items()):
            raw = (ROOT / name).read_bytes()
            snapshot_path = folder / 'source_snapshot' / name
            snapshot = snapshot_path.read_bytes()
            metadata_bytes[snapshot_path] = snapshot
            if sha256(raw) != expected or sha256(snapshot) != expected:
                raise ValueError(f'Active run raw source/snapshot mismatch: {run_id}: {name}')
            if name in frozen_sources and frozen_sources[name] != expected:
                raise ValueError('Active runs disagree on source identity: ' + name)
            frozen_sources[name] = expected
            normalized_sha = sha256(raw.replace(b'\r\n', b'\n'))
            if 'source_lf_sha256' in metadata and metadata['source_lf_sha256'].get(name) != normalized_sha:
                raise ValueError('Declared LF mapping changed: ' + name)
            per_run.append(dict(path=name, frozen_sha256=expected, live_raw_sha256=sha256(raw),
                                snapshot_raw_sha256=sha256(snapshot), lf_sha256=normalized_sha))
        reference_count += len(per_run)
        runs.append(dict(run_id=run_id, metadata_file=metadata_name, metadata_sha256=sha256(metadata_bytes[metadata_path]),
                         source_reference_count=len(per_run), sources=per_run, supplementary_links=supplementary_links,
                         scope='Prepared source identity only; does not certify inference/evaluation completion'))
    new_paths = paths(git('ls-files', '--others', '--exclude-standard', '-z'))
    new_paths |= paths(git('diff', '--cached', '--name-only', '--diff-filter=A', '-z'))
    additional_new_sources = {name for name in new_paths if technical_source(name)}
    names = set(frozen_sources) | additional_new_sources | ADDITIONAL_SOURCES
    files, mappings, captured = [], {}, {}
    for name in sorted(names):
        raw = (ROOT / name).read_bytes()
        captured[name] = sha256(raw)
        if name in ADDITIONAL_SOURCES and (b'\r' in raw or not raw.endswith(b'\n') or raw.endswith(b'\n\n')):
            raise ValueError('New source must use LF and exactly one final newline: ' + name)
        raw_oid = git('hash-object', '--no-filters', '--stdin', data=raw).decode().strip()
        clean_oid = git('hash-object', '--path=' + name, '--stdin', data=raw).decode().strip()
        in_head = name in head_paths
        old_blob = git('cat-file', 'blob', head + ':' + name) if in_head else None
        new_file = not in_head
        if raw_oid != clean_oid:
            if new_file or name not in legacy:
                raise ValueError('Unexpected Git source normalization: ' + name)
            normalized = raw.replace(b'\r\n', b'\n')
            prior = legacy[name]
            normalized_oid = git('hash-object', '--no-filters', '--stdin', data=normalized).decode().strip()
            if b'\r\n' not in raw or b'\r' in normalized or normalized != old_blob or normalized_oid != clean_oid:
                raise ValueError('Legacy difference is not exactly CRLF-to-LF at HEAD: ' + name)
            if sha256(raw) != prior['raw_frozen_sha256'] or sha256(normalized) != prior['git_blob_sha256']:
                raise ValueError('Legacy raw/normalized hashes differ from Sept 23 evidence: ' + name)
            mappings[name] = dict(raw_frozen_sha256=sha256(raw), git_blob_sha256=sha256(normalized),
                head_blob_sha256=sha256(old_blob), raw_git_object=raw_oid, filtered_git_object=clean_oid,
                crlf_count=raw.count(b'\r\n'), relation='Only preserved historical CRLF-to-LF; normalized bytes exactly equal Git HEAD blob')
        if new_file and raw_oid != clean_oid:
            raise ValueError('New file would change bytes in Git: ' + name)
        files.append(dict(path=name, new_at_audited_head=new_file, frozen_in_active_run=name in frozen_sources,
                          raw_sha256=sha256(raw), raw_git_object=raw_oid, filtered_git_object=clean_oid,
                          git_filters_preserve_bytes=raw_oid == clean_oid,
                          head_blob_sha256=sha256(old_blob) if old_blob is not None else None,
                          same_as_head_after_filter=clean_oid == git('rev-parse', head + ':' + name).decode().strip() if in_head else None))
    if set(mappings) != set(legacy):
        raise ValueError('Expected exactly the same eight inherited newline mappings')
    failed_folder = ROOT / '.runtime/experiments' / FAILED
    failed_meta_path = failed_folder / 'method_config.json'
    metadata_bytes[failed_meta_path] = failed_meta_path.read_bytes()
    failed_meta = read(failed_meta_path)
    failed_sources, revised = [], set()
    for name, expected in sorted(failed_meta['source_sha256'].items()):
        saved_path = failed_folder / 'source_snapshot' / name
        saved = saved_path.read_bytes()
        metadata_bytes[saved_path] = saved
        if sha256(saved) != expected:
            raise ValueError('Failed attempt snapshot changed: ' + name)
        live = sha256((ROOT / name).read_bytes())
        if live != expected:
            revised.add(name)
        failed_sources.append(dict(path=name, frozen_sha256=expected, snapshot_sha256=sha256(saved),
                                   live_sha256=live, live_matches_failed_attempt=live == expected))
    if len(failed_sources) != 59:
        raise ValueError('The preserved failed-attempt source count changed')
    if revised != EXPECTED_REVISED:
        raise ValueError('Unexpected live revisions relative to failed camera attempt: ' + repr(sorted(revised)))
    section_history = historical_section(metadata_bytes)
    # 历史实验保留当时的字节；不能把修过的测试夹具倒灌回旧快照。
    for path, raw in metadata_bytes.items():
        if path.read_bytes() != raw:
            raise ValueError('Freeze metadata changed during audit: ' + str(path))
    for name, sha in captured.items():
        if sha256((ROOT / name).read_bytes()) != sha:
            raise ValueError('Live source changed during audit: ' + name)
    if git('rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('Git HEAD changed during audit')
    result = dict(state='complete', audited_git_head=head, active_frozen_run_count=len(runs),
        historical_frozen_run_count=16, added_frozen_run_count=5,
        explicit_continuation_source_count=len(ADDITIONAL_SOURCES),
        active_source_references=reference_count, unique_active_source_files=len(frozen_sources),
        git_checked_source_files=len(files), new_source_files_byte_exact=sum(f['new_at_audited_head'] for f in files),
        legacy_newline_mapping_count=len(mappings), legacy_newline_mappings=mappings, runs=runs, files=files,
        failed_attempt=dict(run_id=FAILED, metadata_sha256=sha256(metadata_bytes[failed_meta_path]),
            source_reference_count=len(failed_sources), sources=failed_sources, live_revised_files=sorted(revised),
            policy='All failed-attempt snapshots checked against frozen raw hashes; live helper/runner intentionally differ for r2; no failed output rewritten'),
        historical_section_attempt=section_history,
        audit_script_sha256=sha256(Path(__file__).read_bytes()), prior_newline_evidence_sha256=sha256(PRIOR.read_bytes()),
        prior_continuation_audit_sha256=sha256(metadata_bytes[PREVIOUS_AUDIT]),
        reproduction='Load scripts/Enter-CreatorEnvironment.ps1, then backends/da3/.venv/Scripts/python.exe -B scripts/audit_heldout_source_identity_20260926.py --output <new-output.json>',
        git_operations='Read-only hash-object --no-filters --stdin versus hash-object --path=<repo path> --stdin; legacy bytes also checked with cat-file blob <audited HEAD>:<path>',
        scope='Raw live/snapshot/freeze equality for twenty-one active runs plus Git byte policy, explicit held-out reporting/checking sources, and preserved historical camera/section attempts. Run completion, scores, and documentation links are outside this source-only audit.')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print('HELDOUT_SOURCE_IDENTITY', len(runs), 'runs;', reference_count, 'active source references;',
          len(files), 'Git source files;', result['new_source_files_byte_exact'], 'new byte-exact;',
          len(mappings), 'legacy LF mappings;', len(failed_sources), 'preserved failed camera snapshots;',
          section_history['source_reference_count'], 'historical section snapshots', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit((ROOT / args.output).resolve())
