"""Audit all prior Sept 24 sources plus the finite camera envelope continuation.

This checks prepared sources, not whether ongoing experiments have completed.
Historical CRLF bytes remain in their snapshots; only the previously documented
8 files may normalize to LF. The failed first camera attempt is snapshot-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
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
OUTPUT = ROOT / 'docs/experiments/results/2026-09-24-envelope-git-freeze-audit.json'


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


def audit(output):
    if output.exists():
        raise FileExistsError('Preserve prior source audit; choose another --output')
    head = git('rev-parse', 'HEAD').decode().strip()
    head_paths = paths(git('ls-tree', '-r', '--name-only', '-z', head))
    legacy = read(PRIOR)['legacy_newline_mappings']
    if len(legacy) != 8:
        raise ValueError('The historical newline allowlist changed')
    metadata_bytes, frozen_sources, runs, reference_count = {}, {}, [], 0
    for run_id, metadata_name in RUNS:
        folder = ROOT / '.runtime/experiments' / run_id
        metadata_path = folder / metadata_name
        metadata_bytes[metadata_path] = metadata_path.read_bytes()
        metadata = read(metadata_path)
        per_run = []
        for name, expected in sorted(metadata['source_sha256'].items()):
            raw = (ROOT / name).read_bytes()
            snapshot = (folder / 'source_snapshot' / name).read_bytes()
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
                         source_reference_count=len(per_run), sources=per_run,
                         scope='Prepared source identity only; does not certify inference/evaluation completion'))
    new_paths = paths(git('ls-files', '--others', '--exclude-standard', '-z'))
    new_paths |= paths(git('diff', '--cached', '--name-only', '--diff-filter=A', '-z'))
    additional_new_sources = {name for name in new_paths if technical_source(name)}
    names = set(frozen_sources) | additional_new_sources | {Path(__file__).relative_to(ROOT).as_posix()}
    files, mappings, captured = [], {}, {}
    for name in sorted(names):
        raw = (ROOT / name).read_bytes()
        captured[name] = sha256(raw)
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
        saved = (failed_folder / 'source_snapshot' / name).read_bytes()
        if sha256(saved) != expected:
            raise ValueError('Failed attempt snapshot changed: ' + name)
        live = sha256((ROOT / name).read_bytes())
        if live != expected:
            revised.add(name)
        failed_sources.append(dict(path=name, frozen_sha256=expected, snapshot_sha256=sha256(saved),
                                   live_sha256=live, live_matches_failed_attempt=live == expected))
    if revised != EXPECTED_REVISED:
        raise ValueError('Unexpected live revisions relative to failed camera attempt: ' + repr(sorted(revised)))
    # 失败实验保留当时的字节；不能要求工作区倒回去，假装r2没有修过问题。
    for path, raw in metadata_bytes.items():
        if path.read_bytes() != raw:
            raise ValueError('Freeze metadata changed during audit: ' + str(path))
    for name, sha in captured.items():
        if sha256((ROOT / name).read_bytes()) != sha:
            raise ValueError('Live source changed during audit: ' + name)
    if git('rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('Git HEAD changed during audit')
    result = dict(state='complete', audited_git_head=head, active_frozen_run_count=len(runs),
        active_source_references=reference_count, unique_active_source_files=len(frozen_sources),
        git_checked_source_files=len(files), new_source_files_byte_exact=sum(f['new_at_audited_head'] for f in files),
        legacy_newline_mapping_count=len(mappings), legacy_newline_mappings=mappings, runs=runs, files=files,
        failed_attempt=dict(run_id=FAILED, metadata_sha256=sha256(metadata_bytes[failed_meta_path]),
            source_reference_count=len(failed_sources), sources=failed_sources, live_revised_files=sorted(revised),
            policy='All failed-attempt snapshots checked against frozen raw hashes; live helper/runner intentionally differ for r2; no failed output rewritten'),
        audit_script_sha256=sha256(Path(__file__).read_bytes()), prior_newline_evidence_sha256=sha256(PRIOR.read_bytes()),
        reproduction='Load scripts/Enter-CreatorEnvironment.ps1, then backends/da3/.venv/Scripts/python.exe -B scripts/audit_camera_envelope_source_identity_20260924.py --output <new-output.json>',
        git_operations='Read-only hash-object --no-filters --stdin versus hash-object --path=<repo path> --stdin; legacy bytes also checked with cat-file blob <audited HEAD>:<path>',
        scope='Raw live/snapshot/freeze equality for twelve active runs plus Git byte policy. Ongoing run completion, scores, and documentation links are outside this source-only audit.')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print('SPRINT_SOURCE_IDENTITY', len(runs), 'runs;', reference_count, 'active source references;',
          len(files), 'Git source files;', result['new_source_files_byte_exact'], 'new byte-exact;',
          len(mappings), 'legacy LF mappings;', len(failed_sources), 'preserved failed snapshots', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit((ROOT / args.output).resolve())
