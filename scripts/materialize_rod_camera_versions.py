"""Materialize explicit initial/corrected camera versions and reversible research patches."""
from __future__ import annotations

import sys

import numpy as np
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import checked_run
from run_rod_identity_stress import reject_truth_open
from run_rod_reference import ROOT, canonical_hash, digest, read_json, write_json

sys.path.insert(0, str(ROOT / 'reconstruction/src'))
from creator_recon.domain.camera_point_snapshot import (  # noqa: E402
    native_intrinsics,
    write_camera_snapshot,
)
from creator_recon.domain.point_patch import (  # noqa: E402
    compose,
    content_hash,
    open_candidate_view,
    write_candidate_view,
    write_patch,
)

RUN_ID = 'rod-height-camera-versions-v1-20260923'
PARENT = 'rod-height-bundle-v1-20260923'


def run():
    parent, _, frozen, manifest = checked_run(PARENT)
    source_index = read_json(parent / 'rod-inference.json')
    if source_index['state'] != 'complete' or source_index['source_sha256'] != frozen['source_sha256'] or source_index['input_sha256'] != frozen['input_sha256']:
        raise ValueError('Detached rod inference')
    assert source_index['camera_inference_sha256'] == digest(parent / 'inference.json')
    predictions = read_json(parent / 'predictions.json')
    assert predictions['state'] == 'complete' and predictions['input_sha256'] == frozen['input_sha256']
    destination = ROOT / '.runtime/experiments' / RUN_ID
    destination.mkdir(exist_ok=False)
    sources = source_snapshot(destination, [*frozen['source_sha256'], 'scripts/materialize_rod_camera_versions.py',
        'reconstruction/src/creator_recon/domain/camera_point_snapshot.py', 'reconstruction/src/creator_recon/domain/point_patch.py'])
    plan = [r for r in source_index['records'] if r['camera_variant'] in ('initial', 'common_K_focal_trimmed')]
    if len(plan) != 2 * len(manifest['cases']):
        raise ValueError('Expected paired initial/two-stage cameras for every case')
    policy = dict(run_id=RUN_ID, parent=PARENT, camera_variants=['initial', 'common_K_focal_trimmed'],
        depth_policy='unchanged DA3 samples; new camera reprojection is an integration version, not depth repair',
        source_sha256=sources, parent_rod_sha256=digest(parent / 'rod-inference.json'), parent_predictions_sha256=digest(parent / 'predictions.json'),
        skip_withheld_camera=True, tasks=plan)
    write_json(destination / 'protocol.json', policy)
    sys.addaudithook(reject_truth_open)
    rows = []
    for entry in plan:
        cid, name = entry['case_id'], entry['camera_variant']
        assert digest(parent / entry['path']) == entry['sha256']
        geometry = read_json(parent / entry['path'])
        if name != 'initial' and geometry['camera_decision']['state'] != 'candidate_camera_correction':
            rows.append(dict(case_id=cid, camera_variant=name, state='withheld_camera_not_materialized'))
            continue
        prediction = next(r for r in predictions['records'] if r['case_id'] == cid)
        assert digest(ROOT / prediction['path']) == prediction['prediction_sha256']
        assert digest(ROOT / prediction['report']) == prediction['report_sha256']
        case = next(c for c in manifest['cases'] if c['case_id'] == cid)
        assert [f['rgb_sha256'] for f in case['frames']] == [f['sha256'] for f in read_json(ROOT / prediction['report'])['images']]
        with np.load(ROOT / prediction['path'], allow_pickle=False) as native:
            depth = native['depth'].copy()
            original_k = native['intrinsics'].copy()
        height, width = depth.shape[1:]
        k = native_intrinsics(geometry['intrinsics'], case['frames'][0]['size_wh'], [width, height])
        e = np.asarray(geometry['extrinsics'])
        if name == 'initial':
            np.testing.assert_allclose(k, original_k, atol=1e-10)
        frames = [dict(frame_id=f['view_id'], image_sha256=f['rgb_sha256'], prediction_size_wh=[width, height]) for f in case['frames']]
        folder = destination / (cid + '-' + name)
        receipt = write_camera_snapshot(folder, depth, k, e, frames=frames, prediction_sha256=prediction['prediction_sha256'], camera_source_sha256=entry['sha256'])
        base = compose(folder / 'base')
        source = read_json(folder / 'camera-depth-source.json')
        assert content_hash(source['source']) == source['geometry_source_id'] == receipt['geometry_source_id']
        checks, max_pixel, max_z = 0, 0., 0.
        for view in range(len(frames)):
            present = np.flatnonzero(base['point_ids'][:, 0] == view)
            selected = present[np.linspace(0, len(present) - 1, min(1024, len(present)), dtype=int)]
            xyz = base['points'][selected] @ e[view, :, :3].T + e[view, :, 3]
            uv = xyz @ k[view].T
            ids = base['point_ids'][selected]
            max_pixel = max(max_pixel, float(abs(uv[:, :2] / uv[:, 2, None] - ids[:, [2, 1]]).max()))
            max_z = max(max_z, float(abs(xyz[:, 2] - depth[view, ids[:, 1], ids[:, 2]]).max()))
            checks += len(selected)
        assert max_pixel < 1e-6 and max_z < 1e-6
        patch_ids = []
        for result in geometry['records']:
            segments = np.asarray(result['identity']['segments'], float).reshape(-1, 2, 3)
            patch_folder = folder / ('patch-' + result['method'])
            evidence = [dict(decision='accept', line_ids=[f'line-{i:04d}' for i in range(len(segments))], suppression_range=None,
                view_ids=[a['view_id'] for a in geometry['query']['anchors']], source_sha256=entry['sha256'],
                note='Research-only frozen curve output bound to this camera/depth version; no physical correctness certificate')] if len(segments) else []
            patch_id = write_patch(patch_folder, folder / 'base', segments, np.empty((0, 3), np.uint32),
                selection_sha256=canonical_hash(geometry['query']), method=dict(id=result['method'], version='camera-version-1',
                config_sha256=canonical_hash(manifest['rod_method']), seed=0), evidence=evidence,
                unresolved=[] if len(segments) else [result['identity']['reason']])
            for enabled in (True, False):
                path = folder / (result['method'] + ('-enabled.json' if enabled else '-withdrawn.json'))
                write_candidate_view(path, folder / 'base', patch_folder, enabled=enabled)
                opened = open_candidate_view(path)
                assert opened['points'].tobytes() == base['points'].tobytes()
                assert opened['point_ids'].tobytes() == base['point_ids'].tobytes()
                np.testing.assert_array_equal(opened['segments'], segments if enabled else np.empty((0, 2, 3)))
            patch_ids.append(dict(method=result['method'], patch_id=patch_id, segment_count=len(segments)))
        row = dict(case_id=cid, camera_variant=name, state='materialized_research_version', **receipt,
            pixel_checks=checks, maximum_pixel_roundtrip_error=max_pixel, maximum_depth_roundtrip_error=max_z,
            source_identity_verified=True, rollback_bytes_equal=True, saved_views_reopened=True, patches=patch_ids)
        write_json(folder / 'receipt.json', row)
        rows.append(row)
        print('CAMERA_VERSION', cid, name, receipt['point_count'], flush=True)
    for cid in [c['case_id'] for c in manifest['cases']]:
        pair = [r for r in rows if r['case_id'] == cid and 'snapshot_id' in r]
        if len(pair) == 2:
            assert pair[0]['snapshot_id'] != pair[1]['snapshot_id'] and pair[0]['geometry_source_id'] != pair[1]['geometry_source_id']
            assert pair[0]['depth_content_sha256'] == pair[1]['depth_content_sha256']
    for name, sha in sources.items():
        assert digest(ROOT / name) == sha == digest(destination / 'source_snapshot' / name)
    assert digest(parent / 'rod-inference.json') == policy['parent_rod_sha256']
    assert digest(parent / 'predictions.json') == policy['parent_predictions_sha256']
    report = dict(state='complete', **policy, rows=rows, gt_read=False, original_predictions_unchanged=True,
                  no_dense_accuracy_or_common_reader_claim=True, no_suppression_applied=True)
    write_json(destination / 'summary.json', report)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-camera-versions.json', report)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
