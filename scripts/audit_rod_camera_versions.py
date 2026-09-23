"""Reopen camera/depth identities and reject stale patches on actual new bases."""
from __future__ import annotations

import sys

import jsonschema
import numpy as np
from materialize_rod_camera_versions import PARENT, ROOT, RUN_ID
from run_rod_identity_stress import reject_truth_open
from run_rod_reference import digest, read_json, write_json

# The version runner adds reconstruction/src; keep these imports after it.
# isort: split
from creator_recon.domain.camera_point_snapshot import native_intrinsics
from creator_recon.domain.point_patch import compose, content_hash, load_snapshot


def audit():
    run = ROOT / '.runtime/experiments' / RUN_ID
    parent = ROOT / '.runtime/experiments' / PARENT
    report = read_json(run / 'summary.json')
    sys.addaudithook(reject_truth_open)
    assert report['state'] == 'complete' and not report['gt_read']
    for name, sha in report['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(run / 'source_snapshot' / name)
    assert digest(parent / 'rod-inference.json') == report['parent_rod_sha256']
    assert digest(parent / 'predictions.json') == report['parent_predictions_sha256']
    predictions = read_json(parent / 'predictions.json')
    validator = jsonschema.Draft202012Validator(read_json(ROOT / 'schemas/pilot/point-patch-0.1.schema.json'))
    schema_count, pixel_count, stale_count = 0, 0, 0
    for row in report['rows']:
        if 'snapshot_id' not in row:
            continue
        cid, variant = row['case_id'], row['camera_variant']
        folder = run / (cid + '-' + variant)
        source = read_json(folder / 'camera-depth-source.json')
        assert content_hash(source['source']) == source['geometry_source_id'] == row['geometry_source_id']
        snapshot, arrays = load_snapshot(folder / 'base')
        assert snapshot['content_id'] == row['snapshot_id']
        assert snapshot['metadata']['world_frame_id'] == 'camera-depth:' + row['geometry_source_id']
        record = next(r for r in report['tasks'] if r['case_id'] == cid and r['camera_variant'] == variant)
        assert digest(parent / record['path']) == record['sha256'] == source['source']['camera_source_sha256']
        geometry = read_json(parent / record['path'])
        prediction = next(r for r in predictions['records'] if r['case_id'] == cid)
        assert digest(ROOT / prediction['path']) == prediction['prediction_sha256'] == source['source']['source_prediction_sha256']
        with np.load(ROOT / prediction['path'], allow_pickle=False) as native:
            depth = native['depth']
        k = native_intrinsics(geometry['intrinsics'], [640, 480], list(depth.shape[:0:-1]))
        np.testing.assert_array_equal(k, source['source']['native_intrinsics'])
        np.testing.assert_array_equal(geometry['extrinsics'], source['source']['world_to_camera_cv'])
        for view in range(len(depth)):
            indices = np.flatnonzero(arrays['point_ids'][:, 0] == view)
            selected = indices[np.linspace(0, len(indices) - 1, 128, dtype=int)]
            ids = arrays['point_ids'][selected]
            pixels = np.c_[ids[:, 2], ids[:, 1], np.ones(len(ids))]
            xyz = (pixels @ np.linalg.inv(k[view]).T) * depth[view, ids[:, 1], ids[:, 2], None]
            inverse = np.linalg.inv(np.vstack([geometry['extrinsics'][view], [0, 0, 0, 1]]))
            expected = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            np.testing.assert_allclose(arrays['points'][selected], expected, atol=1e-12, rtol=0)
            pixel_count += len(ids)
        for path in [folder / 'base/manifest.json', *folder.glob('patch-*/manifest.json')]:
            validator.validate(read_json(path))
            schema_count += 1
        if variant != 'initial':
            stale = run / (cid + '-initial') / 'patch-baseline'
            try:
                compose(folder / 'base', stale)
            except ValueError as error:
                assert 'Stale patch' in str(error)
                stale_count += 1
            else:
                raise AssertionError('An old-camera patch was accepted on new geometry')
    result = dict(state='complete', run_id=RUN_ID, summary_sha256=digest(run / 'summary.json'),
        audit_source_sha256=digest(__file__), schema_count=schema_count, independent_source_pixel_checks=pixel_count,
        actual_stale_patches_rejected=stale_count, gt_read=False)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-camera-versions-audit.json', result)
    print('CAMERA_VERSION_AUDIT', result, flush=True)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit()
