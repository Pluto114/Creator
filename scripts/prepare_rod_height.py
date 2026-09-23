"""Frozen height intervention, using the established paired export implementation."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from run_foreground_estimated_pilot import require_camera_free
from run_rod_identity_blender import (
    ROOT,
    digest,
    locations,
    raster_center_truth,
    read_json,
    validate_render_manifest,
    write_json,
)

CONFIG = ROOT / 'configs/rod_height_v1.json'


def source_snapshot(run, names):
    hashes = {}
    for name in sorted(set(names)):
        dest = run / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        hashes[name] = digest(dest)
    return hashes


def checked_scene(run_id):
    run, inputs, truth, _ = locations(run_id)
    frozen = read_json(run / 'prepared.json')
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT / name) != sha or digest(run / 'source_snapshot' / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    if digest(inputs / 'manifest.json') != frozen['input_sha256']:
        raise ValueError('RGB manifest changed')
    manifest = read_json(inputs / 'manifest.json')
    require_camera_free(manifest)
    return run, inputs, truth, frozen, manifest


def prepare(path):
    from thin_pack_gt import MeshRays, check_blender_rays
    config = read_json(path)
    run, inputs, truth, evaluation = locations(config['run_id'])
    for folder in (run, inputs, truth, evaluation):
        if folder.exists():
            raise FileExistsError('Preserve earlier reference experiments: ' + str(folder))
    for folder in (run, inputs, truth):
        folder.mkdir(parents=True)
    write_json(run / 'protocol.json', config)
    names = set(config['sources'])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'experiments/src/creator_eval').glob('*.py'))
    hashes = source_snapshot(run, names)
    write_json(run / 'generation-freeze.json', dict(source_sha256=hashes, protocol_sha256=digest(run / 'protocol.json')))
    environment = os.environ.copy()
    for key in ('CONFIG', 'SCRIPTS', 'EXTENSIONS', 'DATAFILES'):
        folder = ROOT / '.local/blender-profile' / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        environment['BLENDER_USER_' + key] = str(folder)
    batches = []
    for i, case in enumerate(config['cases']):
        generator = {**config['generator'], 'reference_seed': case['reference_seed'], 'lens_by_view_mm': case['lens_by_view_mm']}
        request = dict(root=str(ROOT), inputs=inputs.relative_to(ROOT).as_posix(), truth=truth.relative_to(ROOT).as_posix(),
                       config={**config, 'generator': generator, 'cases': [case]}, render_manifest_name=f'render-{i:02d}.json')
        request_path = run / f'case-{i:02d}-render_request.json'
        write_json(request_path, request)
        with (run / f'blender-{i:02d}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run([config['blender'], '--background', '--factory-startup', '--disable-autoexec',
                '--python-exit-code', '1', '--python', str(ROOT / 'scripts/blender_rod_height_pack.py'), '--', str(request_path)],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'Render failed; preserve blender-{i:02d}.log')
        batches.append(read_json(truth / f'render-{i:02d}.json'))
        print('REFERENCE_RENDER', case['case_id'], flush=True)
    render = {**batches[0], 'cases': [c for batch in batches for c in batch['cases']]}
    write_json(truth / 'render_manifest.json', render)
    validate_render_manifest(render, config, inputs, truth)
    cases, truths, probes = [], [], []
    for rendered, declared in zip(render['cases'], config['cases']):
        cid = declared['case_id']
        assert rendered['case_id'] == cid
        rays = MeshRays(truth / cid / 'mesh.npz')
        frames, cameras, truth_frames = [], [], []
        for frame in rendered['frames']:
            camera, vid = frame['camera'], frame['frame_id']
            native = truth / cid / vid
            native.mkdir()
            arrays = raster_center_truth(rays, camera)
            for name, array in arrays.items():
                np.save(native / (name + '.npy'), array, allow_pickle=False)
            check = check_blender_rays(rays, camera, frame['blender_ray_probes'], .0001)
            probes.append(check)
            rgb = inputs / frame['rgb']
            frames.append(dict(view_id=vid, rgb=rgb.relative_to(ROOT).as_posix(), rgb_sha256=digest(rgb),
                               size_wh=camera['size_wh'], guide_xyxy=frame['guide_xyxy'], guide_source=frame['guide_source']))
            cameras.append(camera)
            truth_frames.append(dict(view_id=vid, target_visible_pixels=int(arrays['target_visible'].sum()),
                                     ray_check=check, projection_max_px=frame['projection_check_max_px']))
        cases.append(dict(case_id=cid, frames=frames))
        truths.append(dict(case_id=cid, declared=declared, cameras=cameras, frames=truth_frames, mesh_path=f'{cid}/mesh.npz'))
    clean = dict(cases=cases, method=config['tracks'])
    require_camera_free(clean)
    write_json(inputs / 'manifest.json', clean)
    write_json(truth / 'manifest.json', dict(cases=truths, input_sha256=digest(inputs / 'manifest.json')))
    artifact_hashes = {p.relative_to(truth).as_posix(): digest(p) for p in sorted(truth.rglob('*')) if p.is_file()}
    write_json(truth / 'artifact_hashes.json', artifact_hashes)
    write_json(run / 'prepared.json', dict(run_id=config['run_id'], source_sha256=hashes,
        input_sha256=digest(inputs / 'manifest.json'), truth_sha256=digest(truth / 'manifest.json'),
        truth_artifacts_sha256=digest(truth / 'artifact_hashes.json'), protocol_sha256=digest(run / 'protocol.json'),
        independent_ray_samples=sum(p['samples'] for p in probes), surface_id_mismatches=sum(p['surface_id_mismatches'] for p in probes)))
    print('REFERENCE_PREPARED', len(cases), 'camera-free RGB groups', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    prepare(args.config)
