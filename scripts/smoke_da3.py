"""Real upstream DA3 GPU smoke test; this does not implement the Creator adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(
    args: argparse.Namespace, root: Path, destination: Path, *,
    inference_options: dict | None = None, inference_context=None,
) -> dict:
    from environment_paths import require_project_environment

    require_project_environment(root)
    manifest = json.loads((root / 'configs/models.lock.json').read_text(encoding='utf-8'))
    entry = next(model for model in manifest['models'] if model['name'] == args.model)
    model_dir = root / 'models' / f'da3-{args.model}' / entry['revision']
    for file in entry['files']:
        path = model_dir / file['name']
        if not path.is_file() or path.stat().st_size != file['bytes']:
            raise RuntimeError(f'Missing or incomplete model file: {path}')
        if file['sha256'] and sha256(path) != file['sha256']:
            raise RuntimeError(f'Model hash mismatch: {path}')
    for path in args.images:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Loading must use the already verified local snapshot, including in offline runs.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    package = importlib.metadata.distribution('depth-anything-3')
    direct_url = json.loads(package.read_text('direct_url.json') or '{}')
    actual_commit = direct_url.get('vcs_info', {}).get('commit_id')
    if actual_commit != manifest['upstream_commit']:
        raise RuntimeError(f'Unexpected installed DA3 source commit: {actual_commit}')

    if not torch.cuda.is_available():
        raise RuntimeError('This smoke test requires a CUDA GPU; no silent CPU fallback.')
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(model_dir), local_files_only=True).to('cuda')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    # Ordinary RGB runs keep exactly the old call. The separate oracle worker supplies
    # known cameras and a read-only observer so API pose replacement cannot hide in the report.
    with inference_context(model) if inference_context is not None else nullcontext():
        prediction = model.inference(
            [str(path.resolve()) for path in args.images], process_res=args.process_res,
            infer_gs=False, use_ray_pose=getattr(args, 'use_ray_pose', False),
            ref_view_strategy='saddle_balanced',
            export_dir=None,
            **(inference_options or {}),
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    arrays = {name: getattr(prediction, name) for name in
              ('depth', 'conf', 'intrinsics', 'extrinsics', 'processed_images')}
    count = len(args.images)
    if arrays['depth'] is None or arrays['depth'].ndim != 3:
        raise RuntimeError('Missing [N,H,W] depth output')
    shape = arrays['depth'].shape
    expected = {'depth': shape, 'conf': shape, 'intrinsics': (count, 3, 3),
                'extrinsics': (count, 3, 4), 'processed_images': (*shape, 3)}
    if shape[0] != count:
        raise RuntimeError('Input/output view count mismatch')
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.shape != expected[name]:
            raise RuntimeError(f'Unexpected output shape for {name}')
        if not np.isfinite(value).all():
            raise RuntimeError(f'Non-finite {name}')
    if not (arrays['depth'] > 0).all():
        raise RuntimeError('Non-positive depth')
    sky = getattr(prediction, 'sky', None)
    if sky is not None:
        if not isinstance(sky, np.ndarray) or sky.shape != shape or sky.dtype != np.bool_:
            raise RuntimeError('Unexpected upstream sky mask')
        arrays['sky'] = sky
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'prediction.npz', **arrays)
    report = {
        'ok': True, 'scope': 'upstream inference smoke, not a research benchmark',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'upstream_commit': actual_commit,
        'model': entry['repo_id'], 'model_revision': entry['revision'],
        'images': [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in args.images],
        'versions': {p: importlib.metadata.version(p) for p in
                     ('torch', 'torchvision', 'numpy', 'depth-anything-3')},
        'process_res': args.process_res, 'process_res_method': 'upper_bound_resize',
        'infer_gs': False, 'use_ray_pose': getattr(args, 'use_ray_pose', False),
        'ref_view_strategy': 'saddle_balanced',
        'sky_available': sky is not None,
        'is_metric': (int(prediction.is_metric)
                      if isinstance(prediction.is_metric, (bool, int, np.integer)) else None),
        'gpu': torch.cuda.get_device_name(0), 'cuda_runtime': torch.version.cuda,
        'load_seconds': round(load_seconds, 3),
        'inference_seconds': round(inference_seconds, 3),
        'gpu_free_before_mib': round(free_before / 1024**2, 1),
        'gpu_total_mib': round(total_memory / 1024**2, 1),
        'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        'outputs': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                    for k, v in arrays.items()},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('base', 'large'), default='base')
    parser.add_argument('--images', type=Path, nargs='+', required=True)
    parser.add_argument('--process-res', type=int, default=504)
    parser.add_argument('--use-ray-pose', action='store_true')
    args = parser.parse_args()
    if len(args.images) < 2 or args.process_res < 28:
        parser.error('Use at least two images and a processing resolution >= 28.')
    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination = root / '.runtime' / 'smoke' / f'da3-{args.model}' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / 'report.json'
    initial = {'ok': False, 'state': 'running', 'run_id': run_id, 'model': args.model}
    report_path.write_text(json.dumps(initial, indent=2), encoding='utf-8')
    try:
        report = run(args, root, destination)
    except Exception as error:
        initial.update(state='failed', error=repr(error))
        report_path.write_text(json.dumps(initial, indent=2), encoding='utf-8')
        raise
    report.update(state='succeeded', run_id=run_id)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(report_path)


if __name__ == '__main__':
    main()