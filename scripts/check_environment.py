"""Probe installed environments without changing Creator's scaffold capability status."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from environment_paths import require_project_environment

ROOT = Path(__file__).resolve().parents[1]

CORE_PROBE = """
import importlib.metadata as metadata
import json
import numpy as np
import scipy.linalg
import cv2
import pydantic
import trimesh
assert np.allclose(scipy.linalg.solve(np.eye(3), np.ones(3)), np.ones(3))
assert np.allclose(cv2.Rodrigues(np.zeros(3))[0], np.eye(3))
assert np.isclose(trimesh.creation.box().volume, 1.0)
print(json.dumps({'versions': {p: metadata.version(p) for p in
    ['numpy', 'scipy', 'opencv-python-headless', 'pydantic', 'Pillow', 'trimesh']}}))
"""
GPU_PROBE = """
import importlib.metadata as metadata
import json
import torch
import torchvision
import xformers
from depth_anything_3.api import DepthAnything3
assert torch.cuda.is_available(), 'CUDA unavailable'
a = torch.eye(32, device='cuda')
assert torch.allclose(a @ a, a)
boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device='cuda')
keep = torchvision.ops.nms(boxes, torch.tensor([1., .5], device='cuda'), .5)
assert keep.tolist() == [0], 'torchvision CUDA NMS failed'
torch.cuda.synchronize()
print(json.dumps({'versions': {p: metadata.version(p) for p in
    ['torch', 'torchvision', 'xformers', 'numpy', 'opencv-python',
     'depth-anything-3', 'huggingface-hub', 'open3d', 'pycolmap', 'ipywidgets']},
    'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'bf16': torch.cuda.is_bf16_supported(),
    'da3_api_import': True, 'cuda_tensor_and_nms': True}))
"""


def main() -> int:
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'environment only; no reconstruction/refinement quality claim',
              'components': {}, 'cache_paths': {key: os.environ.get(key) for key in
                  ('TEMP', 'UV_CACHE_DIR', 'HF_HOME', 'TORCH_HOME', 'MPLCONFIGDIR')}}
    try:
        require_project_environment(ROOT)
    except RuntimeError as error:
        report.update(ok=False, error=str(error))
        destination = ROOT / '.runtime' / 'environment.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(error, file=sys.stderr)
        return 2
    success = True
    for name, project, code in (
        ('core', 'reconstruction', CORE_PROBE),
        ('evaluation', 'experiments', CORE_PROBE + '\nimport creator_eval'),
        ('da3', 'backends/da3', GPU_PROBE),
    ):
        executable = ROOT / project / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not executable.is_file():
            report['components'][name] = {'ok': False, 'error': f'Missing {executable}'}
            success = False
            continue
        try:
            result = subprocess.run([str(executable), '-c', code], cwd=ROOT,
                                    capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=180, check=False)
            item = {'ok': result.returncode == 0, 'python': str(executable),
                    'stdout': result.stdout, 'stderr': result.stderr}
            if item['ok']:
                item['details'] = json.loads(result.stdout.strip().splitlines()[-1])
            else:
                success = False
            report['components'][name] = item
            print(f'{name}: {"PASS" if item["ok"] else "FAIL"}', flush=True)
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            report['components'][name] = {'ok': False, 'error': str(error)}
            success = False
    report['ok'] = success
    report['project_drive_free_gib'] = round(shutil.disk_usage(ROOT).free / 1024**3, 2)
    destination = ROOT / '.runtime' / 'environment.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(destination)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())