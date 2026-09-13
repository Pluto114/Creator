"""Prevent setup checks from silently using default C/user cache directories."""
import os
from pathlib import Path

CACHE_KEYS = (
    'TEMP', 'TMP', 'UV_CACHE_DIR', 'PIP_CACHE_DIR', 'HF_HOME', 'HF_HUB_CACHE',
    'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR',
    'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'IMAGEIO_USERDIR',
)


def require_project_environment(root: Path) -> None:
    for key in CACHE_KEYS:
        value = os.environ.get(key)
        path = Path(value).resolve() if value else None
        if path is None or not path.is_relative_to(root.resolve()) or not path.is_dir():
            raise RuntimeError(
                f'{key} must be an existing directory inside {root}; '
                'dot-source scripts/Enter-CreatorEnvironment.ps1 first.'
            )