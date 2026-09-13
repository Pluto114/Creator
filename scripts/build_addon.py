"""Build a small, reproducible legacy Blender add-on ZIP using only the stdlib."""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path


def build() -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "blender_addon" / "creator_recon"
    files = sorted(source.rglob("*.py"))
    if not (source / "__init__.py").is_file() or not files:
        raise ValueError("Blender add-on source is missing")
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Source escapes add-on directory: {path.name}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    destination = root / "dist" / "creator_recon-0.1.0.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = Path("creator_recon") / path.relative_to(source)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Add-on archive integrity check failed")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    print(build())
