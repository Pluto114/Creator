"""Validate scaffold syntax and failure boundaries without ML dependencies."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def check() -> None:
    root = Path(__file__).resolve().parents[1]
    components = (
        ("creator_recon", "reconstruction"),
        ("creator_da3", "backends/da3"),
        ("creator_eval", "experiments"),
    )
    source_roots = [root / project / "src" for _, project in components]
    source_roots.extend((root / "blender_addon", root / "scripts"))
    source_count = 0
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            source_count += 1
    for _, project in components:
        tomllib.loads((root / project / "pyproject.toml").read_text(encoding="utf-8"))
    for module, project in components:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / project / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory(prefix="creator-scaffold-") as temporary:
            request = Path(temporary) / "request.json"
            request.write_text("{}", encoding="utf-8")
            for args in (("--help",), ("--version",), ("status",)):
                result = subprocess.run(
                    [sys.executable, "-m", module, *args],
                    cwd=temporary,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=20,
                    check=False,
                )
                if result.returncode != 0:
                    raise AssertionError(f"{module} {args} failed: {result.stderr}")
                if args == ("status",):
                    if not isinstance(json.loads(result.stdout), dict):
                        raise AssertionError(f"{module} status is not a JSON object")
            result = subprocess.run(
                [sys.executable, "-m", module, "run", "--request", str(request)],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
                check=False,
            )
            if result.returncode == 0 or "NOT_IMPLEMENTED" not in result.stderr:
                raise AssertionError(f"{module} did not reject its unimplemented operation")
            if list(Path(temporary).iterdir()) != [request]:
                raise AssertionError(f"{module} created unexpected artifacts")
    print(
        f"Scaffold checks passed: {source_count} Python sources and three isolated CLI boundaries."
    )


if __name__ == "__main__":
    check()
