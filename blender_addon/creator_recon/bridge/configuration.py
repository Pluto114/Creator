"""Read-only path validation; does not import or execute the external core."""

from pathlib import Path


def validate_configuration(core_python: str, project_root: str) -> tuple[str, ...]:
    """Return path issues; existing files do not prove a working environment."""
    issues = []
    for label, value, is_directory in (
        ("Core Python", core_python, False),
        ("Project root", project_root, True),
    ):
        if not value.strip():
            issues.append(f"{label} is not configured.")
            continue
        path = Path(value)
        if not path.is_absolute():
            issues.append(f"{label} must be an absolute path.")
            continue
        try:
            if is_directory:
                if not path.is_dir():
                    issues.append("Project root is not an existing directory.")
                elif not (path / "reconstruction" / "pyproject.toml").is_file():
                    issues.append("Project root has no reconstruction/pyproject.toml.")
            elif not path.is_file():
                issues.append("Core Python is not an existing file.")
        except (OSError, ValueError):
            issues.append(f"{label} could not be inspected.")
    return tuple(issues)
