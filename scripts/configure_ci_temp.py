"""Give CI subprocesses one canonical spelling of the existing temp directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def canonical_temp_environment():
    # Windows runners may expose RUNNER~1 while Path.resolve expands runneradmin.
    # Preserve the actual directory; do not weaken receipt/path-boundary checks.
    directory = str(Path(tempfile.gettempdir()).resolve(strict=True))
    if "\n" in directory or "\r" in directory:
        raise ValueError("CI temp directory must be a single line")
    return dict(TMPDIR=directory, TEMP=directory, TMP=directory)


def main():
    environment = canonical_temp_environment()
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as handle:
        for key, value in environment.items():
            handle.write(f"{key}={value}\n")
    print("CI temporary directory canonicalized (same directory, all three aliases).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
