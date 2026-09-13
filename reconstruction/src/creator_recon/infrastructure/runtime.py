"""Explicit per-run paths; construction does not create directories or acquire resources."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Provisional in-process path context, not a serialized runtime configuration.

    Paths must eventually be resolved and authorized by the application boundary.
    Merely constructing this record does not validate them or permit any I/O.
    """

    project_root: Path
    run_dir: Path
    scratch_dir: Path
