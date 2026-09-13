"""Validate inputs, invoke the isolated backend, normalize and publish a base snapshot; not implemented."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never

from creator_recon.errors import FeatureUnavailableError
from creator_recon.infrastructure.runtime import RuntimeContext


def reconstruct(request: Mapping[str, object], runtime: RuntimeContext) -> Never:
    """Reserved use-case entry over a provisional envelope; no output is written."""
    raise FeatureUnavailableError("reconstruct")
