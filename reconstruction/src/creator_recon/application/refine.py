"""Read a fixed base and ROI, execute the method, validate and publish a patch; not implemented."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never

from creator_recon.errors import FeatureUnavailableError
from creator_recon.infrastructure.runtime import RuntimeContext


def refine(request: Mapping[str, object], runtime: RuntimeContext) -> Never:
    """Reserved use-case entry over a provisional envelope; no output is written."""
    raise FeatureUnavailableError("refine")
