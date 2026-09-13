"""Single future owner of use-case dispatch and run publication ordering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never

from creator_recon.contracts.control import CancellationToken, ProgressSink
from creator_recon.errors import FeatureUnavailableError


class RunService:
    """Reserved coordinator; no request is accepted as a validated RunRequest yet."""

    def execute(
        self,
        request: Mapping[str, object],
        cancel: CancellationToken,
        progress: ProgressSink,
    ) -> Never:
        """Fail before creating a run, launching a backend, or publishing any status."""
        raise FeatureUnavailableError("research job execution")
