"""Future multi-view candidate method; the algorithm has not been implemented or validated."""

from __future__ import annotations

from typing import Any, Never

from creator_recon.contracts.control import CancellationToken, ProgressSink
from creator_recon.errors import FeatureUnavailableError
from creator_recon.refinement.api import RefinementContext


class MultiViewCurveRefiner:
    """Reserved first method entry; no rods, patches or evidence are synthesized."""

    def refine(
        self,
        context: RefinementContext[Any, Any, Any, Any, Any],
        cancel: CancellationToken,
        progress: ProgressSink,
    ) -> Never:
        """Do not disguise absent code as a valid no_supported_change result."""
        raise FeatureUnavailableError("multi-view thin-structure refinement")
