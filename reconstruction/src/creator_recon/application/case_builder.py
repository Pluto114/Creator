"""Case import boundary; immutable input preparation and EXIF handling are pending."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Never

from creator_recon.contracts.control import CancellationToken
from creator_recon.errors import FeatureUnavailableError


class CaseBuilder:
    """Future owner of case validation and transactional input preparation."""

    def create(
        self,
        source_paths: Sequence[Path],
        options: Mapping[str, object],
        output_dir: Path,
        cancel: CancellationToken,
    ) -> Never:
        """Reserved contract: no photo copying, directory creation, or manifest writes."""
        raise FeatureUnavailableError("case import")

    def validate_existing(self, case_ref: object) -> Never:
        """Reserved until ArtifactRef, DTO and content-integrity checks are implemented."""
        raise FeatureUnavailableError("case validation")


def create_case_from_request(request: Mapping[str, object], cancel: CancellationToken) -> Never:
    """CLI boundary over an unvalidated envelope; never treats it as a valid DTO."""
    raise FeatureUnavailableError("case import")
