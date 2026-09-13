"""Explicit scaffold failures, separate from successful no-supported-change results."""


class FeatureUnavailableError(NotImplementedError):
    """A requested feature has no implementation and cannot publish an artifact."""

    code = "E_NOT_IMPLEMENTED"

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(
            f"{feature} is not implemented in this scaffold; no result has been generated."
        )


class OperationCancelledError(Exception):
    """Cooperative cancellation; never a successful empty refinement result."""
