"""The small replaceable boundaries shared by application and refinement code."""

from typing import Protocol


class CancellationToken(Protocol):
    """Cooperative cancellation checked at bounded work and stage boundaries."""

    def is_requested(self) -> bool:
        """Return whether the operation should stop."""
        ...

    def raise_if_requested(self) -> None:
        """Raise OperationCancelledError if cancellation has been requested."""
        ...


class ProgressSink(Protocol):
    """Progress reports are observations, never publication of a successful result."""

    def report(
        self,
        stage: str,
        completed: int | None,
        total: int | None,
        message: str,
    ) -> None:
        """Report measured work; unknown totals remain None."""
        ...


class NeverCancelled:
    """Synchronous shell token; process/file cancellation is a later milestone."""

    def is_requested(self) -> bool:
        return False

    def raise_if_requested(self) -> None:
        return None


class NullProgressSink:
    """Discard progress for the currently non-executing shell."""

    def report(
        self,
        stage: str,
        completed: int | None,
        total: int | None,
        message: str,
    ) -> None:
        return None
