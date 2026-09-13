"""Shared task vocabulary; these enums are not a RunRequest or RunStatus schema."""

from enum import StrEnum


class RunKind(StrEnum):
    """The four documented research task kinds."""

    RECONSTRUCT = "reconstruct"
    REFINE = "refine"
    EXPORT_PREVIEW = "export_preview"
    EVALUATE = "evaluate"


class RunState(StrEnum):
    """Lifecycle vocabulary; transition enforcement is not implemented yet."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
