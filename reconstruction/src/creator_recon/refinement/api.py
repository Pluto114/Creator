"""Refinement interface independent of Blender, model packages and evaluation answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from creator_recon.contracts.control import CancellationToken, ProgressSink

CaseT = TypeVar("CaseT")
BaseT = TypeVar("BaseT")
RegionsT = TypeVar("RegionsT")
ImagesT = TypeVar("ImagesT")
ConfigT = TypeVar("ConfigT")
ContextT_contra = TypeVar("ContextT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class RefinementContext(Generic[CaseT, BaseT, RegionsT, ImagesT, ConfigT]):
    """In-process context shape; full domain/DTO types will bind the generic members.

    No ground truth, evaluation report or Blender objects belong here. Frozen
    prevents rebinding only; future readers must also enforce array immutability
    and method-visible frame restrictions. This is not a wire schema or validator.
    """

    case: CaseT
    base: BaseT
    regions: RegionsT
    images: ImagesT
    method_config: ConfigT
    scratch_dir: Path


class Refiner(Protocol[ContextT_contra, ResultT_co]):
    """Replaceable method boundary; result type becomes PatchResult after DTO implementation."""

    def refine(
        self,
        context: ContextT_contra,
        cancel: CancellationToken,
        progress: ProgressSink,
    ) -> ResultT_co:
        """Return a validated candidate, or propagate failure; never fabricate success."""
        ...
