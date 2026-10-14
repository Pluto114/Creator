"""A separately named native-patch refinement, never a common-reader switch.

The original baseline is preserved as a control. Apply the same measured
four-view outer envelope to every baseline candidate's own finite axis;
there is no depth refit, endpoint extension, interior fill or case selection.
"""

import numpy as np

from .consensus_boundary_readout import trim_consensus_envelope
from .rgb_chain_support import DEFAULTS
from .rod_fixture_finite import canonical_hash

POLICY = dict(method="baseline_boundary", version="boundary-patch-1",
    minimum_boundary_views=DEFAULTS["minimum_chain_views"], horizontal_boundary_padding_px=0.,
    axis_policy="unchanged_native_baseline_axis", extent_policy="outer_intersection_only",
    interior_policy="preserve_original_components_and_gaps", base_point_suppression=False,
    input_context="shared_rgb_foreground_strokes_all_admissible_same_chain_assignments")


def refine_boundary(segments, context):
    segments = np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Finite Mx2x3 native baseline segments required")
    if context["association"]["fallback_to_raw_union"]:
        raise ValueError("Complete original foreground chain context required")
    result = trim_consensus_envelope(segments, context)
    return dict(result, state="accepted" if len(result["segments"]) else "rejected",
        rejection_reasons=[] if len(result["segments"]) else ["no_four_view_native_boundary_support"],
        method="baseline_boundary", policy=dict(POLICY), method_sha256=canonical_hash(POLICY),
        parent_native_sha256=canonical_hash(segments), context_sha256=context["sha256"],
        gt_read=False, extra_annotations=0, native_axis_refitted=False,
        original_baseline_preserved_as_control=True, common_reader_changed=False)
