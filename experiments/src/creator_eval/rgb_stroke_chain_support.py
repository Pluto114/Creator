"""Sparse image foreground strokes: the same candidate must support all marks.

No stroke interpolation, 3D point correspondences, geometry fitting, or GT.
Each point remains an explicit uncertain foreground-membership claim.
"""

import copy

from .rgb_anchored_chain_support import condition_context
from .rgb_chain_support import build_context as build_all_chain_context
from .rod_fixture_finite import canonical_hash, json_ready

SCOPE = ("Positive support conditional on all explicit sparse foreground stroke marks in each view; "
         "no interpolation between marks and no unique target identity claim")


def condition_stroke_context(context, evidence, cameras, anchors):
    groups = {}
    for anchor in anchors:
        groups.setdefault(anchor["view_id"], []).append(anchor)
    if len(groups) < 2 or any(len(items) < 2 for items in groups.values()):
        raise ValueError("At least two marks in each of at least two independent stroke views required")
    for items in groups.values():
        # Near-vertical candidate family: marks must constrain distinct rows.
        if len({float(a["xy"][1]) for a in items}) < 2:
            raise ValueError("Each sparse stroke must cover distinct image rows")
        items.sort(key=lambda a: (a["xy"][1], a["xy"][0]))
    ordered = [groups[name] for name in sorted(groups)]
    conditioned, layers = context, []
    for index in range(max(len(items) for items in ordered)):
        # Repeating an already-tested last mark is a no-op for shorter strokes;
        # it keeps the existing independent-view validation at every layer.
        marks = [items[min(index, len(items)-1)] for items in ordered]
        conditioned = condition_context(conditioned, evidence, cameras, marks)
        result = conditioned["association"]
        layers.append(dict(mark_index=index, context_sha256=conditioned["sha256"],
            assignment_sha256=result["assignment_sha256"], chain_count=result["chain_count"],
            chain_state_counts=result["chain_state_counts"], anchor_audit=result["anchor_audit"],
            reason=result["reason"]))
    association = copy.deepcopy(conditioned["association"])
    all_marks = [anchor for items in ordered for anchor in items]
    association.update(scope=SCOPE, anchors=json_ready(all_marks), anchors_sha256=canonical_hash(all_marks),
        anchor_view_count=len(groups), anchor_point_count=len(all_marks),
        original_chain_count=context["association"]["chain_count"],
        unconditioned_context_sha256=context["sha256"],
        unconditioned_assignment_sha256=context["association"]["assignment_sha256"],
        conditioning_layers=layers, stroke_interpolation_performed=False,
        anchor_audit=[audit for layer in layers for audit in layer["anchor_audit"]],
        target_identity_confirmed=False)
    return dict(views=context["views"], row_members=context["row_members"], association=association,
        sha256=canonical_hash(association))


def build_context(evidence, cameras, anchors, policy=None):
    return condition_stroke_context(build_all_chain_context(evidence, cameras, policy), evidence, cameras, anchors)
