"""Unchanged actual-geometry TLS with explicit sparse-stroke support labels."""

from .rgb_chain_axis_readout import ChainAxisReadout
from .rgb_stroke_chain_support import SCOPE


class StrokeChainAxisReadout(ChainAxisReadout):
    def __call__(self, segments, call_policy):
        result = super().__call__(segments, call_policy)
        result.update(measurement_scope=SCOPE+"; output fitted only to actual 3D geometry",
            support_domain_kind="stroke_conditioned_positive_only_not_all_possible",
            anchor_claims_sha256=self.context["association"]["anchors_sha256"],
            anchor_view_count=self.context["association"]["anchor_view_count"],
            anchor_point_count=self.context["association"]["anchor_point_count"],
            stroke_interpolation_performed=False)
        return result
