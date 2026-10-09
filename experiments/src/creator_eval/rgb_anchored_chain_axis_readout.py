"""Unchanged TLS, explicitly labelled anchor-conditioned positive support."""

from .rgb_anchored_chain_support import SCOPE
from .rgb_chain_axis_readout import ChainAxisReadout


class AnchoredChainAxisReadout(ChainAxisReadout):
    def __call__(self, segments, call_policy):
        result = super().__call__(segments, call_policy)
        result.update(measurement_scope=SCOPE+"; output fitted only to actual 3D geometry",
            support_domain_kind="anchor_conditioned_positive_only_not_all_possible",
            anchor_claims_sha256=self.context["association"]["anchors_sha256"],
            anchor_view_count=self.context["association"]["anchor_view_count"])
        return result
