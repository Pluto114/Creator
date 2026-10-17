"""Same actual-coordinate input admission for base and all candidate arms."""

import numpy as np

from . import raster_band_readout as reader
from .common_readout import sample_segments
from .rgb_chain_support import chain_support_mask
from .rgb_single_axis_readout import MAXIMUM_CURVE_SAMPLES
from .rgb_stroke_selected_axis_readout import StrokeSelectedAxisReadout


class RasterBandReadout(StrokeSelectedAxisReadout):
    def __call__(self, segments, call_policy):
        if set(call_policy) != {"voxel_size", "origin"}:
            raise ValueError("Only common frozen voxel_size and origin accepted")
        size = call_policy["voxel_size"]
        if isinstance(size, (bool, np.bool_)) or not np.isfinite(size) or size <= 0:
            raise ValueError("Positive finite voxel size required")
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 actual segments required")
        samples = sample_segments(segments, size/2, MAXIMUM_CURVE_SAMPLES)
        selected = chain_support_mask(samples, self.context)
        graph = reader.readout(np.concatenate((self.points, samples[selected])), size,
                               call_policy["origin"], self.context)
        graph.update(full_input_point_count=self.input_point_count,
            supported_base_point_count=len(self.points), full_input_segment_count=len(segments),
            full_curve_sample_count=len(samples), supported_curve_sample_count=int(selected.sum()),
            chain_context_sha256=self.context["sha256"], call_policy=call_policy,
            anchor_point_count=self.context["association"]["anchor_point_count"],
            anchor_claims_sha256=self.context["association"]["anchors_sha256"],
            input_support_intervention=True, extra_annotations=0,
            base_curve_sampling_and_ranking_identical=True)
        return graph
