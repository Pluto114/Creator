"""Positive support must include explicit foreground views in the same chain.

This is a stricter observation policy, not proof that excluded points or
unknown image rows contain no rod. It never changes actual point coordinates.
"""

import numpy as np

from .rgb_candidate_readout import POLICY, support_votes
from .rgb_chain_support import POINT_BATCH_SIZE, point_memberships


def consistent_required_view_mask(memberships, assignments, required_views, minimum_views=3):
    arrays = [np.asarray(a) for a in memberships]
    required = tuple(required_views)
    if (not arrays or type(minimum_views) is not int or not 1 <= minimum_views <= len(arrays)
            or any(a.dtype != np.bool_ or a.ndim != 2 for a in arrays)
            or len({len(a) for a in arrays}) != 1
            or any(type(i) is not int or not 0 <= i < len(arrays) for i in required)
            or len(set(required)) != len(required)):
        raise ValueError("Matching boolean memberships and distinct in-range required views needed")
    required = set(required)
    count = len(arrays[0])
    columns, caches = [], []
    for array in arrays:
        packed = np.packbits(array, axis=0, bitorder="little")
        columns.append([int.from_bytes(packed[:, i].tobytes(), "little") for i in range(array.shape[1])])
        caches.append({(): 0})
    result, universe = 0, (1 << count)-1
    for assignment in assignments:
        if len(assignment) != len(arrays):
            raise ValueError("One candidate-index set per view required")
        at_least, mandatory = [universe]+[0]*minimum_views, universe
        for vi, (array, indices, bits, cache) in enumerate(zip(arrays, assignment, columns, caches)):
            ids = np.asarray(indices)
            if len(ids) and (ids.ndim != 1 or ids.dtype.kind not in "iu"
                    or np.any(ids < 0) or np.any(ids >= array.shape[1])):
                raise ValueError("In-range integer candidate indices required")
            key = tuple(int(i) for i in ids)
            if key not in cache:
                union = 0
                for index in key:
                    union |= bits[index]
                cache[key] = union
            supported = cache[key]
            if vi in required:
                mandatory &= supported
            for level in range(minimum_views, 0, -1):
                at_least[level] |= at_least[level-1] & supported
        result |= at_least[minimum_views] & mandatory
    packed_result = np.frombuffer(result.to_bytes((count+7)//8, "little"), np.uint8)
    return np.unpackbits(packed_result, bitorder="little", count=count).astype(bool)


def required_stroke_support_mask(points, context, raw_votes=None):
    votes = support_votes(points, context["views"], POLICY) if raw_votes is None else np.asarray(raw_votes)
    raw = votes >= POLICY["minimum_views"]
    association = context["association"]
    # An incomplete search still cannot silently exclude unsearched branches.
    if association["fallback_to_raw_union"]:
        return raw.copy()
    names = {a["view_id"] for a in association["anchors"]}
    required = [i for i, view in enumerate(context["views"]) if view["view_id"] in names]
    if len(names) < 2 or len(required) != len(names):
        raise ValueError("At least two validated foreground views required")
    mask = np.zeros(len(points), bool)
    if not association["assignments"]:
        return mask
    values, indices = np.asarray(points, float), np.flatnonzero(raw)
    for start in range(0, len(indices), POINT_BATCH_SIZE):
        batch = indices[start:start+POINT_BATCH_SIZE]
        memberships = point_memberships(values[batch], context)
        mask[batch] = consistent_required_view_mask(memberships, association["assignments"], required,
            POLICY["minimum_views"])
    return mask
