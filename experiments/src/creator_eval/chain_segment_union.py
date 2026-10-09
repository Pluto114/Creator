"""Only machine-collinear interval union; never average distinct hypotheses."""

import numpy as np


def merge_collinear_segments(segments):
    values = np.asarray(segments, float)
    if values.size == 0:
        return np.empty((0, 2, 3))
    if values.ndim != 3 or values.shape[1:] != (2, 3) or not np.isfinite(values).all():
        raise ValueError("Finite Mx2x3 segments required")
    lengths = np.linalg.norm(values[:, 1]-values[:, 0], axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Nonzero segments required")
    ordered = []
    for segment in values:
        ends = sorted(tuple(float(v) for v in point) for point in segment)
        ordered.append(ends)
    ordered = np.asarray(sorted(ordered))
    groups = []
    for segment in ordered:
        direction = segment[1]-segment[0]
        length = np.linalg.norm(direction)
        unit = direction/length
        for group in groups:
            anchor, axis, reference_length, members = group
            tolerance = 1e-10*max(1., length, reference_length)
            if (np.linalg.norm(np.cross(unit, axis)) <= 1e-10
                    and np.max(np.linalg.norm(np.cross(segment-anchor, axis), axis=1)) <= tolerance):
                members.append(segment)
                break
        else:
            groups.append((segment[0], unit, length, [segment]))
    output = []
    for anchor, axis, _, members in groups:
        intervals = []
        for segment in members:
            coordinates = (segment-anchor) @ axis
            a, b = np.argsort(coordinates)
            intervals.append((float(coordinates[a]), float(coordinates[b]), tuple(segment[a]), tuple(segment[b])))
        intervals.sort()
        _, right, first, last = intervals[0]
        for begin, end, start_point, end_point in intervals[1:]:
            # No tolerance here: a positive physical gap is never closed.
            if begin <= right:
                if end > right:
                    right, last = end, end_point
            else:
                output.append(sorted((first, last)))
                right, first, last = end, start_point, end_point
        output.append(sorted((first, last)))
    # Endpoints are selected from the original segments, not projected or averaged.
    return np.asarray(sorted(output), float).reshape(-1, 2, 3)
