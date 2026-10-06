"""Repair voxel-center connectivity using observed training coordinates.

Keep the frozen cell-neighborhood edges, but also connect training cells whose
actual transverse samples are within the SAME physical connection radius.
Held-out samples never nominate an edge. They may attach only to one already
fixed training component. This changes grouping, not section acceptance gates.
"""

import itertools

import numpy as np
from scipy.spatial import cKDTree

from .section_evidence import _basis


def groups(points, train_mask, anchor, axis, p):
    xy = (points-anchor) @ _basis(axis)
    size = p["section_cell_voxels"]*p["voxel_size"]
    coordinates = xy/size
    if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**52):
        raise ValueError("Section coordinates exceed exact grid precision")
    cells, inverse = np.unique(np.floor(coordinates).astype(np.int64), axis=0, return_inverse=True)
    if len(cells) > p["section_maximum_cells"]:
        return None, dict(state="unmeasurable", reason="section_transverse_cell_budget_exceeded")
    train_cells = sorted(set(inverse[train_mask].tolist()))
    lookup = {tuple(cells[i]): i for i in train_cells}
    radius = p["section_connection_voxels"]/p["section_cell_voxels"]
    physical = p["section_connection_voxels"]*p["voxel_size"]
    # Cell-box distance provides a conservative finite search envelope; it is
    # never itself a new acceptance or connection radius.
    extent = int(np.ceil(radius))+1
    offsets = [delta for delta in itertools.product(range(-extent, extent+1), repeat=2)
               if sum(max(0, abs(x)-1)**2 for x in delta) <= radius*radius+1e-12]
    legacy = {delta for delta in offsets if sum(x*x for x in delta) <= radius*radius+1e-12}
    order = np.argsort(inverse[train_mask], kind="stable")
    indices = np.flatnonzero(train_mask)[order]
    labels = inverse[indices]
    support = {i: xy[indices[np.searchsorted(labels, i):np.searchsorted(labels, i, side="right")]]
               for i in train_cells}
    trees = {i: cKDTree(value) for i, value in support.items()}
    parents = {i: i for i in train_cells}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    repairs = []
    for i in train_cells:
        for delta in offsets:
            j = lookup.get(tuple(cells[i]+delta))
            if j is None or j <= i or root(i) == root(j):
                continue
            distance = None
            if delta not in legacy:
                distance = float(trees[j].query(support[i], k=1)[0].min())
                if distance > physical:
                    continue
            parents[root(j)] = root(i)
            if distance is not None:
                repairs.append(dict(first_cell=cells[i].tolist(), second_cell=cells[j].tolist(),
                                    training_distance_m=distance))
    roots = sorted({root(i) for i in train_cells})
    if len(roots) > p["section_maximum_groups"]:
        return None, dict(state="unmeasurable", reason="section_group_budget_exceeded")
    numbering = {value: i for i, value in enumerate(roots)}
    group_of = {i: numbering[root(i)] for i in train_cells}
    cell_assignment = np.full(len(cells), -1, dtype=np.int64)
    for cell, number in group_of.items():
        cell_assignment[cell] = number
    assignment = cell_assignment[inverse]
    # Validation can attach to a unique training group but cannot bridge two.
    # Process only unassigned cells; each query uses original training support.
    for i in sorted(set(inverse[~train_mask].tolist())-set(train_cells)):
        members = np.flatnonzero(inverse == i)
        candidates = [set() for _ in members]
        for delta in offsets:
            j = lookup.get(tuple(cells[i]+delta))
            if j is None:
                continue
            close = (np.ones(len(members), dtype=bool) if delta in legacy
                     else trees[j].query(xy[members], k=1)[0] <= physical)
            for k in np.flatnonzero(close):
                candidates[k].add(group_of[j])
        for index, choices in zip(members, candidates):
            if len(choices) == 1:
                assignment[index] = choices.pop()
    return [np.flatnonzero(assignment == i) for i in range(len(roots))], dict(
        transverse_cells=len(cells), training_transverse_cells=len(train_cells),
        training_group_count=len(roots), unassigned_validation_points=int((assignment < 0).sum()),
        metric_grouping=dict(training_only=True, legacy_edges_preserved=True,
                             physical_connection_radius_m=physical, repairs=repairs,
                             validation_can_bridge=False))
