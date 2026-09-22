"""Descriptor-only tracks and leakage-aware splits, not a correspondence certificate."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


def components(nodes, edges):
    adjacency = {n: set() for n in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen, result = set(nodes), []
    while unseen:
        stack, found = [min(unseen)], set()
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(adjacency[node] - found)
        unseen -= found
        result.append(sorted(found))
    return result


def build_tracks(pairs):
    """Require all direct matches in a >=3-view component; reject view conflicts.

    A three-view cycle can still follow the wrong repeated brick. Keeping this
    function camera-free makes that limitation testable instead of hiding it in BA.
    """
    edges = set()
    for pair in pairs:
        a, b = pair["first_view"], pair["second_view"]
        if not 0 <= a < b:
            raise ValueError("Ordered, distinct nonnegative view IDs required")
        ids = np.asarray(pair["keypoint_ids"], dtype=np.int64).reshape(-1, 2)
        if (ids < 0).any() or len(set(ids[:, 0])) != len(ids) or len(set(ids[:, 1])) != len(ids):
            raise ValueError("Pair matching must be one-to-one")
        edges.update(((a, int(i)), (b, int(j))) for i, j in ids)
    nodes = {n for edge in edges for n in edge}
    adjacency = {n: set() for n in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    triangle_edges = {edge for edge in edges if adjacency[edge[0]] & adjacency[edge[1]]}
    tracks, rejected = [], []
    for group in components(nodes, edges):
        views = [node[0] for node in group]
        if len(views) != len(set(views)):
            reason = "multiple_features_in_one_view"
        elif len(group) < 3:
            reason = "fewer_than_three_views"
        elif not all((a, b) in edges for a, b in combinations(group, 2)):
            reason = "missing_direct_cross_view_match"
        else:
            tracks.append(group)
            continue
        rejected.append(dict(nodes=group, reason=reason))
    return dict(tracks=tracks, rejected_components=rejected, triangle_edges=sorted(triangle_edges),
                pair_edge_count=len(edges), triangle_edge_count=len(triangle_edges))


def split_tracks(tracks, features, config):
    """Group tracks sharing a cell in ANY view before assigning train/validation.

    We do not rebalance a bad split after seeing residuals. Repeated descriptors
    can join a huge group; in that case there may simply be no useful holdout.
    """
    if config["cell_size_px"] <= 0 or config["validation_modulus"] < 2 or not 0 <= config["validation_residue"] < config["validation_modulus"]:
        raise ValueError("Invalid spatial split")
    cells, owners, edges = [], {}, set()
    for index, track in enumerate(tracks):
        occupied = set()
        for view, feature in track:
            xy = np.asarray(features[view]["xy"][feature], float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Invalid feature coordinate")
            x, y = np.floor(xy / config["cell_size_px"]).astype(int)
            cell = (view, int(x), int(y))
            occupied.add(cell)
            if cell in owners:
                edges.add((owners[cell], index))
            else:
                owners[cell] = index
        cells.append(occupied)
    validation = np.zeros(len(tracks), bool)
    groups = []
    for indices in components(range(len(tracks)), edges):
        group_cells = sorted(set().union(*(cells[i] for i in indices)))
        key = str(config["seed"]) + ":" + repr(group_cells)
        is_validation = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % config["validation_modulus"] == config["validation_residue"]
        validation[indices] = is_validation
        groups.append(dict(track_ids=indices, cells=group_cells, validation=is_validation))
    train_cells = set().union(*(cells[i] for i in range(len(tracks)) if not validation[i]))
    val_cells = set().union(*(cells[i] for i in range(len(tracks)) if validation[i]))
    return validation, dict(groups=groups, train_tracks=int((~validation).sum()), validation_tracks=int(validation.sum()),
                           shared_cells=len(train_cells & val_cells), cell_size_px=config["cell_size_px"])


def pair_track_labels(pairs, graph, validation):
    """Attach track IDs; -1 means no complete track, never a negative example."""
    lookup = {edge: i for i, track in enumerate(graph["tracks"]) for edge in combinations(map(tuple, track), 2)}
    triangles = {tuple(map(tuple, edge)) for edge in graph["triangle_edges"]}
    labels = []
    for pair in pairs:
        edges = [((pair["first_view"], int(a)), (pair["second_view"], int(b))) for a, b in pair["keypoint_ids"]]
        track_ids = np.array([lookup.get(edge, -1) for edge in edges], dtype=int)
        heldout = np.array([bool(validation[i]) if i >= 0 else False for i in track_ids], bool)
        labels.append(dict(track_ids=track_ids, validation=heldout, triangle=np.array([edge in triangles for edge in edges], bool)))
    return labels
