"""Deterministic local training directions for minority straight members.

This nominates axes only. The caller must independently fit and validate every
proposal; local PCA is not acceptance evidence and never sees held-out points.
"""

from __future__ import annotations

import numpy as np

from .common_readout_split import principal_axes

DEFAULTS = dict(local_proposal_neighbors=12, local_proposal_maximum_seeds=20000,
                local_proposal_maximum_count=2048)


def policy(config):
    p = {**DEFAULTS, **{key: value for key, value in config.items() if key.startswith("local_proposal_")}}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown local direction proposal policy")
    for value in p.values():
        if type(value) is not int or not 1 <= value <= 200000:
            raise ValueError("Bounded positive integer local proposal budgets required")
    if not 3 <= p["local_proposal_neighbors"] <= 128 or p["local_proposal_maximum_count"] > 4096:
        raise ValueError("Bounded local neighborhood and proposal count required")
    return p


def propose(training, config):
    """One deterministic observed seed per training voxel, then local k-NN PCA.

    All occupied training cells are visited, not merely high-density cells or
    points from the dominant member. Budgets fail explicitly instead of silently
    dropping the remaining spatial region. Exact input duplicates/order cannot
    alter the proposal pool; primitive type, labels and validation are absent.
    """
    from scipy.spatial import cKDTree

    p = policy(config)
    training = np.asarray(training, float)
    if training.ndim != 2 or training.shape[1:] != (3,) or not np.isfinite(training).all():
        raise ValueError("Finite Nx3 training points required")
    voxel, origin = config["voxel_size"], np.asarray(config["origin"], float)
    ratio, radius = config["maximum_second_eigen_ratio"], config["ridge_radius_voxels"]*voxel
    if (not np.isfinite(voxel) or voxel <= 0 or origin.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(radius) or radius <= 0 or not 0 < ratio < 1):
        raise ValueError("Finite geometric proposal scales required")
    training = np.unique(training, axis=0)
    diagnostic = dict(state="complete", training_only=True, deterministic=True,
        training_points=len(training), seed_count=0, proposed=0, rejected_degenerate=0,
        rejected_nonlinear=0, duplicate_neighborhoods=0,
        scope="Local training directions only; no acceptance, identity or branch completeness claim")
    if len(training) < p["local_proposal_neighbors"]:
        return [], {**diagnostic, "reason": "insufficient_training_neighbors"}
    coordinates = (training-origin)/voxel
    if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**52):
        raise ValueError("Local proposal coordinates exceed exact grid precision")
    _, seeds = np.unique(np.floor(coordinates).astype(np.int64), axis=0, return_index=True)
    diagnostic["seed_count"] = len(seeds)
    if len(seeds) > p["local_proposal_maximum_seeds"]:
        return [], {**diagnostic, "state": "unmeasurable", "reason": "local_proposal_seed_budget_exceeded"}
    tree = cKDTree(training)
    _, neighbors = tree.query(training[seeds], k=p["local_proposal_neighbors"], workers=1)
    output, seen = [], set()
    for indices in neighbors:
        key = tuple(sorted(indices.tolist()))
        if key in seen:
            diagnostic["duplicate_neighborhoods"] += 1
            continue
        seen.add(key)
        cloud = training[list(key)]
        anchor, eigen, _, axis = principal_axes(cloud)
        if eigen[-1] <= 1e-20 or np.ptp((cloud-anchor) @ axis) < 2*radius:
            diagnostic["rejected_degenerate"] += 1
            continue
        if eigen[-2]/eigen[-1] > ratio:
            diagnostic["rejected_nonlinear"] += 1
            continue
        output.append((anchor, axis))
        if len(output) > p["local_proposal_maximum_count"]:
            return [], {**diagnostic, "state": "unmeasurable", "proposed": len(output),
                        "reason": "local_proposal_count_budget_exceeded"}
    diagnostic["proposed"] = len(output)
    return output, diagnostic
