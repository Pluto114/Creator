"""Train-only section frames on fixed native voxel blocks; never selects a model."""
from __future__ import annotations

import hashlib

import numpy as np

from .section_model_diagnostic import DEFAULTS as MODEL_DEFAULTS
from .section_model_diagnostic import distances, fit_model, summarize

SPLIT_DEFAULTS = dict(native_axis=2, grid_origin=[0., 0., 0.], block_cells=10,
    blocks=6, guard_cells_each_edge=2, first_cell=0)
DEFAULTS = dict(model=MODEL_DEFAULTS, split=SPLIT_DEFAULTS)


def array_identity(values):
    values = np.ascontiguousarray(values)
    return dict(shape=list(values.shape), dtype=values.dtype.str,
        sha256=hashlib.sha256(values.tobytes()).hexdigest())


def native_groups(points, voxel, split=None):
    policy = SPLIT_DEFAULTS if split is None else split
    if policy != SPLIT_DEFAULTS:
        raise ValueError("Fixed native six-block split required")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all() or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Finite Nx3 observations and positive voxel required")
    points = np.unique(points, axis=0)
    cells = np.floor((points-policy["grid_origin"])/voxel).astype(np.int64)
    coordinate = cells[:, policy["native_axis"]]-policy["first_cell"]
    block = np.floor_divide(coordinate, policy["block_cells"])
    offset = np.mod(coordinate, policy["block_cells"])
    inside = (block >= 0) & (block < policy["blocks"])
    guard = inside & ((offset < policy["guard_cells_each_edge"]) |
                      (offset >= policy["block_cells"]-policy["guard_cells_each_edge"]))
    assignment = np.where(~inside, -2, np.where(guard, -1, block))
    return points, cells, assignment


def representation(points, voxel, name):
    # Each side owns its counts. Held-out density cannot change fit weights.
    cells, inverse, counts = np.unique(np.floor(points/voxel).astype(np.int64),
        axis=0, return_inverse=True, return_counts=True)
    if name == "voxel_centers":
        cloud, weights = (cells.astype(float)+.5)*voxel, np.ones(len(cells))
    elif name == "equal_voxel_raw_points":
        cloud, weights = points, 1./counts[inverse]
    else:
        raise ValueError("Unknown section representation")
    return cloud, weights, cells


def prepare_fold(points, voxel, parity, policy=None):
    p = DEFAULTS if policy is None else policy
    if set(p) != {"model", "split"} or set(p["model"]) != set(MODEL_DEFAULTS) or parity not in (0, 1):
        raise ValueError("Explicit two-fold method policy required")
    points, cells, assignment = native_groups(points, voxel, p["split"])
    train_ids, test_ids = list(range(parity, 6, 2)), list(range(1-parity, 6, 2))
    train = np.isin(assignment, train_ids)
    test = np.isin(assignment, test_ids)
    train_cells, test_cells = np.unique(cells[train], axis=0), np.unique(cells[test], axis=0)
    frame = dict(state="unavailable", reason="insufficient_training_voxels")
    if len(train_cells) >= p["model"]["minimum_fit_points"]:
        centers = (train_cells.astype(float)+.5)*voxel
        origin = centers.mean(axis=0)
        eigen, vectors = np.linalg.eigh((centers-origin).T@(centers-origin)/len(centers))
        # Signs and radius search come from training observations alone.
        for axis in range(3):
            if vectors[np.argmax(np.abs(vectors[:, axis])), axis] < 0:
                vectors[:, axis] *= -1
        axial = (centers-origin)@vectors[:, -1]
        low, high = np.quantile(axial, p["model"]["axial_quantiles"])
        maximum_radius = (high-low)/voxel*p["model"]["maximum_radius_span_fraction"]
        frame = dict(state="available", origin=origin.tolist(), vectors=vectors.tolist(),
            covariance_eigenvalues=eigen.tolist(), axial_quantiles_m=[float(low), float(high)],
            maximum_radius_voxels=float(maximum_radius), source="training_occupied_voxel_centers_only")
    roles = {"train": train, "heldout": test, "guard": assignment == -1, "outside_window": assignment == -2}
    diagnostics = dict(fold=parity, train_blocks=train_ids, heldout_blocks=test_ids,
        role_points={k:int(v.sum()) for k,v in roles.items()},
        role_cells={k:len(np.unique(cells[v], axis=0)) for k,v in roles.items()},
        train_cell_identity=array_identity(train_cells), heldout_cell_identity=array_identity(test_cells),
        train_point_identity=array_identity(points[train]), heldout_point_identity=array_identity(points[test]),
        nominal_minimum_native_z_gap_voxels=4., frame=frame,
        frame_shared_between_representations=True, frame_shared_between_folds=False,
        weighting="Within-side occupied-cell inverse population; heldout weights only score predictions")
    return points, assignment, diagnostics


def diagnose(points, voxel, policy=None):
    p = DEFAULTS if policy is None else policy
    result = dict(state="complete_diagnostic", policy=p, voxel_size=voxel,
        emits_axis=False, qualification=None, folds=[], representations=[],
        scope="Development replay; native grouping and fitted preprocessing exclude heldout observations")
    prepared = [prepare_fold(points, voxel, parity, p) for parity in (0, 1)]
    result["folds"] = [item[2] for item in prepared]
    for name in ("voxel_centers", "equal_voxel_raw_points"):
        rows = []
        for cloud, assignment, fold in prepared:
            train_ids, test_ids = fold["train_blocks"], fold["heldout_blocks"]
            training, weights, _ = representation(cloud[np.isin(assignment, train_ids)], voxel, name)
            frame = fold["frame"]
            if frame["state"] == "available":
                origin, vectors = np.array(frame["origin"]), np.array(frame["vectors"])
                values = (training-origin)@vectors/voxel
            for model in ("ellipse", "two_circles", "line"):
                fitted = dict(state="unavailable", reason=frame.get("reason"))
                if frame["state"] == "available":
                    try:
                        fitted = fit_model(model, values[:, :2], weights, frame["maximum_radius_voxels"], p["model"])
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                        fitted = dict(state="numerical_failure", reason=str(error))
                    if fitted["state"] == "fitted":
                        fitted["training_residual"] = summarize(distances(model, values[:, :2], fitted, p["model"]), weights)
                heldout = []
                for block in test_ids:
                    testing, test_weights, _ = representation(cloud[assignment == block], voxel, name)
                    if not len(testing):
                        score = dict(state="missing_test_points")
                    elif fitted["state"] != "fitted":
                        score = dict(state="not_scored_missing_fit", point_count=len(testing))
                    else:
                        values_test = (testing-origin)@vectors/voxel
                        score = summarize(distances(model, values_test[:, :2], fitted, p["model"]), test_weights)
                    heldout.append(dict(slice=block, **score))
                rows.append(dict(model=model, fold=fold["fold"], train_slices=train_ids, test_slices=test_ids,
                    fit=fitted, heldout=heldout, train_representation_identity=array_identity(training),
                    training_weights_identity=array_identity(weights), training_occupied_weight=float(weights.sum())))
        result["representations"].append(dict(name=name, rows=rows))
    return result
