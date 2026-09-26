"""Conditional cross-section predictions; this module never accepts a rod or emits an axis.

The shared PCA frame defines six axial slices. Shape parameters use only alternate
slices, and predictions are evaluated on the other three. Both folds are retained.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

DEFAULTS = dict(axial_slices=6, axial_quantiles=[.05, .95], maximum_nfev=80,
    ellipse_seed_angles=64, ellipse_refine_steps=12, minimum_radius_voxels=.25,
    maximum_radius_span_fraction=.25, minimum_fit_points=12)


def ellipse_distances(points, parameters, seeds=64, steps=12):
    center, radii, angle = parameters[:2], np.exp(parameters[2:4]), parameters[4]
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    values = (points - center) @ rotation
    theta = np.linspace(0., 2*np.pi, seeds, endpoint=False)
    curve = np.c_[radii[0]*np.cos(theta), radii[1]*np.sin(theta)]
    nearest = cKDTree(curve).query(values)[1]
    theta = theta[nearest]
    # 先找附近的弧，再细化。只做固定次数，不看新测试结果改精度。
    for _ in range(steps):
        cosine, sine = np.cos(theta), np.sin(theta)
        x = radii[0]*cosine - values[:, 0]
        y = radii[1]*sine - values[:, 1]
        first = -radii[0]*sine*x + radii[1]*cosine*y
        second = radii[0]**2*sine**2 + radii[1]**2*cosine**2 - radii[0]*cosine*x - radii[1]*sine*y
        update = np.divide(first, second, out=np.zeros_like(first), where=np.abs(second)>1e-12)
        theta -= np.clip(update, -.2, .2)
    closest = np.c_[radii[0]*np.cos(theta), radii[1]*np.sin(theta)]
    distance = np.linalg.norm(values-closest, axis=1)
    sign = np.where(np.sum((values/radii)**2, axis=1) >= 1., 1., -1.)
    return distance*sign


def two_circle_distances(points, parameters):
    first = np.linalg.norm(points-parameters[:2], axis=1)-np.exp(parameters[2])
    second = np.linalg.norm(points-parameters[3:5], axis=1)-np.exp(parameters[5])
    return np.where(np.abs(first) <= np.abs(second), first, second)


def algebraic_circle(points):
    design = np.c_[2*points, np.ones(len(points))]
    value, _, rank, _ = np.linalg.lstsq(design, np.sum(points**2, axis=1), rcond=None)
    radius2 = value[2] + value[:2]@value[:2]
    return None if rank < 3 or radius2 <= 0 else np.r_[value[:2], .5*np.log(radius2)]


def ellipse_initial(points):
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T, bias=True))
    radii = np.sqrt(2*np.maximum(values, 1e-8))
    return np.r_[center, np.log(radii), np.arctan2(vectors[1, 0], vectors[0, 0])]


def conic_initial(points):
    mean = points.mean(axis=0)
    scale = max(float(np.std(points)), 1e-8)
    x, y = ((points-mean)/scale).T
    design = np.c_[x*x, x*y, y*y, x, y]
    value, _, rank, _ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)
    matrix = np.array([[value[0], value[1]/2], [value[1]/2, value[2]]])
    eigen, vectors = np.linalg.eigh(matrix)
    if rank < 5 or eigen.min() <= 1e-10:
        return None
    center = -.5*np.linalg.solve(matrix, value[3:])
    factor = 1.+center@matrix@center
    if factor <= 0:
        return None
    radii = np.sqrt(factor/eigen)*scale
    return np.r_[mean+center*scale, np.log(radii), np.arctan2(vectors[1, 0], vectors[0, 0])]


def summarize(distance, weights):
    distance, weights = np.abs(distance), np.asarray(weights)
    ordered = np.argsort(distance, kind="stable")
    cumulative = np.cumsum(weights[ordered])/weights.sum()
    quantiles = [float(distance[ordered[min(np.searchsorted(cumulative, q), len(ordered)-1)]]) for q in (.5, .9, .95)]
    return dict(point_count=len(distance), occupied_weight=float(weights.sum()),
        rmse_voxels=float(np.sqrt(np.average(distance**2, weights=weights))),
        median_voxels=quantiles[0], p90_voxels=quantiles[1], p95_voxels=quantiles[2])


def fit_model(model, points, weights, maximum_radius, policy):
    if len(points) < policy["minimum_fit_points"]:
        return dict(state="unavailable", reason="insufficient_training_points")
    if model == "line":
        center = np.average(points, axis=0, weights=weights)
        covariance = ((points-center)*weights[:, None]).T@(points-center)/weights.sum()
        eigen, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        return dict(state="fitted", parameters=np.r_[center, normal].tolist(), converged=True,
                    at_parameter_boundary=False, starts=1, evaluations=0, parameter_count=2, numerical_jacobian_rank=None, covariance_eigenvalues=eigen.tolist())
    low, high = points.min(axis=0)-maximum_radius, points.max(axis=0)+maximum_radius
    log_min, log_max = np.log(policy["minimum_radius_voxels"]), np.log(maximum_radius)
    if log_max <= log_min:
        return dict(state="unavailable", reason="insufficient_axial_span")
    if model == "ellipse":
        initial = [ellipse_initial(points)]
        algebraic = conic_initial(points)
        if algebraic is not None:
            initial.append(algebraic)
        lower, upper = np.r_[low, log_min, log_min, -np.inf], np.r_[high, log_max, log_max, np.inf]
        def residual(parameters):
            return ellipse_distances(points, parameters, policy["ellipse_seed_angles"], policy["ellipse_refine_steps"])*np.sqrt(weights)
    elif model == "two_circles":
        eigen, vectors = np.linalg.eigh(np.cov(points.T, bias=True))
        axis = vectors[:, -1]
        projection = points@axis
        initial = []
        for split in (.4, .5, .6):
            keep = projection <= np.quantile(projection, split)
            circles = [algebraic_circle(part) for part in (points[keep], points[~keep])]
            if all(circle is not None for circle in circles):
                initial.append(np.concatenate(circles))
        if not initial:
            center = points.mean(axis=0)
            radius = max(np.sqrt(max(eigen[-1], 0.))/2, policy["minimum_radius_voxels"])
            initial.append(np.r_[center-axis*radius, np.log(radius), center+axis*radius, np.log(radius)])
        lower, upper = np.tile(np.r_[low, log_min], 2), np.tile(np.r_[high, log_max], 2)
        def residual(parameters):
            return two_circle_distances(points, parameters)*np.sqrt(weights)
    else:
        raise ValueError("Unknown diagnostic model")
    fits = []
    for start in initial:
        start = np.clip(start, lower+1e-10, upper-1e-10)
        fitted = least_squares(residual, start, bounds=(lower, upper), max_nfev=policy["maximum_nfev"],
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        fits.append(fitted)
    best = min(fits, key=lambda fit: float(np.sum(fit.fun**2)))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    cutoff = np.finfo(float).eps*max(best.jac.shape)*singular[0] if len(singular) else 0.
    starts = [dict(converged=bool(f.success), at_parameter_boundary=bool(np.any(f.active_mask)),
                   evaluations=f.nfev, squared_loss=float(np.sum(f.fun**2)), parameters=f.x.tolist()) for f in fits]
    return dict(state="fitted", parameters=best.x.tolist(), converged=bool(best.success),
        at_parameter_boundary=bool(np.any(best.active_mask)), starts=len(initial),
        evaluations=sum(fit.nfev for fit in fits), training_squared_loss=float(np.sum(best.fun**2)),
        jacobian_singular_values=singular.tolist(), numerical_jacobian_rank=int(np.count_nonzero(singular>cutoff)),
        parameter_count=len(best.x), starts_diagnostics=starts)


def distances(model, points, fitted, policy):
    parameters = np.asarray(fitted["parameters"])
    if model == "line":
        return (points-parameters[:2])@parameters[2:]
    if model == "ellipse":
        return ellipse_distances(points, parameters, policy["ellipse_seed_angles"], policy["ellipse_refine_steps"])
    return two_circle_distances(points, parameters)


def diagnose(points, voxel, policy=None):
    p = {**DEFAULTS, **(policy or {})}
    if set(p) != set(DEFAULTS) or p["axial_slices"] != 6:
        raise ValueError("Frozen six-slice diagnostic policy required")
    points = np.unique(np.asarray(points, float), axis=0)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 12 or not np.isfinite(points).all() or voxel <= 0:
        raise ValueError("Finite Nx3 points and positive voxel required")
    cells, inverse, counts = np.unique(np.floor(points/voxel).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    centers = (cells.astype(float)+.5)*voxel
    origin = centers.mean(axis=0)
    _, vectors = np.linalg.eigh((centers-origin).T@(centers-origin)/len(centers))
    if vectors[np.argmax(np.abs(vectors[:, -1])), -1] < 0:
        vectors[:, -1] *= -1
    # 两种表示共用同一坐标框架。这里测的是条件截面预测，不是假装完全留出PCA。
    axial = (centers-origin)@vectors[:, -1]
    edges = np.linspace(*np.quantile(axial, p["axial_quantiles"]), 7)
    maximum_radius = (edges[-1]-edges[0])/voxel*p["maximum_radius_span_fraction"]
    result = dict(state="complete_diagnostic", policy=p, voxel_size=voxel, point_count=len(points),
        occupied_voxels=len(cells), origin=origin.tolist(), frame=vectors.tolist(),
        axial_boundaries=edges.tolist(), representations=[], emits_axis=False,
        scope="Conditional prediction evidence only; no model accepted, no rod identity or probability claim")
    for name, cloud, weights in (("voxel_centers", centers, np.ones(len(centers))),
                                ("equal_voxel_raw_points", points, 1./counts[inverse])):
        values = (cloud-origin)@vectors/voxel
        assignment = np.searchsorted(edges/voxel, values[:, -1], side="right")-1
        assignment[values[:, -1] == edges[-1]/voxel] = 5
        models = []
        for parity in (0, 1):
            train_ids, test_ids = list(range(parity, 6, 2)), list(range(1-parity, 6, 2))
            train = np.isin(assignment, train_ids)
            for model in ("ellipse", "two_circles", "line"):
                try:
                    fitted = fit_model(model, values[train, :2], weights[train], maximum_radius, p)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    fitted = dict(state="numerical_failure", reason=str(error))
                scores = []
                if fitted["state"] == "fitted":
                    fitted["training_residual"] = summarize(distances(model, values[train, :2], fitted, p), weights[train])
                    for slot in test_ids:
                        selected = assignment == slot
                        if selected.any():
                            scores.append(dict(slice=slot, **summarize(distances(model, values[selected, :2], fitted, p), weights[selected])))
                        else:
                            scores.append(dict(slice=slot, state="missing_test_points"))
                models.append(dict(model=model, train_slices=train_ids, test_slices=test_ids, fit=fitted, heldout=scores))
        result["representations"].append(dict(name=name, rows=models))
    return result
