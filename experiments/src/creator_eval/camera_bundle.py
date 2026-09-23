"""Small RGB-track bundle adjustment with explicit gauge and held-out observations."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from creator_eval.background_correspondences import residual_summary


def validate_cameras(intrinsics, extrinsics):
    k, e = np.asarray(intrinsics, float), np.asarray(extrinsics, float)
    if k.ndim != 3 or k.shape[1:] != (3, 3) or e.shape != (len(k), 3, 4) or len(k) < 3:
        raise ValueError("At least three matching camera matrices required")
    if not np.isfinite(k).all() or not np.isfinite(e).all() or np.any(k[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Finite cameras and positive focal lengths required")
    if not np.allclose(k[:, 2], [0, 0, 1]) or not np.allclose(e[:, :, :3] @ e[:, :, :3].transpose(0, 2, 1), np.eye(3), atol=1e-4):
        raise ValueError("Expected pinhole K and rigid world-to-camera rotations")
    if not np.allclose(np.linalg.det(e[:, :, :3]), 1, atol=1e-4):
        raise ValueError("Camera rotations must be proper")
    return k, e


def centers(extrinsics):
    e = np.asarray(extrinsics)
    return -np.einsum("nij,ni->nj", e[:, :, :3], e[:, :, 3])


def project(points, intrinsic, extrinsic):
    xyz = np.asarray(points) @ extrinsic[:, :3].T + extrinsic[:, 3]
    q = xyz @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[..., :2] / q[..., 2, None], xyz[..., 2]


def triangulate(observations, intrinsics, extrinsics):
    rows = []
    if len(observations) < 2 or len({o["view"] for o in observations}) != len(observations):
        raise ValueError("Distinct views required for triangulation")
    for observation in observations:
        view = observation["view"]
        xy = np.asarray(observation["xy"], float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not 0 <= view < len(intrinsics):
            raise ValueError("Invalid image observation")
        ray = np.linalg.solve(intrinsics[view], np.r_[xy, 1.])
        p = extrinsics[view]
        rows.extend((ray[0] * p[2] - p[0], ray[1] * p[2] - p[1]))
    _, _, vt = np.linalg.svd(rows)
    point = vt[-1]
    return point[:3] / point[3] if abs(point[3]) > 1e-12 else np.full(3, np.nan)


def observations_from_record(record, label):
    return [dict(track_id=i, observations=[dict(view=v, xy=record["feature_xy"][v][f]) for v, f in track])
            for i, track in enumerate(record["graph"]["tracks"]) if record["band_labels"][i] == label]


def corrupt_training(tracks, fraction, seed):
    import copy
    result = copy.deepcopy(tracks)
    if not 0 <= fraction <= 1:
        raise ValueError("Corruption fraction must lie in [0,1]")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(result))[:int(np.floor(fraction * len(result)))]
    changes = []
    for index in selected:
        observation = result[index]["observations"][int(rng.integers(len(result[index]["observations"])))]
        candidates = [(j, o) for j, track in enumerate(tracks) if j != index for o in track["observations"]
                      if o["view"] == observation["view"] and np.linalg.norm(np.asarray(o["xy"]) - observation["xy"]) >= 20]
        if not candidates:
            continue
        source, replacement = candidates[int(rng.integers(len(candidates)))]
        changes.append(dict(track_id=result[index]["track_id"], view=observation["view"], original_xy=observation["xy"],
                            replacement_xy=replacement["xy"], donor_track_id=tracks[source]["track_id"]))
        observation["xy"] = copy.deepcopy(replacement["xy"])
    return result, changes


class CameraGauge:
    """Fix first pose and first-to-last distance; no GT or soft scale penalty."""
    def __init__(self, intrinsics, extrinsics, focal):
        self.k, e = validate_cameras(intrinsics, extrinsics)
        camera_centers = centers(e)
        self.origin = camera_centers[0]
        self.scale = float(np.linalg.norm(camera_centers[-1] - self.origin))
        if self.scale < 1e-8:
            raise ValueError("First/last camera baseline is degenerate")
        self.centers = (camera_centers - self.origin) / self.scale
        self.rotations = e[:, :, :3].copy()
        self.focal = focal
        direction = self.centers[-1]
        basis = np.eye(3)[np.argmin(abs(direction))]
        self.u = np.cross(direction, basis)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(direction, self.u)
        self.slices, count = {}, 0
        for i in range(1, len(e)):
            size = 5 if i == len(e) - 1 else 6
            self.slices[i] = slice(count, count + size)
            count += size
        self.focal_index = count if focal else None
        self.parameter_count = count + int(focal)

    def decode(self, parameters, native=False):
        from scipy.spatial.transform import Rotation
        k, rotations, locations = self.k.copy(), self.rotations.copy(), self.centers.copy()
        for view, block in self.slices.items():
            delta = parameters[block]
            rotations[view] = Rotation.from_rotvec(delta[:3]).as_matrix() @ self.rotations[view]
            if view == len(k) - 1:
                direction = self.centers[-1] + delta[3] * self.u + delta[4] * self.v
                locations[view] = direction / np.linalg.norm(direction)
            else:
                locations[view] += delta[3:]
        if self.focal:
            factor = np.exp(parameters[self.focal_index])
            k[:, 0, 0] *= factor
            k[:, 1, 1] *= factor
        if native:
            locations = self.origin + self.scale * locations
        translation = -np.einsum("nij,nj->ni", rotations, locations)
        return k, np.concatenate([rotations, translation[:, :, None]], axis=2)


def optimize_cameras(intrinsics, extrinsics, training, config, *, focal=False, loss="soft_l1"):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    gauge = CameraGauge(intrinsics, extrinsics, focal)
    k, e = gauge.decode(np.zeros(gauge.parameter_count))
    points, kept, discarded = [], [], []
    for track in training:
        point = triangulate(track["observations"], k, e)
        positive = np.isfinite(point).all() and all(project(point[None], k[o["view"]], e[o["view"]])[1][0] > 1e-6 for o in track["observations"])
        if positive:
            points.append(point)
            kept.append(track)
        else:
            discarded.append(track["track_id"])
    if len(points) < config["minimum_initial_tracks"]:
        return dict(state="insufficient_initial_structure", discarded_track_ids=discarded, kept_track_count=len(points))
    obs = [(j, o["view"], o["xy"]) for j, track in enumerate(kept) for o in track["observations"]]
    pi, vi = np.array([o[0] for o in obs]), np.array([o[1] for o in obs])
    xy = np.array([o[2] for o in obs], float)
    x0 = np.r_[np.zeros(gauge.parameter_count), np.asarray(points).ravel()]
    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    for block in gauge.slices.values():
        lower[block.start:block.start + 3], upper[block.start:block.start + 3] = -config["rotation_bound_rad"], config["rotation_bound_rad"]
        lower[block.start + 3:block.stop], upper[block.start + 3:block.stop] = -config["center_bound_baseline_fraction"], config["center_bound_baseline_fraction"]
    if focal:
        lower[gauge.focal_index], upper[gauge.focal_index] = np.log(config["focal_scale_bounds"])
    sparsity = lil_matrix((3 * len(obs), len(x0)), dtype=int)
    for i, (point_index, view, _) in enumerate(obs):
        rows = slice(3 * i, 3 * i + 3)
        if view:
            sparsity[rows, gauge.slices[view]] = 1
        if focal:
            sparsity[rows, gauge.focal_index] = 1
        start = gauge.parameter_count + 3 * point_index
        sparsity[rows, start:start + 3] = 1

    def residual(parameters):
        current_k, current_e = gauge.decode(parameters)
        world = parameters[gauge.parameter_count:].reshape(-1, 3)[pi]
        camera = np.einsum("nij,nj->ni", current_e[vi, :, :3], world) + current_e[vi, :, 3]
        projected = np.einsum("nij,nj->ni", current_k[vi], camera)
        # Behind-camera points must not silently look like valid 2D fits. Keep a
        # barrier in the same sparse residual, and report final cheirality too.
        uv = projected[:, :2] / np.maximum(camera[:, 2, None], 1e-6)
        penalty = np.maximum(1e-5 - camera[:, 2], 0) * 1000
        return np.c_[uv - xy, penalty].ravel()

    initial_residual = residual(x0).reshape(-1, 3)
    solution = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
        loss=loss, f_scale=config["robust_scale_px"], x_scale="jac", max_nfev=config["max_nfev"],
        ftol=config["tolerance"], xtol=config["tolerance"], gtol=config["tolerance"])
    native_k, native_e = gauge.decode(solution.x, native=True)
    final_residual = residual(solution.x).reshape(-1, 3)
    return dict(state="converged" if solution.success else "iteration_limit_or_failure", success=bool(solution.success),
        solver_status=int(solution.status), message=solution.message, nfev=int(solution.nfev), optimality=float(solution.optimality),
        cost=float(solution.cost), intrinsics=native_k, extrinsics=native_e,
        points=gauge.origin + gauge.scale * solution.x[gauge.parameter_count:].reshape(-1, 3),
        kept_track_ids=[t["track_id"] for t in kept], discarded_track_ids=discarded, observation_count=len(obs),
        gauge=dict(first_pose_fixed=True, first_last_distance=gauge.scale, exact_scale_constraint=True),
        focal_scale=float(np.exp(solution.x[gauge.focal_index])) if focal else 1.,
        at_parameter_bound=bool(np.any(solution.active_mask[:gauge.parameter_count])),
        negative_depth_observations=int(np.sum(final_residual[:, 2] > 0)),
        train_before=residual_summary(np.linalg.norm(initial_residual[:, :2], axis=1)),
        train_after=residual_summary(np.linalg.norm(final_residual[:, :2], axis=1)))


def validation_plan(tracks, initial_extrinsics):
    c = centers(initial_extrinsics)
    plans = []
    for track in tracks:
        observations = track["observations"]
        if len(observations) < 3:
            raise ValueError("Validation needs two triangulation views and an unused view")
        pair = max(combinations(range(len(observations)), 2), key=lambda ij: np.linalg.norm(c[observations[ij[0]]["view"]] - c[observations[ij[1]]["view"]]))
        plans.append(dict(track_id=track["track_id"], triangulation=[observations[i] for i in pair],
                          scoring=[o for i, o in enumerate(observations) if i not in pair]))
    return plans


def validate_heldout(intrinsics, extrinsics, plans):
    k, e = validate_cameras(intrinsics, extrinsics)
    errors, rows = [], []
    for plan in plans:
        point = triangulate(plan["triangulation"], k, e)
        front = np.isfinite(point).all() and all(project(point[None], k[o["view"]], e[o["view"]])[1][0] > 0 for o in plan["triangulation"])
        for observation in plan["scoring"]:
            view = observation["view"]
            uv, depth = project(point[None], k[view], e[view])
            error = float(np.linalg.norm(uv[0] - observation["xy"])) if front and depth[0] > 0 and np.isfinite(uv).all() else np.nan
            errors.append(error)
            rows.append(dict(track_id=plan["track_id"], view=view, error_px=error if np.isfinite(error) else None))
    return dict(summary=residual_summary(errors), rows=rows,
                scope="Track coordinates never enter BA; triangulate two predetermined views, score other observations without refitting")


def correction_decision(result, before, after, policy):
    a, b = before["summary"], after["summary"]
    reasons = []
    if not result.get("success", False):
        reasons.append("optimizer_not_converged")
    if result.get("at_parameter_bound", True) or result.get("negative_depth_observations", 1):
        reasons.append("parameter_bound_or_negative_depth")
    if b["sample_count"] < policy["minimum_scored_observations"] or b["finite_count"] != b["sample_count"]:
        reasons.append("insufficient_finite_heldout_support")
    if a["median_px"] is None or b["median_px"] is None or b["median_px"] > a["median_px"] * policy["maximum_median_ratio"]:
        reasons.append("no_required_heldout_improvement")
    if b["p95_px"] is None or b["p95_px"] > policy["maximum_p95_px"] or (b["fraction_within_2px"] or 0) < policy["minimum_fraction_within_2px"]:
        reasons.append("heldout_absolute_error_too_large")
    return dict(state="candidate_camera_correction" if not reasons else "withhold_correction", reasons=reasons,
                scope="RGB-only development gate; no dense snapshot or rod patch is automatically changed")
