"""New geometric challenges and explicit non-identifiability pairs, before reader replay.

Labels describe the generating construction, not information available to either
reader. Identical point arrays with contradictory semantic interpretations are
kept deliberately: pure geometry cannot know which story produced those points.
"""
from __future__ import annotations

import numpy as np

FAMILIES = (
    "full_member", "half_member", "gapped_member", "member_local_clutter",
    "member_near_plane", "crossing_members", "side_branch_members", "elliptic_member",
    "long_planar_strip", "bent_wide_ribbon", "corrugated_sheet", "isotropic_clutter",
    "line_as_member", "line_as_wall_seam", "arc_as_member", "arc_as_open_trough",
    "linear_points_as_clutter", "linear_points_as_members",
)


def rotation_xyz(degrees):
    x, y, z = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def member(segment, radius, *, noise, seed, angular_range=(0., 2 * np.pi), elliptic_ratio=1.):
    segment = np.asarray(segment, float)
    axis = segment[1] - segment[0]
    axis /= np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(*angular_range, 29, endpoint=False)
    centers = segment[0] + np.linspace(0, 1, 213)[:, None] * (segment[1] - segment[0])
    ring = radius * (np.cos(angles)[:, None] * u + elliptic_ratio * np.sin(angles)[:, None] * v)
    points = (centers[:, None] + ring).reshape(-1, 3)
    return points + np.random.default_rng(seed).normal(0, noise, points.shape)


def panel(length, width, *, kind="flat"):
    z, x = np.meshgrid(np.linspace(0, length, 213), np.linspace(-width / 2, width / 2, 43), indexing="ij")
    y = np.zeros_like(x)
    if kind == "bent":
        y = .17 * np.sin(np.pi * z / length)
    elif kind == "corrugated":
        y = .035 * np.sin(2 * np.pi * x / .12)
    elif kind != "flat":
        raise ValueError("Unknown panel construction")
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def local_cases(config):
    length, radius, noise, seed = (config[k] for k in ("length_m", "radius_m", "noise_sigma_m", "seed"))
    axis = np.array([[0., 0., 0.], [0., 0., length]])
    whole = member(axis, radius, noise=noise, seed=seed)
    half = member(axis, radius, noise=noise, seed=seed + 1, angular_range=(-np.pi / 2, np.pi / 2))
    gap = np.array([[[0., 0., 0.], [0., 0., .43 * length]], [[0., 0., .64 * length], [0., 0., length]]])
    gapped = np.concatenate([member(s, radius, noise=noise, seed=seed + 2 + i) for i, s in enumerate(gap)])
    random = np.random.default_rng(seed + 4)
    clutter = random.uniform([-.14, -.13, .29 * length], [.14, .13, .54 * length], (1700, 3))
    wall = panel(length, .28) + [.17, .10, 0.]
    crossing = np.array([[-.34, 0., .25 * length], [.34, 0., .83 * length]])
    side = np.array([[0., 0., .55 * length], [.33, 0., .55 * length]])
    ellipse = member(axis, .065, noise=noise, seed=seed + 8, elliptic_ratio=.43)
    flat = panel(length, .24)
    bent = panel(length, .24, kind="bent")
    corrugated = panel(length, .28, kind="corrugated")
    isotropic = np.random.default_rng(seed + 9).uniform(-.16, .16, (4500, 3)) + [0., 0., .5 * length]
    # 完全一样的点，也可能是杆，也可能只是墙缝。别在算法输入里悄悄塞答案。
    line = np.c_[np.zeros(257), np.zeros(257), np.linspace(0, length, 257)]
    arc = member(axis, .054, noise=noise, seed=seed + 10, angular_range=(-.60 * np.pi, .23 * np.pi))
    linear = np.array([[[radius, 0., .37 * length], [.27, .04, .46 * length]],
                       [[-.04, radius, .68 * length], [.02, .25, .74 * length]]])
    local_lines = np.concatenate([s[0] + np.linspace(0, 1, 71)[:, None] * (s[1] - s[0]) for s in linear])
    contaminated = np.concatenate([whole, local_lines])
    positive = {
        "full_member": (whole, axis[None], None),
        "half_member": (half, axis[None], None),
        "gapped_member": (gapped, gap, np.array([gap[0, 1], gap[1, 0]])),
        "member_local_clutter": (np.concatenate([whole, clutter]), axis[None], None),
        "member_near_plane": (np.concatenate([whole, wall]), axis[None], None),
        "crossing_members": (np.concatenate([whole, member(crossing, .019, noise=noise, seed=seed + 5)]), np.array([axis, crossing]), None),
        "side_branch_members": (np.concatenate([whole, member(side, .016, noise=noise, seed=seed + 6)]), np.array([axis, side]), None),
        "elliptic_member": (ellipse, axis[None], None),
    }
    negative = {"long_planar_strip": flat, "bent_wide_ribbon": bent, "corrugated_sheet": corrugated, "isotropic_clutter": isotropic}
    ambiguous = {
        "line_as_member": (line, axis[None], "line_identity", "member"),
        "line_as_wall_seam": (line, np.empty((0, 2, 3)), "line_identity", "background"),
        "arc_as_member": (arc, axis[None], "open_arc_identity", "member"),
        "arc_as_open_trough": (arc, np.empty((0, 2, 3)), "open_arc_identity", "background"),
        "linear_points_as_clutter": (contaminated, axis[None], "linear_clutter_identity", "clutter"),
        "linear_points_as_members": (contaminated, np.concatenate([axis[None], linear]), "linear_clutter_identity", "members"),
    }
    for family in FAMILIES:
        if family in positive:
            points, truth, missing = positive[family]
            labels = dict(expectation="positive", interpretation="declared_member_axes", ambiguity_pair=None)
        elif family in negative:
            points, truth, missing = negative[family], np.empty((0, 2, 3)), None
            labels = dict(expectation="negative", interpretation="complete_background_surface_or_volume", ambiguity_pair=None)
        else:
            points, truth, pair, interpretation = ambiguous[family]
            missing = None
            labels = dict(expectation="unidentifiable", interpretation=interpretation, ambiguity_pair=pair)
        yield family, np.asarray(points, dtype="<f8"), np.asarray(truth, dtype="<f8"), missing, labels


def generate(config):
    """Two frozen transforms/resolutions; these repeated conditions are not objects."""
    for index, placement in enumerate(config["placements"]):
        rotation, origin = rotation_xyz(placement["euler_degrees"]), np.array(placement["origin"])
        for family, points, truth, gap, labels in local_cases(config):
            if labels["ambiguity_pair"] is not None:
                labels["ambiguity_pair"] += f"-placement-{index}"
            labels.update(family=family, placement=index, voxel_size=placement["voxel_m"],
                semantics_scope="construction-conditioned geometry targets; not observable physical identity")
            yield (points @ rotation.T + origin, truth @ rotation.T + origin,
                   None if gap is None else gap @ rotation.T + origin, labels)
