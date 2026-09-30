"""Fixed analytic controls for a point-cloud/finite-curve common reader.

These constructions are independent of the three fixture images, not independent
physical objects, real photographs, or a semantic rod detector qualification.
Only ``case["inputs"]`` belongs in a reader call. Construction truth and family
names are deliberately separated and must be withheld during inference.

Geometry is fixed in metres across grid scales. The phase moves the entire scene
relative to the same grid origin; it never changes a gap or a surface dimension.
No reader implementation, fixture, runtime artifact, or evaluation data is read.
"""

from __future__ import annotations

import numpy as np

CASE_IDS = (
    "axis_scatter",
    "scatter_only",
    "full_cylinder",
    "half_cylinder",
    "thin_sheet",
    "nearby_lines",
    "true_gap",
    "side_branch",
    "sparse_cylinder",
    "sparse_half_cylinder",
    "rotating_sparse_arc",
)
GRID_SCALES_M = (0.006, 0.010)
SUBVOXEL_PHASES = ((0.15, 0.25, 0.35), (0.65, 0.75, 0.85))
DEFAULT_SEED = 410731
LENGTH_M = 1.8
GAP_LIMITS_M = (0.70, 1.10)
SCOPE = (
    "analytic_construction_conditioned_geometry_only; not_real_photos; "
    "not_independent_physical_objects; not_foreground_identity"
)


def _rotation():
    x, y, z = np.deg2rad([17.0, -23.0, 11.0])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _surface(radius, *, half=False, x_offset=0.0):
    # Uniform surface support is deterministic, with no fitted axis inserted.
    angles = (
        np.linspace(0.0, np.pi, 21)
        if half
        else np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    )
    z = np.linspace(0.0, LENGTH_M, 241)
    ring = np.c_[radius * np.cos(angles) + x_offset, radius * np.sin(angles)]
    return np.c_[np.tile(ring, (len(z), 1)), np.repeat(z, len(angles))]


def make_case(case_id, *, seed=DEFAULT_SEED, phase=0, voxel_size=GRID_SCALES_M[0]):
    """Return a fresh input/truth-separated case without running a reader.

    ``phase`` indexes the two predeclared translations, measured in voxels.
    ``voxel_size`` must select one of the predeclared grid resolutions. Seed
    variation is available for deterministic generator tests, not post-score
    retries; a formal protocol must record its chosen seed before inference.
    """
    if case_id not in CASE_IDS:
        raise ValueError("Unknown mixed-readout control family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(phase) is not int or not 0 <= phase < len(SUBVOXEL_PHASES):
        raise ValueError("Phase must index a predeclared subvoxel translation")
    if type(voxel_size) not in (int, float) or voxel_size not in GRID_SCALES_M:
        raise ValueError("Choose a predeclared voxel scale")

    empty = np.empty((0, 2, 3), dtype="<f8")
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH_M]])
    # Paired cases retain precisely the same scatter points, not fresh noise.
    scatter = np.random.default_rng(seed).uniform(
        [-0.060, -0.050, 0.0], [0.060, 0.050, LENGTH_M], size=(3600, 3)
    )
    segments = empty.copy()
    truth = axis[None].copy()
    forbidden = empty.copy()
    construction = empty.copy()
    expected = "line"

    if case_id == "axis_scatter":
        points, segments = scatter, axis[None].copy()
    elif case_id == "scatter_only":
        points, truth, expected = scatter, empty.copy(), "abstain"
    elif case_id == "full_cylinder":
        points = _surface(0.025)
    elif case_id == "half_cylinder":
        points = _surface(0.025, half=True)
    elif case_id == "thin_sheet":
        # Sparse cross-section but dense axial columns: a line voter must not
        # mistake every densely sampled sheet column for a separate member.
        x, z = np.meshgrid(np.linspace(-0.070, 0.070, 9), np.linspace(0.0, LENGTH_M, 241))
        points = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        truth, expected = empty.copy(), "abstain"
    elif case_id == "nearby_lines":
        offsets = (-0.018, 0.018)
        points = np.concatenate([_surface(0.003, x_offset=x) for x in offsets])
        segments = np.array([axis + [x, 0.0, 0.0] for x in offsets])
        truth, expected = segments.copy(), "two_lines"
    elif case_id == "true_gap":
        low, high = GAP_LIMITS_M
        points = scatter[(scatter[:, 2] <= low) | (scatter[:, 2] >= high)]
        segments = np.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, low]],
             [[0.0, 0.0, high], [0.0, 0.0, LENGTH_M]]]
        )
        truth, expected = segments.copy(), "gap"
        forbidden = np.array([[[0.0, 0.0, low], [0.0, 0.0, high]]])
    elif case_id == "side_branch":
        branch = np.array([[0.0, 0.0, 0.9], [0.36, 0.0, 0.9]])
        points, segments = scatter, np.array([axis, branch])
        truth, expected = segments.copy(), "branches"
    else:
        z = np.linspace(0.0, LENGTH_M, 241)
        if case_id == "sparse_cylinder":
            angles = np.tile(np.linspace(0.0, 2 * np.pi, 4, endpoint=False), (len(z), 1))
        elif case_id == "sparse_half_cylinder":
            angles = np.tile(np.linspace(0.0, np.pi, 4), (len(z), 1))
        else:
            # Across the entire length this projects to a full ring, but a
            # local section sees only four points spanning sixty degrees.
            angles = np.linspace(-np.pi / 6, np.pi / 6, 4)[None] + 2 * np.pi * z[:, None] / LENGTH_M
        points = np.c_[
            0.025 * np.cos(angles).ravel(),
            0.025 * np.sin(angles).ravel(),
            np.repeat(z, 4),
        ]
        construction = axis[None].copy()
        truth, expected = empty.copy(), "unresolved"

    rotation = _rotation()
    translation = np.array([-0.30, 0.20, 0.50]) + np.array(SUBVOXEL_PHASES[phase]) * voxel_size

    def transform(values):
        return np.asarray(values @ rotation.T + translation, dtype="<f8").copy()

    scale_index = GRID_SCALES_M.index(voxel_size)
    return {
        "case_id": case_id,
        "condition_id": f"{case_id}-grid-{scale_index}-phase-{phase}",
        "inputs": {"points": transform(points), "segments": transform(segments)},
        "truth": {
            "expected": expected,
            "expected_segments": transform(truth),
            "forbidden_segments": transform(forbidden),
            "construction_segments": transform(construction),
            "scope": SCOPE,
            "abstention_is_recovery": False,
            "negative_labels_are_construction_assumptions": True,
        },
        "protocol": {
            "seed": seed,
            "phase": phase,
            "phase_voxels": list(SUBVOXEL_PHASES[phase]),
            "voxel_size": float(voxel_size),
            "grid_origin": [0.0, 0.0, 0.0],
            "local_to_world_rotation": rotation.tolist(),
            "local_to_world_translation": translation.tolist(),
            "length_m": LENGTH_M,
            "scope": SCOPE,
            "scale_and_phase_are_repeated_conditions_not_objects": True,
        },
    }


def generate(*, seed=DEFAULT_SEED):
    """Yield the fixed 11 families × 2 grid scales × 2 phases (44 conditions)."""
    for case_id in CASE_IDS:
        for voxel_size in GRID_SCALES_M:
            for phase in range(len(SUBVOXEL_PHASES)):
                yield make_case(case_id, seed=seed, phase=phase, voxel_size=voxel_size)
