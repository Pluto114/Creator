"""Authored sampling-continuity/side-branch controls, with separate construction truth.

Only ``inputs`` is available to readers. Neither reader implementations nor
runtime data, fixture truth, or scores are imported or inspected. The prior 44
mixed controls and 36 section controls are separate scored development cohorts.
These 40 conditions repeat ten analytic constructions; they are not forty
independent physical objects or a blind real-photograph qualification set.
"""

from __future__ import annotations

import numpy as np

from .mixed_readout_controls import SCOPE
from .readout_background_controls import rotation_xyz

COHORT = "new_sampling_branch_controls_v1"
CASE_IDS = (
    "full_variable_density",
    "half_variable_density",
    "staggered_twin_surfaces",
    "short_true_gap",
    "long_true_gap",
    "shallow_side_branch",
    "steep_side_branch",
    "scatter_only",
    "rotating_sparse_arc",
    "helical_trace",
)
GRID_SCALES_M = (0.005, 0.009)
DENSITIES = ("dense", "coarse")
AXIAL_STEPS_M = {"dense": 0.0065, "coarse": 0.017}
SAMPLING_PHASE = 0.23
SECOND_TWIN_SAMPLING_PHASE = 0.64
DEFAULT_SEED = 431029
LENGTH_M = 1.7
RADIUS_M = 0.021
SMALL_RADIUS_M = 0.0045
TWIN_AXIS_DISTANCE_M = 0.048
SPARSE_INTERVAL_M = (0.42 * LENGTH_M, 0.73 * LENGTH_M)
GAP_LENGTHS_M = {"short_true_gap": 0.09, "long_true_gap": 0.27}
BRANCH_ANGLES_DEGREES = {"shallow_side_branch": 32.0, "steep_side_branch": 68.0}
BRANCH_LENGTHS_M = {"shallow_side_branch": 0.24, "steep_side_branch": 0.16}
ROTATION_DEGREES = (38.0, -17.0, 71.0)
TRANSLATION_M = (-0.23, 0.36, 0.54)
ARC_COVERAGE_DEGREES = 75.0
ARC_SWEEP_TURNS = 1.25
HELIX_TURNS = 1.75


def _axial_samples(density, *, phase=SAMPLING_PHASE, gap=None):
    """Piecewise physical sampling pitches, not a reader-dependent gap policy.

    The middle region has twice the base step. Its transition rings are observed
    explicitly, as are finite endpoints and both boundaries of any true gap.
    """
    step = AXIAL_STEPS_M[density]
    low, high = SPARSE_INTERVAL_M
    zones = ((0.0, low, step), (low, high, 2 * step), (high, LENGTH_M, step))
    parts = []
    for first, last, pitch in zones:
        lattice = (np.arange(int(np.ceil(LENGTH_M / pitch)) + 1) + phase) * pitch
        parts.append(np.r_[first, lattice[(lattice > first) & (lattice < last)], last])
    values = np.unique(np.concatenate(parts))
    if gap is not None:
        first, last = gap
        values = np.unique(np.r_[values[(values <= first) | (values >= last)], first, last])
    return values


def _surface(z, radius, *, half=False, x_offset=0.0):
    angles = np.linspace(-np.pi / 2, np.pi / 2, 19) if half else np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
    ring = np.c_[radius * np.cos(angles) + x_offset, radius * np.sin(angles)]
    return np.c_[np.tile(ring, (len(z), 1)), np.repeat(z, len(angles))]


def make_case(case_id, *, seed=DEFAULT_SEED, density="dense", voxel_size=GRID_SCALES_M[0]):
    """Return fresh unlabelled arrays and separately declared construction truth."""
    if case_id not in CASE_IDS:
        raise ValueError("Unknown sampling/branch control family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if density not in DENSITIES:
        raise ValueError("Choose a predeclared physical sampling density")
    if type(voxel_size) not in (int, float) or voxel_size not in GRID_SCALES_M:
        raise ValueError("Choose a predeclared voxel scale")

    empty = np.empty((0, 2, 3), dtype="<f8")
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH_M]])
    gap = None
    if case_id in GAP_LENGTHS_M:
        half_gap = GAP_LENGTHS_M[case_id] / 2
        gap = (LENGTH_M / 2 - half_gap, LENGTH_M / 2 + half_gap)
    z = _axial_samples(density, gap=gap)
    segments = empty.copy()
    target, construction, forbidden = axis[None].copy(), empty.copy(), empty.copy()
    expected, radius, angular_samples = "line", RADIUS_M, None
    branch_angle, branch_length, second_layers = None, None, None

    if case_id in ("full_variable_density", "half_variable_density", "short_true_gap", "long_true_gap"):
        half = case_id != "full_variable_density"
        points = _surface(z, RADIUS_M, half=half)
        angular_samples = 19 if half else 36
        if gap is not None:
            low, high = gap
            target = np.array([
                [[0.0, 0.0, 0.0], [0.0, 0.0, low]],
                [[0.0, 0.0, high], [0.0, 0.0, LENGTH_M]],
            ])
            forbidden = np.array([[[0.0, 0.0, low], [0.0, 0.0, high]]])
            expected = "gap"
    elif case_id == "staggered_twin_surfaces":
        other_z = _axial_samples(density, phase=SECOND_TWIN_SAMPLING_PHASE)
        offsets = (-TWIN_AXIS_DISTANCE_M / 2, TWIN_AXIS_DISTANCE_M / 2)
        points = np.concatenate([
            _surface(z, SMALL_RADIUS_M, x_offset=offsets[0]),
            _surface(other_z, SMALL_RADIUS_M, x_offset=offsets[1]),
        ])
        target = np.array([axis + [offset, 0.0, 0.0] for offset in offsets])
        expected, radius, angular_samples = "two_lines", SMALL_RADIUS_M, 36
        second_layers = len(other_z)
    elif case_id in BRANCH_ANGLES_DEGREES or case_id == "scatter_only":
        # Both branch families use exactly the negative arm's scatter points.
        # The primitive arrays contain no source/method/foreground identifiers.
        xy = np.random.default_rng(seed).uniform([-0.068, -0.052], [0.068, 0.052], size=(len(z) * 8, 2))
        points = np.c_[xy, np.repeat(z, 8)]
        radius = None
        if case_id == "scatter_only":
            target, expected = empty.copy(), "abstain"
        else:
            branch_angle, branch_length = BRANCH_ANGLES_DEGREES[case_id], BRANCH_LENGTHS_M[case_id]
            angle = np.deg2rad(branch_angle)
            start = np.array([0.0, 0.0, 0.63 * LENGTH_M])
            end = start + branch_length * np.array([np.sin(angle), 0.0, np.cos(angle)])
            segments = np.array([axis, [start, end]])
            target, expected = segments.copy(), "branches"
    else:
        offsets = np.linspace(-ARC_COVERAGE_DEGREES / 2, ARC_COVERAGE_DEGREES / 2, 3) if case_id == "rotating_sparse_arc" else np.array([0.0])
        turns = ARC_SWEEP_TURNS if case_id == "rotating_sparse_arc" else HELIX_TURNS
        angles = np.deg2rad(offsets)[None] + 2 * np.pi * turns * z[:, None] / LENGTH_M
        points = np.c_[RADIUS_M * np.cos(angles).ravel(), RADIUS_M * np.sin(angles).ravel(),
                       np.repeat(z, len(offsets))]
        target, construction, expected = empty.copy(), axis[None].copy(), "unresolved"
        angular_samples = len(offsets)

    rotation = rotation_xyz(ROTATION_DEGREES)
    translation = np.array(TRANSLATION_M)

    def transform(values):
        return np.asarray(values @ rotation.T + translation, dtype="<f8").copy()

    return {
        "cohort": COHORT,
        "case_id": case_id,
        "condition_id": f"{case_id}-grid-{GRID_SCALES_M.index(voxel_size)}-density-{density}",
        "inputs": {"points": transform(points), "segments": transform(segments)},
        "truth": {
            "expected": expected,
            "expected_segments": transform(target),
            "forbidden_segments": transform(forbidden),
            "construction_segments": transform(construction),
            "scope": SCOPE,
            "abstention_is_recovery": False,
            "negative_labels_are_construction_assumptions": True,
            "unresolved_construction_is_not_negative_foreground_truth": expected == "unresolved",
            "branch_completeness_requires_boundary_metrics_not_only_global_recovery": expected == "branches",
        },
        "protocol": {
            "cohort": COHORT,
            "seed": seed,
            "density": density,
            "axial_step_m": AXIAL_STEPS_M[density],
            "axial_step_multipliers": [1.0, 2.0, 1.0],
            "sparse_interval_m": list(SPARSE_INTERVAL_M),
            "sampling_phase": SAMPLING_PHASE,
            "sampling_phase_unit": "fraction_of_local_physical_axial_step",
            "second_surface_sampling_phase": SECOND_TWIN_SAMPLING_PHASE if second_layers else None,
            "voxel_size": float(voxel_size),
            "grid_origin": [0.0, 0.0, 0.0],
            "local_to_world_rotation": rotation.tolist(),
            "local_to_world_translation": translation.tolist(),
            "rotation_degrees": list(ROTATION_DEGREES),
            "length_m": LENGTH_M,
            "radius_m": radius,
            "angular_samples_per_surface": angular_samples,
            "axial_layer_count": len(z),
            "second_surface_axial_layer_count": second_layers,
            "branch_angle_degrees": branch_angle,
            "branch_length_m": branch_length,
            "gap_length_m": GAP_LENGTHS_M.get(case_id),
            "finite_endpoints_and_gap_boundaries_observed": True,
            "sampling_region_boundaries_observed": True,
            "scope": SCOPE,
            "scale_and_density_are_repeated_conditions_not_objects": True,
            "prior_scored_44_and_36_are_separate_development_regression": True,
            "paired_scatter_has_no_foreground_identity_label": True,
        },
    }


def generate(*, seed=DEFAULT_SEED):
    """Yield only the new 10 families × 2 scales × 2 densities (40 conditions)."""
    for case_id in CASE_IDS:
        for voxel_size in GRID_SCALES_M:
            for density in DENSITIES:
                yield make_case(case_id, seed=seed, density=density, voxel_size=voxel_size)
