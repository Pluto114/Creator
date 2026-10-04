"""Independent authored local-evidence controls; normal inputs contain no labels.

The 32 conditions are repeated analytical constructions, not new photographs or
independent physical objects. Only ``inputs`` may be passed to a reader. This
module does not inspect a reader, runtime artifact, fixture, or evaluation.

Bare feature lines are deliberate positives, including three noncoplanar lines
and a short real branch. The sparse plane's negative role is an authored surface
assumption, not proof that point geometry distinguishes every sampled sheet from
a coplanar set of real rods. Incomplete curved observations remain unresolved.
"""

from __future__ import annotations

import numpy as np

from .mixed_readout_controls import SCOPE
from .readout_background_controls import rotation_xyz

COHORT = "new_local_evidence_controls_v1"
CASE_IDS = (
    "full_shell",
    "half_shell",
    "true_gap_half",
    "twin_bare_lines",
    "noncoplanar_rods_with_branch",
    "rotating_sparse_arc",
    "short_pitch_helix",
    "jittered_sparse_plane",
)
GRID_SCALES_M = (0.0045, 0.0075)
PHASES = (0, 1)
SAMPLING_PHASES = (0.17, 0.73)
DENSITY = "fixed_13mm"
AXIAL_STEP_M = 0.013
DEFAULT_SEED = 441107
LENGTH_M = 1.73
RADIUS_M = 0.024
RADIAL_NOISE_BOUND_M = 0.00018
LINE_TRANSVERSE_NOISE_BOUND_M = 0.00009
PLANE_TRANSVERSE_NOISE_BOUND_M = 0.00016
PLANE_AXIAL_NOISE_BOUND_M = 0.00045
PLANE_WIDTH_M = 0.132
PLANE_COLUMNS = 7
TWIN_SEPARATION_M = 0.043
MULTI_CENTERS_M = ((-0.041, -0.023), (0.039, -0.017), (-0.008, 0.048))
BRANCH_LENGTH_M = 0.190
BRANCH_ANGLE_DEGREES = 55.0
BRANCH_ANCHOR_Z_M = 0.86
GAP_LIMITS_M = (0.69, 0.90)
ROTATING_ARC_DEGREES = 70.0
ROTATING_ARC_TURNS = 2.6
HELIX_TURNS = 8.25
ROTATION_DEGREES = (23.0, -47.0, 62.0)
TRANSLATION_M = (-0.19, 0.28, 0.41)


def _samples(length, phase, *, intervals=None):
    lattice = (np.arange(int(np.ceil(length / AXIAL_STEP_M)) + 1) + SAMPLING_PHASES[phase]) * AXIAL_STEP_M
    return np.unique(np.concatenate([
        np.r_[low, lattice[(lattice > low) & (lattice < high)], high]
        for low, high in (intervals or ((0.0, length),))
    ]))


def _bare_line(segment, phase, random, *, anchor=None):
    start, end = segment
    length = np.linalg.norm(end - start)
    direction = (end - start) / length
    along = _samples(length, phase)
    if anchor is not None:
        along = np.unique(np.r_[along, anchor])
    helper = np.eye(3)[int(np.argmin(np.abs(direction)))]
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    noise = random.uniform(-LINE_TRANSVERSE_NOISE_BOUND_M, LINE_TRANSVERSE_NOISE_BOUND_M, (len(along), 2))
    noise[(along == 0.0) | (along == length)] = 0.0
    if anchor is not None:
        noise[along == anchor] = 0.0
    return start + along[:, None] * direction + noise[:, :1] * u + noise[:, 1:] * v


def make_case(case_id, *, seed=DEFAULT_SEED, phase=0, voxel_size=GRID_SCALES_M[0]):
    """Return unlabelled point-only inputs and separately owned author truth."""
    if case_id not in CASE_IDS:
        raise ValueError("Unknown local-evidence control family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(phase) is not int or phase not in PHASES:
        raise ValueError("Choose a predeclared physical sampling phase")
    if type(voxel_size) not in (int, float) or voxel_size not in GRID_SCALES_M:
        raise ValueError("Choose a predeclared voxel scale")

    random = np.random.default_rng(seed)
    empty = np.empty((0, 2, 3), dtype="<f8")
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH_M]])
    target, construction, forbidden = axis[None].copy(), empty.copy(), empty.copy()
    expected, angular_samples, angular_support = "line", None, None
    intervals = ((0.0, GAP_LIMITS_M[0]), (GAP_LIMITS_M[1], LENGTH_M)) if case_id == "true_gap_half" else None
    z = _samples(LENGTH_M, phase, intervals=intervals)
    surface = case_id in ("full_shell", "half_shell", "true_gap_half", "rotating_sparse_arc", "short_pitch_helix")

    if case_id in ("twin_bare_lines", "noncoplanar_rods_with_branch"):
        centers = ((-TWIN_SEPARATION_M / 2, 0.0), (TWIN_SEPARATION_M / 2, 0.0)) if case_id == "twin_bare_lines" else MULTI_CENTERS_M
        target = np.array([[[x, y, 0.0], [x, y, LENGTH_M]] for x, y in centers])
        clouds = [_bare_line(segment, phase, random, anchor=BRANCH_ANCHOR_Z_M if index == 2 else None)
                  for index, segment in enumerate(target)]
        expected = "two_lines"
        if case_id == "noncoplanar_rods_with_branch":
            start = np.array([*MULTI_CENTERS_M[2], BRANCH_ANCHOR_Z_M])
            angle = np.deg2rad(BRANCH_ANGLE_DEGREES)
            branch = np.array([start, start + BRANCH_LENGTH_M * np.array([np.sin(angle), 0.0, np.cos(angle)])])
            target = np.concatenate((target, branch[None]))
            clouds.append(_bare_line(branch, phase, random))
            expected = "branches"
        points = np.concatenate(clouds)
    elif case_id == "jittered_sparse_plane":
        x, axial = np.meshgrid(np.linspace(-PLANE_WIDTH_M / 2, PLANE_WIDTH_M / 2, PLANE_COLUMNS), z)
        transverse = random.uniform(-PLANE_TRANSVERSE_NOISE_BOUND_M, PLANE_TRANSVERSE_NOISE_BOUND_M, (x.size, 2))
        axial_noise = random.uniform(-PLANE_AXIAL_NOISE_BOUND_M, PLANE_AXIAL_NOISE_BOUND_M, x.size)
        axial_noise[(axial.ravel() == 0.0) | (axial.ravel() == LENGTH_M)] = 0.0
        points = np.c_[x.ravel() + transverse[:, 0], transverse[:, 1], axial.ravel() + axial_noise]
        target, expected = empty.copy(), "abstain"
    else:
        if case_id == "full_shell":
            angles = np.tile(np.linspace(0.0, 2 * np.pi, 28, endpoint=False), (len(z), 1))
            angular_support = 360.0
        elif case_id in ("half_shell", "true_gap_half"):
            angles = np.tile(np.linspace(-np.pi / 2, np.pi / 2, 15), (len(z), 1))
            angular_support = 180.0
            if case_id == "true_gap_half":
                low, high = GAP_LIMITS_M
                target = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, low]], [[0.0, 0.0, high], [0.0, 0.0, LENGTH_M]]])
                forbidden = np.array([[[0.0, 0.0, low], [0.0, 0.0, high]]])
                expected = "gap"
        else:
            offsets = np.linspace(-ROTATING_ARC_DEGREES / 2, ROTATING_ARC_DEGREES / 2, 4) if case_id == "rotating_sparse_arc" else np.array([0.0])
            turns = ROTATING_ARC_TURNS if case_id == "rotating_sparse_arc" else HELIX_TURNS
            angles = np.deg2rad(offsets)[None] + 2 * np.pi * turns * z[:, None] / LENGTH_M
            target, construction, expected = empty.copy(), axis[None].copy(), "unresolved"
            angular_support = ROTATING_ARC_DEGREES if case_id == "rotating_sparse_arc" else 0.0
        angular_samples = angles.shape[1]
        radius = RADIUS_M + random.uniform(-RADIAL_NOISE_BOUND_M, RADIAL_NOISE_BOUND_M, angles.shape)
        points = np.c_[(radius * np.cos(angles)).ravel(), (radius * np.sin(angles)).ravel(), np.repeat(z, angular_samples)]

    rotation, translation = rotation_xyz(ROTATION_DEGREES), np.array(TRANSLATION_M)

    def transform(values):
        return np.asarray(values @ rotation.T + translation, dtype="<f8").copy()

    return {
        "cohort": COHORT,
        "case_id": case_id,
        "condition_id": f"{case_id}-grid-{GRID_SCALES_M.index(voxel_size)}-phase-{phase}",
        "inputs": {"points": transform(points), "segments": empty.copy()},
        "truth": {
            "expected": expected,
            "expected_segments": transform(target),
            "forbidden_segments": transform(forbidden),
            "construction_segments": transform(construction),
            "scope": SCOPE,
            "abstention_is_recovery": False,
            "negative_labels_are_construction_assumptions": True,
            "unresolved_construction_is_not_negative_foreground_truth": expected == "unresolved",
            "unresolved_scope": "Incomplete local circular support does not certify a straight member axis; absence is not proved" if expected == "unresolved" else None,
            "negative_scope": "Authored noisy seven-column thin plane, not proof against every coplanar real-rod interpretation" if expected == "abstain" else None,
        },
        "protocol": {
            "cohort": COHORT, "seed": seed, "density": DENSITY,
            "axial_step_m": AXIAL_STEP_M, "phase": phase, "sampling_phase": SAMPLING_PHASES[phase],
            "sampling_phase_unit": "fraction_of_physical_axial_step", "voxel_size": float(voxel_size),
            "grid_origin": [0.0, 0.0, 0.0], "local_to_world_rotation": rotation.tolist(),
            "local_to_world_translation": translation.tolist(), "rotation_degrees": list(ROTATION_DEGREES),
            "length_m": LENGTH_M, "radius_m": RADIUS_M if surface else None,
            "radial_noise_bound_m": RADIAL_NOISE_BOUND_M if surface else None,
            "line_transverse_noise_bound_m": LINE_TRANSVERSE_NOISE_BOUND_M if "bare" in case_id or "rods" in case_id else None,
            "plane_transverse_noise_bound_m": PLANE_TRANSVERSE_NOISE_BOUND_M if expected == "abstain" else None,
            "plane_axial_noise_bound_m": PLANE_AXIAL_NOISE_BOUND_M if expected == "abstain" else None,
            "plane_columns": PLANE_COLUMNS if expected == "abstain" else None,
            "plane_width_m": PLANE_WIDTH_M if expected == "abstain" else None,
            "angular_samples_per_surface": angular_samples, "local_angular_support_degrees": angular_support,
            "axial_layer_count_before_jitter": len(z), "finite_endpoints_and_gap_boundaries_observed": True,
            "short_branch_length_m": BRANCH_LENGTH_M if expected == "branches" else None,
            "short_branch_angle_degrees": BRANCH_ANGLE_DEGREES if expected == "branches" else None,
            "sampling_phase_changes_observation_locations_not_truth": True,
            "normal_inputs_are_point_only": True,
            "representation_pairs_are_identical_inputs_not_independent_evidence": True,
            "scope": SCOPE, "scale_and_phase_are_repeated_conditions_not_objects": True,
            "prior_scored_44_36_40_32_are_separate_development_regression": True,
        },
    }


def generate(*, seed=DEFAULT_SEED):
    """Yield 8 families × 2 voxel scales × 2 physical sampling phases."""
    for case_id in CASE_IDS:
        for voxel_size in GRID_SCALES_M:
            for phase in PHASES:
                yield make_case(case_id, seed=seed, phase=phase, voxel_size=voxel_size)
