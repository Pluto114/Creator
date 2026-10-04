"""New authored local-section-support controls, with isolated construction truth.

Only ``inputs`` belongs in a normal reader call. No reader implementation, runtime
artifact, fixture data, or output score is inspected. These 32 repeated analytic
conditions are not independent physical objects or real-photograph qualification.
The earlier 44/36/40 scored controls remain separate development-replay cohorts.

Noise is bounded and transverse: radial for circular constructions, normal to the
plane for the plane control. There is no axial noise or hidden input centerline.
Phase changes physical axial sample locations, not the object or grid origin.
"""

from __future__ import annotations

import numpy as np

from .mixed_readout_controls import SCOPE
from .readout_background_controls import rotation_xyz

COHORT = "new_section_support_controls_v1"
CASE_IDS = (
    "full_noisy_sections",
    "half_noisy_sections",
    "true_gap_half",
    "tapered_sections",
    "locally_missing_angles",
    "rotating_sparse_arc",
    "helical_trace",
    "noisy_plane",
)
GRID_SCALES_M = (0.0055, 0.0085)
PHASES = (0, 1)
SAMPLING_PHASES = (0.13, 0.61)
DENSITY = "fixed_19mm"
AXIAL_STEP_M = 0.019
DEFAULT_SEED = 441103
LENGTH_M = 1.45
RADIUS_M = 0.029
RADIAL_NOISE_BOUND_M = 0.00025
PLANE_NOISE_BOUND_M = 0.00020
GAP_LIMITS_M = (0.65, 0.80)
ROTATION_DEGREES = (-41.0, 26.0, 14.0)
TRANSLATION_M = (0.31, 0.18, -0.27)
LOCAL_ARC_DEGREES = 45.0
ROTATING_ARC_DEGREES = 100.0
ROTATING_ARC_TURNS = 1.4
HELIX_TURNS = 2.25


def _axial_samples(phase, *, gapped=False):
    lattice = (np.arange(int(np.ceil(LENGTH_M / AXIAL_STEP_M)) + 1) + SAMPLING_PHASES[phase]) * AXIAL_STEP_M
    intervals = ((0.0, GAP_LIMITS_M[0]), (GAP_LIMITS_M[1], LENGTH_M)) if gapped else ((0.0, LENGTH_M),)
    return np.unique(np.concatenate([
        np.r_[low, lattice[(lattice > low) & (lattice < high)], high]
        for low, high in intervals
    ]))


def make_case(case_id, *, seed=DEFAULT_SEED, phase=0, voxel_size=GRID_SCALES_M[0]):
    """Make one frozen-design condition without invoking any reconstruction code."""
    if case_id not in CASE_IDS:
        raise ValueError("Unknown local-section-support control family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(phase) is not int or phase not in PHASES:
        raise ValueError("Choose a predeclared physical sampling phase")
    if type(voxel_size) not in (int, float) or voxel_size not in GRID_SCALES_M:
        raise ValueError("Choose a predeclared voxel scale")

    random = np.random.default_rng(seed)
    z = _axial_samples(phase, gapped=case_id == "true_gap_half")
    empty = np.empty((0, 2, 3), dtype="<f8")
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH_M]])
    target, construction, forbidden = axis[None].copy(), empty.copy(), empty.copy()
    expected = "line"
    nominal_radius = np.full(len(z), RADIUS_M)
    local_angular_support = None

    if case_id == "noisy_plane":
        x, axial = np.meshgrid(np.linspace(-0.065, 0.065, 31), z)
        points = np.c_[x.ravel(), random.uniform(-PLANE_NOISE_BOUND_M, PLANE_NOISE_BOUND_M, x.size), axial.ravel()]
        target, expected = empty.copy(), "abstain"
        angular_samples = None
    else:
        if case_id in ("half_noisy_sections", "true_gap_half"):
            angles = np.tile(np.linspace(-np.pi / 2, np.pi / 2, 16), (len(z), 1))
            local_angular_support = 180.0
            if case_id == "true_gap_half":
                low, high = GAP_LIMITS_M
                target = np.array([
                    [[0.0, 0.0, 0.0], [0.0, 0.0, low]],
                    [[0.0, 0.0, high], [0.0, 0.0, LENGTH_M]],
                ])
                forbidden = np.array([[[0.0, 0.0, low], [0.0, 0.0, high]]])
                expected = "gap"
        elif case_id in ("full_noisy_sections", "tapered_sections"):
            angles = np.tile(np.linspace(0.0, 2 * np.pi, 30, endpoint=False), (len(z), 1))
            local_angular_support = 360.0
            if case_id == "tapered_sections":
                # A coaxial circular taper still has a declared straight axis.
                # A constant-radius reader's abstention is a positive failure.
                nominal_radius = RADIUS_M * (0.72 + 0.56 * z / LENGTH_M)
        elif case_id == "locally_missing_angles":
            # Four genuinely different visible 45-degree patches, not a random
            # train/validation split of a complete local ring. The total cloud
            # spans much more azimuth than any individual axial block.
            block = np.minimum(np.floor(4 * z / LENGTH_M).astype(int), 3)
            offsets = np.linspace(-LOCAL_ARC_DEGREES / 2, LOCAL_ARC_DEGREES / 2, 11)
            angles = np.deg2rad(90 * block[:, None] + offsets[None])
            target, construction, expected = empty.copy(), axis[None].copy(), "unresolved"
            local_angular_support = LOCAL_ARC_DEGREES
        else:
            offsets = np.linspace(-ROTATING_ARC_DEGREES / 2, ROTATING_ARC_DEGREES / 2, 5) if case_id == "rotating_sparse_arc" else np.array([0.0])
            turns = ROTATING_ARC_TURNS if case_id == "rotating_sparse_arc" else HELIX_TURNS
            angles = np.deg2rad(offsets)[None] + 2 * np.pi * turns * z[:, None] / LENGTH_M
            target, construction, expected = empty.copy(), axis[None].copy(), "unresolved"
            local_angular_support = ROTATING_ARC_DEGREES if case_id == "rotating_sparse_arc" else 0.0
        angular_samples = angles.shape[1]
        radius = nominal_radius[:, None] + random.uniform(-RADIAL_NOISE_BOUND_M, RADIAL_NOISE_BOUND_M, angles.shape)
        points = np.c_[
            (radius * np.cos(angles)).ravel(),
            (radius * np.sin(angles)).ravel(),
            np.repeat(z, angular_samples),
        ]

    rotation = rotation_xyz(ROTATION_DEGREES)
    translation = np.array(TRANSLATION_M)

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
            "unresolved_scope": (
                "No declared locally supported full/half circular member axis; empty reference is not proof of physical absence"
                if expected == "unresolved" else None
            ),
        },
        "protocol": {
            "cohort": COHORT,
            "seed": seed,
            "density": DENSITY,
            "axial_step_m": AXIAL_STEP_M,
            "phase": phase,
            "sampling_phase": SAMPLING_PHASES[phase],
            "sampling_phase_unit": "fraction_of_physical_axial_step",
            "voxel_size": float(voxel_size),
            "grid_origin": [0.0, 0.0, 0.0],
            "local_to_world_rotation": rotation.tolist(),
            "local_to_world_translation": translation.tolist(),
            "rotation_degrees": list(ROTATION_DEGREES),
            "length_m": LENGTH_M,
            "radius_m": None if case_id == "noisy_plane" else RADIUS_M,
            "nominal_radius_range_m": None if case_id == "noisy_plane" else [float(nominal_radius.min()), float(nominal_radius.max())],
            "radial_noise_bound_m": None if case_id == "noisy_plane" else RADIAL_NOISE_BOUND_M,
            "plane_normal_noise_bound_m": PLANE_NOISE_BOUND_M if case_id == "noisy_plane" else None,
            "angular_samples_per_surface": angular_samples,
            "local_angular_support_degrees": local_angular_support,
            "axial_layer_count": len(z),
            "finite_endpoints_and_gap_boundaries_observed": True,
            "sampling_phase_changes_observation_locations_not_truth": True,
            "scope": SCOPE,
            "scale_and_phase_are_repeated_conditions_not_objects": True,
            "prior_scored_44_36_40_are_separate_development_regression": True,
        },
    }


def generate(*, seed=DEFAULT_SEED):
    """Yield only new 8 families × 2 scales × 2 sampling phases (32 conditions)."""
    for case_id in CASE_IDS:
        for voxel_size in GRID_SCALES_M:
            for phase in PHASES:
                yield make_case(case_id, seed=seed, phase=phase, voxel_size=voxel_size)
