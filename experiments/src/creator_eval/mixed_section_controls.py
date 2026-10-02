"""New analytic section controls, separate from the scored mixed-readout set.

Only ``case["inputs"]`` is normal reader input. The other fields describe the
construction for a separately stored evaluation manifest. No reader, fixture,
runtime artifact, or score is accessed. These are authored analytic shapes, not
new physical objects, real photographs, or blind foreground-identity evidence.

The 36 conditions are nine families crossed with two voxel scales and two
physical axial sampling steps. Sampling phase, world pose, and dimensions are
fixed before inference; changing a reader scale never changes the point cloud.
The old 44 conditions must remain a separately named development-regression set.
"""

from __future__ import annotations

import numpy as np

from .mixed_readout_controls import SCOPE
from .readout_background_controls import rotation_xyz

COHORT = "new_section_analytic_controls_v1"
CASE_IDS = (
    "full_cylinder",
    "half_cylinder",
    "small_twin_cylinders",
    "gapped_half_cylinder",
    "plane",
    "rotating_sparse_arc",
    "helical_trace",
    "axis_scatter",
    "scatter_only",
)
GRID_SCALES_M = (0.006, 0.010)
DENSITIES = ("dense", "coarse")
AXIAL_STEPS_M = {"dense": 0.004, "coarse": 0.012}
SAMPLING_PHASE = 0.37
DEFAULT_SEED = 421017
LENGTH_M = 1.6
RADIUS_M = 0.018
SMALL_RADIUS_M = 0.004
TWIN_AXIS_DISTANCE_M = 0.044
GAP_LIMITS_M = (0.64, 0.96)
ROTATION_DEGREES = (-29.0, 34.0, 53.0)
TRANSLATION_M = (0.27, -0.41, 0.63)


def _axial_samples(density, *, gapped=False):
    """One global phased lattice; declared finite boundaries are always observed."""
    step = AXIAL_STEPS_M[density]
    lattice = (np.arange(int(np.ceil(LENGTH_M / step)) + 1) + SAMPLING_PHASE) * step
    intervals = ((0.0, GAP_LIMITS_M[0]), (GAP_LIMITS_M[1], LENGTH_M)) if gapped else ((0.0, LENGTH_M),)
    return np.unique(np.concatenate([
        np.r_[low, lattice[(lattice > low) & (lattice < high)], high]
        for low, high in intervals
    ]))


def _surface(z, radius, *, half=False, x_offset=0.0):
    angles = np.linspace(-np.pi / 2, np.pi / 2, 25) if half else np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    ring = np.c_[radius * np.cos(angles) + x_offset, radius * np.sin(angles)]
    return np.c_[np.tile(ring, (len(z), 1)), np.repeat(z, len(angles))]


def make_case(case_id, *, seed=DEFAULT_SEED, density="dense", voxel_size=GRID_SCALES_M[0]):
    """Make one input/truth-separated condition without running any method.

    Seed variation exists for deterministic generator contracts, not post-score
    retries. A formal run must freeze the chosen seed, density, and voxel scale.
    """
    if case_id not in CASE_IDS:
        raise ValueError("Unknown mixed-section control family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if density not in DENSITIES:
        raise ValueError("Choose a predeclared axial sampling density")
    if type(voxel_size) not in (int, float) or voxel_size not in GRID_SCALES_M:
        raise ValueError("Choose a predeclared voxel scale")

    empty = np.empty((0, 2, 3), dtype="<f8")
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH_M]])
    z = _axial_samples(density, gapped=case_id == "gapped_half_cylinder")
    segments = empty.copy()
    target, forbidden, construction = axis[None].copy(), empty.copy(), empty.copy()
    expected = "line"
    radius = RADIUS_M
    angular_samples = None

    if case_id in ("full_cylinder", "half_cylinder", "gapped_half_cylinder"):
        half = case_id != "full_cylinder"
        points = _surface(z, RADIUS_M, half=half)
        angular_samples = 25 if half else 48
        if case_id == "gapped_half_cylinder":
            low, high = GAP_LIMITS_M
            target = np.array([
                [[0.0, 0.0, 0.0], [0.0, 0.0, low]],
                [[0.0, 0.0, high], [0.0, 0.0, LENGTH_M]],
            ])
            forbidden = np.array([[[0.0, 0.0, low], [0.0, 0.0, high]]])
            expected = "gap"
    elif case_id == "small_twin_cylinders":
        offsets = (-TWIN_AXIS_DISTANCE_M / 2, TWIN_AXIS_DISTANCE_M / 2)
        points = np.concatenate([_surface(z, SMALL_RADIUS_M, x_offset=x) for x in offsets])
        target = np.array([axis + [x, 0.0, 0.0] for x in offsets])
        expected, radius, angular_samples = "two_lines", SMALL_RADIUS_M, 48
    elif case_id == "plane":
        x, axial = np.meshgrid(np.linspace(-0.070, 0.070, 25), z)
        points = np.c_[x.ravel(), np.zeros(x.size), axial.ravel()]
        target, expected, radius = empty.copy(), "abstain", None
    elif case_id in ("rotating_sparse_arc", "helical_trace"):
        # Global azimuth coverage cannot supply unobserved local circular sections.
        offsets = np.linspace(-np.pi / 6, np.pi / 6, 4) if case_id == "rotating_sparse_arc" else np.array([0.0])
        angles = offsets[None] + 2 * np.pi * z[:, None] / LENGTH_M
        points = np.c_[RADIUS_M * np.cos(angles).ravel(), RADIUS_M * np.sin(angles).ravel(),
                       np.repeat(z, len(offsets))]
        target, construction, expected = empty.copy(), axis[None].copy(), "unresolved"
        angular_samples = len(offsets)
    else:
        # Same seed and exact points in the paired axis/scatter-only arms. Each
        # axial observation layer has independent lateral samples, not repeated
        # columns which would manufacture several long finite input lines.
        xy = np.random.default_rng(seed).uniform([-0.060, -0.050], [0.060, 0.050], size=(len(z) * 8, 2))
        points = np.c_[xy, np.repeat(z, 8)]
        radius = None
        if case_id == "axis_scatter":
            segments = axis[None].copy()
        else:
            target, expected = empty.copy(), "abstain"

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
        },
        "protocol": {
            "cohort": COHORT,
            "seed": seed,
            "density": density,
            "axial_step_m": AXIAL_STEPS_M[density],
            "sampling_phase": SAMPLING_PHASE,
            "sampling_phase_unit": "fraction_of_physical_axial_step",
            "voxel_size": float(voxel_size),
            "grid_origin": [0.0, 0.0, 0.0],
            "local_to_world_rotation": rotation.tolist(),
            "local_to_world_translation": translation.tolist(),
            "rotation_degrees": list(ROTATION_DEGREES),
            "length_m": LENGTH_M,
            "radius_m": radius,
            "angular_samples_per_surface": angular_samples,
            "axial_layer_count": len(z),
            "finite_endpoints_and_gap_boundaries_observed": True,
            "scope": SCOPE,
            "scale_and_density_are_repeated_conditions_not_objects": True,
            "old_44_scored_conditions_are_separate_development_regression": True,
        },
    }


def generate(*, seed=DEFAULT_SEED):
    """Yield only new 9 × 2 scales × 2 densities, never the old 44 conditions."""
    for case_id in CASE_IDS:
        for voxel_size in GRID_SCALES_M:
            for density in DENSITIES:
                yield make_case(case_id, seed=seed, density=density, voxel_size=voxel_size)
