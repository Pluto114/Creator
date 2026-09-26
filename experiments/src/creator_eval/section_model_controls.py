"""Paired synthetic section observations; semantic families never enter model inputs."""
from __future__ import annotations

import itertools

import numpy as np

FAMILIES = ("circle", "ellipse_close_extent", "ellipse_wide_extent", "two_circles_close", "two_circles_wide",
            "flat_strip", "folded_sheet", "corrugated_sheet")


def rotation(degrees):
    x, y, z = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz@ry@rx


def generate(protocol):
    g = protocol["generation"]
    if tuple(g["families"]) != FAMILIES or len(g["phases_voxels"]) != 2:
        raise ValueError("Exactly the frozen eight paired families and two phases required")
    angle = np.linspace(0., 2*np.pi, g["angular_samples"], endpoint=False)
    z = np.linspace(0., g["length_voxels"], g["axial_samples"])
    transform = rotation(g["rotation_degrees"])
    cases, truth = [], []
    for number, family in enumerate(FAMILIES):
        centers = []
        if family in ("circle", "ellipse_close_extent", "ellipse_wide_extent"):
            a, b = {"circle": (2.75, 2.75), "ellipse_close_extent": (3.25, 1.5), "ellipse_wide_extent": (4., 1.5)}[family]
            section = np.c_[a*np.cos(angle), b*np.sin(angle)]
            centers = [[0., 0.]]
        elif family.startswith("two_circles"):
            a = angle[::2]
            distance = 3.5 if family.endswith("close") else 5.
            centers = [[-distance/2, 0.], [distance/2, 0.]]
            section = np.concatenate([np.c_[1.5*np.cos(a)+c[0], 1.5*np.sin(a)] for c in centers])
        else:
            x = np.linspace(-4., 4., g["angular_samples"])
            y = np.zeros_like(x)
            if family == "folded_sheet":
                y = 1.5*np.abs(x/4.)
            elif family == "corrugated_sheet":
                y = 1.5*np.sin(np.pi*x/2.)
            section = np.c_[x, y]
        points = np.concatenate([np.c_[section, np.full(len(section), height)] for height in z])
        # 同一个几何族的噪声、可见范围和相位都用这一份随机扰动，别把种子换了再谈因果。
        noise = np.random.default_rng(g["seed"]+number).normal(size=points.shape)
        visible = section[:, 1] >= -1e-12 if centers else section[:, 0] >= 0.
        half = np.tile(visible, len(z))
        for sigma, visibility, phase in itertools.product(g["noise_sigmas_voxels"], g["visibility"], range(2)):
            keep = np.ones(len(points), bool) if visibility == "full" else half
            values = ((points+sigma*noise)[keep]@transform.T + g["phases_voxels"][phase])*g["voxel_m"]
            case_id = f"section-{len(cases):04d}"
            cases.append(dict(case_id=case_id, points=values.astype("<f8"), voxel_size=g["voxel_m"]))
            truth.append(dict(case_id=case_id, family=family, noise_sigma_voxels=sigma, visibility=visibility, phase=phase,
                paired_noise_seed=g["seed"]+number, expected_axis_count=len(centers) if centers else None,
                role="declared_member_geometry" if centers else "declared_sheet_geometry",
                cross_section_centers_voxels=centers, complete_surface=(visibility == "full"),
                missing_evidence_is_not_free_space=True))
    assert len(cases) == len(truth) == 64
    return cases, truth
