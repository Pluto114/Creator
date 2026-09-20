"""Independent quadratic tangent controls for the pixel cylinder screen."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_cylinder_gate import DEFAULT_POLICY, screen_cylinder  # noqa: E402
from run_rod_identity_blender import digest, write_json  # noqa: E402


def analytic_case(radius=.08, tilt=.0, noise=.0, seed=20260920, counts=None):
    """Solve the ray/cylinder intersection discriminant, not the screen's planes."""
    anchor, axis = np.array([.1, -.2, 5.]), np.array([tilt, 1., .15 * tilt])
    axis /= np.linalg.norm(axis)
    rng = np.random.default_rng(seed)
    views, observations = [], []
    for i, cx in enumerate((-1.2, -.45, .45, 1.2)):
        center = np.array([cx, .1 * (i - 1), .1 * i])
        forward = anchor - center
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        intrinsic = np.array([[350 + 35 * i, 0, 159.5], [0, 370 + 25 * i, 119.5], [0, 0, 1.]])
        inverse = np.linalg.inv(intrinsic)
        q = anchor - center
        q -= (q @ axis) * axis
        a = rotation.T @ inverse[:, 0]
        a -= (a @ axis) * axis
        rows = []
        for y in np.linspace(25, 215, (counts or [96] * 4)[i]):
            b = rotation.T @ inverse @ [0, y, 1]
            b -= (b @ axis) * axis
            scale = q @ q - radius * radius
            coeff = [(a @ q) ** 2 - scale * (a @ a),
                     2 * ((a @ q) * (b @ q) - scale * (a @ b)),
                     (b @ q) ** 2 - scale * (b @ b)]
            roots = np.roots(coeff)
            if not np.isreal(roots).all():
                raise ValueError("Analytic tangent quadratic has no real roots")
            left, right_x = sorted(roots.real)
            if noise:
                left, right_x = np.array([left, right_x]) + rng.uniform(-noise, noise, 2)
            if left >= right_x:
                # A subpixel band can become unmeasurable; do not fabricate an ordered observation.
                continue
            rows.append({"y": float(y), "guide_x": float((left + right_x) / 2),
                         "candidates": [{"center_x": float((left + right_x) / 2), "width": float(right_x - left),
                                         "left_edge": {"x": float(left)}, "right_edge": {"x": float(right_x)}}]})
        yy = np.array([r["y"] for r in rows])
        xx = np.array([r["guide_x"] for r in rows])
        slope, intercept = np.linalg.lstsq(np.c_[yy, np.ones(len(yy))], xx, rcond=None)[0]
        line = np.array([1., -slope, -intercept])
        line /= np.linalg.norm(line[:2])
        view = {"view_id": str(i), "K_index": intrinsic, "world_to_camera_cv": np.c_[rotation, -rotation @ center],
                "size_wh": [320, 240], "y_range": [25, 215], "guide_line": line,
                "candidates": [{"line": line, "row_matches": [[j, 0, 0] for j in range(len(rows))]}]}
        views.append(view)
        observations.append({"rows": rows})
    hypothesis = {"model": {"anchor": anchor, "direction": axis}, "supporting_views": list(range(4)),
                  "matches": [{"candidate_index": 0} for _ in range(4)]}
    return hypothesis, views, observations


def run_controls(output):
    rows = []
    for radius in (.004, .015, .08, .25):
        for tilt in (0., .18):
            for noise in (0., .25, .5, 1.):
                h, views, obs = analytic_case(radius, tilt, noise)
                result = screen_cylinder(h, views, obs)
                rows.append({"control": "bounded_uniform_edge_noise", "radius": radius, "tilt": tilt, "noise_px": noise,
                             "retained_rows": [len(o["rows"]) for o in obs], "diagnostic": result})
    for mode in ("axis_shift", "one_bad_side_in_one_view", "coherent_other_real_cylinder", "camera_translation_error"):
        h, views, obs = analytic_case()
        if mode == "axis_shift":
            h["model"]["anchor"][0] += .06
        elif mode == "one_bad_side_in_one_view":
            for row in obs[-1]["rows"]:
                row["candidates"][0]["right_edge"]["x"] += 3
        elif mode == "camera_translation_error":
            views[-1]["world_to_camera_cv"][0, 3] += .04
        result = screen_cylinder(h, views, obs)
        rows.append({"control": mode, "diagnostic": result,
                     "interpretation": "another self-consistent cylinder must pass; identity remains external" if mode == "coherent_other_real_cylinder" else "sensitivity control"})
    # Complete acceptance assertions only where the analytic oracle warrants them.
    for row in rows:
        if row["control"] == "bounded_uniform_edge_noise" and row["noise_px"] == 0 and not row["diagnostic"]["passed"]:
            raise AssertionError("Exact tangent control was rejected")
    artifact = {"state": "complete", "policy": copy.deepcopy(DEFAULT_POLICY), "rows": rows,
                "source_sha256": {p: digest(ROOT / p) for p in ("scripts/run_rod_cylinder_controls.py", "experiments/src/creator_eval/rod_cylinder_gate.py")},
                "scope": "analytic conditional-axis mechanism controls; not rendered RGB or independent-object validation"}
    write_json(output, artifact)
    print("CYLINDER_CONTROLS", len(rows), "cases", sum(r["diagnostic"]["passed"] for r in rows), "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-20-cylinder-controls.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run_controls(args.output)
