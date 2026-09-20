"""Test five-view support dropout without a rendered scene or truth files."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_cylinder_gate import DEFAULT_POLICY, recheck_finite_cylinder  # noqa: E402
from creator_eval.rod_cylinder_support import select_supported_cylinders  # noqa: E402
from run_rod_cylinder_controls import analytic_case  # noqa: E402
from run_rod_identity_blender import digest, write_json  # noqa: E402

EXTENT = {"minimum_second_to_first_plane_ratio": 1e-5, "minimum_direction_gap_ratio": 1e-8}


def five_view_case(bad_views=0):
    # Generate a fifth independent camera through the same quadratic oracle.
    h, views, observations = analytic_case()
    fifth, fifth_obs = copy.deepcopy(views[0]), copy.deepcopy(observations[0])
    # Rotate one calibrated camera around the cylinder axis. The infinite cylinder
    # is unchanged, so its quadratic-oracle edge pixels remain valid in that camera.
    angle = .12
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    shift = h["model"]["anchor"] - rotation @ h["model"]["anchor"]
    ext = fifth["world_to_camera_cv"]
    camera_rotation = ext[:, :3] @ rotation.T
    fifth["world_to_camera_cv"] = np.c_[camera_rotation, ext[:, 3] - camera_rotation @ shift]
    views.append(fifth)
    observations.append(fifth_obs)
    views[-1]["view_id"] = "4"
    h["supporting_views"] = list(range(5))
    h["matches"] += [{"candidate_index": 0}]
    for index in range(5 - bad_views, 5):
        # Keep the center line exactly fixed but corrupt the measured side pair.
        # This isolates width evidence from axis-fit errors.
        for row in observations[index]["rows"]:
            row["candidates"][0]["left_edge"]["x"] -= 3
            row["candidates"][0]["right_edge"]["x"] += 3
    association = {"state": "accepted", "selected": h, "alternatives": [], "search_complete": True}
    return association, views, observations


def run_controls(output):
    rows = []
    for bad_views in (0, 1, 2):
        association, views, obs = five_view_case(bad_views)
        original = copy.deepcopy(association)
        result = select_supported_cylinders(association, views, obs, EXTENT)
        expected = "accepted" if bad_views < 2 else "rejected"
        if result["state"] != expected or association["selected"]["supporting_views"] != original["selected"]["supporting_views"]:
            raise AssertionError("Support dropout control failed or mutated input")
        finite = bound_selected_candidate(result, views, obs)
        finite = recheck_finite_cylinder(finite, result["selected"], views, obs)
        if bad_views == 1 and finite["candidate_selection"][-1]["row_matches"]:
            raise AssertionError("An excluded view still supplied finite evidence")
        rows.append({"bad_views": bad_views, "expected": expected, "state": result["state"],
                     "surviving_views": result["selected"]["supporting_views"] if result["selected"] else [],
                     "finite_state": finite["state"], "support_audit": result["cylinder_support"]})
    paths = ["scripts/run_rod_cylinder_support_controls.py", "scripts/run_rod_cylinder_controls.py",
             "experiments/src/creator_eval/rod_cylinder_gate.py", "experiments/src/creator_eval/rod_cylinder_support.py"]
    write_json(output, {"state": "complete", "policy": DEFAULT_POLICY, "rows": rows,
                        "source_sha256": {p: digest(ROOT / p) for p in paths},
                        "scope": "mechanism controls for one-pass view exclusion; not independent rendered validation"})
    print("SUPPORT_CONTROLS", len(rows), "passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-20-cylinder-support-controls.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run_controls(args.output)
