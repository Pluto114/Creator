"""View-dropout contracts for the cylinder screen, including finite evidence."""
import copy
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_cylinder_support import select_supported_cylinders  # noqa: E402
from run_rod_cylinder_support_controls import EXTENT, five_view_case  # noqa: E402


class CylinderSupportTests(unittest.TestCase):
    def test_single_bad_view_excluded_from_fit_and_finite_evidence(self):
        association, views, obs = five_view_case(1)
        result = select_supported_cylinders(association, views, obs, EXTENT)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["selected"]["supporting_views"], [0, 1, 2, 3])
        finite = bound_selected_candidate(result, views, obs)
        self.assertEqual(finite["state"], "accepted")
        self.assertEqual(finite["candidate_selection"][-1]["row_matches"], [])
        self.assertFalse(np.any(np.asarray(finite["evidence"]["view_states"])[-1]))
        self.assertEqual(association["selected"]["supporting_views"], list(range(5)))

    def test_two_bad_views_cannot_get_three_view_acceptance(self):
        association, views, obs = five_view_case(2)
        result = select_supported_cylinders(association, views, obs, EXTENT)
        self.assertEqual(result["state"], "rejected")
        self.assertIsNone(result["selected"])

    def test_competing_assignments_remain_ambiguous(self):
        association, views, obs = five_view_case()
        other = copy.deepcopy(association["selected"])
        for i, view in enumerate(views):
            view["candidates"].append(copy.deepcopy(view["candidates"][0]))
            other["matches"][i]["candidate_index"] = 1
        association["alternatives"] = [other]
        result = select_supported_cylinders(association, views, obs, EXTENT)
        self.assertEqual(result["state"], "ambiguous")
        self.assertEqual(result["cylinder_screen"]["survivor_count"], 2)

    def test_budget_exhaustion_stays_ambiguous_after_good_support(self):
        association, views, obs = five_view_case(1)
        association["search_complete"] = False
        self.assertEqual(select_supported_cylinders(association, views, obs, EXTENT)["state"], "ambiguous")

    def test_new_replay_clis_start_without_import_side_effects(self):
        for filename in ("run_rod_cylinder_screen.py", "run_rod_cylinder_support.py", "run_rod_cylinder_support_controls.py", "run_rod_cylinder_validation.py"):
            with self.subTest(filename=filename):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / filename), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
