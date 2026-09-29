"""Known-answer tests for the opt-in unresolved narrow evidence branch."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_narrow_controls import (  # noqa: E402
    acceptance_failures,
    evaluate_narrow_control,
    render_narrow_fixture,
    run_narrow_control_methods,
)


class NarrowControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/rod_narrow_controls_v1.json").read_text(encoding="utf-8")
        )
        cls.rows = []
        cls.outcomes = {}
        for ordinal, case in enumerate(cls.protocol["cases"]):
            image, truth = render_narrow_fixture(
                case,
                cls.protocol["generator"],
                seed=cls.protocol["generator"]["seed"] + ordinal,
            )
            raw, methods = run_narrow_control_methods(
                image, cls.protocol["guide_xyxy"], cls.protocol["method"]
            )
            cls.outcomes[case["case_id"]] = (raw, methods, truth)
            for method, prediction in methods.items():
                score, _ = evaluate_narrow_control(
                    prediction, truth, cls.protocol["evaluation"]
                )
                score["method"] = method
                cls.rows.append(score)

    def row(self, case_id, method="unresolved_narrow_pairs"):
        return next(
            row for row in self.rows if row["case_id"] == case_id and row["method"] == method
        )

    def test_predeclared_acceptance_contract(self):
        self.assertEqual(
            acceptance_failures(self.rows, self.protocol["acceptance_contract"]), []
        )

    def test_legacy_branch_still_rejects_one_pixel_width(self):
        raw, _, _ = self.outcomes["n01"]
        self.assertTrue(all(not row["candidates"] for row in raw["resolved_pairs"]["rows"]))
        self.assertEqual(self.row("n01", "resolved_pairs")["selected_rows"], 0)

    def test_narrow_candidate_does_not_claim_physical_width(self):
        raw, _, _ = self.outcomes["n01"]
        candidates = [
            candidate
            for row in raw["unresolved_narrow_pairs"]["rows"]
            for candidate in row["candidates"]
        ]
        self.assertTrue(candidates)
        self.assertTrue(all(c["physical_width_resolved"] is False for c in candidates))
        self.assertTrue(all("foreground identity" in c["limitation"] for c in candidates))

    def test_small_true_gap_is_not_interpolated(self):
        self.assertEqual(self.row("n03")["gap_false_acceptance_fraction"], 0)

    def test_low_contrast_and_occlusion_do_not_become_positive(self):
        self.assertEqual(self.row("n04")["selected_rows"], 0)
        self.assertEqual(self.row("n06")["selected_occluded_rows"], 0)

    def test_equal_neighbor_lines_are_ambiguous(self):
        self.assertEqual(self.row("n05")["fit_state"], "ambiguous")
        self.assertEqual(self.row("n05")["selected_rows"], 0)

    def test_painted_line_exposes_single_view_identity_limit(self):
        row = self.row("n07")
        self.assertEqual(row["false_positive_empty_row_fraction"], 1)
        self.assertGreater(row["unresolved_narrow_selected_rows"], 0)


if __name__ == "__main__":
    unittest.main()
