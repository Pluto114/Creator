"""Ablation plans must separate identity guides from the fixed candidate evidence."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_candidate_ablation import (  # noqa: E402
    canonical_hash,
    checked_artifact,
    expected_keys,
    inference_view,
    require_legacy_regression,
    validate_plan,
)
from run_rod_identity_blender import digest, write_json  # noqa: E402


class CandidateAblationTests(unittest.TestCase):
    def fixture(self):
        config = {"case_ids": ["a", "b"], "search_offsets_px": [0, 8],
                  "identity_offsets_px": [0, 8, 16], "candidate_caps": [8, 16],
                  "association_budget": 40960}
        frames = {key: [{"view_id": str(index)} for index in range(5)] for key in config["case_ids"]}
        return config, frames

    def test_budget_must_cover_largest_cap_and_plan_is_a_product(self):
        config, frames = self.fixture()
        validate_plan(config, frames)
        self.assertEqual(len(expected_keys(config)), 24)
        with self.assertRaisesRegex(ValueError, "budgets"):
            validate_plan({**config, "association_budget": 40959}, frames)

    def test_duplicate_or_non_nested_conditions_fail(self):
        config, frames = self.fixture()
        for field, value in (("candidate_caps", [16, 8]), ("candidate_caps", [8, 8]),
                             ("identity_offsets_px", [0, 0]), ("case_ids", ["a", "a"])):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan({**config, field: value}, frames)
        frames["a"][1]["view_id"] = frames["a"][0]["view_id"]
        with self.assertRaisesRegex(ValueError, "distinct views"):
            validate_plan(config, frames)

    def test_identity_offset_changes_only_the_guide_not_evidence(self):
        frame = {"view_id": "0", "K_index": np.eye(3).tolist(),
                 "world_to_camera_cv": np.eye(4).tolist(), "size_wh": [640, 480],
                 "guide_xyxy": [[319.5, 20], [319.5, 459]]}
        original = copy.deepcopy(frame)
        candidates = [{"line": [1, 0, -320], "row_matches": [[0, 0, 0.0]]}]
        before = inference_view(frame, candidates, 0)
        after = inference_view(frame, candidates, 8)
        self.assertIs(before["candidates"], after["candidates"])
        self.assertEqual(frame, original)
        for key in ("K_index", "world_to_camera_cv", "size_wh", "y_range"):
            self.assertEqual(before[key], after[key])
        self.assertAlmostEqual(abs(after["guide_line"][2] - before["guide_line"][2]), 8)
        self.assertEqual(canonical_hash({"a": np.array([1.0])}), canonical_hash({"a": [1.0]}))

    def test_failed_legacy_control_cannot_be_published_as_complete(self):
        good = {"association_exact_match": True, "guarded_state_equal": True, "finite_exact_match": True}
        require_legacy_regression([good])
        for records in ([], [{**good, "association_exact_match": False}],
                        [{**good, "finite_exact_match": False}], [{}]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, "Legacy regression"):
                require_legacy_regression(records)

    def test_cached_derived_artifact_cannot_silently_drift_or_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            path = run / "pool.json"
            write_json(path, {"rows": [1]})
            entry = {"path": "pool.json", "sha256": digest(path)}
            self.assertEqual(checked_artifact(run, entry), {"rows": [1]})
            path.write_text('{"rows":[2]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_artifact(run, entry)
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_artifact(run, {"path": "../outside.json", "sha256": digest(outside)})


if __name__ == "__main__":
    unittest.main()
