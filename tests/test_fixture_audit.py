"""Audit contract checks use tracked code and in-memory cameras, never run data."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_rod_fixture import (  # noqa: E402
    SCENE,
    block_truth,
    camera_physical,
    independent_observation_score,
)


class FixtureAuditTests(unittest.TestCase):
    def test_guard_rejects_relative_bytes_and_path_truth(self):
        for name in ("data/eval_gt/new/a.json", "data/evaluation/new/a.json", "docs/experiments/results/new.json"):
            relative = Path(ROOT / name)
            for value in (str(relative), bytes(str(relative), "utf-8"), relative):
                with self.assertRaises(PermissionError):
                    block_truth("open", (value,))
        old = Path.cwd()
        try:
            os.chdir(ROOT)
            for value in ("data/eval_gt/a.json", b"data/evaluation/a.json", Path("docs/experiments/results/a.json")):
                with self.assertRaises(PermissionError):
                    block_truth("open", (value,))
        finally:
            os.chdir(old)

    def test_guard_rejects_render_protocol_and_allows_declared_cad(self):
        for path in (SCENE / "protocol.json", SCENE / "case-00-render_request.json",
                     SCENE / "generation-checks.json", SCENE / "rendered-rgb/a.png"):
            with self.assertRaises(PermissionError):
                block_truth("open", (path,))
        for path in (ROOT / "configs/rod_fixture_cad_v1.json", ROOT / "data/inputs/a.png", SCENE / "generation-freeze.json"):
            block_truth("open", (path,))
        block_truth("open", (5,))
        block_truth("other", ("data/eval_gt/a",))

    def test_both_normal_guards_reject_relative_physical_results(self):
        from run_fixture_calibration import block_truth as calibration_guard
        from run_rod_fixture_finite import reject_truth_open as finite_guard
        old = Path.cwd()
        try:
            os.chdir(ROOT)
            for guard in (calibration_guard, finite_guard):
                for path in ("data/eval_gt/a.json", b"data/evaluation/a.json", "docs/experiments/results/a.json"):
                    with self.assertRaises(PermissionError):
                        guard("open", (path,))
        finally:
            os.chdir(old)

    def test_independent_projection_uses_index_centres_without_shift(self):
        k = [[100,0,49.5],[0,100,39.5],[0,0,1]]
        e = np.c_[np.eye(3), np.zeros(3)].tolist()
        data = dict(points_world=[[0,0,2],[.2,.1,2]],corners_xy=[[49.5,39.5],[59.5,44.5]])
        result = independent_observation_score(data,k,e)
        self.assertEqual(result["maximum_px"],0.)
        self.assertTrue(result["positive_depth"])
        data["corners_xy"][0][0] += .5
        self.assertEqual(independent_observation_score(data,k,e)["maximum_px"],.5)

    def test_empty_observations_do_not_become_perfect_score(self):
        result = independent_observation_score(dict(points_world=[],corners_xy=[]),np.eye(3),np.eye(4))
        self.assertIsNone(result["p95_px"])
        self.assertFalse(result["positive_depth"])

    def test_camera_comparison_is_direct_world_without_alignment(self):
        edge = np.array([[100.,0,50.],[0,100.,40.],[0,0,1.]])
        index = edge.copy()
        index[:2,2] -= .5
        truth = dict(K_edge=edge.tolist(),K_index=index.tolist(),world_to_camera_cv=np.eye(4).tolist())
        e = np.c_[np.eye(3),np.array([.1,0,0])]
        predicted = dict(state="validated",K_index=index.tolist(),world_to_camera_cv=e.tolist())
        result = camera_physical(predicted,truth)
        self.assertAlmostEqual(result["center_error_m"],.1)
        self.assertEqual(result["principal_error_px"],0.)
        self.assertFalse(result["alignment_performed"])
        truth["K_index"] = edge.tolist()
        with self.assertRaises(AssertionError):
            camera_physical(predicted,truth)

    def test_missing_camera_error_is_undefined(self):
        edge = np.array([[100.,0,50.],[0,100.,40.],[0,0,1.]])
        index = edge.copy()
        index[:2,2] -= .5
        result = camera_physical(dict(state="unavailable",K_index=None),
            dict(K_edge=edge.tolist(),K_index=index.tolist()))
        self.assertIsNone(result["center_error_m"])
        self.assertEqual(result["state"],"unavailable")


if __name__ == "__main__":
    unittest.main()
