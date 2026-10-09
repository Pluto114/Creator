"""Independent per-chain boolean oracle for mandatory foreground observations."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.rgb_required_view_support import (  # noqa: E402
    consistent_required_view_mask,
    required_stroke_support_mask,
)
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_rgb_stroke_chain_support import marks  # noqa: E402


class RequiredViewsContracts(unittest.TestCase):
    def test_packed_matches_independent_boolean_oracle_at_boundaries(self):
        rng = np.random.default_rng(934)
        for count in (0, 1, 7, 8, 9, 65, 4099):
            arrays = [rng.random((count, width)) < .4 for width in (2, 4, 0, 3, 2)]
            assignments = [[tuple(np.flatnonzero(rng.random(a.shape[1]) < .5)) for a in arrays] for _ in range(20)]
            for required in ((), (0, 3), (0, 1, 2, 3, 4)):
                for minimum in (1, 3, 5):
                    expected = np.zeros(count, bool)
                    for assignment in assignments:
                        per_view = [a[:, ids].any(axis=1) if ids else np.zeros(count, bool)
                                    for a, ids in zip(arrays, assignment)]
                        eligible = np.sum(per_view, axis=0) >= minimum
                        for vi in required:
                            eligible &= per_view[vi]
                        expected |= eligible
                    np.testing.assert_array_equal(
                        consistent_required_view_mask(arrays, assignments, required, minimum), expected)

    def test_three_other_views_cannot_bypass_two_required_views(self):
        arrays = [np.array([[value]], bool) for value in (False, True, True, False, True)]
        assignments = [((0,),)*5]
        self.assertTrue(chain.consistent_chain_mask(arrays, assignments)[0])
        self.assertFalse(consistent_required_view_mask(arrays, assignments, (0, 3))[0])

    def test_required_views_cannot_mix_different_assignments(self):
        arrays = [np.array([[True, False]], bool) for _ in range(5)]
        assignments = [((0,), (0,), (0,), (1,), (0,)), ((1,), (0,), (0,), (0,), (0,))]
        self.assertTrue(chain.consistent_chain_mask(arrays, assignments)[0])
        self.assertFalse(consistent_required_view_mask(arrays, assignments, (0, 3))[0])

    def test_required_views_alone_do_not_replace_minimum_three(self):
        arrays = [np.array([[v]], bool) for v in (True, False, False, True, False)]
        self.assertFalse(consistent_required_view_mask(arrays, [((0,),)*5], (0, 3))[0])

    def test_unknown_required_image_row_is_not_positive(self):
        frames, cameras = fixture()
        selected = marks(frames)
        # Remove a non-mark row in one required view; other three views remain.
        frames[0]["observations"]["rows"][50].update(status="unknown", candidates=[])
        frames[0]["observation_sha256"] = canonical_hash(frames[0]["observations"])
        context = stroke.build_context(frames, cameras, selected)
        point = [[0., .3, 3.]]
        self.assertTrue(chain.chain_support_mask(point, context)[0])
        self.assertFalse(required_stroke_support_mask(point, context)[0])
        self.assertFalse(context["association"]["target_identity_confirmed"])

    def test_translation_and_original_arrays_preserved(self):
        frames, cameras = fixture()
        points = np.array([[0., 0., 3.], [0., .3, 3.], [.2, .3, 3.]])
        original = points.copy()
        before = stroke.build_context(frames, cameras, marks(frames))
        shift = np.array([5., -11., 23.])
        moved = copy.deepcopy(cameras)
        for camera in moved:
            e = np.array(camera["world_to_camera_cv"])
            e[:, 3] -= e[:, :3] @ shift
            camera["world_to_camera_cv"] = e.tolist()
        after = stroke.build_context(frames, moved, marks(frames))
        np.testing.assert_array_equal(required_stroke_support_mask(points, before),
            required_stroke_support_mask(points+shift, after))
        np.testing.assert_array_equal(points, original)
        self.assertEqual(before["sha256"], canonical_hash(before["association"]))

    def test_incomplete_search_preserves_explicit_raw_fallback(self):
        frames, cameras = fixture()
        context = stroke.build_context(frames, cameras, marks(frames), {**chain.DEFAULTS, "maximum_pairs": 1})
        points = np.array([[0., 0., 3.], [1., 0., 3.]])
        np.testing.assert_array_equal(required_stroke_support_mask(points, context), chain.chain_support_mask(points, context))
        self.assertTrue(context["association"]["fallback_to_raw_union"])

    def test_invalid_required_views_rejected(self):
        arrays = [np.ones((2, 1), bool)]*3
        for required in ((-1,), (3,), (True,), (0, 0)):
            with self.assertRaises(ValueError):
                consistent_required_view_mask(arrays, [((0,),)*3], required)


if __name__ == "__main__":
    unittest.main()
