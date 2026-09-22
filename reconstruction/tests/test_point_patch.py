"""Persist/apply/withdraw contracts; these checks do not claim geometric accuracy."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from creator_recon.domain.point_patch import (
    compose,
    content_hash,
    load_bundle,
    load_snapshot,
    open_candidate_view,
    write_candidate_view,
    write_patch,
    write_snapshot,
)


class PointPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.frames = [dict(frame_id="view-a", image_sha256="a" * 64, prediction_size_wh=[4, 3])]
        self.points = np.arange(18, dtype=float).reshape(6, 3)
        self.ids = np.array([[0, r, c] for r in range(2) for c in range(3)], dtype=np.uint32)
        self.snapshot_id = self.snapshot(self.base)

    def snapshot(self, path, **changes):
        args = dict(frames=self.frames, world_frame_id="prediction:one", length_unit="reconstruction_unit", source_prediction_sha256="b" * 64, point_policy="all valid model pixels")
        args.update(changes)
        return write_snapshot(path, self.points, self.ids, **args)

    def patch(self, path, *, changed=True, suppressed=None, evidence=None):
        segments = np.array([[[0., 0., 0.], [0., 0., 2.]]]) if changed else np.empty((0, 2, 3))
        suppression = np.empty((0, 3), np.uint32) if suppressed is None else suppressed
        if evidence is None:
            evidence = [dict(decision="accept", line_ids=["line-0000"], suppression_range=[0, len(suppression)] if len(suppression) else None, view_ids=["view-a"], source_sha256="c" * 64, note="Analytic pipeline fixture, not research evidence")] if changed else []
        return write_patch(path, self.base, segments, suppression, selection_sha256="d" * 64,
                           method=dict(id="control", version="1", config_sha256="e" * 64, seed=0), evidence=evidence, unresolved=[] if changed else ["insufficient geometry"])

    def test_apply_withdraw_reopen_preserves_exact_source_identity(self):
        patch = self.root / "patch"
        self.patch(patch, suppressed=self.ids[[1, 4]])
        candidate = compose(self.base, patch)
        np.testing.assert_array_equal(candidate["points"], self.points[[0, 2, 3, 5]])
        np.testing.assert_array_equal(candidate["point_ids"], self.ids[[0, 2, 3, 5]])
        self.assertEqual(candidate["segments"].shape, (1, 2, 3))
        for _ in range(2):
            restored = compose(self.base, patch, enabled=False)
            np.testing.assert_array_equal(restored["points"], self.points)
            np.testing.assert_array_equal(restored["point_ids"], self.ids)
            self.assertEqual(restored["segments"].shape, (0, 2, 3))
        self.assertEqual(load_snapshot(self.base)[0]["content_id"], self.snapshot_id)

    def test_saved_enabled_and_disabled_views_reopen_without_losing_identity(self):
        patch = self.root / "patch"
        self.patch(patch, suppressed=self.ids[[1, 4]])
        for enabled in (True, False):
            path = self.root / f"view-{enabled}.json"
            write_candidate_view(path, self.base, patch, enabled=enabled)
            saved = open_candidate_view(path)
            direct = compose(self.base, patch, enabled=enabled)
            for key in ("points", "point_ids", "segments"):
                np.testing.assert_array_equal(saved[key], direct[key])
        stale = self.root / "other"
        self.snapshot(stale, source_prediction_sha256="f" * 64)
        path = self.root / "view-True.json"
        body = json.loads(path.read_text())
        body["references"]["snapshot"] = "other"
        body["content_id"] = content_hash({k: v for k, v in body.items() if k != "content_id"})
        path.write_text(json.dumps(body))
        with self.assertRaises(ValueError):
            open_candidate_view(path)

    def test_invalid_toggle_and_unreadable_reference_name_reject_before_save(self):
        for invalid in ("false", 0, 1, None):
            with self.subTest(enabled=invalid), self.assertRaises(ValueError):
                compose(self.base, enabled=invalid)
        other = self.root / "base.with.dot"
        self.snapshot(other)
        output = self.root / "invalid-view.json"
        with self.assertRaises(ValueError):
            write_candidate_view(output, other)
        self.assertFalse(output.exists())

    def test_no_change_equals_the_complete_base(self):
        patch = self.root / "none"
        self.patch(patch, changed=False)
        candidate = compose(self.base, patch)
        np.testing.assert_array_equal(candidate["points"], self.points)
        self.assertEqual(load_bundle(patch, "curve_patch")[0]["metadata"]["outcome"], "no_supported_change")

    def test_content_identity_stable_and_different_source_rejects_stale_patch(self):
        other = self.root / "other"
        self.assertEqual(self.snapshot(other), self.snapshot_id)
        stale = self.root / "stale"
        self.snapshot(stale, source_prediction_sha256="f" * 64)
        patch = self.root / "patch"
        self.patch(patch)
        with self.assertRaises(ValueError):
            compose(stale, patch)

    def test_missing_or_rejected_evidence_cannot_authorize_addition(self):
        for label, evidence in (("missing", []), ("rejected", [dict(decision="reject", line_ids=["line-0000"], suppression_range=None, view_ids=["view-a"], source_sha256="c" * 64, note="no")])):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.patch(self.root / label, evidence=evidence)
            self.assertFalse((self.root / label).exists())

    def test_absent_unsorted_or_duplicate_source_ids_reject(self):
        for label, ids in (("absent", np.array([[0, 2, 0]], np.uint32)), ("unsorted", self.ids[[2, 1]]), ("duplicate", self.ids[[1, 1]])):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.patch(self.root / label, suppressed=ids)

    def test_raster_bounds_and_nonfinite_geometry_reject_before_publication(self):
        self.frames[0]["prediction_size_wh"] = [1, 1]
        with self.assertRaises(ValueError):
            self.snapshot(self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())
        self.points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.snapshot(self.root / "nan")

    def test_existing_and_partial_bundles_are_not_overwritten_or_loaded(self):
        with self.assertRaises(FileExistsError):
            self.snapshot(self.base)
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "points.npy").write_bytes(b"interrupted")
        with self.assertRaises(FileNotFoundError):
            load_snapshot(partial)

    def test_tampered_arrays_and_manifest_do_not_load(self):
        (self.base / "points.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load_snapshot(self.base)
        other = self.root / "other"
        self.snapshot(other)
        path = other / "manifest.json"
        m = json.loads(path.read_text())
        m["metadata"]["world_frame_id"] = "another-world"
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            load_snapshot(other)

    def test_path_escape_rejects_even_with_recomputed_manifest_id(self):
        path = self.base / "manifest.json"
        m = json.loads(path.read_text())
        m["arrays"]["points"]["file"] = "../external.npy"
        m["content_id"] = content_hash({k: v for k, v in m.items() if k != "content_id"})
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            load_snapshot(self.base)

    def test_withdrawing_does_not_hide_a_stale_or_corrupt_patch(self):
        patch = self.root / "patch"
        self.patch(patch)
        (patch / "segments.npy").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            compose(self.base, patch, enabled=False)


if __name__ == "__main__":
    unittest.main()
