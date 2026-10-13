"""Small, mock-only entry contracts for the fresh G1 pipeline."""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_g1_end_to_end as runner  # noqa: E402


class EndToEndContracts(unittest.TestCase):
    def test_run_id_cannot_escape_experiment_directory(self):
        self.assertEqual(runner.run_path("g1-e2e-20261013-001"),
                         ROOT / ".runtime/experiments/g1-e2e-20261013-001")
        for name in ("", ".", "..", "../outside", "..\\outside", "x/y", "_bad", "C:\\outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.run_path(name)

    def test_array_hash_ignores_archive_time_but_preserves_values_shape_dtype(self):
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        payload = io.BytesIO()
        np.save(payload, value, allow_pickle=False)
        archives, hashes = [], []
        for year in (2025, 2026):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr(zipfile.ZipInfo("depth.npy", (year, 1, 1, 0, 0, 0)), payload.getvalue())
            archives.append(stream.getvalue())
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as native:
                hashes.append(runner.array_hash(native["depth"]))
        self.assertNotEqual(archives[0], archives[1])
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[0], runner.array_hash(np.asfortranarray(value)))
        for changed in (value + 1, value.reshape(3, 2), value.astype(np.float64)):
            with self.subTest(shape=changed.shape, dtype=changed.dtype):
                self.assertNotEqual(hashes[0], runner.array_hash(changed))

    def test_existing_worker_refuses_before_rgb_gpu_or_legacy_pipeline(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "fresh-run"
            folder = run / "repeat-0/chair01"
            folder.mkdir(parents=True)
            runner.write(run / "plan.json", dict(run_id=run.name))
            runner.write(folder / "keep.json", dict(previous_attempt=True))
            before = (folder / "keep.json").read_bytes()
            with patch.object(runner.sys, "addaudithook"), \
                    patch.object(runner.rgb, "infer_case") as infer, \
                    patch.object(runner.smoke_da3, "run") as gpu, \
                    patch.object(runner.point, "depth") as old_depth, \
                    patch.object(runner.point, "compose_case") as old_compose, \
                    self.assertRaises(FileExistsError):
                runner.worker(run, 0, "chair01")
            for mocked in (infer, gpu, old_depth, old_compose):
                mocked.assert_not_called()
            self.assertEqual(before, (folder / "keep.json").read_bytes())
            self.assertFalse((folder / "started.json").exists())

    def test_incomplete_or_duplicate_normal_inventory_cannot_read_truth(self):
        run = ROOT / ".runtime/experiments/mock-e2e-contract"
        rows = [dict(repeat=r, case_id=c, variant=a, fraction=s)
                for r in runner.REPEATS for c in runner.CASES for a in runner.ARMS for s in runner.SCALES]
        self.assertEqual(len(rows), 40)
        for changed in (rows[:-1], rows[:-1] + [rows[0]]):
            documents = {run / "manifest.json": dict(normal_sha256="normal-sha"),
                         run / "normal.json": dict(rows=changed, gt_read=False)}

            def read_normal_only(path):
                if path not in documents:
                    raise AssertionError("Unexpected read before inventory accepted: " + str(path))
                return documents[path]

            with self.subTest(row_count=len(changed)), \
                    patch.object(runner, "read", side_effect=read_normal_only) as read, \
                    patch.object(runner, "digest", return_value="normal-sha") as digest, \
                    self.assertRaisesRegex(ValueError, "40-row normal inventory"):
                runner.evaluate(run)
            self.assertEqual([call.args[0] for call in read.call_args_list], list(documents))
            digest.assert_called_once_with(run / "normal.json")

    def test_worker_exception_keeps_failure_record_and_reraises(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "fresh-run"
            run.mkdir()
            runner.write(run / "plan.json", dict(run_id=run.name, direct_hashes={},
                cases=[dict(case_id="chair01", frames=[])], cad={}, method={}))
            with patch.object(runner.sys, "addaudithook"), \
                    patch.object(runner.rgb, "infer_case", side_effect=RuntimeError("mock calibration failed")), \
                    patch.object(runner.smoke_da3, "run") as gpu, \
                    self.assertRaisesRegex(RuntimeError, "mock calibration failed"):
                runner.worker(run, 0, "chair01")
            gpu.assert_not_called()
            folder = run / "repeat-0/chair01"
            failure = runner.read(folder / "failure.json")
            self.assertEqual(failure["stage"], "fresh_rgb_calibration")
            self.assertIn("mock calibration failed", failure["reason"])
            self.assertEqual((failure["repeat"], failure["case_id"]), (0, "chair01"))
            self.assertFalse(failure["gt_read"])
            self.assertTrue((folder / "started.json").is_file())
            self.assertFalse((folder / "normal.json").exists())


if __name__ == "__main__":
    unittest.main()
