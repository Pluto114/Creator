"""Audit persisted pilot bundles, frozen sources and native pixel identities."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_recon.domain.point_patch import (  # noqa: E402
    file_hash,
    load_snapshot,
    open_candidate_view,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run():
    from jsonschema import Draft202012Validator

    validators = {"bundle": Draft202012Validator(read(ROOT / "schemas/pilot/point-patch-0.1.schema.json")),
                  "view": Draft202012Validator(read(ROOT / "schemas/pilot/candidate-view-0.1.schema.json"))}
    rows, documents, frozen_sources = [], 0, 0
    for name in ("g1-point-patch-v1-20260922", "g1-point-patch-v1-20260922r2", "g1-point-patch-components-v1-20260922"):
        folder = ROOT / ".runtime/experiments" / name
        frozen = read(folder / "prepared.json")
        for relative, sha in frozen["source_sha256"].items():
            assert file_hash(folder / "source_snapshot" / relative) == sha
            if name != "g1-point-patch-v1-20260922":
                assert file_hash(ROOT / relative) == sha
            frozen_sources += 1
        assert file_hash(folder / "protocol.json") == frozen["protocol_sha256"]
        assert file_hash(ROOT / "data/inputs" / name / "manifest.json") == frozen["input_sha256"]
        if name == "g1-point-patch-v1-20260922":
            continue  # 原始运行按当时源码保留，不能假装它用了今天的边界修复。
        inference = read(folder / "inference.json")
        assert inference["state"] == "complete" and not inference["gt_read_during_inference"]
        assert len(inference["records"]) == 7
        for entry in inference["records"]:
            assert file_hash(folder / entry["path"]) == entry["sha256"]
            record = read(folder / entry["path"])
            job_dir = (folder / entry["path"]).parent
            for relative, kind in (("base/manifest.json", "bundle"), ("patch/manifest.json", "bundle"),
                                   ("view-enabled.json", "view"), ("view-withdrawn.json", "view")):
                validators[kind].validate(read(job_dir / relative))
                documents += 1
            manifest, base = load_snapshot(job_dir / "base")
            active = open_candidate_view(job_dir / "view-enabled.json")
            restored = open_candidate_view(job_dir / "view-withdrawn.json")
            for key in ("points", "point_ids"):
                assert np.array_equal(base[key], active[key])  # 本次回放的抑制列表确实为空。
                assert base[key].tobytes() == restored[key].tobytes()
            assert len(restored["segments"]) == 0
            assert manifest["content_id"] == record["snapshot_id"]
            for graph in record["graphs"]:
                assert file_hash(folder / graph["path"]) == graph["sha256"]

            # 独立投影核验，避免“反投影函数自己证明自己”。检查的是模型像素来源，
            # 不把估计相机的内部自洽说成物理相机正确；Blender GT仍留在评价侧。
            job = record["job"]
            prediction = ROOT / job["prediction_path"]
            assert file_hash(prediction) == job["prediction_sha256"]
            max_pixel, max_relative_z, sampled = 0., 0., 0
            with np.load(prediction, allow_pickle=False) as native:
                k = native["intrinsics"].astype(float)
                if job["camera_mode"] == "oracle_camera":
                    k[:, :2, 2] -= .5
                for frame, camera in enumerate(native["extrinsics"]):
                    indices = np.flatnonzero(base["point_ids"][:, 0] == frame)
                    indices = indices[np.unique(np.linspace(0, len(indices) - 1, 1024).astype(int))]
                    ids = base["point_ids"][indices]
                    xyz = base["points"][indices] @ camera[:, :3].T + camera[:, 3]
                    image = xyz @ k[frame].T
                    xy = image[:, :2] / image[:, 2, None]
                    z = native["depth"][frame, ids[:, 1], ids[:, 2]]
                    max_pixel = max(max_pixel, float(np.abs(xy - ids[:, [2, 1]]).max()))
                    max_relative_z = max(max_relative_z, float(np.max(np.abs(xyz[:, 2] - z) / z)))
                    sampled += len(ids)
            assert max_pixel < .002 and max_relative_z < 5e-6
            rows.append(dict(run_id=name, job_id=job["candidate_job_id"], points=len(base["points"]),
                             projection_samples=sampled, maximum_pixel_error=max_pixel, maximum_relative_depth_error=max_relative_z,
                             rollback_bytes_equal=True, schema_documents=4))
    old = read(ROOT / "docs/experiments/results/2026-09-22-point-patch.json")
    revised = read(ROOT / "docs/experiments/results/2026-09-22-point-patch-r2.json")
    assert old["rows"] == revised["rows"]
    for a, b in zip(old["jobs"], revised["jobs"]):
        assert a["snapshot_id"] == b["snapshot_id"] and a["patch_id"] == b["patch_id"]
        assert a["graphs"] == b["graphs"]
    report = dict(state="complete", schema_document_count=documents, frozen_source_count=frozen_sources,
                  initial_revision_geometry_and_metrics_unchanged=True, rows=rows,
                  scope="native prediction pixel identity and exact rollback, not estimated-camera physical accuracy")
    path = ROOT / "docs/experiments/results/2026-09-22-point-patch-integrity.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print("POINT_PATCH_INTEGRITY", documents, "schema documents;", sum(r["projection_samples"] for r in rows),
          "sampled pixel projections; maximum error", max(r["maximum_pixel_error"] for r in rows))


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
