"""Verify the frozen split-reader/camera-transfer extension and actual saved views."""
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

    schemas = {"bundle": Draft202012Validator(read(ROOT / "schemas/pilot/point-patch-0.1.schema.json")),
               "view": Draft202012Validator(read(ROOT / "schemas/pilot/candidate-view-0.1.schema.json"))}
    source_count, schema_count, view_count, projections = 0, 0, 0, []
    for run_id in ("common-readout-split-controls-v1-20260922", "g1-point-patch-split-v1-20260922", "foreground-estimated-pilot-v1-20260922"):
        folder = ROOT / ".runtime/experiments" / run_id
        frozen = read(folder / "prepared.json")
        for name, sha in frozen["source_sha256"].items():
            assert file_hash(ROOT / name) == sha == file_hash(folder / "source_snapshot" / name)
            source_count += 1
        assert file_hash(folder / "protocol.json") == frozen["protocol_sha256"]
        assert file_hash(ROOT / "data/inputs" / run_id / "manifest.json") == frozen["input_sha256"]
        inference = read(folder / "inference.json")
        assert inference["state"] == "complete" and not inference["gt_read_during_inference"]
        for base_path in sorted(folder.glob("*/base")):
            job = base_path.parent
            manifest, base = load_snapshot(base_path)
            schemas["bundle"].validate(manifest)
            schema_count += 1
            for patch in sorted(job.glob("patch*/manifest.json")):
                schemas["bundle"].validate(read(patch))
                schema_count += 1
            views = sorted(job.glob("*-enabled.json")) + sorted(job.glob("*-withdrawn.json"))
            for path in views:
                schemas["view"].validate(read(path))
                candidate = open_candidate_view(path)
                assert candidate["points"].tobytes() == base["points"].tobytes()
                assert candidate["point_ids"].tobytes() == base["point_ids"].tobytes()
                if path.stem.endswith("withdrawn"):
                    assert len(candidate["segments"]) == 0
                schema_count += 1
                view_count += 1
            prediction = job / "prediction/prediction.npz"
            if not prediction.exists():
                prediction = ROOT / read(job / "result.json")["job"]["prediction_path"]
            assert file_hash(prediction) == manifest["metadata"]["source_prediction_sha256"]
            maximum, count = 0., 0
            with np.load(prediction, allow_pickle=False) as native:
                ks = native["intrinsics"].astype(float)
                if manifest["metadata"]["length_unit"] == "meter":
                    ks[:, :2, 2] -= .5
                for frame, (k, camera) in enumerate(zip(ks, native["extrinsics"])):
                    indices = np.flatnonzero(base["point_ids"][:, 0] == frame)
                    indices = indices[np.unique(np.linspace(0, len(indices) - 1, 256).astype(int))]
                    ids = base["point_ids"][indices]
                    xyz = base["points"][indices] @ camera[:, :3].T + camera[:, 3]
                    pixel = xyz @ k.T
                    maximum = max(maximum, float(np.max(np.abs(pixel[:, :2] / pixel[:, 2, None] - ids[:, [2, 1]]))))
                    count += len(ids)
            assert maximum < .002
            projections.append(dict(run_id=run_id, case_id=job.name, sampled_pixels=count, maximum_error_px=maximum,
                                    base_points=len(base["points"]), saved_view_count=len(views)))
    assert len(projections) == 11 and view_count == 50 and schema_count == 86
    result = dict(state="complete", frozen_source_count=source_count, schema_document_count=schema_count,
                  saved_view_count=view_count, projections=projections,
                  scope="native pixel identity and exact unchanged base; this does not validate estimated camera physical accuracy")
    output = ROOT / "docs/experiments/results/2026-09-22-g1-extension-integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("G1_EXTENSION_INTEGRITY", schema_count, "schema files;", view_count, "saved views;", sum(r["sampled_pixels"] for r in projections), "pixel checks")


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
