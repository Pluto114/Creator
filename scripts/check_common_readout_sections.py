"""Audit frozen section-reader replays and unchanged historical geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_recon.domain.point_patch import (  # noqa: E402
    load_bundle,
    load_snapshot,
    open_candidate_view,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(output):
    analytic_id, point_id = "common-readout-sections-controls-v1-20260923r2", "g1-point-patch-sections-v1-20260923r2"
    counts = dict(frozen_runs=0, frozen_source_files=0, identical_analytic_inputs=0, identical_v3_regression_rows=0,
                  identical_base_snapshots=0, identical_patch_geometry=0, saved_views=0, correct_reader_outputs=0)
    for run_id in (analytic_id, point_id):
        run = ROOT / ".runtime/experiments" / run_id
        prepared = read(run / "prepared.json")
        for name, sha in prepared["source_sha256"].items():
            assert digest(ROOT / name) == sha == digest(run / "source_snapshot" / name), name
            counts["frozen_source_files"] += 1
        assert digest(ROOT / "data/inputs" / run_id / "manifest.json") == prepared["input_sha256"]
        assert digest(run / "protocol.json") == prepared["protocol_sha256"]
        inference = read(run / "inference.json")
        assert inference["state"] == "complete" and inference["gt_read_during_inference"] is False
        for item in inference["records"]:
            assert digest(run / item["path"]) == item["sha256"]
        counts["frozen_runs"] += 1
    old_id = "common-readout-split-controls-v1-20260922"
    old_inputs = read(ROOT / "data/inputs" / old_id / "manifest.json")
    new_inputs = read(ROOT / "data/inputs" / analytic_id / "manifest.json")
    assert old_inputs["cases"] == new_inputs["cases"]
    for case in new_inputs["cases"]:
        assert digest(ROOT / "data/inputs" / analytic_id / case["points_file"]) == case["points_sha256"]
        counts["identical_analytic_inputs"] += 1
    assert digest(ROOT / "data/eval_gt" / analytic_id / "truth.json") == digest(ROOT / "data/eval_gt" / old_id / "truth.json")
    previous = read(ROOT / "docs/experiments/results/2026-09-22-readout-split-controls.json")
    current = read(ROOT / "docs/experiments/results/2026-09-23-readout-sections-controls-r2.json")
    older = {row["case_id"]: row for row in previous["rows"] if row["reader"] == "transverse_split"}
    for row in current["rows"]:
        if row["reader"] != "transverse_split":
            continue
        for key in ("state", "segments", "metrics", "gap", "diagnostic_qualified"):
            assert row[key] == older[row["case_id"]][key], (row["case_id"], key)
        counts["identical_v3_regression_rows"] += 1
    run = ROOT / ".runtime/experiments" / point_id
    old_run = ROOT / ".runtime/experiments/g1-point-patch-split-v1-20260922"
    inference = read(run / "inference.json")
    for record in inference["records"]:
        result = read(run / record["path"])
        destination = (run / record["path"]).parent
        old_destination = old_run / destination.name
        snapshot, base = load_snapshot(destination / "base")
        old_snapshot, _ = load_snapshot(old_destination / "base")
        assert snapshot["content_id"] == old_snapshot["content_id"]
        counts["identical_base_snapshots"] += 1
        _, patch = load_bundle(destination / "patch", "curve_patch")
        _, old_patch = load_bundle(old_destination / "patch", "curve_patch")
        for key in ("segments", "suppressed_ids"):
            np.testing.assert_array_equal(patch[key], old_patch[key])
        counts["identical_patch_geometry"] += 1
        for label in ("enabled", "withdrawn"):
            view = open_candidate_view(destination / ("view-" + label + ".json"))
            for key in ("points", "point_ids"):
                assert view[key].tobytes() == base[key].tobytes()
            expected = patch["segments"] if label == "enabled" else np.empty((0, 2, 3))
            np.testing.assert_array_equal(view["segments"], expected)
            counts["saved_views"] += 1
        for graph in result["graphs"]:
            assert digest(run / graph["path"]) == graph["sha256"]
            reader = read(run / graph["path"])
            assert reader["scope"] == "robust_straight_surface_section_diagnostic_not_general_skeleton_or_graph_topology"
            assert "circle_trials" in reader["config"] and "maximum_transverse_voxels" not in reader["config"]
            counts["correct_reader_outputs"] += 1
    assert counts["identical_analytic_inputs"] == counts["identical_v3_regression_rows"] == 198
    assert counts["identical_base_snapshots"] == counts["identical_patch_geometry"] == 7
    assert counts["saved_views"] == 14 and counts["correct_reader_outputs"] == 28
    first_analytic = read(ROOT / "docs/experiments/results/2026-09-23-readout-sections-controls.json")
    assert current["rows"] == first_analytic["rows"], "r2 analytic semantics changed"
    counts["identical_r1_analytic_rows"] = len(current["rows"])
    first_points = read(ROOT / "docs/experiments/results/2026-09-23-point-patch-sections.json")
    second_points = read(ROOT / "docs/experiments/results/2026-09-23-point-patch-sections-r2.json")
    assert first_points["rows"] == second_points["rows"], "r2 model metric semantics changed"
    counts["identical_r1_model_rows"] = len(second_points["rows"])
    first_run = ROOT / ".runtime/experiments/g1-point-patch-sections-v1-20260923"
    same_geometry = 0
    for record in inference["records"]:
        result = read(run / record["path"])
        first_result = read(first_run / record["path"])
        assert result["snapshot_id"] == first_result["snapshot_id"]
        assert result["patch_id"] == first_result["patch_id"]
        for graph in result["graphs"]:
            assert read(run / graph["path"]) == read(first_run / graph["path"])
            same_geometry += 1
    counts["identical_r1_model_reader_outputs"] = same_geometry
    assert counts["identical_r1_analytic_rows"] == 396 and counts["identical_r1_model_rows"] == 84 and same_geometry == 28
    report = dict(state="passed", counts=counts, analytic_run=analytic_id, point_run=point_id,
                  gt_read_during_inference=False, reader_qualification="not_established", audit_source_sha256=digest(__file__))
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-23-readout-sections-audit-r2.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit(args.output)
