"""Audit frozen background and v5 runs, source mappings, baseline parity and real views."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_valley import DEFAULTS as OPEN_DEFAULTS  # noqa: E402
from creator_eval.common_readout_valley import readout as open_readout  # noqa: E402
from creator_eval.common_readout_valley_closed import DEFAULTS as CLOSED_DEFAULTS  # noqa: E402
from creator_eval.common_readout_valley_closed import readout as closed_readout  # noqa: E402
from creator_recon.domain.point_patch import (  # noqa: E402
    load_bundle,
    load_snapshot,
    open_candidate_view,
)

RUNS = {
    "readout-background-challenges-v1-20260924": "2026-09-24-readout-background-challenges.json",
    "readout-valley-development-v1-20260924": "2026-09-24-readout-valley-development.json",
    "readout-valley-closed-development-v1-20260924": "2026-09-24-readout-valley-closed-development.json",
    "g1-point-patch-valley-v1-20260924": "2026-09-24-point-patch-valley.json",
    "g1-point-patch-valley-closed-v1-20260924": "2026-09-24-point-patch-valley-closed.json",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(output):
    counts = dict(frozen_runs=0, frozen_source_records=0, hashed_inference_records=0, exact_public_summaries=0,
        identical_v4_development_rows=0, identical_historical_snapshots=0, identical_historical_patches=0,
        reopened_exact_views=0, verified_model_reader_outputs=0, preserved_semantic_ambiguity_pairs=0)
    for run_id, filename in RUNS.items():
        run = ROOT / ".runtime/experiments" / run_id
        prepared = read(run / "prepared.json")
        for name, expected in prepared["source_sha256"].items():
            saved = run / "source_snapshot" / name
            assert digest(saved) == expected, name
            canonical = hashlib.sha256(saved.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            assert prepared["source_lf_sha256"][name] == canonical, name
            assert digest(ROOT / name) in (expected, canonical), name
            counts["frozen_source_records"] += 1
        assert digest(ROOT / "data/inputs" / run_id / "manifest.json") == prepared["input_sha256"]
        assert digest(run / "protocol.json") == prepared["protocol_sha256"]
        inference = read(run / "inference.json")
        assert inference["state"] == "complete" and inference["gt_read_during_inference"] is False
        for item in inference["records"]:
            assert digest(run / item["path"]) == item["sha256"]
            counts["hashed_inference_records"] += 1
        shared = ROOT / "docs/experiments/results" / filename
        assert digest(shared) == digest(ROOT / "data/evaluation" / run_id / "summary.json")
        report = read(shared)
        counts["exact_public_summaries"] += 1
        if run_id.startswith("readout-"):
            for pair in report["ambiguity_pairs"]:
                assert pair["input_arrays_identical"] and pair["both_reader_outputs_identical"] and pair["qualified"] is None
                counts["preserved_semantic_ambiguity_pairs"] += 1
            if "baseline_parity" in report:
                protocol = read(run / "protocol.json")
                for dataset in protocol["datasets"]:
                    prior = {row["case_id"]: row for row in read(ROOT / dataset["source_results_file"])["rows"] if row["reader"] == "robust_sections"}
                    for row in report["rows"]:
                        if row["reader"] == "robust_sections" and row["source_dataset"] == dataset["name"]:
                            for key in ("state", "segments", "metrics", "gap", "diagnostic_qualified"):
                                assert row[key] == prior[row["original_case_id"]][key]
                            counts["identical_v4_development_rows"] += 1
        else:
            old_run = ROOT / ".runtime/experiments/g1-point-patch-sections-v1-20260923r2"
            expected_scope = "persistent_valley_closed_band_section_diagnostic_not_general_skeleton_or_graph_topology" if "-closed-" in run_id else "persistent_valley_section_diagnostic_not_general_skeleton_or_graph_topology"
            for item in inference["records"]:
                result = read(run / item["path"])
                destination, old_destination = (run / item["path"]).parent, (old_run / item["path"]).parent
                manifest, base = load_snapshot(destination / "base")
                previous, _ = load_snapshot(old_destination / "base")
                assert manifest["content_id"] == previous["content_id"]
                counts["identical_historical_snapshots"] += 1
                patch, arrays = load_bundle(destination / "patch", "curve_patch")
                old_patch, old_arrays = load_bundle(old_destination / "patch", "curve_patch")
                assert patch["content_id"] == old_patch["content_id"]
                for key in ("segments", "suppressed_ids"):
                    np.testing.assert_array_equal(arrays[key], old_arrays[key])
                counts["identical_historical_patches"] += 1
                for label in ("enabled", "withdrawn"):
                    view = open_candidate_view(destination / ("view-" + label + ".json"))
                    for key in ("points", "point_ids"):
                        assert view[key].tobytes() == base[key].tobytes()
                    np.testing.assert_array_equal(view["segments"], arrays["segments"] if label == "enabled" else np.empty((0, 2, 3)))
                    counts["reopened_exact_views"] += 1
                for graph in result["graphs"]:
                    assert digest(run / graph["path"]) == graph["sha256"]
                    reader = read(run / graph["path"])
                    assert reader["scope"] == expected_scope
                    defaults = CLOSED_DEFAULTS if "-closed-" in run_id else OPEN_DEFAULTS
                    assert reader["config"] == {**defaults, "voxel_size": result["camera_span"] * graph["fraction"]}
                    counts["verified_model_reader_outputs"] += 1
        counts["frozen_runs"] += 1
    assert OPEN_DEFAULTS == CLOSED_DEFAULTS
    voxel = 1 / 32
    x, z = np.meshgrid((np.arange(8) - 3.5) * voxel, (np.arange(40) + .5) * voxel, indexing="ij")
    plane = np.c_[x.ravel(), np.full(x.size, voxel / 2), z.ravel()]
    original = open_readout(plane, np.empty((0, 2, 3)), {"voxel_size": voxel})
    corrected = closed_readout(plane, np.empty((0, 2, 3)), {"voxel_size": voxel})
    assert original["components"] == 2 and corrected["components"] == 0
    assert counts["frozen_runs"] == 5 and counts["identical_v4_development_rows"] == 468
    assert counts["identical_historical_snapshots"] == counts["identical_historical_patches"] == 14
    assert counts["reopened_exact_views"] == 28 and counts["verified_model_reader_outputs"] == 56
    assert counts["preserved_semantic_ambiguity_pairs"] == 18
    command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
    new_sources = [Path(__file__).relative_to(ROOT).as_posix()]
    for run_id in RUNS:
        prepared = read(ROOT / ".runtime/experiments" / run_id / "prepared.json")
        new_sources += [name for name in prepared["source_sha256"] if any(tag in name for tag in ("valley", "background_controls", "background_challenges", "common_readout_background"))]
    for name in set(new_sources):
        raw = (ROOT / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n"), name
        assert subprocess.check_output(command + ["--no-filters", name], cwd=ROOT) == subprocess.check_output(command + ["--path=" + name, name], cwd=ROOT), name
    report = dict(state="passed", counts=counts, runs=RUNS, git_safe_new_source_count=len(set(new_sources)),
        policy_thresholds_identical_between_v5_revisions=True, exact_grid_original_false_lines=2, exact_grid_corrected_false_lines=0,
        audit_source_sha256=digest(__file__), qualification="not_established; all v5 comparisons are development")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-24-readout-valley-audit.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit(args.output)
