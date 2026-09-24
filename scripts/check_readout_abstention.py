"""Read-only audit of frozen abstention replay and unchanged source point/patch views."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_abstention import DEFAULTS  # noqa: E402
from creator_eval.common_readout_valley_closed import DEFAULTS as CLOSED_DEFAULTS  # noqa: E402
from creator_recon.domain.point_patch import (  # noqa: E402
    load_bundle,
    load_snapshot,
    open_candidate_view,
)

RUNS = {
    "readout-abstention-development-v1-20260924": "2026-09-24-readout-abstention-development.json",
    "g1-point-patch-abstention-v1-20260924": "2026-09-24-point-patch-abstention.json",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolution_contract(output):
    assert output["state"] == "complete"
    unresolved = output["unresolved_components"]
    assert output["unresolved_component_count"] == len(unresolved)
    has_curve = len(output["segments"]) > 0
    expected = ("partial" if has_curve else "unresolved") if unresolved else ("resolved" if has_curve else "no_supported_curve")
    assert output["resolution_state"] == expected
    ids = [item["component_id"] for item in unresolved]
    assert len(ids) == len(set(ids))
    checks = {item["component_id"]: item for item in output["split_valley_checks"]}
    for item in unresolved:
        assert item["state"] == "unresolved" and item["proposed_parts"] == 2 and item["emitted_segments"] == 0
        assert item["valley"] == checks[item["component_id"]]
        assert item["valley"]["decision"] == "abstain_unresolved"
        assert item["reason"] in ("transverse_valley_not_persistent", "supported_split_exceeds_depth_budget")
    assert output["scope"] == "abstaining_component_section_diagnostic_not_general_skeleton_or_graph_topology"


def audit(output):
    counts = Counter()
    reports = {}
    for run_id, filename in RUNS.items():
        run = ROOT / ".runtime/experiments" / run_id
        prepared = read(run / "prepared.json")
        for name, expected in prepared["source_sha256"].items():
            saved = run / "source_snapshot" / name
            assert digest(saved) == expected
            canonical = hashlib.sha256(saved.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            assert prepared["source_lf_sha256"][name] == canonical
            assert digest(ROOT / name) in (expected, canonical)
            counts["frozen_source_records"] += 1
        inputs = ROOT / "data/inputs" / run_id
        assert digest(inputs / "manifest.json") == prepared["input_sha256"]
        assert digest(run / "protocol.json") == prepared["protocol_sha256"]
        manifest = read(inputs / "manifest.json")
        if "cases" in manifest:
            for case in manifest["cases"]:
                assert digest(inputs / case["points_file"]) == case["points_sha256"]
                counts["anonymous_input_files"] += 1
            assert digest(ROOT / "data/eval_gt" / run_id / "truth.json") == prepared["gt_sha256"]
        inference = read(run / "inference.json")
        assert inference["state"] == "complete" and inference["gt_read_during_inference"] is False
        for item in inference["records"]:
            assert digest(run / item["path"]) == item["sha256"]
            counts["hashed_inference_records"] += 1
        shared = ROOT / "docs/experiments/results" / filename
        assert digest(shared) == digest(ROOT / "data/evaluation" / run_id / "summary.json")
        reports[run_id] = read(shared)
        counts["exact_public_summaries"] += 1
        if run_id.startswith("readout-"):
            reference = ROOT / ".runtime/experiments/readout-valley-closed-development-v1-20260924"
            for item in inference["records"]:
                result = read(run / item["path"])
                if item["reader"] == "persistent_valley_closed":
                    assert result == read(reference / item["path"])
                    counts["identical_closed_full_outputs"] += 1
                else:
                    resolution_contract(result)
                    counts["verified_development_resolution_outputs"] += 1
            for pair in reports[run_id]["ambiguity_pairs"]:
                assert pair["input_arrays_identical"] and pair["both_reader_outputs_identical"] and pair["qualified"] is None
                counts["semantic_ambiguity_pairs"] += 1
        else:
            old_run = ROOT / ".runtime/experiments/g1-point-patch-valley-closed-v1-20260924"
            for item in inference["records"]:
                result = read(run / item["path"])
                destination = (run / item["path"]).parent
                prior = (old_run / item["path"]).parent
                snapshot, base = load_snapshot(destination / "base")
                old_snapshot, _ = load_snapshot(prior / "base")
                assert snapshot["content_id"] == old_snapshot["content_id"]
                patch, arrays = load_bundle(destination / "patch", "curve_patch")
                old_patch, old_arrays = load_bundle(prior / "patch", "curve_patch")
                assert patch["content_id"] == old_patch["content_id"]
                for key in ("segments", "suppressed_ids"):
                    np.testing.assert_array_equal(arrays[key], old_arrays[key])
                counts["identical_historical_base_patch_pairs"] += 1
                for label in ("enabled", "withdrawn"):
                    view = open_candidate_view(destination / ("view-" + label + ".json"))
                    for key in ("points", "point_ids"):
                        assert view[key].tobytes() == base[key].tobytes()
                    np.testing.assert_array_equal(view["segments"], arrays["segments"] if label == "enabled" else np.empty((0, 2, 3)))
                    counts["exact_reopened_views"] += 1
                for graph in result["graphs"]:
                    assert digest(run / graph["path"]) == graph["sha256"]
                    reader = read(run / graph["path"])
                    resolution_contract(reader)
                    assert reader["config"] == {**DEFAULTS, "voxel_size": result["camera_span"] * graph["fraction"]}
                    assert reader["resolution_state"] == graph["resolution_state"]
                    assert reader["unresolved_component_count"] == graph["unresolved_component_count"]
                    counts["verified_model_resolution_outputs"] += 1
        counts["frozen_runs"] += 1
    assert DEFAULTS == CLOSED_DEFAULTS
    assert counts["identical_closed_full_outputs"] == counts["verified_development_resolution_outputs"] == 234
    assert counts["identical_historical_base_patch_pairs"] == 7 and counts["exact_reopened_views"] == 14
    assert counts["verified_model_resolution_outputs"] == 28 and counts["semantic_ambiguity_pairs"] == 6
    command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
    files = ["experiments/src/creator_eval/common_readout_abstention.py", "tests/test_common_readout_abstention.py",
             "scripts/run_readout_abstention_development.py", "scripts/run_g1_point_patch_abstention.py",
             "configs/readout_abstention_development_v1.json", "configs/g1_point_patch_abstention_v1.json",
             Path(__file__).relative_to(ROOT).as_posix()]
    for name in files:
        raw = (ROOT / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        assert subprocess.check_output(command + ["--no-filters", name], cwd=ROOT) == subprocess.check_output(command + ["--path=" + name, name], cwd=ROOT)
    development = reports[next(iter(RUNS))]
    groups = []
    for dataset in ("prior198", "background36"):
        for expectation in ("positive", "negative", "unidentifiable"):
            rows = [r for r in development["rows"] if r["reader"] == "abstaining_components" and r["source_dataset"] == dataset and r["expectation"] == expectation]
            if rows:
                groups.append(dict(dataset=dataset, expectation=expectation, total=len(rows),
                    old_geometry_metric_pass=sum(r["diagnostic_qualified"] is True for r in rows),
                    resolution_states=dict(Counter(r["resolution_state"] for r in rows)),
                    abstaining_rows=sum(r["unresolved_component_count"] > 0 for r in rows),
                    metric_pass_with_abstention=sum(r["diagnostic_qualified"] is True and r["unresolved_component_count"] > 0 for r in rows)))
    result = dict(state="passed", counts=dict(counts), runs=RUNS, defaults_unchanged=True,
        development_status_groups=groups, audit_source_sha256=digest(__file__),
        interpretation="Unresolved/partial are not correct background classifications. Legacy no-false-curve metrics are retained separately. No qualification established.")
    with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-24-readout-abstention-audit.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit(args.output)
