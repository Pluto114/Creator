"""Independent frozen-byte, pairing and completeness audit; no model fitting rerun."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_model_controls import generate  # noqa: E402

RUN_ID = "section-model-diagnostic-v1-20260926"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit():
    run = ROOT / ".runtime/experiments" / RUN_ID
    inputs = ROOT / "data/inputs" / RUN_ID
    gt = ROOT / "data/eval_gt" / RUN_ID
    frozen = read(run / "prepared.json")
    command = ["git", "-c", "safe.directory="+ROOT.as_posix(), "hash-object"]
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT/name) == digest(run/"source_snapshot"/name) == sha
        raw = (ROOT/name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        assert subprocess.check_output(command+["--no-filters",name],cwd=ROOT) == subprocess.check_output(command+["--path="+name,name],cwd=ROOT)
    assert digest(run/"source_freeze.json") == frozen["source_freeze_sha256"]
    assert read(run/"source_freeze.json")["source_sha256"] == frozen["source_sha256"]
    assert digest(run/"protocol.json") == frozen["protocol_sha256"]
    assert digest(inputs/"manifest.json") == frozen["input_sha256"]
    assert digest(gt/"manifest.json") == frozen["truth_sha256"]
    manifest, truth = read(inputs/"manifest.json"), read(gt/"manifest.json")
    assert truth["input_sha256"] == frozen["input_sha256"]
    regenerated, labels = generate(read(run/"protocol.json"))
    assert labels == truth["cases"]
    cases = {case["case_id"]:case for case in manifest["cases"]}
    assert len(cases) == len(manifest["cases"]) == len(regenerated) == 64
    for generated in regenerated:
        case = cases[generated["case_id"]]
        assert set(case) == {"case_id","points_file","points_sha256","voxel_size"}
        assert digest(inputs/case["points_file"]) == case["points_sha256"]
        np.testing.assert_array_equal(np.load(inputs/case["points_file"],allow_pickle=False),generated["points"])
        assert case["voxel_size"] == generated["voxel_size"]
    index = read(run/"inference.json")
    assert index["state"] == "complete" and index["gt_read_during_inference"] is False and index["truth_read_tripwire_enabled"]
    assert index["source_sha256"] == frozen["source_sha256"] and index["input_sha256"] == frozen["input_sha256"]
    assert len(index["records"]) == 64 and {r["case_id"] for r in index["records"]} == set(cases)
    shared = ROOT / "docs/experiments/results/2026-09-26-section-model-diagnostic.json"
    assert shared.read_bytes() == (ROOT/"data/evaluation"/RUN_ID/"summary.json").read_bytes()
    report = read(shared)
    assert report["inference_sha256"] == digest(run/"inference.json")
    assert report["qualification"] is None and report["emits_axis"] is False
    rows = {(r["case_id"],r["representation"],r["train_slices"][0],r["model"]):r for r in report["rows"]}
    assert len(rows) == len(report["rows"]) == 768
    count = 0
    for entry in index["records"]:
        assert digest(run/entry["path"]) == entry["sha256"]
        assert entry["input_sha256"] == cases[entry["case_id"]]["points_sha256"]
        output = read(run/entry["path"])
        assert output["emits_axis"] is False and output["policy"] == manifest["method"]
        assert len(output["representations"]) == 2
        for representation in output["representations"]:
            assert len(representation["rows"]) == 6
            for model in representation["rows"]:
                row = rows[(entry["case_id"],representation["name"],model["train_slices"][0],model["model"])]
                assert row["accepted_model"] is False
                for key in ("fit","heldout","train_slices","test_slices"):
                    assert model[key] == row[key]
                assert set(model["train_slices"]).isdisjoint(model["test_slices"])
                assert sorted(model["train_slices"]+model["test_slices"]) == list(range(6))
                count += 1
    assert count == 768 and len(report["comparisons"]) == 256
    assert all(r["selected_model"] is None for r in report["comparisons"])
    result = dict(run_id=RUN_ID,state="passed",frozen_sources=len(frozen["source_sha256"]),
        regenerated_identical_inputs=64,verified_inference_records=64,verified_model_fold_rows=768,
        representations=128,unselected_comparisons=256,public_runtime_bytes_identical=True,
        gt_read_during_inference=False,source_sha256=frozen["source_sha256"],
        audit_source_sha256=digest(__file__),summary_sha256=digest(shared),
        scope="Provenance, paired generation and complete score preservation; no physical classification or model-accuracy qualification")
    destination = ROOT / "docs/experiments/results/2026-09-26-section-model-diagnostic-audit.json"
    with destination.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit()
