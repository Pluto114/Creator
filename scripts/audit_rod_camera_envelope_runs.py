"""Independent, genuinely pre/post audit of two frozen camera-envelope runs.

The pre stage refuses existing evaluations and blocks all GT/evaluation file opens.
The post stage checks identities and table completeness, not physical score quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_camera_training_sensitivity import describe_identity
from prepare_rod_reference import checked_scene
from run_camera_training_sensitivity import checked as checked_parent
from run_rod_camera_envelope import RUNS, checked, validate_analytic_record
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan
from creator_eval.camera_bundle import validation_plan
from creator_eval.camera_training_groups import build_training_group_plan
from creator_eval.rod_camera_envelope import build_envelope

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ID = "rod-camera-envelope-independent-audit-v1-20260924"
AUDIT = ROOT / ".runtime/experiments" / AUDIT_ID
SELF = Path(__file__).relative_to(ROOT).as_posix()
OUTPUT = ROOT / "docs/experiments/results/2026-09-24-rod-camera-envelope-audit.json"
SLOTS = ["control", *[f"leave_group_{i}" for i in range(4)]]
COUNTS = {"replay": 14, "analytic": 16}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    name = path.relative_to(ROOT.resolve()).as_posix()
    sha = digest(path)
    if expected is not None:
        assert sha == expected, name
    assert name not in hashes or hashes[name] == sha, name
    hashes[name] = sha
    return sha


def source_pairs(run, mapping, hashes):
    for name, sha in mapping.items():
        receipt(ROOT / name, hashes, sha)
        receipt(run / "source_snapshot" / name, hashes, sha)


def unique(values, key):
    result = {key(value): value for value in values}
    assert len(result) == len(values), "Repeated identity"
    return result


def five_slots(values):
    ids = [value["condition_id"] for value in values]
    assert len(ids) == 5 and sorted(ids) == sorted(SLOTS), ids


def plan_check(case, method, analytic=False):
    plan = case["group_plan"]
    rebuilt = build_training_group_plan(case["training"],
        validation_track_ids=plan["validation_track_ids"], view_count=len(case["view_ids"]),
        seed=plan["policy"]["seed"], minimum_tracks=method["optimizer"]["minimum_initial_tracks"],
        minimum_tracks_per_view=method["optimizer"]["minimum_trimmed_tracks_per_view"])
    assert plan == rebuilt
    five_slots(plan["conditions"])
    assert len(case["view_ids"]) == len(set(case["view_ids"])) == 5
    if analytic:
        assert sorted(t["track_id"] for t in case["validation"]) == plan["validation_track_ids"]
        assert case["original_plan"] == validation_plan(case["validation"], case["initial_extrinsics"])
        assert case["all_view_plan"] == all_view_plan(case["validation"], case["initial_extrinsics"],
            training_track_ids=[t["track_id"] for t in case["training"]])
        assert canonical_hash(case["rod_tracks"]) == case["rod_support_sha256"]
        for segment in case["rod_tracks"]:
            assert len(segment["endpoint_tracks"]) == 2
            for endpoint in segment["endpoint_tracks"]:
                views = [o["view"] for o in endpoint["observations"]]
                assert len(views) == len(set(views)) and set(views) <= set(range(5))


def collect_before():
    """Read no GT or physical score; all recorded hashes refer to actual current bytes."""
    hashes, counts, runs = {}, Counter(), {}
    parent, parent_config, parent_manifest = checked_parent()
    source_pairs(parent, parent_config["source_sha256"], hashes)
    for name in ("method_config.json", "inference.json"):
        receipt(parent / name, hashes)
    receipt(ROOT / "data/inputs" / parent_config["run_id"] / "manifest.json", hashes, parent_config["input_sha256"])
    parent_index = read(parent / "inference.json")
    parent_tasks = unique(parent_manifest["tasks"], lambda t: (t["parent"], t["case_id"]))
    parent_records = unique(parent_index["records"], lambda r: (r["parent"], r["case_id"], r["condition"]))
    assert len(parent_tasks) == 7 and len(parent_records) == 35
    for task in parent_tasks.values():
        plan_check(task, task["method"])
    for mode, rid in RUNS.items():
        run, config, manifest = checked(mode)
        source_pairs(run, config["source_sha256"], hashes)
        counts["source_pair_records"] += len(config["source_sha256"])
        for name in ("method_config.json", "source_freeze.json", "inference.json"):
            receipt(run / name, hashes)
        frozen = read(run / "source_freeze.json")
        assert frozen["source_sha256"] == config["source_sha256"]
        assert frozen["evaluation"] == config["evaluation"]
        receipt(ROOT / "data/inputs" / rid / "manifest.json", hashes, config["input_sha256"])
        for name, sha in config["receipts"].items():
            receipt(ROOT / name, hashes, sha)
        index = read(run / "inference.json")
        assert index["state"] == "complete" and index["gt_read_during_inference"] is False
        assert index["config_sha256"] == digest(run / "method_config.json")
        assert index["input_sha256"] == config["input_sha256"]
        families = unique(index["families"], lambda row: row["task_id"])
        assert len(families) == COUNTS[mode]
        records = unique(index["records"], lambda row: (row["case_id"], row["condition"]))
        if mode == "analytic":
            cases = unique(manifest["cases"], lambda row: row["case_id"])
            assert len(cases) == 16
            expected = {(cid, slot) for cid in cases for slot in SLOTS}
            assert set(records) == expected and len(records) == 80
            for case in cases.values():
                plan_check(case, manifest["method"], analytic=True)
                counts["analytic_plans_checked"] += 1
            for key, entry in records.items():
                receipt(run / entry["path"], hashes, entry["sha256"])
                validate_analytic_record(read(run / entry["path"]), entry, cases[key[0]], manifest["method"])
                counts["analytic_records_validated"] += 1
            assert set(families) == set(cases)
        else:
            assert records == {}
            tasks = unique(manifest["tasks"], lambda row: row["task_id"])
            assert set(families) == set(tasks)
        for task_id, entry in families.items():
            receipt(run / entry["path"], hashes, entry["sha256"])
            family = read(run / entry["path"])
            task, envelope = family["task"], family["envelope"]
            assert task["task_id"] == task_id and task["case_id"] == entry["case_id"]
            assert family["gt_read_during_inference"] is False
            five_slots(task["packets"])
            five_slots(envelope["conditions"])
            packets = unique(task["packets"], lambda packet: packet["condition_id"])
            assert envelope["expected_condition_count"] == 5
            assert envelope["expected_condition_ids"] == SLOTS[1:]
            assert envelope == build_envelope(packets["control"], [packets[cid] for cid in SLOTS[1:]],
                initial_extrinsics=task["initial_extrinsics"])
            for slot in envelope["conditions"]:
                assert slot["packet_sha256"] == canonical_hash(packets[slot["condition_id"]])
            if mode == "analytic":
                case = cases[task["case_id"]]
                assert task_id == case["case_id"] and task["method"] == "provided_endpoints"
                assert task["initial_extrinsics"] == case["initial_extrinsics"]
                for cid, packet in packets.items():
                    assert packet == read(run / records[(task_id, cid)]["path"])["packet"]
                    counts["analytic_family_packet_links"] += 1
            else:
                assert task == tasks[task_id]
                old_task = parent_tasks[(task["parent"], task["case_id"])]
                assert task["initial_extrinsics"] == old_task["initial_extrinsics"]
                pool_path = ROOT / old_task["pool_path"]
                receipt(pool_path, hashes, old_task["pool_sha256"])
                frames = read(pool_path)["frames"]
                assert [frame["view_id"] for frame in frames] == old_task["view_ids"]
                for cid, packet in packets.items():
                    old_receipt = parent_records[(task["parent"], task["case_id"], cid)]
                    path = parent / old_receipt["path"]
                    receipt(path, hashes, old_receipt["sha256"])
                    record = read(path)
                    assert record["task_sha256"] == canonical_hash(old_task)
                    assert record["condition"] == next(c for c in old_task["group_plan"]["conditions"] if c["condition_id"] == cid)
                    assert not record["gt_read_during_inference"]
                    rod = next((r for r in record["rods"] if r["method"] == task["method"]), None)
                    selected = describe_identity(rod, old_task, frames)["selected"] if rod else None
                    support = None if selected is None else dict(
                        assignment_sha256=canonical_hash({k: selected[k] for k in ("hypothesis_assignments", "finite_assignments")}),
                        source_rows_sha256=selected["source_rows_sha256"], source_row_count=selected["source_row_count"],
                        pool_sha256=old_task["pool_sha256"])
                    expected_packet = dict(condition_id=cid, intrinsics=record["result"].get("intrinsics"),
                        extrinsics=record["result"].get("extrinsics"), camera_decision=record["decision"],
                        identity=rod["identity"] if rod else dict(state="missing", segments=[]), support=support,
                        parent_record_sha256=old_receipt["sha256"])
                    assert packet == expected_packet
                    counts["replay_parent_packet_links"] += 1
            counts[mode + "_families"] += 1
            counts[mode + "_slots"] += len(packets)
        record_paths = sorted(str(path.relative_to(run).as_posix()) for path in (run / "records").glob("*.json"))
        expected_paths = sorted([row["path"] for row in index["records"]] + [row["path"] for row in index["families"]])
        assert record_paths == expected_paths
        runs[mode] = dict(run_id=rid, inference_sha256=digest(run / "inference.json"), record_paths=record_paths)
    assert counts["analytic_family_packet_links"] == counts["analytic_records_validated"] == 80
    assert counts["replay_parent_packet_links"] == 70
    for name, sha in hashes.items():
        assert digest(ROOT / name) == sha, "Changed during audit collection: " + name
    return dict(receipts=hashes, counts=dict(counts), runs=runs)


def freeze():
    raw = (ROOT / SELF).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
    assert subprocess.check_output(command + ["--no-filters", SELF], cwd=ROOT) == subprocess.check_output(command + ["--path=" + SELF, SELF], cwd=ROOT)
    AUDIT.mkdir(parents=True, exist_ok=False)
    destination = AUDIT / "source_snapshot" / SELF
    destination.parent.mkdir(parents=True)
    shutil.copy2(ROOT / SELF, destination)
    write(AUDIT / "frozen.json", dict(audit_id=AUDIT_ID, created_at_utc=now(), audit_source=SELF,
        source_sha256=digest(ROOT / SELF), run_ids=RUNS, pre_not_yet_executed=True))
    print("AUDIT_FROZEN", AUDIT_ID, digest(ROOT / SELF), flush=True)


def checked_self():
    frozen = read(AUDIT / "frozen.json")
    assert frozen["audit_id"] == AUDIT_ID and frozen["run_ids"] == RUNS
    assert frozen["source_sha256"] == digest(ROOT / SELF) == digest(AUDIT / "source_snapshot" / SELF)
    return frozen


def pre():
    frozen = checked_self()
    assert not (AUDIT / "before.json").exists(), "Preserve existing before snapshot"
    absent = []
    for mode, rid in RUNS.items():
        for path in (ROOT / "data/evaluation" / rid,
                     ROOT / "docs/experiments/results" / f"2026-09-24-{mode}-rod-camera-envelope.json"):
            assert not path.exists(), "Pre audit must precede evaluation: " + str(path)
            absent.append(path.relative_to(ROOT).as_posix())
    sys.addaudithook(reject_truth_open)
    before = collect_before()
    before.update(audit_id=AUDIT_ID, captured_at_utc=now(), state="pre_passed", evaluation_paths_absent=absent,
        gt_read=False, physical_scores_read=False, truth_read_tripwire_enabled=True, audit_frozen_sha256=digest(AUDIT / "frozen.json"),
        audit_source_sha256=frozen["source_sha256"])
    write(AUDIT / "before.json", before)
    print("PRE_PASSED", json.dumps(before["counts"]), digest(AUDIT / "before.json"), flush=True)


def post():
    frozen = checked_self()
    before_path = AUDIT / "before.json"
    before_sha = digest(before_path)
    before = read(before_path)
    assert before["state"] == "pre_passed" and before["gt_read"] is False and before["physical_scores_read"] is False
    assert before["audit_frozen_sha256"] == digest(AUDIT / "frozen.json")
    for name, sha in before["receipts"].items():
        assert digest(ROOT / name) == sha, "Changed after pre: " + name
    # 再跑输入链检查可以发现新增/删掉记录；这里仍不重新估计相机或物理分数。
    current = collect_before()
    for key in ("receipts", "counts", "runs"):
        assert current[key] == before[key], key
    counts, identities, summary_hashes = Counter(), [], {}
    for mode, rid in RUNS.items():
        run, config, manifest = checked(mode)
        shared = ROOT / "docs/experiments/results" / f"2026-09-24-{mode}-rod-camera-envelope.json"
        runtime = ROOT / "data/evaluation" / rid / "summary.json"
        assert shared.read_bytes() == runtime.read_bytes()
        summary = read(shared)
        assert summary["state"] == "complete" and summary["run_id"] == rid and summary["mode"] == mode
        assert summary["config"] == config and summary["inference_sha256"] == digest(run / "inference.json")
        assert not summary["gt_read_during_inference"] and summary["no_automatic_promotion"]
        index = read(run / "inference.json")
        families = unique(index["families"], lambda item: item["task_id"])
        rows = unique(summary["rows"], lambda row: row["task_id"])
        assert set(rows) == set(families) and len(rows) == COUNTS[mode]
        if mode == "analytic":
            truth_root = ROOT / "data/eval_gt" / rid
            truth_frozen = read(truth_root / "frozen.json")
            assert truth_frozen["input_sha256"] == config["input_sha256"]
            assert truth_frozen["source_freeze_sha256"] == digest(run / "source_freeze.json")
            assert truth_frozen["truth_sha256"] == digest(truth_root / "manifest.json")
            truth = read(truth_root / "manifest.json")
            assert truth["dataset_id"] == manifest["dataset_id"] and truth["schema_version"] == manifest["schema_version"]
            cases = unique(manifest["cases"], lambda c: c["case_id"])
            truth_cases = unique(truth["cases"], lambda c: c["case_id"])
            assert set(cases) == set(truth_cases) == set(rows)
            assert summary["truth_receipts"][str(truth_root.relative_to(ROOT))] == truth_frozen
            for cid, case in cases.items():
                cameras = truth_cases[cid]["cameras"]
                assert case["view_ids"] == [c["view_id"] for c in cameras]
                assert all(camera["size_wh"] == case["size_wh"] for camera in cameras)
                identities.append(dict(mode=mode, case_id=cid, view_ids=case["view_ids"],
                    input_sha256=config["input_sha256"], truth_sha256=truth_frozen["truth_sha256"]))
                counts["analytic_case_view_links"] += 1
        for task_id, row in rows.items():
            family = read(run / families[task_id]["path"])
            task = family["task"]
            assert row["case_id"] == task["case_id"] and row["parent"] == task.get("parent") and row["method"] == task["method"]
            assert row["envelope"] == family["envelope"]
            physical = row["physical"]
            five_slots(physical["rows"])
            assert [r["condition_id"] for r in physical["rows"]] == [s["condition_id"] for s in family["envelope"]["conditions"]]
            if physical["state"] == "scored":
                assert physical["alignment_reused_for_all_conditions"] is True
            else:
                assert physical["alignment"] is None and all(r["metrics"] is None and r["reason"] for r in physical["rows"])
            counts[mode + "_physical_rows"] += len(physical["rows"])
            counts[mode + "_public_families"] += 1
            if mode == "replay":
                parent = ROOT / ".runtime/experiments" / task["parent"]
                parent_frozen = read(parent / "prepared.json")
                assert digest(parent / "protocol.json") == parent_frozen["protocol_sha256"]
                scene_run, _, truth_root, scene, clean = checked_scene(read(parent / "protocol.json")["run_id"])
                assert digest(scene_run / "prepared.json") == parent_frozen["scene_prepared_sha256"]
                truth = read(truth_root / "manifest.json")
                assert digest(truth_root / "manifest.json") == scene["truth_sha256"]
                assert truth["input_sha256"] == scene["input_sha256"] == parent_frozen["scene_input_sha256"]
                clean_cases = unique(clean["cases"], lambda c: c["case_id"])
                truth_cases = unique(truth["cases"], lambda c: c["case_id"])
                assert set(clean_cases) == set(truth_cases)
                case, target = clean_cases[task["case_id"]], truth_cases[task["case_id"]]
                view_ids = [f["view_id"] for f in case["frames"]]
                assert view_ids == [f["view_id"] for f in target["frames"]]
                assert len(case["frames"]) == len(target["cameras"]) == 5
                # 旧相机对象没有view_id，按冻结的render frame绑定，并核对相机内容身份。
                artifact_index = read(truth_root / "artifact_hashes.json")
                assert digest(truth_root / "artifact_hashes.json") == scene["truth_artifacts_sha256"]
                assert digest(truth_root / "render_manifest.json") == artifact_index["render_manifest.json"]
                rendered = unique(read(truth_root / "render_manifest.json")["cases"], lambda c: c["case_id"])[task["case_id"]]
                assert view_ids == [f["frame_id"] for f in rendered["frames"]]
                assert [canonical_hash(f["camera"]) for f in rendered["frames"]] == [canonical_hash(c) for c in target["cameras"]]
                assert all(frame["size_wh"] == camera["size_wh"] for frame, camera in zip(case["frames"], target["cameras"]))
                _, _, sensitivity_inputs = checked_parent()
                source_task = next(t for t in sensitivity_inputs["tasks"] if (t["parent"], t["case_id"]) == (task["parent"], task["case_id"]))
                assert view_ids == source_task["view_ids"]
                expected = dict(scene_run_id=read(parent / "protocol.json")["run_id"],
                    scene_prepared_sha256=digest(scene_run / "prepared.json"), truth_sha256=scene["truth_sha256"], scene_input_sha256=scene["input_sha256"])
                assert summary["truth_receipts"][task["parent"]] == expected
                identities.append(dict(mode=mode, task_id=task_id, case_id=task["case_id"], view_ids=view_ids,
                    input_sha256=scene["input_sha256"], truth_sha256=scene["truth_sha256"]))
                counts["replay_case_view_links"] += 1
        summary_hashes[mode] = dict(public_sha256=digest(shared), runtime_sha256=digest(runtime))
    assert counts["replay_physical_rows"] == 70 and counts["analytic_physical_rows"] == 80
    assert digest(before_path) == before_sha
    result = dict(audit_id=AUDIT_ID, state="passed", pre_captured_at_utc=before["captured_at_utc"], post_captured_at_utc=now(),
        before_sha256=before_sha, audit_source_sha256=frozen["source_sha256"], before_receipt_count=len(before["receipts"]),
        all_before_hashes_unchanged=True, pre_gt_read=False, pre_physical_scores_read=False,
        pre_counts=before["counts"], post_counts=dict(counts), runs=RUNS, summary_sha256=summary_hashes, truth_identity_links=identities,
        scope="Independent pre/post provenance and table-completeness audit. No physical score recomputation, confidence-interval claim or accuracy qualification.")
    write(AUDIT / "after.json", result)
    write(OUTPUT, result)
    print("POST_PASSED", json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pre", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {"freeze": freeze, "pre": pre, "post": post}[args.stage]()
