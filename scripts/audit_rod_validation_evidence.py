"""Independent pre/post audit of the paired rod-evidence experiment.

The pre stage installs the truth tripwire and records actual pre-evaluation
bytes. Only post reads truth, replays generation, and checks physical readouts.
No stage optimizes cameras or changes inference/experiment artifacts.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import run_rod_validation_evidence as target
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_evidence_challenges import generate_paired_challenges

AUDIT_ID = "rod-validation-evidence-audit-v1-20260926"
AUDIT = ROOT / ".runtime/experiments" / AUDIT_ID
OUTPUT = ROOT / "docs/experiments/results/2026-09-26-rod-validation-evidence-audit.json"
EXTRA_SOURCES = {
    "scripts/audit_rod_validation_evidence.py",
    "tests/test_rod_validation_evidence_runner.py",
    "tests/test_camera_evidence_challenges.py",
    "tests/test_rod_validation_evidence.py",
}


def same(left, right, message):
    if canonical_hash(left) != canonical_hash(right):
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_audit():
    frozen = read_json(AUDIT / "prepared.json")
    assert frozen["audit_run_id"] == AUDIT_ID and frozen["target_run_id"] == target.RUN_ID
    assert digest(target.RUN / "method_config.json") == frozen["target_config_sha256"]
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT / name) == sha == digest(AUDIT / "source_snapshot" / name), name
    return frozen


def freeze():
    if AUDIT.exists() or OUTPUT.exists():
        raise FileExistsError("Keep the previous audit attempt")
    config, _ = target.checked()
    AUDIT.mkdir(parents=True)
    sources = source_snapshot(AUDIT, set(config["source_sha256"]) | EXTRA_SOURCES)
    write_json(AUDIT / "prepared.json", dict(audit_run_id=AUDIT_ID, target_run_id=target.RUN_ID,
        target_config_sha256=digest(target.RUN / "method_config.json"), source_sha256=sources,
        created_at_utc=now(), truth_numeric_read=False,
        scope="Separate auditor source snapshot; pre must genuinely precede public evaluation"))
    print("AUDIT_FROZEN", len(sources), flush=True)


def inspect_inference():
    frozen = checked_audit()
    config, manifest, index, records, receipts = target.audit_records()
    parent_config_names = [name for name in config["receipts"] if name.endswith("/method_config.json")]
    assert len(parent_config_names) == 1
    parent_config = read_json(ROOT / parent_config_names[0])
    parent_input_name = "data/inputs/" + parent_config["run_id"] + "/manifest.json"
    assert config["receipts"][parent_input_name] == parent_config["input_sha256"] == digest(ROOT / parent_input_name)
    assert not parent_config["gt_read_during_inference"]
    parent_inputs = read_json(ROOT / parent_input_name)
    same(manifest["method"], parent_inputs["method"], "Camera optimizer policy changed from its frozen parent")
    same(manifest["variant"], parent_inputs["variant"], "Camera parameter variant changed from its frozen parent")
    for name, sha in parent_config["source_sha256"].items():
        assert config["source_sha256"][name] == sha
    groups = target.unique(manifest["camera_groups"], lambda item: item["camera_group_id"])
    cases = target.unique(manifest["cases"], lambda item: item["case_id"])
    source_freeze = read_json(target.RUN / "source_freeze.json")
    same(source_freeze["source_sha256"], config["source_sha256"], "Source freeze/config disagree")
    assert tuple(config["arms"]) == target.ARMS and set(target.SLOTS) == {
        "control", "leave_group_0", "leave_group_1", "leave_group_2", "leave_group_3"}
    assert len(groups) == 20 and len(cases) == 80 and len(records) == 100
    assert len({receipt["path"] for receipt in records.values()}) == 100
    groups_summary, condition_summary = [], []
    expected_decisions, actual_decisions, expected_rods, actual_rods = set(), set(), set(), set()
    for gid, group in groups.items():
        siblings = {case["case_id"]: case for case in cases.values() if case["camera_group_id"] == gid}
        assert len(siblings) == 4 and len(group["frames"]) == 5
        assert [frame["view_id"] for frame in group["frames"]] == group["view_ids"]
        training_ids = [track["track_id"] for track in group["training"]]
        validation_ids = [track["track_id"] for track in group["validation"]]
        assert len(set(training_ids)) == len(training_ids) == 160
        assert len(set(validation_ids)) == len(validation_ids) == 64
        assert not set(training_ids) & set(validation_ids)
        for frame in group["frames"]:
            bank = frame["measurement_bank"]
            assert frame["source_sha256"] == canonical_hash(bank)
            assert frame["source_kind"] == "synthetic_pixel_measurements"
            assert len(bank["training"]) == 160 and len(bank["validation"]) == 64
            assert len(bank["rod_observations"]) == 8
            assert {row["case_id"] for row in bank["rod_observations"]} == set(siblings)
        assert {condition["condition_id"] for condition in group["group_plan"]["conditions"]} == set(target.SLOTS)
        for condition in group["group_plan"]["conditions"]:
            kept, removed = set(condition["kept_track_ids"]), set(condition["removed_track_ids"])
            assert not kept & removed and kept | removed == set(training_ids)
            assert not (kept | removed) & set(validation_ids)
        groups_summary.append(dict(camera_group_id=gid, case_ids=sorted(siblings),
            initial_intrinsics_sha256=canonical_hash(group["initial_intrinsics"]),
            initial_extrinsics_sha256=canonical_hash(group["initial_extrinsics"]),
            training_sha256=canonical_hash(group["training"]), validation_sha256=canonical_hash(group["validation"]),
            group_plan_sha256=canonical_hash(group["group_plan"]),
            frame_source_sha256=[frame["source_sha256"] for frame in group["frames"]]))
        for slot in target.SLOTS:
            receipt = records[(gid, slot)]
            assert receipt["path"] == f"records/{gid}-{slot}.json", "Unexpected record path"
            record = read_json(target.RUN / receipt["path"])
            assert record["condition"]["condition_id"] == slot
            rods = target.unique(record["rods"], lambda rod: rod["case_id"])
            assert set(rods) == set(siblings)
            if "extrinsics" in record["fit"]:
                assert record["gauge"]["comparable"]
            rod_rows = []
            for case_id, case in siblings.items():
                expected_rods.add((gid, slot, case_id))
                rod = rods[case_id]
                actual_rods.add((gid, slot, rod["case_id"]))
                assert rod["case_sha256"] == canonical_hash(case)
                assert rod["geometry_changed_by_checks"] is False
                evidence = target.unique(rod["evidence"], lambda item: item["evidence_id"])
                raw_evidence = target.unique(case["evidence_sets"], lambda item: item["evidence_id"])
                assert len(evidence) == 3 and set(evidence) == set(raw_evidence)
                for eid, item in evidence.items():
                    assert item["evidence_sha256"] == canonical_hash(raw_evidence[eid])
                    assert set(item["decisions"]) == set(target.ARMS)
                    for arm in target.ARMS:
                        expected_decisions.add((gid, slot, case_id, eid, arm))
                        actual_decisions.add((gid, slot, rod["case_id"], item["evidence_id"], arm))
                rod_rows.append(dict(case_id=case_id, identity_sha256=canonical_hash(rod["identity"]),
                                     endpoint_sha256=canonical_hash(rod["endpoint"]), evidence_ids=sorted(evidence)))
            condition_summary.append(dict(camera_group_id=gid, condition=slot, record_sha256=receipt["sha256"],
                fit_sha256=canonical_hash(record["fit"]), fit_state=record["fit"]["state"],
                camera_decision=record["camera"]["state"], rods=rod_rows))
    assert actual_rods == expected_rods and len(actual_rods) == 400
    assert actual_decisions == expected_decisions and len(actual_decisions) == 4800
    receipts[(AUDIT / "prepared.json").relative_to(ROOT).as_posix()] = digest(AUDIT / "prepared.json")
    for name, sha in frozen["source_sha256"].items():
        receipts[name] = sha
        receipts[(AUDIT / "source_snapshot" / name).relative_to(ROOT).as_posix()] = sha
    return dict(receipts=receipts, inference_sha256=digest(target.RUN / "inference.json"),
                camera_group_count=20, case_count=80, camera_condition_count=100,
                rod_condition_count=400, decision_count=4800, groups=groups_summary,
                conditions=condition_summary, gt_read=False, physical_scores_read=False)


def pre():
    if OUTPUT.exists() or target.PUBLIC.exists() or (ROOT / "data/evaluation" / target.RUN_ID).exists():
        raise FileExistsError("Pre evidence must precede evaluation, never reconstruct it afterwards")
    if (AUDIT / "before-evaluation.json").exists():
        raise FileExistsError("Keep the real pre-evaluation audit")
    sys.addaudithook(reject_truth_open)
    inspected = inspect_inference()
    write_json(AUDIT / "before-evaluation.json", dict(state="passed", recorded_at_utc=now(), **inspected))
    print("INDEPENDENT_PRE_PASSED", len(inspected["receipts"]), "receipts; 100/400/4800 fixed identities", flush=True)


def projection(camera, point):
    # A direct homogeneous projection cross-check, separate from the generator.
    transformed = np.asarray(camera["world_to_camera_cv"]) @ np.r_[point, 1.]
    assert transformed[2] > 0
    pixel = np.asarray(camera["K_index"]) @ transformed[:3]
    return pixel[:2] / pixel[2]


def pairing_check(manifest, truth):
    protocol = read_json(ROOT / "configs/camera_evidence_challenges_v1.json")
    generated, hidden = generate_paired_challenges(protocol)
    saved = {key: value for key, value in manifest.items() if key not in ("method", "variant")}
    saved = dict(saved, camera_groups=[{key: value for key, value in group.items()
        if key not in ("group_plan", "original_plan", "all_view_plan")} for group in manifest["camera_groups"]])
    same(generated, saved, "Frozen inputs do not replay from the declared generator")
    same(hidden, truth, "Truth does not replay from the same frozen generation")
    gt_groups = {group["camera_group_id"]: group for group in truth["camera_groups"]}
    gt_cases = {case["case_id"]: case for case in truth["cases"]}
    maximum_noise_mismatch, maximum_anchor_mismatch = 0., 0.
    for group in manifest["camera_groups"]:
        gid = group["camera_group_id"]
        cameras = gt_groups[gid]["cameras"]
        assert [camera["view_id"] for camera in cameras] == group["view_ids"]
        siblings = {gt_cases[case["case_id"]]["mode"]: case for case in manifest["cases"]
                    if case["camera_group_id"] == gid}
        assert set(siblings) == {"ordinary", "endpoint_swap", "coherent_lateral", "coherent_depth"}
        noise_reference = None
        for mode in ("ordinary", "coherent_lateral", "coherent_depth"):
            case = siblings[mode]
            geometry = gt_cases[case["case_id"]]["measurement_rod_segments"][0]
            noise = np.asarray([[np.asarray(obs["xy"]) - projection(cameras[obs["view"]], geometry[i])
                for obs in endpoint["observations"]] for i, endpoint in enumerate(case["rod_tracks"][0]["endpoint_tracks"])])
            if noise_reference is None:
                noise_reference = noise
            else:
                maximum_noise_mismatch = max(maximum_noise_mismatch, float(abs(noise - noise_reference).max()))
        clean = siblings["ordinary"]["rod_tracks"][0]["endpoint_tracks"]
        swap = siblings["endpoint_swap"]["rod_tracks"][0]["endpoint_tracks"]
        for endpoint in range(2):
            for view in range(5):
                assert swap[endpoint]["observations"][view] == clean[1 - endpoint if view == 2 else endpoint]["observations"][view]
        for case in siblings.values():
            hidden_case = gt_cases[case["case_id"]]
            labels = target.unique(hidden_case["evidence_labels"], lambda item: item["evidence_id"])
            for evidence in case["evidence_sets"]:
                label = labels[evidence["evidence_id"]]
                geometry = np.asarray(hidden_case["measurement_rod_segments"] if label["physical_source"] == "measurement_rod"
                                      else hidden_case["rod_segments"])[0]
                for index, anchor in enumerate(evidence["anchors"]):
                    point = geometry[0] + label["point_fractions"][index] * (geometry[1] - geometry[0])
                    camera = cameras[group["view_ids"].index(anchor["view_id"])]
                    expected = projection(camera, point) + label["noise_uv_px"][index]
                    maximum_anchor_mismatch = max(maximum_anchor_mismatch, float(abs(expected - anchor["xy"]).max()))
    assert maximum_noise_mismatch < 1e-10 and maximum_anchor_mismatch < 1e-10
    return dict(input_generation_replay_exact=True, truth_generation_replay_exact=True,
                maximum_shared_endpoint_noise_mismatch_px=maximum_noise_mismatch,
                maximum_independent_projection_anchor_mismatch_px=maximum_anchor_mismatch)


def post():
    checked_audit()
    if OUTPUT.exists() or (AUDIT / "summary.json").exists():
        raise FileExistsError("Preserve previous post audit")
    before_path = AUDIT / "before-evaluation.json"
    before = read_json(before_path)
    assert before["state"] == "passed" and not before["gt_read"] and not before["physical_scores_read"]
    after = inspect_inference()
    same({key: value for key, value in before.items() if key not in ("state", "recorded_at_utc")},
         after, "Inference or source changed since the real pre audit")
    for name, sha in before["receipts"].items():
        assert digest(ROOT / name) == sha, name
    config, manifest = target.checked()
    root_before = read_json(target.RUN / "before_evaluation.json")
    assert root_before["state"] == "passed" and not root_before["gt_read"]
    for name, sha in root_before["receipts"].items():
        assert before["receipts"][name] == sha == digest(ROOT / name)
    evaluation_path = ROOT / "data/evaluation" / target.RUN_ID / "summary.json"
    assert digest(evaluation_path) == digest(target.PUBLIC)
    result = read_json(evaluation_path)
    assert result["state"] == "complete" and result["run_id"] == target.RUN_ID
    assert result["inference_sha256"] == before["inference_sha256"]
    assert result["before_sha256"] == digest(target.RUN / "before_evaluation.json")
    assert not result["gt_read_during_inference"] and result["no_automatic_promotion"]
    same(result["config"], config, "Evaluation config detached")
    truth_frozen = read_json(target.TRUTH / "frozen.json")
    assert truth_frozen["input_sha256"] == config["input_sha256"]
    assert truth_frozen["source_freeze_sha256"] == config["source_freeze_sha256"]
    assert truth_frozen["truth_sha256"] == digest(target.TRUTH / "manifest.json")
    same(result["truth_frozen"], truth_frozen, "Published truth receipt changed")
    truth = read_json(target.TRUTH / "manifest.json")
    pairing = pairing_check(manifest, truth)
    gt_groups = target.unique(truth["camera_groups"], lambda item: item["camera_group_id"])
    gt_cases = target.unique(truth["cases"], lambda item: item["case_id"])
    groups = target.unique(manifest["camera_groups"], lambda item: item["camera_group_id"])
    cases = target.unique(manifest["cases"], lambda item: item["case_id"])
    assert set(groups) == set(gt_groups) and set(cases) == set(gt_cases)
    index = read_json(target.RUN / "inference.json")
    records = target.unique(index["records"], lambda item: (item["camera_group_id"], item["condition"]))
    alignment_rows = target.unique(result["cameras"], lambda item: item["camera_group_id"])
    assert set(alignment_rows) == set(groups)
    matrices = {}
    for gid, group in groups.items():
        control = read_json(target.RUN / records[(gid, "control")]["path"])
        fit = control["fit"]
        alignment = None
        if "extrinsics" in fit:
            try:
                alignment = target.camera_score(np.asarray(fit["intrinsics"]), np.asarray(fit["extrinsics"]),
                                                 gt_groups[gid]["cameras"], config["evaluation"]["alignment"])
            except ValueError:
                pass
        same(alignment_rows[gid]["alignment"], alignment, "Control-only alignment changed")
        matrices[gid] = np.asarray(alignment["prediction_world_to_gt_world"]) if alignment is not None else None
    expected_rows = {(case["camera_group_id"], case["case_id"], slot) for case in cases.values() for slot in target.SLOTS}
    rows = target.unique(result["rows"], lambda row: (row["camera_group_id"], row["case_id"], row["condition"]))
    assert set(rows) == expected_rows and len(rows) == 400
    count, nonemitted = 0, 0
    for (gid, cid, slot), row in rows.items():
        record = read_json(target.RUN / records[(gid, slot)]["path"])
        rod = next(item for item in record["rods"] if item["case_id"] == cid)
        assert row["shared_alignment_camera_group"] == gid and gt_cases[cid]["camera_group_id"] == gid
        assert row["family"] == gt_cases[cid]["mode"]
        same(row["evidence_labels"], gt_cases[cid]["evidence_labels"], "Evidence label detached")
        same(row["camera_decision"], record["camera"], "Camera decision changed")
        same(row["endpoint_check"], rod["endpoint"], "Endpoint diagnostic changed")
        same(row["target_checks"], [dict(evidence_id=item["evidence_id"], check=item["target"]) for item in rod["evidence"]],
             "Target diagnostic changed")
        metrics, matrix = None, matrices[gid]
        if matrix is not None:
            native = np.asarray(rod["identity"]["segments"]).reshape(-1, 2, 3)
            aligned = native @ matrix[:3, :3].T + matrix[:3, 3]
            metrics = target.curve_metrics(aligned, gt_cases[cid]["rod_segments"],
                tolerance=config["evaluation"]["curve_tolerance_m"], spacing=config["evaluation"]["curve_spacing_m"])
        same(row["geometry_metrics"], metrics, "A fold used a different alignment or physical readout")
        decisions = target.unique(row["decisions"], lambda item: (item["evidence_id"], item["arm"]))
        expected_decisions = {(evidence["evidence_id"], arm): value for evidence in rod["evidence"]
                              for arm, value in evidence["decisions"].items()}
        assert set(decisions) == set(expected_decisions) and len(decisions) == 12
        for key, value in expected_decisions.items():
            emitted = decisions[key]
            assert emitted["state"] == value["state"] and emitted["reasons"] == value["reasons"]
            accepted = value["state"] == "accepted"
            expected_r = 0. if not accepted else None if metrics is None else metrics["recovery_fraction"]
            expected_p = metrics["precision_fraction"] if accepted and metrics is not None else None
            assert emitted["emitted_recovery"] == expected_r and emitted["emitted_precision"] == expected_p
            nonemitted += int(not accepted)
            count += 1
    assert count == 4800
    for name, sha in before["receipts"].items():
        assert digest(ROOT / name) == sha, name
    checked_audit()
    report = dict(state="complete", audit_run_id=AUDIT_ID, target_run_id=target.RUN_ID,
        audit_prepared_sha256=digest(AUDIT / "prepared.json"), before_evaluation_sha256=digest(before_path),
        pre_recorded_at_utc=before["recorded_at_utc"], post_recorded_at_utc=now(),
        inference_sha256=before["inference_sha256"], evaluation_sha256=digest(evaluation_path),
        public_copy_sha256=digest(target.PUBLIC), truth_frozen=truth_frozen,
        source_and_inference_unchanged_across_evaluation=True, unchanged_receipt_count=len(before["receipts"]),
        camera_group_count=20, case_count=80, camera_fit_count=100, physical_row_count=400, decision_count=count,
        nonemitted_decisions_checked_zero_recovery_null_precision=nonemitted,
        alignment_policy_verified="One full-control camera-only Sim3 per group for all four modes and all five fits",
        pairing=pairing, pre_gt_read=False, pre_physical_scores_read=False,
        groups=before["groups"], condition_identity_sha256=canonical_hash(before["conditions"]),
        scope="An audit of declared synthetic pairing and unchanged evidence; no independent real-object or target-recognition claim")
    write_json(AUDIT / "summary.json", report)
    write_json(OUTPUT, report)
    print("INDEPENDENT_POST_PASSED", len(before["receipts"]), "unchanged receipts; 400 rows; 4800 decisions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pre", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {"freeze": freeze, "pre": pre, "post": post}[args.stage]()
