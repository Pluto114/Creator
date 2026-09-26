"""Independent before/after audit of new-view RGB localization and rod checks.

Pre reconstructs RGB matches and map observations under the truth/public-score
tripwire. Post alone reads truth and old physical scores. No BA, target fitting,
threshold selection, or changes to old predictions occur in this auditor.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import run_rod_heldout_views as target
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json
from run_rod_candidate_ablation import canonical_hash

# isort: split
from creator_eval.heldout_view_pose import DEFAULTS, _scores

AUDIT_ID = "rod-heldout-view-audit-v1-20260926"
AUDIT = ROOT / ".runtime/experiments" / AUDIT_ID
OUTPUT = ROOT / "docs/experiments/results/2026-09-26-heldout-view-audit.json"
OWN_SOURCE = "scripts/audit_rod_heldout_views.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def same(a, b, message):
    if canonical_hash(a) != canonical_hash(b):
        raise ValueError(message)


def unique(rows, key):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "Duplicate audited identity"
    return result


def checked_audit():
    frozen = read_json(AUDIT / "prepared.json")
    assert frozen["audit_run_id"] == AUDIT_ID and frozen["target_run_id"] == target.RUN_ID
    assert frozen["target_config_sha256"] == digest(target.RUN / "method_config.json")
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT / name) == sha == digest(AUDIT / "source_snapshot" / name), name
    return frozen


def freeze():
    if AUDIT.exists() or OUTPUT.exists():
        raise FileExistsError("Preserve earlier audit attempts")
    config, _ = target.checked()
    AUDIT.mkdir(parents=True)
    sources = source_snapshot(AUDIT, set(config["source_sha256"]) | {OWN_SOURCE})
    write_json(AUDIT / "prepared.json", dict(audit_run_id=AUDIT_ID, target_run_id=target.RUN_ID,
        target_config_sha256=digest(target.RUN / "method_config.json"), source_sha256=sources,
        created_at_utc=now(), truth_numeric_read=False, physical_scores_read=False,
        scope="Explicit target source closure plus this auditor; no directory glob"))
    print("HELDOUT_AUDIT_FROZEN", len(sources), flush=True)


def verify_saved_pose(task, camera, frame, matching, map_sha, saved):
    """Rebuild inputs and score a sealed pose; never run PnP again."""
    observations, missing = target.map_observations(task, camera, matching)
    same(saved["observations"], observations, "Map observations detached from RGB/old map")
    same(saved["missing"], missing, "Excluded map observations changed")
    k = target.shared_k(camera["result"]["intrinsics"])
    source = dict(source_kind="rgb", rgb_sha256=frame["rgb_sha256"], map_sha256=map_sha,
                  intrinsics_sha256=canonical_hash(k), size_wh=frame["size_wh"])
    same(saved["source"], source, "Pose source detached")
    def arrays(role):
        rows = observations[role]
        return (np.asarray([r["xyz"] for r in rows], float).reshape(-1, 3),
                np.asarray([r["xy"] for r in rows], float).reshape(-1, 2),
                [r["track_id"] for r in rows])
    xyz, xy, ids = arrays("train")
    proposal = saved["proposal"]
    same(proposal["config"], DEFAULTS, "PnP policy changed")
    training_source = dict(source, observation_role="background_pose_training")
    same(proposal["source"], training_source, "Training source detached")
    same(proposal["K"], k, "New K changed from the declared old-map prior")
    assert proposal["training_track_ids"] == ids and proposal["mapped_training_count"] == len(ids)
    assert proposal["input_sha256"] == canonical_hash(dict(points3d=xyz, xy=xy, K=k, track_ids=ids, source=training_source))
    assert proposal["proposal_sha256"] == canonical_hash({key: value for key, value in proposal.items() if key != "proposal_sha256"})
    if proposal["E"] is not None:
        assert proposal["state"] == "fitted" and not proposal["reasons"]
        e = np.asarray(proposal["E"])
        scores, _, _ = _scores(xyz, xy, k, e, ids)
        same(scores, proposal["all_training"], "Saved pose does not reproduce its training projections")
        selected = np.asarray(proposal["ransac_inlier_indices"], int)
        assert len(selected) == len(set(selected.tolist())) >= DEFAULTS["minimum_inlier_tracks"]
        assert np.all((selected >= 0) & (selected < len(ids)))
        assert proposal["ransac_inlier_track_ids"] == [ids[i] for i in selected]
        inlier_scores, valid, errors = _scores(xyz[selected], xy[selected], k, e, [ids[i] for i in selected])
        same(inlier_scores, proposal["refined_ransac_inliers"], "Refined inlier projections changed")
        assert valid.all()
        assert proposal["final_support_count"] == int(np.count_nonzero(valid & (errors <= DEFAULTS["reprojection_threshold_px"])))
    xyz, xy, ids = arrays("validation")
    verified = target.verify_pose(proposal, xyz, xy, track_ids=ids,
                                  source=dict(source, observation_role="background_pose_validation"))
    same(verified, saved["verification"], "Held-out validation changed a sealed pose or its decision")
    assert proposal["proposal_sha256"] == canonical_hash({key: value for key, value in proposal.items() if key != "proposal_sha256"})
    return saved


def inspect():
    import cv2

    frozen = checked_audit()
    config, index, records, receipts = target.audit_normal()
    _, manifest = target.checked()
    _, _, sensitivity = target.checked_training()
    source_tasks = unique(sensitivity["tasks"], lambda task: (task["parent"], task["case_id"]))
    tasks = unique(manifest["tasks"], lambda row: row["task_id"])
    assert len(tasks) == 6 and len(records) == 30
    receipts.update(config["receipts"])
    receipts[(AUDIT / "prepared.json").relative_to(ROOT).as_posix()] = digest(AUDIT / "prepared.json")
    for name, sha in frozen["source_sha256"].items():
        receipts[name] = sha
        receipts[(AUDIT / "source_snapshot" / name).relative_to(ROOT).as_posix()] = sha
    for name, sha in config["source_sha256"].items():
        receipts[(target.RUN / "source_snapshot" / name).relative_to(ROOT).as_posix()] = sha
    scene = read_json(target.SCENE / "prepared.json")
    scene_inputs = read_json(target.SCENE_INPUTS / "manifest.json")
    assert scene["input_sha256"] == digest(target.SCENE_INPUTS / "manifest.json")
    provenance = scene_inputs["scene_source_provenance"]
    assert digest(ROOT / provenance["parent_input_manifest"]) == provenance["parent_input_sha256"]
    assert digest(ROOT / provenance["parent_prepared_path"]) == provenance["parent_prepared_sha256"]
    for name, sha in scene["source_sha256"].items():
        assert digest(ROOT / name) == sha == digest(target.SCENE / "source_snapshot" / name)
        receipts[(target.SCENE / "source_snapshot" / name).relative_to(ROOT).as_posix()] = sha
    annotations = read_json(target.ANNOTATIONS)
    annotation_freeze_path = target.SCENE / "annotations-frozen.json"
    annotation_freeze = read_json(annotation_freeze_path)
    assert annotation_freeze["annotations_sha256"] == digest(target.ANNOTATIONS)
    assert annotation_freeze["scene_prepared_sha256"] == digest(target.SCENE / "prepared.json")
    assert annotation_freeze["before_any_new_pose_or_projection"] and annotation_freeze["new_inference_directory_absent"]
    assert annotations["source_input_sha256"] == scene["input_sha256"]
    assert not annotations["ground_truth_used"] and not annotations["predicted_geometry_used"]
    receipts[annotation_freeze_path.relative_to(ROOT).as_posix()] = digest(annotation_freeze_path)
    for name in ("generation-freeze.json", "generation-checks.json"):
        path = target.SCENE / name
        expected = scene["generation_freeze_sha256" if name == "generation-freeze.json" else "generation_checks_sha256"]
        assert digest(path) == expected
        # Hash receipts only; no generation check numbers enter the inference audit.
        receipts[path.relative_to(ROOT).as_posix()] = expected
    cv2.setNumThreads(1)
    match_cache, task_summary = {}, []
    for tid, task in tasks.items():
        maps = unique(task["maps"], lambda row: row["condition"])
        assert [frame["view_id"] for frame in task["old_frames"]] == source_tasks[(task["parent"], task["case_id"])]["view_ids"]
        assert set(maps) == set(target.CONDITIONS)
        assert [frame["view_id"] for frame in task["new_frames"]] == ["view_h00", "view_h01"]
        assert {f["method"] for f in task["families"]} == {"baseline", "cylinder_support"}
        assert len(task["families"]) == 2
        new_case = next(case for case in scene_inputs["cases"] if case["case_id"] == task["case_id"])
        same(task["new_frames"], new_case["frames"], "New image identity detached")
        assert not {f["rgb_sha256"] for f in task["old_frames"]} & {f["rgb_sha256"] for f in task["new_frames"]}
        for frame in [*task["old_frames"], *task["new_frames"]]:
            assert digest(ROOT / frame["rgb"]) == frame["rgb_sha256"]
            receipts[frame["rgb"]] = frame["rgb_sha256"]
        matched = target.match_task(task)
        saved_matches = read_json(target.RUN / "matches" / (tid + ".json"))
        same(matched, saved_matches, "SIFT extraction/consensus does not replay")
        match_cache[tid] = matched
        tracks = read_json(ROOT / task["tracks_path"])
        for frame in matched["frames"]:
            ids = [row["new_feature_id"] for row in frame["matches"]]
            assert len(ids) == len(set(ids))
            accepted = [row for row in frame["matches"] if row["state"] == "matched"]
            track_ids = [row["old_track_ids"][0] for row in accepted]
            assert len(track_ids) == len(set(track_ids))
            for row in accepted:
                assert len(row["old_track_ids"]) == 1 and len(set(row["old_view_ids"])) >= 2
                label = tracks["band_labels"][row["old_track_ids"][0]]
                expected = "train" if label == "train" and row["xy"][1] >= 192 else (
                    "validation" if label == "validation" and row["xy"][1] < 128 else "excluded")
                assert row["role"] == expected
        task_summary.append(dict(task_id=tid, case_id=task["case_id"], parent=task["parent"],
            old_rgb_sha256=[f["rgb_sha256"] for f in task["old_frames"]],
            new_rgb_sha256=[f["rgb_sha256"] for f in task["new_frames"]],
            matching_sha256=canonical_hash(saved_matches), query_sha256=canonical_hash(task["query"])))
    by_record = unique(records, lambda row: (row["task_id"], row["condition"]))
    expected = {(tid, slot) for tid in tasks for slot in target.CONDITIONS}
    assert set(by_record) == expected
    index_rows = unique(index["records"], lambda row: (row["task_id"], row["condition"]))
    assert set(index_rows) == expected
    geometry, poses, checks = set(), set(), []
    anchor_slots = 0
    for (tid, slot), record in by_record.items():
        task = tasks[tid]
        entry = next(row for row in task["maps"] if row["condition"] == slot)
        receipt = index_rows[(tid, slot)]
        assert receipt["path"] == f"records/{tid}-{slot}.json"
        assert record["fixed_map_sha256"] == entry["sha256"] == digest(ROOT / entry["path"])
        assert (record["parent"], record["case_id"]) == (task["parent"], task["case_id"])
        camera = read_json(ROOT / entry["path"])
        assert camera["condition"]["condition_id"] == entry["condition"] == slot
        assert record["old_geometry_modified"] is False and not record["gt_read_during_inference"]
        assert [pose["view_id"] for pose in record["poses"]] == [f["view_id"] for f in task["new_frames"]]
        fresh_poses = []
        for frame, matching, saved_pose in zip(task["new_frames"], match_cache[tid]["frames"], record["poses"]):
            fresh = verify_saved_pose(task, camera, frame, matching, entry["sha256"], saved_pose)
            assert saved_pose["source"]["rgb_sha256"] == frame["rgb_sha256"]
            assert saved_pose["source"]["map_sha256"] == entry["sha256"]
            assert saved_pose["source"]["intrinsics_sha256"] == canonical_hash(target.shared_k(camera["result"]["intrinsics"]))
            train, validation = fresh["observations"]["train"], fresh["observations"]["validation"]
            assert not {r["track_id"] for r in train} & {r["track_id"] for r in validation}
            assert not {r["new_feature_id"] for r in train} & {r["new_feature_id"] for r in validation}
            assert not fresh["proposal"]["map_modified"] and not fresh["proposal"]["intrinsics_modified"]
            assert not fresh["proposal"]["target_observations_used"] and not fresh["verification"]["pose_modified"]
            poses.add((tid, slot, frame["view_id"]))
            fresh_poses.append(fresh)
        frames = [dict(view_id=f["view_id"], size_wh=f["size_wh"], source_sha256=f["rgb_sha256"],
                       source_kind="rgb", K_index=p["proposal"]["K"], world_to_camera_cv=p["proposal"]["E"])
                  for f, p in zip(task["new_frames"], fresh_poses)]
        pose_ok = [p["verification"]["state"] == "validated"
                   and not any(item["role"] == "validation" for item in p["missing"]) for p in fresh_poses]
        rods = unique(record["rods"], lambda rod: rod["method"])
        assert set(rods) == {"baseline", "cylinder_support"}
        for method, rod in rods.items():
            family = next(f for f in task["families"] if f["method"] == method)
            packet = next(p for p in family["packets"] if p["condition_id"] == slot)
            assert packet["parent_record_sha256"] == entry["sha256"]
            identity = packet["identity"] or dict(state="unavailable", segments=[])
            check = target.check_target_anchors(frames, identity["segments"], task["query"]["anchors"], threshold_px=2.)
            same(check, rod["target_check"], "Target diagnostic changed or omitted a missing-view slot")
            assert len(check["per_anchor"]) == 2
            anchor_slots += len(check["per_anchor"])
            assert rod["geometry_sha256"] == canonical_hash(identity["segments"])
            assert rod["old_packet_sha256"] == canonical_hash(packet)
            if identity["state"] != "accepted":
                expected_state = "original_identity_not_accepted"
            elif packet["camera_decision"]["state"] != "candidate_camera_correction":
                expected_state = "withheld_original_camera"
            elif not all(pose_ok):
                expected_state = "unresolved_new_pose"
            else:
                expected_state = {"supported": "retained_candidate", "contradicted": "rejected_new_target"}.get(
                    check["state"], "unresolved_new_target")
            assert rod["state"] == expected_state
            geometry.add((tid, slot, method))
        checks.append(dict(task_id=tid, condition=slot, fixed_map_sha256=entry["sha256"],
            pose_states=[p["verification"]["state"] for p in fresh_poses],
            rod_states={method: rod["state"] for method, rod in rods.items()},
            pose_input_sha256=[p["proposal"]["input_sha256"] for p in fresh_poses]))
    assert len(poses) == 60 and len(geometry) == 60 and anchor_slots == 120
    for name, sha in receipts.items():
        assert digest(ROOT / name) == sha
    return dict(receipts=receipts, task_count=6, map_count=30, pose_count=60, joint_rod_count=60,
                anchor_projection_slot_count=anchor_slots, unique_new_rgb_count=6,
                tasks=task_summary, checks=checks, inference_sha256=digest(target.RUN / "inference.json"),
                gt_read=False, physical_scores_read=False)


def pre():
    if target.PUBLIC.exists() or (target.RUN / "evaluation.json").exists() or OUTPUT.exists():
        raise FileExistsError("Record pre audit before evaluation, never reconstruct it later")
    if (AUDIT / "before-evaluation.json").exists():
        raise FileExistsError("Preserve real pre evidence")
    sys.addaudithook(target.block_truth)
    checked = inspect()
    write_json(AUDIT / "before-evaluation.json", dict(state="passed", recorded_at_utc=now(), **checked))
    print("HELDOUT_EXTERNAL_PRE", len(checked["receipts"]), "receipts; 60 poses/60 rods/120 anchor slots", flush=True)


def post():
    checked_audit()
    if OUTPUT.exists() or (AUDIT / "summary.json").exists():
        raise FileExistsError("Preserve previous post audit")
    before_path = AUDIT / "before-evaluation.json"
    before = read_json(before_path)
    assert before["state"] == "passed" and not before["gt_read"] and not before["physical_scores_read"]
    after = inspect()
    same({k: v for k, v in before.items() if k not in ("state", "recorded_at_utc")}, after,
         "Source, matches, poses or old geometry changed across evaluation")
    config, manifest = target.checked()
    root_pre = read_json(target.RUN / "before-evaluation.json")
    assert root_pre["no_truth_read"]
    for name, sha in root_pre["receipts"].items():
        assert before["receipts"][name] == sha == digest(ROOT / name)
    result = read_json(target.RUN / "evaluation.json")
    assert digest(target.RUN / "evaluation.json") == digest(target.PUBLIC)
    assert result["run_id"] == target.RUN_ID and result["state"] == "complete"
    assert result["inference_sha256"] == before["inference_sha256"]
    assert result["before_sha256"] == digest(target.RUN / "before-evaluation.json")
    same(config, result["config"], "Published config detached")
    assert digest(target.PHYSICAL) == config["physical_source_sha256"] == result["physical_source_sha256"]
    physical = read_json(target.PHYSICAL)
    _, envelope_config, _ = target.checked_envelope("replay")
    assert physical["run_id"] == envelope_config["run_id"]
    same(physical["config"], envelope_config, "Old physical score configuration detached")
    old_run = ROOT / ".runtime/experiments" / envelope_config["run_id"]
    assert physical["inference_sha256"] == digest(old_run / "inference.json")
    families = unique(physical["rows"], lambda family: family["task_id"])
    scene = read_json(target.SCENE / "prepared.json")
    scene_input = read_json(target.SCENE_INPUTS / "manifest.json")
    truth_root = ROOT / "data/eval_gt" / target.SCENE_ID
    assert digest(truth_root / "manifest.json") == scene["truth_sha256"] == result["truth_sha256"]
    truth = read_json(truth_root / "manifest.json")
    assert truth["input_sha256"] == scene["input_sha256"] == digest(target.SCENE_INPUTS / "manifest.json")
    assert digest(truth_root / "artifact_hashes.json") == scene["truth_artifacts_sha256"]
    new_artifacts = read_json(truth_root / "artifact_hashes.json")
    gt_cases = unique(truth["cases"], lambda case: case["case_id"])
    tasks = unique(manifest["tasks"], lambda task: task["task_id"])
    input_cases = unique(scene_input["cases"], lambda case: case["case_id"])
    expected_pose_keys = {(tid, slot, view) for tid in tasks for slot in target.CONDITIONS for view in ("view_h00", "view_h01")}
    expected_rod_keys = {(family["task_id"], slot) for task in tasks.values() for family in task["families"] for slot in target.CONDITIONS}
    pose_rows = unique(result["pose_rows"], lambda row: (row["task_id"], row["condition"], row["view_id"]))
    rod_rows = unique(result["rows"], lambda row: (row["task_id"], row["condition"]))
    assert set(pose_rows) == expected_pose_keys and set(rod_rows) == expected_rod_keys
    matrices, compared_meshes = {}, []
    for tid, task in tasks.items():
        cid = task["case_id"]
        gt = gt_cases[cid]
        assert [camera["view_id"] for camera in gt["cameras"]] == [f["view_id"] for f in input_cases[cid]["frames"]]
        assert all(camera["size_wh"] == frame["size_wh"] for camera, frame in zip(gt["cameras"], input_cases[cid]["frames"]))
        for true_frame, rgb_frame in zip(gt["frames"], input_cases[cid]["frames"]):
            assert true_frame["view_id"] == rgb_frame["view_id"] and true_frame["rgb_sha256"] == rgb_frame["rgb_sha256"]
        old_bundle = ROOT / ".runtime/experiments" / task["parent"]
        parent_scene_id = read_json(old_bundle / "protocol.json")["run_id"]
        parent_scene = ROOT / ".runtime/experiments" / parent_scene_id
        assert read_json(old_bundle / "prepared.json")["scene_prepared_sha256"] == digest(parent_scene / "prepared.json")
        old_truth_root = ROOT / "data/eval_gt" / parent_scene_id
        old_scene_frozen = read_json(parent_scene / "prepared.json")
        assert digest(old_truth_root / "manifest.json") == old_scene_frozen["truth_sha256"]
        assert digest(old_truth_root / "artifact_hashes.json") == old_scene_frozen["truth_artifacts_sha256"]
        old_mesh = old_truth_root / cid / "mesh.npz"
        new_mesh = truth_root / cid / "mesh.npz"
        assert digest(old_mesh) == read_json(old_truth_root / "artifact_hashes.json")[f"{cid}/mesh.npz"]
        assert digest(new_mesh) == new_artifacts[f"{cid}/mesh.npz"]
        with np.load(old_mesh, allow_pickle=False) as a, np.load(new_mesh, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            assert all(np.array_equal(a[key], b[key]) for key in a.files), "World geometry differs from the old Sim3 frame"
        compared_meshes.append(dict(task_id=tid, parent_scene_id=parent_scene_id,
                                    old_mesh_sha256=digest(old_mesh), new_mesh_sha256=digest(new_mesh), arrays_equal=True))
        baseline_family = next(f for f in task["families"] if f["method"] == "baseline")
        matrix = np.asarray(families[baseline_family["task_id"]]["physical"]["alignment"]["prediction_world_to_gt_world"])
        assert matrix.shape == (4, 4)
        matrices[tid] = matrix
    index = read_json(target.RUN / "inference.json")
    nonemitted = 0
    for receipt in index["records"]:
        record = read_json(target.RUN / receipt["path"])
        tid, slot = record["task_id"], record["condition"]
        matrix = matrices[tid]
        scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
        rotation = matrix[:3, :3] / scale
        gt_cameras = {camera["view_id"]: camera for camera in gt_cases[record["case_id"]]["cameras"]}
        for pose in record["poses"]:
            row = pose_rows[(tid, slot, pose["view_id"])]
            same(row["proposal"], pose["proposal"], "Published pose changed")
            same(row["verification"], pose["verification"], "Published validation changed")
            same(row["missing"], pose["missing"], "Missing observation dropped")
            e = pose["proposal"].get("E")
            center_error, angle_error = None, None
            if e is not None:
                e = np.asarray(e)
                gt_e = np.asarray(gt_cameras[pose["view_id"]]["world_to_camera_cv"])[:3]
                center = -e[:, :3].T @ e[:, 3]
                true_center = -gt_e[:, :3].T @ gt_e[:, 3]
                center_error = float(np.linalg.norm(matrix[:3, :3] @ center + matrix[:3, 3] - true_center))
                relative = (e[:, :3] @ rotation.T) @ gt_e[:, :3].T
                angle_error = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))
            assert row["center_error_m"] == center_error and row["rotation_error_degrees"] == angle_error
        for rod in record["rods"]:
            row = rod_rows[(rod["task_id"], slot)]
            for key, value in rod.items():
                same(row[key], value, "Published rod decision changed: " + key)
            source = families[rod["task_id"]]
            assert (source["parent"], source["case_id"], source["method"]) == (record["parent"], record["case_id"], rod["method"])
            old = next(item for item in source["physical"]["rows"] if item["condition_id"] == slot)
            same(row["physical"], old, "Old physical score joined to a different condition")
            metrics = old["metrics"]
            retained = rod["state"] == "retained_candidate"
            expected_r = metrics["recovery_fraction"] if retained and metrics else None if retained else 0.
            expected_p = metrics["precision_fraction"] if retained and metrics else None
            assert row["emitted_recovery"] == expected_r and row["emitted_precision"] == expected_p
            nonemitted += int(not retained)
    for name, sha in before["receipts"].items():
        assert digest(ROOT / name) == sha
    checked_audit()
    report = dict(state="complete", audit_run_id=AUDIT_ID, target_run_id=target.RUN_ID,
        prepared_sha256=digest(AUDIT / "prepared.json"), before_evaluation_sha256=digest(before_path),
        pre_recorded_at_utc=before["recorded_at_utc"], post_recorded_at_utc=now(),
        inference_sha256=before["inference_sha256"], evaluation_sha256=digest(target.PUBLIC),
        input_sha256=config["input_sha256"], truth_sha256=scene["truth_sha256"],
        physical_source_sha256=digest(target.PHYSICAL), source_and_inference_unchanged=True,
        unchanged_receipt_count=len(before["receipts"]), task_count=6, old_map_count=30,
        pose_count=60, joint_rod_count=60, anchor_projection_slot_count=120, unique_new_rgb_count=6,
        nonemitted_R0_Pnull_count=nonemitted, unchanged_geometry=True, no_old_rejection_promoted=True,
        matching_and_pose_inputs_replay_exact=True, sealed_pose_training_and_validation_recomputed=True,
        pnp_refitted_during_audit=False, old_physical_join_exact=True, new_truth_ID_and_RGB_chain_exact=True,
        old_world_frame_mesh_checks=compared_meshes, tasks=before["tasks"],
        scope="New RGB evidence on three known Blender layouts, localized in old estimated maps. Correlated maps, views and methods; no independent metric-camera or real-object claim.")
    write_json(AUDIT / "summary.json", report)
    write_json(OUTPUT, report)
    print("HELDOUT_EXTERNAL_POST", len(before["receipts"]), "unchanged receipts; 60 poses/60 rods", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pre", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {"freeze": freeze, "pre": pre, "post": post}[args.stage]()
