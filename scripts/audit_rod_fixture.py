"""Independent before/after receipts and direct declared-world fixture evaluation.

The metric definition was frozen before normal inference. Pre never opens truth;
post verifies the actual pre bytes. These Python guards are not OS isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.fixture_calibration import (  # noqa: E402
    POLICY as CAMERA_POLICY,
)
from creator_eval.fixture_calibration import (  # noqa: E402
    coverage_reasons,
    detect_markers,
    observations,
    validate_cad,
)
from creator_eval.fixture_physical_metrics import (  # noqa: E402
    POLICY,
    score_finite_structure,
)
from creator_eval.rod_fixture_finite import (  # noqa: E402
    canonical_hash,
    extract_fixture_evidence,
)

AUDIT_ID = "rod-fixture-audit-v1-20260927"
SCENE_ID = "rod-fixture-scenes-v1-20260927"
CAMERA_ID = "rod-fixture-calibration-v1-20260927"
FINITE_ID = "rod-fixture-finite-v1-20260927"
RUN = ROOT / ".runtime/experiments" / AUDIT_ID
SCENE = ROOT / ".runtime/experiments" / SCENE_ID
CAMERA = ROOT / ".runtime/experiments" / CAMERA_ID
FINITE = ROOT / ".runtime/experiments" / FINITE_ID
TRUTH = ROOT / "data/eval_gt" / SCENE_ID
EVALUATION = ROOT / "data/evaluation" / AUDIT_ID
PUBLIC = ROOT / "docs/experiments/results/2026-09-27-rod-fixture-physical.json"
PUBLIC_AUDIT = ROOT / "docs/experiments/results/2026-09-27-rod-fixture-audit.json"
CASE_IDS = ["r01", "r02", "r03"]
VIEW_IDS = [f"view_{i:02d}" for i in range(5)]
SOURCES = {"scripts/audit_rod_fixture.py", "tests/test_fixture_audit.py", "scripts/thin_pack_gt.py"}


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


def block_truth(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    roots = [ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results", SCENE / "rendered-rgb"]
    if any(path.is_relative_to(folder) for folder in roots) or path.name in {"protocol.json", "generation-checks.json"} or path.name.endswith("render_request.json"):
        raise PermissionError("Pre-audit cannot read truth, render requests, or physical results")


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    name = path.relative_to(ROOT).as_posix()
    value = digest(path)
    if expected is not None:
        assert value == expected, name
    assert name not in hashes or hashes[name] == value, name
    hashes[name] = value
    return value


def source_pairs(folder, mapping, hashes):
    for name, sha in mapping.items():
        receipt(ROOT / name, hashes, sha)
        receipt(folder / "source_snapshot" / name, hashes, sha)


def early_definition(hashes):
    frozen = read(RUN / "evaluation-freeze.json")
    receipt(RUN / "evaluation-freeze.json", hashes)
    assert frozen["gt_read"] is False and frozen["audit_run_id"] == AUDIT_ID
    assert frozen["protocol"]["physical_policy"] == POLICY
    for name, sha in frozen["source_sha256"].items():
        receipt(ROOT / name, hashes, sha)
        receipt(RUN / "evaluation_source_snapshot" / name, hashes, sha)
    return frozen


def freeze():
    sys.addaudithook(block_truth)
    early = early_definition({})
    sources = set(early["source_sha256"]) | SOURCES
    for folder in (SCENE, CAMERA, FINITE):
        sources.update(read(folder / "prepared.json")["source_sha256"])
    mapping = {}
    for name in sorted(sources):
        path, target = ROOT / name, RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("Preserve prior audit source snapshot")
        shutil.copy2(path, target)
        mapping[name] = digest(path)
        assert mapping[name] == digest(target)
    write(RUN / "prepared.json", dict(run_id=AUDIT_ID, source_sha256=mapping,
        evaluation_freeze_sha256=digest(RUN / "evaluation-freeze.json"),
        protocol_sha256=digest(ROOT / "configs/rod_fixture_evaluation_v1.json"),
        target_run_ids=[SCENE_ID, CAMERA_ID, FINITE_ID], created_at_utc=now(), gt_read=False))
    print("FIXTURE_AUDIT_FROZEN", len(mapping), "source pairs", flush=True)


def assert_close(a, b, label):
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10, err_msg=label)


def independent_observation_score(data, k, extrinsic):
    # Recompute in this auditor rather than trusting the runner's cached p95.
    xyz = np.array(data["points_world"], float).reshape(-1, 3)
    if not len(xyz):
        return dict(count=0, median_px=None, p95_px=None, maximum_px=None, positive_depth=False, errors_px=[])
    e = np.asarray(extrinsic, float)
    camera = np.c_[xyz, np.ones(len(xyz))] @ e[:3].T
    homogeneous = (np.asarray(k, float) @ camera.T).T
    xy = homogeneous[:, :2] / homogeneous[:, 2, None]
    errors = np.sqrt(np.sum((xy - np.asarray(data["corners_xy"]))**2, axis=1))
    return dict(count=len(xyz), median_px=float(np.median(errors)), p95_px=float(np.quantile(errors, .95)),
        maximum_px=float(errors.max()), positive_depth=bool(np.all(camera[:, 2] > 0)), errors_px=errors.tolist())


def same_score(a, b):
    for name in ("count", "positive_depth"):
        assert a[name] == b[name]
    for name in ("median_px", "p95_px", "maximum_px", "errors_px"):
        if a[name] is None:
            assert b[name] is None
        else:
            assert_close(a[name], b[name], name)


def cad_declaration_check(cad):
    markers = validate_cad(cad)
    assert set(markers) == set(range(30)) and cad["marker_side_m"] == .30
    assert cad["artwork"]["quad_side_to_marker_side"] == 1.25
    # The independent public drawing fixes ID -> panel/row/column and corners.
    # It is the input ruler, not a ruler recovered from the target's truth.
    for mid, marker in markers.items():
        panel, slot = divmod(mid, 15)
        row, col = divmod(slot, 3)
        center = np.array([[-1.45, .8, 0], [1.45, -.15, 0]][panel], float)
        center += [(col-1)*.42, 0, (2-row)*.42]
        offsets = np.array([[-.15,0,.15],[.15,0,.15],[.15,0,-.15],[-.15,0,-.15]])
        assert_close(marker["corners_world"], center + offsets, "declared CAD drawing")
        assert marker["role"] == ("validation" if col == 1 else "training")
        assert marker["plane_id"] == f"plane_{panel}"
    return dict(markers=30, training_markers=20, validation_markers=10,
        provenance="Public drawing frozen before renderer; same drawing is supplied to calibration")


def collect_before():
    from run_fixture_calibration import checked as checked_camera
    from run_rod_fixture_finite import checked as checked_finite
    prepared = read(RUN / "prepared.json")
    hashes = {}
    source_pairs(RUN, prepared["source_sha256"], hashes)
    receipt(RUN / "prepared.json", hashes)
    early_definition(hashes)
    receipt(RUN / "evaluation-freeze.json", hashes, prepared["evaluation_freeze_sha256"])
    receipt(ROOT / "configs/rod_fixture_evaluation_v1.json", hashes, prepared["protocol_sha256"])
    cp, scene_input = checked_camera()
    fp, finite_input, config = checked_finite()
    sp = read(SCENE / "prepared.json")
    for folder, parent in ((SCENE,sp),(CAMERA,cp),(FINITE,fp)):
        receipt(folder / "prepared.json", hashes)
        source_pairs(folder, parent["source_sha256"], hashes)
        for name, sha in parent.get("receipts", {}).items():
            receipt(ROOT / name, hashes, sha)
    scene_manifest = ROOT / "data/inputs" / SCENE_ID / "manifest.json"
    receipt(scene_manifest, hashes, sp["input_sha256"])
    receipt(ROOT / "data/inputs" / FINITE_ID / "manifest.json", hashes, fp["input_sha256"])
    receipt(FINITE / "method_config.json", hashes, fp["method_config_sha256"])
    receipt(FINITE / "source_freeze.json", hashes, fp["source_freeze_sha256"])
    receipt(SCENE / "generation-freeze.json", hashes, sp["generation_freeze_sha256"])
    generation = read(SCENE / "generation-freeze.json")
    assert generation["source_sha256"] == sp["source_sha256"] and generation["declaration_precedes_render"] is True
    assert generation["declared_cad_sha256"] == sp["declared_cad_sha256"]
    assert datetime.fromisoformat(generation["created_at_utc"]) < datetime.fromisoformat(cp["created_at_utc"])
    cad_path = ROOT / scene_input["declared_cad"]["path"]
    receipt(cad_path, hashes, sp["declared_cad_sha256"])
    receipt(ROOT / "configs/rod_fixture_cad_v1.json", hashes, sp["declared_cad_sha256"])
    cad = read(cad_path)
    drawing = cad_declaration_check(cad)
    for name, sha in scene_input["fixture_artwork_sha256"].items():
        receipt(ROOT / "data/inputs" / SCENE_ID / name, hashes, sha)
    ci, fi = read(CAMERA / "inference.json"), read(FINITE / "inference.json")
    receipt(CAMERA / "inference.json", hashes)
    receipt(FINITE / "inference.json", hashes)
    assert ci["gt_read"] is False and fi["gt_read"] is False and fi["state"] == "complete"
    assert ci["prepared_sha256"] == digest(CAMERA / "prepared.json")
    assert ci["source_sha256"] == cp["source_sha256"] and ci["input_sha256"] == cp["input_sha256"]
    assert ci["cad_sha256"] == cp["cad_sha256"] and ci["policy"] == CAMERA_POLICY
    assert fi["source_sha256"] == fp["source_sha256"] and fi["input_sha256"] == fp["input_sha256"]
    assert fi["config_sha256"] == fp["method_config_sha256"]
    for rows in (scene_input["cases"], finite_input["cases"], ci["cases"], fi["records"]):
        assert [r["case_id"] for r in rows] == CASE_IDS
    counts = dict(camera_frames=0, rod_method_rows=0, rgb_detection_replays=0, rgb_rod_evidence_replays=0)
    normal_states = []
    for scene_case, case, cc, index in zip(scene_input["cases"], finite_input["cases"], ci["cases"], fi["records"]):
        assert scene_case["frames"] == case["frames"] and case["cameras"] == cc["cameras"]
        assert case["camera_case_sha256"] == canonical_hash(cc)
        assert [f["view_id"] for f in case["frames"]] == VIEW_IDS
        assert [f["view_id"] for f in cc["frames"]] == [c["view_id"] for c in cc["cameras"]] == VIEW_IDS
        record_path = FINITE / index["path"]
        receipt(record_path, hashes, index["sha256"])
        record = read(record_path)
        assert record["case_id"] == case["case_id"] and record["input_case_sha256"] == canonical_hash(case)
        assert record["cameras"] == case["cameras"] and record["gt_read_during_inference"] is False
        result = record["result"]
        assert result["camera_input_sha256"] == canonical_hash(case["cameras"])
        assert result["evidence_input_sha256"] == canonical_hash(record["frames"])
        assert result["method_sha256"] == config["method_sha256"]
        assert result["camera_modified"] is False and result["target_geometry_used"] is False
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
            cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
            cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        assert cc["flags"] == flags and not flags & cv2.CALIB_FIX_PRINCIPAL_POINT
        width, height = case["frames"][0]["size_wh"]
        focal = CAMERA_POLICY["initial_focal_image_max_fraction"] * max(width, height)
        assert_close(cc["initial_K"], [[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]], "declared initial K")
        images = []
        for frame, detected, camera in zip(case["frames"], cc["frames"], cc["cameras"]):
            receipt(ROOT / frame["rgb"], hashes, frame["rgb_sha256"])
            with Image.open(ROOT / frame["rgb"]) as image:
                assert list(image.size) == frame["size_wh"]
                rgb = np.asarray(image.convert("RGB"))
            images.append(rgb)
            assert detect_markers(rgb, cad) == detected["detection"]
            assert detected["rgb_sha256"] == frame["rgb_sha256"]
            reasons = []
            for role in ("training", "validation"):
                obs = observations(detected["detection"], cad, role)
                assert obs == camera[role]
                if camera["K_index"] is not None:
                    score = independent_observation_score(obs, camera["K_index"], camera["world_to_camera_cv"])
                    same_score(score, camera[role+"_score"])
                    if role == "training":
                        assert coverage_reasons(obs, cad, role) == []
                    if role == "validation":
                        reasons.extend(coverage_reasons(obs, cad, role))
                    if not score["positive_depth"]:
                        reasons.append(role + "_cheirality")
                    if score["p95_px"] is None or score["p95_px"] > CAMERA_POLICY["maximum_"+role+"_p95_px"]:
                        reasons.append(role+"_residual")
            assert not set(camera["training"]["marker_ids"]) & set(camera["validation"]["marker_ids"])
            if camera["K_index"] is not None:
                assert camera["state"] == ("validated" if not reasons else "withheld")
                assert sorted(reasons) == sorted(camera["reasons"])
                assert_close(camera["K_index"], cc["cameras"][0]["K_index"], "shared case K")
                k = np.asarray(camera["K_index"])
                assert k[0,1] == 0 and k[0,0] == k[1,1] and k[0,0] > 0
                assert np.all(np.asarray(cc["distortion"]) == 0)
            else:
                assert camera["state"] == "unavailable"
            counts["camera_frames"] += 1
            counts["rgb_detection_replays"] += 1
        replay = extract_fixture_evidence(case["frames"], images, config["method"])
        assert canonical_hash(replay) == canonical_hash(record["frames"])
        counts["rgb_rod_evidence_replays"] += len(replay)
        assert [m["method"] for m in result["methods"]] == ["baseline", "cylinder_support"]
        assert result["camera_gate"]["passed"] == all(c["state"] == "validated" for c in case["cameras"])
        for method in result["methods"]:
            segments = np.asarray(method["segments"], float).reshape(-1,2,3)
            assert len(segments) == method["segment_count"] and np.isfinite(segments).all()
            assert_close(np.linalg.norm(segments[:,1]-segments[:,0], axis=1).sum(), method["total_length_m"], "reported arc length")
            assert_close(segments.reshape(-1,3), np.asarray(method["endpoints"]).reshape(-1,3), "native endpoints")
            if method["state"] != "accepted" or not result["camera_gate"]["passed"]:
                assert not len(segments)
            counts["rod_method_rows"] += 1
            normal_states.append(dict(case_id=case["case_id"], method=method["method"], state=method["state"], segments=len(segments)))
    assert counts == dict(camera_frames=15, rod_method_rows=6, rgb_detection_replays=15, rgb_rod_evidence_replays=15)
    return dict(hashes=hashes, counts=counts, normal_states=normal_states, declared_cad=drawing,
        scope="Hashes plus replay of frozen RGB extractors and independent reprojection arithmetic; not an independent detector implementation or OS sandbox")


def pre():
    if PUBLIC.exists() or EVALUATION.exists() or (RUN / "pre.json").exists():
        raise FileExistsError("A genuine pre must precede every physical evaluation")
    sys.addaudithook(block_truth)
    result = collect_before()
    write(RUN / "pre.json", dict(state="passed", audit_run_id=AUDIT_ID, created_at_utc=now(),
        evaluation_existed=False, gt_read=False, **result))
    print("FIXTURE_PRE_PASSED", len(result["hashes"]), "receipts", result["counts"], flush=True)


def check_before():
    before = read(RUN / "pre.json")
    assert before["state"] == "passed" and before["gt_read"] is False and before["evaluation_existed"] is False
    for name, sha in before["hashes"].items():
        assert digest(ROOT / name) == sha, name
    return before


def camera_physical(predicted, truth):
    edge, index = np.asarray(truth["K_edge"]), np.asarray(truth["K_index"])
    expected = edge.copy()
    expected[:2,2] -= .5
    assert_close(index, expected, "single edge -> index half pixel conversion")
    if predicted["K_index"] is None:
        return dict(state="unavailable", normal_state=predicted["state"], center_error_m=None,
            rotation_error_deg=None, focal_relative_error=None, principal_error_px=None)
    pk, pe = np.asarray(predicted["K_index"]), np.asarray(predicted["world_to_camera_cv"])
    te = np.asarray(truth["world_to_camera_cv"])
    pc, tc = -pe[:3,:3].T @ pe[:3,3], -te[:3,:3].T @ te[:3,3]
    cosine = (np.trace(pe[:3,:3] @ te[:3,:3].T)-1)/2
    return dict(state="scored", normal_state=predicted["state"], center_error_m=float(np.linalg.norm(pc-tc)),
        rotation_error_deg=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
        focal_relative_error=float((pk[0,0]-index[0,0])/index[0,0]),
        principal_error_px=float(np.linalg.norm(pk[:2,2]-index[:2,2])),
        predicted_center_world=pc.tolist(), truth_center_world=tc.tolist(), alignment_performed=False)


def evaluate_payload():
    from thin_pack_gt import MeshRays, check_blender_rays, verify_projection_roundtrip
    before = check_before()
    sp, truth = read(SCENE / "prepared.json"), read(TRUTH / "manifest.json")
    assert digest(TRUTH / "manifest.json") == sp["truth_sha256"]
    assert digest(TRUTH / "artifact_hashes.json") == sp["truth_artifacts_sha256"]
    truth_hashes = {}
    for name, sha in read(TRUTH / "artifact_hashes.json").items():
        receipt(TRUTH / name, truth_hashes, sha)
    receipt(TRUTH / "artifact_hashes.json", truth_hashes, sp["truth_artifacts_sha256"])
    assert truth["input_sha256"] == sp["input_sha256"] and truth["declared_cad_sha256"] == sp["declared_cad_sha256"]
    inputs = read(ROOT / "data/inputs" / SCENE_ID / "manifest.json")
    camera_index, finite_index = read(CAMERA / "inference.json"), read(FINITE / "inference.json")
    render = read(TRUTH / "render_manifest.json")
    assert [c["case_id"] for c in truth["cases"]] == [c["case_id"] for c in render["cases"]] == CASE_IDS
    cameras, rods, paired = [], [], []
    for clean, gt, rendered, predicted, entry in zip(inputs["cases"], truth["cases"], render["cases"], camera_index["cases"], finite_index["records"]):
        cid = clean["case_id"]
        assert {cid,gt["case_id"],rendered["case_id"],predicted["case_id"],entry["case_id"]} == {cid}
        assert [f["view_id"] for f in gt["frames"]] == [c["view_id"] for c in gt["cameras"]] == VIEW_IDS
        rays = MeshRays(TRUTH / gt["mesh_path"])
        for frame, gf, gc, rf, pc in zip(clean["frames"],gt["frames"],gt["cameras"],rendered["frames"],predicted["cameras"]):
            assert frame["view_id"] == gf["view_id"] == gc["view_id"] == pc["view_id"]
            assert gf["generation_frame_id"] == rf["frame_id"] and gf["rgb_sha256"] == frame["rgb_sha256"]
            assert {k:v for k,v in gc.items() if k != "view_id"} == rf["camera"]
            native = TRUTH / gf["array_directory"]
            depth = np.load(native / "depth_z.npy", allow_pickle=False)
            ray_check = check_blender_rays(rays,gc,rf["blender_ray_probes"],.0001)
            projection = verify_projection_roundtrip(gc,depth)
            assert ray_check == gf["ray_check"]
            paired.append(dict(case_id=cid,view_id=frame["view_id"],rgb_sha256=frame["rgb_sha256"],
                ray_check=ray_check,projection_max_px=projection))
            cameras.append(dict(case_id=cid,view_id=frame["view_id"],**camera_physical(pc,gc)))
        declared = gt["declared"]
        target = declared["target"]
        segments = target.get("segments",[target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        record = read(FINITE / entry["path"])
        for method in record["result"]["methods"]:
            score = score_finite_structure(method["segments"],segments,gaps)
            rods.append(dict(case_id=cid,method=method["method"],normal_state=method["state"],
                normal_record_sha256=entry["sha256"],prediction_segments=method["segments"],truth_segments=segments,
                declared_gap_segments=gaps,rejection_reasons=method["rejection_reasons"],physical=score))
    assert len(cameras)==15 and len(rods)==6 and len(paired)==15
    return dict(audit_run_id=AUDIT_ID,state="evaluated",pre_sha256=digest(RUN / "pre.json"),
        evaluation_definition_sha256=digest(RUN / "evaluation-freeze.json"),prepared_sha256=digest(RUN / "prepared.json"),
        normal_receipt_count=len(before["hashes"]),truth_hashes=truth_hashes,policy=POLICY,
        camera_rows=cameras,rod_rows=rods,pairing_checks=paired,alignment_performed=False,
        scope="Changed declared fixture acquisition on three old procedural objects; no generic calibration, automatic identity, dense point-cloud repair, or new qualification claim")


def evaluate():
    if EVALUATION.exists() or PUBLIC.exists():
        raise FileExistsError("Preserve prior physical evaluation")
    result = evaluate_payload()
    EVALUATION.mkdir(parents=True)
    write(EVALUATION / "summary.json",result)
    write(PUBLIC,result)
    print("FIXTURE_PHYSICAL_EVALUATED 15 cameras / 6 finite rows",flush=True)


def post():
    before = check_before()
    public = read(PUBLIC)
    assert PUBLIC.read_bytes() == (EVALUATION / "summary.json").read_bytes()
    recomputed = evaluate_payload()
    assert public == recomputed, "Independent audit replay differs from published physical table"
    result = dict(state="passed",audit_run_id=AUDIT_ID,created_at_utc=now(),
        pre_sha256=digest(RUN / "pre.json"),prepared_sha256=digest(RUN / "prepared.json"),
        normal_receipts_unchanged=len(before["hashes"]),truth_receipts_checked=len(public["truth_hashes"]),
        camera_rows=15,rod_method_rows=6,rgb_gt_pairs=15,normal_replay_counts=before["counts"],
        physical_sha256=digest(PUBLIC),public_evaluation_bytes_equal=True,
        physical_rows_recomputed=True,coordinate_alignment_performed=False,
        limitations=["Known CAD drawing and ideal rendering; not a physical manufactured fixture",
            "Read guards and source receipts are application evidence, not OS isolation",
            "RGB detector/extractor replay uses frozen implementations; reprojection arithmetic is separate",
            "Physical recomputation uses the same frozen, known-answer-tested metric implementation",
            "Shared camera K fits train markers in all five views; held-out IDs test location, not held-out camera poses",
            "Candidate search is complete only inside retained cap8 pools; the strip declares one target"])
    write(RUN / "post.json",result)
    write(PUBLIC_AUDIT,result)
    print("FIXTURE_POST_PASSED",len(before["hashes"]),"unchanged pre receipts",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","pre","evaluate","post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    globals()[args.stage]()
