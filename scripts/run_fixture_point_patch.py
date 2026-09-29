"""Fresh conditioned DA3 depth -> full points -> reversible rods -> shared readout."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_eval.common_readout_abstention import readout  # noqa: E402
from creator_eval.fixture_depth_contract import (  # noqa: E402
    camera_inputs,
    exported_cameras,
    source_roundtrip,
)
from creator_eval.rod_fixture_finite import (  # noqa: E402
    canonical_hash,
    narrow_method_policy,
    reconstruct_fixture_narrow,
)
from creator_recon.domain.camera_point_snapshot import materialize  # noqa: E402
from creator_recon.domain.point_patch import (  # noqa: E402
    compose,
    open_candidate_view,
    write_candidate_view,
    write_patch,
    write_snapshot,
)
from run_fixture_challenges import (  # noqa: E402
    block_truth,
    digest,
    now,
    physical_summary,
    read,
    receipt,
    write,
)

RUN_ID = "fixture-point-patch-v1-20260929"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
CONFIG = ROOT / "configs/fixture_point_patch_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-09-29-fixture-point-patch.json"
AUDIT = ROOT / "docs/experiments/results/2026-09-29-fixture-point-patch-audit.json"


def checked():
    before = read(RUN / "pre.json")
    if before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]:
        raise ValueError("Invalid pre-inference receipt")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen input/source changed: " + name)
    return before, read(INPUTS / "manifest.json"), read(CONFIG)


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError("Preserve earlier point-patch attempt")
    config = read(CONFIG)
    if config["run_id"] != RUN_ID or config["align_to_input_ext_scale"] is not True:
        raise ValueError("Wrong conditioned point protocol")
    parent = ROOT / ".runtime/experiments" / config["source_rod_run_id"]
    scene = ROOT / ".runtime/experiments" / config["scene_run_id"]
    hashes = {}
    for path in (parent / "prepared.json", parent / "inference.json", parent / "method_config.json", scene / "prepared.json"):
        receipt(path, hashes)
    previous, method = read(parent / "inference.json"), read(parent / "method_config.json")["method"]
    if previous["state"] != "complete" or previous["gt_read"]:
        raise ValueError("Parent normal inference incomplete")
    narrow_method_policy(method)
    scene_path = ROOT / "data/inputs" / config["scene_run_id"] / "manifest.json"
    receipt(scene_path, hashes, read(scene / "prepared.json")["input_sha256"])
    scene_input = read(scene_path)
    if [r["case_id"] for r in previous["records"]] != config["case_ids"]:
        raise ValueError("Parent case order differs")
    cases = []
    for entry, source in zip(previous["records"], scene_input["cases"]):
        path = parent / entry["path"]
        receipt(path, hashes, entry["sha256"])
        record = read(path)
        if record["case_id"] != source["case_id"] or record["gt_read_during_inference"]:
            raise ValueError("Parent record detached")
        camera_inputs(record["cameras"])
        for frame, original in zip(record["frames"], source["frames"]):
            if frame["rgb_sha256"] != original["rgb_sha256"] or frame["view_id"] != original["view_id"]:
                raise ValueError("RGB/camera source pairing differs")
            receipt(ROOT / frame["rgb"], hashes, frame["rgb_sha256"])
        cases.append(dict(case_id=entry["case_id"], parent_record=path.relative_to(ROOT).as_posix(),
            parent_sha256=entry["sha256"], frames=source["frames"], cameras=record["cameras"]))
    sources = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments/src/creator_eval").glob("*.py")}
    sources.update({
        "scripts/run_fixture_point_patch.py", "scripts/run_fixture_challenges.py", "scripts/smoke_da3.py",
        "scripts/thin_pack_oracle_worker.py", "scripts/thin_pack_gt.py", "scripts/environment_paths.py",
        "tests/test_fixture_depth_contract.py", "configs/fixture_point_patch_v1.json", "configs/models.lock.json",
        "configs/rod_fixture_narrow_v1.json", "configs/rod_fixture_evaluation_v1.json",
        "reconstruction/src/creator_recon/domain/point_patch.py",
        "reconstruction/src/creator_recon/domain/camera_point_snapshot.py",
    })
    # Record actual installed upstream Python, not merely a second checkout.
    spec = importlib.util.find_spec("depth_anything_3")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    if len(locations) != 1:
        raise ValueError("One installed DA3 package location required")
    upstream = Path(locations[0])
    sources.update(p.relative_to(ROOT).as_posix() for p in upstream.rglob("*.py"))
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    for name in sorted(sources):
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, cases=cases, method=method, truth_excluded=True))
    receipt(INPUTS / "manifest.json", hashes)
    write(RUN / "evaluation-freeze.json", dict(created_at_utc=now(), config=config,
        physical_policy=read(ROOT / "configs/rod_fixture_evaluation_v1.json")["physical_policy"],
        gt_read=False, expected_readout_rows=18, reader_qualified=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), hashes=hashes,
        source_count=len(sources), gt_read=False, inference_existed=False))
    checked()
    print("FIXTURE_POINTS_PREPARED", len(hashes), "receipts", flush=True)


def depth(case_id):
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    case = next(c for c in manifest["cases"] if c["case_id"] == case_id)
    folder = RUN / case_id
    folder.mkdir(exist_ok=False)
    from smoke_da3 import run
    from thin_pack_oracle_worker import observe_oracle_inference

    edge, extrinsics = camera_inputs(case["cameras"])
    observer_input = dict(frames=[dict(K_edge=k.tolist(), width=f["size_wh"][0], height=f["size_wh"][1])
                                 for k, f in zip(edge, case["frames"])])
    audit, raw = {}, {}
    report = run(argparse.Namespace(images=[ROOT / f["rgb"] for f in case["frames"]],
                 model=config["model"], process_res=config["process_res"], use_ray_pose=False), ROOT, folder,
        inference_options=dict(extrinsics=extrinsics, intrinsics=edge, align_to_input_ext_scale=True),
        inference_context=functools.partial(observe_oracle_inference, audit=audit, raw_arrays=raw, cameras=observer_input))
    with np.load(folder / "prediction.npz", allow_pickle=False) as native:
        exported_cameras(native, case["cameras"], case["frames"][0]["size_wh"])
    with (folder / "raw-before-api.npz").open("xb") as stream:
        np.savez_compressed(stream, **raw)
    report.update(scope="Fresh RGB-estimated-fixture-conditioned depth; NOT oracle camera or unconstrained pose recovery",
        state="complete", case_id=case_id, gt_read=False, camera_input_sha256=canonical_hash(case["cameras"]),
        prediction_sha256=digest(folder / "prediction.npz"), raw_sha256=digest(folder / "raw-before-api.npz"),
        api_observation=audit, principal_point_encoder_offset_px=(edge[:, :2, 2]-np.array(case["frames"][0]["size_wh"])/2).tolist(),
        encoder_models_principal_point=False)
    write(folder / "depth-report.json", report)
    checked()
    print("FIXTURE_DEPTH", case_id, report["inference_seconds"], "s", flush=True)


def compose_case(case, method, config):
    folder = RUN / case["case_id"]
    report = read(folder / "depth-report.json")
    if (report["state"] != "complete" or report["gt_read"] or report["prediction_sha256"] != digest(folder / "prediction.npz")
            or report["raw_sha256"] != digest(folder / "raw-before-api.npz")
            or report["camera_input_sha256"] != canonical_hash(case["cameras"])):
        raise ValueError("Depth provenance differs")
    with np.load(folder / "prediction.npz", allow_pickle=False) as native:
        depth_array = native["depth"].copy()
        k, e, cameras = exported_cameras(native, case["cameras"], case["frames"][0]["size_wh"])
    parent = read(ROOT / case["parent_record"])
    rods = reconstruct_fixture_narrow(parent["frames"], cameras, method)
    write(folder / "rods.json", dict(cameras=cameras, result=rods, gt_read=False,
                                    prediction_sha256=report["prediction_sha256"]))
    points, ids = materialize(depth_array, k, e)
    roundtrip = source_roundtrip(points, ids, depth_array, k, e)
    frames = [dict(frame_id=f["view_id"], image_sha256=f["rgb_sha256"],
                   prediction_size_wh=[depth_array.shape[2], depth_array.shape[1]]) for f in case["frames"]]
    bundle = folder / "bundle"
    snapshot_id = write_snapshot(bundle / "base", points, ids, frames=frames,
        world_frame_id="fixture-conditioned-prediction:"+report["prediction_sha256"], length_unit="meter",
        source_prediction_sha256=report["prediction_sha256"],
        point_policy="All finite positive NEW conditioned depth pixels; declared fixture metre scale, not independently certified metric accuracy")
    base = compose(bundle / "base")
    collections, patches = {"base": base}, []
    for result in rods["methods"]:
        name, segments = result["method"], np.asarray(result["segments"], float).reshape(-1, 2, 3)
        evidence = [dict(decision="accept", line_ids=[f"line-{i:04d}" for i in range(len(segments))],
            suppression_range=None, view_ids=[f["frame_id"] for f in frames], source_sha256=digest(folder / "rods.json"),
            note="Research finite curves recomputed using this exact depth export camera; no surface or identity certification")] if len(segments) else []
        patch_id = write_patch(bundle / ("patch-"+name), bundle / "base", segments, np.empty((0, 3), np.uint32),
            selection_sha256=canonical_hash(case["frames"]), method=dict(id=name, version="fixture-conditioned-1",
            config_sha256=canonical_hash(method), seed=method["image_hypotheses"]["seed"]), evidence=evidence,
            unresolved=[] if len(segments) else result["rejection_reasons"])
        for enabled, suffix in ((True, "enabled"), (False, "withdrawn")):
            path = bundle / (name+"-"+suffix+".json")
            write_candidate_view(path, bundle / "base", bundle / ("patch-"+name), enabled=enabled)
            opened = open_candidate_view(path)
            if any(opened[key].tobytes() != base[key].tobytes() for key in ("points", "point_ids")):
                raise ValueError("Candidate changed base points or rollback bytes")
            if not np.array_equal(opened["segments"], segments if enabled else np.empty((0, 2, 3))):
                raise ValueError("Saved view geometry differs")
            if enabled:
                collections[name] = opened
        patches.append(dict(method=name, patch_id=patch_id, segment_count=len(segments)))
    region = dict(minimum_views=config["region_minimum_views"], views=[dict(view_id=f["view_id"],
        K_index=c["K_index"], world_to_camera_cv=c["world_to_camera_cv"],
        guides=[dict(xyxy=f["guide_xyxy"], half_width_px=config["region_half_width_px"])]) for f, c in zip(case["frames"], cameras)])
    write(folder / "region.json", region)
    centers = np.linalg.inv(np.concatenate((e, np.broadcast_to([0, 0, 0, 1], (5, 1, 4))), axis=1))[:, :3, 3]
    span = float(np.linalg.norm(centers[:, None]-centers, axis=-1).max())
    graphs = []
    for fraction in config["voxel_camera_span_fractions"]:
        policy = {**config["readout"], "voxel_size": span*fraction}
        for name, collection in collections.items():
            output = readout(collection["points"], collection["segments"], policy, region)
            path = folder / f"readout-{fraction}-{name}.json"
            write(path, output)
            graphs.append(dict(variant=name, fraction=fraction, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                state=output["state"], resolution_state=output["resolution_state"], segment_count=len(output["segments"]),
                region_point_count=output["region_point_count"], occupied_voxels=output["occupied_voxels"]))
            print("FIXTURE_READOUT", case["case_id"], fraction, name, output["state"], len(output["segments"]), flush=True)
    row = dict(case_id=case["case_id"], snapshot_id=snapshot_id, full_point_count=len(points), patches=patches,
        prediction_sha256=report["prediction_sha256"], graphs=graphs, roundtrip=roundtrip,
        no_suppression=True, base_bytes_preserved=True, withdrawn_bytes_equal=True, saved_views_reopened=True, gt_read=False)
    write(folder / "integration.json", row)
    return row


def infer():
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    if (RUN / "inference.json").exists() or any((RUN / cid).exists() for cid in config["case_ids"]):
        raise FileExistsError("Preserve prior inference attempt")
    started, rows = time.perf_counter(), []
    for case in manifest["cases"]:
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "depth", "--case", case["case_id"]], cwd=ROOT, check=True)
        rows.append(compose_case(case, manifest["method"], config))
    checked()
    outputs = {}
    for cid in config["case_ids"]:
        for path in sorted((RUN / cid).rglob("*")):
            if path.is_file():
                receipt(path, outputs)
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", rows=rows, outputs=outputs,
        gt_read=False, pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started))
    print("FIXTURE_POINTS_COMPLETE", sum(r["full_point_count"] for r in rows), "points", flush=True)


def evaluation_payload():
    before, _, config = checked()
    inference = read(RUN / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"] or inference["pre_sha256"] != digest(RUN / "pre.json"):
        raise ValueError("Incomplete normal inference")
    for name, sha in inference["outputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Inference output changed: "+name)
    scene = ROOT / ".runtime/experiments" / config["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / config["scene_run_id"] / "manifest.json"
    if digest(truth_path) != read(scene / "prepared.json")["truth_sha256"]:
        raise ValueError("Scene truth changed")
    truth = {c["case_id"]: c for c in read(truth_path)["cases"]}
    rows = []
    for record in inference["rows"]:
        gt = truth[record["case_id"]]
        target = gt["declared"]["target"]
        segments = target.get("segments", [target["endpoints"]]) if target["present"] else []
        gaps = [gt["declared"]["gap_segment"]] if "gap_segment" in gt["declared"] else []
        for entry in record["graphs"]:
            graph = read(RUN / entry["path"])
            score, error = None, None
            if graph["state"] == "complete":
                try:
                    score = physical_summary(graph["segments"], segments, gaps)
                except ValueError as exc:
                    error = str(exc)
            rows.append(dict(case_id=record["case_id"], **entry, physical=score, scoring_error=error,
                unresolved_component_count=graph.get("unresolved_component_count"), reason=graph.get("reason")))
    if len(rows) != 18 or [r["case_id"] for r in inference["rows"]] != config["case_ids"]:
        raise ValueError("Incomplete paired inventory")
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), output_receipts=len(inference["outputs"]),
        elapsed_seconds=inference["elapsed_seconds"], integration=[{k:v for k,v in r.items() if k != "graphs"} for r in inference["rows"]],
        readout_rows=rows, reader_qualified=False, g1_passed=False, alignment_performed=False, scope=config["scope"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve previous evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_POINTS_EVALUATED", len(result["readout_rows"]), "rows", flush=True)


def post():
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(),
        public_sha256=digest(PUBLIC), unchanged_pre_receipts=public["unchanged_pre_receipts"],
        output_receipts=public["output_receipts"], exact_evaluation_replay=True,
        source_pixel_roundtrip_checked=True, saved_views_and_rollback_checked=True, reader_qualified=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("FIXTURE_POINTS_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "depth", "evaluate", "post"))
    parser.add_argument("--case", choices=("r01", "r02", "r03"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "depth":
        if args.case is None:
            parser.error("depth requires --case")
        depth(args.case)
    else:
        globals()[args.stage]()
