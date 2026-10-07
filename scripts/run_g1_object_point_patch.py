"""Fresh two-object DA3 depth, complete point snapshots and four reversible patches."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import run_g1_object_rgb as parent

ROOT = parent.ROOT
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_eval import fixture_naive_controls as controls  # noqa: E402
from creator_eval.fixture_depth_contract import (  # noqa: E402
    camera_inputs,
    exported_cameras,
    source_roundtrip,
)
from creator_eval.rgb_candidate_readout import POLICY, support_views, support_votes  # noqa: E402
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

read, write, digest, receipt, block_truth = parent.read, parent.write, parent.digest, parent.receipt, parent.block_truth
RUN_ID = "g1-object-point-patch-v1-20261006"
RUN, INPUTS = ROOT / ".runtime/experiments" / RUN_ID, ROOT / "data/inputs" / RUN_ID
CONFIG = ROOT / "configs/g1_object_point_patch_v1.json"
TEMPLATE = ROOT / ".runtime/experiments/fixture-point-patch-v1-20260929"
OWN_SOURCES = {"scripts/run_g1_object_point_patch.py", "configs/g1_object_point_patch_v1.json",
    "tests/test_g1_object_point_patch.py", "experiments/src/creator_eval/fixture_naive_controls.py",
    "tests/test_fixture_naive_fit.py", "tests/test_fixture_naive_fit_deepseek.py",
    "experiments/src/creator_eval/rgb_candidate_readout.py",
    "reconstruction/src/creator_recon/__init__.py", "reconstruction/src/creator_recon/domain/__init__.py"}


def runtime():
    return dict(**parent.runtime(), installed_packages={name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "depth-anything-3")})


def validated_config():
    config = read(CONFIG)
    expected = dict(run_id=RUN_ID, source_rod_run_id=parent.RUN_ID,
        source_rod_prepared_sha256="9ee22dc740bf261bb0931a64984e42ad74daac8ac29c99f7d5d65653e3cc28cc",
        source_rod_inference_sha256="4fb1c2ceca56a32288fb83c577daa68aeef5d180aaa2d4eba1bc36c6f811327b",
        scene_run_id=parent.SCENE_ID, source_template_run_id=TEMPLATE.name,
        source_template_pre_sha256="b6d53cfcaf9b4c681f208ae360c57feb4fca72f14f50623e041b330002e3436a",
        case_ids=["chair01", "aframe01"], view_ids=[f"view_{i:02d}" for i in range(5)],
        patch_methods=["baseline", "cylinder_support", *controls.METHODS], model="large", process_res=504,
        align_to_input_ext_scale=True, depth_repeats_per_object=1, expected_objects=2, expected_patch_count=8,
        narrow_method_sha256="b33ce471de3fbe6b7387d78494c7675108a68793fa9d68711a5eb6a1f4c3072e",
        naive_defaults=controls.DEFAULTS, support_policy=POLICY)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen object point-patch protocol differs")
    return config


def source_inventory(template_pre, rgb_pre, upstream_names):
    prefix = TEMPLATE.relative_to(ROOT).as_posix() + "/source_snapshot/"
    inherited = {name[len(prefix):] for name in template_pre["hashes"] if name.startswith(prefix)}
    if len(inherited) != template_pre["source_count"]:
        raise ValueError("Closed original point source inventory differs")
    return sorted(inherited | set(rgb_pre["source_sha256"]) | OWN_SOURCES | set(upstream_names))


def installed_upstream_sources():
    spec = importlib.util.find_spec("depth_anything_3")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    if len(locations) != 1:
        raise ValueError("One already installed DA3 source package required; never auto-install")
    package = Path(locations[0]).resolve()
    if not package.is_relative_to(ROOT):
        raise ValueError("DA3 package must be the project-local locked installation")
    return sorted(path.relative_to(ROOT).as_posix() for path in package.rglob("*.py"))


def verify_models(config, hashes):
    lock_path = ROOT / "configs/models.lock.json"
    receipt(lock_path, hashes)
    lock = read(lock_path)
    selected = [row for row in lock["models"] if row["name"] == config["model"]]
    if len(selected) != 1:
        raise ValueError("One pinned LARGE model required")
    model = selected[0]
    for entry in model["files"]:
        path = ROOT / "models" / f"da3-{config['model']}" / model["revision"] / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or not entry["sha256"]:
            raise ValueError("Missing/incomplete locked local model; downloads are not permitted")
        receipt(path, hashes, entry["sha256"])
    return dict(model=model["repo_id"], revision=model["revision"], upstream_commit=lock["upstream_commit"])


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or before["runtime"] != runtime() or before["config"] != config):
        raise ValueError("Invalid object point freeze/runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen input/source changed: " + name)
    manifest = read(INPUTS / "manifest.json")
    if ([case["case_id"] for case in manifest["cases"]] != config["case_ids"]
            or manifest["run_id"] != RUN_ID or not manifest["truth_excluded"]
            or canonical_hash(manifest["method"]) != config["narrow_method_sha256"]):
        raise ValueError("Frozen complete normal object inventory/method differs")
    narrow_method_policy(manifest["method"])
    return before, manifest, config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError("Preserve every previous object point-patch attempt")
    config, hashes = validated_config(), {}
    receipt(parent.RUN / "prepared.json", hashes, config["source_rod_prepared_sha256"])
    receipt(parent.RUN / "inference.json", hashes, config["source_rod_inference_sha256"])
    rgb_pre, scene, method_config = parent.checked()
    normal, records = parent.verified_inference()
    for name, sha in rgb_pre["receipts"].items():
        receipt(ROOT / name, hashes, sha)
    receipt(parent.RUN / "method_config.json", hashes, rgb_pre["method_config_sha256"])
    receipt(parent.INPUTS / "manifest.json", hashes, rgb_pre["input_sha256"])
    receipt(TEMPLATE / "pre.json", hashes, config["source_template_pre_sha256"])
    template_pre = read(TEMPLATE / "pre.json")
    cases = []
    for entry, record, source in zip(normal["records"], records, scene["cases"]):
        receipt(parent.RUN / entry["path"], hashes, entry["sha256"])
        if (record["state"] != "complete" or record["case_id"] != source["case_id"]
                or [frame["view_id"] for frame in record["frames"]] != config["view_ids"]):
            raise ValueError("Complete original RGB evidence and validated cameras required")
        camera_inputs(record["cameras"])
        for frame, original in zip(record["frames"], source["frames"]):
            if any(frame[key] != original[key] for key in parent.FRAME_FIELDS):
                raise ValueError("Normal RGB/camera/evidence pairing differs")
            receipt(ROOT / frame["rgb"], hashes, frame["rgb_sha256"])
        cases.append(dict(case_id=entry["case_id"], parent_record=(parent.RUN / entry["path"]).relative_to(ROOT).as_posix(),
            parent_sha256=entry["sha256"], frames=source["frames"], cameras=record["cameras"]))
    model_identity = verify_models(config, hashes)
    sources = source_inventory(template_pre, rgb_pre, installed_upstream_sources())
    prefix = TEMPLATE.relative_to(ROOT).as_posix() + "/source_snapshot/"
    for name, sha in template_pre["hashes"].items():
        if name.startswith(prefix):
            receipt(ROOT / name, hashes, sha)
            receipt(ROOT / name[len(prefix):], hashes, sha)
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    for name in sources:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, cases=cases, method=method_config["method"],
        truth_excluded=True, independent_real_capture=False))
    receipt(INPUTS / "manifest.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, hashes=hashes, source_count=len(sources), config=config,
        runtime=runtime(), model_identity=model_identity, gt_read=False, inference_existed=False,
        expected_objects=2, expected_patch_count=8, depth_repeats_per_object=1, g1_passed=False))
    checked()
    print("G1_OBJECT_POINTS_PREPARED", len(sources), "sources", len(hashes), "receipts", flush=True)


def depth(case_id):
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    cases = [case for case in manifest["cases"] if case["case_id"] == case_id]
    if len(cases) != 1:
        raise ValueError("One declared object required")
    case = cases[0]
    folder = RUN / case_id
    folder.mkdir(exist_ok=False)
    from smoke_da3 import run as run_model
    from thin_pack_oracle_worker import observe_oracle_inference

    edge, extrinsics = camera_inputs(case["cameras"])
    observer_input = dict(frames=[dict(K_edge=k.tolist(), width=frame["size_wh"][0], height=frame["size_wh"][1])
                                 for k, frame in zip(edge, case["frames"])])
    audit, raw = {}, {}
    try:
        report = run_model(argparse.Namespace(images=[ROOT / frame["rgb"] for frame in case["frames"]],
            model=config["model"], process_res=config["process_res"], use_ray_pose=False), ROOT, folder,
            inference_options=dict(extrinsics=extrinsics, intrinsics=edge, align_to_input_ext_scale=True),
            inference_context=functools.partial(observe_oracle_inference, audit=audit, raw_arrays=raw, cameras=observer_input))
        with np.load(folder / "prediction.npz", allow_pickle=False) as native:
            exported_cameras(native, case["cameras"], case["frames"][0]["size_wh"])
        if set(raw) != {"depth", "intrinsics", "extrinsics"} or not audit:
            raise ValueError("Actual pre-API arrays and camera replacement audit required")
        with (folder / "raw-before-api.npz").open("xb") as stream:
            np.savez_compressed(stream, **raw)
        report.update(state="complete", case_id=case_id, gt_read=False,
            camera_input_sha256=canonical_hash(case["cameras"]), prediction_sha256=digest(folder / "prediction.npz"),
            raw_sha256=digest(folder / "raw-before-api.npz"), pre_sha256=digest(RUN / "pre.json"),
            api_observation=audit, principal_point_encoder_offset_px=(edge[:, :2, 2]-np.array(case["frames"][0]["size_wh"])/2).tolist(),
            encoder_models_principal_point=False, depth_repeats_for_object=1,
            scope="Fresh RGB-estimated-fixture-conditioned DA3; not generated/oracle cameras or learned pose recovery")
        write(folder / "depth-report.json", report)
    except Exception as exc:
        write(folder / "depth-failure.json", dict(state="error", case_id=case_id, gt_read=False,
            reason=type(exc).__name__ + ": " + str(exc), pre_sha256=digest(RUN / "pre.json")))
        raise
    checked()
    print("G1_OBJECT_DEPTH", case_id, report["inference_seconds"], "seconds", flush=True)


def verified_depth(case):
    folder = RUN / case["case_id"]
    if (folder / "depth-failure.json").exists():
        raise ValueError("A failed depth attempt cannot be relabelled complete")
    report = read(folder / "depth-report.json")
    if (report["state"] != "complete" or report["case_id"] != case["case_id"] or report["gt_read"]
            or report["pre_sha256"] != digest(RUN / "pre.json")
            or report["prediction_sha256"] != digest(folder / "prediction.npz")
            or report["raw_sha256"] != digest(folder / "raw-before-api.npz")
            or report["camera_input_sha256"] != canonical_hash(case["cameras"])
            or report["process_res"] != 504 or report["depth_repeats_for_object"] != 1
            or not report["api_observation"]):
        raise ValueError("Fresh depth/camera provenance differs")
    return report


def validate_candidate(base, candidate, segments, *, enabled):
    if any(candidate[key] != base[key] for key in ("base_snapshot_id", "world_frame_id", "length_unit")):
        raise ValueError("Candidate coordinate/snapshot identity changed")
    for key in ("points", "point_ids"):
        if (candidate[key].dtype != base[key].dtype or candidate[key].shape != base[key].shape
                or candidate[key].tobytes() != base[key].tobytes()):
            raise ValueError("Saved candidate changed complete base points or source IDs")
    expected = segments if enabled else np.empty((0, 2, 3))
    if not np.array_equal(candidate["segments"], expected):
        raise ValueError("Saved enabled/withdrawn geometry differs")


def persist_patch(bundle, base, frames, name, segments, *, source_sha256, selection_sha256,
                  method_sha256, seed, reasons, source_state, evidence_view_ids=None):
    segments = np.asarray(segments, float).reshape(-1, 2, 3)
    evidence = [dict(decision="accept", line_ids=[f"line-{i:04d}" for i in range(len(segments))],
        suppression_range=None, view_ids=[frame["frame_id"] for frame in frames] if evidence_view_ids is None else evidence_view_ids,
        source_sha256=source_sha256,
        note="Normal research curve recomputed in this exact conditioned depth export frame; no point suppression")] if len(segments) else []
    patch_dir = bundle / ("patch-" + name)
    patch_id = write_patch(patch_dir, bundle / "base", segments, np.empty((0, 3), np.uint32),
        selection_sha256=selection_sha256, method=dict(id=name, version="g1-object-point-1", config_sha256=method_sha256, seed=seed),
        evidence=evidence, unresolved=list(reasons))
    paths = {}
    for enabled, state in ((True, "enabled"), (False, "withdrawn")):
        path = bundle / f"{name}-{state}.json"
        write_candidate_view(path, bundle / "base", patch_dir, enabled=enabled)
        validate_candidate(base, open_candidate_view(path), segments, enabled=enabled)
        paths[state + "_path"] = path.relative_to(ROOT).as_posix()
    files = list(patch_dir.iterdir()) + [ROOT / path for path in paths.values()]
    return dict(method=name, patch_id=patch_id, segment_count=len(segments), source_state=source_state,
        outcome="accepted_change" if len(segments) else "no_supported_change", native_segments=segments.tolist(),
        source_sha256=source_sha256, selection_sha256=selection_sha256, **paths,
        added_storage_bytes=sum(path.stat().st_size for path in files),
        no_suppression=True, base_bytes_preserved=True, withdrawn_bytes_equal=True, saved_views_reopened=True)


def camera_span(extrinsics):
    homogeneous = np.concatenate((extrinsics, np.broadcast_to([0, 0, 0, 1], (len(extrinsics), 1, 4))), axis=1)
    centers = np.linalg.inv(homogeneous)[:, :3, 3]
    return float(np.linalg.norm(centers[:, None]-centers, axis=-1).max())


def compose_case(case, method, config):
    started = time.perf_counter()
    folder = RUN / case["case_id"]
    if (folder / "bundle").exists() or (folder / "integration.json").exists() or (folder / "rods.json").exists():
        raise FileExistsError("Preserve previous object integration, including partial attempts")
    report = verified_depth(case)
    with np.load(folder / "prediction.npz", allow_pickle=False) as native:
        depth_array = native["depth"].copy()
        k, e, cameras = exported_cameras(native, case["cameras"], case["frames"][0]["size_wh"])
    if digest(ROOT / case["parent_record"]) != case["parent_sha256"]:
        raise ValueError("Frozen parent RGB evidence changed")
    source = read(ROOT / case["parent_record"])
    rod_started = time.perf_counter()
    rods = reconstruct_fixture_narrow(source["frames"], cameras, method)
    rod_seconds = time.perf_counter()-rod_started
    if [row["method"] for row in rods["methods"]] != ["baseline", "cylinder_support"]:
        raise ValueError("Complete ordered recomputed rod method inventory required")
    write(folder / "rods.json", dict(cameras=cameras, result=rods, gt_read=False,
        prediction_sha256=report["prediction_sha256"], parent_record_sha256=case["parent_sha256"],
        cameras_recomputed_from_prediction_export=True, elapsed_seconds=rod_seconds))
    points, ids = materialize(depth_array, k, e)
    roundtrip = source_roundtrip(points, ids, depth_array, k, e)
    frames = [dict(frame_id=frame["view_id"], image_sha256=frame["rgb_sha256"],
        prediction_size_wh=[depth_array.shape[2], depth_array.shape[1]]) for frame in case["frames"]]
    bundle = folder / "bundle"
    snapshot_id = write_snapshot(bundle / "base", points, ids, frames=frames,
        world_frame_id="fixture-conditioned-prediction:" + report["prediction_sha256"], length_unit="meter",
        source_prediction_sha256=report["prediction_sha256"],
        point_policy="All finite positive NEW conditioned depth pixels; declared fixture metre scale, not independently certified metric accuracy")
    base = compose(bundle / "base")
    patches = []
    for result in rods["methods"]:
        patches.append(persist_patch(bundle, base, frames, result["method"], result["segments"],
            source_sha256=digest(folder / "rods.json"), selection_sha256=canonical_hash(case["frames"]),
            method_sha256=canonical_hash(method), seed=method["image_hypotheses"]["seed"],
            reasons=result["rejection_reasons"], source_state=result["state"]))
    views = support_views(source["frames"], cameras)
    votes = support_votes(points, views, config["support_policy"])
    mask, span = votes >= config["support_policy"]["minimum_views"], camera_span(e)
    fits = controls.fit_controls(points, ids, span, mask)
    if [fit["method"] for fit in fits] != list(controls.METHODS):
        raise ValueError("Complete ordered naive control inventory required")
    for fit in fits:
        if fit["input_support_count"] != int(mask.sum()) or fit["state"] != "complete" or fit["gt_read"]:
            raise ValueError("Naive control changed the normal RGB mask or failed")
        path = folder / f"{fit['method']}-fit.json"
        write(path, fit)
        patches.append(persist_patch(bundle, base, frames, fit["method"], fit["segments"],
            source_sha256=digest(path), selection_sha256=fit["input_support_sha256"],
            method_sha256=canonical_hash(controls.DEFAULTS), seed=controls.DEFAULTS["seed"],
            reasons=[fit["reason"]] if fit["reason"] else [], source_state=fit["outcome"],
            evidence_view_ids=[frames[int(index)]["frame_id"] for index, count in fit["source_view_counts"].items() if count]))
    if [patch["method"] for patch in patches] != config["patch_methods"]:
        raise ValueError("Complete four-patch inventory required")
    write(folder / "support.json", dict(gt_read=False, policy=config["support_policy"], camera_span_m=span,
        full_point_count=len(points), supported_point_count=int(mask.sum()),
        vote_histogram=np.bincount(votes, minlength=len(views)+1).tolist(),
        input_support_sha256=fits[0]["input_support_sha256"],
        selected_source_view_counts=fits[0]["source_view_counts"],
        camera_sha256=canonical_hash(cameras), raw_frame_sha256=canonical_hash(source["frames"])))
    row = dict(case_id=case["case_id"], state="complete", snapshot_id=snapshot_id, full_point_count=len(points),
        points_per_view=[int(np.sum(ids[:, 0] == index)) for index in range(5)], patches=patches,
        prediction_sha256=report["prediction_sha256"], camera_export_sha256=canonical_hash(cameras),
        parent_record_sha256=case["parent_sha256"], roundtrip=roundtrip, camera_span_m=span,
        supported_base_point_count=int(mask.sum()), support_sha256=fits[0]["input_support_sha256"],
        elapsed_seconds=time.perf_counter()-started, rod_recompute_seconds=rod_seconds,
        naive_fit_seconds={fit["method"]: fit["elapsed_seconds"] for fit in fits},
        depth_inference_seconds=report["inference_seconds"], depth_load_seconds=report["load_seconds"],
        gpu_peak_allocated_mib=report["peak_allocated_mib"],
        base_storage_bytes=sum(path.stat().st_size for path in (bundle / "base").iterdir()),
        patch_storage_bytes=sum(patch["added_storage_bytes"] for patch in patches),
        no_suppression=True, base_bytes_preserved=True, withdrawn_bytes_equal=True, saved_views_reopened=True,
        depth_repeats_for_object=1, end_to_end_repeat_performed=False, independent_real_capture=False,
        gt_read=False, g1_passed=False)
    write(folder / "integration.json", row)
    return row


def normal_records():
    """Verify complete normal artifact inventory before any downstream evaluator."""
    before, manifest, config = checked()
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json")
            or [row["case_id"] for row in normal["rows"]] != config["case_ids"]
            or normal["depth_repeats_per_object"] != 1 or normal["end_to_end_repeat_performed"]):
        raise ValueError("Complete two-object normal inference required")
    actual_paths = {path.relative_to(ROOT).as_posix() for cid in config["case_ids"]
                    for path in (RUN / cid).rglob("*") if path.is_file()}
    if set(normal["outputs"]) != actual_paths:
        raise ValueError("Complete normal output artifact inventory differs")
    for name, sha in normal["outputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Normal point/patch artifact changed: " + name)
    for case, row in zip(manifest["cases"], normal["rows"]):
        folder = RUN / case["case_id"]
        report = verified_depth(case)
        if (read(folder / "integration.json") != row or row["state"] != "complete" or row["gt_read"]
                or row["prediction_sha256"] != report["prediction_sha256"]
                or [item["method"] for item in row["patches"]] != config["patch_methods"]):
            raise ValueError("Normal integration identity or four-patch inventory differs")
        base = compose(folder / "bundle/base")
        if base["base_snapshot_id"] != row["snapshot_id"] or len(base["points"]) != row["full_point_count"]:
            raise ValueError("Complete base snapshot identity differs")
        for item in row["patches"]:
            segments = np.asarray(item["native_segments"], float).reshape(-1, 2, 3)
            for enabled, name in ((True, "enabled"), (False, "withdrawn")):
                path = ROOT / item[name + "_path"]
                if path != folder / f"bundle/{item['method']}-{name}.json":
                    raise ValueError("Normal saved view path differs")
                candidate = open_candidate_view(path)
                validate_candidate(base, candidate, segments, enabled=enabled)
                if candidate["patch_id"] != item["patch_id"]:
                    raise ValueError("Normal patch content identity differs")
    return before, manifest, config, normal


def infer():
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    if (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior complete object point inference")
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, gt_read=False,
        depth_repeats_per_object=1, case_ids=config["case_ids"]))
    started, rows = time.perf_counter(), []
    for case in manifest["cases"]:
        folder = RUN / case["case_id"]
        if not folder.exists():
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_g1_object_point_patch.py"),
                            "depth", "--case", case["case_id"]], cwd=ROOT, check=True, timeout=1800)
        # The optional explicit depth stage may already have created this exact
        # receipt. It is verified, never repeated or overwritten by infer.
        verified_depth(case)
        rows.append(compose_case(case, manifest["method"], config))
    checked()
    if [row["case_id"] for row in rows] != config["case_ids"] or sum(len(row["patches"]) for row in rows) != 8:
        raise ValueError("Complete two-object/eight-patch inventory required")
    outputs = {}
    for cid in config["case_ids"]:
        for path in sorted((RUN / cid).rglob("*")):
            if path.is_file():
                receipt(path, outputs)
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows, outputs=outputs,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started,
        depth_repeats_per_object=1, end_to_end_repeat_performed=False,
        independent_real_capture=False, g1_passed=False, evaluation_status="Normal point/patch integration only; no GT or readout"))
    normal_records()
    print("G1_OBJECT_POINTS_COMPLETE", sum(row["full_point_count"] for row in rows), "points", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "depth", "infer"))
    parser.add_argument("--case", choices=("chair01", "aframe01"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "depth":
        if args.case is None:
            parser.error("depth requires --case")
        depth(args.case)
    else:
        if args.case is not None:
            parser.error("--case is only valid for depth")
        globals()[args.stage]()
