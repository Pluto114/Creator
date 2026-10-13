"""Two objects x two fresh processes: RGB calibration, DA3, saved patches, readout.

Development evidence, not a blind test or automatic G1 qualification. No old
prediction, camera record or candidate is an input; existing outputs are never
resumed. Evaluate is a separate command after the complete normal inventory.
"""

import argparse
import functools
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import run_g1_object_point_patch as point
import run_g1_object_rgb as rgb
import smoke_da3
from creator_eval.rgb_occupied_axis_readout import OccupiedAxisReadout
from creator_eval.rgb_stroke_chain_support import build_context
from thin_pack_oracle_worker import observe_oracle_inference

ROOT = rgb.ROOT
CASES = ("chair01", "aframe01")
ARMS = ("base", "baseline", "cylinder_support", *point.controls.METHODS)
SCALES = (.0025, .005)
REPEATS = (0, 1)
ANCHORS = ROOT / "data/inputs/rgb-foreground-strokes-v1-20261009/manifest.json"
FLAGS = dict(development_run=True, blind_evaluation=False, independent_real_capture=False,
             reader_qualified=False, g1_passed=False)
read, write, digest = rgb.read, rgb.write, rgb.digest


def run_path(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}", name):
        raise ValueError("A single safe, unused run ID is required")
    return ROOT / ".runtime/experiments" / name


def array_hash(value):
    value = np.ascontiguousarray(value)
    digestor = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digestor.update(value.tobytes())
    return digestor.hexdigest()


def verify_hashes(hashes):
    for name, sha in hashes.items():
        if digest(ROOT / name) != sha:
            raise ValueError("Run input/source changed: " + name)


def prepare_plan(run):
    config, rgb_config = point.validated_config(), rgb.validated_config()
    scene_path = rgb.SCENE_INPUTS / "manifest.json"
    if digest(scene_path) != rgb_config["scene_input_sha256"]:
        raise ValueError("Original normal image inventory changed")
    scene = rgb.validate_manifest(read(scene_path), rgb_config)
    cad_path = ROOT / scene["declared_cad"]["path"]
    if digest(cad_path) != scene["declared_cad"]["sha256"]:
        raise ValueError("Declared metric fixture CAD changed")
    cad = read(cad_path)
    rgb.validate_cad(cad)
    method_path = rgb.RUN / "method_config.json"
    method = read(method_path)["method"]
    if rgb.canonical_hash(method) != config["narrow_method_sha256"]:
        raise ValueError("Original method parameters changed")
    rgb.narrow_method_policy(method)
    marks = read(ANCHORS)
    anchors = {case["case_id"]: case["anchors"] for case in marks["cases"] if case["case_id"] in CASES}
    if set(anchors) != set(CASES):
        raise ValueError("Shared foreground annotations for both objects required")
    inputs = [scene_path, cad_path, method_path, ANCHORS, point.CONFIG, rgb.CONFIG, ROOT / "configs/models.lock.json"]
    inputs.extend(ROOT / frame["rgb"] for case in scene["cases"] for frame in case["frames"])
    sources = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and not any(part.startswith(".venv") for part in path.parts):
                sources.add(path)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(inputs) | sources)}
    for path in sources:
        target = run / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return dict(run_id=run.name, cases=scene["cases"], cad=cad, method=method, config=config,
        anchors=anchors, direct_hashes=hashes, repeats=list(REPEATS), arms=list(ARMS), scales=list(SCALES),
        expected_workers=4, expected_rows=40, extra_annotations=0,
        inherited_annotations=sum(map(len, anchors.values())), gt_read=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        policy="Fresh RGB calibration and DA3 in each case/repeat subprocess; no output cache or resume",
        source_policy="Direct inputs and imported project sources fixed at dispatch; no recursive ancestry audit", **FLAGS)


def integrate(folder, case, source, plan, report):
    with np.load(folder / "depth/prediction.npz", allow_pickle=False) as native:
        depth = native["depth"].copy()
        prediction_arrays = {name: array_hash(native[name]) for name in native.files}
        k, e, cameras = point.exported_cameras(native, source["cameras"], case["frames"][0]["size_wh"])
    rods = rgb.reconstruct_fixture_narrow(source["frames"], cameras, plan["method"])
    if [r["method"] for r in rods["methods"]] != ["baseline", "cylinder_support"]:
        raise ValueError("Both original rod methods required")
    if any(r["state"] not in ("accepted", "rejected") for r in rods["methods"]):
        raise ValueError("Rod reconstruction failed; preserve worker failure")
    write(folder / "rods.json", dict(cameras=cameras, result=rods, gt_read=False,
        prediction_sha256=report["prediction_sha256"], cameras_recomputed_from_prediction_export=True))
    points, ids = point.materialize(depth, k, e)
    roundtrip = point.source_roundtrip(points, ids, depth, k, e)
    frames = [dict(frame_id=frame["view_id"], image_sha256=frame["rgb_sha256"],
        prediction_size_wh=[depth.shape[2], depth.shape[1]]) for frame in case["frames"]]
    bundle = folder / "bundle"
    point.write_snapshot(bundle / "base", points, ids, frames=frames,
        world_frame_id="fixture-conditioned-prediction:" + report["prediction_sha256"], length_unit="meter",
        source_prediction_sha256=report["prediction_sha256"],
        point_policy="All finite positive fresh conditioned depth pixels; no suppression or confidence filter")
    base = point.compose(bundle / "base")
    patches = []
    for result in rods["methods"]:
        patches.append(point.persist_patch(bundle, base, frames, result["method"], result["segments"],
            source_sha256=digest(folder / "rods.json"), selection_sha256=rgb.canonical_hash(case["frames"]),
            method_sha256=rgb.canonical_hash(plan["method"]), seed=plan["method"]["image_hypotheses"]["seed"],
            reasons=result["rejection_reasons"], source_state=result["state"]))
    votes = point.support_votes(points, point.support_views(source["frames"], cameras), point.POLICY)
    mask, span = votes >= point.POLICY["minimum_views"], point.camera_span(e)
    for fit in point.controls.fit_controls(points, ids, span, mask):
        if fit["state"] != "complete" or fit["gt_read"] or fit["input_support_count"] != int(mask.sum()):
            raise ValueError("Naive control failed or changed the shared support domain")
        write(folder / (fit["method"] + "-fit.json"), fit)
        patches.append(point.persist_patch(bundle, base, frames, fit["method"], fit["segments"],
            source_sha256=digest(folder / (fit["method"] + "-fit.json")), selection_sha256=fit["input_support_sha256"],
            method_sha256=rgb.canonical_hash(point.controls.DEFAULTS), seed=point.controls.DEFAULTS["seed"],
            reasons=[fit["reason"]] if fit["reason"] else [], source_state=fit["outcome"],
            evidence_view_ids=[frames[int(i)]["frame_id"] for i, count in fit["source_view_counts"].items() if count]))
    if [patch["method"] for patch in patches] != list(ARMS[1:]):
        raise ValueError("All four patches, including rejected/empty outcomes, required")
    context = build_context(source["frames"], cameras, plan["anchors"][case["case_id"]])
    adapter = OccupiedAxisReadout(base["points"], context)
    rows = []
    for arm in ARMS:
        patch = next((p for p in patches if p["method"] == arm), None)
        candidate = base if patch is None else point.open_candidate_view(ROOT / patch["enabled_path"])
        segments = np.empty((0, 2, 3)) if patch is None else candidate["segments"]
        for fraction in SCALES:
            started, error = time.perf_counter(), None
            try:
                full = adapter(segments, dict(voxel_size=span*fraction, origin=[0., 0., 0.]))
                graph = {key: value for key, value in full.items() if value is None or isinstance(value, (str, bool, int, float))}
                graph["segments"] = full["segments"]
                graph["selected_band_score"] = full.get("selected_band_score")
            except Exception as exc:
                error = repr(exc)
                graph = dict(state="error", reason=error, segments=[])
            rows.append(dict(case_id=case["case_id"], variant=arm, fraction=fraction, graph=graph,
                native_state="complete", native_segments=segments, native_source_state="base" if patch is None else patch["source_state"],
                error=error, reader_elapsed_seconds=time.perf_counter()-started))
    write(folder / "integration.json", dict(patches=patches, full_point_count=len(points), roundtrip=roundtrip,
        points_sha256=array_hash(points), point_ids_sha256=array_hash(ids), prediction_array_sha256=prediction_arrays,
        camera_sha256=rgb.canonical_hash(cameras), source_frame_sha256=rgb.canonical_hash(source["frames"]),
        camera_span_m=span, gt_read=False, base_bytes_preserved=True, withdrawn_bytes_equal=True, saved_views_reopened=True))
    return rows


def worker(run, repeat, case_id):
    sys.addaudithook(rgb.block_truth)
    plan = read(run / "plan.json")
    if repeat not in REPEATS or case_id not in CASES or plan["run_id"] != run.name:
        raise ValueError("Worker must belong to the fixed run inventory")
    folder = run / f"repeat-{repeat}" / case_id
    folder.mkdir(parents=True, exist_ok=False)
    started, stage = time.perf_counter(), "input_identity"
    write(folder / "started.json", dict(pid=os.getpid(), repeat=repeat, case_id=case_id,
        started_at=datetime.now(timezone.utc).isoformat(), plan_sha256=digest(run / "plan.json")))
    try:
        verify_hashes(plan["direct_hashes"])
        case = next(case for case in plan["cases"] if case["case_id"] == case_id)
        stage = "fresh_rgb_calibration"
        source = rgb.infer_case(case, plan["cad"], plan["method"])
        write(folder / "rgb.json", source)
        if source["state"] != "complete":
            raise RuntimeError("Fresh RGB failed: " + repr(source["errors"]))
        print("FRESH_RGB", repeat, case_id, source["elapsed_seconds"], flush=True)
        stage = "fresh_da3"
        edge, extrinsics = point.camera_inputs(source["cameras"])
        observer = dict(frames=[dict(K_edge=k.tolist(), width=f["size_wh"][0], height=f["size_wh"][1])
                               for k, f in zip(edge, case["frames"])])
        audit, raw = {}, {}
        depth_folder = folder / "depth"
        depth_folder.mkdir(exist_ok=False)
        report = smoke_da3.run(argparse.Namespace(images=[ROOT / f["rgb"] for f in case["frames"]],
            model="large", process_res=504, use_ray_pose=False), ROOT, depth_folder,
            inference_options=dict(extrinsics=extrinsics, intrinsics=edge, align_to_input_ext_scale=True),
            inference_context=functools.partial(observe_oracle_inference, audit=audit, raw_arrays=raw, cameras=observer))
        if set(raw) != {"depth", "intrinsics", "extrinsics"} or not audit:
            raise ValueError("Fresh API observation and pre-API arrays required")
        with (depth_folder / "raw-before-api.npz").open("xb") as stream:
            np.savez_compressed(stream, **raw)
        report.update(prediction_sha256=digest(depth_folder / "prediction.npz"), api_observation=audit,
            raw_sha256=digest(depth_folder / "raw-before-api.npz"), camera_input_sha256=rgb.canonical_hash(source["cameras"]),
            scope="Fresh RGB-calibrated fixture-conditioned DA3; not oracle camera or learned-pose accuracy", gt_read=False)
        write(depth_folder / "report.json", report)
        print("FRESH_DA3", repeat, case_id, report["inference_seconds"], flush=True)
        stage = "saved_patches_and_common_readout"
        rows = integrate(folder, case, source, plan, report)
        write(folder / "normal.json", dict(rows=rows, case_id=case_id, repeat=repeat, pid=os.getpid(),
            plan_sha256=digest(run / "plan.json"), gt_read=False, elapsed_seconds=time.perf_counter()-started, **FLAGS))
        print("FRESH_COMPLETE", repeat, case_id, len(rows), flush=True)
    except Exception as exc:
        write(folder / "failure.json", dict(stage=stage, reason=repr(exc), elapsed_seconds=time.perf_counter()-started,
            pid=os.getpid(), case_id=case_id, repeat=repeat, gt_read=False))
        raise


def normal(run):
    sys.addaudithook(rgb.block_truth)
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = prepare_plan(run)
    write(run / "plan.json", plan)
    rows, workers = [], []
    for repeat in REPEATS:
        for case_id in CASES:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--run-id", run.name,
                       "--repeat", str(repeat), "--case", case_id]
            label = f"r{repeat}-{case_id}"
            before = time.perf_counter()
            with (run / (label + ".stdout.log")).open("xb") as stdout, (run / (label + ".stderr.log")).open("xb") as stderr:
                process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            folder = run / f"repeat-{repeat}" / case_id
            capture = dict(repeat=repeat, case_id=case_id, command=command, exit_code=process.returncode,
                elapsed_seconds=time.perf_counter()-before, stdout=label+".stdout.log", stderr=label+".stderr.log")
            if process.returncode == 0:
                data = read(folder / "normal.json")
                if data["plan_sha256"] != digest(run / "plan.json") or data["gt_read"] or len(data["rows"]) != 10:
                    raise ValueError("Complete fresh worker result required")
                capture.update(normal_sha256=digest(folder / "normal.json"), pid=data["pid"],
                               integration_sha256=digest(folder / "integration.json"))
                rows.extend(dict(row, repeat=repeat) for row in data["rows"])
            else:
                reason = read(folder / "failure.json") if (folder / "failure.json").exists() else dict(reason="worker_exit_"+str(process.returncode))
                capture["failure"] = reason
                rows.extend(dict(case_id=case_id, repeat=repeat, variant=arm, fraction=fraction,
                    graph=dict(state="error", reason=reason, segments=[]), error=reason,
                    native_state="error", native_segments=[], native_source_state="worker_failed")
                    for arm in ARMS for fraction in SCALES)
            workers.append(capture)
            write(run / (label + "-capture.json"), capture)
            print("E2E_WORKER", label, process.returncode, round(capture["elapsed_seconds"], 3), flush=True)
    verify_hashes(plan["direct_hashes"])
    complete = all(w["exit_code"] == 0 for w in workers)
    write(run / "normal.json", dict(run_id=run.name, rows=rows, workers=workers, gt_read=False,
        plan_sha256=digest(run / "plan.json"), end_to_end_repeat_performed=complete,
        fresh_da3_calls=4 if complete else None, elapsed_seconds=time.perf_counter()-started, **FLAGS))
    write(run / "manifest.json", dict(state="normal_complete", normal_sha256=digest(run / "normal.json"),
        expected_rows=40, all_workers_complete=complete, plan_sha256=digest(run / "plan.json")))
    print("E2E_NORMAL_COMPLETE", len(rows), "workers_complete", complete, flush=True)


def evaluate(run):
    import run_g1_object_stroke_chain_readout as scoring

    manifest, data = read(run / "manifest.json"), read(run / "normal.json")
    expected = {(r, c, a, s) for r in REPEATS for c in CASES for a in ARMS for s in SCALES}
    if (digest(run / "normal.json") != manifest["normal_sha256"] or data["gt_read"]
            or len(data["rows"]) != 40 or {(r["repeat"], r["case_id"], r["variant"], r["fraction"]) for r in data["rows"]} != expected):
        raise ValueError("Complete unchanged 40-row normal inventory required")
    truth_path = ROOT / "data/eval_gt/g1-object-scenes-v1-20261006/manifest.json"
    expected_truth = read(rgb.SCENE / "prepared.json")["truth_sha256"]
    if digest(truth_path) != expected_truth:
        raise ValueError("Original complete truth changed")
    truths = {c["case_id"]: c["declared"] for c in read(truth_path)["cases"]}
    for row in data["rows"]:
        declared = truths[row["case_id"]]
        target = declared["target"]
        truth = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        row["physical"], row["scoring_error"] = scoring.score_geometry(row["graph"]["state"], row["graph"]["segments"], truth, gaps)
        row["native_physical"], row["native_scoring_error"] = scoring.score_geometry(row["native_state"], row["native_segments"], truth, gaps)
    pairs = []
    for case_id in CASES:
        first, second = [read(run / f"repeat-{r}" / case_id / "integration.json") for r in REPEATS] if manifest["all_workers_complete"] else ({}, {})
        pairs.append(dict(case_id=case_id, available=bool(first and second),
            equality={key: first.get(key) == second.get(key) if first and second else None for key in
                      ("points_sha256", "point_ids_sha256", "prediction_array_sha256", "camera_sha256", "source_frame_sha256")}))
    lookup = {(r["repeat"], r["case_id"], r["variant"], r["fraction"]): r for r in data["rows"]}
    row_pairs = []
    for case_id in CASES:
        for arm in ARMS:
            for scale in SCALES:
                a, b = [lookup[(repeat, case_id, arm, scale)] for repeat in REPEATS]
                row_pairs.append(dict(case_id=case_id, variant=arm, fraction=scale,
                    available=a["graph"]["state"] == b["graph"]["state"] == "complete",
                    segments_equal=a["graph"]["segments"] == b["graph"]["segments"],
                    physical_equal=a["physical"] == b["physical"], native_equal=a["native_segments"] == b["native_segments"]))
    write(run / "evaluation.json", dict(**data, normal_sha256=manifest["normal_sha256"], evaluation_gt_read=True,
        truth_sha256=expected_truth, repeat_pairs=pairs, row_repeat_pairs=row_pairs))
    print("E2E_EVALUATED", len(data["rows"]), "pairs", len(row_pairs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("normal", "worker", "evaluate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat", type=int, choices=REPEATS)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.stage == "worker" and (args.repeat is None or args.case is None):
        parser.error("worker requires --repeat and --case")
    run = run_path(args.run_id)
    if args.stage == "normal":
        normal(run)
    elif args.stage == "worker":
        worker(run, args.repeat, args.case)
    else:
        evaluate(run)


if __name__ == "__main__":
    main()
