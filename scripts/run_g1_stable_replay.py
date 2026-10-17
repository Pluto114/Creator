"""Split-stage same-version replay of four saved independent DA3 workers.

No new DA3, RGB inference, calibration, controls fit, or baseline reconstruction.
Fresh subprocesses apply boundary-patch-1 and the current stable common reader
to each worker's own saved sources. All 48 normal rows precede GT evaluation.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import run_boundary_patch_sprint as boundary
import run_g1_end_to_end as upstream
import run_g1_object_point_patch as point
import run_readout_sprint as sprint
from creator_eval import rod_boundary_patch as method
from creator_eval.rgb_bracketed_occupancy_readout import BracketedOccupancyReadout
from creator_eval.rgb_stroke_chain_support import build_context

ROOT = upstream.ROOT
UPSTREAM = ROOT / ".runtime/experiments/g1-end-to-end-20261013-001"
STABLE = ROOT / ".runtime/experiments/boundary-patch-sprint-20261014-001"
ARMS = (*sprint.ARMS, "baseline_boundary")
CASES, REPEATS, SCALES = upstream.CASES, upstream.REPEATS, upstream.SCALES
read, write, digest = upstream.read, upstream.write, upstream.digest
FLAGS = dict(split_stage_replay=True, fresh_da3_calls=0, cached_independent_da3_workers=4,
             fresh_rgb_calibration_calls=0, fresh_native_baseline_reconstructions=0,
             fresh_last_mile_workers=4, fresh_common_readouts=48, fresh_stable_patches=4,
             end_to_end_repeat_performed=False, blind_evaluation=False,
             independent_real_capture=False, g1_passed=False)


def expected_keys():
    return {(r, c, a, s) for r in REPEATS for c in CASES for a in ARMS for s in SCALES}


def validate(data):
    keys = [(r["repeat"], r["case_id"], r["variant"], r["fraction"]) for r in data["rows"]]
    if len(keys) != 48 or set(keys) != expected_keys() or data["gt_read"]:
        raise ValueError("Complete unmodified 48-row normal inventory required")


def prepare_plan(run):
    old_manifest, old = read(UPSTREAM / "manifest.json"), read(UPSTREAM / "normal.json")
    stable_manifest, stable = read(STABLE / "manifest.json"), read(STABLE / "normal.json")
    if (not old_manifest["all_workers_complete"] or old["fresh_da3_calls"] != 4
            or digest(UPSTREAM / "normal.json") != old_manifest["normal_sha256"]
            or digest(UPSTREAM / "plan.json") != old_manifest["plan_sha256"]):
        raise ValueError("Original complete four-worker independent upstream required")
    if (stable_manifest["state"] != "normal_complete" or len(stable["rows"]) != 60
            or stable_manifest["normal_sha256"] != digest(STABLE / "normal.json")
            or stable["adapter"] != boundary.ADAPTER or stable["method_policy"] != method.POLICY):
        raise ValueError("Same current stable boundary policy and reader required")
    old_plan = read(UPSTREAM / "plan.json")
    if digest(upstream.ANCHORS) != old_plan["direct_hashes"][upstream.ANCHORS.relative_to(ROOT).as_posix()]:
        raise ValueError("Original shared annotations changed")
    workers = []
    paths = {UPSTREAM / "normal.json", UPSTREAM / "manifest.json", UPSTREAM / "plan.json",
             STABLE / "normal.json", STABLE / "manifest.json", upstream.ANCHORS}
    for repeat in REPEATS:
        for case_id in CASES:
            folder = UPSTREAM / f"repeat-{repeat}" / case_id
            capture = next(w for w in old["workers"] if w["repeat"] == repeat and w["case_id"] == case_id)
            native = read(folder / "normal.json")
            if (capture["exit_code"] != 0 or digest(folder / "normal.json") != capture["normal_sha256"]
                    or digest(folder / "integration.json") != capture["integration_sha256"]
                    or native["pid"] != capture["pid"] or native["gt_read"]
                    or native["plan_sha256"] != old_manifest["plan_sha256"]):
                raise ValueError("Upstream worker source binding changed")
            paths.update(folder / name for name in ("rgb.json", "rods.json", "integration.json", "normal.json",
                         "depth/report.json", "depth/prediction.npz"))
            paths.update(path for path in (folder / "bundle").rglob("*") if path.is_file())
            paths.update(folder / (arm + "-fit.json") for arm in point.controls.METHODS)
            workers.append(dict(repeat=repeat, case_id=case_id, upstream_pid=capture["pid"],
                upstream_folder=folder.relative_to(ROOT).as_posix(),
                upstream_normal_sha256=capture["normal_sha256"],
                upstream_integration_sha256=capture["integration_sha256"]))
    if len({w["upstream_pid"] for w in workers}) != 4:
        raise ValueError("Four distinct original DA3 worker processes required")
    sources = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and not any(p.startswith(".venv") for p in path.parts):
                sources.add(path)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths | sources)}
    changed = [name for name, sha in hashes.items() if name in stable_manifest["direct_hashes"]
               and sha != stable_manifest["direct_hashes"][name]]
    if changed:
        raise ValueError("Current stable shared direct sources changed: " + repr(changed))
    return dict(run_id=run.name, workers=workers, direct_hashes=hashes,
        anchors=old_plan["anchors"], arms=list(ARMS), scales=list(SCALES), method_policy=method.POLICY,
        adapter=boundary.ADAPTER, stable_normal_sha256=digest(STABLE / "normal.json"),
        upstream_normal_sha256=digest(UPSTREAM / "normal.json"), gt_read=False, **FLAGS)


def worker(run, repeat, case_id):
    sys.addaudithook(upstream.rgb.block_truth)
    started = time.perf_counter()
    plan = read(run / "plan.json")
    item = next(w for w in plan["workers"] if w["repeat"] == repeat and w["case_id"] == case_id)
    folder = run / f"repeat-{repeat}" / case_id
    folder.mkdir(parents=True, exist_ok=False)
    source = ROOT / item["upstream_folder"]
    try:
        upstream.verify_hashes(plan["direct_hashes"])
        integration, raw, rods = [read(source / name) for name in ("integration.json", "rgb.json", "rods.json")]
        report = read(source / "depth/report.json")
        cameras = rods["cameras"]
        base_path = source / "bundle/base"
        base = point.compose(base_path)
        hashes = dict(points_sha256=upstream.array_hash(base["points"]),
            point_ids_sha256=upstream.array_hash(base["point_ids"]),
            camera_sha256=point.canonical_hash(cameras), source_frame_sha256=point.canonical_hash(raw["frames"]))
        if any(integration[key] != sha for key, sha in hashes.items()):
            raise ValueError("Worker-specific actual base, cameras or evidence changed")
        prediction_sha = digest(source / "depth/prediction.npz")
        base_metadata = read(base_path / "manifest.json")["metadata"]
        if not (prediction_sha == report["prediction_sha256"] == rods["prediction_sha256"]
                == base_metadata["source_prediction_sha256"]):
            raise ValueError("Actual saved base not bound to this independent DA3 prediction")
        native, states = {"base": np.empty((0, 2, 3))}, {"base": "base"}
        for patch in integration["patches"]:
            name = patch["method"]
            candidate_path = ROOT / patch["enabled_path"]
            if not candidate_path.is_relative_to(source):
                raise ValueError("Do not substitute another upstream worker's candidate")
            candidate = point.open_candidate_view(candidate_path)
            native[name] = candidate["segments"]
            point.validate_candidate(base, candidate, np.asarray(patch["native_segments"]).reshape(-1, 2, 3), enabled=True)
            point.validate_candidate(base, point.open_candidate_view(ROOT / patch["withdrawn_path"]), native[name], enabled=False)
            bound_source = source / ("rods.json" if name in ("baseline", "cylinder_support") else name + "-fit.json")
            if digest(bound_source) != patch["source_sha256"]:
                raise ValueError("Original per-worker native fit binding changed")
            states[name] = patch["source_state"]
        if set(native) != set(sprint.ARMS):
            raise ValueError("All original five arms required from this worker")
        context = build_context(raw["frames"], cameras, plan["anchors"][case_id])
        refinement = method.refine_boundary(native["baseline"], context)
        record = folder / "refinement.json"
        write(record, dict(original_native_segments=native["baseline"], refinement=refinement,
            context_sha256=context["sha256"], policy=method.POLICY, upstream_binding=item, gt_read=False))
        bundle = folder / "bundle"
        shutil.copytree(base_path, bundle / "base")
        stable_patch = boundary.persist(bundle, base, refinement, context, record, method.POLICY)
        native["baseline_boundary"] = point.open_candidate_view(ROOT / stable_patch["enabled_path"])["segments"]
        states["baseline_boundary"] = refinement["state"]
        source_base_bytes = {p.name: digest(p) for p in base_path.iterdir() if p.is_file()}
        copied_base_bytes = {p.name: digest(p) for p in (bundle / "base").iterdir() if p.is_file()}
        if source_base_bytes != copied_base_bytes:
            raise ValueError("Persisting stable patch changed copied base bytes")
        adapter = BracketedOccupancyReadout(base["points"], context)
        span = integration["camera_span_m"]
        rows = []
        for arm in ARMS:
            for fraction in SCALES:
                before = time.perf_counter()
                graph = sprint.compact(adapter(native[arm], dict(voxel_size=span*fraction, origin=[0., 0., 0.])))
                rows.append(dict(kind="object", case_id=case_id, repeat=repeat, variant=arm, fraction=fraction,
                    graph=graph, native_segments=native[arm], native_state="complete", native_source_state=states[arm],
                    error=None, reader_elapsed_seconds=time.perf_counter()-before))
        write(folder / "integration.json", dict(**hashes, upstream_binding=item,
            prediction_sha256=prediction_sha, context_sha256=context["sha256"], patch=stable_patch,
            base_file_sha256=source_base_bytes, copied_base_files_identical=True,
            base_bytes_preserved=True, withdrawn_bytes_equal=True, saved_views_reopened=True, gt_read=False))
        write(folder / "normal.json", dict(rows=rows, pid=os.getpid(), upstream_binding=item,
            plan_sha256=digest(run / "plan.json"), gt_read=False, elapsed_seconds=time.perf_counter()-started))
        print("STABLE_REPLAY_COMPLETE", repeat, case_id, len(rows), flush=True)
    except Exception as exc:
        write(folder / "failure.json", dict(reason=repr(exc), pid=os.getpid(), upstream_binding=item, gt_read=False))
        raise


def normal(run):
    sys.addaudithook(upstream.rgb.block_truth)
    started = time.perf_counter()
    run.mkdir(parents=True, exist_ok=False)
    plan = prepare_plan(run)
    write(run / "plan.json", plan)

    def dispatch(item):
        repeat, case_id = item["repeat"], item["case_id"]
        label = f"r{repeat}-{case_id}"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--run-id", run.name,
                   "--repeat", str(repeat), "--case", case_id]
        before = time.perf_counter()
        with (run / (label + ".stdout.log")).open("xb") as stdout, (run / (label + ".stderr.log")).open("xb") as stderr:
            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        capture = dict(**item, command=command, exit_code=process.returncode,
            elapsed_seconds=time.perf_counter()-before, stdout=label+".stdout.log", stderr=label+".stderr.log")
        write(run / (label + "-capture.json"), capture)
        print("STABLE_REPLAY_WORKER", label, process.returncode, flush=True)
        return capture

    with ThreadPoolExecutor(max_workers=4) as pool:
        workers = list(pool.map(dispatch, plan["workers"]))
    if any(w["exit_code"] != 0 for w in workers):
        raise RuntimeError("Replay worker failed; preserve logs and do not evaluate partial rows")
    rows = []
    for worker_info in workers:
        folder = run / f"repeat-{worker_info['repeat']}" / worker_info["case_id"]
        data = read(folder / "normal.json")
        binding = next(w for w in plan["workers"] if w["repeat"] == worker_info["repeat"] and w["case_id"] == worker_info["case_id"])
        if data["gt_read"] or len(data["rows"]) != 12 or data["upstream_binding"] != binding or data["plan_sha256"] != digest(run / "plan.json"):
            raise ValueError("Fresh subprocess output/source binding mismatch")
        worker_info.update(pid=data["pid"], normal_sha256=digest(folder / "normal.json"),
                           integration_sha256=digest(folder / "integration.json"))
        rows.extend(data["rows"])
    if len({w["pid"] for w in workers}) != 4:
        raise ValueError("Four distinct fresh replay worker processes required")
    upstream.verify_hashes(plan["direct_hashes"])
    data = dict(run_id=run.name, rows=rows, workers=workers, gt_read=False, adapter=boundary.ADAPTER,
        method_policy=method.POLICY, elapsed_seconds=time.perf_counter()-started, **FLAGS)
    validate(data)
    write(run / "normal.json", data)
    write(run / "manifest.json", dict(state="normal_complete", expected_rows=48,
        normal_sha256=digest(run / "normal.json"), plan_sha256=digest(run / "plan.json"), all_workers_complete=True))
    print("STABLE_REPLAY_NORMAL_COMPLETE", len(rows), flush=True)


def evaluate(run):
    import run_g1_object_stroke_chain_readout as scoring

    started = time.perf_counter()
    if (run / "evaluation.json").exists():
        raise FileExistsError("Previous scores are immutable")
    manifest, data, plan = [read(run / name) for name in ("manifest.json", "normal.json", "plan.json")]
    if (manifest["state"] != "normal_complete" or digest(run / "normal.json") != manifest["normal_sha256"]
            or digest(run / "plan.json") != manifest["plan_sha256"]):
        raise ValueError("Complete unchanged normal must precede GT")
    validate(data)
    truth_path = ROOT / "data/eval_gt/g1-object-scenes-v1-20261006/manifest.json"
    expected_truth = read(upstream.rgb.SCENE / "prepared.json")["truth_sha256"]
    if digest(truth_path) != expected_truth:
        raise ValueError("Original complete object GT changed")
    truth_cases = read(truth_path)["cases"]
    if set(c["case_id"] for c in truth_cases) != set(CASES):
        raise ValueError("Both complete original object truths required")
    truths = {c["case_id"]: c["declared"] for c in truth_cases}
    for row in data["rows"]:
        declared = truths[row["case_id"]]
        target = declared["target"]
        truth = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        row["physical"], row["scoring_error"] = scoring.score_geometry(row["graph"]["state"], row["graph"]["segments"], truth, gaps)
        row["native_physical"], row["native_scoring_error"] = scoring.score_geometry(row["native_state"], row["native_segments"], truth, gaps)
    lookup = {(r["repeat"], r["case_id"], r["variant"], r["fraction"]): r for r in data["rows"]}
    stable_eval = read(STABLE / "evaluation.json")
    if stable_eval["normal_sha256"] != plan["stable_normal_sha256"]:
        raise ValueError("Stable reference scores not bound to same unchanged normal")
    stable_rows = {(r["case_id"], r["variant"], r["fraction"]): r for r in stable_eval["rows"] if r["case_id"] in CASES}
    comparisons = []
    compare_keys = ("native_segments", "native_state", "physical", "native_physical", "scoring_error", "native_scoring_error")
    for case_id in CASES:
        for arm in ARMS:
            for scale in SCALES:
                first, second = [lookup[(r, case_id, arm, scale)] for r in REPEATS]
                reference = stable_rows[(case_id, arm, scale)]
                pair = {key: first[key] == second[key] for key in compare_keys}
                pair["common_segments"] = first["graph"]["segments"] == second["graph"]["segments"]
                pair["common_state"] = first["graph"]["state"] == second["graph"]["state"] == "complete"
                matches = []
                for row in (first, second):
                    equal = {key: row[key] == reference[key] for key in compare_keys}
                    equal["common_segments"] = row["graph"]["segments"] == reference["graph"]["segments"]
                    equal["common_state"] = row["graph"]["state"] == reference["graph"]["state"] == "complete"
                    matches.append(equal)
                comparisons.append(dict(case_id=case_id, variant=arm, fraction=scale,
                    repeat_equality=pair, stable_reference_equality_by_repeat=matches))
    integrations = [read(run / f"repeat-{r}" / c / "integration.json") for r in REPEATS for c in CASES]
    summary = dict(all_24_repeat_pairs_equal=all(all(c["repeat_equality"].values()) for c in comparisons),
        all_48_rows_equal_stable_reference=all(all(eq.values()) for c in comparisons for eq in c["stable_reference_equality_by_repeat"]),
        all_four_saved_reopened_withdrawn=all(i["base_bytes_preserved"] and i["withdrawn_bytes_equal"]
            and i["saved_views_reopened"] and i["copied_base_files_identical"] for i in integrations),
        complete_rows=sum(r["graph"]["state"] == "complete" for r in data["rows"]),
        scoring_errors=sum(bool(r["scoring_error"] or r["native_scoring_error"]) for r in data["rows"]))
    result = dict(**data, normal_sha256=manifest["normal_sha256"], plan_sha256=manifest["plan_sha256"],
        evaluation_gt_read=True, truth_sha256=expected_truth, comparisons=comparisons, integrations=integrations,
        summary=summary, evaluation_elapsed_seconds=time.perf_counter()-started)
    write(run / "evaluation.json", result)
    output = ROOT / "docs/experiments/results/2026-10-17-g1-stable-replay.json"
    if output.exists():
        raise FileExistsError("Do not overwrite a previous public replay report")
    write(output, result)
    print("STABLE_REPLAY_EVALUATED", summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("normal", "worker", "evaluate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat", type=int, choices=REPEATS)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    run = upstream.run_path(args.run_id)
    if args.stage == "worker":
        if args.repeat is None or args.case is None:
            parser.error("Worker requires --repeat and --case")
        worker(run, args.repeat, args.case)
    elif args.stage == "normal":
        normal(run)
    else:
        evaluate(run)


if __name__ == "__main__":
    main()
