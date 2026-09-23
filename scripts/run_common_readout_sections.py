"""Replay all 198 previously seen controls with a separately frozen section reader."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from check_common_readout_ceiling import tube
from run_rod_identity_blender import clean, digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_sections import readout as section_readout  # noqa: E402
from creator_eval.common_readout_split import readout as split_readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from run_rod_identity_stress import reject_truth_open  # noqa: E402

CONFIG = ROOT / "configs/common_readout_sections_v1.json"
SOURCES = ["scripts/run_common_readout_sections.py", "scripts/check_common_readout_ceiling.py",
           "scripts/run_rod_identity_blender.py", "scripts/run_rod_identity_stress.py",
           "experiments/src/creator_eval/common_readout.py", "experiments/src/creator_eval/common_readout_components.py",
           "experiments/src/creator_eval/common_readout_split.py", "experiments/src/creator_eval/line_controls.py",
           "experiments/src/creator_eval/common_readout_sections.py"]
READERS = dict(transverse_split=split_readout, robust_sections=section_readout)


def locations(run_id):
    if not run_id or Path(run_id).name != run_id or ":" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("Run ID must be one directory component")
    return ROOT / ".runtime/experiments" / run_id, ROOT / "data/inputs" / run_id, ROOT / "data/eval_gt" / run_id


def analytic_cases(config):
    fresh = config["fresh"]
    for split, origin, vector, radii, noises, voxels, separation, seed in (
        ("prior_development", [.013, .017, .011], [.35, .17, 1.], [0., .005, .02, .05], [0., .0025], [.02, .04], .16, 22092026),
        ("prior_fresh_now_regression", [.028, .033, -.027], [.43, -.25, 1.25], [0., .009, .024, .042], [0., .004], [.018, .036], .16, 22092027),
        ("fresh_full_surface", fresh["origin"], fresh["vector"], fresh["radii"], fresh["noise_sigmas"], fresh["voxels"], fresh["parallel_spacing"], fresh["seed"]),
    ):
        a, length = np.array(origin), vector[2]
        layouts = dict(straight=[[a, a + [0, 0, length]]], tilted=[[a, a + vector]],
                       gap=[[a, a + [0, 0, .4 * length]], [a + [0, 0, .6 * length], a + [0, 0, length]]],
                       parallel=[[a, a + [0, 0, length]], [a + [separation, 0, 0], a + [separation, 0, length]]])
        for layout, truth in layouts.items():
            truth = np.asarray(truth, float)
            for radius in radii:
                for noise in noises:
                    points = np.concatenate([tube(s, radius, noise, seed + i) for i, s in enumerate(truth)])
                    for voxel in voxels:
                        gaps = [truth[0, 1], truth[1, 0]] if layout == "gap" else None
                        yield points, truth, gaps, dict(split=split, layout=layout, radius_m=radius, noise_sigma_m=noise, voxel_m=voxel, expectation="positive")

    a = np.array(fresh["origin"])
    single = np.array([[a, a + [0, 0, 1.17]]])
    surface = tube(single[0], .05, .003, fresh["seed"] + 100)
    plane_x, plane_y = np.meshgrid(np.arange(31) * .017, np.arange(31) * .017)
    plane = np.c_[plane_x.ravel(), plane_y.ravel(), np.zeros(plane_x.size)] + a
    yield plane, np.empty((0, 2, 3)), None, dict(split="challenge", layout="plane_negative", voxel_m=.017, expectation="negative")
    yield surface[surface[:, 0] > a[0]], single, None, dict(split="challenge", layout="half_surface", voxel_m=.017, expectation="positive")
    random = np.random.default_rng(fresh["seed"] + 101)
    clutter = a + random.uniform([-.12, -.12, .3], [.12, .12, .6], size=(1500, 3))
    yield np.concatenate((surface, clutter)), single, None, dict(split="challenge", layout="local_clutter", voxel_m=.017, expectation="positive")
    branch = tube(np.array([a + [0, 0, .55], a + [.35, 0, .55]]), 0., 0., 1)
    yield np.concatenate((surface, branch)), np.empty((0, 2, 3)), None, dict(split="challenge", layout="branch_negative", voxel_m=.017, expectation="negative")
    crossing = np.array([[a + [-.35, 0, .25], a + [.35, 0, .92]]])
    yield np.concatenate((surface, tube(crossing[0], .013, .003, fresh["seed"] + 102))), np.concatenate((single, crossing)), None, dict(split="challenge", layout="crossing", voxel_m=.017, expectation="positive")
    nearby = single + [.095, 0, 0]
    yield np.concatenate((tube(single[0], .037, .003, fresh["seed"] + 103), tube(nearby[0], .037, .003, fresh["seed"] + 104))), np.concatenate((single, nearby)), None, dict(split="challenge", layout="very_close_parallel", voxel_m=.034, expectation="positive")


def prepare(path):
    config = read_json(path)
    run, inputs, truth_root = locations(config["run_id"])
    if any(p.exists() for p in (run, inputs, truth_root)):
        raise FileExistsError("Keep previous readout runs")
    for folder in (run, inputs, truth_root):
        folder.mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = digest(ROOT / name)
    inputs_list, truth_list = [], []
    for number, (points, truth, gaps, labels) in enumerate(analytic_cases(config)):
        case_id = f"control-{number:04d}"
        path_points = inputs / (case_id + ".npy")
        with path_points.open("xb") as stream:
            np.save(stream, points, allow_pickle=False)
        inputs_list.append(dict(case_id=case_id, points_file=path_points.name, points_sha256=digest(path_points), voxel_size=labels["voxel_m"]))
        truth_list.append(dict(case_id=case_id, truth=truth, gap=gaps, labels=labels))
    assert len(inputs_list) == 198
    write_json(inputs / "manifest.json", dict(run_id=config["run_id"], cases=inputs_list, readers=config["readers"], readout=config["readout"], gt_excluded=True))
    write_json(truth_root / "truth.json", truth_list)
    write_json(run / "protocol.json", config)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources, input_sha256=digest(inputs / "manifest.json"),
        gt_sha256=digest(truth_root / "truth.json"), protocol_sha256=digest(run / "protocol.json")))
    print("PREPARED_SPLIT_CONTROLS", len(inputs_list), "frozen inputs", flush=True)


def checked(run_id):
    run, inputs, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    for name, sha in prepared["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source changed: " + name)
    if digest(inputs / "manifest.json") != prepared["input_sha256"]:
        raise ValueError("Frozen input changed")
    return run, inputs, prepared


def infer(run_id):
    run, inputs, prepared = checked(run_id)
    manifest = read_json(inputs / "manifest.json")
    sys.addaudithook(reject_truth_open)
    records, started = [], time.perf_counter()
    for case in manifest["cases"]:
        path = inputs / case["points_file"]
        if digest(path) != case["points_sha256"]:
            raise ValueError("Control point array changed")
        points = np.load(path, allow_pickle=False)
        for name in manifest["readers"]:
            start = time.perf_counter()
            result = READERS[name](points, np.empty((0, 2, 3)), {**manifest["readout"], "voxel_size": case["voxel_size"]})
            relative = case["case_id"] + "-" + name + ".json"
            write_json(run / relative, result)
            records.append(dict(case_id=case["case_id"], reader=name, path=relative, sha256=digest(run / relative), elapsed_seconds=time.perf_counter() - start))
        if len(records) % 64 == 0:
            print("SPLIT_CONTROL_PROGRESS", len(records), flush=True)
    write_json(run / "inference.json", dict(state="complete", records=records, source_sha256=prepared["source_sha256"],
        input_sha256=prepared["input_sha256"], gt_read_during_inference=False, numpy_version=np.__version__,
        python_version=sys.version, elapsed_seconds=time.perf_counter() - started))
    print("INFERRED_SPLIT_CONTROLS", len(records), flush=True)


def evaluate(run_id, output):
    run, inputs, prepared = checked(run_id)
    _, _, truth_root = locations(run_id)
    if digest(truth_root / "truth.json") != prepared["gt_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Frozen evaluation changed")
    truth = {r["case_id"]: r for r in read_json(truth_root / "truth.json")}
    config, manifest, inference = read_json(run / "protocol.json"), read_json(inputs / "manifest.json"), read_json(run / "inference.json")
    expected = {(case["case_id"], name) for case in manifest["cases"] for name in manifest["readers"]}
    actual = [(r["case_id"], r["reader"]) for r in inference["records"]]
    if inference["state"] != "complete" or inference["source_sha256"] != prepared["source_sha256"] or inference["input_sha256"] != prepared["input_sha256"] or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete/detached paired inference")
    rows = []
    for record in inference["records"]:
        if digest(run / record["path"]) != record["sha256"]:
            raise ValueError("Inference result changed")
        result, target = read_json(run / record["path"]), truth[record["case_id"]]
        score = curve_metrics(result["segments"], target["truth"], tolerance=config["tolerance_m"], spacing=config["spacing_m"])
        gap = gap_coverage(result["segments"], target["gap"], tolerance=config["tolerance_m"], spacing=config["spacing_m"]) if target["gap"] else None
        limits = config["qualification"]
        qualified = result["state"] == "complete"
        if target["labels"]["expectation"] == "negative":
            length = sum(np.linalg.norm(np.subtract(s[1], s[0])) for s in result["segments"])
            qualified &= length <= limits["negative_maximum_predicted_length_m"]
        else:
            qualified &= score["recovery_fraction"] >= limits["positive_minimum_recovery"] and score["precision_fraction"] is not None and score["precision_fraction"] >= limits["positive_minimum_precision"]
        rows.append(dict(case_id=record["case_id"], reader=record["reader"], **target["labels"], state=result["state"],
                         segments=result["segments"], metrics=score, gap=gap, diagnostic_qualified=bool(qualified),
                         accepted_splits=result.get("accepted_splits"), split_decisions=result.get("split_decisions"),
                         sections=result.get("sections"), rejected_side_branches=result.get("rejected_side_branches"),
                         rejected_unsupported_sections=result.get("rejected_unsupported_sections")))
    report = dict(run_id=run_id, state="complete", source_sha256=prepared["source_sha256"], input_sha256=prepared["input_sha256"],
                  gt_sha256=prepared["gt_sha256"], protocol_sha256=prepared["protocol_sha256"], inference_sha256=digest(run / "inference.json"),
                  scope=config["scope"], qualification=config["qualification"], elapsed_seconds=inference["elapsed_seconds"], rows=rows)
    evaluation = ROOT / "data/evaluation" / run_id
    evaluation.mkdir(parents=True, exist_ok=False)
    write_json(evaluation / "summary.json", report)
    write_json(output, report)
    for split in sorted({r["split"] for r in rows}):
        for reader in manifest["readers"]:
            subset = [r for r in rows if r["split"] == split and r["reader"] == reader]
            print(split, reader, sum(r["diagnostic_qualified"] for r in subset), "/", len(subset), flush=True)
    return clean(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default="common-readout-sections-controls-v1-20260923r2")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-23-readout-sections-controls-r2.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
