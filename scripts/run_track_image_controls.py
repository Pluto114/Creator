"""Run the unchanged track extractor on analytic RGB controls; score separately."""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from run_foreground_tracks import (
    ROOT,
    checked_tracks,
    digest,
    infer,
    locations,
    read_json,
    write_json,
)

# isort: split
from creator_eval.track_image_controls import control_camera, physical_pair, render_planes

CONFIG = ROOT / "configs/track_image_controls_v1.json"


def prepare(path):
    config = read_json(path)
    _, parent, previous = checked_tracks(config["track_source_run_id"])
    run, inputs = locations(config["run_id"])
    truth = ROOT / "data/eval_gt" / config["run_id"]
    for folder in (run, inputs, truth):
        if folder.exists():
            raise FileExistsError("Keep prior image controls")
    for folder in (run, inputs, truth):
        folder.mkdir(parents=True)
    sources = {}
    for name in sorted(set(parent["source_sha256"]) | {"scripts/run_track_image_controls.py", "experiments/src/creator_eval/track_image_controls.py"}):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = digest(target)
    # Freeze before rendering. Generation metadata belongs only to evaluation.
    write_json(run / "protocol.json", config)
    write_json(run / "generation-freeze.json", dict(source_sha256=sources, protocol_sha256=digest(run / "protocol.json")))
    cases, truths = [], []
    for case in config["cases"]:
        frames, cameras = [], []
        for angle in config["angles_degrees"]:
            camera = control_camera(angle, config["size_wh"])
            rgb = render_planes(camera, case["planes"], case["repeat"], config["texture_seed"])
            view_id = f"view_{angle:+03d}"
            dest = inputs / case["case_id"] / (view_id + ".png")
            dest.parent.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(dest)
            frames.append(dict(view_id=view_id, rgb=dest.relative_to(ROOT).as_posix(), rgb_sha256=digest(dest), size_wh=config["size_wh"], guide_xyxy=[[319.5, 20], [319.5, 459]]))
            cameras.append(camera)
        cases.append(dict(case_id=case["case_id"], frames=frames))
        truths.append(dict(case_id=case["case_id"], planes=case["planes"], cameras=cameras))
    write_json(inputs / "manifest.json", dict(cases=cases, method=previous["method"]))
    write_json(truth / "manifest.json", dict(cases=truths, input_sha256=digest(inputs / "manifest.json")))
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources, protocol_sha256=digest(run / "protocol.json"),
        input_sha256=digest(inputs / "manifest.json"), truth_sha256=digest(truth / "manifest.json")))
    print("PREPARED_TRACK_IMAGE_CONTROLS", len(cases) * len(config["angles_degrees"]), "images", flush=True)


def evaluate(run_id, output):
    run, frozen, inputs = checked_tracks(run_id)
    truth_path = ROOT / "data/eval_gt" / run_id / "manifest.json"
    if digest(truth_path) != frozen["truth_sha256"] or digest(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Control truth/protocol changed")
    truth = read_json(truth_path)
    if truth["input_sha256"] != frozen["input_sha256"]:
        raise ValueError("Control truth detached from RGB")
    index = read_json(run / "inference.json")
    if index["state"] != "complete" or index["source_sha256"] != frozen["source_sha256"] or index["input_sha256"] != frozen["input_sha256"]:
        raise ValueError("Incomplete/detached control inference")
    expected = {(c["case_id"], d) for c in inputs["cases"] for d in inputs["method"]["detectors"]}
    actual = [(r["case_id"], r["detector"]) for r in index["records"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Control plan incomplete")
    rows = []
    for entry in index["records"]:
        if digest(run / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen RGB control result changed")
        record = read_json(run / entry["path"])
        gt = next(c for c in truth["cases"] if c["case_id"] == entry["case_id"])
        track_edges = [[] for _ in record["graph"]["tracks"]]
        pairs = []
        for pair in record["pairs"]:
            a, b = pair["first_view"], pair["second_view"]
            labels = physical_pair(np.asarray(pair["first_xy"]).reshape(-1, 2), np.asarray(pair["second_xy"]).reshape(-1, 2), gt["cameras"][a], gt["cameras"][b], gt["planes"])
            selected = np.asarray(pair["track_ids"]) >= 0
            for i in np.flatnonzero(selected):
                track_edges[pair["track_ids"][i]].append(labels[i])
            pairs.append(dict(first_view=a, second_view=b, mutual_counts=dict(Counter(labels)),
                complete_track_edge_counts=dict(Counter(labels[selected])), rgb_geometry=pair["geometry"]))
        track_labels = ["wrong" if "wrong" in checks else ("correct" if checks and all(v == "correct" for v in checks) else "indeterminate") for checks in track_edges]
        rows.append({**{key: record[key] for key in ("case_id", "detector", "feature_counts", "coverage", "availability", "nonplanar_validated_pairs", "split")},
            "complete_tracks": len(track_labels), "track_truth_counts": dict(Counter(track_labels)), "pairs": pairs})
    destination = ROOT / "data/evaluation" / run_id
    destination.mkdir(exist_ok=False)
    result = dict(state="complete", run_id=run_id, rows=rows, source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"],
        protocol_sha256=frozen["protocol_sha256"], truth_sha256=frozen["truth_sha256"], inference_sha256=digest(run / "inference.json"),
        inference_seconds=index["elapsed_seconds"], scope=read_json(run / "protocol.json")["scope"])
    write_json(destination / "summary.json", result)
    write_json(output, result)
    print("EVALUATED_TRACK_IMAGE_CONTROLS", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default="track-image-controls-v1-20260922")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-track-image-controls.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
