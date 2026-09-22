"""Freeze RGB descriptor tracks before a separate mesh-truth audit."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from run_foreground_estimated_pilot import (
    ROOT,
    checked,
    checked_case,
    digest,
    locations,
    read_json,
    write_json,
)
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.background_correspondences import (
    detect_features,
    exclusion_mask,
    fit_rgb_geometry,
    mutual_ratio_matches,
)
from creator_eval.correspondence_tracks import build_tracks, pair_track_labels, split_tracks

CONFIG = ROOT / "configs/foreground_tracks_v1.json"
SOURCES = ["scripts/run_foreground_tracks.py", "experiments/src/creator_eval/correspondence_tracks.py",
           "experiments/src/creator_eval/background_correspondences.py", "experiments/src/creator_eval/camera_diagnostics.py",
           "scripts/run_background_correspondences.py", "scripts/thin_pack_gt.py"]


def prepare(path):
    config = read_json(path)
    parent, parent_inputs, previous, method = checked(config["source_run_id"])
    run, inputs = locations(config["run_id"])
    if run.exists() or inputs.exists():
        raise FileExistsError("Keep previous track experiments")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    cases = []
    for entry in method["cases"]:
        old = checked_case(parent_inputs, entry)
        # No camera or pool is needed for descriptor matching. Leave them out
        # entirely so a later refactor cannot quietly start depending on them.
        cases.append(dict(case_id=entry["case_id"], frames=[{key: f[key] for key in
                     ("view_id", "rgb", "rgb_sha256", "size_wh", "guide_xyxy")} for f in old["frames"]]))
    allowed = {key: config[key] for key in ("detectors", "ratio", "target_exclusion_half_width_px", "split", "geometry", "availability")}
    write_json(inputs / "manifest.json", dict(cases=cases, method=allowed))
    write_json(run / "protocol.json", config)
    sources = {}
    for name in sorted(set(previous["source_sha256"]) | set(SOURCES)):
        dest = run / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        sources[name] = digest(dest)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources,
        input_sha256=digest(inputs / "manifest.json"), protocol_sha256=digest(run / "protocol.json"),
        parent_prepared_sha256=digest(parent / "prepared.json")))
    print("PREPARED_RGB_TRACKS", len(cases), flush=True)


def checked_tracks(run_id):
    run, inputs = locations(run_id)
    frozen = read_json(run / "prepared.json")
    if frozen["run_id"] != run_id or digest(inputs / "manifest.json") != frozen["input_sha256"]:
        raise ValueError("Track input identity changed")
    for name, sha in frozen["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(run / "source_snapshot" / name) != sha:
            raise ValueError("Frozen track source changed: " + name)
    return run, frozen, read_json(inputs / "manifest.json")


def reject_model_open(event, arguments):
    reject_truth_open(event, arguments)
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0])
        if path.name in ("prediction.npz", "geometry.json"):
            raise PermissionError("RGB track inference must not read model geometry")


def infer(run_id):
    import cv2

    run, frozen, inputs = checked_tracks(run_id)
    sys.addaudithook(reject_model_open)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(113)
    method, records, started = inputs["method"], [], time.perf_counter()
    for case in inputs["cases"]:
        images, masks = [], []
        for frame in case["frames"]:
            if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
                raise ValueError("RGB changed")
            rgb = cv2.cvtColor(cv2.imread(str(ROOT / frame["rgb"])), cv2.COLOR_BGR2RGB)
            if list(rgb.shape[1::-1]) != frame["size_wh"]:
                raise ValueError("RGB size changed")
            (x0, y0), (x1, y1) = frame["guide_xyxy"]
            masks.append(exclusion_mask(frame["size_wh"], {"selected": [x0, x1]}, [y0, y1], method["target_exclusion_half_width_px"]))
            images.append(rgb)
        for detector, settings in method["detectors"].items():
            features = [detect_features(rgb, mask, detector, settings) for rgb, mask in zip(images, masks)]
            pairs = []
            for a, b in combinations(range(len(features)), 2):
                ids, distances = mutual_ratio_matches(features[a], features[b], detector, method["ratio"])
                pairs.append(dict(first_view=a, second_view=b, keypoint_ids=ids,
                    first_xy=features[a]["xy"][ids[:, 0]], second_xy=features[b]["xy"][ids[:, 1]], descriptor_distances=distances))
            graph = build_tracks(pairs)
            validation, split = split_tracks(graph["tracks"], features, method["split"])
            labels = pair_track_labels(pairs, graph, validation)
            nonplanar_pairs = 0
            for pair, label in zip(pairs, labels):
                chosen = label["track_ids"] >= 0
                geometry = fit_rgb_geometry(pair["first_xy"][chosen], pair["second_xy"][chosen], label["validation"][chosen], method["geometry"])
                if geometry.get("models", {}).get("F", {}).get("state") == "validation_consistent" and not geometry["planar_or_repetitive_explanation_possible"]:
                    nonplanar_pairs += 1
                pair.update(**label, geometry=geometry)
            coverage = []
            for view in range(len(features)):
                present = np.array([any(v == view for v, _ in track) for track in graph["tracks"]], bool)
                coverage.append(dict(view=view, train=int((present & ~validation).sum()), validation=int((present & validation).sum())))
            policy = method["availability"]
            sufficient = all(row["train"] >= policy["minimum_train_tracks_per_view"] and row["validation"] >= policy["minimum_validation_tracks_per_view"] for row in coverage)
            sufficient &= nonplanar_pairs >= policy["minimum_nonplanar_validated_pairs"]
            path = run / (case["case_id"] + "-" + detector + ".json")
            write_json(path, dict(case_id=case["case_id"], detector=detector, feature_counts=[len(f["xy"]) for f in features],
                feature_xy=[f["xy"] for f in features], graph=graph, validation_tracks=validation, split=split, pairs=pairs, coverage=coverage,
                availability="eligible_for_further_validation" if sufficient else "insufficient_for_camera_optimization",
                nonplanar_validated_pairs=nonplanar_pairs, camera_changes_applied=False))
            records.append(dict(path=path.name, sha256=digest(path), case_id=case["case_id"], detector=detector))
            print("RGB_TRACKS", case["case_id"], detector, graph["pair_edge_count"], "edges;", len(graph["tracks"]), "complete tracks;", sufficient, flush=True)
    checked_tracks(run_id)
    write_json(run / "inference.json", dict(state="complete", input_sha256=frozen["input_sha256"], source_sha256=frozen["source_sha256"],
        records=records, gt_read_during_inference=False, model_geometry_read_during_inference=False,
        opencv_version=cv2.__version__, numpy_version=np.__version__, elapsed_seconds=time.perf_counter() - started))


def evaluate(run_id, output):
    from creator_eval.background_correspondences import mesh_direction
    from run_background_correspondences import _mesh_report
    from thin_pack_gt import MeshRays

    run, frozen, inputs = checked_tracks(run_id)
    if digest(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Protocol changed")
    config = read_json(run / "protocol.json")
    parent, _, previous, _ = checked(config["source_run_id"])
    if digest(parent / "prepared.json") != frozen["parent_prepared_sha256"]:
        raise ValueError("Parent changed")
    pack, pack_inputs = locations(config["evaluation"]["gt_bundle"])
    if digest(pack / "prepared.json") != previous["parent_prepared_sha256"]:
        raise ValueError("Truth source changed")
    pack_meta = read_json(pack / "prepared.json")
    if digest(pack_inputs / "manifest.json") != pack_meta["input_manifest_sha256"]:
        raise ValueError("GT image index changed")
    pack_frames = {c["case_id"]: c["frames"] for c in read_json(pack_inputs / "manifest.json")["cases"]}
    truth_root = ROOT / "data/eval_gt" / config["evaluation"]["gt_bundle"]
    if digest(truth_root / "artifact_hashes.json") != pack_meta["truth_artifacts_sha256"]:
        raise ValueError("GT index changed")
    for name, sha in read_json(truth_root / "artifact_hashes.json").items():
        if digest(truth_root / name) != sha:
            raise ValueError("GT changed")
    targets = {c["case_id"]: read_json(truth_root / c["path"]) for c in read_json(truth_root / "manifest.json")["cases"]}
    inference = read_json(run / "inference.json")
    if inference["state"] != "complete" or inference["input_sha256"] != frozen["input_sha256"] or inference["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("Incomplete/detached RGB tracks")
    expected = {(c["case_id"], d) for c in inputs["cases"] for d in inputs["method"]["detectors"]}
    actual = [(r["case_id"], r["detector"]) for r in inference["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete/duplicated RGB plan")
    destination = ROOT / "data/evaluation" / run_id
    destination.mkdir(exist_ok=False)
    summaries, rows, artifacts = [], [], []
    for entry in inference["records"]:
        if digest(run / entry["path"]) != entry["sha256"]:
            raise ValueError("RGB tracks changed")
        record, target = read_json(run / entry["path"]), targets[entry["case_id"]]
        case = next(c for c in inputs["cases"] if c["case_id"] == entry["case_id"])
        if [(f["view_id"], f["rgb_sha256"]) for f in case["frames"]] != [(f["view_id"], f["rgb_sha256"]) for f in pack_frames[entry["case_id"]]]:
            raise ValueError("GT belongs to different RGB")
        if [f["view_id"] for f in case["frames"]] != [f["view_id"] for f in target["frames"]]:
            raise ValueError("GT frame order changed")
        cameras = [read_json(truth_root / entry["case_id"] / f["camera_path"]) for f in target["frames"]]
        mesh = MeshRays(truth_root / target["mesh_path"])
        track_edge_results = [[] for _ in record["graph"]["tracks"]]
        for pair in record["pairs"]:
            a, b = pair["first_view"], pair["second_view"]
            xy_a, xy_b = np.asarray(pair["first_xy"], float).reshape(-1, 2), np.asarray(pair["second_xy"], float).reshape(-1, 2)
            if len(xy_a):
                forward = mesh_direction(xy_a, xy_b, cameras[a], cameras[b], mesh.cast, config["evaluation"]["visibility_depth_tolerance_m"])
                reverse = mesh_direction(xy_b, xy_a, cameras[b], cameras[a], mesh.cast, config["evaluation"]["visibility_depth_tolerance_m"])
            else:
                forward = reverse = {key: np.zeros(0, bool) for key in ("source_hit", "source_point_visible_in_target", "target_in_frame", "occluded", "target_no_hit", "depth_inconsistent", "invalid_projection")}
                forward["transfer_error_px"] = np.empty(0)
            both = forward["source_point_visible_in_target"] & reverse["source_point_visible_in_target"]
            any_visible = forward["source_point_visible_in_target"] | reverse["source_point_visible_in_target"]
            worst = np.fmax(forward["transfer_error_px"], reverse["transfer_error_px"])
            correct = both & (worst <= 2)
            wrong = any_visible & (worst > 5)
            ids = np.asarray(pair["track_ids"], int)
            for i in np.flatnonzero(ids >= 0):
                track_edge_results[ids[i]].append("correct" if correct[i] else ("wrong" if wrong[i] else "indeterminate"))
            selected, validation = ids >= 0, np.asarray(pair["validation"], bool)
            subsets = dict(mutual=np.ones(len(ids), bool), triangle=np.asarray(pair["triangle"], bool), complete_track=selected,
                           track_train=selected & ~validation, track_validation=selected & validation)
            detail = destination / f"{entry['case_id']}-{entry['detector']}-{a}-{b}.json"
            # MeshRays marks unknown projections as NaN. Keep them as null in
            # the evidence file; do not turn missing observations into a pass.
            from run_background_correspondences import clean
            write_json(detail, clean(dict(forward=forward, reverse=reverse, track_ids=ids)))
            artifacts.append(dict(path=detail.name, sha256=digest(detail)))
            rows.append(dict(case_id=entry["case_id"], detector=entry["detector"], first_view=a, second_view=b,
                rgb_geometry=pair["geometry"], subsets={name: _mesh_report(forward, reverse, mask) for name, mask in subsets.items()}))
        track_labels = ["wrong" if "wrong" in checks else ("correct" if checks and all(v == "correct" for v in checks) else "indeterminate") for checks in track_edge_results]
        summaries.append({**{key: record[key] for key in ("case_id", "detector", "feature_counts", "coverage", "availability", "nonplanar_validated_pairs", "split")},
            "pair_edge_count": record["graph"]["pair_edge_count"], "triangle_edge_count": record["graph"]["triangle_edge_count"],
            "complete_track_count": len(track_labels), "track_truth_counts": dict(Counter(track_labels)), "track_truth_labels": track_labels,
            "rejected_components": dict(Counter(c["reason"] for c in record["graph"]["rejected_components"]))})
    checked_tracks(run_id)
    report = dict(state="complete", run_id=run_id, scope=config["scope"], summaries=summaries, pairs=rows, artifacts=artifacts,
        source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"], protocol_sha256=frozen["protocol_sha256"],
        inference_sha256=digest(run / "inference.json"), truth_artifacts_sha256=pack_meta["truth_artifacts_sha256"],
        inference_seconds=inference["elapsed_seconds"], camera_changes_applied=False,
        limits="First-hit geometry verifies point location, not descriptor support/antialiasing. Unknown stays unknown. No GT result selects tracks or updates cameras.")
    write_json(destination / "summary.json", report)
    write_json(output, report)
    print("EVALUATED_RGB_TRACKS", len(rows), "pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default="foreground-tracks-v1-20260922")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-foreground-tracks.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
