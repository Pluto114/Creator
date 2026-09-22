"""Second frozen track split: fixed spatial bands with a guard row, no refit to GT."""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import run_foreground_tracks as prior
from run_foreground_tracks import (
    ROOT,
    build_tracks,
    checked_tracks,
    detect_features,
    digest,
    exclusion_mask,
    fit_rgb_geometry,
    mutual_ratio_matches,
    pair_track_labels,
    reject_model_open,
    write_json,
)

# isort: split
from creator_eval.correspondence_band_split import band_split


def prepare(path):
    prior.SOURCES = [*prior.SOURCES, "scripts/run_foreground_track_bands.py", "experiments/src/creator_eval/correspondence_band_split.py"]
    prior.prepare(path)


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
            labels, split = band_split(graph["tracks"], features, method["split"])
            validation = labels == "validation"
            pair_labels = pair_track_labels(pairs, graph, validation)
            nonplanar_pairs = 0
            for pair, label in zip(pairs, pair_labels):
                # Keep excluded tracks in the audit. Only their fitting eligibility
                # changes, so this new split cannot make the truth audit look nicer.
                ids = label["track_ids"]
                eligible = np.array([i >= 0 and labels[i] != "excluded" for i in ids], bool)
                geometry = fit_rgb_geometry(pair["first_xy"][eligible], pair["second_xy"][eligible], label["validation"][eligible], method["geometry"])
                if geometry.get("models", {}).get("F", {}).get("state") == "validation_consistent" and not geometry["planar_or_repetitive_explanation_possible"]:
                    nonplanar_pairs += 1
                pair.update(**label, split_eligible=eligible, geometry=geometry)
            coverage = []
            for view in range(len(features)):
                present = np.array([any(v == view for v, _ in track) for track in graph["tracks"]], bool)
                coverage.append(dict(view=view, train=int((present & (labels == "train")).sum()), validation=int((present & validation).sum())))
            policy = method["availability"]
            sufficient = all(row["train"] >= policy["minimum_train_tracks_per_view"] and row["validation"] >= policy["minimum_validation_tracks_per_view"] for row in coverage)
            sufficient &= nonplanar_pairs >= policy["minimum_nonplanar_validated_pairs"]
            path = run / (case["case_id"] + "-" + detector + ".json")
            write_json(path, dict(case_id=case["case_id"], detector=detector, feature_counts=[len(f["xy"]) for f in features],
                feature_xy=[f["xy"] for f in features], graph=graph, validation_tracks=validation, split=split, pairs=pairs, coverage=coverage,
                band_labels=labels, availability="eligible_for_further_validation" if sufficient else "insufficient_for_camera_optimization",
                nonplanar_validated_pairs=nonplanar_pairs, camera_changes_applied=False))
            records.append(dict(path=path.name, sha256=digest(path), case_id=case["case_id"], detector=detector))
            print("BANDED_TRACKS", case["case_id"], detector, split["train_tracks"], split["validation_tracks"], split["excluded_tracks"], sufficient, flush=True)
    checked_tracks(run_id)
    write_json(run / "inference.json", dict(state="complete", input_sha256=frozen["input_sha256"], source_sha256=frozen["source_sha256"],
        records=records, gt_read_during_inference=False, model_geometry_read_during_inference=False,
        opencv_version=cv2.__version__, numpy_version=np.__version__, elapsed_seconds=time.perf_counter() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foreground_track_bands_v1.json")
    parser.add_argument("--run-id", default="foreground-track-bands-v1-20260922")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
