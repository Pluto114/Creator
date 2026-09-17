"""Inspect frozen RGB edge choices against independent labels; never produce method inputs."""
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from run_thin_line_controls import clean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARE = ROOT / "docs/experiments/results/2026-09-17-edge-attribution.json"


def visible_intervals(mask_row):
    selected = np.flatnonzero(mask_row)
    if not len(selected):
        return []
    runs = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    return [[float(r[0] - .5), float(r[-1] + .5)] for r in runs]


def projected_axis_at_rows(segments, camera, rows):
    ext, intrinsic = np.array(camera["world_to_camera_cv"]), np.array(camera["K_index"])
    xyz = np.array(segments) @ ext[:3, :3].T + ext[:3, 3]
    if np.any(xyz[..., 2] <= camera["clip_start"]):
        raise ValueError("Truth segment behind near plane")
    uv = xyz @ intrinsic.T
    uv = uv[..., :2] / uv[..., 2, None]
    result = np.full(len(rows), np.nan)
    for first, last in uv:
        fraction = (rows - first[1]) / (last[1] - first[1])
        included = (fraction >= 0) & (fraction <= 1)
        if np.any(np.isfinite(result[included])):
            raise ValueError("Overlapping projected centerline row ranges require explicit handling")
        result[included] = first[0] + fraction[included] * (last[0] - first[0])
    return result


def audit(protocol_path, share_path=DEFAULT_SHARE):
    protocol = read_json(protocol_path)
    run_id = protocol["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid audit run ID")
    if share_path.exists():
        raise FileExistsError(f"Keep the old shared result; choose a new --share-json: {share_path}")
    output = ROOT / "data/evaluation" / run_id
    output.mkdir(exist_ok=False)
    shutil.copy2(protocol_path, output / "protocol.json")
    shutil.copy2(__file__, output / "audit_rod_edges.py")
    source = ROOT / ".runtime/experiments" / protocol["observation_run"]
    source_manifest = read_json(source / "manifest.json")
    bundle = ROOT / "data/inputs" / protocol["input_bundle"]
    if source_manifest["state"] != "complete" or sha256(bundle / "manifest.json") != source_manifest["input_manifest_sha256"]:
        raise ValueError("Input/observation identity mismatch")
    inputs = read_json(bundle / "manifest.json")
    gt = ROOT / "data/eval_gt" / protocol["gt_bundle"]
    status = read_json(gt / "status.json")
    if status["state"] != "complete" or sha256(gt / "artifact_hashes.json") != status["artifact_hashes_sha256"]:
        raise ValueError("GT hash index mismatch")
    hashes = read_json(gt / "artifact_hashes.json")

    def checked(relative):
        path = gt / relative
        if sha256(path) != hashes[Path(relative).as_posix()]["sha256"]:
            raise ValueError("GT artifact mismatch")
        return path

    if read_json(checked("validation_summary.json"))["input_manifest_sha256"] != source_manifest["input_manifest_sha256"]:
        raise ValueError("GT belongs to different RGB")
    result = {"state": "running", "run_id": run_id, "scope": protocol["scope"], "rows": [], "frames": [], "profiles": [],
              "source_manifest_sha256": sha256(source / "manifest.json"), "input_manifest_sha256": sha256(bundle / "manifest.json"),
              "gt_hash_index_sha256": status["artifact_hashes_sha256"], "protocol_sha256": sha256(output / "protocol.json"),
              "producer_sha256": sha256(output / "audit_rod_edges.py")}
    for case in protocol["cases"]:
        group = next(g for g in inputs["groups"] if g["case_id"] == case)
        artifact = source_manifest["observations"][case]
        if sha256(source / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen observations changed")
        observations = read_json(source / artifact["path"])
        geometry = read_json(checked(Path(case) / "geometry.json"))
        for index, frame in enumerate(group["frames"]):
            rgb_path = bundle / frame["rgb"]
            if sha256(rgb_path) != frame["sha256"]:
                raise ValueError("RGB changed")
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            gray = rgb.astype(float) @ np.array([.2126, .7152, .0722])
            camera = read_json(checked(Path(case) / frame["frame_id"] / "camera.json"))
            ids = np.load(checked(Path(case) / frame["frame_id"] / "native/rod_id.npy"), allow_pickle=False)
            for target, rod_id in protocol["target_rod_ids"].items():
                obs = observations[target][index]
                if obs["frame_id"] != frame["frame_id"]:
                    raise ValueError("Observation order mismatch")
                selected = {m["row_index"]: m["candidate_index"] for m in obs["fitted"]["row_matches"]} if obs["fitted"]["usable"] else {}
                rows = obs["extracted"]["rows"]
                truth = [o["world_centerline_endpoints"] for o in geometry["objects"]
                         if o["rod_id"] == rod_id and o["duplicate_geometry_of"] is None]
                axes = projected_axis_at_rows(truth, camera, np.array([r["y"] for r in rows]))
                local = []
                for ri, row in enumerate(rows):
                    intervals = visible_intervals(ids[row["y"]] == rod_id)
                    candidate = row["candidates"][selected[ri]] if ri in selected else None
                    entry = {"case_id": case, "frame_id": frame["frame_id"], "target": target, "y": row["y"],
                             "detector_state": row["status"], "line_usable": obs["fitted"]["usable"],
                             "candidate_count": len(row["candidates"]), "selected": candidate is not None,
                             "axis_x": axes[ri], "visible_intervals": intervals}
                    if candidate is not None:
                        left, right = candidate["left_edge"]["x"], candidate["right_edge"]["x"]
                        entry.update(center_x=candidate["center_x"], pair_width_px=candidate["width"], edge_x=[left, right],
                                     center_minus_axis_px=candidate["center_x"] - axes[ri], classification="no_single_visible_finite_truth_interval")
                        if len(intervals) == 1 and np.isfinite(axes[ri]):
                            low, high = intervals[0]
                            tolerance = protocol["boundary_tolerance_px"]
                            same_silhouette = abs(left - low) <= tolerance and abs(right - high) <= tolerance
                            narrow_inside = left >= low - tolerance and right <= high + tolerance
                            entry.update(mask_width_px=high - low, pair_to_mask_width=candidate["width"] / (high - low),
                                         mask_midpoint_minus_axis_px=(low + high) / 2 - axes[ri],
                                         classification="both_visible_boundaries" if same_silhouette else (
                                             "inside_visible_rod_nonboundary_pair" if narrow_inside else "pair_extends_beyond_visible_rod"))
                    if row["y"] in protocol["fixed_profile_rows"]:
                        left = max(0, int(np.floor(row["guide_x"])) - protocol["profile_margin_px"])
                        right = min(gray.shape[1], int(np.ceil(row["guide_x"])) + protocol["profile_margin_px"] + 1)
                        result["profiles"].append({**entry, "x": list(range(left, right)), "gray": gray[row["y"], left:right],
                                                   "all_pairs": [{"center_x": c["center_x"], "edge_x": [c["left_edge"]["x"], c["right_edge"]["x"]]} for c in row["candidates"]]})
                    if candidate is not None:
                        local.append(entry)
                        result["rows"].append(entry)
                finite = [r for r in local if np.isfinite(r["axis_x"])]
                mask = [r for r in finite if "pair_to_mask_width" in r]
                result["frames"].append({"case_id": case, "frame_id": frame["frame_id"], "target": target,
                                         "line_state": obs["fitted"]["state"], "selected_rows": len(local), "finite_axis_rows": len(finite),
                                         "single_interval_rows": len(mask), "classification_counts": {c: sum(r.get("classification") == c for r in local) for c in
                                             ["both_visible_boundaries", "inside_visible_rod_nonboundary_pair", "pair_extends_beyond_visible_rod", "no_single_visible_finite_truth_interval"]},
                                         "signed_center_bias_median_px": float(np.median([r["center_minus_axis_px"] for r in finite])) if finite else None,
                                         "pair_to_mask_width_median": float(np.median([r["pair_to_mask_width"] for r in mask])) if mask else None})
    result["state"] = "complete"
    write_json(output / "summary.json", clean(result))
    write_json(share_path,
               clean({k: v for k, v in result.items() if k not in ("rows", "profiles")}))
    print("EDGE_AUDIT", output, flush=True)
    for frame in result["frames"]:
        print(frame, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/rod_edge_audit_v1.json")
    parser.add_argument("--share-json", type=Path, default=DEFAULT_SHARE)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit(args.protocol, args.share_json)
