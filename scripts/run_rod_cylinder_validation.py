"""Freeze new within-family Blender layouts and evaluate a fixed cylinder method."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_candidate_association import associate_multiview_lines_cached  # noqa: E402
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_candidate_pool import enumerate_image_line_pool  # noqa: E402
from creator_eval.rod_cylinder_gate import recheck_finite_cylinder  # noqa: E402
from creator_eval.rod_cylinder_support import select_supported_cylinders  # noqa: E402
from creator_eval.rod_multiview_candidates import apply_guide_identity_guard  # noqa: E402
from creator_eval.rod_observations import extract_rod_observations  # noqa: E402
from run_rod_candidate_ablation import (  # noqa: E402
    artifact_record,
    canonical_hash,
    checked_artifact,
    expected_keys,
    inference_view,
)
from run_rod_identity_blender import digest, locations, prepare, read_json, write_json  # noqa: E402
from run_rod_identity_stress import (  # noqa: E402
    frozen_source_check,
    reject_truth_open,
    validate_inference_identity,
)
from run_rod_support_recheck import score_record, verify_final_support  # noqa: E402

CONFIG = ROOT / "configs/rod_cylinder_validation_v1.json"
RUN_ID = "rod-cylinder-new-layout-v1-20260920"
SHARE = ROOT / "docs/experiments/results/2026-09-20-cylinder-new-layout.json"


def checked_run(run_id):
    run, inputs, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["state"] != "prepared" or prepared["run_id"] != run_id:
        raise ValueError("Prepared run identity changed")
    frozen_source_check(run, prepared)
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    method, manifest = read_json(run / "method_config.json"), read_json(inputs / "manifest.json")
    if set(c["case_id"] for c in manifest["cases"]) != set(method["case_ids"]):
        raise ValueError("Method case plan differs from rendered input manifest")
    return run, inputs, prepared, method, manifest


def infer_pool(run_id, case, search):
    run, inputs, _, method, _ = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    begin = time.perf_counter()
    frames = []
    for frame in case["frames"]:
        path = inputs / frame["rgb"]
        if digest(path) != frame["rgb_sha256"]:
            raise ValueError("Frozen RGB changed")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        guide = np.asarray(frame["guide_xyxy"], float).copy()
        guide[:, 0] += search
        raw = extract_rod_observations(rgb, guide, method["observation"])
        pool = enumerate_image_line_pool(raw, method["image_hypotheses"], require_refit_support=True)
        frames.append({**frame, "rgb": path.relative_to(ROOT).as_posix(), "observations": raw, "pool": pool})
    verify_final_support(frames, method["image_hypotheses"])
    pool_path = run / "pools" / f"{case['case_id']}-search-{search}.json"
    write_json(pool_path, {"case_id": case["case_id"], "search_offset_px": search, "frames": frames})
    pool_record = artifact_record(run, pool_path, case_id=case["case_id"], search_offset_px=search)
    records, brief = [], []
    for cap in method["candidate_caps"]:
        views = [inference_view(f, f["pool"]["candidates"][:cap], 0) for f in frames]
        raw = [f["observations"] for f in frames]
        started = time.perf_counter()
        association = associate_multiview_lines_cached(views, method["association_common"])
        seconds = time.perf_counter() - started
        if not association["search_complete"]:
            raise ValueError("New-layout retained-pool search incomplete")
        filtered = select_supported_cylinders(association, views, raw, method["extent"], method["cylinder_screen"])
        for name, proposal in (("baseline", association), ("cylinder_support", filtered)):
            sha = canonical_hash(proposal)
            variants = []
            for identity in method["identity_offsets_px"]:
                variant_views = [inference_view(f, f["pool"]["candidates"][:cap], identity) for f in frames]
                guarded = apply_guide_identity_guard(proposal, variant_views, method["guide_guard"])
                finite = bound_selected_candidate(guarded, variant_views, raw, method["extent"])
                if name == "cylinder_support":
                    finite = recheck_finite_cylinder(finite, guarded["selected"], variant_views, raw, method["cylinder_screen"])
                variants.append({"identity_offset_px": identity, "association_sha256": sha,
                                 "guarded": {k: guarded[k] for k in ("state", "reason", "guide_identity_guard")}, "finite": finite})
            if canonical_hash(proposal) != sha:
                raise ValueError("Variant mutated the shared association")
            path = run / "associations" / f"{case['case_id']}-search-{search}-cap-{cap}-{name}.json"
            write_json(path, {"case_id": case["case_id"], "search_offset_px": search, "cap": cap, "pool_sha256": pool_record["sha256"],
                              "method": name, "association": proposal, "association_sha256": sha, "association_seconds": seconds,
                              "source_geometry_sha256": canonical_hash(association), "variants": variants})
            records.append(artifact_record(run, path, method=name))
            brief.append([cap, name, proposal["state"], [v["finite"]["state"] for v in variants]])
    return {"pool": pool_record, "associations": records, "elapsed_seconds": time.perf_counter() - begin, "brief": brief}


def infer(run_id):
    run, _, prepared, method, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    begin = time.perf_counter()
    (run / "pools").mkdir()
    (run / "associations").mkdir()
    records = []
    with ProcessPoolExecutor(max_workers=method["parallel_workers"]) as executor:
        futures = [executor.submit(infer_pool, run_id, case, search) for case, search in itertools.product(manifest["cases"], method["search_offsets_px"])]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print("NEW_LAYOUT_POOL", record["pool"]["case_id"], record["pool"]["search_offset_px"], record["brief"], flush=True)
    records.sort(key=lambda r: (r["pool"]["case_id"], r["pool"]["search_offset_px"]))
    write_json(run / "inference.json", {
        **{k: prepared[k] for k in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256")},
        "state": "inferred", "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "elapsed_seconds": time.perf_counter() - begin, "workers": method["parallel_workers"],
        "pools": [r["pool"] for r in records], "associations": [a for r in records for a in r["associations"]],
    })


def evaluate(config_path, share):
    config = read_json(config_path)
    run, _, prepared, method, _ = checked_run(config["run_id"])
    _, _, truth, output = locations(config["run_id"])
    if output.exists() or share.exists():
        raise FileExistsError("Keep earlier validation scores")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Frozen evaluation protocol changed")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth / filename) != prepared[key]:
            raise ValueError("Truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError("Truth artifact changed")
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    inference = read_json(run / "inference.json")
    validate_inference_identity(inference, prepared)
    pool_hashes = {}
    for entry in inference["pools"]:
        pool = checked_artifact(run, entry)
        key = (pool["case_id"], pool["search_offset_px"])
        if key in pool_hashes:
            raise ValueError("Duplicate pool")
        verify_final_support(pool["frames"], method["image_hypotheses"])
        pool_hashes[key] = entry["sha256"]
    if set(pool_hashes) != set(itertools.product(method["case_ids"], method["search_offsets_px"])):
        raise ValueError("Incomplete new-layout pools")
    rows = {"baseline": [], "cylinder_support": []}
    hashes = {}
    for entry in inference["associations"]:
        record = checked_artifact(run, entry)
        key = (record["case_id"], record["search_offset_px"], record["cap"])
        name = entry["method"]
        if name != record["method"] or record["pool_sha256"] != pool_hashes[key[:2]]:
            raise ValueError("Method or pool identity mismatch")
        hashes.setdefault(key, []).append(record["source_geometry_sha256"])
        if name == "baseline" and record["source_geometry_sha256"] != record["association_sha256"]:
            raise ValueError("Baseline is not the shared geometric association")
        if name == "cylinder_support":
            for variant in record["variants"]:
                finite = variant["finite"]
                if finite["state"] == "accepted" and finite.get("final_cylinder_screen", {}).get("passed") is not True:
                    raise ValueError("Accepted axis lacks final screen")
        scored = score_record(record, targets[key[0]], config["evaluation"])
        for row in scored:
            row["label"] = targets[key[0]]["label"]
        rows[name].extend(scored)
    for name, values in rows.items():
        keys = [(r["case_id"], r["search_offset_px"], r["cap"], r["identity_offset_px"]) for r in values]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys(method):
            raise ValueError("Missing or duplicate validation conditions:" + name)
    if any(len(values) != 2 or len(set(values)) != 1 for values in hashes.values()):
        raise ValueError("Methods did not share exactly the same geometric association")
    probes = [f for t in targets.values() for f in t["frames"]]
    result = {"state": "complete", "run_id": config["run_id"], "scope": config["scope"],
              "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(run / "inference.json"),
              "row_count": len(rows["cylinder_support"]), "rows": rows["cylinder_support"], "baseline_rows_rescored": rows["baseline"],
              "elapsed_seconds": inference["elapsed_seconds"], "workers": inference["workers"],
              "independent_blender_checks": {"frames": len(probes), "ray_count": sum(f["blender_ray_check"]["samples"] for f in probes),
                   "surface_id_mismatches": sum(f["blender_ray_check"]["surface_id_mismatches"] for f in probes),
                   "max_z_error_m": max(f["blender_ray_check"]["max_z_error_m"] for f in probes),
                   "projection_max_px": max(f["blender_projection_error_px"] for f in probes)},
              "limitations": ["New within-family layouts, not independent objects or photos; exact cameras.",
                              "All parameters frozen before render; after this scoring these scenes are no longer blind.",
                              "Only retained best/support competitors screened; no full image-hypothesis completeness.",
                              "No DA3 correction, original 0.01-pack recovery, or formal patch claim."]}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share, result)
    print("EVALUATED_NEW_LAYOUT", len(rows["cylinder_support"]), "paired conditions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--share-json", type=Path, default=SHARE)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.config, args.share_json)
