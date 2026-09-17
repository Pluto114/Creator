"""Prepare matched material controls, freeze RGB-only observations, then evaluate."""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_observations import extract_rod_observations, fit_robust_image_line
from evaluate_rod_evidence import centering_report
from run_thin_line_controls import clean

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rod_appearance_control_v1.json"
SHARE = ROOT / "docs/experiments/results/2026-09-17-appearance-control.json"


def locations(config):
    rid = config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", rid):
        raise ValueError("Invalid appearance run ID")
    return ROOT / ".runtime/experiments" / rid, ROOT / "data/inputs" / rid, ROOT / "data/eval_gt" / rid


def frozen_config(current, run, status):
    frozen = read_json(run / "protocol.json")
    if sha256(run / "protocol.json") != status["protocol_sha256"] or current != frozen:
        raise ValueError("Current appearance configuration differs from the frozen protocol")
    return frozen


def require_exact_replica(records):
    if not records or not all(r["decoded_exact"] and r["max_channel_difference"] == 0 for r in records):
        raise ValueError("Original-material replicas must exactly reproduce original RGB")


def prepare():
    config = read_json(CONFIG)
    run, inputs, truth = locations(config)
    if sha256(ROOT / config["source_scene"]) != config["source_sha256"]:
        raise ValueError("Scene changed")
    for directory in (run, inputs, truth):
        directory.mkdir(exist_ok=False)
    write_json(run / "protocol.json", config)
    for file in (Path(__file__), ROOT / "scripts/blender_rod_appearance_control.py"):
        shutil.copy2(file, run / file.name)
    request = {"root": str(ROOT), "inputs": inputs.relative_to(ROOT).as_posix(), "truth": truth.relative_to(ROOT).as_posix(), "config": config}
    write_json(run / "request.json", request)
    blender = Path(r"D:\CloudMusic\steam\steamapps\common\Blender\blender.exe")
    environment = os.environ.copy()
    for setting in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        profile = ROOT / ".local/blender-profile" / setting.lower()
        profile.mkdir(parents=True, exist_ok=True)
        environment["BLENDER_USER_" + setting] = str(profile)
    with (run / "blender.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([str(blender), "--background", "--factory-startup", "--disable-autoexec", str(ROOT / config["source_scene"]), "--python-exit-code", "1",
                                    "--python", str(ROOT / "scripts/blender_rod_appearance_control.py"), "--", str(run / "request.json")],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError("Blender control failed; see preserved blender.log")
    if sha256(ROOT / config["source_scene"]) != config["source_sha256"]:
        raise ValueError("Source scene changed")
    render = read_json(truth / "render_manifest.json")
    manifest = {"scope": config["scope"], "groups": []}
    replica = []
    reference_inputs = ROOT / "data/inputs" / config["reference_bundle"]
    reference_manifest = read_json(reference_inputs / "manifest.json")
    reference_group = next(g for g in reference_manifest["groups"] if g["case_id"] == config["reference_case"])
    for group in render["groups"]:
        frames = []
        for frame in group["frames"]:
            relative = Path(group["condition"]) / (frame["frame_id"] + ".png")
            frames.append({**frame, "rgb": relative.as_posix(), "sha256": sha256(inputs / relative)})
            if group["condition"] == "original_replica":
                old = next(f for f in reference_group["frames"] if f["frame_id"] == frame["frame_id"])
                if sha256(reference_inputs / old["rgb"]) != old["sha256"]:
                    raise ValueError("Original RGB changed")
                diff = np.array(Image.open(inputs / relative), dtype=int) - np.array(Image.open(reference_inputs / old["rgb"]), dtype=int)
                replica.append({"frame_id": frame["frame_id"], "original_sha256": old["sha256"], "decoded_exact": bool(np.all(diff == 0)),
                                "max_channel_difference": int(np.max(abs(diff))), "mean_absolute_channel_difference": float(np.mean(abs(diff)))})
        manifest["groups"].append({"condition": group["condition"], "frames": frames})
    write_json(inputs / "manifest.json", manifest)
    write_json(truth / "replica_check.json", replica)
    require_exact_replica(replica)
    write_json(truth / "artifact_hashes.json", {p.relative_to(truth).as_posix(): sha256(p) for p in sorted(truth.rglob("*")) if p.is_file()})
    write_json(run / "status.json", {"state": "prepared", "protocol_sha256": sha256(run / "protocol.json"),
                                     "input_manifest_sha256": sha256(inputs / "manifest.json"), "gt_index_sha256": sha256(truth / "artifact_hashes.json"),
                                     "reference_input_manifest_sha256": sha256(reference_inputs / "manifest.json"), "replicas_exact": True})
    print("PREPARED", replica, flush=True)


def infer():
    config = read_json(CONFIG)
    run, inputs, _ = locations(config)
    status = read_json(run / "status.json")
    config = frozen_config(config, run, status)
    if status["state"] != "prepared" or sha256(run / "protocol.json") != status["protocol_sha256"] or sha256(inputs / "manifest.json") != status["input_manifest_sha256"]:
        raise ValueError("Unprepared, already inferred or changed input")
    if not status.get("replicas_exact", False):
        raise ValueError("Replica identity gate has not passed")
    # The old method is reused verbatim. Today's truth is never opened here.
    old = ROOT / ".runtime/experiments" / config["frozen_method_run"]
    old_manifest = read_json(old / "manifest.json")
    for name in ("protocol", "selection"):
        if sha256(old / (name + ".json")) != old_manifest[name + "_sha256"]:
            raise ValueError("Old method identity mismatch")
    method, selection = read_json(old / "protocol.json"), read_json(old / "selection.json")
    frozen = run / "observation_sources"
    frozen.mkdir()
    shutil.copy2(ROOT / "experiments/src/creator_eval/rod_observations.py", frozen / "rod_observations.py")
    if sha256(frozen / "rod_observations.py") != old_manifest["source_hashes"]["rod_observations.py"]:
        raise ValueError("Detector changed since the reference experiment")
    records = []
    for group in read_json(inputs / "manifest.json")["groups"]:
        for frame in group["frames"]:
            path = inputs / frame["rgb"]
            if sha256(path) != frame["sha256"]:
                raise ValueError("Control RGB changed")
            rgb = np.array(Image.open(path).convert("RGB"))
            for target in config["target_aliases"]:
                xs = selection["line_control"]["guides"][frame["frame_id"]][target]
                ys = selection["line_control"]["guide_y"]
                rows = extract_rod_observations(rgb, np.c_[xs, ys], method["observation"])
                fitted = fit_robust_image_line(rows, method["line_fit"])
                records.append({"condition": group["condition"], "frame_id": frame["frame_id"], "target": target,
                                "extracted": rows, "fitted": fitted, "rgb_sha256": frame["sha256"]})
                print("APPEARANCE_OBSERVATIONS", group["condition"], frame["frame_id"], target, fitted["state"], flush=True)
    write_json(run / "observations.json", clean(records))
    status.update(state="inferred", gt_read_in_infer=False, observations_sha256=sha256(run / "observations.json"),
                  detector_sha256=sha256(frozen / "rod_observations.py"), reference_method_manifest_sha256=sha256(old / "manifest.json"))
    write_json(run / "status.json", status)


def evaluate():
    config = read_json(CONFIG)
    run, _, truth = locations(config)
    status = read_json(run / "status.json")
    config = frozen_config(config, run, status)
    if status["state"] != "inferred" or sha256(run / "observations.json") != status["observations_sha256"] or sha256(truth / "artifact_hashes.json") != status["gt_index_sha256"]:
        raise ValueError("Incomplete or changed appearance experiment")
    for relative, digest in read_json(truth / "artifact_hashes.json").items():
        if sha256(truth / relative) != digest:
            raise ValueError("Appearance GT changed")
    require_exact_replica(read_json(truth / "replica_check.json"))
    output = ROOT / "data/evaluation" / config["run_id"]
    # 换run ID不等于允许重写上一次的报告，分享文件也得换一个名字。
    if SHARE.exists():
        raise FileExistsError(f"Keep the old shared result; choose a new --share-json: {SHARE}")
    output.mkdir(exist_ok=False)
    result = {"state": "complete", "scope": config["scope"], "rows": [], "replica_check": read_json(truth / "replica_check.json"),
              "status_sha256": sha256(run / "status.json"), "protocol": config, "identities": status}
    for row in read_json(run / "observations.json"):
        geometry = read_json(truth / row["condition"] / "geometry.json")
        camera = read_json(truth / row["condition"] / (row["frame_id"] + "-camera.json"))
        rod_id = config["target_rod_ids_evaluation_only"][row["target"]]
        segments = [o["world_centerline_endpoints"] for o in geometry["objects"] if o["rod_id"] == rod_id and o["duplicate_geometry_of"] is None]
        xy = []
        if row["fitted"]["usable"]:
            for match in row["fitted"]["row_matches"]:
                observed = row["extracted"]["rows"][match["row_index"]]
                xy.append([observed["candidates"][match["candidate_index"]]["center_x"], observed["y"]])
        result["rows"].append({"condition": row["condition"], "frame_id": row["frame_id"], "target": row["target"],
                               "line_state": row["fitted"]["state"], **centering_report(xy, segments, camera)})
    write_json(output / "summary.json", clean(result))
    write_json(SHARE, clean(result))
    print("APPEARANCE_EVALUATION", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "evaluate"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--share-json", type=Path, default=SHARE)
    args = parser.parse_args()
    CONFIG = args.config
    SHARE = args.share_json
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {"prepare": prepare, "infer": infer, "evaluate": evaluate}[args.stage]()
