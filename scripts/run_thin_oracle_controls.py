"""Explicit oracle-camera controls; never mixed into the ordinary RGB-only runs."""
import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from thin_pack_gt import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "thin-oracle-controls-v1-20260914"


def main(run_id):
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    protocol = read_json(ROOT / "configs/thin_pack_controls_v1.json")
    bundle = ROOT / "data/inputs/thin-pack-v2-20260914r1"
    gt = ROOT / "data/eval_gt/thin-pack-v2-20260914r1"
    manifest = read_json(bundle / "manifest.json")
    status = read_json(gt / "status.json")
    if status["state"] != "complete" or sha256(gt / "artifact_hashes.json") != status["artifact_hashes_sha256"]:
        raise ValueError("GT identity mismatch")
    hashes = read_json(gt / "artifact_hashes.json")
    target = ROOT / ".runtime/experiments" / run_id
    target.mkdir(exist_ok=False)
    frozen = target / "producer_sources"
    frozen.mkdir()
    for source in [Path(__file__), ROOT / "scripts/thin_pack_oracle_worker.py", ROOT / "scripts/smoke_da3.py"]:
        shutil.copy2(source, frozen / source.name)
    record = {
        "run_id": run_id, "state": "running", "scope": "oracle_camera only; not fair RGB-only baseline",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_manifest_sha256": sha256(bundle / "manifest.json"),
        "gt_artifact_hashes_sha256": status["artifact_hashes_sha256"],
        "source_hashes": {p.name: sha256(p) for p in frozen.iterdir()}, "jobs": [],
    }
    write_json(target / "protocol.json", protocol)
    record["protocol_sha256"] = sha256(target / "protocol.json")
    write_json(target / "inference_manifest.json", record)
    for case in protocol["oracle_cases"]:
        group = next(g for g in manifest["groups"] if g["case_id"] == case)
        folder = target / (case + "-504")
        folder.mkdir()
        cameras = {"coordinate_convention": "opencv_world_to_camera_meters_K_edge", "frames": []}
        frames = []
        for frame in group["frames"]:
            relative = Path(case) / frame["frame_id"] / "camera.json"
            path = gt / relative
            if sha256(path) != hashes[relative.as_posix()]["sha256"]:
                raise ValueError("Camera hash mismatch")
            camera = read_json(path)
            cameras["frames"].append({
                "rgb_sha256": frame["sha256"], "world_to_camera_cv": camera["world_to_camera_cv"],
                "K_edge": camera["K_edge"], "width": camera["size_wh"][0], "height": camera["size_wh"][1],
            })
            frames.append({"rgb": str((bundle / frame["rgb"]).resolve()), "sha256": frame["sha256"]})
        write_json(folder / "oracle_cameras.json", cameras)
        request = {
            "model": "large", "process_res": 504, "frames": frames,
            "camera_file": str((folder / "oracle_cameras.json").resolve()),
            "camera_sha256": sha256(folder / "oracle_cameras.json"),
        }
        write_json(folder / "request.json", request)
        print("ORACLE_INFER", case, flush=True)
        started = time.perf_counter()
        with (folder / "inference.log").open("w", encoding="utf-8") as log:
            outcome = subprocess.run(
                [sys.executable, str(ROOT / "scripts/thin_pack_oracle_worker.py"),
                 "--request", str(folder / "request.json")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        report = read_json(folder / "report.json") if (folder / "report.json").exists() else {}
        entry = {
            "job_id": case+"-504", "case_id": case, "process_res": 504,
            "return_code": outcome.returncode, "wall_seconds": time.perf_counter()-started,
            "state": "succeeded" if outcome.returncode == 0 and report.get("ok") else "failed",
            "request_sha256": sha256(folder / "request.json"),
        }
        if entry["state"] == "succeeded":
            entry.update(prediction_sha256=sha256(folder / "prediction.npz"), report_sha256=sha256(folder / "report.json"))
        record["jobs"].append(entry)
        write_json(target / "inference_manifest.json", record)
        print("ORACLE", entry["state"], case, flush=True)
    record["state"] = "complete" if all(j["state"] == "succeeded" for j in record["jobs"]) else "complete_with_failures"
    write_json(target / "inference_manifest.json", record)
    if record["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    main(args.run_id)
