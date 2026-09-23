"""Replay the frozen robust-section reader on all seven complete historical pairs."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import run_g1_point_patch_components as pipeline
from run_rod_identity_blender import write_json
from run_rod_identity_stress import reject_truth_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_sections import readout  # noqa: E402

RUN_ID = "g1-point-patch-sections-v1-20260923r2"
CONFIG = ROOT / "configs/g1_point_patch_sections_v1.json"
SOURCES = [*pipeline.SOURCES, "scripts/run_g1_point_patch_sections.py", "experiments/src/creator_eval/common_readout_split.py", "experiments/src/creator_eval/common_readout_sections.py"]


def infer_job(run_id, job):
    # Windows重新启动子进程，不继承父进程的函数替换。每个worker明确选读取器。
    pipeline.readout = readout
    return pipeline.infer_job(run_id, job)


def infer(run_id):
    run, method, frozen = pipeline.checked(run_id)
    sys.addaudithook(reject_truth_open)
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=method["parallel_workers"]) as pool:
        jobs = [pool.submit(infer_job, run_id, job) for job in method["jobs"]]
        for job in as_completed(jobs):
            record = job.result()
            records.append(record)
            print("SPLIT_POINT_PATCH_JOB", record["path"], flush=True)
    write_json(run / "inference.json", dict(run_id=run_id, input_sha256=frozen["input_sha256"], source_sha256=frozen["source_sha256"],
        state="complete", gt_read_during_inference=False, truth_read_tripwire_enabled=True, records=sorted(records, key=lambda r: r["path"]),
        elapsed_seconds=time.perf_counter() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--share-json", type=Path, default=ROOT / "docs/experiments/results/2026-09-23-point-patch-sections-r2.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        pipeline.SOURCES = SOURCES
        pipeline.prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        pipeline.evaluate(args.config, args.share_json)
