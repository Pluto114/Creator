"""Read-only gate tracing of the frozen September 30 ridge reader.

Replay selected existing development inputs with their original policies and
assert exact output equality. No truth, scores, new algorithms, sweeps, parameter
changes or output files are permitted. JSON lines are printed to stdout only.
"""

from __future__ import annotations

import collections
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "mixed-readout-ridges-v1-20260930"
INPUT_IDS = (
    "control-0000", "control-0001", "control-0002", "control-0003",
    "control-0026", "control-0027", "control-0030", "control-0031",
)
sys.dont_write_bytecode = True


def guard(event, arguments):
    if event != "open" or not arguments:
        return
    name, mode, flags = arguments
    if (isinstance(mode, str) and any(mark in mode for mark in "wax+")) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    ):
        raise PermissionError("Diagnostic cannot write files")
    if not isinstance(name, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(name)).resolve()
    forbidden = (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments")
    if any(path.is_relative_to(folder) for folder in forbidden) or path.name in {
        "evaluation.json", "post.json", "protocol.json", "generation-checks.json",
    }:
        raise PermissionError("Diagnostic cannot read truth or physical scores")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.addaudithook(guard)
    sys.path.insert(0, str(ROOT / "experiments/src"))
    from creator_eval import common_readout_ridges as reader

    run = ROOT / ".runtime/experiments" / RUN_ID
    source = Path(reader.__file__)
    frozen = run / "source_snapshot" / source.relative_to(ROOT)
    if source.read_bytes() != frozen.read_bytes():
        raise ValueError("Original frozen reader differs; diagnostic must not trace new code")
    traced = (reader._ridge_runs, reader._fit_ridge)
    sources = {function.__code__: inspect.getsourcelines(function) for function in traced}
    for input_id in INPUT_IDS:
        path = run / "records" / f"{input_id}-ridges-native.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        input_path = ROOT / "data/inputs" / RUN_ID / f"{input_id}.npz"
        input_sha = digest(input_path)
        if record["input_id"] != input_id or record["gt_read"] or input_sha != record["input_sha256"]:
            raise ValueError("Original normal record or input identity differs")
        with np.load(input_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"points", "segments"}:
                raise ValueError("Only unlabelled point and segment inputs are allowed")
            points, segments = arrays["points"].copy(), arrays["segments"].copy()
        stats, checks, peers = collections.Counter(), [], []

        def trace(frame, event, argument):
            if frame.f_code not in sources:
                return None
            lines, start = sources[frame.f_code]
            offset = frame.f_lineno-start
            text = lines[offset].strip() if 0 <= offset < len(lines) else ""
            values = frame.f_locals
            if frame.f_code is reader._ridge_runs.__code__:
                if event == "call":
                    stats["ridge_calls"] += 1
                if event == "line" and text.startswith("if span <"):
                    stats["raw_axial_runs"] += 1
                    stats["short_run"] += int(values["span"] < values["p"]["minimum_run_length_voxels"]*values["voxel"])
                if event == "line" and text.startswith("if (coverage <"):
                    policy = values["p"]
                    reasons = []
                    decisions = {
                        "coverage": values["coverage"] < policy["minimum_axial_coverage"],
                        "fold": min(values["validation"], values["fitting"]) < policy["minimum_validation_coverage"],
                        "core_fraction": values["fraction"] < policy["minimum_core_fraction"],
                    }
                    for reason, rejected in decisions.items():
                        if rejected:
                            stats[reason] += 1
                            reasons.append(reason)
                    detail = {key: float(values[key]) for key in (
                        "span", "coverage", "fitting", "validation", "fraction", "core_count", "expanded_count",
                    )}
                    detail.update(core_points=int(values["support"].sum()),
                                  expanded_points=int(values["expanded"].sum()), reasons=reasons)
                    checks.append(detail)
                if event == "line" and text == "if competing:":
                    stats["peer_reject"] += int(values["competing"])
                    stats["passed_all"] += int(not values["competing"])
                    peers.append(dict(span=float(values["span"]), competing=bool(values["competing"])))
            if frame.f_code is reader._fit_ridge.__code__ and event == "return":
                stats["fit_calls"] += 1
                stats["fit_none"] += int(argument is None)
            return trace

        sys.settrace(trace)
        try:
            result = reader.readout(points, segments, record["call_policy"])
        finally:
            sys.settrace(None)
        if not np.array_equal(result["segments"], np.asarray(record["result"]["segments"]).reshape(-1, 2, 3)):
            raise ValueError("Instrumented geometry differs from frozen normal output")
        if result["config"] != record["effective_policy"]:
            raise ValueError("Instrumented policy differs from frozen policy")
        payload = dict(input_id=input_id, exact_geometry_replay=True, gt_read=False,
            reader_sha256=digest(source), input_sha256=input_sha,
            voxel_size=result["config"]["voxel_size"],
            counts_include_pre_and_post_merge_calls=True, gate_counts=dict(stats),
            proposals=result["ridge_proposals"], unique_fits=result["unique_fits"],
            accepted_pre_merge=result["accepted_ridge_runs"],
            post_merge=len(result["post_merge_ridge_evidence"]),
            parallel_rejected=result["ambiguous_parallel_runs"],
            longest_run_checks=sorted(checks, key=lambda item: -item["span"])[:6],
            peer_checks=peers,
            output_lengths=np.linalg.norm(result["segments"][:, 1]-result["segments"][:, 0], axis=1).tolist())
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
