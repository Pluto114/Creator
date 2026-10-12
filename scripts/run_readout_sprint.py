"""Single-pass 50-configuration development screen, explicitly not G1 acceptance."""

import argparse
import importlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import foreground_consensus_runner as legacy
import numpy as np

ROOT = legacy.ROOT
DEFAULT_ADAPTER = "creator_eval.rgb_foreground_consensus_axis_readout:ForegroundConsensusAxisReadout"
ARMS = ("base", "depth_tls_single_span", "depth_ransac_single_span", "baseline", "cylinder_support")
SCALES = (.0025, .005)
FLAGS = dict(development_quick_screen=True, g1_passed=False, reader_qualified=False,
             blind_evaluation=False, fresh_repeats=0, end_to_end_repeat_performed=False)


def load_adapter(name):
    module, attribute = name.split(":")
    return getattr(importlib.import_module(module), attribute)


def compact(graph):
    result = {k: v for k, v in graph.items() if v is None or isinstance(v, (str, bool, int, float))}
    for name in ("state", "segments", "retained_groups", "competition_order", "effective_policy",
                 "selected_band_score", "foreground_bands"):
        if name in graph:
            result[name] = graph[name]
    result["group_summary"] = []
    for row in graph.get("readouts", []):
        summary = {k: v for k, v in row.items() if v is None or isinstance(v, (str, bool, int, float))}
        summary["band_score"] = row.get("band_score")
        candidate = row.get("graph") or row.get("candidate_graph") or {}
        summary["candidate_graph"] = {k: candidate[k] for k in ("state", "reason", "model", "segments") if k in candidate}
        result["group_summary"].append(summary)
    return result


def normal(run, adapter_name):
    run.mkdir(parents=True, exist_ok=False)
    started, opened, rows = time.perf_counter(), set(), []
    sys.addaudithook(legacy.objects.block_truth)
    def track(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if isinstance(args[1], str) and not any(letter in args[1] for letter in ("r", "+")):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if (path.is_relative_to(ROOT) and not path.is_relative_to(run)
                and not any(part.startswith(".venv") or part in (".local", "__pycache__") for part in path.parts)):
            opened.add(path)
    sys.addaudithook(track)
    adapter_type = load_adapter(adapter_name)
    for kind, case_ids in legacy.CASES.items():
        runner = legacy.Runner(kind)
        cases = legacy.read(runner.point.INPUTS / "manifest.json")["cases"]
        if [case["case_id"] for case in cases] != case_ids:
            raise ValueError("Full original five-case inventory required")
        references = {(row["case_id"], row["variant"]): row for row in legacy.read(runner.old.RUN / "inference.json")["rows"]
                      if row["repeat"] == 0 and row["fraction"] == SCALES[0] and row["reader"] == "stroke_chain_axis"}
        for case in cases:
            case_started, cid = time.perf_counter(), case["case_id"]
            opened.update((ROOT / frame["rgb"]).resolve() for frame in case["frames"])
            raw = legacy.read(ROOT / case["parent_record"])
            cameras = legacy.read(runner.point.RUN / cid / "rods.json")["cameras"]
            anchors, _ = runner.old.anchor_input(cid)
            context = runner.old.condition_stroke_context(runner.old.build_context(raw["frames"], cameras), raw["frames"], cameras, anchors)
            span = legacy.fixture.previous.camera_span(cameras)
            base = runner.old.compose(runner.candidates(cid, "base", 0)[0])
            adapter = adapter_type(base["points"], context, runner.old.POLICY)
            for arm in ARMS:
                reference = legacy.read(runner.old.RUN / references[cid, arm]["path"])
                segments = np.empty((0, 2, 3)) if arm == "base" else runner.old.open_candidate_view(runner.candidates(cid, arm, 0)[1])["segments"]
                if not np.array_equal(segments, np.asarray(reference["native_segments"]).reshape(-1, 2, 3)):
                    raise ValueError("Original native candidate geometry changed")
                for fraction in SCALES:
                    call_started, error = time.perf_counter(), None
                    try:
                        graph = compact(adapter(segments, dict(voxel_size=span*fraction, origin=[0., 0., 0.])))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        graph = dict(state="error", reason=error, segments=[])
                    rows.append(dict(kind=kind, case_id=cid, variant=arm, fraction=fraction, graph=graph,
                                     native_segments=segments, native_state=reference["native_state"],
                                     error=error, reader_elapsed_seconds=time.perf_counter()-call_started))
            print("SPRINT_NORMAL", cid, "rows", len(rows), "case_seconds", round(time.perf_counter()-case_started, 3), flush=True)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and not any(p.startswith(".venv") for p in path.parts):
                opened.add(path)
    hashes = {path.relative_to(ROOT).as_posix(): legacy.objects.digest(path) for path in sorted(opened)}
    legacy.write(run / "normal.json", dict(run_id=run.name, adapter=adapter_name, gt_read=False,
                 rows=rows, elapsed_seconds=time.perf_counter()-started, **FLAGS))
    legacy.write(run / "manifest.json", dict(run_id=run.name, state="normal_complete", gt_read=False,
                 adapter=adapter_name, normal_sha256=legacy.digest(run / "normal.json"), direct_hashes=hashes,
                 expected_rows=50, source_policy="direct inputs, referenced RGB and imported project sources hashed once after normal; not an immutable ancestry freeze", **FLAGS))
    print("SPRINT_NORMAL_COMPLETE", run.name, len(rows), "ready_for_explicit_evaluate", flush=True)


def evaluate(run):
    started = time.perf_counter()
    if (run / "evaluation.json").exists():
        raise FileExistsError("Preserve previous screen scores")
    manifest, data = legacy.read(run / "manifest.json"), legacy.read(run / "normal.json")
    if manifest["normal_sha256"] != legacy.digest(run / "normal.json") or manifest["state"] != "normal_complete" or len(data["rows"]) != 50:
        raise ValueError("Complete unchanged normal and manifest required before scores")
    expected_rows = {(kind, cid, arm, fraction) for kind, ids in legacy.CASES.items() for cid in ids for arm in ARMS for fraction in SCALES}
    if {(r["kind"], r["case_id"], r["variant"], r["fraction"]) for r in data["rows"]} != expected_rows or data["gt_read"] is not False:
        raise ValueError("Exactly all five cases, five arms and two scales required")
    truths, truth_hashes = {}, {}
    for kind, ids in legacy.CASES.items():
        runner = legacy.Runner(kind)
        scene = legacy.read(runner.point.CONFIG)["scene_run_id"]
        path = ROOT / "data/eval_gt" / scene / "manifest.json"
        expected = legacy.read(ROOT / ".runtime/experiments" / scene / "prepared.json")["truth_sha256"]
        if legacy.digest(path) != expected:
            raise ValueError("Original full GT bytes changed")
        truth_hashes[path.relative_to(ROOT).as_posix()] = expected
        truth = legacy.read(path)
        if [case["case_id"] for case in truth["cases"]] != ids:
            raise ValueError("Full original truth cases required")
        truths.update({case["case_id"]: case["declared"] for case in truth["cases"]})
    for row in data["rows"]:
        declared = truths[row["case_id"]]
        target = declared["target"]
        truth = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        row["physical"], row["scoring_error"] = legacy.objects.score_geometry(row["graph"]["state"], row["graph"]["segments"], truth, gaps)
        row["native_physical"], row["native_scoring_error"] = legacy.objects.score_geometry(row["native_state"], row["native_segments"], truth, gaps)
    legacy.write(run / "evaluation.json", dict(**data, normal_sha256=manifest["normal_sha256"],
                 evaluation_gt_read=True, truth_hashes=truth_hashes, evaluation_elapsed_seconds=time.perf_counter()-started))
    print("SPRINT_EVALUATED", run.name, len(data["rows"]), "development_only_not_G1", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("normal", "evaluate"))
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.stage == "evaluate" and not args.run_id:
        parser.error("evaluate requires the completed normal run ID")
    name = args.run_id or "readout-sprint-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}", name):
        parser.error("A single safe new run-directory name is required")
    run, started = ROOT / ".runtime/experiments" / name, time.perf_counter()
    if args.stage == "normal" and run.exists():
        raise FileExistsError("Use a new run ID; previous attempts are preserved")
    try:
        normal(run, args.adapter) if args.stage == "normal" else evaluate(run)
    finally:
        if run.exists() and not (run / (args.stage + "-work-log.json")).exists():
            legacy.write(run / (args.stage + "-work-log.json"), dict(stage=args.stage, adapter=args.adapter if args.stage == "normal" else None,
                         elapsed_seconds=time.perf_counter()-started, error=None if sys.exc_info()[1] is None else repr(sys.exc_info()[1]), **FLAGS))


if __name__ == "__main__":
    main()
