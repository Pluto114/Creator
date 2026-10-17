"""One complete 60-row reader comparison on unchanged saved native candidates."""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import foreground_consensus_runner as legacy
import numpy as np
import run_boundary_patch_sprint as boundary
import run_g1_object_point_patch as point
import run_readout_sprint as sprint

ROOT = legacy.ROOT
PARENT_ID = "boundary-patch-sprint-20261014-001"
PARENT = ROOT / ".runtime/experiments" / PARENT_ID
ADAPTER = "creator_eval.rgb_raster_band_readout:RasterBandReadout"
ARMS = boundary.ARMS
FLAGS = dict(sprint.FLAGS, reader_only_intervention=True, fresh_readout_rows=60,
    reused_control_rows=0, reference_only_old_rows=60, fresh_patch_count=0,
    fresh_depth_processes=0, native_geometry_unchanged=True,
    development_prior_scores_seen=True)


def normal(run):
    run.mkdir(parents=True, exist_ok=False)
    started, opened, rows, cases_log = time.perf_counter(), set(), [], []
    sys.addaudithook(legacy.objects.block_truth)

    def track(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if isinstance(args[1], str) and not any(c in args[1] for c in ("r", "+")):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if (path.is_relative_to(ROOT) and not path.is_relative_to(run)
                and not any(p.startswith(".venv") or p in (".local", "__pycache__") for p in path.parts)):
            opened.add(path)

    sys.addaudithook(track)
    from creator_eval import raster_band_readout as reader
    parent_manifest, parent = legacy.read(PARENT / "manifest.json"), legacy.read(PARENT / "normal.json")
    parent_sha = legacy.digest(PARENT / "normal.json")
    if (parent_manifest["normal_sha256"] != parent_sha
            or parent_manifest["state"] != "normal_complete" or parent["adapter"] != boundary.ADAPTER):
        raise ValueError("Unchanged complete stable parent normal required")
    boundary.validate_rows(parent, ARMS)
    references = {(r["case_id"], r["variant"], r["fraction"]): r for r in parent["rows"]}
    patches = {p["case_id"]: p for p in parent["patches"]}
    adapter_type = sprint.load_adapter(ADAPTER)
    for kind, case_ids in legacy.CASES.items():
        runner = legacy.Runner(kind)
        cases = legacy.read(runner.point.INPUTS / "manifest.json")["cases"]
        if [c["case_id"] for c in cases] != case_ids:
            raise ValueError("Full original five-case inventory required")
        for case in cases:
            case_started, cid = time.perf_counter(), case["case_id"]
            opened.update((ROOT / f["rgb"]).resolve() for f in case["frames"])
            raw = legacy.read(ROOT / case["parent_record"])
            cameras = legacy.read(runner.point.RUN / cid / "rods.json")["cameras"]
            anchors, _ = runner.old.anchor_input(cid)
            context = runner.old.condition_stroke_context(runner.old.build_context(raw["frames"], cameras),
                raw["frames"], cameras, anchors)
            span = legacy.fixture.previous.camera_span(cameras)
            base_path = runner.candidates(cid, "base", 0)[0]
            base = point.compose(base_path)
            adapter = adapter_type(base["points"], context, runner.old.POLICY)
            sources = []
            for arm in ARMS:
                path = None if arm == "base" else (
                    ROOT / patches[cid]["enabled_path"] if arm == "baseline_boundary"
                    else runner.candidates(cid, arm, 0)[1])
                candidate = None if path is None else point.open_candidate_view(path)
                segments = np.empty((0, 2, 3)) if candidate is None else candidate["segments"]
                if candidate is not None:
                    point.validate_candidate(base, candidate, segments, enabled=True)
                for fraction in sprint.SCALES:
                    reference = references[cid, arm, fraction]
                    if not np.array_equal(segments, np.asarray(reference["native_segments"]).reshape(-1, 2, 3)):
                        raise ValueError("Original native geometry changed: " + cid + "/" + arm)
                    if context["sha256"] != reference["graph"]["chain_context_sha256"]:
                        raise ValueError("Original shared RGB/stroke context changed")
                    call_started, error = time.perf_counter(), None
                    try:
                        graph = sprint.compact(adapter(segments, dict(voxel_size=span*fraction, origin=[0., 0., 0.])))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        graph = dict(state="error", reason=error, segments=[])
                    rows.append(dict(kind=kind, case_id=cid, variant=arm, fraction=fraction,
                        graph=graph, native_segments=segments, native_state=reference["native_state"],
                        error=error, reader_elapsed_seconds=time.perf_counter()-call_started,
                        reused_control=False, native_source_run_id=PARENT_ID))
                sources.append(dict(variant=arm, candidate_path=None if path is None else path.relative_to(ROOT).as_posix(),
                    candidate_sha256=None if path is None else legacy.digest(path),
                    saved_patch_id=patches[cid]["patch_id"] if arm == "baseline_boundary" else None,
                    native_geometry_equal_parent=True, original_base_bytes_preserved=True))
            cases_log.append(dict(case_id=cid, context_sha256=context["sha256"],
                base_path=base_path.relative_to(ROOT).as_posix(), candidates=sources,
                elapsed_seconds=time.perf_counter()-case_started))
            print("RASTER_BAND_NORMAL", cid, "rows", len(rows), "case_seconds",
                  round(time.perf_counter()-case_started, 3), flush=True)
    data = dict(run_id=run.name, adapter=ADAPTER, gt_read=False, rows=rows, cases=cases_log,
        reference_source=dict(run_id=PARENT_ID, normal_sha256=parent_sha,
            manifest_sha256=legacy.digest(PARENT / "manifest.json")),
        reader_policy=reader.POLICY, elapsed_seconds=time.perf_counter()-started, **FLAGS)
    boundary.validate_rows(data, ARMS)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and not any(p.startswith(".venv") for p in path.parts):
                opened.add(path)
    hashes = {p.relative_to(ROOT).as_posix(): legacy.digest(p) for p in sorted(opened)}
    changed = [name for name, sha in hashes.items() if name in parent_manifest["direct_hashes"]
               and parent_manifest["direct_hashes"][name] != sha]
    if changed:
        raise ValueError("Direct stable input/source changed: " + repr(changed))
    legacy.write(run / "normal.json", data)
    legacy.write(run / "manifest.json", dict(run_id=run.name, state="normal_complete", gt_read=False,
        adapter=ADAPTER, normal_sha256=legacy.digest(run / "normal.json"), direct_hashes=hashes,
        expected_rows=60, reference_source=data["reference_source"],
        source_policy="Direct inputs and imported sources hashed once; saved parent normal checked; no recursive ancestry audit",
        **FLAGS))
    print("RASTER_BAND_NORMAL_COMPLETE", run.name, len(rows), flush=True)


def evaluate(run):
    # The six-arm inventory and original full-case score function are unchanged.
    boundary.evaluate(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("normal", "evaluate"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}", args.run_id):
        parser.error("A single safe unused run-directory name is required")
    run, started = ROOT / ".runtime/experiments" / args.run_id, time.perf_counter()
    if args.stage == "normal" and run.exists():
        raise FileExistsError("Use a new run ID; preserve all previous attempts")
    try:
        normal(run) if args.stage == "normal" else evaluate(run)
    finally:
        log = run / (args.stage + "-work-log.json")
        if run.exists() and not log.exists():
            legacy.write(log, dict(stage=args.stage, elapsed_seconds=time.perf_counter()-started,
                error=None if sys.exc_info()[1] is None else repr(sys.exc_info()[1]), **FLAGS))


if __name__ == "__main__":
    main()
