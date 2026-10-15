"""Five real foreground-axis patches, ten fresh readouts, seventy reused controls."""

import argparse
import copy
import os
import re
import shutil
import sys
import time
from pathlib import Path

import foreground_consensus_runner as legacy
import numpy as np
import run_g1_object_point_patch as point
import run_readout_sprint as sprint

ROOT = legacy.ROOT
PARENT_ID = "selected-boundary-patch-sprint-20261015-001"
PARENT = ROOT / ".runtime/experiments" / PARENT_ID
ADAPTER = "creator_eval.rgb_bracketed_occupancy_readout:BracketedOccupancyReadout"
NEW_ARM = "baseline_foreground_axis"
PARENT_ARMS = (*sprint.ARMS, "baseline_boundary", "baseline_selected_boundary")
ARMS = (*PARENT_ARMS, NEW_ARM)
FLAGS = dict(sprint.FLAGS, old_control_readouts_recomputed=False, fresh_patch_count=5,
             reused_control_rows=70, fresh_readout_rows=10, fresh_depth_processes=0)


def inventory(arms):
    return {(kind, cid, arm, fraction) for kind, ids in legacy.CASES.items()
            for cid in ids for arm in arms for fraction in sprint.SCALES}


def validate_rows(data, arms):
    expected = inventory(arms)
    keys = [(r["kind"], r["case_id"], r["variant"], r["fraction"]) for r in data["rows"]]
    if len(keys) != len(expected) or set(keys) != expected or data["gt_read"] is not False:
        raise ValueError("Complete original cases, methods and scales required before GT")


def persist(bundle, base, refinement, context, original_native, record, policy):
    segments = np.asarray(refinement["segments"], float).reshape(-1, 2, 3)
    if refinement["state"] not in ("accepted", "rejected", "complete"):
        raise ValueError("Boundary method failed; preserve the partial normal run")
    source_sha = legacy.digest(record)
    selection_sha = point.canonical_hash(dict(context_sha256=context["sha256"],
        original_native_sha256=point.canonical_hash(original_native),
        anchors=context["association"]["anchors"]))
    evidence = [dict(decision="accept", line_ids=[f"line-{i:04d}" for i in range(len(segments))],
        suppression_range=None, view_ids=[v["view_id"] for v in context["views"]],
        source_sha256=source_sha,
        note="Measured foreground-axis refinement of native baseline; no point suppression or GT")] if len(segments) else []
    patch_dir = bundle / ("patch-" + NEW_ARM)
    patch_id = point.write_patch(patch_dir, bundle / "base", segments, np.empty((0, 3), np.uint32),
        selection_sha256=selection_sha,
        method=dict(id=NEW_ARM, version="foreground-axis-patch-1", config_sha256=point.canonical_hash(policy), seed=0),
        evidence=evidence, unresolved=list(refinement["rejection_reasons"]))
    paths = {}
    for enabled, state in ((True, "enabled"), (False, "withdrawn")):
        path = bundle / f"{NEW_ARM}-{state}.json"
        point.write_candidate_view(path, bundle / "base", patch_dir, enabled=enabled)
        point.validate_candidate(base, point.open_candidate_view(path), segments, enabled=enabled)
        paths[state + "_path"] = path.relative_to(ROOT).as_posix()
    files = list(patch_dir.iterdir()) + [ROOT / path for path in paths.values()]
    return dict(method=NEW_ARM, patch_id=patch_id, native_segments=segments,
        segment_count=len(segments), source_state=refinement["state"], source_sha256=source_sha,
        selection_sha256=selection_sha, method_version="foreground-axis-patch-1", **paths,
        no_suppression=True, base_bytes_preserved=True, withdrawn_bytes_equal=True,
        saved_views_reopened=True, added_storage_bytes=sum(path.stat().st_size for path in files),
        copied_base_bytes=sum(path.stat().st_size for path in (bundle / "base").iterdir()))


def normal(run):
    run.mkdir(parents=True, exist_ok=False)
    started, opened = time.perf_counter(), set()
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
    from creator_eval import rod_foreground_axis_patch as method
    parent_manifest = legacy.read(PARENT / "manifest.json")
    parent = legacy.read(PARENT / "normal.json")
    parent_sha = legacy.digest(PARENT / "normal.json")
    if (parent_manifest["normal_sha256"] != parent_sha or parent_manifest["state"] != "normal_complete"
            or parent["adapter"] != ADAPTER):
        raise ValueError("Unchanged complete bracketed parent normal required")
    validate_rows(parent, PARENT_ARMS)
    rows = copy.deepcopy(parent["rows"])
    for row in rows:
        row.update(reused_control=True, source_run_id=PARENT_ID, source_normal_sha256=parent_sha)
    patches = []
    adapter_type = sprint.load_adapter(ADAPTER)
    for kind, case_ids in legacy.CASES.items():
        runner = legacy.Runner(kind)
        cases = legacy.read(runner.point.INPUTS / "manifest.json")["cases"]
        if [c["case_id"] for c in cases] != case_ids:
            raise ValueError("Original full five-case inventory required")
        for case in cases:
            case_started, cid = time.perf_counter(), case["case_id"]
            folder = run / cid
            folder.mkdir()
            opened.update((ROOT / f["rgb"]).resolve() for f in case["frames"])
            raw = legacy.read(ROOT / case["parent_record"])
            cameras = legacy.read(runner.point.RUN / cid / "rods.json")["cameras"]
            anchors, _ = runner.old.anchor_input(cid)
            context = runner.old.condition_stroke_context(runner.old.build_context(raw["frames"], cameras),
                raw["frames"], cameras, anchors)
            base_path, native_path, _ = runner.candidates(cid, "baseline", 0)
            base = point.compose(base_path)
            native = point.open_candidate_view(native_path)["segments"]
            references = [r for r in parent["rows"] if r["case_id"] == cid and r["variant"] == "baseline"]
            if any(not np.array_equal(native, np.asarray(r["native_segments"]).reshape(-1, 2, 3)) for r in references):
                raise ValueError("Original baseline geometry changed")
            refinement_started = time.perf_counter()
            refinement = method.refine_boundary(native, context)
            refinement_seconds = time.perf_counter() - refinement_started
            record = folder / "refinement.json"
            legacy.write(record, dict(case_id=cid, gt_read=False, original_native_segments=native,
                original_candidate_path=native_path.relative_to(ROOT).as_posix(),
                original_candidate_sha256=legacy.digest(native_path), context_sha256=context["sha256"],
                policy=method.POLICY, refinement=refinement, elapsed_seconds=refinement_seconds))
            bundle = folder / "bundle"
            shutil.copytree(base_path, bundle / "base")
            patch = persist(bundle, base, refinement, context, native, record, method.POLICY)
            patch.update(case_id=cid, refinement_path=record.relative_to(ROOT).as_posix(),
                refinement_elapsed_seconds=refinement_seconds,
                original_base_path=base_path.relative_to(ROOT).as_posix())
            patches.append(patch)
            segments = point.open_candidate_view(ROOT / patch["enabled_path"])["segments"]
            adapter = adapter_type(base["points"], context, runner.old.POLICY)
            span = legacy.fixture.previous.camera_span(cameras)
            for fraction in sprint.SCALES:
                call_started, error = time.perf_counter(), None
                try:
                    graph = sprint.compact(adapter(segments, dict(voxel_size=span*fraction, origin=[0., 0., 0.])))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    graph = dict(state="error", reason=error, segments=[])
                rows.append(dict(kind=kind, case_id=cid, variant=NEW_ARM, fraction=fraction, graph=graph,
                    native_segments=segments, native_state="complete", native_source_state=refinement["state"],
                    error=error, reader_elapsed_seconds=time.perf_counter()-call_started,
                    reused_control=False, saved_patch_id=patch["patch_id"]))
            print("FOREGROUND_AXIS_PATCH_NORMAL", cid, "rows", len(rows), "case_seconds",
                  round(time.perf_counter()-case_started, 3), flush=True)
    data = dict(run_id=run.name, adapter=ADAPTER, gt_read=False, rows=rows, patches=patches,
        controls_source=dict(run_id=PARENT_ID, normal_sha256=parent_sha,
            manifest_sha256=legacy.digest(PARENT / "manifest.json"), control_rows=70),
        method_policy=method.POLICY, elapsed_seconds=time.perf_counter()-started, **FLAGS)
    validate_rows(data, ARMS)
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
        raise ValueError("Reused-control direct input/source changed: " + repr(changed))
    legacy.write(run / "normal.json", data)
    legacy.write(run / "manifest.json", dict(run_id=run.name, state="normal_complete", gt_read=False,
        adapter=ADAPTER, normal_sha256=legacy.digest(run / "normal.json"), direct_hashes=hashes,
        expected_rows=80, controls_source=data["controls_source"],
        source_policy="Direct inputs and imported project sources hashed once; reused parent normal hash checked; no recursive ancestry audit",
        **FLAGS))
    print("FOREGROUND_AXIS_PATCH_NORMAL_COMPLETE", run.name, len(rows), flush=True)


def evaluate(run):
    started = time.perf_counter()
    if (run / "evaluation.json").exists():
        raise FileExistsError("Preserve previous scores")
    manifest, data = legacy.read(run / "manifest.json"), legacy.read(run / "normal.json")
    if manifest["state"] != "normal_complete" or manifest["normal_sha256"] != legacy.digest(run / "normal.json"):
        raise ValueError("Complete unchanged normal required before GT")
    validate_rows(data, ARMS)
    truths, truth_hashes = {}, {}
    for kind, ids in legacy.CASES.items():
        runner = legacy.Runner(kind)
        scene = legacy.read(runner.point.CONFIG)["scene_run_id"]
        path = ROOT / "data/eval_gt" / scene / "manifest.json"
        expected = legacy.read(ROOT / ".runtime/experiments" / scene / "prepared.json")["truth_sha256"]
        if legacy.digest(path) != expected:
            raise ValueError("Original full GT bytes changed")
        truth = legacy.read(path)
        if [case["case_id"] for case in truth["cases"]] != ids:
            raise ValueError("Full original five-case truth inventory required")
        truth_hashes[path.relative_to(ROOT).as_posix()] = expected
        truths.update({c["case_id"]: c["declared"] for c in truth["cases"]})
    for row in data["rows"]:
        declared = truths[row["case_id"]]
        target = declared["target"]
        truth = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        row["physical"], row["scoring_error"] = legacy.objects.score_geometry(row["graph"]["state"], row["graph"]["segments"], truth, gaps)
        row["native_physical"], row["native_scoring_error"] = legacy.objects.score_geometry(row["native_state"], row["native_segments"], truth, gaps)
    legacy.write(run / "evaluation.json", dict(**data, normal_sha256=manifest["normal_sha256"],
        evaluation_gt_read=True, truth_hashes=truth_hashes, evaluation_elapsed_seconds=time.perf_counter()-started))
    print("FOREGROUND_AXIS_PATCH_EVALUATED", run.name, len(data["rows"]), "development_only_not_G1", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("normal", "evaluate"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}", args.run_id):
        parser.error("A single safe unused run-directory name is required")
    run = ROOT / ".runtime/experiments" / args.run_id
    if args.stage == "normal" and run.exists():
        raise FileExistsError("Use a new run ID; previous attempts are preserved")
    started = time.perf_counter()
    try:
        normal(run) if args.stage == "normal" else evaluate(run)
    finally:
        log = run / (args.stage + "-work-log.json")
        if run.exists() and not log.exists():
            legacy.write(log, dict(stage=args.stage, elapsed_seconds=time.perf_counter()-started,
                error=None if sys.exc_info()[1] is None else repr(sys.exc_info()[1]), **FLAGS))


if __name__ == "__main__":
    main()
