"""Reader-only guided-block-axis replay of all sixty frozen fixture/control candidates."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np
import run_fixture_independent_axis_readout as previous_sources
import run_fixture_naive_controls as previous
from run_fixture_challenges import digest, now, physical_summary, read, receipt, write

ROOT, parent, analytic = previous.ROOT, previous.parent, previous.analytic
from creator_eval import guided_block_axis_readout as reader  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY  # noqa: E402
from creator_eval.rgb_guided_block_axis_readout import (  # noqa: E402
    SCOPE,
    GuidedBlockAxisReadout,
    guided_support_views,
)
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "fixture-guided-block-axis-readout-v1-20261008"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_guided_block_axis_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-08-fixture-guided-block-axis-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-08-fixture-guided-block-axis-readout-audit.json"
block_truth = previous.block_truth
inventory, row_key = previous.inventory, previous.row_key
COUNT_KEYS = ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
              "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram")
NATIVE_KEYS = ("native_segments", "native_state", "native_metric_scope", "control", "raw_row_states")


def validated_config():
    config = read(CONFIG)
    required = dict(run_id=RUN_ID, parent_run_id=previous.RUN_ID, case_ids=["r01", "r02", "r03"],
        voxel_camera_span_fractions=[.0025, .005],
        variants=["base", *previous.controls.METHODS, "baseline", "cylinder_support"],
        repeats=[0, 1], expected_normal_rows=60, support_policy=POLICY, reader_overrides={},
        development_prior_scores_seen=True, blind_evaluation=False)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen guided-block-axis reader-only inventory or policy differs")
    return config


def source_inventory():
    return sorted(set(previous_sources.source_inventory()) | {
        "scripts/run_fixture_guided_block_axis_readout.py", "configs/fixture_guided_block_axis_readout_v1.json",
        "tests/test_fixture_guided_block_axis_readout.py", "experiments/src/creator_eval/guided_block_axis_readout.py",
        "experiments/src/creator_eval/rgb_guided_block_axis_readout.py", "tests/test_guided_block_axis_readout.py",
        "tests/test_guided_block_axis_readout_deepseek.py",
        "scripts/run_g1_object_guided_block_readout.py", "configs/g1_object_guided_block_readout_v1.json",
        "tests/test_g1_object_guided_block_readout.py",
        "experiments/src/creator_eval/single_axis_readout.py",
        "experiments/src/creator_eval/rgb_single_axis_readout.py", "tests/test_single_axis_readout.py"})


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or not before["reader_only_intervention"] or before["runtime"] != analytic.runtime()
            or before.get("development_prior_scores_seen") is not True or before.get("blind_evaluation") is not False):
        raise ValueError("Invalid guided-block-axis development freeze or changed runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen guided-block-axis receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != reader.DEFAULTS
            or frozen["expected_normal_rows"] != len(inventory(config))
            or frozen.get("development_prior_scores_seen") is not True or frozen.get("blind_evaluation") is not False):
        raise ValueError("Frozen guided-block-axis policy or inventory differs")
    return before, config


def parent_normal(config):
    before, old_config = previous.checked()
    if inventory(old_config) != inventory(config):
        raise ValueError("All sixty original repeat/case/scale/variant pairs required")
    normal, pairs = previous.verified_normal(old_config)
    graphs = {row_key(entry): graph for entry, (graph, _) in zip(normal["rows"], pairs)}
    return before, normal, graphs


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve every prior guided-block-axis attempt")
    config = validated_config()
    before, normal, _ = parent_normal(config)
    hashes = {}
    for name, sha in {**before["hashes"], **normal["outputs"]}.items():
        receipt(ROOT / name, hashes, sha)
    for path in (previous.RUN / "pre.json", previous.RUN / "inference.json"):
        receipt(path, hashes)
    for row in normal["rows"]:
        receipt(previous.RUN / row["path"], hashes, row["sha256"])
    names = source_inventory()
    if any(not (ROOT / name).is_file() for name in names):
        raise FileNotFoundError("All explicit reader and independent-test sources must exist before freeze")
    RUN.mkdir(parents=True)
    for name in names:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_normal_rows=len(inventory(config)),
        reader_defaults=reader.DEFAULTS, physical_policy=read(previous.RUN / "evaluation-freeze.json")["physical_policy"],
        gt_read=False, reader_only_intervention=True, native_geometry_unchanged=True,
        parent_inference_sha256=digest(previous.RUN / "inference.json"),
        development_prior_scores_seen=True, blind_evaluation=False, measurement_scope=SCOPE,
        fresh_process_repeats=True, independent_objects=False, reader_qualified=False, g1_passed=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False, inference_existed=False,
        reader_only_intervention=True, runtime=analytic.runtime(), development_prior_scores_seen=True, blind_evaluation=False,
        source_count=len(names), hashes=hashes))
    checked()
    print("FIXTURE_GUIDED_BLOCK_AXIS_PREPARED", len(names), "sources", len(hashes), "receipts", flush=True)


def candidate_paths(cid, variant, repeat):
    if cid not in ("r01", "r02", "r03") or repeat not in (0, 1):
        raise ValueError("Unknown frozen case or repeat")
    base = previous.RUN / cid / "bundle/base"
    if variant == "base":
        return base, None, None
    if variant in previous.controls.METHODS:
        bundle, name = previous.RUN / cid / "bundle", f"{variant}-r{repeat}"
    elif variant in ("baseline", "cylinder_support"):
        bundle, name = parent.RUN / cid / "bundle", variant
    else:
        raise ValueError("Unknown frozen variant")
    return base, bundle / f"{name}-enabled.json", bundle / f"{name}-withdrawn.json"


def validate_pair(graph, old):
    for key in (*COUNT_KEYS, *NATIVE_KEYS, "call_policy"):
        if graph[key] != old[key]:
            raise ValueError("Reader-only pairing changed: " + key)
    if graph["effective_policy"] != {**reader.DEFAULTS, **graph["call_policy"]}:
        raise ValueError("Guided-block-axis effective policy differs")


def infer_repeat(repeat):
    sys.addaudithook(block_truth)
    _, config = checked()
    if type(repeat) is not int or repeat not in config["repeats"]:
        raise ValueError("Unknown guided-block-axis repeat")
    inventory(config, repeat)
    record_dir, repeat_path = RUN / f"records-r{repeat}", RUN / f"repeat-{repeat}.json"
    if record_dir.exists() or repeat_path.exists():
        raise FileExistsError("Preserve every prior guided-block-axis repeat")
    _, old_normal, old_graphs = parent_normal(config)
    old_rows = {row_key(row): row for row in old_normal["rows"]}
    _, manifest, _ = parent.checked()
    point_rows = {row["case_id"]: row for row in read(parent.RUN / "inference.json")["rows"]}
    record_dir.mkdir()
    rows, started = [], time.perf_counter()
    for case in manifest["cases"]:
        cid = case["case_id"]
        raw = read(ROOT / case["parent_record"])
        cameras = read(parent.RUN / cid / "rods.json")["cameras"]
        views = guided_support_views(raw["frames"], cameras)
        base = compose(candidate_paths(cid, "base", repeat)[0])
        original = point_rows[cid]
        if (base["base_snapshot_id"] != original["snapshot_id"] or len(base["points"]) != original["full_point_count"]
                or base["length_unit"] != "meter"
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + original["prediction_sha256"]):
            raise ValueError("Original full base snapshot/prediction frame differs")
        candidates = {"base": base}
        for variant in config["variants"][1:]:
            _, enabled, withdrawn = candidate_paths(cid, variant, repeat)
            candidates[variant] = open_candidate_view(enabled)
            previous.validate_candidate(base, candidates[variant])
            previous.validate_candidate(base, open_candidate_view(withdrawn), withdrawn=True)
        adapter = GuidedBlockAxisReadout(base["points"], views, config["support_policy"])
        for fraction in config["voxel_camera_span_fractions"]:
            for variant in config["variants"]:
                key = repeat, cid, fraction, variant
                old, reference = old_graphs[key], old_rows[key]
                native = np.asarray(old["native_segments"], float).reshape(-1, 2, 3)
                if not np.array_equal(candidates[variant]["segments"], native):
                    raise ValueError("Reopened original geometry differs from its paired native record")
                call = old["call_policy"]
                if call != dict(voxel_size=previous.camera_span(cameras)*fraction, origin=[0., 0., 0.]):
                    raise ValueError("Original camera-span scale/origin changed")
                call_started = time.perf_counter()
                output = adapter(candidates[variant]["segments"], call)
                output.update({name: old[name] for name in NATIVE_KEYS})
                output.update(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                    reader_elapsed_seconds=time.perf_counter()-call_started, reader_only_intervention=True,
                    development_prior_scores_seen=True, blind_evaluation=False)
                validate_pair(output, old)
                path = record_dir / f"{cid}-{fraction}-{variant}.json"
                write(path, output)
                rows.append(dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                    path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                    previous_path=reference["path"], previous_sha256=reference["sha256"]))
                print("FIXTURE_GUIDED_BLOCK_AXIS", repeat, cid, fraction, variant, output["state"], len(output["segments"]), flush=True)
    checked()
    if [row_key(row) for row in rows] != inventory(config, repeat):
        raise ValueError("Incomplete guided-block-axis repeat inventory")
    write(repeat_path, dict(run_id=RUN_ID, repeat=repeat, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started,
        development_prior_scores_seen=True, blind_evaluation=False))


def repeat_normal(config, repeat):
    normal = read(RUN / f"repeat-{repeat}.json")
    if (normal["run_id"] != RUN_ID or normal["repeat"] != repeat or normal["state"] != "complete"
            or normal["gt_read"] or normal["pre_sha256"] != digest(RUN / "pre.json")
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config, repeat)):
        raise ValueError("Complete guided-block-axis repeat inventory required")
    return normal


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    started = time.perf_counter()
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, created_at_utc=now(), repeats=config["repeats"],
        development_prior_scores_seen=True, blind_evaluation=False))
    rows, repeats = [], []
    for repeat in config["repeats"]:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_fixture_guided_block_axis_readout.py"),
            "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=1800)
        normal = repeat_normal(config, repeat)
        rows.extend(normal["rows"])
        path = RUN / f"repeat-{repeat}.json"
        repeats.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
            elapsed_seconds=normal["elapsed_seconds"]))
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        repeats=repeats, fresh_process_repeats=True, reader_only_intervention=True,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started,
        development_prior_scores_seen=True, blind_evaluation=False))
    verified_normal(config)


def verified_normal(config):
    """All sixty fresh graphs and their original bundles are verified before GT."""
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or not normal["fresh_process_repeats"]
            or not normal["reader_only_intervention"]
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config)):
        raise ValueError("All 60 guided-block-axis normal rows required before truth")
    if [entry["repeat"] for entry in normal["repeats"]] != config["repeats"]:
        raise ValueError("Missing fresh-process repeat receipts")
    repeated = []
    for entry in normal["repeats"]:
        path = RUN / f"repeat-{entry['repeat']}.json"
        if entry["path"] != path.relative_to(RUN).as_posix() or digest(path) != entry["sha256"]:
            raise ValueError("Guided-block-axis repeat receipt changed")
        repeated.extend(repeat_normal(config, entry["repeat"])["rows"])
    if repeated != normal["rows"]:
        raise ValueError("Guided-block-axis repeat row references differ")
    _, old_normal, old_graphs = parent_normal(config)
    graphs = []
    for row, reference in zip(normal["rows"], old_normal["rows"]):
        if row_key(row) != row_key(reference) or row["previous_path"] != reference["path"] or row["previous_sha256"] != reference["sha256"]:
            raise ValueError("Exact old/new normal pairing differs")
        expected = f"records-r{row['repeat']}/{row['case_id']}-{row['fraction']}-{row['variant']}.json"
        if row["path"] != expected or digest(RUN / row["path"]) != row["sha256"]:
            raise ValueError("Guided-block-axis normal graph changed")
        graph, old = read(RUN / row["path"]), old_graphs[row_key(row)]
        if (graph["gt_read"] or not graph["reader_only_intervention"]
                or graph.get("development_prior_scores_seen") is not True or graph.get("blind_evaluation") is not False
                or any(graph[key] != row[key] for key in ("repeat", "case_id", "fraction", "variant"))):
            raise ValueError("Guided-block-axis normal identity differs")
        validate_pair(graph, old)
        graphs.append((graph, old))
    return normal, graphs


def evaluation_payload():
    before, config = checked()
    normal, graphs = verified_normal(config)
    scene_id = read(parent.CONFIG)["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
    if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
        raise ValueError("Original truth identity differs")
    truth = {case["case_id"]: case for case in read(truth_path)["cases"]}
    rows, projections = [], {}
    for entry, (graph, old) in zip(normal["rows"], graphs):
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        scores, errors = [], []
        for state, values in ((graph["state"], graph["segments"]), (old["state"], old["segments"]),
                              (graph["native_state"], graph["native_segments"]), (old["native_state"], old["native_segments"])):
            score, error = None, None
            if state == "complete":
                try:
                    score = physical_summary(values, segments, gaps)
                except ValueError as exc:
                    error = str(exc)
            scores.append(score)
            errors.append(error)
        if scores[2] != scores[3] or errors[2] != errors[3]:
            raise ValueError("Unchanged native geometry physical recomputation differs")
        key = entry["case_id"], entry["fraction"], entry["variant"]
        projection = previous.physical_repeat_projection(graph)
        repeated = None if entry["repeat"] == 0 else projection == projections[key]
        if entry["repeat"] == 0:
            projections[key] = projection
        rows.append(dict(**entry, state=graph["state"], reason=graph.get("reason"),
            output_segment_count=len(graph["segments"]), previous_state=old["state"],
            previous_output_segment_count=len(old["segments"]), native_segment_count=len(graph["native_segments"]),
            physical=scores[0], previous_physical=scores[1], native_physical=scores[2],
            previous_native_physical=scores[3], native_physical_unchanged=True,
            scoring_error=errors[0], previous_scoring_error=errors[1], native_scoring_error=errors[2],
            native_metric_scope=graph["native_metric_scope"], control=graph["control"],
            reader_elapsed_seconds=graph["reader_elapsed_seconds"], physical_equal_to_repeat_zero=repeated,
            **{key: graph[key] for key in COUNT_KEYS}))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), parent_inference_sha256=digest(previous.RUN / "inference.json"),
        truth_sha256=digest(truth_path), unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"],
        rows=rows, base_rows=[row for row in rows if row["variant"] == "base"],
        fresh_process_repeats=normal["repeats"], physical_repeat_pairs=30,
        physical_repeat_equal=sum(row["physical_equal_to_repeat_zero"] is True for row in rows),
        native_physical_recomputed_from_original_geometry=True, native_unchanged_rows=len(rows),
        reader_only_intervention=True, reader_qualified=False, independent_objects=False, g1_passed=False,
        physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"], measurement_scope=SCOPE,
        development_prior_scores_seen=True, blind_evaluation=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior guided-block-axis evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_GUIDED_BLOCK_AXIS_EVALUATED", len(result["rows"]), flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve prior guided-block-axis post audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Guided-block-axis exact evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        reader_only_intervention=True, original_bundles_unchanged=True, measurement_counts_unchanged=True,
        native_unchanged_rows=public["native_unchanged_rows"], physical_repeat_pairs=30,
        physical_repeat_equal=public["physical_repeat_equal"], reader_qualified=False, g1_passed=False,
        independent_objects=False, measurement_scope=SCOPE,
        development_prior_scores_seen=True, blind_evaluation=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("FIXTURE_GUIDED_BLOCK_AXIS_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "infer-repeat", "evaluate", "post"))
    parser.add_argument("repeat", type=int, nargs="?")
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "infer-repeat":
        infer_repeat(args.repeat)
    elif args.repeat is not None:
        parser.error("repeat is only valid for infer-repeat")
    else:
        globals()[args.stage]()
