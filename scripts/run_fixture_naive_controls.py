"""Frozen conventional depth controls, reversible patches and fresh-process replay."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np
import run_fixture_metric_readout as previous
from run_fixture_challenges import digest, now, physical_summary, read, receipt, write

ROOT, parent, analytic = previous.ROOT, previous.parent, previous.analytic
from creator_eval import common_readout_metric as reader  # noqa: E402
from creator_eval import fixture_naive_controls as controls  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY, support_views  # noqa: E402
from creator_eval.rgb_fixture_metric_readout import SCOPE, FixtureMetricReadout  # noqa: E402
from creator_recon.domain.point_patch import (  # noqa: E402
    compose,
    content_hash,
    load_snapshot,
    open_candidate_view,
    write_candidate_view,
    write_patch,
)

RUN_ID = "fixture-naive-controls-v1-20261006"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_naive_controls_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-06-fixture-naive-controls.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-06-fixture-naive-controls-audit.json"
block_truth = analytic.block_truth
validate_candidate = previous.validate_candidate


def inventory(config, repeat=None):
    repeats = config["repeats"] if repeat is None else [repeat]
    if any(type(value) is not int or value not in config["repeats"] for value in repeats):
        raise ValueError("Unknown fresh-process repeat")
    return [(r, cid, fraction, variant) for r in repeats for cid in config["case_ids"]
            for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"]]


def row_key(row):
    return tuple(row[key] for key in ("repeat", "case_id", "fraction", "variant"))


def validated_config():
    config = read(CONFIG)
    required = dict(run_id=RUN_ID, parent_run_id=parent.RUN_ID, comparison_run_id=previous.RUN_ID,
        case_ids=["r01", "r02", "r03"], voxel_camera_span_fractions=[.0025, .005],
        variants=["base", *controls.METHODS, "baseline", "cylinder_support"], repeats=[0, 1],
        expected_normal_rows=60, support_policy=POLICY, reader_overrides={}, naive_defaults=controls.DEFAULTS)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen naive-control protocol identity differs")
    return config


def source_inventory():
    return sorted(set(previous.source_inventory()) | {
        "scripts/run_fixture_naive_controls.py", "configs/fixture_naive_controls_v1.json",
        "tests/test_fixture_naive_controls.py", "tests/test_fixture_naive_fit.py",
        "tests/test_fixture_naive_fit_deepseek.py", "experiments/src/creator_eval/fixture_naive_controls.py",
        "experiments/src/creator_eval/line_controls.py"})


def comparison_rows():
    _, config = previous.checked()
    normal, _ = previous.verified_normal(config)
    return {(r["case_id"], r["fraction"], r["variant"]): r for r in normal["rows"]}


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or not before["development_prior_scores_seen"] or before["runtime"] != analytic.runtime()):
        raise ValueError("Invalid naive-control freeze or changed runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != reader.DEFAULTS
            or frozen["naive_defaults"] != controls.DEFAULTS or frozen["expected_normal_rows"] != len(inventory(config))):
        raise ValueError("Frozen policies or complete inventory differ")
    return before, config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve every prior naive-control attempt")
    config = validated_config()
    before, _ = previous.checked()
    hashes = {}
    for name, sha in before["hashes"].items():
        receipt(ROOT / name, hashes, sha)
    references = comparison_rows()
    for path in (previous.RUN / "pre.json", previous.RUN / "inference.json"):
        receipt(path, hashes)
    for row in references.values():
        receipt(previous.RUN / row["path"], hashes, row["sha256"])
    RUN.mkdir(parents=True)
    for cid in config["case_ids"]:
        source, target = parent.RUN / cid / "bundle/base", RUN / cid / "bundle/base"
        shutil.copytree(source, target)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                receipt(target / path.relative_to(source), hashes, receipt(path, hashes))
        validate_candidate(compose(source), compose(target), withdrawn=True)
    for name in source_inventory():
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_normal_rows=len(inventory(config)),
        reader_defaults=reader.DEFAULTS, naive_defaults=controls.DEFAULTS,
        physical_policy=read(previous.RUN / "evaluation-freeze.json")["physical_policy"],
        gt_read=False, development_prior_scores_seen=True, measurement_scope=SCOPE,
        fresh_process_repeats=True, independent_objects=False, g1_passed=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False, inference_existed=False,
        runtime=analytic.runtime(), development_prior_scores_seen=True, hashes=hashes))
    checked()
    print("FIXTURE_NAIVE_PREPARED", len(hashes), "receipts", flush=True)


def camera_span(cameras):
    extrinsics = np.asarray([c["world_to_camera_cv"] for c in cameras])
    matrices = np.concatenate((extrinsics, np.broadcast_to([0, 0, 0, 1], (len(cameras), 1, 4))), axis=1)
    centers = np.linalg.inv(matrices)[:, :3, 3]
    return float(np.linalg.norm(centers[:, None]-centers, axis=-1).max())


def persist_control(bundle, fit, repeat, base, outputs):
    """Persist a real non-suppressing patch, reopen both states, bind every byte."""
    if fit["method"] not in controls.METHODS or fit["state"] != "complete" or fit["gt_read"]:
        raise ValueError("Invalid normal naive fit")
    segments = np.asarray(fit["segments"], float).reshape((-1, 2, 3))
    snapshot, _ = load_snapshot(bundle / "base")
    view_ids = [snapshot["metadata"]["frames"][int(index)]["frame_id"]
                for index, count in fit["source_view_counts"].items() if count]
    line_ids = [f"line-{index:04d}" for index in range(len(segments))]
    evidence = [dict(decision="accept" if line_ids else "unresolved", line_ids=line_ids,
        suppression_range=None, view_ids=view_ids, source_sha256=fit["input_support_sha256"],
        note="Conventional fit on unchanged RGB-supported depth points; single observed span, "
             "without candidate endpoints or inferred gap splitting.")]
    name = fit["method"] + f"-r{repeat}"
    patch_dir = bundle / ("patch-" + name)
    patch_id = write_patch(patch_dir, bundle / "base", segments, np.empty((0, 3), np.uint32),
        selection_sha256=fit["input_support_sha256"],
        method=dict(id=fit["method"], version="fixture-naive-1", config_sha256=content_hash(controls.DEFAULTS),
                    seed=controls.DEFAULTS["seed"]), evidence=evidence,
        unresolved=[fit["reason"]] if fit["reason"] else [])
    candidates = {}
    for enabled, state in ((True, "enabled"), (False, "withdrawn")):
        path = bundle / f"{name}-{state}.json"
        write_candidate_view(path, bundle / "base", patch_dir, enabled=enabled)
        candidate = open_candidate_view(path)
        validate_candidate(base, candidate, withdrawn=not enabled)
        if enabled and not np.array_equal(candidate["segments"], segments):
            raise ValueError("Reopened naive geometry differs")
        receipt(path, outputs)
        candidates[state] = candidate
    files = sorted(patch_dir.iterdir())
    for path in files:
        receipt(path, outputs)
    return candidates["enabled"], dict(patch_id=patch_id,
        enabled_path=(bundle / f"{name}-enabled.json").relative_to(ROOT).as_posix(),
        withdrawn_path=(bundle / f"{name}-withdrawn.json").relative_to(ROOT).as_posix(),
        added_storage_bytes=sum(path.stat().st_size for path in files)
            + sum((bundle / f"{name}-{state}.json").stat().st_size for state in ("enabled", "withdrawn")),
        base_unchanged=True, withdrawal_verified=True, saved_reopen_verified=True)


def infer_repeat(repeat):
    sys.addaudithook(block_truth)
    _, config = checked()
    if type(repeat) is not int or repeat not in config["repeats"]:
        raise ValueError("Unknown fresh-process repeat")
    record_dir, repeat_path = RUN / f"records-r{repeat}", RUN / f"repeat-{repeat}.json"
    if record_dir.exists() or repeat_path.exists():
        raise FileExistsError("Preserve every prior fresh-process attempt")
    _, manifest, _ = parent.checked()
    references = comparison_rows()
    point_rows = {r["case_id"]: r for r in read(parent.RUN / "inference.json")["rows"]}
    record_dir.mkdir()
    started, rows, outputs = time.perf_counter(), [], {}
    for case in manifest["cases"]:
        cid = case["case_id"]
        original_folder, bundle = parent.RUN / cid, RUN / cid / "bundle"
        raw, cameras = read(ROOT / case["parent_record"]), read(original_folder / "rods.json")["cameras"]
        views, base = support_views(raw["frames"], cameras), compose(bundle / "base")
        original = point_rows[cid]
        if (base["base_snapshot_id"] != original["snapshot_id"] or len(base["points"]) != original["full_point_count"]
                or base["length_unit"] != "meter"
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + original["prediction_sha256"]):
            raise ValueError("Complete original snapshot/frame differs")
        adapter = FixtureMetricReadout(base["points"], views, config["support_policy"])
        fits = controls.fit_controls(base["points"], base["point_ids"], camera_span(cameras), adapter.mask)
        if [fit["method"] for fit in fits] != list(controls.METHODS):
            raise ValueError("Complete ordered naive fit inventory required")
        candidates, metadata = {"base": base}, {}
        for fit in fits:
            if fit["input_support_count"] != int(adapter.mask.sum()):
                raise ValueError("Naive fit changed the original support mask")
            candidate, persistence = persist_control(bundle, fit, repeat, base, outputs)
            path = record_dir / f"{cid}-{fit['method']}-fit.json"
            write(path, dict(**fit, persistence=persistence, repeat=repeat, case_id=cid))
            receipt(path, outputs)
            candidates[fit["method"]] = candidate
            metadata[fit["method"]] = dict(fit_record_path=path.relative_to(RUN).as_posix(),
                fit_record_sha256=digest(path), fit_elapsed_seconds=fit["elapsed_seconds"],
                input_support_sha256=fit["input_support_sha256"], outcome=fit["outcome"], **persistence)
        for variant in ("baseline", "cylinder_support"):
            candidates[variant] = open_candidate_view(original_folder / f"bundle/{variant}-enabled.json")
            validate_candidate(base, candidates[variant])
            validate_candidate(base, open_candidate_view(original_folder / f"bundle/{variant}-withdrawn.json"), withdrawn=True)
        for fraction in config["voxel_camera_span_fractions"]:
            for variant in config["variants"]:
                reference_variant = variant if variant not in controls.METHODS else "base"
                reference_row = references[(cid, fraction, reference_variant)]
                reference = read(previous.RUN / reference_row["path"])
                call = reference["call_policy"]
                if call != dict(voxel_size=camera_span(cameras)*fraction, origin=[0., 0., 0.]):
                    raise ValueError("Original camera-span scale or origin differs")
                call_started = time.perf_counter()
                output = adapter(candidates[variant]["segments"], call)
                elapsed = time.perf_counter()-call_started
                keys = ["full_input_point_count", "supported_base_point_count", "base_vote_histogram"]
                if variant not in controls.METHODS:
                    keys += ["full_curve_sample_count", "supported_curve_sample_count", "curve_vote_histogram"]
                if any(output[key] != reference[key] for key in keys):
                    raise ValueError("Original support or sampling identity differs")
                output.update(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                    reader_elapsed_seconds=elapsed, native_segments=candidates[variant]["segments"],
                    native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality",
                    control=metadata.get(variant),
                    raw_row_states=[dict(view_id=v["view_id"], **v["row_states"]) for v in views])
                path = record_dir / f"{cid}-{fraction}-{variant}.json"
                write(path, output)
                rows.append(dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                    path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                    previous_path=reference_row["path"], previous_sha256=reference_row["sha256"]))
                print("FIXTURE_NAIVE", repeat, cid, fraction, variant, output["state"], len(output["segments"]), flush=True)
    checked()
    if [row_key(row) for row in rows] != inventory(config, repeat):
        raise ValueError("Incomplete repeat inventory")
    write(repeat_path, dict(run_id=RUN_ID, repeat=repeat, state="complete", gt_read=False, rows=rows,
        outputs=outputs, pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started))


def repeat_normal(config, repeat):
    normal = read(RUN / f"repeat-{repeat}.json")
    if (normal["run_id"] != RUN_ID or normal["repeat"] != repeat or normal["state"] != "complete"
            or normal["gt_read"] or normal["pre_sha256"] != digest(RUN / "pre.json")
            or [row_key(row) for row in normal["rows"]] != inventory(config, repeat)):
        raise ValueError("Complete fresh-process normal inventory required")
    for name, sha in normal["outputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Persisted naive bundle or fit changed")
    return normal


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    started = time.perf_counter()
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, created_at_utc=now(), repeats=config["repeats"]))
    rows, outputs, repeats = [], {}, []
    for repeat in config["repeats"]:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_fixture_naive_controls.py"),
                        "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=1800)
        normal = repeat_normal(config, repeat)
        rows.extend(normal["rows"])
        outputs.update(normal["outputs"])
        path = RUN / f"repeat-{repeat}.json"
        receipt(path, outputs)
        repeats.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                            elapsed_seconds=normal["elapsed_seconds"]))
    checked()
    normal = dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows, outputs=outputs,
        repeats=repeats, fresh_process_repeats=True, independent_objects=False,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started)
    normal["fit_repeat_comparisons"] = fit_repeat_comparisons(normal, config)
    write(RUN / "inference.json", normal)
    verified_normal(config)


def fit_repeat_comparisons(normal, config):
    """Keep all six model-repeat results, including any nonidentical pairs."""
    results = []
    for cid in config["case_ids"]:
        for method in controls.METHODS:
            projections, identities = [], []
            for repeat in config["repeats"]:
                path = RUN / f"records-r{repeat}/{cid}-{method}-fit.json"
                name = path.relative_to(ROOT).as_posix()
                sha = digest(path)
                if normal["outputs"].get(name) != sha:
                    raise ValueError("Fit-repeat source is not bound by normal outputs")
                fit = read(path)
                if (fit["gt_read"] or fit["case_id"] != cid or fit["method"] != method
                        or fit["repeat"] != repeat or fit["config"] != controls.DEFAULTS or fit["readout_scale_used"]):
                    raise ValueError("Fit-repeat source identity differs")
                projections.append({key: fit[key] for key in ("segments", "model", "input_support_sha256", "config")})
                identities.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=sha))
            results.append(dict(case_id=cid, method=method, source_fits=identities,
                model_repeat_equal=projections[0] == projections[1],
                compared_fields=["segments", "model", "input_support_sha256", "config"]))
    return results


def verified_normal(config):
    """Verify all 60 rows, controls, bundle bytes and references before any GT."""
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or not normal["fresh_process_repeats"]
            or normal["independent_objects"] or [row_key(row) for row in normal["rows"]] != inventory(config)):
        raise ValueError("All 60 frozen normal rows must finish before truth access")
    for name, sha in normal["outputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Persisted naive bundle or fit changed")
    repeat_rows = [row for repeat in config["repeats"] for row in repeat_normal(config, repeat)["rows"]]
    if repeat_rows != normal["rows"]:
        raise ValueError("Fresh-process row references differ")
    references, graphs = comparison_rows(), []
    for row in normal["rows"]:
        variant = row["variant"] if row["variant"] not in controls.METHODS else "base"
        old_row = references[(row["case_id"], row["fraction"], variant)]
        if row["previous_path"] != old_row["path"] or row["previous_sha256"] != old_row["sha256"]:
            raise ValueError("Paired normal identity differs")
        path, old_path = RUN / row["path"], previous.RUN / old_row["path"]
        if digest(path) != row["sha256"] or digest(old_path) != old_row["sha256"]:
            raise ValueError("Paired normal graph changed")
        graph, old = read(path), read(old_path)
        if (graph["gt_read"] or any(graph[key] != row[key] for key in ("repeat", "case_id", "fraction", "variant"))
                or graph["call_policy"] != old["call_policy"]
                or graph["effective_policy"] != {**reader.DEFAULTS, **graph["call_policy"]}):
            raise ValueError("Normal identity or policy differs")
        if row["variant"] in controls.METHODS:
            control = graph["control"]
            fit_path = RUN / control["fit_record_path"]
            if (digest(fit_path) != control["fit_record_sha256"]
                    or normal["outputs"].get(fit_path.relative_to(ROOT).as_posix()) != control["fit_record_sha256"]):
                raise ValueError("Fit record is not bound by normal output receipts")
            fit = read(fit_path)
            if (fit["gt_read"] or fit["method"] != row["variant"] or fit["repeat"] != row["repeat"]
                    or fit["case_id"] != row["case_id"] or fit["segments"] != graph["native_segments"]
                    or fit["config"] != controls.DEFAULTS or fit["readout_scale_used"]):
                raise ValueError("Scale-independent native fit identity differs")
        graphs.append((graph, old))
    if normal["fit_repeat_comparisons"] != fit_repeat_comparisons(normal, config):
        raise ValueError("Saved fit-repeat comparisons differ from normal models")
    return normal, graphs


def physical_repeat_projection(graph):
    """Timings and receipt paths are not physical reproducibility criteria."""
    keys = ("state", "resolution_state", "reason", "segments", "native_segments", "native_state",
            "full_input_point_count", "supported_base_point_count", "full_curve_sample_count",
            "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram",
            "call_policy", "effective_policy", "raw_row_states")
    result = {key: graph.get(key) for key in keys}
    control = graph.get("control")
    result["control"] = None if control is None else {
        key: control[key] for key in ("input_support_sha256", "outcome", "base_unchanged",
                                      "withdrawal_verified", "saved_reopen_verified")}
    return result


def evaluation_payload():
    before, config = checked()
    normal, graphs = verified_normal(config)
    # All intended normal rows and saved artifacts are verified above.
    scene_id = read(parent.CONFIG)["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
    if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
        raise ValueError("Original truth identity differs")
    truth = {c["case_id"]: c for c in read(truth_path)["cases"]}
    rows, projected = [], {}
    for entry, (graph, old) in zip(normal["rows"], graphs):
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = (target["segments"] if "segments" in target else [target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        scores, errors = [], []
        for state, values in ((graph["state"], graph["segments"]), (graph["native_state"], graph["native_segments"]),
                              (old["state"], old["segments"])):
            score, error = None, None
            if state == "complete":
                try:
                    score = physical_summary(values, segments, gaps)
                except ValueError as exc:
                    error = str(exc)
            scores.append(score)
            errors.append(error)
        key = entry["case_id"], entry["fraction"], entry["variant"]
        projection = physical_repeat_projection(graph)
        repeated = None if entry["repeat"] == 0 else projection == projected[key]
        if entry["repeat"] == 0:
            projected[key] = projection
        rows.append(dict(**entry, state=graph["state"], reason=graph.get("reason"),
            output_segment_count=len(graph["segments"]), native_segment_count=len(graph["native_segments"]),
            physical=scores[0], native_physical=scores[1], previous_physical=scores[2],
            scoring_error=errors[0], native_scoring_error=errors[1], previous_scoring_error=errors[2],
            native_metric_scope=graph["native_metric_scope"], control=graph["control"],
            reader_elapsed_seconds=graph["reader_elapsed_seconds"], physical_equal_to_repeat_zero=repeated,
            full_input_point_count=graph["full_input_point_count"],
            supported_base_point_count=graph["supported_base_point_count"]))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"], rows=rows,
        fresh_process_repeats=normal["repeats"], independent_objects=False,
        fit_repeat_comparisons=normal["fit_repeat_comparisons"],
        fit_repeat_pairs=len(normal["fit_repeat_comparisons"]),
        fit_repeat_equal=sum(pair["model_repeat_equal"] for pair in normal["fit_repeat_comparisons"]),
        physical_repeat_pairs=30, physical_repeat_equal=sum(row["physical_equal_to_repeat_zero"] is True for row in rows),
        physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"], measurement_scope=SCOPE,
        development_prior_scores_seen=True, no_independent_holdout_claim=True, reader_qualified=False,
        g1_passed=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior naive-control evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_NAIVE_EVALUATED", len(result["rows"]), flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve prior naive-control audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Naive-control evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        parent_outputs_unchanged=True, same_region_for_all_variants=True,
        physical_repeat_pairs=public["physical_repeat_pairs"], physical_repeat_equal=public["physical_repeat_equal"],
        fit_repeat_pairs=public["fit_repeat_pairs"], fit_repeat_equal=public["fit_repeat_equal"],
        development_prior_scores_seen=True, independent_objects=False, reader_qualified=False, g1_passed=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("FIXTURE_NAIVE_POST_PASSED", flush=True)


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
