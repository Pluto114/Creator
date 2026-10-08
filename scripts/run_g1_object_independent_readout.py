"""Paired two-reader/two-process readout of immutable new-object point patches."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import run_g1_object_point_patch as parent
import run_g1_object_support_readout as previous_sources
from run_fixture_challenges import physical_summary

ROOT = parent.ROOT
read, write, digest, receipt, block_truth = parent.read, parent.write, parent.digest, parent.receipt, parent.block_truth
from creator_eval import independent_axis_readout as independent  # noqa: E402
from creator_eval import single_axis_readout as single  # noqa: E402
from creator_eval.fixture_physical_metrics import POLICY as PHYSICAL_POLICY  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY, support_views  # noqa: E402
from creator_eval.rgb_independent_axis_readout import IndependentAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "g1-object-independent-readout-v1-20261008"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/g1_object_independent_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-08-g1-object-independent-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-08-g1-object-independent-readout-audit.json"
ADAPTERS = dict(single_axis=SingleAxisReadout, independent_axis=IndependentAxisReadout)
COUNT_KEYS = ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
              "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram")
KEYS = ("repeat", "case_id", "fraction", "variant", "reader")
OWN_SOURCES = {"scripts/run_g1_object_independent_readout.py", "configs/g1_object_independent_readout_v1.json", "tests/test_g1_object_independent_readout.py",
    "experiments/src/creator_eval/single_axis_readout.py", "experiments/src/creator_eval/rgb_single_axis_readout.py",
    "tests/test_single_axis_readout.py", "tests/test_single_axis_readout_deepseek.py",
    "experiments/src/creator_eval/independent_axis_readout.py", "experiments/src/creator_eval/rgb_independent_axis_readout.py",
    "tests/test_independent_axis_readout.py", "tests/test_independent_axis_readout_reused_deepseek.py"}


def now():
    return datetime.now(timezone.utc).isoformat()


def defaults():
    return dict(single_axis=single.DEFAULTS, independent_axis=independent.DEFAULTS)


def inventory(config, repeat=None):
    repeats = config["repeats"] if repeat is None else [repeat]
    if any(type(value) is not int or value not in config["repeats"] for value in repeats):
        raise ValueError("Unknown reader-only repeat")
    return [(r, cid, fraction, variant, reader) for r in repeats for cid in config["case_ids"]
            for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"] for reader in config["readers"]]


def row_key(row):
    return tuple(row[key] for key in KEYS)


def validated_config():
    config = read(CONFIG)
    expected = dict(run_id=RUN_ID, parent_run_id=parent.RUN_ID,
        parent_pre_sha256="807d7fadedc47f39e30db3cb78dfeb8934b8e706119c5e342afbccf5bd2a0af1",
        parent_inference_sha256="74bc654b83ed830c9ab53f7c9e4c194c50d7b085152c95c010eb0c9d05eea7b3",
        parent_pre_receipts=674, parent_output_receipts=62, scene_run_id=parent.parent.SCENE_ID,
        case_ids=["chair01", "aframe01"], variants=["base", *parent.controls.METHODS, "baseline", "cylinder_support"],
        voxel_camera_span_fractions=[.0025, .005], readers=["single_axis", "independent_axis"], repeats=[0, 1],
        expected_normal_rows=80, support_policy=POLICY, reader_overrides={},
        development_prior_scores_seen=True, blind_evaluation=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen object readout inventory/policy differs")
    return config


def source_inventory(point_pre):
    return sorted(set(previous_sources.source_inventory(point_pre)) | OWN_SOURCES)


def parent_normal(config):
    if (digest(parent.RUN / "pre.json") != config["parent_pre_sha256"]
            or digest(parent.RUN / "inference.json") != config["parent_inference_sha256"]):
        raise ValueError("Original point/patch parent receipt differs")
    before, manifest, _, normal = parent.normal_records()
    if (len(before["hashes"]) != config["parent_pre_receipts"]
            or len(normal["outputs"]) != config["parent_output_receipts"]
            or [row["case_id"] for row in normal["rows"]] != config["case_ids"]):
        raise ValueError("Complete frozen point/patch parent inventory required")
    return before, manifest, normal


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or not before["reader_only_intervention"] or before["runtime"] != parent.runtime()
            or before.get("development_prior_scores_seen") is not True or before.get("blind_evaluation") is not False):
        raise ValueError("Invalid object readout freeze/runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen object readout receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != defaults()
            or frozen["expected_normal_rows"] != len(inventory(config)) or frozen["physical_policy"] != PHYSICAL_POLICY):
        raise ValueError("Frozen readers, evaluation policy or inventory differs")
    return before, config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve previous object readout attempt")
    config = validated_config()
    before, _, normal = parent_normal(config)
    hashes = {}
    for name, sha in {**before["hashes"], **normal["outputs"]}.items():
        receipt(ROOT / name, hashes, sha)
    for path in (parent.RUN / "pre.json", parent.RUN / "inference.json"):
        receipt(path, hashes)
    names = source_inventory(before)
    if any(not (ROOT / name).is_file() for name in names):
        raise FileNotFoundError("All explicit readout sources and tests must exist before freeze")
    RUN.mkdir(parents=True)
    for name in names:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, reader_defaults=defaults(),
        expected_normal_rows=len(inventory(config)), physical_policy=PHYSICAL_POLICY,
        gt_read=False, reader_only_intervention=True, native_geometry_unchanged=True,
        fresh_reader_processes=2, end_to_end_repeat_performed=False, independent_real_capture=False,
        reader_qualified=False, g1_passed=False, development_prior_scores_seen=True, blind_evaluation=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), hashes=hashes, source_count=len(names),
        runtime=parent.runtime(), gt_read=False, inference_existed=False, reader_only_intervention=True,
        development_prior_scores_seen=True, blind_evaluation=False))
    checked()
    print("G1_OBJECT_INDEPENDENT_READOUT_PREPARED", len(names), "sources", len(hashes), "receipts", flush=True)


def validate_pair(first, second):
    for key in (*COUNT_KEYS, "call_policy", "native_segments", "native_source_state", "native_patch_id"):
        if first[key] != second[key]:
            raise ValueError("Paired readout changed original sampling/native identity: " + key)


def point_rows(normal):
    return {row["case_id"]: row for row in normal["rows"]}


def native_metadata(row, variant):
    if variant == "base":
        return dict(native_segments=[], native_source_state="unchanged_base", native_patch_id=None,
                    native_outcome="no_added_geometry", native_reason=[])
    patch = next(item for item in row["patches"] if item["method"] == variant)
    manifest = read(parent.RUN / row["case_id"] / f"bundle/patch-{variant}/manifest.json")
    if manifest["content_id"] != patch["patch_id"]:
        raise ValueError("Original native patch manifest identity differs")
    return dict(native_segments=patch["native_segments"], native_source_state=patch["source_state"],
                native_patch_id=patch["patch_id"], native_outcome=patch["outcome"], native_reason=manifest["metadata"]["unresolved"])


def infer_repeat(repeat):
    sys.addaudithook(block_truth)
    _, config = checked()
    if type(repeat) is not int or repeat not in config["repeats"]:
        raise ValueError("Unknown reader-only repeat")
    record_dir, repeat_path = RUN / f"records-r{repeat}", RUN / f"repeat-{repeat}.json"
    if record_dir.exists() or repeat_path.exists():
        raise FileExistsError("Preserve previous object reader repeat")
    _, manifest, original = parent_normal(config)
    originals = point_rows(original)
    record_dir.mkdir()
    rows, started = [], time.perf_counter()
    for case in manifest["cases"]:
        cid, folder, bound = case["case_id"], parent.RUN / case["case_id"], originals[case["case_id"]]
        raw = read(ROOT / case["parent_record"])
        cameras = read(folder / "rods.json")["cameras"]
        support = read(folder / "support.json")
        if (canonical_hash(cameras) != bound["camera_export_sha256"]
                or canonical_hash(cameras) != support["camera_sha256"]
                or canonical_hash(raw["frames"]) != support["raw_frame_sha256"]):
            raise ValueError("Exact exported camera/raw RGB support identity differs")
        views, base = support_views(raw["frames"], cameras), compose(folder / "bundle/base")
        if (base["base_snapshot_id"] != bound["snapshot_id"] or len(base["points"]) != bound["full_point_count"]
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + bound["prediction_sha256"]
                or base["length_unit"] != "meter"):
            raise ValueError("Complete original base snapshot identity differs")
        adapters = {name: ADAPTERS[name](base["points"], views, config["support_policy"]) for name in config["readers"]}
        first, second = (adapters[name] for name in config["readers"])
        if (not np.array_equal(first.mask, second.mask) or not np.array_equal(first.votes, second.votes)
                or len(first.points) != bound["supported_base_point_count"]
                or np.bincount(first.votes, minlength=len(views)+1).tolist() != support["vote_histogram"]):
            raise ValueError("Original complete base RGB mask/votes changed")
        span = parent.camera_span(np.asarray([camera["world_to_camera_cv"] for camera in cameras]))
        if span != bound["camera_span_m"]:
            raise ValueError("Original exported camera span changed")
        candidates, native = {"base": base}, {}
        for variant in config["variants"]:
            native[variant] = native_metadata(bound, variant)
            if variant != "base":
                candidates[variant] = open_candidate_view(folder / f"bundle/{variant}-enabled.json")
                segments = np.asarray(native[variant]["native_segments"], float).reshape(-1, 2, 3)
                parent.validate_candidate(base, candidates[variant], segments, enabled=True)
                parent.validate_candidate(base, open_candidate_view(folder / f"bundle/{variant}-withdrawn.json"), segments, enabled=False)
        for fraction in config["voxel_camera_span_fractions"]:
            call = dict(voxel_size=span*fraction, origin=[0., 0., 0.])
            for variant in config["variants"]:
                pair = []
                for name in config["readers"]:
                    call_started = time.perf_counter()
                    output = adapters[name](candidates[variant]["segments"], call)
                    output.update(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        reader_elapsed_seconds=time.perf_counter()-call_started, reader_only_intervention=True,
                        native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality",
                        base_snapshot_id=base["base_snapshot_id"], source_support_sha256=bound["support_sha256"],
                        raw_row_states=[dict(view_id=view["view_id"], **view["row_states"]) for view in views], **native[variant])
                    if output["effective_policy"] != {**defaults()[name], **call}:
                        raise ValueError("Unchanged reader defaults/call policy differ")
                    pair.append(output)
                    if len(pair) == 2:
                        validate_pair(*pair)
                    path = record_dir / f"{cid}-{fraction}-{variant}-{name}.json"
                    write(path, output)
                    rows.append(dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
                    print("G1_OBJECT_INDEPENDENT_READOUT", repeat, cid, fraction, variant, name, output["state"], len(output["segments"]), flush=True)
    checked()
    if [row_key(row) for row in rows] != inventory(config, repeat):
        raise ValueError("Complete forty-row reader repeat required")
    write(repeat_path, dict(run_id=RUN_ID, repeat=repeat, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), reader_only_intervention=True, elapsed_seconds=time.perf_counter()-started))


def repeat_normal(config, repeat):
    normal = read(RUN / f"repeat-{repeat}.json")
    if (normal["run_id"] != RUN_ID or normal["repeat"] != repeat or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or not normal["reader_only_intervention"]
            or [row_key(row) for row in normal["rows"]] != inventory(config, repeat)):
        raise ValueError("Complete forty-row reader repeat required")
    return normal


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    if (RUN / "inference.json").exists():
        raise FileExistsError("Preserve completed object readout")
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, created_at_utc=now(), repeats=config["repeats"]))
    rows, repeats, started = [], [], time.perf_counter()
    for repeat in config["repeats"]:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_g1_object_independent_readout.py"),
                        "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=3600)
        normal = repeat_normal(config, repeat)
        rows.extend(normal["rows"])
        path = RUN / f"repeat-{repeat}.json"
        repeats.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                            elapsed_seconds=normal["elapsed_seconds"]))
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows, repeats=repeats,
        pre_sha256=digest(RUN / "pre.json"), reader_only_intervention=True, fresh_reader_processes=2,
        end_to_end_repeat_performed=False, elapsed_seconds=time.perf_counter()-started,
        development_prior_scores_seen=True, blind_evaluation=False))
    verified_normal(config)


def verified_normal(config):
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or not normal["reader_only_intervention"]
            or normal["fresh_reader_processes"] != 2 or normal["end_to_end_repeat_performed"]
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config)):
        raise ValueError("All 80 frozen normal rows required before any truth access")
    if [entry["repeat"] for entry in normal["repeats"]] != config["repeats"]:
        raise ValueError("Both fresh reader-process receipts required")
    repeat_rows = []
    for entry in normal["repeats"]:
        if (entry["path"] != f"repeat-{entry['repeat']}.json"
                or digest(RUN / entry["path"]) != entry["sha256"]):
            raise ValueError("Reader-repeat receipt changed")
        repeat_rows.extend(repeat_normal(config, entry["repeat"])["rows"])
    if repeat_rows != normal["rows"]:
        raise ValueError("Fresh reader-process row pairing differs")
    _, manifest, original = parent_normal(config)
    originals, graphs, pairs = point_rows(original), [], {}
    for row in normal["rows"]:
        expected_path = f"records-r{row['repeat']}/{row['case_id']}-{row['fraction']}-{row['variant']}-{row['reader']}.json"
        if row["path"] != expected_path or digest(RUN / row["path"]) != row["sha256"]:
            raise ValueError("Normal readout graph changed")
        graph, bound = read(RUN / row["path"]), originals[row["case_id"]]
        expected_native = native_metadata(bound, row["variant"])
        call = dict(voxel_size=bound["camera_span_m"]*row["fraction"], origin=[0., 0., 0.])
        if (graph["gt_read"] or any(graph[key] != row[key] for key in KEYS)
                or graph["call_policy"] != call or graph["effective_policy"] != {**defaults()[row["reader"]], **call}
                or graph["base_snapshot_id"] != bound["snapshot_id"]
                or graph["full_input_point_count"] != bound["full_point_count"]
                or graph["supported_base_point_count"] != bound["supported_base_point_count"]
                or graph["source_support_sha256"] != bound["support_sha256"]
                or any(graph[key] != value for key, value in expected_native.items())):
            raise ValueError("Normal reader/complete base/native identity differs")
        key = row_key(row)[:-1]
        if row["reader"] == config["readers"][0]:
            pairs[key] = graph
        else:
            validate_pair(pairs[key], graph)
        graphs.append(graph)
    return normal, graphs, manifest, original


def repeat_projection(graph):
    keys = ("state", "resolution_state", "reason", "segments", "native_segments", "native_source_state",
        "native_patch_id", "native_outcome", "native_reason", "call_policy", "effective_policy", "raw_row_states",
        "base_snapshot_id", "source_support_sha256", *COUNT_KEYS)
    return {key: graph.get(key) for key in keys}


def score_geometry(state, segments, truth, gaps):
    if state != "complete":
        return None, None
    try:
        return physical_summary(segments, truth, gaps), None
    except ValueError as exc:
        return None, str(exc)


def evaluation_payload():
    before, config = checked()
    normal, graphs, manifest, original = verified_normal(config)
    # The full eighty-row normal and complete parent artifact inventories are
    # verified above before this run accesses truth. Prior development scores
    # were already seen; this is not blind evaluation.
    scene = ROOT / ".runtime/experiments" / config["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / config["scene_run_id"] / "manifest.json"
    if digest(truth_path) != read(scene / "prepared.json")["truth_sha256"]:
        raise ValueError("Frozen new-object truth identity differs")
    truth = read(truth_path)
    if (truth["input_sha256"] != digest(ROOT / "data/inputs" / config["scene_run_id"] / "manifest.json")
            or [case["case_id"] for case in truth["cases"]] != config["case_ids"]):
        raise ValueError("New-object truth/normal input pairing differs")
    truths = {}
    for source, actual in zip(manifest["cases"], truth["cases"]):
        if [frame["rgb_sha256"] for frame in source["frames"]] != [frame["rgb_sha256"] for frame in actual["frames"]]:
            raise ValueError("Object truth RGB identity/order differs")
        declared, target = actual["declared"], actual["declared"]["target"]
        segments = (target["segments"] if "segments" in target else [target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        truths[source["case_id"]] = segments, gaps
    rows, native_rows, projections, native_seen = [], [], {}, set()
    for entry, graph in zip(normal["rows"], graphs):
        truth_segments, gaps = truths[entry["case_id"]]
        score, error = score_geometry(graph["state"], graph["segments"], truth_segments, gaps)
        key = entry["case_id"], entry["fraction"], entry["variant"], entry["reader"]
        projection = repeat_projection(graph)
        repeated = None if entry["repeat"] == 0 else projection == projections[key]
        if entry["repeat"] == 0:
            projections[key] = projection
        rows.append(dict(**entry, state=graph["state"], resolution_state=graph.get("resolution_state"),
            reason=graph.get("reason"), output_segment_count=len(graph["segments"]), physical=score, scoring_error=error,
            native_source_state=graph["native_source_state"], native_outcome=graph["native_outcome"],
            native_patch_id=graph["native_patch_id"], native_segment_count=len(graph["native_segments"]),
            reader_elapsed_seconds=graph["reader_elapsed_seconds"], physical_equal_to_repeat_zero=repeated,
            **{name: graph[name] for name in COUNT_KEYS}))
        native_key = entry["case_id"], entry["variant"]
        if native_key not in native_seen:
            native_seen.add(native_key)
            native_score, native_error = score_geometry("complete", graph["native_segments"], truth_segments, gaps)
            native_rows.append(dict(case_id=entry["case_id"], variant=entry["variant"],
                source_state=graph["native_source_state"], outcome=graph["native_outcome"], reasons=graph["native_reason"],
                patch_id=graph["native_patch_id"], segments=graph["native_segments"], physical=native_score,
                scoring_error=native_error, measurement_scope=graph["native_metric_scope"],
                is_point_quality_measurement=False, scored_without_readout_clipping=True))
    if [(row["case_id"], row["variant"]) for row in native_rows] != [(cid, variant) for cid in config["case_ids"] for variant in config["variants"]]:
        raise ValueError("Complete ten native arms required")
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"],
        rows=rows, native_rows=native_rows, reader_repeats=normal["repeats"],
        reader_repeat_pairs=40, reader_repeat_equal=sum(row["physical_equal_to_repeat_zero"] is True for row in rows),
        source_parent_run_id=parent.RUN_ID, source_parent_inference_sha256=config["parent_inference_sha256"],
        source_parent_elapsed_seconds=original["elapsed_seconds"], parent_integration=original["rows"],
        parent_output_receipts=len(original["outputs"]), physical_policy=PHYSICAL_POLICY,
        measurement_scope="declared_central_rod_local_RGB_volume_not_other_members_surfaces_or_whole_object_quality",
        gpu_memory_scope="parent torch peak allocated for the process, not whole-GPU memory utilization",
        native_geometry_unchanged=True, point_inputs_unchanged=True, no_patch_refit=True,
        synthetic_development=True, independent_real_capture=False, end_to_end_repeat_performed=False,
        development_prior_scores_seen=True, blind_evaluation=False,
        alignment_performed=False, reader_qualified=False, g1_passed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve previous object readout evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("G1_OBJECT_INDEPENDENT_READOUT_EVALUATED", len(result["rows"]), "common", len(result["native_rows"]), "native", flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve previous object readout audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Object readout evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        reader_repeat_pairs=public["reader_repeat_pairs"], reader_repeat_equal=public["reader_repeat_equal"],
        same_complete_base_and_mask=True, same_sampling_counts=True, native_geometry_unchanged=True,
        point_inputs_unchanged=True, no_patch_refit=True, reader_only_repeats=True,
        end_to_end_repeat_performed=False, independent_real_capture=False, reader_qualified=False, g1_passed=False,
        development_prior_scores_seen=True, blind_evaluation=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("G1_OBJECT_INDEPENDENT_READOUT_POST_PASSED", flush=True)


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
