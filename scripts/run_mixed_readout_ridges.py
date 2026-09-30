"""Freeze, run and audit unlabelled analytic mixed-geometry reader controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
RUN_ID = "mixed-readout-ridges-v1-20260930"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
TRUTH = ROOT / "data/eval_gt" / RUN_ID / "manifest.json"
CONFIG = ROOT / "configs/mixed_readout_ridges_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-09-30-mixed-readout-ridges.json"
AUDIT = ROOT / "docs/experiments/results/2026-09-30-mixed-readout-ridges-audit.json"


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    name, sha = path.relative_to(ROOT).as_posix(), digest(path)
    if (expected is not None and sha != expected) or (name in hashes and hashes[name] != sha):
        raise ValueError("Frozen receipt changed: " + name)
    hashes[name] = sha
    return sha


def block_truth(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    roots = (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments")
    if (any(path.is_relative_to(folder) for folder in roots)
            or path.name in {"evaluation.json", "post.json", "protocol.json", "generation-checks.json"}
            or path.name.endswith("render_request.json") or "rendered-rgb" in path.parts):
        raise PermissionError("Normal mixed-reader inference cannot access truth or scores")


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=importlib.metadata.version("scipy"))


def readers():
    from creator_eval import common_readout_abstention, common_readout_ridges

    return {"ridges": common_readout_ridges, "abstention": common_readout_abstention}


def validated_config():
    config = read(CONFIG)
    if (config["run_id"] != RUN_ID or config["seed"] != 410731
            or config["condition_count"] != 44 or config["construction_family_count"] != 11
            or config["grid_scales_m"] != [.006, .010] or config["phase_count"] != 2
            or config["readers"] != ["ridges", "abstention"]
            or config["representations"] != ["native", "sampled_points"]
            or config["expected_normal_rows"] != 176 or config["reader_overrides"] != {}
            or config["diagnostic_recovery_precision_threshold"] != .9):
        raise ValueError("Frozen analytic protocol identity differs")
    return config


def inventory(manifest, config):
    return [(row["input_id"], reader, representation) for row in manifest["inputs"]
            for reader in config["readers"] for representation in config["representations"]]


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["inference_existed"]
            or before["control_outputs_or_scores_seen_before_freeze"]
            or not before["construction_truth_generated_separately"] or before["runtime"] != runtime()):
        raise ValueError("Invalid analytic pre-inference freeze or changed runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source/input changed: " + name)
    manifest, frozen = read(INPUTS / "manifest.json"), read(RUN / "evaluation-freeze.json")
    if (manifest["run_id"] != RUN_ID or not manifest["truth_excluded"]
            or len(manifest["inputs"]) != config["condition_count"]
            or [row["input_id"] for row in manifest["inputs"]] != [f"control-{i:04d}" for i in range(44)]
            or any(set(row) != {"input_id", "voxel_size", "path", "sha256"} for row in manifest["inputs"])
            or len(inventory(manifest, config)) != config["expected_normal_rows"]
            or frozen["config"] != config):
        raise ValueError("Unlabelled input inventory differs")
    if frozen["reader_defaults"] != json_ready({name: module.DEFAULTS for name, module in readers().items()}):
        raise ValueError("Reader defaults differ from pre-inference freeze")
    return before, manifest, config


def source_inventory():
    # Conservatively freeze the complete local evaluation package: this covers
    # transitive reader, metric and generator imports without guessing a subset.
    paths = set((ROOT / "experiments/src/creator_eval").glob("*.py"))
    paths.update((ROOT / "tests").glob("test_common_readout*.py"))
    paths.update((ROOT / "scripts").glob("check_common_readout*.py"))
    paths.update(ROOT / name for name in (
        "scripts/run_mixed_readout_ridges.py", "scripts/environment_paths.py",
        "scripts/Enter-CreatorEnvironment.ps1", "tests/test_mixed_readout_controls.py",
        "tests/test_mixed_readout_runner.py",
        "tests/test_fixture_physical_metrics.py", "experiments/pyproject.toml",
        "configs/mixed_readout_ridges_v1.json",
        # Legacy reader tests import these helpers or exercise their --help
        # entrypoints. Freeze that full local import closure as well.
        "scripts/run_rod_identity_blender.py", "scripts/run_g1_point_patch.py",
        "scripts/run_g1_point_patch_components.py", "scripts/run_readout_background_challenges.py",
        "scripts/run_rod_evidence.py", "scripts/run_rod_identity_stress.py",
        "scripts/thin_pack_gt.py", "scripts/run_thin_line_controls.py",
        "reconstruction/src/creator_recon/__init__.py",
        "reconstruction/src/creator_recon/domain/__init__.py",
        "reconstruction/src/creator_recon/domain/point_patch.py",
        "configs/readout_background_challenges_v1.json",
    ))
    return sorted(path.relative_to(ROOT).as_posix() for path in paths)


def prepare():
    if RUN.exists() or INPUTS.exists() or TRUTH.parent.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve any earlier attempt; use a new run ID")
    config = validated_config()
    from creator_eval.fixture_physical_metrics import POLICY
    from creator_eval.mixed_readout_controls import (
        CASE_IDS,
        GRID_SCALES_M,
        SUBVOXEL_PHASES,
        generate,
    )

    if (len(CASE_IDS) != config["construction_family_count"] or list(GRID_SCALES_M) != config["grid_scales_m"]
            or len(SUBVOXEL_PHASES) != config["phase_count"]):
        raise ValueError("Generator factorial design differs")
    sources = source_inventory()
    if not all((ROOT / name).is_file() for name in sources):
        raise FileNotFoundError("Missing frozen source dependency")
    # Validate source policies before creating an immutable attempt directory.
    defaults = {name: module.DEFAULTS for name, module in readers().items()}
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    TRUTH.parent.mkdir(parents=True)
    hashes = {}
    for name in sources:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    inputs, truths = [], []
    for number, case in enumerate(generate(seed=config["seed"])):
        input_id = f"control-{number:04d}"
        if set(case["inputs"]) != {"points", "segments"}:
            raise ValueError("Normal arrays contain undeclared fields")
        path = INPUTS / (input_id + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **case["inputs"])
        sha = receipt(path, hashes)
        inputs.append(dict(input_id=input_id, voxel_size=case["protocol"]["voxel_size"],
                           path=path.relative_to(ROOT).as_posix(), sha256=sha))
        truths.append(dict(input_id=input_id, input_sha256=sha, family=case["case_id"],
                           condition_id=case["condition_id"], protocol=case["protocol"], truth=case["truth"]))
    if len(inputs) != config["condition_count"]:
        raise ValueError("Generator did not produce all 44 normal conditions")
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, truth_excluded=True, inputs=inputs))
    receipt(INPUTS / "manifest.json", hashes)
    write(TRUTH, dict(run_id=RUN_ID, input_manifest_sha256=digest(INPUTS / "manifest.json"), cases=truths,
                     construction_family_count=11, repeated_conditions_not_independent_objects=True))
    truth_sha = digest(TRUTH)
    write(RUN / "evaluation-freeze.json", dict(config=config, reader_defaults=defaults, physical_policy=POLICY,
        expected_readout_rows=176, construction_family_count=11, repeated_condition_count=44,
        physical_object_count=None, alignment_performed=False, truth_file_excluded_from_normal_receipt_reads=True,
        source_files=sources, runtime=runtime(), third_party_freeze="Runtime versions; installed binary packages are not copied"))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), inference_existed=False,
        generation_truth_created=True, construction_truth_generated_separately=True,
        control_outputs_or_scores_seen_before_freeze=False,
        prior_fixture_scores_informed_development=True, runtime=runtime(), hashes=hashes,
        source_count=len(sources), truth_path=TRUTH.relative_to(ROOT).as_posix(), truth_sha256=truth_sha))
    checked()
    print("MIXED_RIDGES_PREPARED", len(hashes), "receipts; 44 conditions / 176 normal rows", flush=True)


def infer():
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior normal inference, including partial failures")
    from creator_eval.common_readout import sample_segments

    modules = readers()
    (RUN / "records").mkdir()
    started, rows = time.perf_counter(), []
    for item in manifest["inputs"]:
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError("Normal input changed")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"points", "segments"}:
                raise ValueError("Normal input must contain only unlabelled arrays")
            points, segments = arrays["points"].copy(), arrays["segments"].copy()
        maximum = min(module.DEFAULTS["maximum_curve_samples"] for module in modules.values())
        samples = sample_segments(segments, item["voxel_size"]/2, maximum)
        options = dict(native=(points, segments),
                       sampled_points=(np.concatenate((points, samples)), np.empty((0, 2, 3))))
        for reader in config["readers"]:
            for representation in config["representations"]:
                source_points, source_segments = options[representation]
                # No construction label, expected axis, truth, or primitive ID is
                # passed to either reader. Copies also prevent one arm mutation.
                call_policy = dict(voxel_size=item["voxel_size"], origin=[0., 0., 0.])
                row_started = time.perf_counter()
                try:
                    result = modules[reader].readout(source_points.copy(), source_segments.copy(), call_policy)
                except Exception as exc:
                    result = dict(state="error", resolution_state="error", segments=[], components=0,
                                  reason=type(exc).__name__ + ": " + str(exc))
                name = f"{item['input_id']}-{reader}-{representation}.json"
                output = RUN / "records" / name
                write(output, dict(input_id=item["input_id"], input_sha256=item["sha256"], reader=reader,
                    representation=representation, gt_read=False, call_policy=call_policy,
                    effective_policy={**modules[reader].DEFAULTS, **call_policy}, result=result,
                    elapsed_seconds=time.perf_counter()-row_started))
                rows.append(dict(input_id=item["input_id"], reader=reader, representation=representation,
                    path=output.relative_to(RUN).as_posix(), sha256=digest(output)))
                print("MIXED_RIDGES", item["input_id"], reader, representation, result["state"],
                      len(result["segments"]), "curves", flush=True)
    checked()
    if [(row["input_id"], row["reader"], row["representation"]) for row in rows] != inventory(manifest, config):
        raise ValueError("Incomplete ordered normal inventory")
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), input_manifest_sha256=digest(INPUTS / "manifest.json"),
        completed_at_utc=now(), elapsed_seconds=time.perf_counter()-started))


def physical_summary(prediction, truth, gaps):
    from creator_eval.fixture_physical_metrics import score_finite_structure

    p = score_finite_structure(prediction, truth, gaps)
    return dict(segment_count=p["prediction_segment_count"], total_length_m=p["length"]["prediction_union_m"],
        truth_length_m=p["length"]["truth_union_m"], signed_length_error_m=p["length"]["signed_error_m"],
        boundary_count=p["predicted_endpoints"]["boundary_endpoint_count"],
        boundary_bijection_state=p["boundary_bijection"]["state"],
        boundary_max_error_m=p["boundary_bijection"]["maximum_error_m"],
        recovery_fraction=p["curves"]["recovery_fraction"], precision_fraction=p["curves"]["precision_fraction"],
        truth_to_prediction_p95_m=p["curves"]["truth_to_prediction"]["distance_p95"],
        prediction_to_truth_p95_m=p["curves"]["prediction_to_truth"]["distance_p95"],
        guarded_gap_false_length_m=p["guarded_gap"]["false_proximity_length_m"],
        guarded_gap_length_m=p["guarded_gap"]["interior_length_m"],
        truth_boundary_to_prediction_boundary=p["truth_boundary_to_prediction_boundary"],
        prediction_boundary_to_truth_boundary=p["prediction_boundary_to_truth_boundary"])


def axis_deviation(prediction, truth):
    prediction, truth = np.asarray(prediction, float).reshape(-1, 2, 3), np.asarray(truth, float).reshape(-1, 2, 3)
    rows = []
    for segment in prediction:
        if not len(truth):
            break
        direction = (segment[1]-segment[0])/np.linalg.norm(segment[1]-segment[0])
        alternatives = []
        for index, target in enumerate(truth):
            axis = (target[1]-target[0])/np.linalg.norm(target[1]-target[0])
            delta = segment-target[0]
            distances = np.linalg.norm(delta-(delta @ axis)[:, None]*axis, axis=1)
            angle = float(np.rad2deg(np.arccos(np.clip(abs(direction @ axis), 0., 1.))))
            alternatives.append((float(distances.mean()), angle, index, float(distances.max())))
        mean, angle, index, maximum = min(alternatives)
        rows.append(dict(nearest_truth_axis_index=index, mean_offset_m=mean, maximum_offset_m=maximum,
                         unsigned_angle_degrees=angle))
    return dict(rows=rows, maximum_offset_m=max((r["maximum_offset_m"] for r in rows), default=None),
                maximum_unsigned_angle_degrees=max((r["unsigned_angle_degrees"] for r in rows), default=None),
                policy="Evaluation-only nearest infinite truth axis; no fitting or alignment; finite recovery and endpoints reported separately")


def evaluation_payload():
    before, manifest, config = checked()
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json")
            or normal["input_manifest_sha256"] != digest(INPUTS / "manifest.json")
            or [(r["input_id"], r["reader"], r["representation"]) for r in normal["rows"]] != inventory(manifest, config)):
        raise ValueError("All 176 frozen normal rows must finish before truth access")
    records = {}
    inputs = {item["input_id"]: item for item in manifest["inputs"]}
    for row in normal["rows"]:
        path = RUN / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Normal output changed")
        record = read(path)
        if (record["gt_read"] or record["input_sha256"] != inputs[row["input_id"]]["sha256"]
                or any(record[key] != row[key] for key in ("input_id", "reader", "representation"))):
            raise ValueError("Normal output/input identity differs")
        records[(row["input_id"], row["reader"], row["representation"])] = record
    # The first truth-file access occurs only after every normal row is checked.
    if digest(TRUTH) != before["truth_sha256"]:
        raise ValueError("Separated construction truth changed")
    truth_manifest = read(TRUTH)
    if (truth_manifest["run_id"] != RUN_ID or truth_manifest["input_manifest_sha256"] != digest(INPUTS / "manifest.json")
            or [row["input_id"] for row in truth_manifest["cases"]] != list(inputs)):
        raise ValueError("Truth/input pairing differs")
    truths = {row["input_id"]: row for row in truth_manifest["cases"]}
    rows, pairs = [], []
    for entry in normal["rows"]:
        record = records[(entry["input_id"], entry["reader"], entry["representation"])]
        graph, case = record["result"], truths[entry["input_id"]]
        if case["input_sha256"] != inputs[entry["input_id"]]["sha256"]:
            raise ValueError("Per-condition truth identity differs")
        truth = case["truth"]
        target, gaps = truth["expected_segments"], truth["forbidden_segments"]
        score, error, axis = None, None, None
        if graph["state"] == "complete":
            try:
                score = physical_summary(graph["segments"], target, gaps)
                axis = axis_deviation(graph["segments"], target)
            except ValueError as exc:
                error = str(exc)
        classification = "error_or_unmeasurable_not_a_rejection"
        if graph["state"] == "complete" and error is None:
            if target:
                classification = "positive_empty_recovery_failure" if not graph["segments"] else "positive_readout_see_metrics"
            elif truth["expected"] == "unresolved":
                classification = ("unresolved_construction_emission" if graph["segments"]
                                  else "construction_conditioned_unresolved_abstention")
            else:
                classification = "negative_emission_failure" if graph["segments"] else "construction_conditioned_empty_rejection_only"
        rows.append(dict(**entry, family=case["family"], condition_id=case["condition_id"],
            voxel_size=case["protocol"]["voxel_size"], phase=case["protocol"]["phase"], expected=truth["expected"],
            state=graph["state"], resolution_state=graph.get("resolution_state"), reason=graph.get("reason"),
            output_segment_count=len(graph["segments"]), physical=score, axis_deviation=axis, scoring_error=error,
            classification=classification, positive_truth_present=bool(target),
            metric_truth_scope=("No declared recoverable axis: empty-reference metrics are not physical negative classification"
                                if truth["expected"] == "unresolved" else "Declared analytic construction, not foreground identity"),
            positive_recovery_precision_at_least_90_percent=(bool(score and target)
                and score["recovery_fraction"] is not None and score["precision_fraction"] is not None
                and score["recovery_fraction"] >= config["diagnostic_recovery_precision_threshold"]
                and score["precision_fraction"] >= config["diagnostic_recovery_precision_threshold"]),
            reader_elapsed_seconds=record["elapsed_seconds"]))
    for item in manifest["inputs"]:
        for reader in config["readers"]:
            a, b = [records[(item["input_id"], reader, representation)]["result"] for representation in config["representations"]]
            keys = ("state", "resolution_state", "reason", "segments", "components", "occupied_voxels")
            differences = [key for key in keys if a.get(key) != b.get(key)]
            pairs.append(dict(input_id=item["input_id"], reader=reader, identical_geometry_and_resolution=not differences,
                              different_fields=differences))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=before["truth_sha256"],
        unchanged_pre_receipts=len(before["hashes"]), physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"],
        construction_family_count=11, repeated_condition_count=44, normal_row_count=176, independent_physical_object_count=None,
        elapsed_seconds=normal["elapsed_seconds"], rows=rows, representation_pairs=pairs,
        diagnostic_recovery_precision_threshold=config["diagnostic_recovery_precision_threshold"],
        diagnostic_threshold_scope=config["diagnostic_threshold_scope"],
        all_representation_pairs_identical=all(row["identical_geometry_and_resolution"] for row in pairs),
        reader_qualified=False, g1_passed=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior analytic evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("MIXED_RIDGES_EVALUATED", len(result["rows"]), "rows", flush=True)


def post():
    if (RUN / "post.json").exists() or AUDIT.exists():
        raise FileExistsError("Preserve prior post audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Analytic evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        normal_row_count=public["normal_row_count"], all_representation_pairs_identical=public["all_representation_pairs_identical"],
        positive_abstentions_remain_failures=True, negative_rejections_are_construction_conditioned=True,
        reader_qualified=False, g1_passed=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("MIXED_RIDGES_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
