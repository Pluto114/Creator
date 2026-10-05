"""Frozen hypothesis-development replay of all 184 previously scored conditions."""

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
RUN_ID = "mixed-readout-bundle-v1-20261005"
REPLAY_COHORT = "development_replay_mixed"
SECTION_REPLAY_COHORT = "development_replay_section"
SAMPLING_REPLAY_COHORT = "development_replay_sampling"
SUPPORT_REPLAY_COHORT = "development_replay_support"
LOCAL_REPLAY_COHORT = "development_replay_local"
PARENT_RUN_ID = "mixed-readout-ridge-sampling-v1-20261004"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
TRUTH = ROOT / "data/eval_gt" / RUN_ID / "manifest.json"
CONFIG = ROOT / "configs/mixed_readout_bundle_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-05-mixed-readout-bundle.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-05-mixed-readout-bundle-audit.json"


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
    from creator_eval import (
        common_readout_bundle,
        common_readout_bundle_small,
        common_readout_ridge_sampling,
    )

    return {"bundle": common_readout_bundle, "bundle_small": common_readout_bundle_small,
            "ridge_sampling": common_readout_ridge_sampling}

def validated_config():
    config = read(CONFIG)
    required = dict(run_id=RUN_ID, parent_run_id=PARENT_RUN_ID,
        condition_count=184, development_replay_conditions=184, development_replay_family_count=46,
        all_inputs_prior_outputs_and_scores_seen=True, new_conditions=0, hypothesis_development_replay=True,
        readers=["bundle", "bundle_small", "ridge_sampling"], representations=["native", "sampled_points"],
        expected_normal_rows=1104, expected_fresh_rows=736, expected_reference_rows=368,
        reuse_parent_baseline=True, reader_overrides={}, diagnostic_recovery_precision_threshold=.9)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen five-cohort development-only protocol identity differs")
    return config

def inventory(manifest, config):
    return [(row["input_id"], reader, representation) for row in manifest["inputs"]
            for reader in config["readers"] for representation in config["representations"]]


def reference_rows():
    manifest = read(RUN / "baseline-references.json")
    expected = [(f"replay-{i:04d}", "ridge_sampling", representation)
                for i in range(184) for representation in ("native", "sampled_points")]
    rows = manifest["rows"]
    if (manifest["run_id"] != RUN_ID or manifest["parent_run_id"] != PARENT_RUN_ID
            or [(r["input_id"], r["reader"], r["representation"]) for r in rows] != expected):
        raise ValueError("Frozen parent baseline reference inventory differs")
    return {(r["input_id"], r["reader"], r["representation"]): r for r in rows}


def reference_result(reference, item, representation, call_policy, defaults):
    """Reuse only a byte-bound, same-input, same-policy normal parent result."""
    path = ROOT / reference["path"]
    if digest(path) != reference["sha256"]:
        raise ValueError("Parent normal reference changed")
    record = read(path)
    if (reference["input_id"] != item["input_id"] or reference["input_sha256"] != item["sha256"]
            or reference["reader"] != "ridge_sampling" or reference["representation"] != representation
            or record["input_id"] != reference["original_input_id"] or record["input_sha256"] != item["sha256"]
            or record["reader"] != "ridge_sampling" or record["representation"] != representation
            or record["gt_read"] or record["call_policy"] != call_policy
            or record["effective_policy"] != json_ready({**defaults, **call_policy})):
        raise ValueError("Parent normal reference identity/policy differs")
    return record["result"]


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["inference_existed"]
            or not before["development_replay_prior_outputs_and_scores_seen"]
            or not before["all_inputs_prior_outputs_and_scores_seen"] or not before["hypothesis_development_replay"]
            or not before["construction_truth_copied_separately"] or before["runtime"] != runtime()):
        raise ValueError("Invalid cohort-specific pre-inference freeze or changed runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source/input changed: " + name)
    manifest, frozen = read(INPUTS / "manifest.json"), read(RUN / "evaluation-freeze.json")
    expected_ids = [f"replay-{i:04d}" for i in range(184)]
    expected_cohorts = [REPLAY_COHORT]*44 + [SECTION_REPLAY_COHORT]*36 + [SAMPLING_REPLAY_COHORT]*40 + [SUPPORT_REPLAY_COHORT]*32 + [LOCAL_REPLAY_COHORT]*32
    if (manifest["run_id"] != RUN_ID or not manifest["truth_excluded"]
            or len(manifest["inputs"]) != config["condition_count"]
            or [row["input_id"] for row in manifest["inputs"]] != expected_ids
            or [row["cohort"] for row in manifest["inputs"]] != expected_cohorts
            or any(set(row) != {"input_id", "cohort", "voxel_size", "path", "sha256"} for row in manifest["inputs"])
            or len(inventory(manifest, config)) != config["expected_normal_rows"]
            or frozen["config"] != config):
        raise ValueError("Unlabelled five-cohort input inventory differs")
    if frozen["reader_defaults"] != json_ready({name: module.DEFAULTS for name, module in readers().items()}):
        raise ValueError("Reader defaults differ from pre-inference freeze")
    reference_rows()
    return before, manifest, config


def source_inventory():
    import run_mixed_readout_ridge_sampling as legacy

    # Closed inherited inventory: never rescan a mutable ancestor source glob.
    sources = set(read(legacy.RUN / "evaluation-freeze.json")["source_files"])
    sources.update({
        "scripts/run_mixed_readout_bundle.py", "configs/mixed_readout_bundle_v1.json",
        "tests/test_mixed_bundle_runner.py", "experiments/src/creator_eval/common_readout_bundle.py",
        "experiments/src/creator_eval/common_readout_bundle_small.py",
        "experiments/src/creator_eval/ridge_bundle_evidence.py",
        "experiments/src/creator_eval/ridge_plane_evidence.py",
        "tests/test_common_readout_bundle.py", "tests/test_bundle_integration.py",
        "tests/test_ridge_bundle_evidence.py", "tests/test_ridge_plane_evidence.py",
        "tests/test_bundle_ablation.py",
        "tests/test_common_readout_bundle_small.py", "tests/test_bundle_small_integration.py",
    })
    return sorted(sources)

def prepare():
    if RUN.exists() or INPUTS.exists() or TRUTH.parent.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve any earlier attempt; use a new run ID")
    config = validated_config()
    import run_mixed_readout_ridge_sampling as legacy
    from creator_eval.fixture_physical_metrics import POLICY

    # Complete 736-row parent provenance validation precedes any truth access.
    old_before, old_inputs, old_config, old_normal, old_records = legacy.normal_records()
    if (legacy.RUN_ID != PARENT_RUN_ID or len(old_inputs["inputs"]) != config["development_replay_conditions"]
            or len(old_normal["rows"]) != 736
            or [(r["input_id"], r["reader"], r["representation"]) for r in old_normal["rows"]]
            != legacy.inventory(old_inputs, old_config)):
        raise ValueError("Complete 184-condition / 736-row development parent required")
    if digest(legacy.TRUTH) != old_before["truth_sha256"]:
        raise ValueError("Original separated replay truth changed")
    old_truth = read(legacy.TRUTH)
    if (old_truth["input_manifest_sha256"] != digest(legacy.INPUTS / "manifest.json")
            or [r["input_id"] for r in old_truth["cases"]] != [r["input_id"] for r in old_inputs["inputs"]]):
        raise ValueError("Original development truth/input pairing differs")
    sources = source_inventory()
    if not all((ROOT / name).is_file() for name in sources):
        raise FileNotFoundError("Missing frozen source dependency")
    defaults = {name: module.DEFAULTS for name, module in readers().items()}
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    TRUTH.parent.mkdir(parents=True)
    hashes = {}
    for path in (legacy.RUN / "pre.json", legacy.INPUTS / "manifest.json", legacy.RUN / "inference.json",
                 legacy.RUN / "evaluation-freeze.json"):
        receipt(path, hashes)
    # Retain every verified parent normal row and its original cache provenance.
    for row in old_normal["rows"]:
        receipt(legacy.RUN / row["path"], hashes, row["sha256"])
        parent_reference = old_records[(row["input_id"], row["reader"], row["representation"])]["parent_reference"]
        if parent_reference is not None:
            receipt(ROOT / parent_reference["path"], hashes, parent_reference["sha256"])
    old_rows = {(r["input_id"], r["reader"], r["representation"]): r for r in old_normal["rows"]}
    references = []
    for number, item in enumerate(old_inputs["inputs"]):
        for representation in config["representations"]:
            old_row = old_rows[(item["input_id"], "ridge_sampling", representation)]
            references.append(dict(input_id=f"replay-{number:04d}", reader="ridge_sampling", representation=representation,
                original_input_id=item["input_id"], input_sha256=item["sha256"],
                path=(legacy.RUN / old_row["path"]).relative_to(ROOT).as_posix(), sha256=old_row["sha256"]))
    write(RUN / "baseline-references.json", dict(run_id=RUN_ID, parent_run_id=PARENT_RUN_ID, rows=references))
    receipt(RUN / "baseline-references.json", hashes)
    for name in sources:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    inputs, truths = [], []
    for number, (item, truth) in enumerate(zip(old_inputs["inputs"], old_truth["cases"])):
        input_id = f"replay-{number:04d}"
        cohort = (REPLAY_COHORT if number < 44 else SECTION_REPLAY_COHORT if number < 80
                  else SAMPLING_REPLAY_COHORT if number < 120 else SUPPORT_REPLAY_COHORT if number < 152
                  else LOCAL_REPLAY_COHORT)
        sha = receipt(ROOT / item["path"], hashes, item["sha256"])
        if truth["input_sha256"] != sha:
            raise ValueError("Original replay per-condition truth identity differs")
        inputs.append(dict(input_id=input_id, cohort=cohort, voxel_size=item["voxel_size"],
                           path=item["path"], sha256=sha))
        truths.append({**truth, "input_id": input_id, "cohort": cohort,
                       "original_input_id": item["input_id"], "prior_outputs_and_scores_seen": True})
    if len(inputs) != config["condition_count"]:
        raise ValueError("Exactly 184 previously scored inputs required; no new generator")
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, truth_excluded=True, inputs=inputs))
    receipt(INPUTS / "manifest.json", hashes)
    cohort_counts = {REPLAY_COHORT: (44, 11), SECTION_REPLAY_COHORT: (36, 9),
                     SAMPLING_REPLAY_COHORT: (40, 10), SUPPORT_REPLAY_COHORT: (32, 8),
                     LOCAL_REPLAY_COHORT: (32, 8)}
    write(TRUTH, dict(run_id=RUN_ID, input_manifest_sha256=digest(INPUTS / "manifest.json"), cases=truths,
        cohorts={name: dict(conditions=n, families=f, prior_outputs_and_scores_seen=True)
                 for name, (n, f) in cohort_counts.items()}, repeated_conditions_not_independent_objects=True,
        all_inputs_prior_outputs_and_scores_seen=True, hypothesis_development_replay=True))
    truth_sha = digest(TRUTH)
    write(RUN / "evaluation-freeze.json", dict(config=config, reader_defaults=defaults, physical_policy=POLICY,
        expected_readout_rows=1104, expected_fresh_rows=736, expected_reference_rows=368,
        cohort_condition_counts={name: n for name, (n, _) in cohort_counts.items()},
        parent_reference_run=PARENT_RUN_ID, parent_ridge_sampling_replay_is_cached=True,
        all_inputs_prior_outputs_and_scores_seen=True, hypothesis_development_replay=True, new_conditions=0,
        physical_object_count=None, alignment_performed=False, truth_file_excluded_from_normal_receipt_reads=True,
        source_files=sources, runtime=runtime(), third_party_freeze="Runtime versions; installed binaries are not copied"))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), inference_existed=False,
        construction_truth_copied_separately=True, development_replay_prior_outputs_and_scores_seen=True,
        all_inputs_prior_outputs_and_scores_seen=True, hypothesis_development_replay=True,
        privileged_prepare_copied_replay_truth=True, runtime=runtime(), hashes=hashes, source_count=len(sources),
        truth_path=TRUTH.relative_to(ROOT).as_posix(), truth_sha256=truth_sha))
    checked()
    print("MIXED_BUNDLE_PREPARED", len(hashes), "receipts; 184 development conditions / 1104 paired rows, 736 fresh", flush=True)

def infer():
    sys.addaudithook(block_truth)
    _, manifest, config = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior normal inference, including partial failures")
    from creator_eval.common_readout import sample_segments

    modules = readers()
    references = reference_rows()
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
                reference = references.get((item["input_id"], reader, representation))
                if reference is not None:
                    result = reference_result(reference, item, representation, call_policy, modules[reader].DEFAULTS)
                    execution_kind = "frozen_parent_reference"
                else:
                    execution_kind = "fresh_inference"
                    try:
                        result = modules[reader].readout(source_points.copy(), source_segments.copy(), call_policy)
                    except Exception as exc:
                        result = dict(state="error", resolution_state="error", segments=[], components=0,
                                      reason=type(exc).__name__ + ": " + str(exc))
                name = f"{item['input_id']}-{reader}-{representation}.json"
                output = RUN / "records" / name
                write(output, dict(input_id=item["input_id"], cohort=item["cohort"], input_sha256=item["sha256"], reader=reader,
                    representation=representation, gt_read=False, call_policy=call_policy,
                    effective_policy={**modules[reader].DEFAULTS, **call_policy}, result=result,
                    execution_kind=execution_kind, parent_reference=reference,
                    elapsed_seconds=0. if reference is not None else time.perf_counter()-row_started))
                rows.append(dict(input_id=item["input_id"], cohort=item["cohort"], reader=reader, representation=representation,
                    path=output.relative_to(RUN).as_posix(), sha256=digest(output)))
                print("MIXED_BUNDLE", item["input_id"], reader, representation, result["state"],
                      len(result["segments"]), "curves", flush=True)
    checked()
    if [(row["input_id"], row["reader"], row["representation"]) for row in rows] != inventory(manifest, config):
        raise ValueError("Incomplete ordered normal inventory")
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        fresh_inference_rows=config["expected_fresh_rows"], frozen_reference_rows=config["expected_reference_rows"],
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


def finite_completeness(score, target, threshold):
    """Predeclared display gate that cannot hide a missed short side branch."""
    if not score or not len(target):
        return False
    return bool(score["segment_count"] == len(target)
        and score["boundary_bijection_state"] == "scored"
        and score["boundary_max_error_m"] is not None
        and score["boundary_max_error_m"] <= .025
        and score["recovery_fraction"] is not None and score["recovery_fraction"] >= threshold
        and score["precision_fraction"] is not None and score["precision_fraction"] >= threshold
        and score["guarded_gap_false_length_m"] == 0.)


def cohort_summaries(rows, config):
    summaries = []
    for cohort, conditions, families, seen in ((REPLAY_COHORT, 44, 11, True),
                                               (SECTION_REPLAY_COHORT, 36, 9, True),
                                               (SAMPLING_REPLAY_COHORT, 40, 10, True),
                                               (SUPPORT_REPLAY_COHORT, 32, 8, True),
                                               (LOCAL_REPLAY_COHORT, 32, 8, True)):
        arms = []
        for reader in config["readers"]:
            for representation in config["representations"]:
                selected = [row for row in rows if (row["cohort"], row["reader"], row["representation"])
                            == (cohort, reader, representation)]
                arms.append(dict(reader=reader, representation=representation, normal_rows=len(selected),
                    positive_conditions=sum(row["positive_truth_present"] for row in selected),
                    positive_recovery_precision_at_least_90_percent=sum(row["positive_recovery_precision_at_least_90_percent"] for row in selected),
                    positive_finite_structure_display_pass=sum(row.get("positive_finite_structure_display_pass", False) for row in selected),
                    error_or_unmeasurable=sum(row["state"] != "complete" or row["scoring_error"] is not None for row in selected),
                    empty_outputs=sum(row["output_segment_count"] == 0 for row in selected)))
        summaries.append(dict(cohort=cohort, repeated_conditions=conditions, construction_families=families,
            prior_outputs_and_scores_seen=seen, independent_physical_objects=None, arms=arms))
    return summaries


def normal_records():
    """Verify all executed and cached rows without opening any truth or score."""
    before, manifest, config = checked()
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json")
            or normal["input_manifest_sha256"] != digest(INPUTS / "manifest.json")
            or [(r["input_id"], r["reader"], r["representation"]) for r in normal["rows"]] != inventory(manifest, config)):
        raise ValueError("All 1104 frozen normal/reference rows must finish before truth access")
    records = {}
    inputs = {item["input_id"]: item for item in manifest["inputs"]}
    references = reference_rows()
    modules = readers()
    fresh_rows, reused_rows = 0, 0
    for row in normal["rows"]:
        path = RUN / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Normal output changed")
        record = read(path)
        if (record["gt_read"] or record["input_sha256"] != inputs[row["input_id"]]["sha256"]
                or record["cohort"] != inputs[row["input_id"]]["cohort"] or row["cohort"] != record["cohort"]
                or any(record[key] != row[key] for key in ("input_id", "reader", "representation"))):
            raise ValueError("Normal output/input identity differs")
        key = row["input_id"], row["reader"], row["representation"]
        reference = references.get(key)
        kind = "frozen_parent_reference" if reference is not None else "fresh_inference"
        item = inputs[row["input_id"]]
        call_policy = dict(voxel_size=item["voxel_size"], origin=[0., 0., 0.])
        if (record["execution_kind"] != kind or record["parent_reference"] != reference
                or record["call_policy"] != call_policy
                or record["effective_policy"] != json_ready({**modules[row["reader"]].DEFAULTS, **call_policy})):
            raise ValueError("Normal/reference execution provenance differs")
        if reference is not None:
            if record["result"] != reference_result(reference, item, row["representation"], call_policy, modules["ridge_sampling"].DEFAULTS):
                raise ValueError("Cached parent result was modified")
            reused_rows += 1
        else:
            fresh_rows += 1
        records[key] = record
    if (fresh_rows != config["expected_fresh_rows"] or reused_rows != config["expected_reference_rows"]
            or normal["fresh_inference_rows"] != fresh_rows or normal["frozen_reference_rows"] != reused_rows):
        raise ValueError("Fresh/reference counts differ before truth access")
    return before, manifest, config, normal, records


def evaluation_payload():
    before, manifest, config, normal, records = normal_records()
    inputs = {item["input_id"]: item for item in manifest["inputs"]}
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
        if (case["input_sha256"] != inputs[entry["input_id"]]["sha256"]
                or case["cohort"] != inputs[entry["input_id"]]["cohort"]):
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
            execution_kind=record["execution_kind"],
            voxel_size=case["protocol"]["voxel_size"], phase=case["protocol"].get("phase"),
            density=case["protocol"].get("density"), sampling_phase=case["protocol"].get("sampling_phase"),
            prior_outputs_and_scores_seen=case["prior_outputs_and_scores_seen"], expected=truth["expected"],
            state=graph["state"], resolution_state=graph.get("resolution_state"), reason=graph.get("reason"),
            output_segment_count=len(graph["segments"]), physical=score, axis_deviation=axis, scoring_error=error,
            classification=classification, positive_truth_present=bool(target),
            positive_finite_structure_display_pass=finite_completeness(
                score, target, config["diagnostic_recovery_precision_threshold"]),
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
            pairs.append(dict(input_id=item["input_id"], cohort=item["cohort"], reader=reader, identical_geometry_and_resolution=not differences,
                              different_fields=differences))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=before["truth_sha256"],
        unchanged_pre_receipts=len(before["hashes"]), physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"],
        cohort_summaries=cohort_summaries(rows, config), repeated_condition_count=184, normal_row_count=1104,
        fresh_inference_row_count=736, frozen_reference_row_count=368,
        development_replay_prior_outputs_and_scores_seen=True, all_inputs_prior_outputs_and_scores_seen=True,
        hypothesis_development_replay=True, new_conditions=0,
        parent_reference_run=PARENT_RUN_ID, parent_ridge_sampling_replay_is_cached=True,
        no_combined_holdout_claim=True, independent_physical_object_count=None,
        elapsed_seconds=normal["elapsed_seconds"], rows=rows, representation_pairs=pairs,
        diagnostic_recovery_precision_threshold=config["diagnostic_recovery_precision_threshold"],
        diagnostic_threshold_scope=config["diagnostic_threshold_scope"],
        finite_structure_display_scope="Positive R/P >= 90%, exact declared segment count, boundary bijection <= 25 mm, zero guarded gap false length. Diagnostic only, not topology or G1 qualification.",
        all_representation_pairs_identical=all(row["identical_geometry_and_resolution"] for row in pairs),
        reader_qualified=False, g1_passed=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior analytic evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("MIXED_BUNDLE_EVALUATED", len(result["rows"]), "rows", flush=True)


def post():
    if (RUN / "post.json").exists() or AUDIT.exists():
        raise FileExistsError("Preserve prior post audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Analytic evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        fresh_inference_row_count=public["fresh_inference_row_count"], frozen_reference_row_count=public["frozen_reference_row_count"],
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        normal_row_count=public["normal_row_count"], all_representation_pairs_identical=public["all_representation_pairs_identical"],
        no_combined_holdout_claim=True, development_replay_prior_outputs_and_scores_seen=True,
        all_inputs_prior_outputs_and_scores_seen=True, hypothesis_development_replay=True, new_conditions=0,
        positive_abstentions_remain_failures=True, negative_rejections_are_construction_conditioned=True,
        reader_qualified=False, g1_passed=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("MIXED_BUNDLE_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
