"""Readout-only development replay on frozen full point/candidate fixture bundles."""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np
import run_fixture_candidate_readout as previous
import run_fixture_point_patch as parent
import run_mixed_readout_supported as analytic
from run_fixture_challenges import digest, now, physical_summary, read, receipt, write

ROOT = parent.ROOT
from creator_eval import common_readout_supported as reader  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY, support_views  # noqa: E402
from creator_eval.rgb_fixture_supported_readout import SCOPE, FixtureSupportedReadout  # noqa: E402
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "fixture-supported-readout-v1-20261004"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_supported_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-04-fixture-supported-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-04-fixture-supported-readout-audit.json"
block_truth = analytic.block_truth


def inventory(config):
    return [(cid, fraction, variant) for cid in config["case_ids"]
            for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"]]


def validated_config():
    config = read(CONFIG)
    required = dict(run_id=RUN_ID, parent_run_id=parent.RUN_ID, comparison_run_id=previous.RUN_ID,
        analytic_prerequisite_run_id=analytic.RUN_ID, analytic_expected_normal_rows=608,
        analytic_expected_fresh_rows=368, analytic_expected_reference_rows=240,
        case_ids=["r01", "r02", "r03"], voxel_camera_span_fractions=[.0025, .005],
        variants=["base", "baseline", "cylinder_support"], expected_normal_rows=18,
        support_policy=POLICY, reader_overrides={})
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen readout-only fixture protocol identity differs")
    return config


def source_inventory():
    sources = set(analytic.source_inventory())
    sources.update({"scripts/run_fixture_supported_readout.py", "configs/fixture_supported_readout_v1.json",
        "tests/test_fixture_supported_readout.py", "experiments/src/creator_eval/rgb_fixture_supported_readout.py",
        "scripts/run_fixture_candidate_readout.py", "scripts/run_fixture_point_patch.py",
        "scripts/run_fixture_rgb_readout.py", "scripts/run_fixture_challenges.py",
        "experiments/src/creator_eval/rgb_candidate_readout.py", "tests/test_fixture_candidate_readout.py",
        "reconstruction/src/creator_recon/domain/point_patch.py"})
    return sorted(sources)


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or not before["development_prior_scores_seen"] or before["runtime"] != analytic.runtime()):
        raise ValueError("Invalid readout-only development freeze or changed runtime")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != reader.DEFAULTS
            or frozen["expected_readout_rows"] != len(inventory(config))):
        raise ValueError("Frozen reader policy or complete inventory differs")
    return before, config


def analytic_receipts(config):
    """Normal/pre only: completeness and provenance, never a score-based gate."""
    before, _, _, normal, records = analytic.normal_records()
    if (len(normal["rows"]) != config["analytic_expected_normal_rows"]
            or normal["fresh_inference_rows"] != config["analytic_expected_fresh_rows"]
            or normal["frozen_reference_rows"] != config["analytic_expected_reference_rows"]):
        raise ValueError("Complete 608-row analytic normal prerequisite required")
    hashes = dict(before["hashes"])
    for path in (analytic.RUN / "pre.json", analytic.RUN / "inference.json"):
        receipt(path, hashes)
    for row in normal["rows"]:
        receipt(analytic.RUN / row["path"], hashes, row["sha256"])
        reference = records[(row["input_id"], row["reader"], row["representation"])]["parent_reference"]
        if reference is not None:
            receipt(ROOT / reference["path"], hashes, reference["sha256"])
    return hashes


def comparison_rows(config):
    normal = read(previous.RUN / "inference.json")
    if (normal["run_id"] != previous.RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(previous.RUN / "pre.json")
            or [(r["case_id"], r["fraction"], r["variant"]) for r in normal["rows"]] != inventory(config)):
        raise ValueError("Complete frozen 18-row comparison inventory required")
    return {(r["case_id"], r["fraction"], r["variant"]): r for r in normal["rows"]}


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve every earlier fixture-supported attempt")
    config = validated_config()
    hashes = analytic_receipts(config)
    before, old_config = previous.checked()
    if previous.inventory(old_config) != inventory(config):
        raise ValueError("Original measurement inventory differs")
    for name, sha in before["hashes"].items():
        receipt(ROOT / name, hashes, sha)
    _, manifest, _ = parent.checked()
    point_normal = read(parent.RUN / "inference.json")
    if (point_normal["run_id"] != parent.RUN_ID or point_normal["state"] != "complete" or point_normal["gt_read"]
            or point_normal["pre_sha256"] != digest(parent.RUN / "pre.json")
            or [c["case_id"] for c in manifest["cases"]] != config["case_ids"]
            or [r["case_id"] for r in point_normal["rows"]] != config["case_ids"]):
        raise ValueError("Complete original point-patch inventory required")
    for name, sha in point_normal["outputs"].items():
        receipt(ROOT / name, hashes, sha)
    for path in (parent.RUN / "pre.json", parent.RUN / "inference.json",
                 previous.RUN / "pre.json", previous.RUN / "inference.json"):
        receipt(path, hashes)
    for row in comparison_rows(config).values():
        receipt(previous.RUN / row["path"], hashes, row["sha256"])
    RUN.mkdir(parents=True)
    for name in source_inventory():
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_readout_rows=len(inventory(config)),
        reader_defaults=reader.DEFAULTS, physical_policy=read(parent.RUN / "evaluation-freeze.json")["physical_policy"],
        gt_read=False, development_prior_scores_seen=True, no_independent_holdout_claim=True,
        analytic_prerequisite_is_normal_completeness_only=True, measurement_scope=SCOPE))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False, inference_existed=False,
        runtime=analytic.runtime(), development_prior_scores_seen=True, hashes=hashes))
    checked()
    print("FIXTURE_SUPPORTED_PREPARED", len(hashes), "receipts", flush=True)


def validate_candidate(base, candidate, *, withdrawn=False):
    for key in ("base_snapshot_id", "world_frame_id", "length_unit"):
        if candidate[key] != base[key]:
            raise ValueError("Candidate snapshot/frame/unit binding differs")
    for key in ("points", "point_ids"):
        if (candidate[key].dtype != base[key].dtype or candidate[key].shape != base[key].shape
                or candidate[key].tobytes() != base[key].tobytes()):
            raise ValueError("Complete immutable base identity differs")
    if withdrawn and len(candidate["segments"]):
        raise ValueError("Withdrawn candidate did not restore empty patch geometry")


def call_policy(reference, cameras, fraction):
    extrinsics = np.asarray([c["world_to_camera_cv"] for c in cameras])
    homogeneous = np.concatenate((extrinsics, np.broadcast_to([0, 0, 0, 1], (len(cameras), 1, 4))), axis=1)
    centers = np.linalg.inv(homogeneous)[:, :3, 3]
    span = float(np.linalg.norm(centers[:, None]-centers, axis=-1).max())
    config = reference["config"]
    if (config["voxel_size"] != span*fraction or config["origin"] != [0., 0., 0.]):
        raise ValueError("Original camera-span voxel/origin differs")
    return dict(voxel_size=config["voxel_size"], origin=list(config["origin"]))


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior fixture-supported inference")
    _, manifest, _ = parent.checked()
    old = comparison_rows(config)
    point_rows = {r["case_id"]: r for r in read(parent.RUN / "inference.json")["rows"]}
    (RUN / "records").mkdir()
    rows, started = [], time.perf_counter()
    for case in manifest["cases"]:
        cid, folder = case["case_id"], parent.RUN / case["case_id"]
        raw, cameras = read(ROOT / case["parent_record"]), read(folder / "rods.json")["cameras"]
        views = support_views(raw["frames"], cameras)
        base = compose(folder / "bundle/base")
        original = point_rows[cid]
        if (base["base_snapshot_id"] != original["snapshot_id"] or len(base["points"]) != original["full_point_count"]
                or base["length_unit"] != "meter"
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + original["prediction_sha256"]):
            raise ValueError("Original complete snapshot/prediction frame differs")
        candidates = {"base": base}
        for variant in config["variants"][1:]:
            candidates[variant] = open_candidate_view(folder / f"bundle/{variant}-enabled.json")
            validate_candidate(base, candidates[variant])
            validate_candidate(base, open_candidate_view(folder / f"bundle/{variant}-withdrawn.json"), withdrawn=True)
        adapter = FixtureSupportedReadout(base["points"], views, config["support_policy"])
        for fraction in config["voxel_camera_span_fractions"]:
            for variant in config["variants"]:
                reference_row = old[(cid, fraction, variant)]
                reference = read(previous.RUN / reference_row["path"])
                output = adapter(candidates[variant]["segments"], call_policy(reference, cameras, fraction))
                for key in ("full_input_point_count", "supported_base_point_count", "full_curve_sample_count",
                            "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram"):
                    if output[key] != reference[key]:
                        raise ValueError("Frozen measurement-volume sampling changed: " + key)
                output.update(gt_read=False, case_id=cid, fraction=fraction, variant=variant,
                    raw_row_states=[dict(view_id=v["view_id"], **v["row_states"]) for v in views])
                path = RUN / "records" / f"{cid}-{fraction}-{variant}.json"
                write(path, output)
                rows.append(dict(case_id=cid, fraction=fraction, variant=variant,
                    path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                    previous_path=reference_row["path"], previous_sha256=reference_row["sha256"]))
                print("FIXTURE_SUPPORTED", cid, fraction, variant, output["state"], len(output["segments"]), "curves", flush=True)
    checked()
    if [(r["case_id"], r["fraction"], r["variant"]) for r in rows] != inventory(config):
        raise ValueError("Incomplete normal inventory")
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started))


def verified_normal(config):
    """Verify every new and reference graph before the caller may open truth."""
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json")
            or [(r["case_id"], r["fraction"], r["variant"]) for r in normal["rows"]] != inventory(config)):
        raise ValueError("All 18 frozen normal rows must finish before truth access")
    references, graphs = comparison_rows(config), []
    for row in normal["rows"]:
        key = row["case_id"], row["fraction"], row["variant"]
        old_row = references[key]
        if row["previous_path"] != old_row["path"] or row["previous_sha256"] != old_row["sha256"]:
            raise ValueError("Paired normal identity differs")
        path, old_path = RUN / row["path"], previous.RUN / old_row["path"]
        if digest(path) != row["sha256"] or digest(old_path) != old_row["sha256"]:
            raise ValueError("Paired readout graph changed")
        graph, old = read(path), read(old_path)
        if (graph["gt_read"] or any(graph[key] != row[key] for key in ("case_id", "fraction", "variant"))
                or graph["call_policy"] != {key: old["config"][key] for key in ("voxel_size", "origin")}
                or graph["effective_policy"] != {**reader.DEFAULTS, **graph["call_policy"]}):
            raise ValueError("Frozen normal graph identity/policy differs")
        graphs.append((graph, old))
    return normal, graphs


def evaluation_payload():
    before, config = checked()
    normal, graphs = verified_normal(config)
    # All normal inventory, identities and paired SHA values are checked above.
    scene_id = read(parent.CONFIG)["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
    if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
        raise ValueError("Original truth identity differs")
    truth = {c["case_id"]: c for c in read(truth_path)["cases"]}
    rows = []
    for entry, (graph, old) in zip(normal["rows"], graphs):
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = (target["segments"] if "segments" in target else [target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        scores, errors = [], []
        for item in (graph, old):
            score, error = None, None
            if item["state"] == "complete":
                try:
                    score = physical_summary(item["segments"], segments, gaps)
                except ValueError as exc:
                    error = str(exc)
            scores.append(score)
            errors.append(error)
        counts = {key: graph[key] for key in ("full_input_point_count", "supported_base_point_count",
            "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram")}
        rows.append(dict(**entry, **counts, state=graph["state"], resolution_state=graph.get("resolution_state"),
            reason=graph.get("reason"), output_segment_count=len(graph["segments"]), physical=scores[0],
            previous_state=old["state"], previous_output_segment_count=len(old["segments"]),
            previous_physical=scores[1], scoring_error=errors[0], previous_scoring_error=errors[1]))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"], rows=rows,
        physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"], measurement_scope=SCOPE,
        development_prior_scores_seen=True, no_independent_holdout_claim=True, reader_qualified=False,
        analytic_prerequisite_is_normal_completeness_only=True, g1_passed=False,
        alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior fixture-supported evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_SUPPORTED_EVALUATED", len(result["rows"]), flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve prior fixture-supported audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Fixture-supported evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        parent_outputs_unchanged=True, same_region_for_base_and_candidate=True,
        development_prior_scores_seen=True, reader_qualified=False, g1_passed=False, measurement_scope=SCOPE)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("FIXTURE_SUPPORTED_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
