"""Frozen candidate-union readout, paired with the previous unique-row reader."""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import unittest

import run_fixture_point_patch as parent
import run_fixture_rgb_readout as previous
from run_fixture_challenges import block_truth, digest, now, physical_summary, read, receipt, write

ROOT = parent.ROOT
from creator_eval.rgb_candidate_readout import (  # noqa: E402
    POLICY,
    CandidateSupportedReadout,
    support_views,
)
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "fixture-candidate-readout-v1-20260930"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_candidate_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-09-30-fixture-candidate-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-09-30-fixture-candidate-readout-audit.json"


def inventory(config):
    return [(cid, f, v) for cid in config["case_ids"] for f in config["voxel_camera_span_fractions"]
            for v in ("base", "baseline", "cylinder_support")]


def checked():
    before = read(RUN / "pre.json")
    if before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"] or not before["development_prior_scores_seen"]:
        raise ValueError("Invalid pre-inference development freeze")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen receipt changed: "+name)
    return before, read(CONFIG)


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists():
        raise FileExistsError("Preserve existing candidate-union attempt")
    config = read(CONFIG)
    if (config["run_id"] != RUN_ID or config["parent_run_id"] != parent.RUN_ID
            or config["comparison_run_id"] != previous.RUN_ID or config["support_policy"] != POLICY):
        raise ValueError("Candidate-union protocol identity differs")
    before, old_config = previous.checked()
    parent.checked()
    old = read(previous.RUN / "inference.json")
    if (old["state"] != "complete" or old["gt_read"] or old["pre_sha256"] != digest(previous.RUN / "pre.json")
            or inventory(config) != inventory(old_config)
            or [(r["case_id"], r["fraction"], r["variant"]) for r in old["rows"]] != inventory(config)):
        raise ValueError("Complete paired comparison inventory required")
    hashes = dict(before["hashes"])
    receipt(previous.RUN / "pre.json", hashes)
    receipt(previous.RUN / "inference.json", hashes)
    for row in old["rows"]:
        receipt(previous.RUN / row["path"], hashes, row["sha256"])
    # Source-controlled analytic inputs only; preserve expected limitations too.
    log = io.StringIO()
    suite = unittest.TestLoader().discover(str(ROOT / "tests"), pattern="test_fixture_candidate_readout.py")
    controls = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not controls.wasSuccessful() or controls.testsRun < 11:
        raise ValueError("Candidate-union analytic controls failed: "+log.getvalue())
    RUN.mkdir(parents=True)
    sources = ["scripts/run_fixture_candidate_readout.py", "experiments/src/creator_eval/rgb_candidate_readout.py",
               "tests/test_fixture_candidate_readout.py", "configs/fixture_candidate_readout_v1.json"]
    for name in sources:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "analytic-controls.json", dict(tests_run=controls.testsRun, state="passed", gt_read=False,
        log=log.getvalue(), expected_limitation="Coarse voxels can still turn nearby supported lines into an unsupported middle axis; not reader qualification"))
    receipt(RUN / "analytic-controls.json", hashes)
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_readout_rows=len(inventory(config)), gt_read=False,
        physical_policy=read(parent.RUN / "evaluation-freeze.json")["physical_policy"],
        development_prior_scores_seen=True, no_independent_holdout_claim=True))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False, inference_existed=False,
                                development_prior_scores_seen=True, hashes=hashes))
    checked()
    print("CANDIDATE_READOUT_PREPARED", len(hashes), "receipts", flush=True)


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior candidate-union inference")
    (RUN / "records").mkdir()
    _, manifest, _ = parent.checked()
    old = {(r["case_id"], r["fraction"], r["variant"]): r for r in read(previous.RUN / "inference.json")["rows"]}
    rows, started = [], time.perf_counter()
    for case in manifest["cases"]:
        folder = parent.RUN / case["case_id"]
        raw = read(ROOT / case["parent_record"])
        views = support_views(raw["frames"], read(folder / "rods.json")["cameras"])
        base = compose(folder / "bundle/base")
        adapter = CandidateSupportedReadout(base["points"], views, config["support_policy"])
        for fraction in config["voxel_camera_span_fractions"]:
            for variant in ("base", "baseline", "cylinder_support"):
                old_row = old[(case["case_id"], fraction, variant)]
                reference = read(previous.RUN / old_row["path"])
                candidate = base if variant == "base" else open_candidate_view(folder / f"bundle/{variant}-enabled.json")
                if any(candidate[key].tobytes() != base[key].tobytes() for key in ("points", "point_ids")):
                    raise ValueError("Complete immutable base identity differs")
                output = adapter(candidate["segments"], reference["config"])
                path = RUN / "records" / f"{case['case_id']}-{fraction}-{variant}.json"
                write(path, output)
                rows.append(dict(case_id=case["case_id"], fraction=fraction, variant=variant,
                    path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                    previous_path=old_row["path"], previous_sha256=old_row["sha256"],
                    raw_row_states=[dict(view_id=v["view_id"], **v["row_states"]) for v in views],
                    previous_supported_curve_sample_count=reference["supported_curve_sample_count"],
                    previous_supported_base_point_count=reference["supported_base_point_count"]))
                print("CANDIDATE_READOUT", case["case_id"], fraction, variant, len(adapter.points), "points;",
                      output["supported_curve_sample_count"], "/", output["full_curve_sample_count"], "curve samples;",
                      len(output["segments"]), "curves", flush=True)
    checked()
    if [(r["case_id"], r["fraction"], r["variant"]) for r in rows] != inventory(config):
        raise ValueError("Incomplete normal inventory")
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started))


def evaluation_payload():
    before, config = checked()
    normal = read(RUN / "inference.json")
    if (normal["state"] != "complete" or normal["gt_read"] or normal["pre_sha256"] != digest(RUN / "pre.json")
            or [(r["case_id"], r["fraction"], r["variant"]) for r in normal["rows"]] != inventory(config)):
        raise ValueError("Complete frozen normal inference required")
    scene_id = read(parent.CONFIG)["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
    if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
        raise ValueError("Original truth identity differs")
    truth = {c["case_id"]: c for c in read(truth_path)["cases"]}
    rows = []
    for entry in normal["rows"]:
        path, old_path = RUN / entry["path"], previous.RUN / entry["previous_path"]
        if digest(path) != entry["sha256"] or digest(old_path) != entry["previous_sha256"]:
            raise ValueError("Paired readout graph changed")
        graph, old = read(path), read(old_path)
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = target.get("segments", [target["endpoints"]]) if target["present"] else []
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
        rows.append(dict(**entry, state=graph["state"], physical=scores[0], previous_physical=scores[1],
            scoring_error=errors[0], previous_scoring_error=errors[1], resolution_state=graph["resolution_state"],
            full_input_point_count=graph["full_input_point_count"], supported_base_point_count=graph["supported_base_point_count"],
            full_curve_sample_count=graph["full_curve_sample_count"], supported_curve_sample_count=graph["supported_curve_sample_count"],
            base_vote_histogram=graph["base_vote_histogram"], curve_vote_histogram=graph["curve_vote_histogram"]))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"], rows=rows,
        reader_qualified=False, g1_passed=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior candidate-union evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("CANDIDATE_READOUT_EVALUATED", len(result["rows"]), flush=True)


def post():
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Candidate-union evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        parent_outputs_unchanged=True, same_region_for_base_and_candidate=True,
        development_prior_scores_seen=True, reader_qualified=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("CANDIDATE_READOUT_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
