"""Paired RGB-supported common readout on the complete fresh fixture bundles."""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import run_fixture_point_patch as parent
from run_fixture_challenges import block_truth, digest, now, physical_summary, read, receipt, write

ROOT = parent.ROOT
from creator_eval.rgb_supported_readout import POLICY, SupportedReadout, support_views  # noqa: E402
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "fixture-rgb-readout-v1-20260929"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_rgb_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-09-29-fixture-rgb-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-09-29-fixture-rgb-readout-audit.json"


def checked():
    before = read(RUN / "pre.json")
    if before["run_id"] != RUN_ID or before["gt_read"] or before["parent_evaluation_existed"]:
        raise ValueError("Invalid pre-evaluation freeze")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen receipt changed: "+name)
    return before, read(CONFIG)


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or parent.PUBLIC.exists() or (parent.RUN / "evaluation.json").exists():
        raise FileExistsError("Region must freeze before either new physical evaluation")
    config = read(CONFIG)
    if config["run_id"] != RUN_ID or config["parent_run_id"] != parent.RUN_ID or config["support_policy"] != POLICY:
        raise ValueError("Readout policy identity differs")
    before, _, _ = parent.checked()
    inference = read(parent.RUN / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"]:
        raise ValueError("Complete GT-free parent required")
    hashes = dict(before["hashes"])
    receipt(parent.RUN / "pre.json", hashes)
    receipt(parent.RUN / "inference.json", hashes)
    for name, sha in inference["outputs"].items():
        receipt(ROOT / name, hashes, sha)
    RUN.mkdir(parents=True)
    sources = ["scripts/run_fixture_rgb_readout.py", "experiments/src/creator_eval/rgb_supported_readout.py",
               "tests/test_fixture_rgb_readout.py", "configs/fixture_rgb_readout_v1.json"]
    for name in sources:
        destination = RUN / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        receipt(destination, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_readout_rows=18, gt_read=False,
        physical_policy=read(parent.RUN / "evaluation-freeze.json")["physical_policy"]))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False,
        parent_evaluation_existed=False, hashes=hashes))
    checked()
    print("RGB_READOUT_PREPARED", len(hashes), "receipts", flush=True)


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve old RGB-supported readout")
    (RUN / "records").mkdir()
    _, manifest, _ = parent.checked()
    rows, started = [], time.perf_counter()
    for case in manifest["cases"]:
        folder = parent.RUN / case["case_id"]
        original = read(ROOT / case["parent_record"])
        cameras = read(folder / "rods.json")["cameras"]
        views = support_views(original["frames"], cameras)
        base = compose(folder / "bundle/base")
        adapter = SupportedReadout(base["points"], views, config["support_policy"])
        for fraction in config["voxel_camera_span_fractions"]:
            broad = read(folder / f"readout-{fraction}-base.json")
            reader_config = broad["config"]
            for variant in ("base", "baseline", "cylinder_support"):
                candidate = base if variant == "base" else open_candidate_view(folder / f"bundle/{variant}-enabled.json")
                if candidate["points"].tobytes() != base["points"].tobytes() or candidate["point_ids"].tobytes() != base["point_ids"].tobytes():
                    raise ValueError("Paired base point identity differs")
                output = adapter(candidate["segments"], reader_config)
                path = RUN / "records" / f"{case['case_id']}-{fraction}-{variant}.json"
                write(path, output)
                rows.append(dict(case_id=case["case_id"], fraction=fraction, variant=variant,
                    path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
                print("RGB_READOUT", case["case_id"], fraction, variant, len(adapter.points), "supported points;", len(output["segments"]), "curves", flush=True)
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started))


def evaluation_payload():
    before, config = checked()
    inference = read(RUN / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"] or inference["pre_sha256"] != digest(RUN / "pre.json"):
        raise ValueError("Incomplete normal supported readout")
    expected = [(cid, f, v) for cid in config["case_ids"] for f in config["voxel_camera_span_fractions"] for v in ("base", "baseline", "cylinder_support")]
    if [(r["case_id"], r["fraction"], r["variant"]) for r in inference["rows"]] != expected:
        raise ValueError("Incomplete method inventory")
    # Parent physical evaluation is now safe; both normal outputs and pre receipts exist.
    broad = parent.evaluation_payload()
    broad_rows = {(r["case_id"], r["fraction"], r["variant"]): r for r in broad["readout_rows"]}
    pc = read(parent.CONFIG)
    truth_path = ROOT / "data/eval_gt" / pc["scene_run_id"] / "manifest.json"
    truth = {c["case_id"]: c for c in read(truth_path)["cases"]}
    rows = []
    for entry in inference["rows"]:
        path = RUN / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Supported graph changed")
        graph = read(path)
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = target.get("segments", [target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        score, error = None, None
        if graph["state"] == "complete":
            try:
                score = physical_summary(graph["segments"], segments, gaps)
            except ValueError as exc:
                error = str(exc)
        rows.append(dict(**entry, state=graph["state"], physical=score, scoring_error=error,
            supported_base_point_count=graph["supported_base_point_count"], full_input_point_count=graph["full_input_point_count"],
            supported_curve_sample_count=graph["supported_curve_sample_count"], resolution_state=graph["resolution_state"],
            broad_strip_physical=broad_rows[(entry["case_id"], entry["fraction"], entry["variant"])]["physical"]))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=broad["truth_sha256"],
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=inference["elapsed_seconds"],
        rows=rows, reader_qualified=False, g1_passed=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists():
        raise FileExistsError("Preserve old supported evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("RGB_READOUT_EVALUATED", len(result["rows"]), flush=True)


def post():
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Supported evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        parent_outputs_unchanged=True, same_region_for_base_and_candidate=True, reader_qualified=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("RGB_READOUT_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
