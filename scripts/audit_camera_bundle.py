"""Audit frozen BA/holdout separation, gauges and evaluation-only 3D diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from run_camera_bundle_pilot import (
    ROOT,
    checked_run,
    corrupt_training,
    digest,
    load_case,
    observations_from_record,
    read_json,
    validation_plan,
    write_json,
)

# isort: split
from creator_eval.camera_bundle import centers, triangulate, validate_heldout
from creator_eval.native_diagnostics import distribution
from creator_eval.track_image_controls import cast_planes

RUN_ID = "camera-bundle-pilot-v1-20260923"


def point_errors(plan, k, e, truth, alignment):
    transform = np.array(alignment["prediction_world_to_gt_world"])
    errors = []
    for track in plan:
        first = track["triangulation"][0]
        target, _, hit = cast_planes(np.array([first["xy"]]), truth["cameras"][first["view"]], truth["planes"])
        point = triangulate(track["triangulation"], k, e)
        transformed = point @ transform[:3, :3].T + transform[:3, 3]
        error = float(np.linalg.norm(transformed - target[0])) if hit[0] >= 0 and np.isfinite(transformed).all() else None
        errors.append(dict(track_id=track["track_id"], error_m=error))
    values = np.array([r["error_m"] if r["error_m"] is not None else np.nan for r in errors])
    return dict(summary=distribution(values), total_tracks=len(values), unknown=int((~np.isfinite(values)).sum()), rows=errors,
                scope="Camera-only Sim3; true first-hit point at first triangulation observation. Includes wrong descriptor tracks; finite physical correspondences are not GT-selected for optimization.")


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, inputs, frozen, manifest = checked_run(RUN_ID)
    summary_path = ROOT / "data/evaluation" / RUN_ID / "summary.json"
    summary = read_json(summary_path)
    inference = read_json(root / "inference.json")
    assert summary["inference_sha256"] == digest(root / "inference.json")
    assert inference["input_sha256"] == frozen["input_sha256"] and not inference["gt_read_during_inference"]
    for path, sha in summary["consumed_truth_sha256"].items():
        assert digest(ROOT / path) == sha
    rows, pair_checks = [], []
    for receipt in inference["records"]:
        assert digest(root / receipt["path"]) == receipt["sha256"]
        record = read_json(root / receipt["path"])
        _, case, tracks, method, k, e = load_case(RUN_ID, receipt["case_id"])
        train, val = observations_from_record(tracks, "train"), observations_from_record(tracks, "validation")
        assert not {t["track_id"] for t in train} & {t["track_id"] for t in val}
        changed, edits = corrupt_training(train, record["condition"]["corrupt_fraction"], method["seed"])
        assert edits == record["corruption"] and len(changed) == record["training_count"]
        assert len({r["track_id"] for r in edits}) == len(edits)
        plan = validation_plan(val, e)
        assert plan == record["validation_plan"]
        assert validate_heldout(k, e, plan) == record["before"]
        assert not record["dense_base_changed"] and not record["gt_read_during_inference"]
        result = record["result"]
        score = next(r for r in summary["rows"] if r["path"] == receipt["path"])
        truth_path = ROOT / "data/eval_gt" / case["source_run_id"] / "manifest.json"
        truth = next(c for c in read_json(truth_path)["cases"] if c["case_id"] == case["case_id"])
        before_points = point_errors(plan, k, e, truth, score["initial_camera"])
        after_points, first_error, scale_error = None, None, None
        if "extrinsics" in result:
            corrected_k, corrected_e = np.array(result["intrinsics"]), np.array(result["extrinsics"])
            first_error = float(abs(corrected_e[0] - e[0]).max())
            scale_error = float(abs(np.linalg.norm(centers(corrected_e)[-1] - centers(corrected_e)[0]) - np.linalg.norm(centers(e)[-1] - centers(e)[0])))
            assert first_error < 1e-6 and scale_error < 1e-6
            assert validate_heldout(corrected_k, corrected_e, plan) == record["after"]
            assert np.allclose(corrected_k[:, [0, 1], [0, 1]] / k[:, [0, 1], [0, 1]], result["focal_scale"])
            np.testing.assert_array_equal(corrected_k[:, :2, 2], k[:, :2, 2])
            assert set(result["kept_track_ids"]) | set(result["discarded_track_ids"]) == {t["track_id"] for t in train}
            assert not set(result["kept_track_ids"]) & set(result["discarded_track_ids"])
            after_points = point_errors(plan, corrected_k, corrected_e, truth, score["candidate_camera"])
        rows.append(dict(case_id=case["case_id"], condition=receipt["condition"], variant=receipt["variant"],
            first_pose_max_error=first_error, baseline_distance_error=scale_error, before_points=before_points, after_points=after_points,
            train_validation_disjoint=True, exact_fixed_holdout_plan=True, corruption_reproduced=True, decision=record["decision"]))
    for case in manifest["cases"]:
        assert digest(inputs / case["tracks_path"]) == case["tracks_sha256"]
        for frame in case["frames"]:
            assert digest(ROOT / frame["rgb"]) == frame["rgb_sha256"]
        if case["reuse"]:
            assert digest(ROOT / case["reuse"]["path"]) == case["reuse"]["prediction_sha256"]
            pair_checks.append(case["case_id"])
    assert len(rows) == 8 and len(inference["skipped"]) == 6
    names = [r["condition"].replace("corrupt_10pct_tracks", "10% corrupted").replace("original_tracks", "Original") + "\n" + r["variant"].replace("shared_focal", "focal") for r in summary["rows"]]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    for axis, key, label in ((axes[0], "median_px", "Held-out median error (px)"), (axes[1], "p95_px", "Held-out p95 error (px)")):
        axis.plot(x, [r["before"][key] for r in summary["rows"]], "o--", label="Uncorrected", color=".5")
        axis.plot(x, [r["after"][key] for r in summary["rows"]], "o-", label="Candidate", color="#2679a6")
        axis.set(ylabel=label, xticks=x, xticklabels=names)
        axis.tick_params(axis="x", labelrotation=65, labelsize=8)
        axis.legend(fontsize=8)
    axes[2].plot(x, [1000 * r["before_points"]["summary"]["median"] for r in rows], "o--", label="Uncorrected", color=".5")
    axes[2].plot(x, [1000 * r["after_points"]["summary"]["median"] if r["after_points"] else np.nan for r in rows], "o-", label="Candidate", color="#2679a6")
    axes[2].set(ylabel="Held-out point median error (mm), GT evaluation", xticks=x, xticklabels=names)
    axes[2].tick_params(axis="x", labelrotation=65, labelsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("DA3-initialized bundle pilot | one known analytic positive layout | no rod efficacy claim", fontsize=13)
    figure = ROOT / "docs/experiments/figures/2026-09-23-camera-bundle.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    report = dict(state="complete", run_id=RUN_ID, summary_sha256=digest(summary_path), rows=rows,
        original_rod_predictions_verified_unchanged=pair_checks, source_sha256=digest(Path(__file__)),
        figure_sha256=digest(figure), no_dense_patch_published=True)
    write_json(ROOT / "docs/experiments/results/2026-09-23-camera-bundle-audit.json", report)
    print("CAMERA_BUNDLE_AUDIT", len(rows), "condition gauges and holdout contracts", flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
