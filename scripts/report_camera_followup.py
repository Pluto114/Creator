"""Verify camera snapshots and exact split replays, then summarize frozen outcomes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from run_foreground_estimated_pilot import (
    ROOT,
    checked,
    checked_case,
    digest,
    read_json,
    write_json,
)
from run_foreground_tracks import checked_tracks

# isort: split
from creator_recon.domain.point_patch import load_snapshot, open_candidate_view

RESULTS = ROOT / "docs/experiments/results"


def verify_ray():
    from jsonschema import Draft202012Validator
    run, inputs, frozen, method = checked("foreground-ray-pose-v1-20260922")
    old, old_inputs, _, old_method = checked("foreground-estimated-pilot-v1-20260922")
    for key in ("model", "process_res", "candidate_cap", "identity_policy", "association_common", "extent", "cylinder_screen", "annotations"):
        assert method[key] == old_method[key]
    assert method["use_ray_pose"] and not old_method["use_ray_pose"]
    assert digest(run / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["source_sha256"].items():
        assert digest(run / "source_snapshot" / name) == sha
    index = read_json(run / "predictions.json")
    schema = Draft202012Validator(read_json(ROOT / "schemas/pilot/point-patch-0.1.schema.json"))
    view_schema = Draft202012Validator(read_json(ROOT / "schemas/pilot/candidate-view-0.1.schema.json"))
    checks, documents, saved_views = [], 0, 0
    for entry in method["cases"]:
        case = checked_case(inputs, entry)
        prior = checked_case(old_inputs, next(c for c in old_method["cases"] if c["case_id"] == entry["case_id"]))
        for a, b in zip(case["frames"], prior["frames"]):
            assert {k: v for k, v in a.items() if k != "rgb"} == {k: v for k, v in b.items() if k != "rgb"}
        folder = run / entry["case_id"]
        native_path = folder / "prediction/prediction.npz"
        receipt = next(r for r in index["records"] if r["case_id"] == entry["case_id"])
        assert digest(native_path) == receipt["prediction_sha256"]
        assert digest(folder / "prediction/report.json") == receipt["report_sha256"]
        manifest, base = load_snapshot(folder / "base")
        schema.validate(manifest)
        documents += 1
        assert manifest["metadata"]["source_prediction_sha256"] == receipt["prediction_sha256"]
        for p in folder.glob("patch-*/manifest.json"):
            schema.validate(read_json(p))
            documents += 1
        for view in sorted(folder.glob("*-enabled.json")) + sorted(folder.glob("*-withdrawn.json")):
            view_schema.validate(read_json(view))
            candidate = open_candidate_view(view)
            assert candidate["points"].tobytes() == base["points"].tobytes()
            assert candidate["point_ids"].tobytes() == base["point_ids"].tobytes()
            if "withdrawn" in view.stem:
                assert len(candidate["segments"]) == 0
            documents += 1
            saved_views += 1
        maximum, count = 0., 0
        with np.load(native_path, allow_pickle=False) as native:
            for frame, (k, e) in enumerate(zip(native["intrinsics"], native["extrinsics"])):
                indices = np.flatnonzero(base["point_ids"][:, 0] == frame)
                indices = indices[np.unique(np.linspace(0, len(indices) - 1, 256).astype(int))]
                ids = base["point_ids"][indices]
                xyz = base["points"][indices] @ e[:, :3].T + e[:, 3]
                uv = xyz @ k.T
                maximum = max(maximum, float(np.max(abs(uv[:, :2] / uv[:, 2, None] - ids[:, [2, 1]]))))
                count += len(ids)
            with np.load(old / entry["case_id"] / "prediction/prediction.npz", allow_pickle=False) as previous:
                difference = float(np.max(abs(native["depth"] - previous["depth"])))
        assert maximum < .002
        checks.append(dict(case_id=entry["case_id"], point_count=len(base["points"]), sampled_source_pixels=count,
            maximum_pixel_error=maximum, maximum_depth_difference_from_decoder=difference))
    assert documents == 58 and saved_views == 36 and len(checks) == 4
    return dict(schema_documents=documents, saved_views=saved_views, checks=checks,
                same_rgb_pools_clicks_and_geometry_rules=True, exact_base_rollback=True)


def verify_replay(old_id, new_id, old_summary_path, control):
    old, old_frozen, old_inputs = checked_tracks(old_id)
    run, frozen, inputs = checked_tracks(new_id)
    assert {k: v for k, v in inputs["method"].items() if k != "split"} == {k: v for k, v in old_inputs["method"].items() if k != "split"}
    assert [(c["case_id"], [f["rgb_sha256"] for f in c["frames"]]) for c in inputs["cases"]] == [(c["case_id"], [f["rgb_sha256"] for f in c["frames"]]) for c in old_inputs["cases"]]
    for root, meta, source_inputs in ((old, old_frozen, old_inputs), (run, frozen, inputs)):
        assert digest(root / "protocol.json") == meta["protocol_sha256"]
        for case in source_inputs["cases"]:
            for frame in case["frames"]:
                assert digest(ROOT / frame["rgb"]) == frame["rgb_sha256"]
    old_index, index = read_json(old / "inference.json"), read_json(run / "inference.json")
    for index_data, meta in ((old_index, old_frozen), (index, frozen)):
        assert index_data["state"] == "complete" and index_data["source_sha256"] == meta["source_sha256"]
        assert index_data["input_sha256"] == meta["input_sha256"] and not index_data["gt_read_during_inference"]
    previous = read_json(old_summary_path)
    assert previous["inference_sha256"] == digest(old / "inference.json")
    rows = []
    assert len(index["records"]) == len(old_index["records"])
    for item in index["records"]:
        original = next(i for i in old_index["records"] if (i["case_id"], i["detector"]) == (item["case_id"], item["detector"]))
        assert digest(run / item["path"]) == item["sha256"] and digest(old / original["path"]) == original["sha256"]
        current, prior = read_json(run / item["path"]), read_json(old / original["path"])
        assert current["graph"] == prior["graph"] and current["feature_xy"] == prior["feature_xy"]
        for a, b in zip(current["pairs"], prior["pairs"]):
            for key in ("first_view", "second_view", "keypoint_ids", "first_xy", "second_xy", "track_ids"):
                assert a[key] == b[key]
        scoring = next(r for r in previous["rows" if control else "summaries"] if (r["case_id"], r["detector"]) == (item["case_id"], item["detector"]))
        assert current["split"]["shared_cells"] == prior["split"]["shared_cells"] == 0
        rows.append({**{k: current[k] for k in ("case_id", "detector", "coverage", "availability", "nonplanar_validated_pairs", "split")},
            "complete_tracks": len(current["graph"]["tracks"]), "unchanged_physical_track_counts": scoring["track_truth_counts"],
            "previous_split": prior["split"], "previous_availability": prior["availability"],
            "geometry_states": dict(Counter(p["geometry"]["state"] for p in current["pairs"])),
            "F_validation_consistent_pairs": sum(p["geometry"].get("models", {}).get("F", {}).get("state") == "validation_consistent" for p in current["pairs"]),
            "planar_or_repetitive_pairs": sum(p["geometry"].get("planar_or_repetitive_explanation_possible", False) for p in current["pairs"])})
    return dict(run_id=new_id, input_sha256=frozen["input_sha256"], inference_sha256=digest(run / "inference.json"),
        unchanged_graphs_pixels_images=True, physical_scores_source=old_summary_path.relative_to(ROOT).as_posix(),
        physical_scores_source_sha256=digest(old_summary_path), rows=rows, inference_seconds=index["elapsed_seconds"])


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ray_integrity = verify_ray()
    real = verify_replay("foreground-tracks-v1-20260922", "foreground-track-bands-v1-20260922", RESULTS / "2026-09-22-foreground-tracks.json", False)
    control = verify_replay("track-image-controls-v1-20260922", "track-image-control-bands-v1-20260922", RESULTS / "2026-09-22-track-image-controls.json", True)
    paths = [RESULTS / f"2026-09-22-foreground-{suffix}.json" for suffix in ("estimated-pilot", "ray-pose")]
    predictions = [read_json(p) for p in paths]
    rows, cameras = [], []
    for mode, report in zip(("decoder", "ray_pose"), predictions):
        for name, sha in report["source_sha256"].items():
            assert digest(ROOT / name) == sha
        root = ROOT / ".runtime/experiments" / report["run_id"]
        assert digest(root / "inference.json") == report["inference_sha256"]
        for record in read_json(root / "inference.json")["records"]:
            assert digest(root / record["path"]) == record["sha256"]
            receipt = read_json(root / record["path"])
            assert digest(root / record["case_id"] / "geometry.json") == receipt["geometry_sha256"]
        for method in ("baseline", "cylinder_support"):
            selected = [r for r in report["rows"] if r["method"] == method and r["role"] == "primary"]
            accepted = [r for r in selected if r["state"] == "accepted"]
            good = [r for r in accepted if r["metrics"]["recovery_fraction"] is not None and r["metrics"]["recovery_fraction"] >= .9 and r["metrics"]["precision_fraction"] >= .9]
            rows.append(dict(mode=mode, method=method, primary_count=len(selected), correct=len(good), wrong_accepted=len(accepted) - len(good),
                no_output=len(selected) - len(accepted), accepted_errors=[dict(query_id=r["query_id"], median_m=r["metrics"]["truth_to_prediction"]["distance_median"]) for r in accepted]))
        cameras.extend(dict(mode=mode, case_id=c["case_id"], rmse_m=c["alignment"]["camera_rmse_m"],
            rotation_min_deg=min(c["alignment"]["orientation_errors_degrees"]), rotation_max_deg=max(c["alignment"]["orientation_errors_degrees"])) for c in report["cases"])
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for i, mode in enumerate(("decoder", "ray_pose")):
        selected = [c for c in cameras if c["mode"] == mode]
        axes[0].bar(np.arange(4) + i * .36, [1000 * c["rmse_m"] for c in selected], .33, label=mode)
    axes[0].set(xticks=np.arange(4) + .18, xticklabels=["p01", "p02", "p03", "p04"], ylabel="Camera-center RMSE after Sim3 (mm)", title="Better global fit, still wrong rods")
    axes[0].legend()
    counts = [r for r in rows if r["method"] == "baseline"]
    bars = axes[1].bar(["Decoder", "Ray pose"], [r["wrong_accepted"] for r in counts], color="#bd4b45")
    axes[1].bar_label(bars)
    axes[1].set(ylim=(0, 4), ylabel="Wrong accepted targets / 8 primary queries", title="Correct outputs: 0 for both")
    categories = [f"{r['case_id']}\n{r['detector']}" for r in real["rows"]]
    bottom = np.zeros(len(categories))
    for label, color in (("correct", "#218163"), ("wrong", "#bd4b45"), ("indeterminate", "#adb8c4")):
        values = np.array([r["unchanged_physical_track_counts"].get(label, 0) for r in real["rows"]])
        axes[2].bar(np.arange(len(categories)), values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[2].set(xticks=np.arange(len(categories)), xticklabels=categories, ylabel="Complete descriptor tracks", title="Cycles still follow wrong bricks")
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("RGB-only camera/track follow-up | same known scenes | G1 NOT passed", fontsize=13)
    figure = ROOT / "docs/experiments/figures/2026-09-22-camera-followup.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    result = dict(state="complete", ray_integrity=ray_integrity, camera_counts=rows, cameras=cameras, real_split_replay=real, control_split_replay=control,
        camera_summary_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, report_source_sha256=digest(Path(__file__)), figure_sha256=digest(figure),
        scope="Development diagnostics; no GT-fed correction, no camera optimizer, no independent rod assets; band splits change fitting subsets only and all physical track scores remain attached to identical graphs")
    write_json(RESULTS / "2026-09-22-camera-followup-summary.json", result)
    print("CAMERA_FOLLOWUP_VERIFIED", ray_integrity["schema_documents"], "schema files; 36 reopened views; 14 identical graph replays", flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
