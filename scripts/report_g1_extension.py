"""Plot frozen reader regressions and estimated-camera counterfactuals honestly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from run_rod_foreground_identity import query_truth
from run_rod_identity_blender import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/experiments/results"
FIGURES = ROOT / "docs/experiments/figures"


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    paths = {key: RESULTS / name for key, name in (
        ("reader", "2026-09-22-readout-split-controls.json"), ("replay", "2026-09-22-point-patch-split.json"),
        ("estimated", "2026-09-22-foreground-estimated-pilot.json"), ("camera", "2026-09-22-camera-counterfactual.json"))}
    data = {key: read_json(path) for key, path in paths.items()}
    for report in data.values():
        for name, sha in report["source_sha256"].items():
            assert digest(ROOT / name) == sha
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summaries = []
    splits = ["prior_development", "prior_fresh_now_regression", "fresh_full_surface", "challenge"]
    for split in splits:
        for reader in ("straight_components", "transverse_split"):
            rows = [r for r in data["reader"]["rows"] if r["split"] == split and r["reader"] == reader]
            summaries.append(dict(split=split, reader=reader, qualified=sum(r["diagnostic_qualified"] for r in rows), count=len(rows),
                maximum_guarded_gap_proximity=max((r["gap"]["guarded_interior"]["covered_fraction"] for r in rows if r["gap"]), default=0)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    for i, reader in enumerate(("straight_components", "transverse_split")):
        selected = [s for s in summaries if s["reader"] == reader]
        bars = axes[0].bar(np.arange(4) + i * .36, [100 * s["qualified"] / s["count"] for s in selected], .32,
                           label="Component v1" if i == 0 else "Transverse split v1")
        axes[0].bar_label(bars, labels=[f"{s['qualified']}/{s['count']}" for s in selected], fontsize=9)
    axes[0].set(xticks=np.arange(4) + .18, xticklabels=["Old 64", "Previous fresh\n(now regression)", "Fresh 64", "Challenges"],
                ylabel="Conditions meeting fixed diagnostic limits (%)", ylim=(0, 122), title="Known failures repaired; new limits remain")
    axes[0].legend(loc="upper right", fontsize=9)
    source = ROOT / ".runtime/experiments" / data["reader"]["run_id"]
    # Use the actual previously failing parallel case, not an invented illustration.
    case = "control-0127"
    rows = [r for r in data["reader"]["rows"] if r["case_id"] == case]
    assert rows[0]["layout"] == "parallel" and rows[0]["radius_m"] == .042
    points = np.load(ROOT / "data/inputs" / data["reader"]["run_id"] / (case + ".npy"), allow_pickle=False)
    axes[1].scatter(points[::9, 0], points[::9, 2], s=2, color=".65", alpha=.35, label="Unchanged surface points")
    for row, color, label in zip(rows, ("#c43d36", "#19825a"), ("Old: false middle axis", "New: two axes")):
        for j, segment in enumerate(row["segments"]):
            segment = np.asarray(segment)
            axes[1].plot(segment[:, 0], segment[:, 2], color=color, lw=2.5, label=label if j == 0 else None)
    axes[1].set(xlabel="World X (m)", ylabel="World Z (m)", title="Preserved regression: noisy parallel tubes")
    axes[1].legend(loc="lower center", fontsize=8)
    fig.suptitle("Shared readout v3 | analytic regression progress, NOT G1 qualification", fontsize=13)
    reader_plot = FIGURES / "2026-09-22-readout-split.png"
    fig.savefig(reader_plot, dpi=150)
    plt.close(fig)
    assert read_json(source / "inference.json")["state"] == "complete"

    rows = data["camera"]["rows"]
    variants = ["estimated", "oracle_intrinsics_only", "oracle_pose_only", "oracle_intrinsics_and_pose"]
    keys = [(r["query_id"], r["method"]) for r in data["estimated"]["rows"]]
    table = np.zeros((len(keys), len(variants)))
    camera_counts = []
    for j, variant in enumerate(variants):
        for i, key in enumerate(keys):
            row = next(r for r in rows if (r["query_id"], r["method"]) == key and r["variant"] == variant)
            if row["state"] == "accepted":
                metric = row["metrics"]
                correct = metric["recovery_fraction"] is not None and metric["recovery_fraction"] >= .9 and metric["precision_fraction"] is not None and metric["precision_fraction"] >= .9
                table[i, j] = 1 if correct else 2
        for method in ("baseline", "cylinder_support"):
            selected = [r for r in rows if r["variant"] == variant and r["method"] == method and r["role"] == "primary"]
            accepted = [r for r in selected if r["state"] == "accepted"]
            correct = [r for r in accepted if r["metrics"]["recovery_fraction"] is not None and r["metrics"]["recovery_fraction"] >= .9 and r["metrics"]["precision_fraction"] is not None and r["metrics"]["precision_fraction"] >= .9]
            camera_counts.append(dict(variant=variant, method=method, primary_count=len(selected), accepted_count=len(accepted),
                                      correct_count=len(correct), wrong_count=len(accepted) - len(correct)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.7), gridspec_kw={"width_ratios": [1.7, 1, 1]}, layout="constrained")
    axes[0].imshow(table, aspect="auto", cmap=ListedColormap(["#dce1e7", "#19825a", "#c43d36"]), vmin=0, vmax=2)
    axes[0].set(xticks=np.arange(4), xticklabels=["Model", "Known K", "Known pose", "Known both"], yticks=np.arange(len(keys)),
                yticklabels=[q + (" / plain" if m == "baseline" else " / cylinder") for q, m in keys], title="Same clicks, same RGB observation pool")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(handles=[Patch(color="#dce1e7", label="No curve"), Patch(color="#19825a", label="R/P >= 90% at 25 mm"), Patch(color="#c43d36", label="Accepted but inaccurate")],
                   loc="upper center", bbox_to_anchor=(.5, -.13), fontsize=8)
    run_folder = ROOT / ".runtime/experiments" / data["estimated"]["run_id"]
    protocol = read_json(run_folder / "protocol.json")
    queries = {q["query_id"]: q for q in read_json(ROOT / protocol["evaluation_queries_file"])["evaluation_queries"]}
    truth_root = ROOT / "data/eval_gt" / protocol["source_run_id"]
    target_lookup = {c["case_id"]: read_json(truth_root / c["path"]) for c in read_json(truth_root / "manifest.json")["cases"]}
    accepted_errors = []
    for ax, query_id in zip(axes[1:], ("p01_A", "p03_B")):
        row = next(r for r in data["estimated"]["rows"] if r["query_id"] == query_id and r["method"] == "baseline")
        target = query_truth(queries[query_id], target_lookup[row["case_id"]], truth_root)
        true_segment = np.asarray(target["target"]["endpoints"])
        ax.plot(true_segment[:, 1], true_segment[:, 2], "--", color="#14375e", lw=2, label="True target")
        for i, segment in enumerate(row["segments"]):
            segment = np.asarray(segment)
            ax.plot(segment[:, 1], segment[:, 2], color="#c43d36", lw=2, label="Accepted model-camera curve" if i == 0 else None)
        error = row["metrics"]["truth_to_prediction"]["distance_median"]
        accepted_errors.append(dict(query_id=query_id, truth_to_prediction_median_m=error, metrics=row["metrics"]))
        ax.set(xlabel="World Y (m)", ylabel="World Z (m)", title=f"{query_id}: median error {error * 100:.1f} cm")
        ax.legend(loc="upper center", fontsize=8)
    fig.suptitle("Camera-only Sim3 held fixed | camera substitutions are privileged DIAGNOSTICS\nNormal model-camera outputs remain wrong; no oracle correction is published as a patch", fontsize=12)
    camera_plot = FIGURES / "2026-09-22-camera-transfer.png"
    fig.savefig(camera_plot, dpi=150)
    plt.close(fig)
    output = dict(state="complete", input_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in paths.values()},
                  readout_summary=summaries, camera_counts=camera_counts, accepted_model_camera_errors=accepted_errors,
                  figures={p.relative_to(ROOT).as_posix(): digest(p) for p in (reader_plot, camera_plot)})
    previous_path = RESULTS / "2026-09-21-foreground-new-queries.json"
    previous = [r for r in read_json(previous_path)["rows"] if r["search_offset_px"] == 0 and r["cap"] == 8 and r["profile"] == "two_clicks"]
    deltas = []
    for row in rows:
        if row["variant"] != "oracle_intrinsics_and_pose":
            continue
        old = next(r for r in previous if r["query_id"] == row["query_id"] and r["method"] == row["method"])
        assert old["state"] == row["state"]
        for key in ("recovery_fraction", "precision_fraction", "false_predicted_length"):
            a, b = row["metrics"][key], old["curve_metrics"][key]
            if a is None or b is None:
                assert a == b
            else:
                deltas.append(abs(a - b))
    assert len(previous) == 18
    output["full_oracle_parity"] = dict(previous_sha256=digest(previous_path), query_method_count=len(previous),
                                       states_equal=True, maximum_core_metric_difference=max(deltas))
    (RESULTS / "2026-09-22-g1-extension-summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(readout_summary=summaries, camera_counts=camera_counts), indent=2))


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
