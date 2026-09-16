"""Read-only held-out overlays: one improved case beside one failed case."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "rod-evidence-dense-v1-20260916"
HELDOUT_ID = "thin-heldout-v1-20260916"
FRAME_ID = "view_+22p5"
CASES = ("brick-texture-thinner", "brick-texture-origin")
VARIANTS = ("legacy_occupancy", "paired_full")
OUTPUT = ROOT / "docs/experiments/assets/2026-09-16-rod-overlay.png"


def project(points, camera):
    points = np.asarray(points, float)
    transform = np.asarray(camera["world_to_camera_cv"])
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    if np.any((xyz[..., 2] <= camera["clip_start"]) | (xyz[..., 2] >= camera["clip_end"])):
        raise ValueError("Overlay candidate crosses camera depth cutoffs")
    uv = xyz @ np.asarray(camera["K_index"]).T
    return uv[..., :2] / uv[..., 2, None]


def load_fixed_sources():
    run = ROOT / ".runtime/experiments" / RUN_ID
    fit_path = run / "manifest.json"
    fit = read_json(fit_path)
    if fit["state"] != "complete" or fit["gt_read_in_fit"] or fit["heldout_read_in_fit"]:
        raise ValueError("Expected completed RGB fit with evaluation-only held-out frames")
    evaluation_path = ROOT / "data/evaluation" / RUN_ID / "summary.json"
    evaluation = read_json(evaluation_path)
    if evaluation["state"] != "complete" or evaluation["fit_manifest_sha256"] != sha256(fit_path):
        raise ValueError("Evaluation must match the frozen completed fit")
    heldout = ROOT / "data/eval_gt" / HELDOUT_ID
    status = read_json(heldout / "status.json")
    if (
        status["state"] != "complete"
        or sha256(heldout / "artifact_hashes.json") != status["artifact_hashes_sha256"]
    ):
        raise ValueError("Held-out export integrity failure")
    hashes = read_json(heldout / "artifact_hashes.json")
    manifest_path = heldout / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest["eval_only"] or evaluation["heldout_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Held-out identity differs from evaluation")
    consumed = [fit_path, evaluation_path, manifest_path, heldout / "artifact_hashes.json"]
    rows = []
    for case in CASES:
        job_id = "oracle_camera--" + case + "-504"
        entry = next(job for job in fit["jobs"] if job["candidate_job_id"] == job_id)
        path = run / job_id / "candidates.json"
        if sha256(path) != entry["candidates_sha256"] or entry["camera_mode"] != "oracle_camera":
            raise ValueError("Expected unchanged privileged camera control")
        candidates = read_json(path)
        consumed.append(path)
        group = next(group for group in manifest["groups"] if group["case_id"] == case)
        frame = next(frame for frame in group["frames"] if frame["frame_id"] == FRAME_ID)
        artifacts = {}
        for key in ("rgb", "camera", "curve_visibility"):
            artifact = heldout / frame[key]
            digest = sha256(artifact)
            if digest != frame[key + "_sha256"] or digest != hashes[frame[key]]["sha256"]:
                raise ValueError("Held-out frame identity changed")
            artifacts[key] = artifact
            consumed.append(artifact)
        camera = read_json(artifacts["camera"])
        with Image.open(artifacts["rgb"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        labels = dict(np.load(artifacts["curve_visibility"], allow_pickle=False))
        rod = labels["rod_id"] == 6
        if not np.any(rod & labels["intentional_gap"]):
            raise ValueError("Expected the physical gapped rod")
        overlays, metrics = {}, {}
        for variant in VARIANTS:
            candidate = candidates["targets"]["gap"]["variants"][variant]
            segments = np.asarray(candidate["segments"], float).reshape(-1, 2, 3)
            overlays[variant] = project(segments, camera) if len(segments) else np.empty((0, 2, 2))
            matching = [
                row
                for row in evaluation["heldout_rows"]
                if row["candidate_job_id"] == job_id
                and row["target"] == "gap"
                and row["variant"] == variant
                and row["frame_id"] == FRAME_ID
                and row["tolerance_px"] == 2
            ]
            if len(matching) != 1:
                raise ValueError("Missing or duplicated frozen 2px metric")
            metrics[variant] = matching[0]
        # 这是评测展示框：所有真端点和两种预测端点一起入框，绝不把难看的延长/缺端裁掉。
        points = np.concatenate(
            [labels["uv_index"][rod]] + [value.reshape(-1, 2) for value in overlays.values()]
        )
        low = np.floor(points.min(0) - [180, 42]).astype(int)
        high = np.ceil(points.max(0) + [180, 42]).astype(int)
        low = np.maximum(low, [0, 0])
        high = np.minimum(high, camera["size_wh"])
        crop = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
        for prediction in overlays.values():
            if len(prediction) and not np.all(
                (prediction >= np.array(crop[:2]) - 0.5) & (prediction <= np.array(crop[2:]) - 0.5)
            ):
                raise ValueError(
                    "An overlay endpoint would be hidden outside the shared display crop"
                )
        rows.append(
            {
                "case": case,
                "rgb": rgb,
                "labels": labels,
                "crop": crop,
                "overlays": overlays,
                "metrics": metrics,
            }
        )
    identities = {path.relative_to(ROOT).as_posix(): sha256(path) for path in consumed}
    return rows, identities


def truth_overlay(axis, labels):
    selected = labels["rod_id"] == 6
    for sid in (6, 7):
        points = labels["uv_index"][selected & (labels["segment_surface_id"] == sid)]
        axis.plot(
            points[:, 0],
            points[:, 1],
            color="#93cce7",
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            alpha=0.8,
            zorder=3,
        )
        for endpoint in points[[0, -1]]:
            axis.plot(
                [endpoint[0] - 6, endpoint[0] + 6],
                [endpoint[1]] * 2,
                color="#b1dcef",
                linewidth=0.85,
                alpha=0.9,
                zorder=3,
            )
    gap = labels["uv_index"][selected & labels["intentional_gap"]]
    axis.plot(
        gap[:, 0],
        gap[:, 1],
        color="#b9c0c7",
        linewidth=0.75,
        linestyle=(0, (2, 4)),
        alpha=0.7,
        zorder=2,
    )
    middle = gap[len(gap) // 2]
    axis.annotate(
        "True empty gap",
        xy=middle,
        xytext=(middle[0] + 54, middle[1] - 7),
        fontsize=8.3,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "#14232f", "alpha": 0.8, "edgecolor": "none", "pad": 3},
        arrowprops={"arrowstyle": "-", "color": "#d8dee3", "lw": 0.8},
        zorder=6,
    )


def main():
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    rows, identities = load_fixed_sources()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fbfcfd"})
    figure, axes = plt.subplots(2, 3, figsize=(12, 17.6))
    figure.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.13, hspace=0.29, wspace=0.08)
    figure.suptitle(
        "A preserved gap is not the whole story", fontsize=17, fontweight="bold", y=0.98
    )
    figure.text(
        0.5,
        0.959,
        "Known cameras  |  Same asset  |  Evaluation-only view +22.5 degrees",
        ha="center",
        fontsize=11,
        color="#384653",
    )
    legend = [
        Line2D(
            [0],
            [0],
            color="#739fba",
            lw=1,
            ls="--",
            label="GT axis / ends (including hidden ends)",
        ),
        Line2D(
            [0],
            [0],
            color="#ed843e",
            lw=1.6,
            marker="o",
            mfc="none",
            label="Legacy candidate / endpoints",
        ),
        Line2D(
            [0],
            [0],
            color="#e661bb",
            lw=1.6,
            marker="o",
            mfc="none",
            label="Paired-full candidate / endpoints",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=3,
    )
    titles = ("A  Original held-out RGB", "B  Legacy occupancy", "C  Paired full")
    colors = {"legacy_occupancy": "#ff914d", "paired_full": "#ff72ce"}
    crop_records = []
    for index, row in enumerate(rows):
        left, top, right, bottom = row["crop"]
        crop_records.append({"case_id": row["case"], "box_xyxy": row["crop"]})
        for column, axis in enumerate(axes[index]):
            axis.imshow(
                row["rgb"][top:bottom, left:right],
                extent=(left - 0.5, right - 0.5, bottom - 0.5, top - 0.5),
                interpolation="nearest",
            )
            axis.set_xlim(left - 0.5, right - 0.5)
            axis.set_ylim(bottom - 0.5, top - 0.5)
            axis.set_axis_off()
            axis.set_title(titles[column], fontsize=11, pad=10, loc="left")
            if not column:
                axis.text(
                    0.5,
                    -0.035,
                    "Unaltered RGB; identical display crop",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.7,
                    color="#384653",
                )
                continue
            variant = VARIANTS[column - 1]
            truth_overlay(axis, row["labels"])
            for segment in row["overlays"][variant]:
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=colors[variant],
                    lw=1.7,
                    alpha=0.97,
                    marker="o",
                    markersize=4.2,
                    markerfacecolor="none",
                    markeredgewidth=1.1,
                    zorder=5,
                )
            metrics = row["metrics"][variant]["metrics"]
            recall = metrics["visible_gt_to_candidate"]["recall_fraction"]
            gap = metrics["gap_clear_guarded_interior_to_candidate"]["false_coverage_fraction"]
            worst = metrics["visible_gt_to_candidate"]["distance_max_px"]
            axis.text(
                0.5,
                -0.035,
                f"Visible-GT recall @ 2 px: {100 * recall:.1f}%   |   Gap proximity: {100 * gap:.1f}%\n"
                f"Worst visible-GT miss: {worst:.1f} px",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8.7,
                linespacing=1.6,
                color="#253440",
            )
    # 行标题明确保留失败例；两行都展示完整杆端，不能只给缺口局部美化图。
    figure.text(
        0.05,
        0.904,
        "THIN  /  Gap preserved, but the finite endpoints are still imperfect",
        fontsize=11.8,
        fontweight="bold",
        color="#253440",
    )
    figure.text(
        0.05,
        0.489,
        "MEDIUM  /  Projection failure: paired-full recall is 0% at 2 px",
        fontsize=11.8,
        fontweight="bold",
        color="#95384c",
    )
    figure.text(
        0.05,
        0.077,
        "Privileged camera control, not the ordinary estimated-camera pipeline. Dense observations; both candidates frozen before held-out evaluation.\n"
        "Gap proximity is false proximity over clear gap-interior samples, with a 2 px endpoint guard; 0% alone does not establish correct geometry.\n"
        "Display crops use all GT and candidate endpoints plus fixed margins, only for this figure. No display crop or held-out image entered fitting.",
        fontsize=8.7,
        color="#384653",
        va="top",
        linespacing=1.6,
    )
    fit_key = f".runtime/experiments/{RUN_ID}/manifest.json"
    evaluation_key = f"data/evaluation/{RUN_ID}/summary.json"
    heldout_key = f"data/eval_gt/{HELDOUT_ID}/manifest.json"
    figure.text(
        0.05,
        0.025,
        f"Source: {RUN_ID}  |  oracle_camera  |  brick-texture-thinner / brick-texture-origin  |  physical rod 6\n"
        f"SHA256 prefixes  fit {identities[fit_key][:12]}  /  evaluation {identities[evaluation_key][:12]}  /  held-out {identities[heldout_key][:12]}  |  Full identities in adjacent JSON.",
        fontsize=7.8,
        color="#5d6b76",
        va="top",
        linespacing=1.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    for relative, digest in identities.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError("Read-only plotting source changed: " + relative)
    metadata = {
        "scope": "read_only_evaluation_figure_not_method_input_or_parameter_selection",
        "run_id": RUN_ID,
        "camera_mode": "oracle_camera",
        "heldout_bundle": HELDOUT_ID,
        "heldout_frame": FRAME_ID,
        "variants": list(VARIANTS),
        "target": "gap",
        "rod_id": 6,
        "cases": list(CASES),
        "crop_rule": "union_of_all_physical_rod_GT_samples_and_both_candidates_endpoints_plus_180px_horizontal_42px_vertical_margin_clipped_to_image",
        "crops": crop_records,
        "source_sha256": identities,
        "plot_script_sha256": sha256(Path(__file__)),
        "image_sha256": sha256(OUTPUT),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("OVERLAY_READY", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
