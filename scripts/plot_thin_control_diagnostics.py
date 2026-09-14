"""Render existing line-control observations and metrics; never changes fits or predictions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "thin-line-controls-v1-20260914"
CASES = ("black-white-thinner", "brick-texture-thinner", "brick-texture-origin")
CASE_LABELS = {"black-white-thinner": "Black/white | thin", "brick-texture-thinner": "Brick | thin", "brick-texture-origin": "Brick | medium"}
METHODS = (
    ("depth_tls_unfiltered", "single_span", "Depth TLS / span"),
    ("depth_tls_unfiltered", "multiview_supported", "Depth TLS / support"),
    ("depth_ransac_unfiltered", "single_span", "Depth RANSAC / span"),
    ("depth_ransac_unfiltered", "multiview_supported", "Depth RANSAC / support"),
    ("rgb_multiview_planes", "single_span", "RGB planes / span"),
    ("rgb_multiview_planes", "multiview_supported", "RGB planes / support"),
)


def observation_figure(run, manifest, output, provenance):
    input_root = ROOT / "data/inputs/thin-pack-v2-20260914r1"
    input_manifest = read_json(input_root / "manifest.json")
    colors = {"whole": "#ffdc40", "gap": "#ff49ee"}
    fig, axes = plt.subplots(3, 5, figsize=(17.5, 18), layout="constrained")
    for row, case in enumerate(CASES):
        job_id = "full--" + case + "-504"
        entry = next(job for job in manifest["jobs"] if job["line_job_id"] == job_id)
        path = run / job_id / "observations.json"
        if sha256(path) != entry["observations_sha256"]:
            raise ValueError("Observation identity mismatch: " + job_id)
        provenance[path] = sha256(path)
        observations = read_json(path)
        group = next(group for group in input_manifest["groups"] if group["case_id"] == case)
        for column, frame in enumerate(group["frames"]):
            axis = axes[row, column]
            image_path = input_root / frame["rgb"]
            if sha256(image_path) != frame["sha256"]:
                raise ValueError("Source image changed")
            with Image.open(image_path) as image:
                rgb = np.array(image.convert("RGB"))
            axis.imshow(rgb, interpolation="nearest")
            all_points = []
            counts = {}
            for target in ("whole", "gap"):
                item = next(item for item in observations[target] if item["frame_id"] == frame["frame_id"])
                points = np.asarray(item["source_uv"], dtype=float).reshape(-1, 2)
                counts[target] = len(points)
                if len(points):
                    all_points.append(points)
                    # Hollow markers leave the rod/background visible. Solid paint would
                    # make a row of bad selections look like a perfectly recovered rod.
                    axis.scatter(points[:, 0], points[:, 1], s=10, linewidths=0.7, facecolors="none", edgecolors=colors[target], label=target)
            if not all_points:
                raise ValueError("No observations to determine a data-only zoom window")
            points = np.concatenate(all_points)
            low = np.maximum(np.floor(points.min(0) - 30), [0, 0])
            high = np.minimum(np.ceil(points.max(0) + 30), [rgb.shape[1] - 1, rgb.shape[0] - 1])
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(high[1], low[1])
            axis.set_aspect("equal")
            axis.tick_params(labelsize=8)
            axis.set_title(frame["frame_id"] + f"\nwhole {counts['whole']} / gap {counts['gap']} points", fontsize=10)
            if column == 0:
                axis.set_ylabel(CASE_LABELS[case] + "\noriginal image y [px]", fontsize=11)
            if row == 2:
                axis.set_xlabel("Original image x [px]", fontsize=9)
    handles = [Line2D([], [], marker="o", markersize=6, markerfacecolor="none", markeredgecolor=colors[target], linestyle="none", label=label) for target, label in (("whole", "Whole-rod candidate points"), ("gap", "Broken-rod candidate points"))]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, fontsize=11)
    fig.suptitle("RGB selection audit | " + run.name + "\nOriginal full-resolution RGB; every saved accepted point shown; zoom uses observations only (no GT)", fontsize=15)
    fig.savefig(output, dpi=170, metadata={"Title": "RGB observation audit", "Run": run.name, "Source hashes": str({str(path): digest for path, digest in provenance.items()})})
    plt.close(fig)


def fmt(value, digits=2):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def mix(color, amount):
    from matplotlib.colors import to_rgb

    base = np.array(to_rgb(color))
    return tuple(1 - min(max(amount, 0), 1) * (1 - base))


def summary_figure(summary, output):
    rows = [row for row in summary["rows"] if np.isclose(row["tolerance_m"], 0.05) and row["kind"] in ("full", "oracle")]
    case_order = {case: index for index, case in enumerate(CASES)}
    jobs = sorted({row["line_job_id"] for row in rows}, key=lambda job: (job.startswith("oracle"), int(job.rsplit("-", 1)[1]), case_order.get(next(row["case_id"] for row in rows if row["line_job_id"] == job), 99)))
    lookup = {(row["line_job_id"], row["target"], row["variant"], row["mode"]): row for row in rows}
    methods = tuple(method for method in METHODS if any(row["variant"] == method[0] for row in rows))
    nrows = len(jobs) * len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(26, max(12, nrows * 0.40 + 2.9)))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.91, bottom=0.12, wspace=0.025)
    for axis, target in zip(axes, ("whole", "gap")):
        headers = ["Input / cameras", "Method", "GT→pred\nmedian cm", "pred→GT\nmedian cm", "Predicted\nlength m", "R5\n%", "P5\n%", "Segments"]
        if target == "gap":
            headers += ["Gap5\n%"]
        cell_text, cell_colors = [], []
        for job in jobs:
            representative = next(row for row in rows if row["line_job_id"] == job)
            camera_name = "KNOWN" if representative["kind"] == "oracle" else "Estimated"
            input_name = CASE_LABELS.get(representative["case_id"], representative["case_id"]) + f"\n{camera_name} / {representative['process_res']}"
            for method_index, (variant, mode, label) in enumerate(methods):
                row = lookup.get((job, target, variant, mode))
                values = [input_name if method_index == 0 else "", label]
                colors = ["#edeafa" if camera_name == "KNOWN" else "#edf2f7", "#f8fafc"]
                if row is None or row.get("state") != "succeeded":
                    values += ["missing/failed"] + ["—"] * (len(headers) - 3)
                    colors += ["#eeeeee"] * (len(headers) - 2)
                    cell_text.append(values)
                    cell_colors.append(colors)
                    continue
                metrics = row["metrics"]
                forward = metrics["truth_to_prediction"]["distance_median"]
                backward = metrics["prediction_to_truth"]["distance_median"]
                predicted_length = metrics["prediction_to_truth"]["source_length"]
                recovery, precision = metrics["recovery_fraction"], metrics["precision_fraction"]
                values += [fmt(None if forward is None else forward * 100), fmt(None if backward is None else backward * 100), fmt(predicted_length), fmt(None if recovery is None else recovery * 100, 1), fmt(None if precision is None else precision * 100, 1), str(row["segment_count"])]
                # Red is positional error, green is actual 5 cm coverage. Length stays
                # neutral: a short wrong line must not look successful just for being short.
                colors += [mix("#ee7777", 0 if forward is None else min(forward / 0.30, 0.70)), mix("#ee7777", 0 if backward is None else min(backward / 0.30, 0.70)), "#f1f5f9", mix("#6abea0", 0 if recovery is None else recovery * 0.80), mix("#6abea0", 0 if precision is None else precision * 0.80), "#f1f5f9"]
                if target == "gap":
                    gap = row.get("gap", {}).get("guarded_interior", {}).get("covered_fraction")
                    text = fmt(None if gap is None else gap * 100, 1)
                    if recovery == 0 and gap == 0:
                        text += " *"
                    values.append(text)
                    colors.append("#e3e3e3" if recovery == 0 else mix("#f7aa56", 0 if gap is None else gap * 0.85))
                cell_text.append(values)
                cell_colors.append(colors)
        widths = [0.24, 0.23, 0.115, 0.115, 0.11, 0.07, 0.07, 0.065]
        if target == "gap":
            widths += [0.085]
        widths = np.asarray(widths) / np.sum(widths)
        table = axis.table(cellText=cell_text, cellColours=cell_colors, colLabels=headers, colWidths=widths, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#e0e5eb")
            cell.set_linewidth(0.35)
            if r == 0:
                cell.set_facecolor("#25344a")
                cell.set_text_props(color="white", weight="bold", fontsize=9)
            elif (r - 1) % len(methods) == 0:
                cell.set_linewidth(1.0)
                cell.set_edgecolor("#65758b")
            if c <= 1 and r > 0:
                cell.set_text_props(ha="left")
        axis.set_axis_off()
        axis.set_title("WHOLE ROD" if target == "whole" else "BROKEN ROD — inspect rod coverage and gap together", fontsize=15, pad=14)
    fig.suptitle("Line controls at 5 cm tolerance | " + summary["run_id"] + "\nSaved metrics only; no refitting, no score changes", fontsize=18)
    fig.text(0.02, 0.075, "R5 = fraction of true rod length within 5 cm of prediction.  P5 = fraction of predicted length within 5 cm of the true rod.\nGT→pred and pred→GT are separate distance medians, not a single accuracy percentage.  span = one continuous interval; support = intervals supported across views.", fontsize=11)
    fig.text(0.02, 0.027, "Gap5 = guarded gap interior within 5 cm of prediction (higher means more unwanted gap coverage).  * R5 = 0: zero gap coverage does NOT demonstrate a preserved gap; the rod itself was missed.\nPurple input cells use known cameras and are diagnostic controls, not ordinary RGB-only reconstruction. Missing/failed values remain unavailable, never zero.", fontsize=11, color="#713918")
    fig.savefig(output, dpi=155, metadata={"Title": "Line-control numerical diagnostics at five centimeters", "Run": summary["run_id"]})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--suffix", default="", help="Optional name suffix for another immutable plot pair")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id) or not re.fullmatch(r"[A-Za-z0-9_-]{0,40}", args.suffix):
        parser.error("Invalid run id or plot suffix")
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    run = ROOT / ".runtime/experiments" / args.run_id
    output = ROOT / "data/evaluation" / args.run_id
    manifest_path, summary_path = run / "manifest.json", output / "summary.json"
    provenance = {manifest_path: sha256(manifest_path), summary_path: sha256(summary_path)}
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    if manifest["state"] != "complete" or summary["state"] != "complete" or summary["fit_manifest_sha256"] != provenance[manifest_path]:
        raise ValueError("Need matching complete fitting and evaluation artifacts")
    suffix = "-" + args.suffix if args.suffix else ""
    observations_path = output / f"rgb-selection-audit{suffix}.png"
    numeric_path = output / f"numeric-controls-5cm{suffix}.png"
    if observations_path.exists() or numeric_path.exists():
        raise FileExistsError("Plots already exist; use --suffix for another plot pair")
    observation_figure(run, manifest, observations_path, provenance)
    summary_figure(summary, numeric_path)
    for path, digest in provenance.items():
        if sha256(path) != digest:
            raise ValueError("An input artifact changed while plotting")
    print(observations_path)
    print(numeric_path)


if __name__ == "__main__":
    main()
