"""Plot the Blender identity cases and the development guide-guard replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "data/inputs/rod-identity-blender-v1-20260919r3"
DEFAULT_IDENTITY = ROOT / "docs/experiments/results/2026-09-19-blender-identity-r3.json"
DEFAULT_GUIDE = ROOT / "docs/experiments/results/2026-09-19-guide-identity-replay.json"
DEFAULT_OUTPUT = ROOT / "docs/experiments/figures/2026-09-19-blender-identity.png"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def count(summary, field, label):
    return int(summary[field].get(label, 0))


def main(inputs, identity_path, guide_path, output):
    manifest = read_json(inputs / "manifest.json")
    identity = read_json(identity_path)
    guide = read_json(guide_path)
    identity_rows = {
        row["case_id"]: row
        for row in identity["rows"]
        if row["variant"] == "geometry_4"
    }
    cases = {case["case_id"]: case for case in manifest["cases"]}
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9])
    for index, case_id in enumerate(sorted(cases)):
        axis = fig.add_subplot(grid[index // 4, index % 4])
        frame = next(row for row in cases[case_id]["frames"] if row["view_id"] == "view_+00")
        axis.imshow(Image.open(inputs / frame["rgb"]).convert("RGB"))
        guide_xy = np.asarray(frame["guide_xyxy"], float)
        axis.plot(guide_xy[:, 0], guide_xy[:, 1], color="#00d4ff", linewidth=1.2)
        row = identity_rows[case_id]
        title = row["label"].replace("_", " ")
        axis.set_title(f"{case_id}  {title}\ngeometry-4: {row['classification']}", fontsize=9)
        axis.set_axis_off()

    summaries = {
        (row["source_key"], row["variant"]): row for row in guide["summaries"]
    }
    keys = [
        ("analytic_controls", "support_3_views", "analytic / 3 views"),
        ("analytic_controls", "support_4_views", "analytic / 4 views"),
        ("blender_identity", "geometry_3", "Blender / 3 views"),
        ("blender_identity", "geometry_4", "Blender / 4 views"),
    ]
    x = np.arange(len(keys))
    before_false = [
        count(summaries[(source, variant)], "baseline_classification_counts", "false_accept_empty")
        for source, variant, _ in keys
    ]
    after_false = [
        count(summaries[(source, variant)], "guarded_classification_counts", "false_accept_empty")
        for source, variant, _ in keys
    ]
    axis = fig.add_subplot(grid[2, :2])
    axis.bar(x - 0.18, before_false, width=0.36, label="before", color="#c75146")
    axis.bar(x + 0.18, after_false, width=0.36, label="after guide guard", color="#3b82a0")
    axis.set_xticks(x, [row[2] for row in keys], rotation=18, ha="right")
    axis.set_ylabel("empty-target false accepts")
    axis.set_ylim(0, max(before_false) + 0.8)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    before_correct = [
        count(summaries[(source, variant)], "baseline_classification_counts", "correct_accept")
        for source, variant, _ in keys
    ]
    after_correct = [
        count(summaries[(source, variant)], "guarded_classification_counts", "correct_accept")
        for source, variant, _ in keys
    ]
    axis = fig.add_subplot(grid[2, 2:])
    axis.bar(x - 0.18, before_correct, width=0.36, label="before", color="#8e6c46")
    axis.bar(x + 0.18, after_correct, width=0.36, label="after guide guard", color="#55a868")
    axis.set_xticks(x, [row[2] for row in keys], rotation=18, ha="right")
    axis.set_ylabel("correct accepts retained")
    axis.set_ylim(0, max(before_correct) + 1.0)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.02, 0.95,
        "Development replay only; the surface-stripe wrong line remains.",
        transform=axis.transAxes, va="top", fontsize=9,
    )
    fig.suptitle(
        "Rod identity: new paired Blender layouts and coarse-guide regression",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.inputs, args.identity, args.guide, args.output)
