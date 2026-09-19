"""Plot frozen multi-view candidate controls and evaluation classifications."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "rod-multiview-candidates-v1-20260919r1"
SUMMARY = ROOT / "data/evaluation" / RUN_ID / "summary.json"
INPUTS = ROOT / "data/inputs" / RUN_ID
OUTPUT = ROOT / "docs/experiments/assets/2026-09-19-multiview-candidates.png"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(output=OUTPUT):
    if output.exists():
        raise FileExistsError(f"Keep existing figure: {output}")
    summary = read_json(SUMMARY)
    manifest = read_json(INPUTS / "manifest.json")
    examples = [
        ("e02_target_plus_new_clutter", "True line plus view-specific clutter"),
        ("e03_consistent_surface_stripe", "Wrong surface stripe, consistent in all views"),
        ("e04_two_new_consistent_lines", "Two equally consistent lines"),
    ]
    figure = plt.figure(figsize=(15, 10))
    grid = figure.add_gridspec(
        4, 5, height_ratios=[1, 1, 1, 1.25],
        left=.12, right=.985, bottom=.08, top=.89, hspace=.05, wspace=.08,
    )
    for row_index, (case_id, label) in enumerate(examples):
        case = next(item for item in manifest["cases"] if item["case_id"] == case_id)
        for view_index, frame in enumerate(case["frames"]):
            axis = figure.add_subplot(grid[row_index, view_index])
            image = np.asarray(Image.open(INPUTS / frame["rgb"]).convert("L"))
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            guide = np.asarray(frame["guide_xyxy"])
            axis.plot(guide[:, 0], guide[:, 1], color="#2ca25f", linestyle="--", linewidth=1.2)
            center_x = float(np.mean(guide[:, 0]))
            axis.set_xlim(center_x - 26, center_x + 26)
            axis.set_ylim(210, 28)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"view {view_index}")
            if view_index == 0:
                axis.set_ylabel(label.replace(", ", "\n"), fontsize=9, rotation=0, labelpad=62, va="center")
    matrix_axis = figure.add_subplot(grid[3, :])
    case_ids = [f"e0{index}" for index in range(1, 7)]
    variants = ["support_3_views", "support_4_views"]
    labels = {
        "correct_accept": ("correct", "#2ca25f"),
        "wrong_line_accept": ("wrong line", "#d95f0e"),
        "false_accept_empty": ("false object", "#b2182b"),
        "safe_identity_ambiguity": ("ambiguous", "#756bb1"),
        "target_refused": ("missed target", "#969696"),
        "safe_empty_refusal": ("safe refusal", "#67a9cf"),
    }
    rows = {(row["case_id"][:3], row["variant"]): row for row in summary["rows"] if row["split"] == "evaluation"}
    for x, case_id in enumerate(case_ids):
        for y, variant in enumerate(variants):
            row = rows[(case_id, variant)]
            text, color = labels[row["classification"]]
            matrix_axis.add_patch(plt.Rectangle((x - .48, y - .42), .96, .84, color=color, alpha=.88))
            matrix_axis.text(x, y, f"{text}\n{row['support_view_count']} views", ha="center", va="center", color="white", fontsize=9, weight="bold")
    matrix_axis.set_xlim(-.5, len(case_ids) - .5)
    matrix_axis.set_ylim(1.5, -.5)
    matrix_axis.set_yticks(range(2), ["minimum 3 views", "minimum 4 views"])
    matrix_axis.set_xticks(
        range(6),
        ["clean\nsloped", "target +\nclutter", "surface\nstripe", "two\nlines", "visible in\n3 views", "consistent\nbackground"],
    )
    matrix_axis.tick_params(length=0)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)
    figure.suptitle(
        "Cross-view consistency removes incoherent clutter, but cannot certify physical identity",
        fontsize=16,
        weight="bold",
        y=.985,
    )
    figure.text(
        .5,
        .945,
        "Frozen evaluation split; exact cameras; green dashed lines are coarse guides, not truth. Infinite-line identity only.",
        ha="center",
        fontsize=11,
        color="#425466",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.output)
