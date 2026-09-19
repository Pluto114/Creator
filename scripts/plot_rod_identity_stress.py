"""Plot the frozen stress results; evaluation overlays never enter inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main(summary_path, output, guide_offset=8):
    if output.exists():
        raise FileExistsError("Keep the existing figure")
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs = ROOT / "data/inputs" / result["run_id"]
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    baseline = {row["case_id"]: row for row in result["rows"] if row["guide_offset_px"] == guide_offset}
    titles = dict(zip([f"s{i:02d}" for i in range(1, 11)],
                      ["Matte rod", "Narrow rod", "Slanted rod", "True gap", "Occlusion",
                       "Surface stripe", "Glossy rod", "Adjacent rods", "Empty target / far rod", "Empty target / near rod"]))
    fig, axes = plt.subplots(2, 5, figsize=(14, 10), constrained_layout=True)
    for ax, case in zip(axes.flat, manifest["cases"]):
        frame = next(frame for frame in case["frames"] if frame["view_id"] == "view_+02")
        row = baseline[case["case_id"]]
        with Image.open(inputs / frame["rgb"]) as image:
            ax.imshow(image)
        guide = np.asarray(frame["guide_xyxy"], float).copy()
        guide[:, 0] += guide_offset
        ax.plot(guide[:, 0], guide[:, 1], "--", color="#e8b33d", lw=1, alpha=.8)
        projection = np.asarray(frame["K_index"]) @ np.asarray(frame["world_to_camera_cv"])[:3]
        for segment in row["segments"]:
            points = np.c_[np.asarray(segment), np.ones(2)] @ projection.T
            pixels = points[:, :2] / points[:, 2, None]
            ax.plot(pixels[:, 0], pixels[:, 1], color="#00e4a5", lw=2)
        ax.set_xlim(245, 395)
        ax.set_ylim(425, 55)
        ax.set_xticks([])
        ax.set_yticks([])
        title = titles[case["case_id"]]
        ax.set_title(f"{case['case_id']}: {title}\n{row['finite_state']}, {row['segment_count']} segments", fontsize=10)
        offsets = [item for item in result["rows"] if item["case_id"] == case["case_id"]]
        notes = [f"+{item['guide_offset_px']} px: {item['guide_classification']}" for item in offsets]
        ax.set_xlabel("\n".join(notes), fontsize=7)
    fig.suptitle("New RGB stress scenes | exact cameras, synthetic screen guides\n"
                 f"Gold: guide at +{guide_offset} px; green: finite output at this offset. Crops are display only.", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/experiments/results/2026-09-19-identity-stress-v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/figures/2026-09-19-identity-stress-v2.png")
    parser.add_argument("--guide-offset", type=int, choices=(0, 8, 16), default=8)
    args = parser.parse_args()
    main(args.summary, args.output, args.guide_offset)
