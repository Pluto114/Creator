"""Compare finite outputs before/after support repair on the same RGB crops."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from environment_paths import require_project_environment
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(summary_path, output):
    if output.exists():
        raise FileExistsError("Keep existing comparison figures")
    summary = read(summary_path)
    if summary["state"] != "complete" or summary["row_count"] != 60:
        raise ValueError("Only draw the completed paired experiment")
    run = ROOT / ".runtime/experiments" / summary["run_id"]
    prepared = read(run / "prepared.json")
    index_path = run / "input_index.json"
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen RGB index changed")
    cases = read(index_path)["cases"]
    # 这张图固定旧默认cap8、身份提示0；两种窗口和全部案例都画，别只挑赢家。
    lookup = [{(r["case_id"], r["search_offset_px"]): r for r in rows
               if r["cap"] == 8 and r["identity_offset_px"] == 0}
              for rows in (summary["baseline_rows_rescored"], summary["rows"])]
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, len(cases) * 3.8 + 1.2), squeeze=False)
    try:
        for row_index, case in enumerate(cases):
            frame = next(f for f in case["frames"] if f["view_id"] == "view_+02")
            image_path = ROOT / frame["rgb"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != frame["rgb_sha256"]:
                raise ValueError("RGB changed")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            projection = np.asarray(frame["K_index"]) @ np.asarray(frame["world_to_camera_cv"])[:3]
            for col, (offset, version) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
                ax, result = axes[row_index, col], lookup[version][case["case_id"], offset]
                ax.imshow(rgb, interpolation="nearest")
                guide = np.asarray(frame["guide_xyxy"])
                ax.plot(guide[:, 0], guide[:, 1], "--", color="#f3c354", lw=.7)
                for segment in result["segments"]:
                    pixels = np.c_[np.asarray(segment), np.ones(2)] @ projection.T
                    if np.any(pixels[:, 2] <= 0):
                        raise ValueError("Predicted segment projects behind the camera")
                    xy = pixels[:, :2] / pixels[:, 2, None]
                    ax.plot(xy[:, 0], xy[:, 1], "o-", color="#00efb2", lw=1.8, markersize=2)
                ax.set_xlim(245, 395)
                ax.set_ylim(425, 55)
                ax.set_xticks([])
                ax.set_yticks([])
                title = "Before" if version == 0 else "After"
                ax.set_title(f"{case['case_id']} | {title} | search +{offset}px\n"
                             f"{result['finite_state']}, {len(result['segments'])} segments", fontsize=10)
        fig.suptitle("Final-support repair | paired development replay\n"
                     "All five cases; cap 8, identity +0px, view +02. Green: finite prediction; gold: identity guide.\n"
                     "Same RGB/camera/crop before and after. Cap 16 and other identity offsets are in the full report.",
                     fontsize=12, y=.992)
        fig.tight_layout(rect=(.015, .01, .985, .944), h_pad=2.0, w_pad=1.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170)
    finally:
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/experiments/results/2026-09-20-refit-support.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/figures/2026-09-20-refit-support.png")
    args = parser.parse_args()
    require_project_environment(ROOT)
    render(args.summary, args.output)
