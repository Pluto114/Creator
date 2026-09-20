"""Show every case with fixed crops and settings, including failures."""
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


def render(summary_path, output, strict_path=None):
    if output.exists():
        raise FileExistsError("Keep all earlier comparison figures")
    summary = read(summary_path)
    if summary["state"] != "complete":
        raise ValueError("Only completed scores can be drawn")
    run = ROOT / ".runtime/experiments" / summary["run_id"]
    prepared = read(run / "prepared.json")
    if "baseline_run_id" in prepared:
        baseline = ROOT / ".runtime/experiments" / prepared["baseline_run_id"]
        cases = read(baseline / "input_index.json")["cases"]
        image_root = ROOT
    else:
        image_root = ROOT / "data/inputs" / summary["run_id"]
        manifest_path = image_root / "manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != prepared["input_manifest_sha256"]:
            raise ValueError("RGB manifest changed")
        cases = read(manifest_path)["cases"]
    columns = [("Baseline, cap 8", summary["baseline_rows_rescored"], 8)]
    if strict_path:
        strict = read(strict_path)
        if strict["baseline_run_id"] != summary["baseline_run_id"]:
            raise ValueError("Strict and support versions have different baselines")
        columns.append(("Strict all views, cap 8", strict["rows"], 8))
    columns.extend([("View support, cap 8", summary["rows"], 8), ("View support, cap 16", summary["rows"], 16)])
    lookups = [{r["case_id"]: r for r in rows if r["search_offset_px"] == 0 and r["identity_offset_px"] == 0 and r["cap"] == cap}
               for _, rows, cap in columns]
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(3.1 * len(columns), 3.3 * len(cases) + 1), squeeze=False)
    try:
        for i, case in enumerate(cases):
            frame = case["frames"][2]  # Fixed third view, not a per-case best-looking image.
            path = image_root / frame["rgb"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame["rgb_sha256"]:
                raise ValueError("Rendered RGB changed")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            projection = np.asarray(frame["K_index"]) @ np.asarray(frame["world_to_camera_cv"])[:3]
            for j, (title, _, _) in enumerate(columns):
                ax, result = axes[i, j], lookups[j][case["case_id"]]
                ax.imshow(rgb, interpolation="nearest")
                guide = np.asarray(frame["guide_xyxy"])
                ax.plot(guide[:, 0], guide[:, 1], "--", color="#f3c354", lw=.6)
                for segment in result["segments"]:
                    projected = np.c_[np.asarray(segment), np.ones(2)] @ projection.T
                    if np.any(projected[:, 2] <= 0):
                        raise ValueError("Prediction behind the camera")
                    xy = projected[:, :2] / projected[:, 2, None]
                    ax.plot(xy[:, 0], xy[:, 1], "o-", color="#00efb2", lw=1.6, markersize=2)
                ax.set_xlim(245, 395)
                ax.set_ylim(435, 45)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{case['case_id']} | {title}\n{result['finite_state']}, {len(result['segments'])} segments", fontsize=9)
        scope = "Old development scenes" if strict_path else "Frozen new layouts, same cylinder family"
        fig.suptitle(f"{scope} | all cases shown\nSearch +0px, identity +0px, third view. Green: finite prediction; gold: guide.\nSame RGB/camera/crop across each row; full offset results remain in JSON.", fontsize=11, y=.995)
        fig.tight_layout(rect=(.01, .01, .99, .946), h_pad=1.7, w_pad=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=155)
    finally:
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--strict", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_project_environment(ROOT)
    render(args.summary, args.output, args.strict)
