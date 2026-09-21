"""Read-only result tables and fixed-setting RGB overlays for the identity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "cylinder_support")
LABELS = ("Geometry + anchors", "Cylinder + anchors")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(data):
    groups = []
    for method in METHODS:
        for profile in dict.fromkeys(r["profile"] for r in data["rows"]):
            for role in dict.fromkeys(r["query_role"] for r in data["rows"]):
                rows = [r for r in data["rows"] if (r["method"], r["profile"], r["query_role"]) == (method, profile, role)]
                groups.append(dict(method=method, profile=profile, role=role, count=len(rows),
                                   classifications=dict(Counter(r["classification"] for r in rows))))
    return dict(run_id=data["run_id"], row_count=data["row_count"], incremental_seconds=data["elapsed_seconds"], groups=groups)


def plot_counts(datasets, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for ax, (title, data) in zip(axes, datasets):
        profiles = list(dict.fromkeys(r["profile"] for r in data["rows"]))
        x = np.arange(len(profiles))
        for method, label, delta, color in zip(METHODS, LABELS, (-.2, .2), ("#3569b7", "#db8232")):
            rows = [r for r in data["rows"] if r["query_role"] == "primary" and r["method"] == method]
            counts = [sum(r["profile"] == p and r["classification"] == "correct_accept" for r in rows) for p in profiles]
            ax.bar(x + delta, counts, width=.38, color=color, label=label)
        positives = sum(r["method"] == METHODS[0] and r["profile"] == "two_clicks" and r["query_role"] == "primary" and "empty" not in r["classification"] for r in data["rows"])
        ax.set_title(f"{title}: {positives} positive query conditions / method")
        ax.set_xticks(x, profiles, rotation=48, ha="right")
        ax.set_ylim(0, positives + 1)
        ax.set_ylabel("Accepted correct axes (partial curves count too)")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Same annotation budget; repeated conditions are not independent objects.\nBlank, mixed-object and deliberately wrong-object controls are reported separately.", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_queries(data, config, path):
    annotations = read(ROOT / config["annotation_file"])["queries"]
    inputs = ROOT / "data/inputs" / data["parent_run_id"]
    manifest = read(inputs / "manifest.json")
    frames = {c["case_id"]: {f["view_id"]: f for f in c["frames"]} for c in manifest["cases"]}
    fig, axes = plt.subplots(len(annotations), 4, figsize=(13, 2.65 * len(annotations)), layout="constrained")
    for row_index, query in enumerate(annotations):
        for mi, (method, label) in enumerate(zip(METHODS, LABELS)):
            result = next(r for r in data["rows"] if (r["query_id"], r["method"], r["search_offset_px"], r["cap"], r["profile"]) == (query["query_id"], method, 0, 8, "two_clicks"))
            for ai, anchor in enumerate(query["anchors"]):
                ax = axes[row_index, 2 * mi + ai]
                frame = frames[query["case_id"]][anchor["view_id"]]
                rgb = inputs / frame["rgb"]
                if sha(rgb) != anchor["rgb_sha256"]:
                    raise ValueError("Plot anchor belongs to a different image")
                # Every query uses the same crop and setting. No picking the pretty run.
                ax.imshow(Image.open(rgb), interpolation="nearest")
                projection = np.asarray(frame["K_index"]) @ np.asarray(frame["world_to_camera_cv"])[:3]
                for segment in result["segments"]:
                    points = np.c_[segment, np.ones(2)] @ projection.T
                    pixels = points[:, :2] / points[:, 2, None]
                    ax.plot(*pixels.T, color="#00ffff", linewidth=1.5)
                ax.scatter(*anchor["xy"], s=34, marker="x", c="#ffd633", linewidths=1.6)
                ax.set_xlim(295, 351)
                ax.set_ylim(402, 78)
                ax.set_aspect("auto")
                ax.set_xticks([300, 320, 340])
                ax.set_yticks([100, 200, 300, 400])
                ax.tick_params(labelsize=7)
                ax.set_title(f"{query['query_id']} | {anchor['view_id']}\n{label}: {result['state']}", fontsize=8)
    fig.suptitle("All fresh queries, fixed search offset 0 / cap 8 / two clicks\nYellow = input claim; cyan = unchanged finite output. RGB crops stretched for inspection.", fontsize=12)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(output_dir):
    files = [ROOT / f"docs/experiments/results/2026-09-21-foreground-{name}.json" for name in ("development", "new-queries")]
    datasets = [(title, read(path)) for title, path in zip(("Seen development scenes", "New procedural layouts"), files)]
    stats_path = output_dir / "results/2026-09-21-foreground-summary.json"
    counts_path = output_dir / "figures/2026-09-21-foreground-perturbations.png"
    queries_path = output_dir / "figures/2026-09-21-foreground-queries.png"
    if any(p.exists() for p in (stats_path, counts_path, queries_path)):
        raise FileExistsError("Keep earlier summaries and figures")
    for p in (stats_path, counts_path, queries_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    truth = ROOT / "data/eval_gt" / datasets[1][1]["parent_run_id"]
    frames = [f for c in read(truth / "manifest.json")["cases"] for f in read(truth / c["path"])["frames"]]
    ray_check = dict(frames=len(frames), samples=sum(f["blender_ray_check"]["samples"] for f in frames),
        surface_id_mismatches=sum(f["blender_ray_check"]["surface_id_mismatches"] for f in frames),
        max_z_error_m=max(f["blender_ray_check"]["max_z_error_m"] for f in frames),
        max_projection_error_px=max(f["blender_projection_error_px"] for f in frames),
        scope="Independent Blender/Open3D pixel-center geometry checks; not RGB antialiasing precision")
    result = dict(state="summarized", source_sha256={p.name: sha(p) for p in files}, runs=[summarize(d) for _, d in datasets], independent_rays=ray_check,
                  caveat="Correct axis acceptance is not complete curve recovery or topological success. Counts repeat conditions and queries on shared objects.")
    with stats_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot_counts(datasets, counts_path)
    plot_queries(datasets[1][1], read(ROOT / "configs/rod_foreground_new_queries_v1.json"), queries_path)
    print("FOREGROUND_SUMMARY", stats_path, ray_check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/experiments")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.output_dir)
