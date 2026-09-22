"""Read frozen results, expose reader failures, and draw paired diagnostic figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from check_common_readout_ceiling import tube
from run_rod_identity_blender import digest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/experiments/results"
FIGURES = ROOT / "docs/experiments/figures"


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {name: RESULTS / file for name, file in (
        ("controls", "2026-09-22-common-readout-components.json"),
        ("local_links", "2026-09-22-point-patch-r2.json"),
        ("straight_components", "2026-09-22-point-patch-components.json"),
    )}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in files.items()}
    for result in data.values():
        for name, sha in result["source_sha256"].items():
            if digest(ROOT / name) != sha:
                raise ValueError("Frozen result source changed: " + name)
    summary, mistakes = [], []
    for split in ("development", "fresh_analytic"):
        for method in ("local_links", "straight_components"):
            rows = [r for r in data["controls"]["rows"] if r["split"] == split and r["method"] == method]
            wrong = [r for r in rows if r["metrics"]["precision_fraction"] is not None and r["metrics"]["precision_fraction"] < .9]
            summary.append(dict(split=split, method=method, count=len(rows),
                recovery_at_least_90_count=sum(r["metrics"]["recovery_fraction"] >= .9 for r in rows),
                mean_recovery=float(np.mean([r["metrics"]["recovery_fraction"] for r in rows])),
                nonempty_precision_below_90_count=len(wrong),
                maximum_guarded_gap_proximity=max(r.get("gap", {}).get("guarded_interior", {}).get("covered_fraction", 0) for r in rows)))
            mistakes.extend(wrong)

    # 这里追查一个很烦的数值反例：格子明明连续，投影后再 floor 却凭空分段。
    # 只审计，不偷偷改变刚跑完的读取器。修复必须另开版本，让两边一起重跑。
    points = tube([[.013, .017, .011], [.013, .017, 1.011]], 0., 0., 22092026)
    voxel = .02
    grid = np.unique(np.floor(points / voxel).astype(np.int64), axis=0)
    cloud = (grid.astype(float) + .5) * voxel
    mean = cloud.mean(axis=0)
    _, vectors = np.linalg.eigh((cloud - mean).T @ (cloud - mean))
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projection = np.sort((cloud - mean) @ axis)
    occupied = np.unique(np.floor(projection / voxel).astype(np.int64))
    exact_bins = np.unique(np.floor(np.round(projection / voxel, 10)).astype(np.int64))
    bin_audit = dict(scope="read-only numerical diagnosis; rounding is not an adopted reader change",
        occupied_voxels=len(grid), voxel_m=voxel,
        all_world_grid_steps_are_one=bool(np.all(np.diff(grid[:, 2]) == 1)),
        maximum_projected_step_m=float(np.max(np.diff(projection))),
        axial_bin_holes=int(np.sum(np.diff(occupied) > 1)),
        diagnostic_rounded_bin_holes=int(np.sum(np.diff(exact_bins) > 1)))
    assert bin_audit["all_world_grid_steps_are_one"] and bin_audit["axial_bin_holes"] > 0

    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    for i, split in enumerate(("development", "fresh_analytic")):
        entries = [r for r in summary if r["split"] == split]
        bars = axes[0].bar(np.arange(2) + i * .36, [r["recovery_at_least_90_count"] for r in entries], width=.32,
                           label="Earlier controls" if i == 0 else "Fresh analytic controls")
        axes[0].bar_label(bars, fmt="%d / 64")
    axes[0].set(xticks=np.arange(2) + .18, xticklabels=["Local links", "Straight components"], ylim=(0, 72),
                ylabel="Conditions with recovery >= 90%", title="A better ceiling is not enough")
    axes[0].legend(loc="upper left")
    failure = next(r for r in mistakes if r["split"] == "fresh_analytic")
    origin, length = np.array([.028, .033, -.027]), 1.25
    truth = [[origin, origin + [0, 0, length]], [origin + [.16, 0, 0], origin + [.16, 0, length]]]
    for i, segment in enumerate(truth):
        cloud = tube(segment, failure["radius_m"], failure["noise_sigma_m"], 22092027 + i)
        axes[1].scatter(cloud[::6, 0], cloud[::6, 2], s=2, alpha=.22, color="#527aab", label="Tube surface samples" if i == 0 else None)
        segment = np.asarray(segment)
        axes[1].plot(segment[:, 0], segment[:, 2], "--", color="#14375e", label="True axes" if i == 0 else None)
    for i, segment in enumerate(failure["segments"]):
        segment = np.asarray(segment)
        axes[1].plot(segment[:, 0], segment[:, 2], color="#c53932", lw=3, label="Readout: false middle axis" if i == 0 else None)
    axes[1].set(xlabel="World X (m)", ylabel="World Z (m)", title="Fresh parallel control: precision = 0%, recovery = 0%")
    axes[1].legend(loc="lower center", fontsize=9)
    fig.suptitle("Common-reader qualification FAILED | analytic controls at 25 mm tolerance", fontsize=13)
    ceiling_plot = FIGURES / "2026-09-22-readout-ceiling.png"
    fig.savefig(ceiling_plot, dpi=150)
    plt.close(fig)

    jobs = [j["job"]["candidate_job_id"] for j in data["local_links"]["jobs"]]
    labels = [j.replace("estimated--", "Est / ").replace("oracle_camera--", "Oracle / ")
              .replace("black-white-thinner", "B&W thin").replace("brick-texture-thinner", "Brick thin")
              .replace("brick-texture-origin", "Brick origin").replace("-504", "\n504").replace("-756", "\n756") for j in jobs]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), sharex=True, sharey=True, layout="constrained")
    primary = []
    for column, method in enumerate(("local_links", "straight_components")):
        for row, fraction in enumerate((.0025, .005)):
            ax = axes[row, column]
            for variant, marker, color in (("base", "o", "#527aab"), ("candidate", "x", "#c53932")):
                values = []
                for job in jobs:
                    item = next(r for r in data[method]["rows"] if r["job_id"] == job and r["fraction"] == fraction and r["variant"] == variant and r["tolerance_m"] == .05)
                    primary.append({"reader": method, **item})
                    values.append(item["metrics"]["recovery_fraction"] * 100)
                ax.plot(np.arange(len(jobs)), values, marker=marker, color=color, label=variant, linestyle="none", markersize=8)
            ax.axvline(3.5, color=".75", lw=1)
            ax.set(title=f"{method.replace('_', ' ')} | voxel = {fraction:g} x camera span", ylim=(-3, 104),
                   xticks=np.arange(len(jobs)), xticklabels=labels)
            ax.tick_params(axis="x", labelrotation=25, labelsize=9)
            if column == 0:
                ax.set_ylabel("Target recovery at 50 mm (%)")
            ax.legend(loc="upper left")
    fig.suptitle("Same 7 historical model outputs | complete base vs complete candidate\nReader-sensitive diagnostics; NOT qualified G1 efficacy evidence", fontsize=13)
    replay_plot = FIGURES / "2026-09-22-point-patch-paired.png"
    fig.savefig(replay_plot, dpi=150)
    plt.close(fig)

    output = dict(state="complete", input_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in files.values()},
        ceiling_summary=summary, false_axis_controls=mistakes, numerical_bin_audit=bin_audit,
        primary_50mm_rows=primary, figures={p.relative_to(ROOT).as_posix(): digest(p) for p in (ceiling_plot, replay_plot)},
        conclusion="Both common readers remain unqualified. This is engineering and failure analysis, not independent-object efficacy.")
    # This derived figure/audit index is reproducible; frozen measurement inputs stay read-only.
    (RESULTS / "2026-09-22-point-patch-audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(ceiling_summary=summary, numerical_bin_audit=bin_audit), indent=2))


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run()
