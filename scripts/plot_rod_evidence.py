"""Plot two fixed, completed rod-evidence evaluations without rerunning any method."""

import json
from pathlib import Path

import numpy as np
from thin_pack_gt import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "sparse": "rod-evidence-v1-20260916",
    "dense": "rod-evidence-dense-v1-20260916",
}
OUTPUT = ROOT / "docs/experiments/assets/2026-09-16-rod-evidence.png"
TOLERANCE_M = 0.05
ORACLE_CASES = [
    ("black-white-thinner", "BW thin"),
    ("brick-texture-thinner", "Brick thin"),
    ("brick-texture-origin", "Brick medium"),
]
ESTIMATED_JOBS = [
    ("black-white-thinner", 504, "BW thin 504"),
    ("brick-texture-thinner", 504, "Brick thin 504"),
    ("brick-texture-origin", 504, "Brick medium 504"),
    ("brick-texture-thinner", 756, "Brick thin 756"),
]
VARIANTS = [
    ("legacy_occupancy", "Legacy"),
    ("paired_occupancy", "Paired occupancy"),
    ("paired_geometry", "Paired + geometry"),
    ("paired_full", "Paired + negative veto"),
]


def evaluation_index(summary):
    """Use fixed identities; duplicates or missing rows must never disappear silently."""
    if summary["state"] != "complete":
        raise ValueError("Only a complete evaluation may be plotted")
    result = {}
    for row in summary["rows"]:
        if abs(row["tolerance_m"] - TOLERANCE_M) > 1e-10:
            continue
        key = (
            row["camera_mode"],
            row["case_id"],
            row["process_res"],
            row["target"],
            row["variant"],
        )
        if key in result:
            raise ValueError(f"Duplicate frozen metric identity: {key}")
        result[key] = row
    return result


def percent(value):
    if value is None:
        return float("nan")
    value = float(value)
    if not np.isfinite(value) or not -1e-8 <= value <= 1 + 1e-8:
        raise ValueError("Expected a fraction or explicit null")
    return 100 * np.clip(value, 0, 1)


def recovery(row):
    return percent(row["metrics"]["recovery_fraction"])


def gap_proximity(row):
    return percent(row["gap"]["guarded_interior"]["covered_fraction"])


def paired_bars(axis, labels, first, second, names, colors, group_size):
    """Each series item contains its metric value and the original row state."""
    positions = np.arange(len(labels), dtype=float)
    height = 0.32
    for offset, values, name, color in zip((-0.18, 0.18), (first, second), names, colors):
        for index, (value, row) in enumerate(values):
            y = positions[index] + offset
            refused = row["state"] != "accepted"
            if np.isfinite(value):
                axis.barh(
                    y,
                    value,
                    height=height,
                    color=color,
                    label=name if index == 0 else None,
                    alpha=0.9,
                )
                caption = f"{value:.1f}%" + ("  [R]" if refused else "")
                axis.text(
                    value + (3 if refused else 1.3),
                    y,
                    caption,
                    va="center",
                    fontsize=7.6,
                    color="#26333c",
                )
                if refused:
                    axis.scatter(
                        [max(value, 0.7)], [y], marker="x", s=22, color="#26333c", zorder=4
                    )
            else:
                # 空预测和不可测不是同一件事。null保留NA，绝不偷换成0或100%。
                axis.text(1.3, y, "NA" + ("  [R]" if refused else ""), va="center", fontsize=7.6)
    axis.set_yticks(positions, labels=labels, fontsize=8.4)
    axis.invert_yaxis()
    axis.set_xlim(0, 128)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Percent at 5 cm distance tolerance", fontsize=9)
    axis.tick_params(axis="y", length=0, pad=6)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", color="#dbe1e5", linewidth=0.6)
    axis.set_axisbelow(True)
    for boundary in range(group_size, len(labels), group_size):
        axis.axhline(boundary - 0.5, color="#c5ced4", linewidth=0.8)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#bcc7ce")
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False, fontsize=8.3)


def main():
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    # 数据选择固定在代码里。只能等两版都评测完成后出图，不能先挑好看的子集。
    sources, indices = {}, {}
    for kind, run_id in RUNS.items():
        path = ROOT / "data/evaluation" / run_id / "summary.json"
        summary = read_json(path)
        if summary["run_id"] != run_id:
            raise ValueError("Evaluation run identity changed")
        indices[kind] = evaluation_index(summary)
        sources[run_id] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"}
    )
    figure, axes = plt.subplots(1, 3, figsize=(20, 9.8))
    figure.subplots_adjust(left=0.1, right=0.99, top=0.79, bottom=0.18, wspace=0.68)

    labels, coverage, false_gap = [], [], []
    for case, case_label in ORACLE_CASES:
        for variant, method_label in VARIANTS:
            row = indices["dense"][("oracle_camera", case, 504, "gap", variant)]
            labels.append(case_label + "\n" + method_label)
            coverage.append((recovery(row), row))
            false_gap.append((gap_proximity(row), row))
    paired_bars(
        axes[0],
        labels,
        coverage,
        false_gap,
        ("Rod recovery", "Gap interior near prediction"),
        ("#2d6789", "#b55748"),
        4,
    )
    axes[0].set_title(
        "A  Oracle cameras: gapped rod\nDense rows; geometry and false filling",
        loc="left",
        pad=48,
        fontsize=11,
        fontweight="bold",
    )

    labels, legacy, proposed = [], [], []
    ordinary_refused = 0
    for case, resolution, case_label in ESTIMATED_JOBS:
        for target in ("whole", "gap"):
            baseline = indices["dense"][("estimated", case, resolution, target, "legacy_occupancy")]
            candidate = indices["dense"][("estimated", case, resolution, target, "paired_full")]
            labels.append(case_label + " / " + target)
            legacy.append((recovery(baseline), baseline))
            proposed.append((recovery(candidate), candidate))
            ordinary_refused += candidate["state"] != "accepted"
    paired_bars(
        axes[1],
        labels,
        legacy,
        proposed,
        ("Legacy occupancy", "Paired full: dense rows"),
        ("#a9b4bb", "#2d6789"),
        2,
    )
    axes[1].set_title(
        f"B  Estimated cameras: both targets\nDense paired full: {ordinary_refused}/{len(proposed)} refused",
        loc="left",
        pad=48,
        fontsize=11,
        fontweight="bold",
    )

    labels, sparse, dense = [], [], []
    for case, case_label in ORACLE_CASES:
        for variant, method_label in VARIANTS:
            key = ("oracle_camera", case, 504, "whole", variant)
            first, second = indices["sparse"][key], indices["dense"][key]
            labels.append(case_label + "\n" + method_label)
            sparse.append((recovery(first), first))
            dense.append((recovery(second), second))
    paired_bars(
        axes[2],
        labels,
        sparse,
        dense,
        ("Rows every 4 pixels", "Rows every 1 pixel"),
        ("#a9b4bb", "#2d6789"),
        4,
    )
    axes[2].set_title(
        "C  Oracle cameras: whole rod\nFrozen sparse / dense sampling control",
        loc="left",
        pad=48,
        fontsize=11,
        fontweight="bold",
    )

    figure.suptitle(
        "RGB evidence prototype: recovery and gap errors",
        x=0.04,
        ha="left",
        y=0.965,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.925,
        "One development asset; manually corresponding strips. Oracle panels use true cameras.",
        fontsize=11,
        color="#485964",
    )
    figure.text(
        0.04,
        0.095,
        "[R] = refused / empty proposal. Zero gap coverage is only useful alongside sufficient rod recovery. NA remains unmeasurable.",
        fontsize=10,
        color="#485964",
    )
    figure.text(
        0.04,
        0.067,
        "Gap bars measure geometric proximity inside the guarded real gap. This figure does not score topology or independent-object generalization.",
        fontsize=10,
        color="#485964",
    )
    matching_scores = True
    for index in indices.values():
        for key, row in index.items():
            if key[-1] == "paired_geometry":
                other = index[(*key[:-1], "paired_full")]
                matching_scores &= all(
                    row.get(field) == other.get(field)
                    for field in ("metrics", "gap", "state", "segment_count")
                )
    if matching_scores:
        figure.text(
            0.04,
            0.039,
            "Paired geometry/full have identical reported scores in both runs; the negative veto showed no added benefit here.",
            fontsize=10,
            color="#485964",
        )
    figure.text(
        0.04,
        0.012,
        "2026-09-16 | Fixed runs: rod-evidence-v1 and rod-evidence-dense-v1 | Complete metric tables remain in the evaluation summaries.",
        fontsize=8.5,
        color="#6b7880",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        dpi=170,
        facecolor="white",
        metadata={
            "Title": "Creator rod-evidence fixed development comparison",
            "Description": json.dumps(sources, ensure_ascii=True),
        },
    )
    plt.close(figure)
    print("PLOT_READY", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
