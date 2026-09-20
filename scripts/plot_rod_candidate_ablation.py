"""Plot matched-row candidate support from frozen RGB-only ablation pools.

Only pool JSON and its RGB files are read. This figure does not use cameras,
geometry, masks, evaluator outputs, or physical ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from environment_paths import require_project_environment
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ["s01", "s04", "s05", "s07", "s10"]
COLORS = {
    "top8": "#00b8ff",
    "extra8": "#ff9f1c",
    "left": "#8cff66",
    "right": "#ff65cb",
    "guide": "#ffffff",
}


def project_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must stay inside the actual project root: {path}")
    if "eval_gt" in path.parts:
        raise ValueError("This RGB-only plot must not read or write eval_gt")
    return path


def read_frame(run_dir, case_id, offset, view_id):
    path = run_dir / "pools" / f"{case_id}-search-{offset:g}.json"
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record["case_id"] != case_id or float(record["search_offset_px"]) != offset:
        raise ValueError(f"Pool identity does not match its requested panel: {path}")
    frames = [frame for frame in record["frames"] if frame["view_id"] == view_id]
    if len(frames) != 1:
        raise ValueError(f"Expected exactly one {view_id} frame in {path}")
    frame = frames[0]
    if not frame["observations"]["rows"]:
        raise ValueError(f"No observation rows available to define a crop: {path}")
    return frame


def matched_support(hypothesis, observations):
    """Return fitted centers and the exact matched raw pair, without interpolation."""
    rows = observations["rows"]
    support, seen = [], set()
    slope, intercept = float(hypothesis["slope"]), float(hypothesis["intercept"])
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("Candidate line coefficients must be finite")
    for match in hypothesis["row_matches"]:
        if len(match) != 3:
            raise ValueError("Expected legacy [row_index, candidate_index, residual] triples")
        row_index, candidate_index, residual = match
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise ValueError("Matched row index must be an integer")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise ValueError("Matched raw candidate index must be an integer")
        if row_index in seen or not 0 <= row_index < len(rows):
            raise ValueError("Matched row index must be unique and inside raw observations")
        seen.add(row_index)
        row = rows[row_index]
        if not 0 <= candidate_index < len(row["candidates"]):
            raise ValueError("Matched candidate index is outside its original row")
        candidate = row["candidates"][candidate_index]
        y = float(row["y"])
        values = [
            slope * y + intercept,
            y,
            float(candidate["left_edge"]["x"]),
            float(candidate["right_edge"]["x"]),
            float(residual),
        ]
        if not np.isfinite(values).all() or values[2] > values[3]:
            raise ValueError("Invalid matched row coordinates or residual")
        support.append(values[:4])
    return np.asarray(support, dtype=float).reshape(-1, 4)


def shared_crop(frames, width, height):
    """Use one crop for both windows, based solely on observation search strips."""
    xs, ys = [], []
    for frame in frames:
        observations = frame["observations"]
        for row in observations["rows"]:
            x, y = float(row["guide_x"]), float(row["y"])
            xs.append(x)
            ys.append(y)
            window = row.get("absence_window_xyxy")
            if window is not None:
                # 名字里有absence，但这里只拿搜索条带的边界，绝不当缺失真值。
                xs.extend([float(window[0]), float(window[2])])
            elif "scan_half_width" in observations.get("config", {}):
                half = float(observations["config"]["scan_half_width"])
                xs.extend([x - half, x + half])
            for candidate in row["candidates"]:
                xs.extend([float(candidate["left_edge"]["x"]), float(candidate["right_edge"]["x"])])
    if not np.isfinite(xs + ys).all():
        raise ValueError("Crop coordinates must be finite")
    left = max(0, int(np.floor(min(xs))) - 8)
    right = min(width, int(np.ceil(max(xs))) + 9)
    top = max(0, int(np.floor(min(ys))) - 8)
    bottom = min(height, int(np.ceil(max(ys))) + 9)
    if right <= left or bottom <= top:
        raise ValueError("Observation crop does not overlap RGB")
    return left, top, right, bottom


def draw_rgb_crop(ax, rgb, bounds):
    """Display the identical pixel crop and coordinate scale in all three columns."""
    left, top, right, bottom = bounds
    ax.imshow(
        rgb[top:bottom, left:right],
        extent=(left - 0.5, right - 0.5, bottom - 0.5, top - 0.5),
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlim(left - 0.5, right - 0.5)
    ax.set_ylim(bottom - 0.5, top - 0.5)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Original RGB x (px; enlarged)", fontsize=9)
    ax.set_ylabel("Original RGB y (px)", fontsize=9)


def draw_panel(ax, frame, rgb, bounds, case_id, offset):
    draw_rgb_crop(ax, rgb, bounds)
    observations = frame["observations"]
    rows = observations["rows"]
    pool = frame["pool"]["candidates"]
    # 只在真实row_matches对应的y画拟合x。不能为了好看把缺行接成一整根线。
    for rank in range(min(16, len(pool)), 0, -1):
        support = matched_support(pool[rank - 1], observations)
        ax.scatter(
            support[:, 0],
            support[:, 1],
            s=3.4 if rank <= 8 else 4.2,
            color=COLORS["top8" if rank <= 8 else "extra8"],
            alpha=0.82 if rank <= 8 else 0.65,
            marker=".",
            linewidths=0,
            zorder=3,
        )
    rank1_rows = 0
    if pool:
        support = matched_support(pool[0], observations)
        rank1_rows = len(support)
        for column, color in [(2, COLORS["left"]), (3, COLORS["right"])]:
            ax.scatter(
                support[:, column],
                support[:, 1],
                s=7,
                color=color,
                marker=".",
                linewidths=0,
                zorder=5,
            )
    guide_x = [row["guide_x"] for row in rows]
    guide_y = [row["y"] for row in rows]
    # guide是搜索位置，不是算法恢复出来的支撑；用独立的白色小横杠标清。
    ax.scatter(
        guide_x,
        guide_y,
        color=COLORS["guide"],
        marker="_",
        s=5,
        linewidths=0.45,
        alpha=0.9,
        zorder=4,
    )
    shown_top, shown_extra = min(8, len(pool)), min(8, max(0, len(pool) - 8))
    ax.set_title(
        f"{case_id} | search offset {offset:+g} px | full sampled pool N={len(pool)}\n"
        f"Shown ranks 1-8: {shown_top}, ranks 9-16: {shown_extra}; rank-1 matched rows: {rank1_rows}\n"
        "Rank 1 is local pool order, not the final multiview selection",
        loc="left",
        fontsize=9.2,
    )
    if not pool:
        ax.text(
            0.5,
            0.5,
            "No supported candidates",
            transform=ax.transAxes,
            ha="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )


def render(run_id, output, cases, offsets, view_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be one directory name")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", case) for case in cases):
        raise ValueError("Case IDs must be simple path-free identifiers")
    if len(cases) != len(set(cases)) or not cases:
        raise ValueError("Choose nonempty, unique case IDs")
    if len(offsets) != 2 or not np.isfinite(offsets).all() or offsets[0] == offsets[1]:
        raise ValueError("Choose exactly two distinct finite offsets")
    if output.exists():
        raise FileExistsError(f"Keep existing figure: {output}")
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Output must be .png, .pdf, or .svg")
    run_dir = ROOT / ".runtime/experiments" / run_id
    panels = []
    for case_id in cases:
        frames = [read_frame(run_dir, case_id, offset, view_id) for offset in offsets]
        rgb_paths = [project_path(frame["rgb"]) for frame in frames]
        if rgb_paths[0] != rgb_paths[1]:
            raise ValueError(f"Window comparison must use the same RGB path: {case_id}")
        with Image.open(rgb_paths[0]) as source:
            rgb = np.asarray(source.convert("RGB"))
        height, width = rgb.shape[:2]
        if any(list(frame["size_wh"]) != [width, height] for frame in frames):
            raise ValueError(f"Frozen size_wh disagrees with RGB dimensions: {case_id}")
        panels.append((case_id, frames, rgb, shared_crop(frames, width, height)))
    # 标题和图例按英寸留位置。按整图百分比挤间距，一行预览会把两行字叠在一起。
    header_inches, footer_inches = 1.0, 1.15
    figure_height = 3.8 * len(cases) + header_inches + footer_inches
    fig, axes = plt.subplots(len(cases), 3, figsize=(19, figure_height), squeeze=False)
    try:
        for row_axes, (case_id, frames, rgb, bounds) in zip(axes, panels):
            # 先让人看清原图。小杆上挤满彩色点时，光看叠加图很容易脑补出轮廓。
            draw_rgb_crop(row_axes[0], rgb, bounds)
            row_axes[0].set_title(
                f"{case_id} | original RGB, no overlays\n"
                f"Same crop as search offsets {offsets[0]:+g} / {offsets[1]:+g} px\n"
                "Pixel coordinates and display scales match both panels on the right",
                loc="left",
                fontsize=9.2,
            )
            for ax, frame, offset in zip(row_axes[1:], frames, offsets):
                draw_panel(ax, frame, rgb, bounds, case_id, offset)
        fig.text(
            0.07,
            1 - 0.18 / figure_height,
            "Candidate cap and search-window diagnostics",
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.07,
            1 - 0.54 / figure_height,
            f"{run_id} | {view_id} | RGB observations only; no physical ground truth",
            fontsize=10,
            va="top",
        )
        legend = [
            Line2D(
                [0],
                [0],
                marker=".",
                ls="none",
                color=COLORS["top8"],
                markersize=8,
                label="Ranks 1-8: fitted x at matched y only",
            ),
            Line2D(
                [0],
                [0],
                marker=".",
                ls="none",
                color=COLORS["extra8"],
                markersize=8,
                label="Ranks 9-16: fitted x at matched y only",
            ),
            Line2D(
                [0],
                [0],
                marker=".",
                ls="none",
                color=COLORS["left"],
                markersize=8,
                label="Rank 1: raw left-edge samples",
            ),
            Line2D(
                [0],
                [0],
                marker=".",
                ls="none",
                color=COLORS["right"],
                markersize=8,
                label="Rank 1: raw right-edge samples",
            ),
            Line2D(
                [0],
                [0],
                marker="_",
                ls="none",
                color=COLORS["guide"],
                markersize=8,
                label="Search-window guide (not an identity variant)",
            ),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=2,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.34 / figure_height),
            facecolor="#252b34",
            edgecolor="none",
            labelcolor="white",
            framealpha=1,
        )
        fig.text(
            0.5,
            0.10 / figure_height,
            "Missing rows remain empty. Common RGB crop per case; x/y display scales differ. "
            "Full pool means the fixed sampled pool, not all possible image explanations.",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        fig.tight_layout(
            rect=(0.015, footer_inches / figure_height, 0.99, 1 - header_inches / figure_height),
            h_pad=2.0,
            w_pad=2.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # xb也挡住检查之后的并发覆盖；这里唯一写入的是新图，预测文件完全只读。
        with output.open("xb") as stream:
            fig.savefig(stream, format=output.suffix[1:].lower(), dpi=180, facecolor="white")
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--offsets", nargs=2, type=float, default=[0, 8])
    parser.add_argument("--view-id", default="view_+02")
    args = parser.parse_args()
    require_project_environment(ROOT)
    print(render(args.run_id, project_path(args.output), args.cases, args.offsets, args.view_id))


if __name__ == "__main__":
    main()
