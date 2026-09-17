"""Show fixed-row luminance and yesterday's choices; truth is display-only."""
from pathlib import Path

import numpy as np
from thin_pack_gt import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    source = ROOT / "data/evaluation/rod-edge-audit-v1-20260917/summary.json"
    data = read_json(source)
    if data["state"] != "complete":
        raise ValueError("Audit incomplete")
    cases = ["brick-texture-thinner", "brick-texture-origin"]
    frames = ["view_-30", "view_+00", "view_+30"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey=True)
    for i, case in enumerate(cases):
        for j, frame in enumerate(frames):
            ax = axes[i, j]
            row = next(r for r in data["profiles"] if (r["case_id"], r["frame_id"], r["target"], r["y"]) == (case, frame, "whole", 300))
            center = row["axis_x"]
            for low, high in row["visible_intervals"]:
                ax.axvspan(low-center, high-center, color="#dce7ed", alpha=.8)
                ax.axvline(low-center, color="#8da8b7", lw=1, ls=":")
                ax.axvline(high-center, color="#8da8b7", lw=1, ls=":")
            ax.plot(np.array(row["x"])-center, row["gray"], color="#283b47", lw=1.7, marker=".", ms=3)
            ax.axvline(0, color="#2076b1", lw=1.8, ls="--")
            for pair in row["all_pairs"]:
                ax.plot(np.array(pair["edge_x"])-center, [18, 18], color="#a5a5a5", lw=3, alpha=.4)
            if row["selected"]:
                ax.plot(np.array(row["edge_x"])-center, [10, 10], color="#d8782f", lw=5, solid_capstyle="butt")
                ax.axvline(row["center_x"]-center, color="#d8782f", lw=1.5)
                message = f"Selected center: {row['center_x']-center:+.2f} px\nSelected width: {row['pair_width_px']:.1f} px"
            else:
                message = "No selected center"
            ax.text(.98, .97, message, ha="right", va="top", transform=ax.transAxes, fontsize=9,
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
            ax.set_title(("Thin (diameter 0.02 m)" if i == 0 else "Medium (diameter 0.10 m)") + "  |  " + frame.removeprefix("view_") + " deg", loc="left", fontsize=11)
            ax.set_xlim(-27, 27)
            ax.set_ylim(0, 255)
            ax.set_xlabel("Horizontal pixels from projected physical axis")
            if j == 0:
                ax.set_ylabel("RGB luminance (0..255)")
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A narrow bright strip can be stable, yet off-center", fontsize=19, x=.055, ha="left", y=.985)
    handles = [Line2D([0], [0], color="#283b47", label="Observed luminance"),
               Patch(color="#dce7ed", label="Visible GT rod footprint"),
               Line2D([0], [0], color="#2076b1", ls="--", label="Projected physical axis"),
               Line2D([0], [0], color="#d8782f", label="Frozen selected pair / center")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.06, right=.99, top=.83, bottom=.16, hspace=.43, wspace=.12)
    fig.text(.055, .075, "Display: whole rod at fixed audit row y=300. The complete audit includes all five views and rows 300 / 500 / 700.", fontsize=10, color="#475c69")
    fig.text(.055, .043, "Truth only centers the display and marks integer-mask boundaries (+/-0.5 px approximation); exact mesh geometry is audited separately.", fontsize=9, color="#475c69")
    output = ROOT / "docs/experiments/assets/2026-09-17-edge-profiles.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, metadata={"audit_summary_sha256": sha256(source)})
    plt.close(fig)
    write_json(output.with_suffix(".json"), {"audit_summary_sha256": sha256(source), "figure_sha256": sha256(output),
                                           "display_only": True, "y": 300, "target": "whole", "cases": cases, "frames": frames})
    print(output)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
