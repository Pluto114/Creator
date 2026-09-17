"""Material-intervention plots and a paired-row diagnostic; no detector changes."""
from pathlib import Path

import numpy as np
from audit_rod_edges import projected_axis_at_rows
from thin_pack_gt import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "rod-appearance-control-v1-20260917"


def selected_points(row):
    if not row["fitted"]["usable"]:
        return {}
    points = {}
    for match in row["fitted"]["row_matches"]:
        observed = row["extracted"]["rows"][match["row_index"]]
        points[observed["y"]] = observed["candidates"][match["candidate_index"]]["center_x"]
    return points


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = ROOT / ".runtime/experiments" / RUN_ID
    evaluation = ROOT / "data/evaluation" / RUN_ID
    truth = ROOT / "data/eval_gt" / RUN_ID
    status = read_json(run / "status.json")
    summary = read_json(evaluation / "summary.json")
    if sha256(run / "observations.json") != status["observations_sha256"] or summary["status_sha256"] != sha256(run / "status.json"):
        raise ValueError("Appearance source identity changed")
    if sha256(truth / "artifact_hashes.json") != status["gt_index_sha256"]:
        raise ValueError("Appearance GT index changed")
    for relative, digest in read_json(truth / "artifact_hashes.json").items():
        if sha256(truth / relative) != digest:
            raise ValueError("Appearance GT changed")
    observations = read_json(run / "observations.json")
    pairs = []
    order = [(frame, target) for frame in ("view_-30", "view_+00", "view_+30") for target in ("whole", "gap")]
    for frame, target in order:
        pair = [next(r for r in observations if (r["condition"], r["frame_id"], r["target"]) == (condition, frame, target))
                for condition in ("original_replica", "uniform_emission")]
        first, second = [selected_points(r) for r in pair]
        rows = np.array(sorted(set(first) & set(second)))
        camera = read_json(truth / "original_replica" / (frame + "-camera.json"))
        geometry = read_json(truth / "original_replica/geometry.json")
        rod_id = summary["protocol"]["target_rod_ids_evaluation_only"][target]
        segments = [o["world_centerline_endpoints"] for o in geometry["objects"] if o["rod_id"] == rod_id and o["duplicate_geometry_of"] is None]
        center = projected_axis_at_rows(segments, camera, rows)
        finite = np.isfinite(center)
        metrics = []
        for mapping in (first, second):
            error = abs(np.array([mapping[y] for y in rows]) - center)[finite]
            metrics.append({"median_abs_error_px": float(np.median(error)) if len(error) else None,
                            "p95_abs_error_px": float(np.quantile(error, .95)) if len(error) else None})
        pairs.append({"frame_id": frame, "target": target, "common_selected_finite_rows": int(finite.sum()),
                      "original_replica": metrics[0], "uniform_emission": metrics[1]})
    paired = {"scope": "Supplementary paired selected-row evaluation; intersection selection is explicit, not full detection recall",
              "source_summary_sha256": sha256(evaluation / "summary.json"), "observations_sha256": status["observations_sha256"], "rows": pairs}
    write_json(evaluation / "common_rows.json", paired)
    write_json(ROOT / "docs/experiments/results/2026-09-17-appearance-common-rows.json", paired)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    x, width = np.arange(len(order)), .35
    for side, ax in enumerate(axes):
        for offset, condition, label, color in ((-width/2, "original_replica", "Original material", "#ca774a"),
                                                 (width/2, "uniform_emission", "Uniform emission", "#3c829d")):
            values = []
            for index, (frame, target) in enumerate(order):
                if side == 0:
                    row = next(r for r in summary["rows"] if (r["condition"], r["frame_id"], r["target"]) == (condition, frame, target))
                    values.append(row["median_horizontal_error_px"])
                else:
                    values.append(pairs[index][condition]["median_abs_error_px"])
            ax.bar(x+offset, [np.nan if v is None else v for v in values], width, color=color, label=label)
            for index, value in enumerate(values):
                ax.text(index+offset, .15 if value is None else value+.15, "NA" if value is None else f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=0 if value is None else 60)
        ax.set_xticks(x, [f"{frame.removeprefix('view_')}\n{target}" for frame, target in order])
        ax.set_xlim(-.6, len(order) - .4)
        ax.set_ylim(0, 9.7)
        ax.set_ylabel("Median horizontal center error (pixels)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.set_title("All selected finite-rod rows" if side == 0 else "Only identical rows selected in both images", loc="left", fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Same geometry, cameras and detector; different target surface response", x=.065, ha="left", fontsize=17)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.23, wspace=.23)
    counts = ", ".join(str(r["common_selected_finite_rows"]) for r in pairs)
    fig.text(.065, .1, f"Shared finite rows, left to right: {counts}. NA: original +30 gap fit was ambiguous; no shared accepted rows.", fontsize=9, color="#425965")
    fig.text(.065, .057, "Three original replicas are pixel-identical to the old RGB. Emission is a controlled appearance intervention, not an algorithm improvement.", fontsize=9, color="#425965")
    output = ROOT / "docs/experiments/assets/2026-09-17-appearance-control.png"
    fig.savefig(output, dpi=160, metadata={"summary_sha256": sha256(evaluation / "summary.json")})
    plt.close(fig)
    print(output, pairs, flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
