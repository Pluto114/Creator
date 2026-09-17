"""Publish every background comparison and visualize the correspondence failure."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "background-correspondences-v1-20260917"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_result(run_id):
    output = ROOT / "data/evaluation" / run_id
    run = ROOT / ".runtime/experiments" / run_id
    source = output / "summary.json"
    summary, manifest = read(source), read(run / "manifest.json")
    protocol = read(run / "protocol.json")
    if summary["state"] != "complete" or manifest["state"] != "complete":
        raise ValueError("Cannot publish incomplete runs")
    if sha(run / "manifest.json") != summary["fit_manifest_sha256"]:
        raise ValueError("Fit manifest changed")
    for name in ("protocol", "selection", "input_manifest"):
        if sha(run / (name + ".json")) != manifest[name + "_sha256"]:
            raise ValueError("Frozen input/config changed")
    for entry in manifest["features"] + manifest["pairs"]:
        if sha(run / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen feature/match file changed")
    for entry in summary["artifacts"]:
        if sha(output / entry["path"]) != entry["sha256"]:
            raise ValueError("Per-match mesh result changed")
    pair_count = len(protocol["cases"]) * len(protocol["detectors"]) * len(protocol["pairs"])
    camera_count = len(protocol["evaluation"]["camera_jobs"]) * len(protocol["detectors"]) * len(protocol["pairs"])
    if len(summary["pairs"]) != pair_count or len(summary["camera_rows"]) != camera_count:
        raise ValueError("Missing comparison rows")
    pairs = copy.deepcopy(summary["pairs"])
    for pair in pairs:
        for model in pair["rgb_geometry"]["models"].values():
            if "train_inlier_mask" in model:
                mask = model.pop("train_inlier_mask")
                model["training_inlier_count"] = sum(mask)
                model["training_mask_count"] = len(mask)
                model.pop("all_residuals_px")
    metadata = {key: value for key, value in summary.items() if key not in ("pairs", "camera_rows")}
    metadata.update(full_summary_sha256=sha(source), protocol=protocol,
                    protocol_sha256=manifest["protocol_sha256"], selection_sha256=manifest["selection_sha256"],
                    input_manifest_sha256=manifest["input_manifest_sha256"], producer_sources=manifest["source_hashes"],
                    features=manifest["features"], fit_started_at=manifest["started_at"], fit_completed_at=manifest["completed_at"],
                    runtime={key: manifest[key] for key in ("opencv_version", "numpy_version", "opencv_threads")},
                    compact_schema_version="1.0.0", pair_count=pair_count, camera_row_count=camera_count,
                    omitted_arrays="Per-match residuals and training masks only; original pair/mesh paths and hashes retained. All pairs, camera conditions, subsets, counts, models and aggregate metrics retained.",
                    publication_script_sha256=sha(Path(__file__)))
    return metadata, pairs, summary["camera_rows"]


def write_compact(destination, metadata, pairs, cameras):
    # One row per comparison keeps a complete failure table readable in a Git diff.
    prefix = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    def rows(values):
        return ",\n".join("    " + json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for value in values)
    text = prefix[:-2] + ',\n  "pairs": [\n' + rows(pairs) + '\n  ],\n  "camera_rows": [\n' + rows(cameras) + '\n  ]\n}\n'
    parsed = json.loads(text)
    if parsed["pairs"] != pairs or parsed["camera_rows"] != cameras:
        raise AssertionError("Compact serialization lost a comparison")
    destination.write_text(text, encoding="utf-8")


def plot(pairs, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = ["black-white-thinner", "brick-texture-thinner", "brick-texture-origin"]
    names = ["B/W thin", "Brick thin", "Brick medium"]
    groups = []
    for case, name in zip(cases, names):
        for detector in ("ORB", "SIFT"):
            entries = [p for p in pairs if p["case_id"] == case and p["detector"] == detector]
            eligible = sum(p["mesh_correspondence"]["all"]["both_directions_visible"] for p in entries)
            good = sum(round((p["mesh_correspondence"]["all"]["both_visible_fraction_within_2px"] or 0)
                             * p["mesh_correspondence"]["all"]["both_directions_visible"]) for p in entries)
            assessed = sum(p["rgb_geometry"]["state"] == "assessed" for p in entries)
            passes = {kind: sum(p["rgb_geometry"]["models"].get(kind, {}).get("state") == "validation_consistent" for p in entries)
                      for kind in ("F", "H")}
            groups.append({"name": name, "detector": detector, "good": good, "eligible": eligible,
                           "matches": sum(p["match_count"] for p in entries), "assessed": assessed, **passes})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 10), facecolor="#fafbfc")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.68, wspace=.23)
    ax = fig.add_subplot(grid[0, 0])
    colors = {"ORB": "#326da8", "SIFT": "#7c5bae"}
    values = [100 * g["good"] / g["eligible"] if g["eligible"] else 0 for g in groups]
    bars = ax.bar(np.arange(6), values, color=[colors[g["detector"]] for g in groups], width=.68)
    for index, (g, bar) in enumerate(zip(groups, bars)):
        if not g["assessed"]:
            bar.set_hatch("///")
        ax.text(index, values[index] + 1.1, f"{g['good']}/{g['eligible']}\n{values[index]:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(6), [g["name"] + "\n" + g["detector"] for g in groups], fontsize=9)
    ax.set_ylim(0, 38)
    ax.set_ylabel("Physical agreement within 2 px (%)")
    ax.set_title("A  Raw matches: changing the detector\ndoes not fix them", loc="left", fontsize=12, fontweight="bold", pad=16)
    ax.grid(axis="y", color="#dde2e8", alpha=.8)
    ax.set_axisbelow(True)
    ax.text(0, -.22, "Agreement / both-visible matches; unevaluable matches excluded here.\nB/W has too few matches for model validation (hatched).", transform=ax.transAxes, fontsize=9, va="top", color="#465260")

    table_ax = fig.add_subplot(grid[0, 1])
    table_ax.axis("off")
    table_ax.set_title("B  Fixed training / validation protocol", loc="left", fontsize=12, fontweight="bold", pad=16)
    rows = [[g["name"] + " " + g["detector"], str(g["matches"]), f"{g['assessed']}/6",
             f"{g['F']}/{g['assessed']}" if g["assessed"] else "inconclusive",
             f"{g['H']}/{g['assessed']}" if g["assessed"] else "inconclusive"] for g in groups]
    table = table_ax.table(cellText=rows, colLabels=["Condition", "Matches", "Assessed\npairs", "F passes", "H passes"],
                          colWidths=[.34, .14, .15, .185, .185], cellLoc="center", bbox=[0, .19, 1, .77])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e1e6ec")
        if row == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#f2f5f8" if row % 2 else "white")
    table_ax.text(0, .08, "Pass: >=60% of held-out residuals <=2 px. F: Sampson; H: symmetric transfer.\nInsufficient fixed-split support stays inconclusive; no resampling.\nZero H passes does not establish that the scene is non-planar.", transform=table_ax.transAxes, fontsize=9, va="top", color="#465260")

    detail = next(p for p in pairs if p["pair_id"] == "brick-texture-origin--ORB--2-3")
    val = detail["mesh_correspondence"]["validation"]
    model = detail["rgb_geometry"]["models"]["F"]["validation"]
    total = val["match_count"]
    consistent = round(model["fraction_within_2px"] * total)
    visible = val["both_directions_visible"]
    correct = round(val["both_visible_fraction_within_2px"] * visible)
    ax = fig.add_subplot(grid[1, :])
    good_color, bad_color, unknown_color = "#2b8c74", "#d88d54", "#cbd2da"
    ax.barh(1, consistent, color=good_color, height=.45)
    ax.barh(1, total - consistent, left=consistent, color=bad_color, height=.45)
    ax.barh(0, correct, color=good_color, height=.45)
    ax.barh(0, visible - correct, left=correct, color=bad_color, height=.45)
    ax.barh(0, total - visible, left=visible, color=unknown_color, height=.45, hatch="///", edgecolor="#8d98a5")
    ax.text(consistent / 2, 1, f"{consistent}/{total} = {100*consistent/total:.1f}%  F agreement", ha="center", va="center", color="white", fontweight="bold")
    ax.text(consistent + (total - consistent) / 2, 1, f"{total-consistent} outside 2 px", ha="center", va="center", fontsize=10)
    ax.text(correct + (visible - correct) / 2, 0, f"{visible-correct} both-visible, outside 2 px", ha="center", va="center")
    ax.text(visible + (total - visible) / 2, 0, f"{total-visible} not\nboth-visible", ha="center", va="center", fontsize=10)
    ax.annotate(f"{correct}/{total} = {100*correct/total:.1f}% verified", xy=(correct / 2, .22), xytext=(0, .56),
                arrowprops={"arrowstyle": "-", "color": good_color}, fontsize=10, color=good_color, fontweight="bold")
    ax.set_yticks([1, 0], ["F Sampson\n<=2 px", "Physical transfer\n<=2 px, both ways"])
    ax.set_xticks(np.arange(0, total + 1, 12))
    ax.set_xlim(0, total)
    ax.set_ylim(-.55, 1.5)
    ax.set_xlabel(f"The same {total} validation matches; neither GT nor validation refits the model")
    ax.set_title("C  A false reassurance: brick medium, ORB, view +0 to +15 degrees", loc="left", fontsize=13, fontweight="bold", pad=18)
    ax.text(.01, -.32, f"Conditional physical agreement: {correct}/{visible} = {100*correct/visible:.1f}% among both-visible matches.\nThe 72.6% and 5.9% figures use different denominators; unknown visibility is not counted as a wrong match.", transform=ax.transAxes, fontsize=10, color="#465260", va="top")
    fig.suptitle("Background correspondence audit: a consistent F can still connect the wrong places", fontsize=17, fontweight="bold", x=.05, ha="left", y=.975)
    fig.text(.05, .935, "Original RGB only for matching and robust fitting. True mesh/cameras only for evaluation. No camera was corrected.", fontsize=11, color="#465260")
    fig.subplots_adjust(left=.13, right=.975, top=.855, bottom=.18)
    fig.text(.05, .025, "One development asset; image pairs are repeated measurements. All 36 pairs and 84 camera comparisons, including failures, are retained in the JSON.", fontsize=9, color="#617082")
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--replace", action="store_true", help="Replace this publication's derived JSON/figure only")
    args = parser.parse_args()
    destination = ROOT / "docs/experiments/results/2026-09-17-background-correspondences.json"
    figure = ROOT / "docs/experiments/assets/2026-09-17-background-correspondences.png"
    if not args.replace and (destination.exists() or figure.exists()):
        raise FileExistsError("Publication exists; use --replace only for the derived publication")
    metadata, pairs, cameras = compact_result(args.run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    write_compact(destination, metadata, pairs, cameras)
    plot(pairs, figure)
    print(destination)
    print(figure)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
