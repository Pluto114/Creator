"""Export the complete frozen score table and a fixed six-profile diagnostic."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from run_rod_profile_controls import (
    CONFIG,
    EVALUATION,
    INPUTS,
    ROOT,
    RUN,
    RUN_ID,
    TRUTH,
    digest,
    read_json,
    require_hash,
    require_project_environment,
    verify_sources,
    write_json,
)

PANEL_CASES = [
    ("p01", "Flat bright rod"),
    ("p07", "Asymmetric bright step"),
    ("p09", "Narrow highlight, weak silhouette"),
    ("p12", "Painted stripe, no rod"),
    ("p14", "Low-contrast physical rod"),
    ("p19", "Background stripe inside a true gap"),
]
PLOT_ROW = 80


def chosen_center(result):
    if not result["fitted"]["usable"]:
        return None
    for match in result["fitted"]["row_matches"]:
        row = result["rows"][match["row_index"]]
        if row["y"] == PLOT_ROW:
            return row["candidates"][match["candidate_index"]]["center_x"]
    return None


def make_plot(predictions, image_entries, truth_entries, output):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, (case_id, label) in zip(axes.flat, PANEL_CASES):
        image_entry = image_entries[case_id]
        image_path = INPUTS / image_entry["image"]
        require_hash(image_path, image_entry["image_sha256"])
        gray = np.asarray(Image.open(image_path))[PLOT_ROW, :, 0]
        truth_entry = truth_entries[case_id]
        truth_path = TRUTH / truth_entry["truth"]
        require_hash(truth_path, truth_entry["truth_sha256"])
        truth = read_json(truth_path)
        objects = next(row["objects"] for row in truth["rows"] if row["y"] == PLOT_ROW)
        for obj in objects:
            ax.axvspan(obj["left_x"], obj["right_x"], color="#8eb9a1", alpha=0.26, zorder=0)
            ax.axvline(obj["center_x"], color="#254d3b", linestyle="--", linewidth=1.2)
        ax.plot(np.arange(len(gray)), gray, color="#333333", linewidth=1.8)
        record = predictions[case_id]
        require_hash(RUN / "predictions" / record["prediction"], record["prediction_sha256"])
        methods = read_json(RUN / "predictions" / record["prediction"])["methods"]
        centers = []
        for method, color, style in [
            ("paired_robust", "#2166ac", "-"),
            ("pair_center_consensus_reject", "#d87720", ":"),
        ]:
            center = chosen_center(methods[method])
            centers.append(center)
            if center is not None:
                ax.axvline(center, color=color, linestyle=style, linewidth=2)
        baseline = "rejected" if centers[0] is None else f"x={centers[0]:.1f}"
        control = "rejected" if centers[1] is None else f"x={centers[1]:.1f}"
        physical = (
            f"Physical center x={objects[0]['center_x']:.1f}"
            if objects
            else "No physical rod at this row"
        )
        ax.text(
            0.02,
            0.96,
            physical + f"\nB: {baseline} | C: {control}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
        ax.set_title(f"{case_id}  {label}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(42, 86)
        ax.set_ylim(0, 275)
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("RGB intensity (0-255)")
    for ax in axes[-1]:
        ax.set_xlabel("Original-image pixel x")
    legend = [
        Line2D([0], [0], color="#333333", lw=2, label="Observed RGB profile"),
        Patch(facecolor="#8eb9a1", alpha=0.4, label="Physical support (evaluation only)"),
        Line2D([0], [0], color="#254d3b", ls="--", label="Physical center (evaluation only)"),
        Line2D([0], [0], color="#2166ac", lw=2, label="B: paired edges + robust line"),
        Line2D(
            [0], [0], color="#d87720", lw=2, ls=":", label="C: pair-center disagreement rejection"
        ),
    ]
    fig.suptitle(
        "Photometric edges can be stable and still miss physical geometry",
        x=0.07,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.946,
        "Fixed analytic controls; row y=80 in every panel. No real-scene performance claim.",
        fontsize=11,
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.024),
    )
    fig.text(
        0.5,
        0.013,
        "Rejected means no accepted center, not zero error. The complete 22-case / 44-method table includes every result.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.015, 0.115, 0.99, 0.925), h_pad=2.2, w_pad=2.5)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    require_project_environment(ROOT)
    prepared = read_json(RUN / "prepared.json")
    verify_sources(prepared)
    require_hash(CONFIG, prepared["config_sha256"])
    inference = read_json(RUN / "inference.json")
    summary = read_json(EVALUATION / "summary.json")
    require_hash(RUN / "inference.json", summary["inference_sha256"])
    require_hash(TRUTH / "manifest.json", prepared["truth_manifest_sha256"])
    require_hash(INPUTS / "manifest.json", prepared["input_manifest_sha256"])
    truth_manifest = read_json(TRUTH / "manifest.json")
    require_hash(TRUTH / "protocol.json", truth_manifest["protocol_copy_sha256"])
    config = read_json(TRUTH / "protocol.json")
    if len(summary["rows"]) != 44 or summary["case_count"] != 22:
        raise ValueError("The full fixed 22-case, two-method protocol must be retained")
    result_path = ROOT / "docs/experiments/results/2026-09-17-profile-controls.json"
    report_path = ROOT / "docs/experiments/2026-09-17-profile-controls.md"
    figure_path = ROOT / "docs/experiments/assets/2026-09-17-profile-controls.png"
    for output in (result_path, report_path, figure_path):
        if output.exists():
            raise FileExistsError(f"Keep previous exported artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    export = dict(summary)
    export["frozen_protocol"] = config
    export["source_summary_sha256"] = digest(EVALUATION / "summary.json")
    export["input_manifest_sha256"] = prepared["input_manifest_sha256"]
    export["method_config_sha256"] = prepared["method_config_sha256"]
    export["boundary_assertions"] = {
        "generator_truth": "Known geometric interval before intensity profile, blur, pixel-area integration, and quantization.",
        "method_inputs": "RGB, a common coarse guide [[64,12],[64,148]], frozen method settings only.",
        "inference_files": "Inference reads data/inputs and .runtime method settings, never eval_gt or fixture definitions.",
        "selection": "Candidate index comes from the method fitter; evaluator never chooses the closest-to-GT pair.",
        "ambiguity": "Two equal physical objects have no unique target identity; nearest-object GT assignment is forbidden.",
        "ordering": "All case predictions were frozen before the separate evaluate command.",
        "absence": "Flat-background-like absence is a hypothesis; score false hypotheses on weak physical rods.",
        "failures": "All 44 method/case records are retained, including rejections, ambiguous fits, and null metrics.",
        "figure": "Six pre-requested case IDs p01,p07,p09,p12,p14,p19, all at row80; a diagnostic selection, not a ranking.",
    }
    write_json(result_path, export)
    make_plot(
        {entry["case_id"]: entry for entry in inference["cases"]},
        {entry["case_id"]: entry for entry in read_json(INPUTS / "manifest.json")["cases"]},
        {entry["case_id"]: entry for entry in truth_manifest["cases"]},
        figure_path,
    )
    introduction = f"""# 2026-09-17：亮度边缘不等于杆的物理轮廓

这是一轮**解析模拟机制检查，不是现实场景性能测试，也不是3D恢复分数**。目的很朴素：杆的物理位置不动，只改亮度，看看双边缘中点还靠不靠谱。

先冻结22种病例，再把输入RGB、解析几何真值、方法输出分别写入不同目录。方法只拿RGB、全病例相同的粗guide和固定参数。输出全部完成后，另一个评价阶段才读取真值。没有用本轮现实场景真值或保留视图选阈值，也没有看完结果删掉难例。

冻结协议SHA256：`{prepared["config_sha256"]}`。完整参数、44条结果和边界断言见 [可提交JSON](results/2026-09-17-profile-controls.json)。

## 结论和代价

- 平顶亮/暗杆能正确定位；对称三角剖面也能碰巧给对中心，但它的轮廓边界仍偏3px。中心对了，不代表找到了物体边界。
- 线性照明与非对称阶跃让原方法稳定偏1.5–2px。新增“多个可靠边缘对的中心分歧大于1px就整行拒绝”的对照拒绝了这4例；它没有恢复中心，目标行接受率从100%降到0%，1px内准确覆盖率仍是0%。
- 窄亮带、窄暗带和模糊后的窄亮带都只有一对可靠边缘。两方法仍全部接受，中心偏4px，边界最大误差的中位数达到10px。新的拒绝规则解决不了这种唯一但错误的答案。
- 纯背景竖纹被两方法100%接受；竖纹穿过真实缺口时，32个缺口采样行也全部被接受。这个数表示错误的行证据，不是拓扑恢复率。
- 低对比实体杆的137个采样行全部产生flat-background-like absence候选。这说明它只能是局部反证假设，不能证明那里没有杆。
- 干净缺口没有被自动填上，亚像素0.75px杆保持拒绝，两个同样合理的物体没有被真值偷偷指定一个目标。这些约束需要继续保留。

原方法16/22病例产生可用拟合，保守对照12/22；这不是正确率。两方法的1px内准确目标行覆盖率在每个病例都相同，新增对照只是减少4个错误接受病例，没有得到新的正确中心。22例是人为指定的机制检查，不能把比例当成现实发生率。

![Fixed analytic profile controls](assets/2026-09-17-profile-controls.png)

图固定展示p01/p07/p09/p12/p14/p19的第80行；物理轮廓与中心只在评价侧画上去。p12和p19这一行没有物理杆，仍可看到足以骗过双边缘方法的背景亮条纹。

## 可复现命令

```powershell
. .\\scripts\\Enter-CreatorEnvironment.ps1
.\\backends\\da3\\.venv\\Scripts\\python.exe scripts/run_rod_profile_controls.py prepare
.\\backends\\da3\\.venv\\Scripts\\python.exe scripts/run_rod_profile_controls.py infer
.\\backends\\da3\\.venv\\Scripts\\python.exe scripts/run_rod_profile_controls.py evaluate
.\\backends\\da3\\.venv\\Scripts\\python.exe scripts/plot_rod_profile_controls.py
.\\reconstruction\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_rod_profile_controls.py -v
```

这些命令采用不可覆盖的新输出目录；已有本轮文件时会明确停止，不会重写原始证据。运行ID为`{RUN_ID}`；输入在`data/inputs`，真值在`data/eval_gt`，推理在`.runtime/experiments`，评分在`data/evaluation`各自同名目录。

11个解析测试覆盖几何/亮度隔离、亚像素面积、确定性噪声、真缺口、中心分歧拒绝、唯一内部高光反例、低对比假absence、拒绝输出的空值语义以及不按最近真值选边或指定双物体身份。

## 完整结果

"""
    table = (EVALUATION / "summary.md").read_text(encoding="utf-8")
    table = table.replace("# Frozen analytic profile controls\n", "", 1).lstrip()
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(introduction + table)
    print(
        json.dumps(
            {"results": str(result_path), "report": str(report_path), "figure": str(figure_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
