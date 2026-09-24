"""Derive readable tables and one figure from the two completed public summaries.

Only the supplied summaries are experiment inputs. No estimator, matching,
alignment, threshold, or score is rerun here. Missing quantities remain null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "replay": ROOT / "docs/experiments/results/2026-09-24-replay-rod-camera-envelope.json",
    "analytic": ROOT / "docs/experiments/results/2026-09-24-analytic-rod-camera-envelope.json",
}
OUTPUT = ROOT / "docs/experiments/results/2026-09-24-rod-camera-envelope-readable.json"
FIGURE = ROOT / "docs/experiments/figures/2026-09-24-rod-camera-envelope.png"
REPORT = ROOT / "docs/experiments/2026-09-24-rod-camera-envelope.md"
EXPECTED_IDS = ["control", *["leave_group_" + str(i) for i in range(4)]]
STATES = {"complete_empirical_envelope": "完整", "unresolved": "未决", "unavailable": "不可用"}
FAMILIES = {
    "volume_near_low_noise": ("体积背景／近杆／低噪声", "volume, near, low noise"),
    "volume_far_low_noise": ("体积背景／远杆／低噪声", "volume, far, low noise"),
    "volume_near_high_noise": ("体积背景／近杆／高噪声", "volume, near, high noise"),
    "volume_far_high_noise": ("体积背景／远杆／高噪声", "volume, far, high noise"),
    "single_depth_near": ("8m平面背景／近杆", "8m plane, near"),
    "far_plane_near": ("24m平面背景／近杆", "24m plane, near"),
    "single_depth_far": ("8m平面背景／远杆", "8m plane, far"),
    "narrow_near": ("窄基线／近杆", "narrow baseline, near"),
    "narrow_far_noisy": ("窄基线／远杆／高噪声", "narrow baseline, far, noisy"),
    "gapped_volume": ("体积背景／断杆", "gapped rod"),
    "coherent_wrong_rod_lateral": ("各视图一致看错横向杆", "coherent wrong rod: lateral"),
    "coherent_wrong_rod_depth": ("各视图一致看错深度", "coherent wrong rod: depth"),
    "shared_frame_pixel_bias": ("背景与杆共享像素偏移", "shared frame pixel bias"),
    "one_view_endpoint_swap": ("一个视图端点对应交换", "one-view endpoint swap"),
    "single_view_rod": ("杆仅单视图可见", "single-view rod"),
    "no_observations": ("没有观测", "no observations"),
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_source(path, mode, count):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["state"] != "complete" or data["mode"] != mode or len(data["rows"]) != count:
        raise ValueError("Expected complete public summary with fixed family count: " + mode)
    ids = [row["task_id"] for row in data["rows"]]
    if len(set(ids)) != count:
        raise ValueError("Duplicate family in public summary")
    if mode == "analytic":
        families = [row["family"] for row in data["rows"]]
        if len(set(families)) != count or set(families) != set(FAMILIES):
            raise ValueError("Analytic case families differ from the declared sixteen")
    else:
        grouped = defaultdict(list)
        for row in data["rows"]:
            grouped[(row["parent"], row["case_id"])].append(row["method"])
        if len(grouped) != 7 or any(sorted(methods) != ["baseline", "cylinder_support"] for methods in grouped.values()):
            raise ValueError("Replay must preserve seven cases and both rod methods")
    return data, dict(path=path.relative_to(ROOT).as_posix(), sha256=digest_bytes(raw), run_id=data["run_id"])


def value_range(values, expected):
    present = [float(v) for v in values if v is not None]
    if any(not np.isfinite(v) for v in present):
        raise ValueError("Nonfinite derived display value")
    return dict(min=min(present) if present else None, max=max(present) if present else None,
                available_count=len(present), expected_count=expected, missing_count=expected - len(present))


def _metric_row(row, slot):
    metrics = row.get("metrics")
    prediction = (metrics or {}).get("prediction_to_truth", {})
    target = (metrics or {}).get("truth_to_prediction", {})
    nonempty = metrics is not None and prediction.get("source_length", 0.) > 0
    # Raw recovery may correctly be zero for an empty prediction. Keep it, but
    # never let absence masquerade as a zero-error geometric observation.
    return dict(condition_id=slot["condition_id"], slot_state=slot["state"],
                camera_decision_state=slot.get("camera_decision_state"),
                identity_state=slot.get("identity_state"), slot_reasons=slot.get("reasons", []),
                physical_reason=row.get("reason"), metrics_present=metrics is not None,
                nonempty_predicted_geometry=bool(nonempty),
                truth_to_prediction_p95_m=target.get("distance_p95"),
                prediction_to_truth_p95_m=prediction.get("distance_p95"),
                recovery_fraction=(metrics or {}).get("recovery_fraction"),
                precision_fraction=(metrics or {}).get("precision_fraction"))


def summarize_family(row, mode):
    envelope, physical = row["envelope"], row["physical"]
    slots = envelope["conditions"]
    if [slot["condition_id"] for slot in slots] != EXPECTED_IDS or envelope["expected_condition_count"] != 5:
        raise ValueError("Envelope lost one of its five planned condition slots")
    by_id = {p["condition_id"]: p for p in physical["rows"]}
    if len(by_id) != len(physical["rows"]) or set(by_id) != set(EXPECTED_IDS):
        raise ValueError("Physical evaluation must also retain all five rows")
    conditions = [_metric_row(by_id[slot["condition_id"]], slot) for slot in slots]
    full, folds = conditions[0], conditions[1:]
    coverage = physical.get("coverage")
    complete = envelope["state"] == "complete_empirical_envelope"
    if complete != (envelope["segment_envelopes"] is not None):
        raise ValueError("Envelope completeness and ranges disagree")
    scale = (physical.get("alignment") or {}).get("scale")
    baseline = envelope["baseline"]
    segments, gaps = [], []
    if complete:
        for segment in envelope["segment_envelopes"]:
            low, high = np.asarray(segment["min_delta_local"]), np.asarray(segment["max_delta_local"])
            if low.shape != (envelope["samples_per_segment"], 3) or high.shape != low.shape:
                raise ValueError("Unexpected parameter-position box shape")
            diagonal = float(np.linalg.norm(high - low, axis=1).max())
            parts = (coverage or {}).get("segments", [])
            matching = next((part for part in parts if part["control_segment_index"] == segment["control_segment_index"]), None)
            segments.append(dict(control_segment_index=segment["control_segment_index"],
                maximum_box_diagonal_native=diagonal,
                maximum_box_diagonal_over_baseline=diagonal / baseline,
                maximum_box_diagonal_m=diagonal * scale if scale is not None else None,
                maximum_pairwise_direction_angle_degrees=segment["maximum_pairwise_direction_angle_degrees"],
                maximum_fractional_displacement_over_baseline=segment["maximum_fractional_displacement_over_baseline"],
                length_over_baseline_min=segment["length_over_baseline_min"],
                length_over_baseline_max=segment["length_over_baseline_max"],
                endpoint_delta_local_min=segment["endpoint_delta_local_min"],
                endpoint_delta_local_max=segment["endpoint_delta_local_max"],
                sampled_truth_containment=matching))
        for gap in envelope["gap_envelopes"]:
            low, high = gap["gap_along_control_axis_native_min"], gap["gap_along_control_axis_native_max"]
            gaps.append(dict(left_control_segment_index=gap["left_control_segment_index"],
                right_control_segment_index=gap["right_control_segment_index"],
                along_control_axis_over_baseline_min=low / baseline,
                along_control_axis_over_baseline_max=high / baseline,
                along_control_axis_m_min=low * scale if scale is not None else None,
                along_control_axis_m_max=high * scale if scale is not None else None))
    if mode == "analytic":
        label, label_en = FAMILIES[row["family"]]
        label = row["case_id"] + " " + label
    else:
        path = "变高" if "height" in row["parent"] else "同高"
        method = "普通" if row["method"] == "baseline" else "圆柱"
        label = path + "-" + row["case_id"] + "-" + method
        label_en = label
    p95_range = value_range([f["truth_to_prediction_p95_m"] for f in folds], 4)
    # Empty predictions have a valid zero recovery. Excluding that zero makes
    # failed folds look successful; only genuinely missing metrics stay null.
    r_range = value_range([f["recovery_fraction"] for f in folds], 4)
    p_range = value_range([f["precision_fraction"] for f in folds], 4)
    box_values = [s["maximum_box_diagonal_m"] for s in segments if s["maximum_box_diagonal_m"] is not None]
    direction_values = [s["maximum_pairwise_direction_angle_degrees"] for s in segments]
    parameter_fraction = coverage.get("fraction") if coverage else None
    return dict(task_id=row["task_id"], case_id=row["case_id"], parent=row.get("parent"), method=row["method"],
        family=row.get("family"), label_zh=label, label_en=label_en, intended_role=row.get("intended_role"),
        expected_limit=row.get("expected_limit"), state=envelope["state"], reasons=envelope["reasons"],
        comparable_condition_count=envelope["comparable_condition_count"], expected_condition_count=5,
        control=full, conditions=conditions, fold_p95_m=p95_range,
        fold_recovery=r_range, fold_precision=p_range,
        physical_state=physical["state"], alignment_scale=scale,
        maximum_box_diagonal_m=max(box_values) if box_values else None,
        maximum_box_diagonal_over_baseline=max((s["maximum_box_diagonal_over_baseline"] for s in segments), default=None),
        maximum_pairwise_direction_angle_degrees=max(direction_values) if direction_values else None,
        segments=segments if complete else None, gap_ranges=gaps if complete else None,
        sampled_truth_containment_state=(coverage or {}).get("state"),
        sampled_truth_containment_equal_segment_fraction=parameter_fraction,
        complete_but_some_truth_samples_outside=bool(complete and parameter_fraction is not None and parameter_fraction < 1.),
        complete_but_all_truth_samples_outside=bool(complete and parameter_fraction == 0.),
        incomplete_conditions=[dict(condition_id=s["condition_id"], state=s["state"], reasons=s["reasons"],
                                    camera_decision_state=s.get("camera_decision_state"), identity_state=s.get("identity_state"))
                               for s in slots if s["state"] != "comparable"])


def summarize(data, mode):
    rows = [summarize_family(row, mode) for row in sorted(data["rows"], key=lambda r: r["task_id"])]
    return dict(expected_family_count=14 if mode == "replay" else 16, family_count=len(rows),
                condition_row_count=sum(len(row["conditions"]) for row in rows),
                state_counts=dict(Counter(row["state"] for row in rows)),
                some_truth_samples_outside_complete_families=[r["task_id"] for r in rows if r["complete_but_some_truth_samples_outside"]],
                all_truth_samples_outside_complete_families=[r["task_id"] for r in rows if r["complete_but_all_truth_samples_outside"]],
                inference_elapsed_seconds=data["elapsed_seconds"], rows=rows)


def number(value, multiplier=1., digits=2):
    return "—" if value is None else f"{value * multiplier:.{digits}f}"


def range_text(value, multiplier=1., digits=2, count=True):
    if not value["available_count"]:
        return "—（0/4）" if count else "—"
    low, high = value["min"], value["max"]
    text = number(low, multiplier, digits)
    if low != high:
        text += "–" + number(high, multiplier, digits)
    return text + (f"（{value['available_count']}/4）" if count else "")


def percent_pair(row):
    return number(row["recovery_fraction"], 100., 1) + " / " + number(row["precision_fraction"], 100., 1)


def state_counts(dataset):
    return "、".join(STATES[state] + str(dataset["state_counts"].get(state, 0)) for state in STATES)


def table(rows):
    lines = [
        "| 条件 | 状态（可比/5） | p95：控制 / 四留组范围 mm | 控制R/P % | 四留组R / P范围 %（各有效数/4） | 最大盒对角 mm | 真值参数点包含 % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["label_zh"], STATES[row["state"]] + f"（{row['comparable_condition_count']}/5）",
            number(row["control"]["truth_to_prediction_p95_m"], 1000.) + " / " + range_text(row["fold_p95_m"], 1000.),
            percent_pair(row["control"]),
            range_text(row["fold_recovery"], 100., 1) + " / "
            + range_text(row["fold_precision"], 100., 1),
            number(row["maximum_box_diagonal_m"], 1000.),
            number(row["sampled_truth_containment_equal_segment_fraction"], 100., 1),
        ]) + " |")
    return lines


def figure(rows, destination):
    fig, ax = plt.subplots(figsize=(13.4, 10.3))
    fig.subplots_adjust(left=.285, right=.79, top=.88, bottom=.14)
    for i, row in enumerate(rows):
        error = row["control"]["truth_to_prediction_p95_m"]
        diagonal = row["maximum_box_diagonal_m"]
        for value, marker, color in ((error, "x", "#b34340"), (diagonal, "o", "#176b91")):
            if value is not None and value > 0:
                ax.scatter(1000 * value, i, marker=marker, color=color, s=42, linewidths=1.5, zorder=3)
        status = {"complete_empirical_envelope": "complete", "unresolved": "unresolved", "unavailable": "unavailable"}[row["state"]]
        notes = [status + f" ({row['comparable_condition_count']}/5)"]
        if error is None:
            notes.append("p95: NA")
        elif error == 0:
            notes.append("p95: 0")
        if diagonal is None:
            notes.append("box: NA")
        elif diagonal == 0:
            notes.append("box: 0")
        ax.text(1.02, i, "; ".join(notes), transform=ax.get_yaxis_transform(), va="center", fontsize=8.4, color="#535b65")
    positives = [v * 1000 for r in rows for v in
                 (r["control"]["truth_to_prediction_p95_m"], r["maximum_box_diagonal_m"])
                 if v is not None and v > 0]
    ax.set_xscale("log")
    if positives:
        ax.set_xlim(min(positives) / 1.8, max(positives) * 1.8)
    else:
        ax.set_xlim(.1, 1000.)
    ax.set_yticks(range(len(rows)), [r["case_id"] + "  " + r["label_en"] for r in rows], fontsize=8.7)
    ax.set_ylim(len(rows) - .35, -.65)
    ax.grid(axis="x", which="both", alpha=.16)
    ax.set_axisbelow(True)
    ax.set_xlabel("Millimetres (log scale); different quantities, same physical units", fontsize=10)
    ax.scatter([], [], color="#b34340", marker="x", s=42, label="Full-control target-to-curve p95")
    ax.scatter([], [], color="#176b91", marker="o", s=42, label="Largest empirical parameter-box diagonal")
    ax.legend(loc="lower left", bbox_to_anchor=(-.05, 1.02), frameon=False, fontsize=9)
    fig.suptitle("Empirical box size and finite-rod error", fontsize=15, y=.967)
    fig.text(.02, .045,
             "16 analytic challenges; endpoint correspondences supplied; common-axis TLS; synthetic camera initialization.\n"
             "One control-camera Sim3 per family. Complete means comparable perturbations, not accuracy or confidence.\n"
             "All 16 rows remain visible. Missing/zero values are labelled, never replaced by points on the logarithmic axis.",
             fontsize=9, va="bottom", color="#424a52")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_report(derived):
    replay, analytic = derived["replay"], derived["analytic"]
    all_rows = replay["rows"] + analytic["rows"]
    complete = [row for row in all_rows if row["state"] == "complete_empirical_envelope"]
    misses = [row for row in complete if row["complete_but_some_truth_samples_outside"]]
    zero = [row for row in complete if row["complete_but_all_truth_samples_outside"]]
    lines = [
        "# 9月24日：窄经验范围不能直接当准确性门",
        "",
        f"本轮固定保留14个历史杆方法族和16个解析反例，共150个条件行（70个回放、80个解析）。历史回放：{state_counts(replay)}；解析反例：{state_counts(analytic)}。",
        f"共{len(complete)}个族得到完整经验范围，其中{len(misses)}个并未包含全部预定真值参数点，{len(zero)}个一个真值参数点也没有包含。这里要求每段33个相同比例位置的真值点同时落入三坐标经验盒，除了浮点数容差不另加宽；这是很严格的参数位置检查，不是概率覆盖率，也不能据此说所有重建曲线都错了。",
        "例如c006的目标到预测曲线p95只有4.000 mm、25 mm容差下R/P均为100%，参数点包含仍为0%。同参数位置不同于到整条曲线的最近距离：端点略偏就会改变比例位置，方向、坐标系或共同偏差也可能让点落在薄盒外。当前实验没有校准这些误差。",
        "",
        "控制结果始终是原来的完整训练结果。四个删组结果只用来展示变化范围，没有选最好的一次、求平均替换控制，或用GT调一个“看起来够窄”的阈值。五个条件槽位都保留；缺失、相机扣留、身份拒绝、像素赋值改变以及有限段无法对应，都会留下原因并使范围未决或不可用。",
        "",
        "## 怎么读这些数",
        "",
        "- 完整只表示预定五条件都有可比较的相机、固定像素支持和有限段对应，不代表真实正确。",
        "- 表中p95是目标有限轴→预测曲线的距离95分位数；反方向的预测曲线→目标有限轴p95也完整保存在可读JSON条件行中，没有把两个方向混成一个数。R是在25mm容差内恢复的目标轴采样比例，P是落在目标轴25mm以内的预测曲线采样比例。它们是有限曲线诊断，不是点云恢复率；100%也只表示达到该容差。",
        "- 每个族只用完整控制相机拟合一次camera-only Sim3，所有fold沿用同一变换。对齐的残余误差和所有fold共享的偏差没有被这个变化范围消除，也没有作为不确定性单独加入。与前一份逐条件分别对齐的训练扰动报告口径不同，不能直接混表。",
        "- 经验盒在固定控制轴的正交坐标系中，比较每段33个相同比例位置。包含率按这些点计数，每段等权，包含端点；不是弧长覆盖率、距离管、体积或置信区间。",
        "- 表中的留组范围分别标注p95、R、P各自有效数/4。空预测的R=0是有效的漏检结果，纳入R范围；没有预测曲线时P未定义，保留缺失。只有无对齐或缺少度量等情况的R才是null，破折号不替代有效的0。五条件始终保留。",
        "",
        "## 14个历史方法族",
        "",
        "这些复用已知RGB、像素池与已计算相机；普通和圆柱方法各七行，没有新增图像或重跑DA3。扣留相机上的数值几何只作为诊断列出，不能发布补丁。",
        "",
        *table(replay["rows"]),
        "",
        "## 16个新解析反例",
        "",
        "这里直接给定跨视图端点对应，对三角化端点做公共直线TLS，同时保留端点和缺口对应。初始相机是预先规定的真实相机扰动，不是DA3预测。它检验相机传播与读数边界，不声称完成照片中杆的检测、身份辨认或真实对象泛化。",
        "",
        "各条件使用不同随机种子。近/远、噪声、背景或基线标签用于描述场景，不能把两行之差当作严格配对的单因素因果效应。",
        "",
        *table(analytic["rows"]),
        "",
        "![控制误差与经验盒大小](figures/2026-09-24-rod-camera-envelope.png)",
        "",
        "图保留全部16行。红叉为控制的目标→曲线p95，蓝点为完整范围中的最大参数位置盒对角；二者量纲相同但含义不同。横轴为毫米对数轴，缺失不画点，真实0另行标注，不塞到某个很小的正数。",
        "",
        "## 完整范围中的方向与缺口",
        "",
        "下表的方向是五个已观测方向之间的最大夹角。缺口是相邻预测有限段在固定控制轴上的间隔范围，只描述预测，不能证明那里真的没有物体。单段没有这一内部缺口指标。",
        "",
        "| 条件 | 最大两两方向角 ° | 预测缺口范围 mm | 最大同参数位移 / B % |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in complete:
        gaps = row["gap_ranges"]
        gap_text = "单段" if not gaps else "；".join(
            f"{g['left_control_segment_index']}→{g['right_control_segment_index']}: "
            + number(g["along_control_axis_m_min"], 1000.) + "–" + number(g["along_control_axis_m_max"], 1000.)
            for g in gaps)
        maximum = max((s["maximum_fractional_displacement_over_baseline"] for s in row["segments"]), default=None)
        lines.append(f"| {row['label_zh']} | {number(row['maximum_pairwise_direction_angle_degrees'], digits=4)} | {gap_text} | {number(maximum, 100., 4)} |")
    if not complete:
        lines.append("| 无完整可比较族 | — | — | — |")
    lines += [
        "",
        "## 一致错误输入的观察结果",
        "",
        "横向看错杆、看错深度、共享像素偏移与端点交换在运行前就已列为反例，不是跑完后按分数挑出来的。它们用来检查“所有fold可能一起错”的边界；不要求每个随机实例都呈现同一种数值趋势。",
        "",
    ]
    for row in analytic["rows"]:
        if row["family"] in {"coherent_wrong_rod_lateral", "coherent_wrong_rod_depth", "shared_frame_pixel_bias", "one_view_endpoint_swap"}:
            lines.append(
                f"- {row['label_zh']}：{STATES[row['state']]}；控制p95 {number(row['control']['truth_to_prediction_p95_m'], 1000., 3)} mm，"
                f"最大盒对角 {number(row['maximum_box_diagonal_m'], 1000., 3)} mm，"
                f"真值参数点包含 {number(row['sampled_truth_containment_equal_segment_fraction'], 100., 1)}%。")
    lines += ["", "## 完整但未包含全部真值参数点的族", ""]
    if misses:
        for row in misses:
            lines.append(f"- {row['label_zh']}：包含 {number(row['sampled_truth_containment_equal_segment_fraction'], 100., 1)}%，"
                         f"控制p95 {number(row['control']['truth_to_prediction_p95_m'], 1000.)} mm，盒对角 {number(row['maximum_box_diagonal_m'], 1000.)} mm。")
    else:
        lines.append("本次完整族的预定真值参数点均被包含；这仍不是其他噪声、其他相机误差或其他对象的覆盖保证。")
    lines += ["", "## 未决与不可用条件：没有从分母里消失", ""]
    for row in all_rows:
        if row["state"] == "complete_empirical_envelope":
            continue
        groups = defaultdict(list)
        for condition in row["incomplete_conditions"]:
            for reason in condition["reasons"]:
                groups[reason].append(condition["condition_id"])
        details = "；".join(", ".join(ids) + ": " + reason for reason, ids in groups.items())
        lines.append(f"- {row['label_zh']}（{STATES[row['state']]}，{row['comparable_condition_count']}/5可比）：{details}。")
    lines += [
        "",
        "这些是四次相关的训练扰动，既没有覆盖所有相机误差，也没有标定概率；杆的像素噪声、有限端点误差与共同偏差也不由它们单独覆盖。它们还能共享相同的对应错误和系统偏差。不能把本次23个零包含结果写成“统计覆盖概率为0”，也不能把某个方向的失配归因于某一种误差而不另做检验。范围窄、真值样本落入盒中或某次R/P达到100%，都不自动构成G1通过或物理正确保证。",
        "",
        "## 来源与复现",
        "",
        "此报告脚本只读取两份已完成的公开汇总，派生JSON保存各自SHA256；不重做拟合、对齐、匹配或评分。原始结果、失败记录和控制均保留。",
        f"源记录的推理耗时：历史回放{replay['inference_elapsed_seconds']:.3f}秒，解析反例{analytic['inference_elapsed_seconds']:.3f}秒；这里没有把准备、冻结、评价和检查时间算成推理时间。",
        "",
        "- [历史回放完整结果](results/2026-09-24-replay-rod-camera-envelope.json)",
        "- [解析反例完整结果](results/2026-09-24-analytic-rod-camera-envelope.json)",
        "- [五槽可读结果与来源SHA](results/2026-09-24-rod-camera-envelope-readable.json)",
        "- [报告生成脚本](../../scripts/report_rod_camera_envelope.py)",
        "",
        "输出采用只创建模式；复现时使用新的输出路径，不能覆盖本次产物。",
        "",
    ]
    return "\n".join(lines)


def run(sources, output, figure_path, report_path):
    for path in (output, figure_path, report_path):
        if path.exists():
            raise FileExistsError("Preserve existing report artifacts: " + str(path))
    loaded = {mode: read_source(path, mode, 14 if mode == "replay" else 16) for mode, path in sources.items()}
    derived = dict(schema_version="rod-camera-envelope-readable-v1",
                   sources={mode: pair[1] for mode, pair in loaded.items()},
                   replay=summarize(loaded["replay"][0], "replay"),
                   analytic=summarize(loaded["analytic"][0], "analytic"),
                   scope="Descriptive fixed-condition ranges, no algorithm changes, new thresholds, confidence intervals, or automatic promotion",
                   p95_direction="finite truth target to predicted curve",
                   fold_metric_policy="Fixed four rows; valid empty-prediction recovery=0 participates; undefined precision and missing metrics remain null; each range reports valid count",
                   box_containment_policy="33 same-fraction samples per segment, equal segment weight, endpoints included, not arc length")
    serialized = json.dumps(derived, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = make_report(derived)
    for path in (output, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure(derived["analytic"]["rows"], figure_path)
    # Source reports may be large, but a render is short; still check that none
    # changed while building the derived output.
    for mode, path in sources.items():
        if digest_bytes(path.read_bytes()) != derived["sources"][mode]["sha256"]:
            raise ValueError("Source changed during report generation")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    print(json.dumps(dict(output=str(output), figure=str(figure_path), report=str(report_path),
        replay_counts=derived["replay"]["state_counts"], analytic_counts=derived["analytic"]["state_counts"],
        replay_rows=len(derived["replay"]["rows"]), analytic_rows=len(derived["analytic"]["rows"])), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-source", type=Path, default=SOURCES["replay"])
    parser.add_argument("--analytic-source", type=Path, default=SOURCES["analytic"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(dict(replay=args.replay_source, analytic=args.analytic_source),
        args.output, args.figure, args.report)
