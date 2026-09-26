"""Publish new-view validation, including failures that still pass pixel checks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json

SOURCE = ROOT / 'docs/experiments/results/2026-09-26-heldout-view-validation.json'
OUT = ROOT / 'docs/experiments/results/2026-09-26-heldout-view-readable.json'
REPORT = ROOT / 'docs/experiments/2026-09-26-heldout-rgb-views.md'
FIGURES = ROOT / 'docs/experiments/figures'
SCENE = ROOT / '.runtime/experiments/rod-heldout-view-scenes-v1-20260926'
ANNOTATIONS = ROOT / 'configs/rod_heldout_views_annotations_v1.json'


def numeric(values):
    valid = [x for x in values if x is not None]
    return dict(total=len(values), count=len(valid), minimum=min(valid) if valid else None,
        median=float(np.median(valid)) if valid else None, maximum=max(valid) if valid else None)


def stratum(row):
    metrics = row['physical']['metrics']
    a, b = metrics['truth_to_prediction']['distance_p95'], metrics['prediction_to_truth']['distance_p95']
    if a is None or b is None:
        return 'unscored'
    return 'within_25mm_both_p95' if max(a, b) <= .025 else 'exceeds_25mm_any_p95'


def eligible(row):
    return row['old_identity_state'] == 'accepted' and row['old_camera_state'] == 'candidate_camera_correction'


def summarize(result):
    assert result['state'] == 'complete' and len(result['rows']) == len(result['pose_rows']) == 60
    pairs = [(r['task_id'], r['condition']) for r in result['rows']]
    assert len(set(pairs)) == 60
    cells = []
    states = ('retained_candidate', 'rejected_new_target', 'unresolved_new_target', 'unresolved_new_pose')
    for control_only in (False, True):
        pool = [r for r in result['rows'] if eligible(r) and (not control_only or r['condition'] == 'control')]
        for label in ('within_25mm_both_p95', 'exceeds_25mm_any_p95', 'unscored'):
            subset = [r for r in pool if stratum(r) == label]
            cells.append(dict(control_only=control_only, physical_stratum=label, total=len(subset),
                states={s: sum(r['state'] == s for r in subset) for s in states}))
    groups = []
    for parent in sorted({r['parent'] for r in result['rows']}):
        for case in ('r01', 'r02', 'r03'):
            pool = [r for r in result['rows'] if r['parent'] == parent and r['case_id'] == case]
            assert len(pool) == 10
            groups.append(dict(parent=parent, case_id=case, count=10, eligible=sum(eligible(r) for r in pool),
                states=dict(Counter(r['state'] for r in pool)),
                maximum_target_distance_px=numeric([r['target_check']['maximum_distance_px'] for r in pool]),
                physical_p95_m=numeric([r['physical']['metrics']['truth_to_prediction']['distance_p95'] for r in pool])))
    poses = result['pose_rows']
    return dict(source_sha256=digest(SOURCE), report_source_sha256=digest(Path(__file__)),
        strata=cells, groups=groups, states=dict(Counter(r['state'] for r in result['rows'])),
        pose_states=dict(Counter(r['verification']['state'] for r in poses)),
        pose_validation_p95_px=numeric([r['verification']['validation']['summary']['p95_px'] if r['verification']['validation'] else None for r in poses]),
        pose_center_error_m=numeric([r['center_error_m'] for r in poses]),
        pose_rotation_error_degrees=numeric([r['rotation_error_degrees'] for r in poses]),
        pose_train_count=numeric([len(r['proposal']['training_track_ids'] or []) for r in poses]),
        pose_validation_count=numeric([len(r['verification']['validation_track_ids'] or []) for r in poses]),
        empty_outputs=sum(r['state'] != 'retained_candidate' for r in result['rows']),
        elapsed_seconds=result['elapsed_seconds'], scope=result['scope'])


def plots(result, rgb_path, error_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = read_json(ROOT / 'data/inputs/rod-heldout-view-scenes-v1-20260926/manifest.json')
    queries = read_json(ANNOTATIONS)
    fig, axes = plt.subplots(3, 2, figsize=(10, 11.5))
    for case, row in zip(manifest['cases'], axes):
        query = next(q for q in queries['queries'] if q['case_id'] == case['case_id'])
        for frame, ax in zip(case['frames'], row):
            assert digest(ROOT / frame['rgb']) == frame['rgb_sha256']
            with Image.open(ROOT / frame['rgb']) as im:
                ax.imshow(np.asarray(im.convert('RGB')))
            point = next(a for a in query['anchors'] if a['view_id'] == frame['view_id'])
            x, y = point['xy']
            ax.plot(x, y, '+', color='#ff7348', ms=11, mew=1.5)
            ax.annotate(f'({x:g}, {y:g})', (x, y), xytext=(15, 8), textcoords='offset points', color='#ffd3bd', fontsize=9)
            ax.set_title(case['case_id'] + ' / ' + frame['view_id'])
            ax.set_axis_off()
    fig.suptitle('Six new rendered RGB views; target clicks frozen before localization', fontsize=13)
    fig.tight_layout(rect=(0, .02, 1, .965))
    fig.text(.03, .015, 'Blender reference-board scenes, not real photographs or new independent objects. Crosses are RGB membership claims.', fontsize=9)
    fig.savefig(rgb_path, dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = dict(r01='#2b8c74', r02='#d28a27', r03='#b14e68')
    for case in colors:
        selected = [r for r in result['rows'] if r['case_id'] == case and eligible(r)]
        xs, ys = [], []
        for row in selected:
            metric = row['physical']['metrics']
            xs.append(1000*max(metric['truth_to_prediction']['distance_p95'], metric['prediction_to_truth']['distance_p95']))
            bounds = [a['maximum_distance_bound_px'] for a in row['target_check']['per_anchor']]
            ys.append(max(bounds) if all(v is not None for v in bounds) else np.nan)
        ax.scatter(xs, ys, label=case + ' (correlated methods/folds)', color=colors[case], alpha=.65, s=44)
    ax.axvline(25, color='#6b7280', linestyle='--', label='Existing 25 mm physical readout')
    ax.axhline(2, color='#934a3a', linestyle=':', label='Fixed target pixel budget')
    ax.set_xlabel('Worse of the two physical curve p95 distances (mm)')
    ax.set_ylabel('Worst new-view target distance upper bound (px)')
    ax.set_ylim(bottom=0, top=max(2.25, ax.get_ylim()[1]))
    ax.grid(alpha=.2)
    ax.set_title('A new-image pixel check can still pass a physically inaccurate rod')
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.text(.04, .025, 'Only old eligible candidates shown; all original refusals remain in the full table. PnP uses the old estimated map and K.', fontsize=9)
    fig.savefig(error_path, dpi=160)
    plt.close(fig)


def bounds(value, scale=1.):
    if not value['count']:
        return '未评分'
    return f"{value['minimum']*scale:.3f}–{value['maximum']*scale:.3f}（中位{value['median']*scale:.3f}；{value['count']}/{value['total']}）"


def markdown(result, summary):
    lines = ['# 9月26日续接：新照片进来了，但定位新照片仍会继承旧地图偏差', '',
        '本轮新增6张Blender RGB，复用已有完整杆、断杆和细杆三个物体，额外拍两个未参与原重建的视角。旧同高度/变高度两条采集路径各有完整控制和四个固定删组，合计30份地图；每份地图定位两个新视图，再检查普通/圆柱两种旧杆，共60个联合决策和120个单视图投影槽。它们相互关联，不是60个独立物体，也不是实拍。', '',
        '所有60个新相机通过预先冻结的背景验证。原51个可继续审查的杆候选也全部通过新目标投影；其中27个双向p95均≤25mm，24个至少一向超出，新增拦截仍是0。另9个原身份不接受仍不接受。新增图像没有自动带来可用的三维筛错能力。', '',
        '## 新图像与目标点，如何隔开', '',
        '新视角相对既有场景只改变角度与高度，场景网格、纹理和光照保持原样；正常输入只含匿名view_h00/view_h01、RGB、SHA及固定屏幕guide。精确相机、深度、ID、可见性和网格独立保存于GT目录，推理禁止读取。生成前冻结显式源码闭包，正常算法另行冻结，没有抓取并行工作中的半成品。', '',
        '六张原图逐一目视检查，目标点选在可见杆的y=135行，随后仅核查该行RGB像素来记录x位置。点击来自助手查看RGB，不是GT投影、模型覆盖图或人类用户实验。半像素矩形只是坐标格表示，不是经过标定的点击误差界；点表示可见杆成员，不能自动解释为物理轴上的对应端点。点击在任何新PnP或杆投影之前已封存。', '',
        '![新RGB与冻结点击](figures/2026-09-26-heldout-rgb.png)', '',
        '## 新相机只由背景定位，旧杆和旧地图不动', '',
        '用原SIFT参数重新提取五张旧图，逐feature坐标/顺序必须与冻结旧track数据完全相同。新图屏幕guide两侧64px内不提供定位特征；匹配采用固定0.75双向比率，一个新feature必须由至少两个旧视图一致指向同一个旧track。冲突或多个新feature争用同一track一律排除，不按拟合分数选赢家。', '',
        '定位点只取旧BA保留的训练三维点，既不重拟合也不换地图。只有旧train且新图y≥192的对应进入PnP；旧validation且新图y<128的点只用于验证，其三维位置只从旧验证像素三角化。中间带与角色不相容的观测明确排除。旧K的焦距几何均值与主点均值给新图使用，是固定镜头假设，不是GT内参。', '',
        'PnP采用固定2px RANSAC、5000次预算、0.99置信参数和固定随机种子；至少24训练点/24内点，三维点及内点不能近共面。LM只用训练内点。验证至少8点，p95≤2px且2px内比例≥0.8，正深度与来源检查同时满足。阈值没有按这批分数调。数值姿态和验证决定分开保存，验证不能改pose；新目标点从未进入定位。', '',
        '这仍不是独立绝对相机标定：旧地图、旧K和旧验证点都依赖同一旧相机解，误差可以共同传递。SIFT有像素邻域支撑，中心排除条和y分带也不等于每一个底层像素严格互不相交。这里只证明新观测ID和数值拟合角色隔离。', '',
        '## 完整读数和遗漏代价', '',
        '| 范围 | 原物理读数 | 原eligible数 | 保留 | 新目标拒绝 | 新目标未决 | 新相机未决 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for cell in summary['strata']:
        label = {'within_25mm_both_p95': '双向p95≤25mm', 'exceeds_25mm_any_p95': '至少一向p95>25mm', 'unscored': '未评分'}[cell['physical_stratum']]
        lines.append('| ' + ' | '.join(['仅完整控制' if cell['control_only'] else '全部五条件', label, str(cell['total']), *[str(v) for v in cell['states'].values()]]) + ' |')
    lines += ['', '这两个物理分组沿用旧25mm读数，只作诊断，不等于整条杆正确性或G1验收。旧几何、像素赋值与控制相机Sim3都完全不变；不能逐fold或新视图重新对齐来消掉偏差。拒绝/未决是空输出，R=0、P未定义。', '',
        '![新图像投影与旧三维误差](figures/2026-09-26-heldout-pixel-vs-physical.png)', '',
        f"60次定位的训练对应数为{bounds(summary['pose_train_count'])}，验证对应数为{bounds(summary['pose_validation_count'])}；背景验证p95为{bounds(summary['pose_validation_p95_px'])}px。", '',
        f"评价阶段再看真值，沿各旧完整控制的固定Sim3，新视图中心误差为{bounds(summary['pose_center_error_m'], 1000)}mm，旋转误差为{bounds(summary['pose_rotation_error_degrees'])}度。这里没有把新相机重新对齐到GT。", '',
        '## 这次结果决定什么', '',
        '单纯再收集视图、用旧地图定位、再核一次同目标投影，尚不能承担三维精度验收。下一步要分清已知内参、独立相机定位和额外目标观测分别能约束什么；参考板若提供已知尺寸或坐标，必须作为明确新增的采集输入，不能从GT悄悄复制。仅把误差阈值调紧没有本轮证据支持。', '',
        '为辨别机制，另做[特权相机替换诊断](results/2026-09-26-heldout-view-oracle.json)：它是在看到正常流程仍全保留之后提出的事后评价，对同一固定杆/控制Sim3分别替换新相机的K或位姿，不是正常方法的可用输入，不用于调这轮门槛。正常结果保持封存。', '',
        '## 检查与产物', '',
        f"新视图匹配、PnP和杆检查合计{summary['elapsed_seconds']:.3f}秒，复用旧DA3与旧BA；不含渲染、GT、冻结和审计，不能当完整系统耗时。", '',
        '- [60相机与60联合杆完整结果](results/2026-09-26-heldout-view-validation.json)',
        '- [可读统计与来源](results/2026-09-26-heldout-view-readable.json)',
        '- [独立评分前后审计](results/2026-09-26-heldout-view-audit.json)',
        '- [新图像生成入口](../../scripts/prepare_rod_heldout_views.py)',
        '- [验证入口](../../scripts/run_rod_heldout_views.py)',
        '- [训练独立截面诊断](2026-09-26-section-model-trainonly.md)', '',
        '复现先加载项目D盘环境。新图生成与点击封存独立在先；验证入口prepare/infer/pre，独立外审pre通过后evaluate，再外审post。旧run与公共结果拒绝覆盖。所有数据、源码快照和失败记录仍保留，G1未通过，无新自动补丁或UI扩建。', '']
    return '\n'.join(lines)


def main():
    rgb_path = FIGURES / '2026-09-26-heldout-rgb.png'
    error_path = FIGURES / '2026-09-26-heldout-pixel-vs-physical.png'
    for path in (OUT, REPORT, rgb_path, error_path):
        if path.exists():
            raise FileExistsError('Preserve existing report: ' + str(path))
    result = read_json(SOURCE)
    summary = summarize(result)
    generation = read_json(SCENE / 'prepared.json')
    summary['generation_prepared_sha256'] = digest(SCENE / 'prepared.json')
    summary['independent_ray_samples'] = generation['independent_ray_samples']
    summary['surface_id_mismatches'] = generation['surface_id_mismatches']
    plots(result, rgb_path, error_path)
    assert digest(SOURCE) == summary['source_sha256']
    write_json(OUT, summary)
    REPORT.write_text(markdown(result, summary), encoding='utf-8', newline='\n')
    print('NEW_VIEW_REPORT 60 poses/60 rods; six source RGB shown; all fixed-fold strata retained', flush=True)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
