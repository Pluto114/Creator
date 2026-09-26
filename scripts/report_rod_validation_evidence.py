"""Summarize all paired evidence decisions without tuning the frozen method."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json

SOURCE = ROOT / 'docs/experiments/results/2026-09-26-rod-validation-evidence.json'
OUTPUT = ROOT / 'docs/experiments/results/2026-09-26-rod-validation-readable.json'
REPORT = ROOT / 'docs/experiments/2026-09-26-paired-target-evidence.md'
FIGURE = ROOT / 'docs/experiments/figures/2026-09-26-paired-target-evidence.png'
MODES = ('ordinary', 'endpoint_swap', 'coherent_lateral', 'coherent_depth')
ROLES = ('target', 'candidate_consistent_target', 'single_view_target')
ARMS = ('baseline', 'endpoints', 'target_claims', 'joint')
STATES = ('accepted', 'rejected', 'unresolved', 'withheld_camera')
NAMES = dict(ordinary='正确端点', endpoint_swap='一个视图交换端点', coherent_lateral='一致认错旁边杆', coherent_depth='一致认错深度')
CLAIMS = dict(target='额外目标点', candidate_consistent_target='跟着候选一起错的点', single_view_target='只提供一个视图')


def numeric(values):
    v = [x for x in values if x is not None]
    return dict(count=len(v), total=len(values), minimum=min(v) if v else None,
        median=float(np.median(v)) if v else None, maximum=max(v) if v else None,
        mean=float(np.mean(v)) if v else None)


def summarize(source):
    assert source['state'] == 'complete' and len(source['rows']) == 400
    assert source['paired_seed_count'] == 20 and source['decision_count'] == 4800
    keys = [(r['camera_group_id'], r['case_id'], r['condition']) for r in source['rows']]
    assert len(set(keys)) == 400
    controls = [r for r in source['rows'] if r['condition'] == 'control']
    assert len(controls) == 80
    tables, families, paired = [], [], []
    for mode in MODES:
        selected = [r for r in controls if r['family'] == mode]
        assert len(selected) == 20 and len({r['camera_group_id'] for r in selected}) == 20
        families.append(dict(family=mode,
            endpoint_states=dict(Counter(r['endpoint_check']['state'] for r in selected)),
            control_p95_m=numeric([r['geometry_metrics']['truth_to_prediction']['distance_p95'] if r['geometry_metrics'] else None for r in selected]),
            control_recovery=numeric([r['geometry_metrics']['recovery_fraction'] if r['geometry_metrics'] else None for r in selected]),
            endpoint_maximum_px=numeric([r['endpoint_check'].get('maximum_residual_px') for r in selected])))
        for role in ROLES:
            for arm in ARMS:
                entries = []
                for row in selected:
                    eid = next(e['evidence_id'] for e in row['evidence_labels'] if e['role'] == role)
                    result = next(d for d in row['decisions'] if d['evidence_id'] == eid and d['arm'] == arm)
                    entries.append(dict(camera_group_id=row['camera_group_id'], case_id=row['case_id'], **result))
                tables.append(dict(family=mode, role=role, arm=arm, seed_count=len(entries),
                    states={state: sum(e['state'] == state for e in entries) for state in STATES},
                    emitted_recovery=numeric([e['emitted_recovery'] for e in entries]),
                    emitted_precision=numeric([e['emitted_precision'] for e in entries]), seed_decisions=entries))
    for gid in sorted({r['camera_group_id'] for r in controls}):
        rows = [r for r in controls if r['camera_group_id'] == gid]
        assert {r['family'] for r in rows} == set(MODES)
        value = dict(camera_group_id=gid, outcomes=[])
        for row in sorted(rows, key=lambda r: r['family']):
            value['outcomes'].append(dict(family=row['family'], case_id=row['case_id'],
                camera=row['camera_decision']['state'], endpoint=row['endpoint_check']['state'],
                raw_geometry_metrics=row['geometry_metrics'], decisions=row['decisions'], evidence_labels=row['evidence_labels']))
        paired.append(value)
    consistency = []
    for case in sorted({r['case_id'] for r in source['rows']}):
        five = [r for r in source['rows'] if r['case_id'] == case]
        assert len(five) == 5 and len({r['condition'] for r in five}) == 5
        row = next(r for r in five if r['condition'] == 'control')
        for label in row['evidence_labels']:
            states = [next(d['state'] for d in r['decisions'] if d['evidence_id'] == label['evidence_id'] and d['arm'] == 'joint') for r in five]
            consistency.append(dict(case_id=case, family=row['family'], role=label['role'], states=states,
                all_five_same=len(set(states)) == 1, all_five_accepted=all(s == 'accepted' for s in states)))
    return dict(source_sha256=digest(SOURCE), source_run_id=source['run_id'], camera_fit_count=100,
        control_count=80, paired_seed_count=20, full_geometry_condition_count=400, full_decision_count=4800,
        control_tables=tables, geometry=families, per_seed_controls=paired, joint_fold_consistency=consistency,
        all_condition_states=dict(Counter(d['state'] for r in source['rows'] for d in r['decisions'])),
        elapsed_seconds=source['elapsed_seconds'], scope='Twenty paired seeds, not hundreds of independent objects. No thresholds selected from these outcomes.')


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8), sharey=True)
    colors = dict(accepted='#19846c', rejected='#ce5653', unresolved='#a3a9b3', withheld_camera='#7654a3')
    titles = ['Additional target claims', 'Candidate-consistent claims', 'Only one claimed view']
    for ax, role, title in zip(axes, ROLES, titles):
        entries = [next(r for r in summary['control_tables'] if (r['family'], r['role'], r['arm']) == (mode, role, 'joint')) for mode in MODES]
        bottom = np.zeros(4)
        for state in STATES:
            values = np.asarray([r['states'][state] for r in entries])
            ax.bar(np.arange(4), values, bottom=bottom, color=colors[state], label=state)
            for i, (base, v) in enumerate(zip(bottom, values)):
                if v:
                    ax.text(i, base + v / 2, str(int(v)), ha='center', va='center', fontsize=10, color='white' if state != 'unresolved' else 'black')
            bottom += values
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(4), ['Clean', 'Swap', 'Wrong\nlateral', 'Wrong\ndepth'])
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, 21, 5))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Fixed full-control seeds (20 per condition)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, .91), frameon=False)
    fig.suptitle('Endpoint + target projection checks: conditional decisions', fontsize=15, y=.98)
    fig.subplots_adjust(top=.76, bottom=.22, wspace=.15)
    fig.text(.04, .06, 'Synthetic pixel and target-claim component study; shared cameras/noise within each seed.\nAccepted means compatible with supplied evidence, not correct in 3D or automatic identity recognition.', fontsize=10, color='#4c5562')
    fig.savefig(path, dpi=165, bbox_inches='tight')
    plt.close(fig)


def text_range(values, scale=1.):
    if values['count'] == 0:
        return '—'
    return f"{values['median']*scale:.2f}（{values['minimum']*scale:.2f}–{values['maximum']*scale:.2f}；{values['count']}/{values['total']}）"


def report(summary):
    lines = ['# 9月26日：分开检查端点对应和目标声明', '',
        '这次先把比较条件锁在一起：20个随机种子，每个种子的四种杆观测共用同一背景、初始相机和二维噪声。相机只拟合100次，复用到400条杆条件；三套目标声明与四个检查组合产生4800个决策。独立重复单位仍是20个种子，不能把它写成4800个独立实验。', '',
        '方法在生成数据和评分前冻结，只跑这一版。正确端点、一个视图交换端点、一致认错横向杆、一致认错深度四种机制是预先指定的，不是从成绩中挑出来的。原来的五条件波动范围没有被当成通过门。', '',
        '## 两类像素证据分别检查什么', '',
        '端点检查对每个给定端点留出一个视图，用其余至少三个视图三角化，再预测留出像素；需要至少四个观测视图。2px预算沿用旧有限段的像素尺度，但这是新的残差语义，尚未被标定为准确性保证。它可以检验对应不一致，不能排除所有视图共同认错。', '',
        '目标检查把额外二维声明与已有有限杆段投影比较，不重拟合杆、不改变候选排序，也不把缺口连成整线。沿用2px预算和至少两不同中心视图，声明不确定矩形的外接圆给出保守距离上下界：整块均相容才supported，明确不相容才contradicted，其余unresolved。', '',
        '本轮目标点是合成测量，噪声流与端点分开，但共享潜在模拟场景。它不是人工用户标注实验，也没有实现从照片自动获得目标身份。跟随错候选的目标声明故意保留：若额外输入也认错，系统可能继续相容。缺少第二视图也必须留在分母中。', '',
        '## 完整训练控制：全部20个种子留在表里', '',
        '下表每格依次是接受 / 拒绝 / 未决 / 相机扣留，总和恒为20。只描述决策，不把接受叫物理正确。baseline同样受原相机门和几何可用性约束；没有从扣留条件里挑出好看的相机。', '',
        '| 杆观测 | 目标声明 | 原控制 | 只查端点 | 只查目标投影 | 两者同时查 |',
        '| --- | --- | --- | --- | --- | --- |']
    for mode in MODES:
        for role in ROLES:
            cells = []
            for arm in ARMS:
                r = next(t for t in summary['control_tables'] if (t['family'], t['role'], t['arm']) == (mode, role, arm))
                cells.append(' / '.join(str(r['states'][s]) for s in STATES))
            lines.append('| ' + ' | '.join([NAMES[mode], CLAIMS[role], *cells]) + ' |')
    lines += ['', 'clean和交换端点条件的“跟着候选”的声明与额外目标点字节相同，是重复等价对照；交换端点不凭空创造另一根物理杆。它们不算新增独立证据或样本。', '',
        '![固定种子的条件决策](figures/2026-09-26-paired-target-evidence.png)', '',
        '## 拒绝带来的损失也照算', '',
        '相机对齐只用每个种子的完整控制相机中心，得到一个Sim3后给四种机制、五个相机条件及所有声明共用。目标真值不参与拟合或对齐。下表先列未过滤几何的真值→曲线p95，再列joint最终输出的平均恢复率；被拒绝/未决/扣留是空输出，R=0、P未定义，不能把漏检删除后再平均。没有可用对齐而仍接受的结果保持未评分。', '',
        '| 杆观测 | 原控制p95：中位（范围；有值数）mm | joint额外目标点平均R % | joint跟随候选点平均R % |',
        '| --- | ---: | ---: | ---: |']
    for mode in MODES:
        g = next(r for r in summary['geometry'] if r['family'] == mode)
        cells = []
        for role in ROLES[:2]:
            v = next(t['emitted_recovery'] for t in summary['control_tables'] if (t['family'], t['role'], t['arm']) == (mode, role, 'joint'))
            cells.append('—' if v['mean'] is None else f"{100*v['mean']:.2f}（{v['count']}/{v['total']}）")
        lines.append('| ' + ' | '.join([NAMES[mode], text_range(g['control_p95_m'], 1000), *cells]) + ' |')
    lines += ['', 'R仍使用原25mm曲线容差及2mm采样，不是像素识别率。反方向p95、P、每个种子和全部五fold数据都在公开JSON中；图和本表没有挑最佳fold。纯几何是否准确与证据是否相容是两个不同问题。', '',
        '## 五条件一致性与保留边界', '',
        '| 杆观测 / 目标声明 | 五次决策全相同 | 五次均接受 |', '| --- | ---: | ---: |']
    for mode in MODES:
        for role in ROLES:
            selected = [r for r in summary['joint_fold_consistency'] if (r['family'], r['role']) == (mode, role)]
            assert len(selected) == 20
            lines.append(f"| {NAMES[mode]} / {CLAIMS[role]} | {sum(r['all_five_same'] for r in selected)}/20 | {sum(r['all_five_accepted'] for r in selected)}/20 |")
    lines += ['', '一致性仍不是准确性证书；这一张表只检查同样证据在既定相机扰动下是否改变决定。没有改端点、延伸有限段、替换初始模型或生成新补丁。G1仍需要实际采集、独立对象、可靠共同读取与错误代价的证据。', '',
        '## 复现与来源', '',
        f"共享相机推理阶段耗时{summary['elapsed_seconds']:.2f}秒，不含准备、冻结、真值评价和审计。", '',
        '- [冻结完整结果](results/2026-09-26-rod-validation-evidence.json)',
        '- [20种子可读对照](results/2026-09-26-rod-validation-readable.json)',
        '- [独立审计](results/2026-09-26-rod-validation-evidence-audit.json)',
        '- [运行入口](../../scripts/run_rod_validation_evidence.py)',
        '- [协议](../../configs/camera_evidence_challenges_v1.json)', '',
        '加载Enter-CreatorEnvironment.ps1，用D盘DA3环境执行入口的prepare、infer、pre、evaluate。正式run拒绝覆盖，源码已有快照；重新设计须另开版本。外部审计在评价前保存真实before、评价后检查来源和配对。旧实验与失败记录完整保留。', '']
    return '\n'.join(lines)


def run(output, figure, markdown):
    for path in (output, figure, markdown):
        if path.exists():
            raise FileExistsError('Keep existing report: ' + str(path))
    summary = summarize(read_json(SOURCE))
    summary['report_source_sha256'] = digest(Path(__file__))
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, figure)
    assert digest(SOURCE) == summary['source_sha256']
    write_json(output, summary)
    markdown.write_text(report(summary), encoding='utf-8', newline='\n')
    print('REPORTED', len(summary['control_tables']), 'paired 20-seed cells;', len(summary['joint_fold_consistency']), 'five-condition families', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--figure', type=Path, default=FIGURE)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.output, args.figure, args.report)
