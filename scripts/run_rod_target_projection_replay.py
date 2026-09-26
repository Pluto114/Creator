"""Recheck frozen RGB target clicks against unchanged finite rod projections.

The clicks already selected the target: this is same-source consistency, not
independent validation. Inference never opens physical scores or truth. Evaluation
joins the existing single-control-camera Sim3 scores without fitting anything.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter

from PIL import Image
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, checked_run, digest, read_json, write_json
from run_rod_camera_envelope import checked as checked_envelope
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.rod_validation_evidence import check_target_anchors

RUN_ID = 'rod-target-projection-replay-v1-20260926'
RUN = ROOT / '.runtime/experiments' / RUN_ID
INPUTS = ROOT / 'data/inputs' / RUN_ID
SOURCE_SUMMARY = ROOT / 'docs/experiments/results/2026-09-24-replay-rod-camera-envelope.json'
PUBLIC = ROOT / 'docs/experiments/results/2026-09-26-target-projection-replay.json'
REPORT = ROOT / 'docs/experiments/2026-09-26-target-projection-replay.md'
CONDITIONS = ['control', *['leave_group_' + str(i) for i in range(4)]]
TRAINING_INPUT = ROOT / 'data/inputs/camera-training-sensitivity-v1-20260924r2/manifest.json'
NEW_SOURCES = ['scripts/run_rod_target_projection_replay.py',
    'experiments/src/creator_eval/rod_validation_evidence.py', 'tests/test_rod_validation_evidence.py']
POLICY = dict(threshold_px=2.0, minimum_anchor_views=2,
    supported='distance + norm(uncertainty_xy_px) <= 2',
    contradicted='max(0, distance - norm(uncertainty_xy_px)) > 2',
    old_rejection_preserved=True, camera_withholding_preserved=True,
    geometry_modified=False, independence='same_source_clicks_already_used_for_identity_selection',
    evaluation='Read frozen Sep24 physical scores with one control-camera Sim3 per family; no realignment',
    physical_strata='Existing 25mm tolerance: both directional p95 within tolerance; separately R/P both 100%; diagnostic only')


def receipt(path):
    return path.relative_to(ROOT).as_posix(), digest(path)


def no_evaluation_open(event, arguments):
    reject_truth_open(event, arguments)
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        from pathlib import Path
        raw = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
        path = Path(raw).resolve()
        if path.is_relative_to(ROOT / 'docs/experiments/results'):
            raise PermissionError('Inference cannot read public physical-score summaries')


def validate_manifest(manifest):
    tasks = manifest['tasks']
    ids = [task['task_id'] for task in tasks]
    assert len(ids) == len(set(ids)) == 14
    cases = Counter((task['parent'], task['case_id']) for task in tasks)
    assert len(cases) == 7 and set(cases.values()) == {2}
    for key in cases:
        assert {t['method'] for t in tasks if (t['parent'], t['case_id']) == key} == {'baseline', 'cylinder_support'}
    for task in tasks:
        source = task['source_task']
        assert all(task[name] == source[name] for name in ('task_id', 'parent', 'case_id', 'method'))
        assert canonical_hash(source) == task['source_task_sha256']
        frames = task['frames']
        assert [f['view_id'] for f in frames] == task['view_ids'] and len(frames) == 5
        assert len(set(task['view_ids'])) == 5
        assert [p['condition_id'] for p in source['packets']] == CONDITIONS
        assert len(task['anchors']) == 2 and len({a['view_id'] for a in task['anchors']}) == 2
        assert task['query']['case_id'] == task['case_id']
        expected = [dict(view_id=a['view_id'], xy=a['xy'], uncertainty_xy_px=a['uncertainty_xy_px'],
                         source_sha256=a['rgb_sha256']) for a in task['query']['anchors']]
        assert task['anchors'] == expected
        by_id = {f['view_id']: f for f in frames}
        for anchor in task['anchors']:
            assert anchor['source_sha256'] == by_id[anchor['view_id']]['source_sha256']
        for packet in source['packets']:
            for name in ('intrinsics', 'extrinsics'):
                assert packet[name] is None or len(packet[name]) == 5


def prepare():
    if RUN.exists() or INPUTS.exists() or PUBLIC.exists() or REPORT.exists():
        raise FileExistsError('Preserve prior attempts; use a new run ID')
    source_run, source_config, old = checked_envelope('replay')
    source_index = read_json(source_run / 'inference.json')
    assert source_index['state'] == 'complete' and not source_index['gt_read_during_inference']
    assert source_index['config_sha256'] == digest(source_run / 'method_config.json')
    assert source_index['input_sha256'] == source_config['input_sha256']
    assert digest(TRAINING_INPUT) == source_config['receipts'][TRAINING_INPUT.relative_to(ROOT).as_posix()]
    training = read_json(TRAINING_INPUT)
    by_case = {(t['parent'], t['case_id']): t for t in training['tasks']}
    assert len(by_case) == len(training['tasks']) == 7
    family_receipts = {r['task_id']: r for r in source_index['families']}
    assert len(family_receipts) == len(source_index['families']) == 14
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    (RUN / 'records').mkdir()
    sources = source_snapshot(RUN, set(source_config['source_sha256']) | set(NEW_SOURCES))
    write_json(RUN / 'source_freeze.json', dict(source_sha256=sources, policy=POLICY,
        physical_scores_not_parsed=True, inference_not_started=True))
    # Pin evaluation bytes now, but do not parse or expose their numeric values
    # to the input manifest or normal inference. The method is already frozen.
    eval_receipt = dict(path=SOURCE_SUMMARY.relative_to(ROOT).as_posix(), sha256=digest(SOURCE_SUMMARY))
    sys.addaudithook(no_evaluation_open)
    receipts = dict(source_config['receipts'])
    receipts.update(receipt(p) for p in [source_run / 'method_config.json', source_run / 'source_freeze.json',
        source_run / 'inference.json', ROOT / 'data/inputs' / source_config['run_id'] / 'manifest.json'])
    tasks, cached = [], {}
    for source in old['tasks']:
        key = source['parent'], source['case_id']
        if key not in cached:
            parent, inputs, frozen, manifest = checked_run(source['parent'])
            annotation_receipt = read_json(parent / 'annotations-frozen.json')
            assert annotation_receipt['before_prediction']
            assert annotation_receipt['input_sha256'] == frozen['input_sha256']
            assert annotation_receipt['sha256'] == digest(parent / 'annotations.json')
            annotations = read_json(parent / 'annotations.json')
            assert not annotations['ground_truth_used_for_coordinates'] and not annotations['predicted_geometry_used_for_coordinates']
            query = next(q for q in annotations['queries'] if q['case_id'] == source['case_id'])
            assert query == by_case[key]['query']
            case = next(c for c in manifest['cases'] if c['case_id'] == source['case_id'])
            assert [f['view_id'] for f in case['frames']] == by_case[key]['view_ids']
            frames = []
            for frame in case['frames']:
                image = ROOT / frame['rgb']
                assert digest(image) == frame['rgb_sha256']
                with Image.open(image) as handle:
                    assert list(handle.size) == frame['size_wh']
                receipts[frame['rgb']] = frame['rgb_sha256']
                frames.append(dict(view_id=frame['view_id'], rgb=frame['rgb'], size_wh=frame['size_wh'],
                    source_sha256=frame['rgb_sha256'], source_kind='rgb'))
            receipts.update(receipt(p) for p in [inputs / 'manifest.json', parent / 'prepared.json',
                                                parent / 'annotations.json', parent / 'annotations-frozen.json'])
            cached[key] = dict(frames=frames, view_ids=by_case[key]['view_ids'], query=query,
                              annotation_origin=annotations['annotation_origin'])
        entry = family_receipts[source['task_id']]
        path = source_run / entry['path']
        assert digest(path) == entry['sha256']
        assert read_json(path)['task'] == source
        receipts[path.relative_to(ROOT).as_posix()] = entry['sha256']
        data = copy.deepcopy(cached[key])
        anchors = [dict(view_id=a['view_id'], xy=a['xy'], uncertainty_xy_px=a['uncertainty_xy_px'],
                        source_sha256=a['rgb_sha256']) for a in data['query']['anchors']]
        tasks.append(dict(**{n: source[n] for n in ('task_id', 'parent', 'case_id', 'method')}, **data,
            anchors=anchors, source_task=source, source_task_sha256=canonical_hash(source),
            source_family_path=path.relative_to(ROOT).as_posix(), source_family_sha256=entry['sha256']))
    manifest = dict(tasks=tasks, expected_family_count=14, expected_condition_count=70,
        scope='Seven known DA3 RGB cases, two rod methods, full plus four training folds; reused target clicks')
    validate_manifest(manifest)
    write_json(INPUTS / 'manifest.json', manifest)
    write_json(RUN / 'method_config.json', dict(run_id=RUN_ID, source_sha256=sources, policy=POLICY,
        input_sha256=digest(INPUTS / 'manifest.json'), source_freeze_sha256=digest(RUN / 'source_freeze.json'),
        receipts=receipts, evaluation_source=eval_receipt, parent_run_id=source_config['run_id'],
        gt_read_during_inference=False, no_automatic_promotion=True))
    print('PREPARED', len(tasks), 'families / 70 conditions', flush=True)


def checked(*, evaluation=False):
    config = read_json(RUN / 'method_config.json')
    assert config['run_id'] == RUN_ID and config['policy'] == POLICY
    assert not config['gt_read_during_inference'] and config['no_automatic_promotion']
    assert digest(INPUTS / 'manifest.json') == config['input_sha256']
    assert digest(RUN / 'source_freeze.json') == config['source_freeze_sha256']
    for name, sha in config['source_sha256'].items():
        assert digest(ROOT / name) == sha == digest(RUN / 'source_snapshot' / name), name
    for name, sha in config['receipts'].items():
        assert digest(ROOT / name) == sha, name
    if evaluation:
        item = config['evaluation_source']
        assert digest(ROOT / item['path']) == item['sha256']
    manifest = read_json(INPUTS / 'manifest.json')
    validate_manifest(manifest)
    for task in manifest['tasks']:
        assert read_json(ROOT / task['source_family_path'])['task'] == task['source_task']
        for frame in task['frames']:
            assert config['receipts'][frame['rgb']] == frame['source_sha256']
    return config, manifest


def infer_row(task, packet):
    frames = []
    for i, frame in enumerate(task['frames']):
        frames.append(dict(frame, K_index=packet['intrinsics'][i] if packet['intrinsics'] is not None else None,
            world_to_camera_cv=packet['extrinsics'][i] if packet['extrinsics'] is not None else None))
    before = canonical_hash(dict(frames=frames, packet=packet, anchors=task['anchors']))
    check = check_target_anchors(frames, packet['identity'].get('segments', []), task['anchors'])
    assert before == canonical_hash(dict(frames=frames, packet=packet, anchors=task['anchors']))
    identity_state, camera_state = packet['identity']['state'], packet['camera_decision']['state']
    eligible = identity_state == 'accepted' and camera_state == 'candidate_camera_correction'
    old_state = ('eligible_candidate' if eligible else
                 ('original_identity_not_accepted' if identity_state != 'accepted' else 'withheld_by_camera'))
    after_state = old_state
    if eligible:
        after_state = {'supported': 'retained_candidate', 'contradicted': 'blocked_by_projection',
                       'unresolved': 'unresolved_projection'}[check['state']]
    return dict(task_id=task['task_id'], parent=task['parent'], case_id=task['case_id'], method=task['method'],
        condition_id=packet['condition_id'], packet_sha256=canonical_hash(packet),
        old_identity_state=identity_state, old_camera_state=camera_state, old_state=old_state,
        old_candidate_eligible=eligible, after_state=after_state,
        retained_candidate=eligible and check['state'] == 'supported', projection_check=check,
        geometry_sha256_before=canonical_hash(packet['identity'].get('segments', [])),
        geometry_sha256_after=canonical_hash(packet['identity'].get('segments', [])),
        evidence_scope='same_source_not_independent', geometry_modified=False, gt_read_during_inference=False)


def infer():
    sys.addaudithook(no_evaluation_open)
    config, manifest = checked()
    if (RUN / 'inference.json').exists():
        raise FileExistsError('Preserve frozen inference')
    started = time.perf_counter()
    receipts = []
    for task in manifest['tasks']:
        rows = [infer_row(task, packet) for packet in task['source_task']['packets']]
        assert len(rows) == 5
        path = RUN / 'records' / (task['task_id'] + '.json')
        write_json(path, dict(task_id=task['task_id'], task_sha256=canonical_hash(task), rows=rows,
            gt_read_during_inference=False))
        receipts.append(dict(task_id=task['task_id'], path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
    checked()
    write_json(RUN / 'inference.json', dict(state='complete', run_id=RUN_ID, families=receipts,
        config_sha256=digest(RUN / 'method_config.json'), input_sha256=config['input_sha256'],
        elapsed_seconds=time.perf_counter() - started, gt_read_during_inference=False))
    print('INFERRED', len(receipts), 'families / 70 conditions', flush=True)


def inference_rows(config, manifest):
    index = read_json(RUN / 'inference.json')
    assert index['state'] == 'complete' and not index['gt_read_during_inference']
    assert index['config_sha256'] == digest(RUN / 'method_config.json')
    assert index['input_sha256'] == config['input_sha256']
    expected = {task['task_id']: task for task in manifest['tasks']}
    assert len(index['families']) == len(expected) == 14
    assert {f['task_id'] for f in index['families']} == set(expected)
    rows = []
    for item in index['families']:
        path = RUN / item['path']
        assert digest(path) == item['sha256']
        record = read_json(path)
        task = expected[item['task_id']]
        assert record['task_id'] == item['task_id'] and record['task_sha256'] == canonical_hash(task)
        assert not record['gt_read_during_inference']
        fresh = [infer_row(task, packet) for packet in task['source_task']['packets']]
        assert fresh == record['rows']
        rows.extend(record['rows'])
    assert len(rows) == len({(r['task_id'], r['condition_id']) for r in rows}) == 70
    assert all(r['geometry_sha256_before'] == r['geometry_sha256_after'] for r in rows)
    assert not any(r['retained_candidate'] and not r['old_candidate_eligible'] for r in rows)
    return index, rows


def physical_class(metrics):
    if metrics is None:
        return dict(p95='unscored', complete_coverage='unscored')
    if metrics['prediction_to_truth']['source_length'] <= 0:
        assert metrics['recovery_fraction'] == 0 and metrics['precision_fraction'] is None
        return dict(p95='empty_prediction', complete_coverage='empty_prediction')
    tolerance = metrics['tolerance']
    assert tolerance == .025
    distances = [metrics[name]['distance_p95'] for name in ('truth_to_prediction', 'prediction_to_truth')]
    p95 = ('both_within_25mm' if all(v <= tolerance for v in distances) else 'at_least_one_exceeds_25mm')
    complete = all(metrics[name] >= 1. - 1e-12 for name in ('recovery_fraction', 'precision_fraction'))
    return dict(p95=p95, complete_coverage='both_RP_100' if complete else 'some_uncovered_samples')


def evaluation_rows(rows, manifest, config):
    old = read_json(ROOT / config['evaluation_source']['path'])
    source_run = ROOT / '.runtime/experiments' / config['parent_run_id']
    assert old['state'] == 'complete' and old['mode'] == 'replay' and old['run_id'] == config['parent_run_id']
    assert old['config'] == read_json(source_run / 'method_config.json')
    assert old['inference_sha256'] == digest(source_run / 'inference.json')
    families = {r['task_id']: r for r in old['rows']}
    assert len(families) == len(old['rows']) == len(manifest['tasks']) == 14
    assert set(families) == {t['task_id'] for t in manifest['tasks']}
    physical = {}
    for task in manifest['tasks']:
        family = families[task['task_id']]
        assert all(family[name] == task[name] for name in ('parent', 'case_id', 'method'))
        package = family['physical']
        assert package.get('alignment_reused_for_all_conditions', package['alignment'] is None)
        assert [r['condition_id'] for r in package['rows']] == CONDITIONS
        for row in package['rows']:
            packet = next(p for p in task['source_task']['packets'] if p['condition_id'] == row['condition_id'])
            if row['metrics'] is not None:
                assert row['identity_state'] == packet['identity']['state']
                assert row['camera_decision'] == packet['camera_decision']['state']
            physical[task['task_id'], row['condition_id']] = row
    result = []
    for row in rows:
        scored = physical[row['task_id'], row['condition_id']]
        result.append(dict(row, physical=copy.deepcopy(scored), physical_class=physical_class(scored['metrics'])))
    return result


def summarise(rows):
    bins = []
    for subset, selected in [('all_conditions', rows), ('control_only', [r for r in rows if r['condition_id'] == 'control'])]:
        eligible = [r for r in selected if r['old_candidate_eligible']]
        for metric in ('p95', 'complete_coverage'):
            counts = Counter((r['physical_class'][metric], r['after_state']) for r in eligible)
            for quality in sorted({r['physical_class'][metric] for r in eligible}):
                for after in ('retained_candidate', 'blocked_by_projection', 'unresolved_projection'):
                    bins.append(dict(subset=subset, stratum=metric, physical_class=quality, after_state=after,
                        count=counts[quality, after], old_eligible_count=len(eligible),
                        stratum_count=sum(r['physical_class'][metric] == quality for r in eligible),
                        row_ids=[r['task_id'] + '/' + r['condition_id'] for r in eligible
                        if r['physical_class'][metric] == quality and r['after_state'] == after]))
    return dict(condition_count=len(rows), family_count=len({r['task_id'] for r in rows}),
        projection_states=dict(Counter(r['projection_check']['state'] for r in rows)),
        old_states=dict(Counter(r['old_state'] for r in rows)), after_states=dict(Counter(r['after_state'] for r in rows)),
        old_eligible_count=sum(r['old_candidate_eligible'] for r in rows),
        retained_count=sum(r['retained_candidate'] for r in rows), cross_tabs=bins)


def fmt(value, factor=1., digits=2):
    return '—' if value is None else f'{value * factor:.{digits}f}'


def label(row):
    return ('变高' if 'height' in row['parent'] else '同高') + '-' + row['case_id'] + '-' + ('普通' if row['method'] == 'baseline' else '圆柱')


def markdown(summary):
    rows, stats = summary['rows'], summary['summary']
    lines = ['# 9月26日：旧点击能检查投影，却不能变成新的独立证据', '',
        f"固定回放7组已知RGB输入、两种杆方法、每组完整相机加四个留组条件，共14个方法族、70行。旧流程可作为候选继续审查的有{stats['old_eligible_count']}行；新增投影检查保留{stats['retained_count']}行。所有缺失和原拒绝都还在表里。", '',
        '这次没有重建一根新杆，也没有给模型加真值。只是把原来那两个点击，投影到原有的有限杆上再问一次：这个位置是否一致。点击早已用于目标选择，所以结果是同源一致性审查；通过不能算新增的独立身份验证。', '',
        '旧身份拒绝继续拒绝；相机扣留继续扣留。新检查不能提升旧结果，也不修改相机、几何、像素赋值或默认发布策略。', '',
        '## 规则和数据来历', '',
        '- 原点击由助手在DA3预测之前查看RGB网格后记录，带逐图SHA；不是GT投影，也不是人类用户研究。7组都只有view_-32和view_+19两个点击，y=135，像素不确定度为[0.5,0.5]。',
        '- 固定2 px阈值继承旧有限段阶段。取不确定矩形的外接圆半径r；投影距离d+r≤2才支持，max(0,d−r)>2才矛盾，其余未决。有限段和缺口保持原样，不按点击重拟合或挑另一个目标。',
        '- 检查至少两个不同相机中心的视图；图域、来源SHA、相机合法性、缺失和相机后方情况都显式检查。无数值就是null，不画成0。',
        '- 检测池每个视图有440行，候选搜索、排序和有限段拟合已使用其中的数据。未被最后选中的像素行也不自动成为留出集。旧两点击是前景成员声明，不是跨视图物理端点对应；不能拿它们硬做端点三角化。',
        '- 保持原完整训练控制和四fold，不平均、不挑最好fold。70行互相关联，只来自7组已知场景，不能按70个独立对象计算成功率。', '',
        '## 保留与拦截，都对照物理读数', '',
        '只复用9月24日冻结评分：每个族以完整控制相机拟合一次Sim3，四fold共享它。没有重新对齐，也没有拿三维分数调2 px阈值。两方向p95都≤原25 mm容差叫“该读数达标”，任一方向超出叫“该读数超出”；另列R/P都100%的分组。两者都不是整条杆物理正确或G1通过的证明。空预测保留R=0、P未定义；相机扣留上的物理分数仍只作诊断。', '',
        '| 范围 | 分组口径 | 原物理读数 | 新检查后 | 数量 |', '| --- | --- | --- | --- | ---: |']
    names = dict(all_conditions='全部70条件', control_only='仅14个控制', p95='双向p95', complete_coverage='R/P全覆盖',
        both_within_25mm='两方向均≤25 mm', at_least_one_exceeds_25mm='至少一方向>25 mm',
        both_RP_100='R/P均100%', some_uncovered_samples='存在未覆盖样本', unscored='无评分', empty_prediction='空预测',
        retained_candidate='保留候选', blocked_by_projection='投影矛盾拦截', unresolved_projection='投影未决')
    for row in stats['cross_tabs']:
        lines.append('| ' + ' | '.join([names[row['subset']], names[row['stratum']], names[row['physical_class']],
                                       names[row['after_state']], str(row['count'])]) + ' |')
    lines += ['', '这张表只在旧身份接受且相机未扣留的候选中比较；下方完整70行仍保留其他情况。达标行被拦也算损失，超出行继续保留也明确列出，不能只挑拒绝数量讲故事。', '',
        '## 完整70行', '',
        '投影列依次为状态、两点击中的最大中心距离px；双向p95依次为真值→预测、预测→真值。R/P沿用25 mm容差，—表示未定义或缺失。old/new状态是诊断候选状态，不是产品发布。', '',
        '| 方法族 | 条件 | 旧身份 / 相机 | 投影检查 / px | 新状态 | 双向p95 mm | R/P % |',
        '| --- | --- | --- | --- | --- | ---: | ---: |']
    for row in rows:
        metrics = row['physical']['metrics'] or {}
        left, right = metrics.get('truth_to_prediction', {}), metrics.get('prediction_to_truth', {})
        lines.append('| ' + ' | '.join([label(row), row['condition_id'], row['old_identity_state'] + ' / ' + row['old_camera_state'],
            row['projection_check']['state'] + ' / ' + fmt(row['projection_check']['maximum_distance_px']), row['after_state'],
            fmt(left.get('distance_p95'), 1000.) + ' / ' + fmt(right.get('distance_p95'), 1000.),
            fmt(metrics.get('recovery_fraction'), 100., 1) + ' / ' + fmt(metrics.get('precision_fraction'), 100., 1)]) + ' |')
    lines += ['', '## 复现与限制', '',
        '新检查没有未使用的额外点击或独立端点数据，所以本轮不能证明新身份识别能力。它只能展示：重复使用旧点击做更直接的投影检查，会保住、丢掉或漏拦哪些已有候选。要验证额外证据是否有效，需要预先冻结且未参与拟合和选择的观测。', '',
        f"推理阶段耗时{summary['inference_elapsed_seconds']:.3f}秒；所有评分来自固定源汇总SHA `{summary['evaluation_source']['sha256']}`。", '',
        '- [70行完整JSON](results/2026-09-26-target-projection-replay.json)',
        '- [独立阶段审计收据](results/2026-09-26-target-projection-replay-audit.json)',
        '- [可执行脚本](../../scripts/run_rod_target_projection_replay.py)', '',
        '运行命令依次为脚本的prepare、audit-pre、infer、evaluate、audit-post。各阶段只创建新产物；重复执行应使用新run ID，旧失败和旧结果不覆盖。', '']
    return '\n'.join(lines)


def evaluate():
    config, manifest = checked(evaluation=True)
    index, rows = inference_rows(config, manifest)
    result = evaluation_rows(rows, manifest, config)
    summary = dict(state='complete', run_id=RUN_ID, rows=result, summary=summarise(result),
        evaluation_source=config['evaluation_source'], policy=POLICY, source_sha256=config['source_sha256'],
        input_sha256=config['input_sha256'], inference_sha256=digest(RUN / 'inference.json'),
        inference_elapsed_seconds=index['elapsed_seconds'], gt_read_during_inference=False,
        alignment_recomputed=False, physical_metrics_recomputed=False, no_automatic_promotion=True)
    destination = ROOT / 'data/evaluation' / RUN_ID
    destination.mkdir(exist_ok=False)
    write_json(destination / 'summary.json', summary)
    write_json(PUBLIC, summary)
    with REPORT.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(markdown(summary))
    checked(evaluation=True)
    print('EVALUATED', summary['summary'], flush=True)


def audit(stage):
    config, manifest = checked(evaluation=stage == 'post')
    receipt_data = dict(run_id=RUN_ID, stage=stage, state='passed', family_count=14, condition_count=70,
        input_sha256=config['input_sha256'], config_sha256=digest(RUN / 'method_config.json'),
        source_freeze_sha256=config['source_freeze_sha256'], source_file_count=len(config['source_sha256']),
        parent_and_rgb_receipt_count=len(config['receipts']), unchanged_geometry=True, no_promotions=True,
        evidence_scope='same_source_not_independent', source_sha256=config['source_sha256'])
    if stage == 'post':
        _, rows = inference_rows(config, manifest)
        expected = evaluation_rows(rows, manifest, config)
        summary = read_json(PUBLIC)
        assert summary['rows'] == expected and summary['summary'] == summarise(expected)
        assert digest(PUBLIC) == digest(ROOT / 'data/evaluation' / RUN_ID / 'summary.json')
        assert REPORT.read_text(encoding='utf-8') == markdown(summary)
        receipt_data.update(summary_sha256=digest(PUBLIC), report_sha256=digest(REPORT),
                            inference_sha256=digest(RUN / 'inference.json'), summary=summary['summary'])
    write_json(RUN / ('audit-' + stage + '.json'), receipt_data)
    if stage == 'post':
        write_json(ROOT / 'docs/experiments/results/2026-09-26-target-projection-replay-audit.json', receipt_data)
    print('AUDIT', stage, 'passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'audit-pre', 'infer', 'evaluate', 'audit-post'])
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage.startswith('audit-'):
        audit(args.stage.split('-')[1])
    else:
        {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[args.stage]()
