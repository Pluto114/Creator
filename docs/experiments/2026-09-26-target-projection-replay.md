# 9月26日：旧点击能检查投影，却不能变成新的独立证据

固定回放7组已知RGB输入、两种杆方法、每组完整相机加四个留组条件，共14个方法族、70行。旧流程可作为候选继续审查的有51行；新增投影检查保留51行。所有缺失和原拒绝都还在表里。

这次没有重建一根新杆，也没有给模型加真值。只是把原来那两个点击，投影到原有的有限杆上再问一次：这个位置是否一致。点击早已用于目标选择，所以结果是同源一致性审查；通过不能算新增的独立身份验证。

旧身份拒绝继续拒绝；相机扣留继续扣留。新检查不能提升旧结果，也不修改相机、几何、像素赋值或默认发布策略。

## 规则和数据来历

- 原点击由助手在DA3预测之前查看RGB网格后记录，带逐图SHA；不是GT投影，也不是人类用户研究。7组都只有view_-32和view_+19两个点击，y=135，像素不确定度为[0.5,0.5]。
- 固定2 px阈值继承旧有限段阶段。取不确定矩形的外接圆半径r；投影距离d+r≤2才支持，max(0,d−r)>2才矛盾，其余未决。有限段和缺口保持原样，不按点击重拟合或挑另一个目标。
- 检查至少两个不同相机中心的视图；图域、来源SHA、相机合法性、缺失和相机后方情况都显式检查。无数值就是null，不画成0。
- 检测池每个视图有440行，候选搜索、排序和有限段拟合已使用其中的数据。未被最后选中的像素行也不自动成为留出集。旧两点击是前景成员声明，不是跨视图物理端点对应；不能拿它们硬做端点三角化。
- 保持原完整训练控制和四fold，不平均、不挑最好fold。70行互相关联，只来自7组已知场景，不能按70个独立对象计算成功率。

## 保留与拦截，都对照物理读数

只复用9月24日冻结评分：每个族以完整控制相机拟合一次Sim3，四fold共享它。没有重新对齐，也没有拿三维分数调2 px阈值。两方向p95都≤原25 mm容差叫“该读数达标”，任一方向超出叫“该读数超出”；另列R/P都100%的分组。两者都不是整条杆物理正确或G1通过的证明。空预测保留R=0、P未定义；相机扣留上的物理分数仍只作诊断。

| 范围 | 分组口径 | 原物理读数 | 新检查后 | 数量 |
| --- | --- | --- | --- | ---: |
| 全部70条件 | 双向p95 | 至少一方向>25 mm | 保留候选 | 24 |
| 全部70条件 | 双向p95 | 至少一方向>25 mm | 投影矛盾拦截 | 0 |
| 全部70条件 | 双向p95 | 至少一方向>25 mm | 投影未决 | 0 |
| 全部70条件 | 双向p95 | 两方向均≤25 mm | 保留候选 | 27 |
| 全部70条件 | 双向p95 | 两方向均≤25 mm | 投影矛盾拦截 | 0 |
| 全部70条件 | 双向p95 | 两方向均≤25 mm | 投影未决 | 0 |
| 全部70条件 | R/P全覆盖 | R/P均100% | 保留候选 | 20 |
| 全部70条件 | R/P全覆盖 | R/P均100% | 投影矛盾拦截 | 0 |
| 全部70条件 | R/P全覆盖 | R/P均100% | 投影未决 | 0 |
| 全部70条件 | R/P全覆盖 | 存在未覆盖样本 | 保留候选 | 31 |
| 全部70条件 | R/P全覆盖 | 存在未覆盖样本 | 投影矛盾拦截 | 0 |
| 全部70条件 | R/P全覆盖 | 存在未覆盖样本 | 投影未决 | 0 |
| 仅14个控制 | 双向p95 | 至少一方向>25 mm | 保留候选 | 6 |
| 仅14个控制 | 双向p95 | 至少一方向>25 mm | 投影矛盾拦截 | 0 |
| 仅14个控制 | 双向p95 | 至少一方向>25 mm | 投影未决 | 0 |
| 仅14个控制 | 双向p95 | 两方向均≤25 mm | 保留候选 | 4 |
| 仅14个控制 | 双向p95 | 两方向均≤25 mm | 投影矛盾拦截 | 0 |
| 仅14个控制 | 双向p95 | 两方向均≤25 mm | 投影未决 | 0 |
| 仅14个控制 | R/P全覆盖 | R/P均100% | 保留候选 | 4 |
| 仅14个控制 | R/P全覆盖 | R/P均100% | 投影矛盾拦截 | 0 |
| 仅14个控制 | R/P全覆盖 | R/P均100% | 投影未决 | 0 |
| 仅14个控制 | R/P全覆盖 | 存在未覆盖样本 | 保留候选 | 6 |
| 仅14个控制 | R/P全覆盖 | 存在未覆盖样本 | 投影矛盾拦截 | 0 |
| 仅14个控制 | R/P全覆盖 | 存在未覆盖样本 | 投影未决 | 0 |

这张表只在旧身份接受且相机未扣留的候选中比较；下方完整70行仍保留其他情况。达标行被拦也算损失，超出行继续保留也明确列出，不能只挑拒绝数量讲故事。

## 完整70行

投影列依次为状态、两点击中的最大中心距离px；双向p95依次为真值→预测、预测→真值。R/P沿用25 mm容差，—表示未定义或缺失。old/new状态是诊断候选状态，不是产品发布。

| 方法族 | 条件 | 旧身份 / 相机 | 投影检查 / px | 新状态 | 双向p95 mm | R/P % |
| --- | --- | --- | --- | --- | ---: | ---: |
| 同高-r01-普通 | control | accepted / candidate_camera_correction | supported / 0.28 | retained_candidate | 9.69 / 9.68 | 100.0 / 100.0 |
| 同高-r01-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.37 | retained_candidate | 7.10 / 7.09 | 99.9 / 100.0 |
| 同高-r01-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.40 | retained_candidate | 8.03 / 8.00 | 100.0 / 100.0 |
| 同高-r01-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.25 | retained_candidate | 18.11 / 18.10 | 100.0 / 100.0 |
| 同高-r01-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.30 | retained_candidate | 14.65 / 14.65 | 100.0 / 100.0 |
| 同高-r01-圆柱 | control | accepted / candidate_camera_correction | supported / 0.48 | retained_candidate | 8.21 / 8.20 | 100.0 / 100.0 |
| 同高-r01-圆柱 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.49 | retained_candidate | 8.47 / 8.45 | 100.0 / 100.0 |
| 同高-r01-圆柱 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.50 | retained_candidate | 8.19 / 8.16 | 100.0 / 100.0 |
| 同高-r01-圆柱 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.55 | retained_candidate | 15.10 / 15.10 | 100.0 / 100.0 |
| 同高-r01-圆柱 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.48 | retained_candidate | 12.54 / 12.54 | 100.0 / 100.0 |
| 同高-r02-普通 | control | accepted / candidate_camera_correction | supported / 0.43 | retained_candidate | 36.80 / 27.96 | 82.3 / 84.0 |
| 同高-r02-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.44 | retained_candidate | 33.21 / 24.53 | 93.6 / 95.0 |
| 同高-r02-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.43 | retained_candidate | 34.45 / 22.32 | 93.6 / 95.3 |
| 同高-r02-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.42 | retained_candidate | 37.66 / 32.21 | 52.8 / 53.7 |
| 同高-r02-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.44 | retained_candidate | 36.99 / 29.17 | 71.6 / 73.3 |
| 同高-r02-圆柱 | control | accepted / candidate_camera_correction | supported / 0.43 | retained_candidate | 36.80 / 27.96 | 82.3 / 84.0 |
| 同高-r02-圆柱 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.44 | retained_candidate | 33.21 / 24.53 | 93.6 / 95.0 |
| 同高-r02-圆柱 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.43 | retained_candidate | 34.45 / 22.32 | 93.6 / 95.3 |
| 同高-r02-圆柱 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.42 | retained_candidate | 37.66 / 32.21 | 52.8 / 53.7 |
| 同高-r02-圆柱 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.44 | retained_candidate | 36.99 / 29.17 | 71.6 / 73.3 |
| 同高-r03-普通 | control | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 28.88 / 28.75 | 67.6 / 68.3 |
| 同高-r03-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.07 | retained_candidate | 35.29 / 35.16 | 30.4 / 30.9 |
| 同高-r03-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 18.63 / 18.50 | 98.6 / 99.3 |
| 同高-r03-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.26 | retained_candidate | 32.10 / 31.89 | 55.4 / 56.2 |
| 同高-r03-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 38.55 / 38.38 | 2.9 / 3.1 |
| 同高-r03-圆柱 | control | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 28.88 / 28.75 | 67.6 / 68.3 |
| 同高-r03-圆柱 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.07 | retained_candidate | 35.29 / 35.16 | 30.4 / 30.9 |
| 同高-r03-圆柱 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 18.63 / 18.50 | 98.6 / 99.3 |
| 同高-r03-圆柱 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.26 | retained_candidate | 32.10 / 31.89 | 55.4 / 56.2 |
| 同高-r03-圆柱 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.15 | retained_candidate | 38.55 / 38.38 | 2.9 / 3.1 |
| 同高-r04-普通 | control | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-普通 | leave_group_0 | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-普通 | leave_group_1 | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-普通 | leave_group_2 | accepted / withhold_correction | supported / 0.74 | withheld_by_camera | 1251.67 / 1251.85 | 0.0 / 0.0 |
| 同高-r04-普通 | leave_group_3 | accepted / withhold_correction | supported / 0.73 | withheld_by_camera | 1294.35 / 1295.14 | 0.0 / 0.0 |
| 同高-r04-圆柱 | control | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-圆柱 | leave_group_0 | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-圆柱 | leave_group_1 | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-圆柱 | leave_group_2 | rejected / withhold_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 同高-r04-圆柱 | leave_group_3 | accepted / withhold_correction | supported / 0.31 | withheld_by_camera | 1293.39 / 1294.13 | 0.0 / 0.0 |
| 变高-r01-普通 | control | accepted / candidate_camera_correction | supported / 0.57 | retained_candidate | 9.62 / 9.62 | 100.0 / 100.0 |
| 变高-r01-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.56 | retained_candidate | 4.47 / 4.44 | 100.0 / 100.0 |
| 变高-r01-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.59 | retained_candidate | 5.26 / 5.24 | 100.0 / 100.0 |
| 变高-r01-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.54 | retained_candidate | 9.53 / 9.53 | 100.0 / 100.0 |
| 变高-r01-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.58 | retained_candidate | 18.95 / 18.97 | 100.0 / 100.0 |
| 变高-r01-圆柱 | control | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r01-圆柱 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.57 | retained_candidate | 3.61 / 3.60 | 100.0 / 100.0 |
| 变高-r01-圆柱 | leave_group_1 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r01-圆柱 | leave_group_2 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r01-圆柱 | leave_group_3 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r02-普通 | control | accepted / candidate_camera_correction | supported / 0.71 | retained_candidate | 14.22 / 14.08 | 100.0 / 100.0 |
| 变高-r02-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.78 | retained_candidate | 12.48 / 12.42 | 100.0 / 100.0 |
| 变高-r02-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.70 | retained_candidate | 11.25 / 11.18 | 100.0 / 100.0 |
| 变高-r02-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.65 | retained_candidate | 20.21 / 20.11 | 100.0 / 100.0 |
| 变高-r02-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.72 | retained_candidate | 15.24 / 15.13 | 100.0 / 100.0 |
| 变高-r02-圆柱 | control | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r02-圆柱 | leave_group_0 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r02-圆柱 | leave_group_1 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r02-圆柱 | leave_group_2 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r02-圆柱 | leave_group_3 | rejected / candidate_camera_correction | unresolved / — | original_identity_not_accepted | — / — | 0.0 / — |
| 变高-r03-普通 | control | accepted / candidate_camera_correction | supported / 0.25 | retained_candidate | 30.64 / 30.35 | 69.4 / 70.5 |
| 变高-r03-普通 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.17 | retained_candidate | 37.87 / 37.57 | 39.1 / 39.9 |
| 变高-r03-普通 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.25 | retained_candidate | 23.91 / 23.65 | 97.8 / 99.0 |
| 变高-r03-普通 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.28 | retained_candidate | 19.30 / 19.08 | 98.4 / 99.0 |
| 变高-r03-普通 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.27 | retained_candidate | 46.72 / 46.40 | 0.0 / 0.0 |
| 变高-r03-圆柱 | control | accepted / candidate_camera_correction | supported / 0.25 | retained_candidate | 30.64 / 30.35 | 69.4 / 70.5 |
| 变高-r03-圆柱 | leave_group_0 | accepted / candidate_camera_correction | supported / 0.17 | retained_candidate | 37.87 / 37.57 | 39.1 / 39.9 |
| 变高-r03-圆柱 | leave_group_1 | accepted / candidate_camera_correction | supported / 0.25 | retained_candidate | 23.91 / 23.65 | 97.8 / 99.0 |
| 变高-r03-圆柱 | leave_group_2 | accepted / candidate_camera_correction | supported / 0.28 | retained_candidate | 19.30 / 19.08 | 98.4 / 99.0 |
| 变高-r03-圆柱 | leave_group_3 | accepted / candidate_camera_correction | supported / 0.27 | retained_candidate | 46.72 / 46.40 | 0.0 / 0.0 |

## 复现与限制

新检查没有未使用的额外点击或独立端点数据，所以本轮不能证明新身份识别能力。它只能展示：重复使用旧点击做更直接的投影检查，会保住、丢掉或漏拦哪些已有候选。要验证额外证据是否有效，需要预先冻结且未参与拟合和选择的观测。

推理阶段耗时1.183秒；所有评分来自固定源汇总SHA `42a2f23dfd91d46b2a2ce94741b2f76e5a6f0e6ebe946b1dbdddcce878c44012`。

- [70行完整JSON](results/2026-09-26-target-projection-replay.json)
- [独立阶段审计收据](results/2026-09-26-target-projection-replay-audit.json)
- [可执行脚本](../../scripts/run_rod_target_projection_replay.py)

运行命令依次为脚本的prepare、audit-pre、infer、evaluate、audit-post。各阶段只创建新产物；重复执行应使用新run ID，旧失败和旧结果不覆盖。
