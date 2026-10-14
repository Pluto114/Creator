# 10月14日续轮：四视图共识外边界

**本路线作为共同 reader 整体拒绝，统一开发 reader 仍为 `bracketed-occupancy-sprint-20261014-001`。** `consensus-boundary-sprint-20261014-001` 完成一次完整 50 配置。A-frame baseline 两尺度 P 从 98% 升至 100%，最大端点误差从约 58.53 mm 降至 16.90 / 16.81 mm；但全部 50 行中 **18 行 R 下降、0 行上升**，包括 chair 和 A-frame 全部 4 行 base。它没有解决一般端点问题，不能只挑 A-frame baseline 的收益用于共同 reader 升级。

## 决策与正常侧检查

起点是统一采用的 `bracketed-occupancy-sprint-20261014-001`。根因诊断确认 A-frame baseline 上端过长已存在于原生候选，不是先前内部短孔连接造成。

第一原型 `AnchoredBoundaryReadout` 要求每个已标记视图都支持外边界，6 个机制测试通过，但 normal-only 诊断发现 view00 的上部观测缺失会把约 204 mm 的有效范围截掉。主代理在完整快筛前停止此原型，**没有启动其 50 normal/evaluate**，不把机制测试通过说成路线成功。

第二候选改用已有 chain 搜索的 `minimum_chain_views=4`：外边界需要同一 chain 内至少 4 个不同 view 的无 padding 测量带支持，不要求每个 anchor view 都有观测。重复 view ID 拒绝，同一 view 的多个候选不重复计票，不把不同 chain 的票拼起来。normal-only 诊断在 A-frame baseline 上给出约 z=0.857 的共识上界，而不是原型的约 z=0.704；随后才批准完整一次快筛。

规则仍保持 bracketed 的输入准入点池、实际 TLS 模型、选轴、内部几何和连通策略，只按强正支撑的外包络向内裁两端。不加固定毫米收缩，不外推、不重新拟合，不以 GT 选阈值；内部 RGB unknown 不新增切孔，原有实际几何 gap 不填。它是正支撑包络，不是物理端点检测器。

## 实际执行

- required 原型 6 项测试：6/6，0.447 s，Ruff 通过。其后只参数化共用 trim 的 mask 回调并抽取所选链上下文，默认 required 语义保留；未重跑旧 6 项。
- consensus 6 项新测试：第一次 5 过、1 个 mock context 因最后新增的 view ID 校验缺 `views` 字段报错。修正测试夹具并加入重复 view ID 拒绝断言后，6/6 通过，**0.440 s**；修改 helper、新 core/adapter 和专属测试 Ruff 通过。没有因这个测试夹具问题改变 core。
- 唯一一次 5 案 × 5 臂 × 2 尺度 = 50 行完整 normal；完整落盘后显式独立进程 evaluate 固定完整原 GT。normal `gt_read=false`，evaluate 才 `evaluation_gt_read=true`。开发历史分数已见，不作盲测声明。
- normal 捕获墙钟 **118.596 s**，evaluate **2.460 s**，合计 **121.055 s**，两阶段退出码均 0。normal、共同读数评分、native 评分错误均 0。
- 无新 DA3、新补丁生成/端到端重跑、第二遍快筛、旧对照重算、全仓测试、双环境或祖先审计。原始运行在 `.runtime/experiments/consensus-boundary-sprint-20261014-001/`；捕获命令/退出码/日志在 `.runtime/validation/foreground-consensus-runners-20261012-002/consensus-boundary-sprint-20261014-001-{normal,evaluate}.*`。

## 完整 50 对照

旧对照是保存的 **bracketed 001 完整 50 行**，不是只比较之前失败版。R/P 为原固定容差下恢复率/精度，比较容差 `1e-12`。

| 指标 | 上升 | 下降 | 相同 |
| --- | ---: | ---: | ---: |
| R | 0 | 18 | 32 |
| P | 2 | 0 | 48 |
| 段数 | 0 | 0 | 50 |
| 输出长度 | 0 | 38 | 12 |

50/50 所选候选和实际 TLS 模型完全相同；50/50 native 状态、几何、完整物理指标及评分错误完全相同。边界可评分行数仍为 36；其中最大端点误差改善 2 行、变差 19 行，其余已可评分 15 行相同。另外 14 行仍无边界配对，null 不是 0。

### 全部 10 个 base

P 全部旧新均 100%，段数全不变。细/粗为 camera-span fraction 0.0025/0.005。

| 案例 | 尺度 | R 旧 → 新 % | 长度旧 → 新 m | 裁去 mm |
| --- | --- | ---: | ---: | ---: |
| r01 | 细 | 95.900 → 95.900 | 1.616153 → 1.615512 | 0.641 |
| r01 | 粗 | 98.800 → 98.800 | 1.823704 → 1.823318 | 0.386 |
| r02 | 细 | 100 → 100 | 1.543052 → 1.541755 | 1.297 |
| r02 | 粗 | 100 → 100 | 1.634148 → 1.634148 | 0 |
| r03 | 细 | 82.700 → 82.700 | 1.195340 → 1.195340 | 0 |
| r03 | 粗 | 87.400 → 87.400 | 1.506246 → 1.506246 | 0 |
| chair01 | 细 | 96.129 → 95.355 | 1.352143 → 1.323320 | 28.823 |
| chair01 | 粗 | 98.194 → 97.548 | 1.455932 → 1.427134 | 28.799 |
| aframe01 | 细 | 33.132 → 32.769 | 0.444639 → 0.435113 | 9.526 |
| aframe01 | 粗 | 33.736 → 32.769 | 0.523060 → 0.502697 | 20.363 |

A-frame 粗 base 最大边界误差 **1119.142 → 1135.501 mm**；其余 9 个 base 仍 `boundary_count_mismatch`。不能以 P 保持 100% 淡化被裁掉的有效覆盖。

### 全部 18 行 R 下降

| 案例/臂 | 配置数 | 细 R 旧 → 新 % | 粗 R 旧 → 新 % |
| --- | ---: | ---: | ---: |
| chair base | 2 | 96.129 → 95.355 | 98.194 → 97.548 |
| chair TLS / RANSAC | 4 | 100 → 99.225806 | 100 → 99.225806 |
| chair baseline / cylinder | 4 | 100 → 99.354839 | 100 → 99.354839 |
| A-frame base / TLS / RANSAC / cylinder | 8 | 33.131802 → 32.769045 | 33.736397 → 32.769045 |

chair 四候选仍为单段、P100%，但最大端点误差由约 10.09–12.71 mm 恶化至 **35.85–36.58 mm**，每行裁掉约 28.79–29.89 mm。正常侧的正支撑更严格，不等同于物理边界更准确。

### 唯一 P 改善：A-frame baseline 两行

| 尺度 | R 旧 → 新 % | P 旧 → 新 % | 最大端点误差旧 → 新 mm | 输出长度旧 → 新 m |
| --- | ---: | ---: | ---: | ---: |
| 细 | 100 → 100 | 98 → 100 | 58.531646 → 16.901058 | 1.699992 → 1.643388 |
| 粗 | 100 → 100 | 98 → 100 | 58.539720 → 16.806789 | 1.699994 → 1.643617 |

两行裁去 56.604517 / 56.377464 mm，但原生 baseline 没有改动。本轮是共同 reader 干预，不能称为原生补丁或实际保存/重开对象已完成端点优化。

### 真实 gap 与原生失败

r02 全部 5 臂两尺度，共 10 行的 `guarded_gap_false_length_m` 均保持 **0 → 0**，R/P 均保持100%。其外边界部分略变，因此不宣称完整物理指标都不变；细 RANSAC/baseline/cylinder 最大端点误差各增加约 0.16–0.23 mm。其他案例没有声明 gap，零值不是 gap 保持证据。

全部原生数据仍不变：A-frame TLS/RANSAC 原生 R=P=0，cylinder 原生为空，baseline 原生 R=100%、P约97.879859%。r03 和 A-frame 的基础支持不足也未解决。

## 交付与下一步

主代理依据全体 base/候选结果拒绝共同 reader 干预，不按 GT 或对象选择启用。本失败版完整保留。下一独立路线将相同边界机制作用于原生 baseline 候选轴，统一新增第六臂 `baseline_boundary`，而非修改共同 reader；旧 baseline 和统一 bracketed reader 都保留。新路线尚未在本报告中计作运行或成功。

全部 50 行新旧 R/P/N/长度/边界/gap、旧新几何、25 个唯一 native、源 SHA、捕获命令/耗时在[紧凑结果 JSON](results/2026-10-14-consensus-boundary.json)。助手日志 `.runtime/work-logs/20261014-boundary-readout-support.md`，导出脚本同目录，只汇总既有分数、不重评分。本轮未新增 DeepSeek 任务。

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_consensus_boundary_readout.py -v
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py normal --adapter creator_eval.rgb_consensus_boundary_readout:ConsensusBoundaryReadout --run-id NEW_UNIQUE_RUN_ID
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py evaluate --run-id NEW_UNIQUE_RUN_ID
```

本次实际 ID `consensus-boundary-sprint-20261014-001`，已有运行拒绝覆盖。以上是复现入口，不代表又跑第二遍。
