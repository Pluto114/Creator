# 10月12日续轮：实际投影的精确占据读取

统一采用 `occupied-axis-sprint-20261012-001`，替代上一轮统一采用的 `stroke-selected-sprint-20261012-002`；不按案例、方法或尺度挑选版本。完整 50 配置恢复率上升 22、下降 0、相同 28；全部 10 个 base 的精度保持 100%。本轮是已见过分数的合成开发快筛，不是盲测、实拍、共同读取资格或正式 G1 验收。

## 方案与唯一一次运行

保留旧 voxel-balanced TLS 轴方向、同 stroke 输入域、实际种子扩展和前景带选择规则。支持长度改为每个实际唯一点在轴上投影的 ±1v 邻域精确并集，截在实际投影 min/max 内，不通过离散采样生成占据，也不插补并集之外的缺口；再按原同链精确 RGB cells 裁剪。

**4v 是整轴实际投影 extent 的资格门限，不再逐段删掉小于 4v 的实际支持碎段。** 合格轴保留正长度碎段，数值长度下限为 `1e-10 m`。这一语义在首次新 normal/评分之前确定，未根据本轮评分回调参数。仍只处理明确标注的一个局部直杆及其实际缺口，不宣称解决任意多轴骨架。

评分前仅运行新 8 项机制测试（NumPy 1.26，8/8，0.396s）及两新核心＋专属测试 Ruff，均通过；包括窄至 0.01v 的缺口不被采样吞掉、整轴不足 4v 拒绝、合格轴实际 3v 片段经过 RGB cells 后仍保留。没有追加全量验证、祖先扫描、双重审计或第二次快筛。

固定 5 案 × 5 臂 × 2 尺度 = 50 配置；完整 normal 和直接来源 manifest 落盘后，显式 evaluate 固定完整原 GT。没有重算旧 controls、没有新 DA3＋补丁端到端运行、没有同版本新进程重复。normal/读数评分/native 评分错误均为 0，两阶段退出码均为 0；这不代表几何正确或 G1 通过。

| 单次真实墙钟耗时 | 旧 stroke-selected 002 | 新 occupied-axis 001 |
| --- | ---: | ---: |
| normal | 58.670s | 112.610s |
| evaluate | 2.347s | 2.436s |
| 合计 | 61.017s | **115.046s** |

本次明显更慢，不能只报道覆盖收益。以上为实际捕获的子进程耗时，不是独占、重复的性能基准。原始 stdout/stderr 位于 `.runtime/validation/foreground-consensus-runners-20261012-002/occupied-axis-sprint-001-{normal,evaluate}.*`；阶段日志、normal/evaluation、来源 manifest 均在 `.runtime/experiments/occupied-axis-sprint-20261012-001/`。

## 全部 10 个 base

R 为恢复率（%）；细/粗 fraction 为 0.0025/0.005。下表全部新旧 P 都为 100%；段数不是正确杆数，长度为米。

| 案例 | 尺度 | 旧 R → 新 R | 旧段数 → 新段数 | 旧长度 → 新长度 |
| --- | --- | ---: | ---: | ---: |
| r01 | 细 | 80.200 → 95.900 | 6 → 10 | 1.398406 → 1.616153 |
| r01 | 粗 | 88.700 → 98.800 | 4 → 6 | 1.607714 → 1.823704 |
| r02 | 细 | 95.636 → 100.000 | 7 → 9 | 1.392007 → 1.543052 |
| r02 | 粗 | 100.000 → 100.000 | 3 → 3 | 1.593749 → 1.634148 |
| r03 | 细 | 43.500 → 82.700 | 4 → 14 | 0.671813 → 1.195340 |
| r03 | 粗 | 53.000 → 87.400 | 3 → 9 | 0.911760 → 1.506246 |
| chair01 | 细 | 91.097 → 96.129 | 6 → 11 | 1.210102 → 1.340018 |
| chair01 | 粗 | 95.355 → 98.194 | 2 → 4 | 1.394236 → 1.443776 |
| aframe01 | 细 | 27.449 → 33.132 | 2 → 5 | 0.356454 → 0.444639 |
| aframe01 | 粗 | 33.736 → 33.736 | 1 → 1 | 0.523060 → 0.523060 |

10 个 base 恢复率 8 升、0 降、2 相同；除 Aframe 粗尺度外，其余 9 行新旧都为 `boundary_count_mismatch`，边界误差为 null，不是 0。Aframe 粗 base 虽能配对边界，最大误差仍为 **1119.142mm**，明显未恢复目标完整长度。

## 候选、缺口与保留的负结果

全部 50 配置 P 上升 0、下降 2、相同 48（R/P 比较容差 `1e-12`）。两项 P 下降均在 Aframe baseline：

| Aframe baseline | R 旧 → 新 % | P 旧 → 新 % | 段数 | 长度旧 → 新 m |
| --- | ---: | ---: | ---: | ---: |
| 细 | 100 → 100 | 98.923 → 98.399 | 2 → 3 | 1.668068 → 1.692081 |
| 粗 | 100 → 100 | 98.467 → 98.419 | 2 → 3 | 1.691113 → 1.691945 |

两尺度仍为边界数量不匹配。chair 的 TLS/RANSAC/baseline/cylinder 四候选臂在两尺度 R 都提升到 100、P 保持 100，但均由 1 段变为 3 段，边界从可评分变为数量不匹配。r01 RANSAC 两尺度也由 1 段变为 2 段并出现边界不匹配。不能将恢复了实际短片段等同于得到正确连通杆或通过边界指标。

唯一含原始声明物理 gap 的 r02，其全部 10 配置的 `guarded_gap_false_length_m` 新旧均为 0，未增加缺口假填；其他案例 gap 字段为 0 不作为缺口保持证据。50 行原 native 几何和物理评分与旧版完全相同。Aframe 的 TLS/RANSAC/cylinder 共同读数仍与 base 相同，不表示原生补丁变正确；原 native TLS/RANSAC 仍 R=P=0、cylinder 仍为空。

全部 50 项新旧完整物理指标、逐项差值、25 个唯一 native 指标及源/run/capture 标识见[紧凑 JSON](results/2026-10-12-occupied-axis-readout.json)，未复制巨型 normal/readouts。

## 运行入口与源码快照

在 `D:/Creator-newage` 执行；必须使用全新 ID，已有运行目录拒绝覆盖。normal 完成后再 evaluate 同一 ID。

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
$runId = 'occupied-axis-sprint-' + (Get-Date -Format 'yyyyMMddTHHmmssfff')
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py normal --adapter creator_eval.rgb_occupied_axis_readout:OccupiedAxisReadout --run-id $runId
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py evaluate --run-id $runId
```

本次实际 ID 为 `occupied-axis-sprint-20261012-001`。两新核心、专属测试和 quick runner 的逐字节副本及 SHA 保存在该 run 的 `source_snapshot/`；其中三项运行源 SHA 与 normal manifest 一致，测试不是推理输入。快照是在运行后保存，不冒充运行前的全祖先冻结。源文件分别为 `occupied_axis_readout.py`、`rgb_occupied_axis_readout.py`、`tests/test_occupied_axis_readout.py`、`scripts/run_readout_sprint.py`。
