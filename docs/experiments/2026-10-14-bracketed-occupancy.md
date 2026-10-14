# 10月14日：保留旧边界的内部短孔洞连接

**统一开发版本采用 `bracketed-occupancy-sprint-20261014-001`，不是生产切换或 G1 资格通过。** 它完成唯一一次完整 50 配置；相对原统一基线 `occupied-axis-sprint-20261012-001`，12 行碎段减少，边界可配对由 26 增至 36 行；全部 50 行恢复率不变，全部 10 个 base 精度仍 100%，r02 全 10 行真实 gap 假邻近仍为 0。采用代价必须保留：**A-frame baseline 两尺度 P 下降约 0.40/0.42 个百分点至 98%，端点误差约 58.53 mm**；A-frame base 恢复仍约 33%。这不是 Pareto 改善，也不能称为准确率提高。

## 方案与失败路线的关系

同日第一候选 `geometric-occupancy-sprint-20261014-001` 无条件移除第二次 RGB 裁剪，导致 r02 全 10 行 gap 假邻近由 0 升至 17.88–53.64 mm，已整体拒绝；失败源/结果保留在[独立报告](2026-10-14-geometric-occupancy.md)。本轮是新源码、新 run ID，不在失败结果上修改评分，也不只相对失败版报告。

第二候选调用原 occupied reader，不动 RGB 准入点池、实际 TLS 模型、选轴排名、实际 ±1v 投影并集和旧外边界。只把满足以下全部条件的相邻旧输出片段连接：它们属于**同一个实际占据区间**，中间缺失区间被两侧旧输出夹住，且其轴向长度不大于 **1v**。原世界端点直接复制，不重新投影；跨实际占据分量的缺口即便小于 1v 也不连接，超过 1v 的 RGB 孔洞保留。

这仍是明确语义变化：同一占据区间内的短 RGB 未知孔洞不再强制切段，而非宣称已证明每个孔洞内是正确物体。尺度阈值在这次 normal 与新评分之前确定，没有事后调参；所有五案五臂两尺度统一用同一规则，不按案例挑版本。

## 实际运行

- 新增 6 项机制测试，6/6 通过，NumPy 1.26 用时 **0.219 s**：内部短孔连接、外端点不延伸、实际缺口不跨、>1v RGB 孔不连、多孔阈值不漂移、真实小 fixture base/curve 同坐标同读数及旧选轴不改。新核心和专属测试 Ruff 通过。
- 一次固定 5 案 × 5 臂 × 2 尺度 = 50 配置完整 normal，之后独立进程 evaluate 固定完整原 GT。normal 进程 `gt_read=false`，评分 `evaluation_gt_read=true`，无盲测声明。
- normal **104.627 s**，evaluate **2.901 s**，合计 **107.529 s**，两次退出码均 0；normal、共同读数评分、native 评分错误均 0。旧 occupied 一次捕获合计 115.046 s，耗时不是独占重复性能基准。
- 复用原 RGB/相机、DA3 点快照、补丁及旧完整对照；没有新 DA3、新端到端重复、第二次快筛、旧 control 重算、双环境、全仓回归或祖先审计。
- 原始产物位于 `.runtime/experiments/bracketed-occupancy-sprint-20261014-001/`；实际命令、退出码、stdout/stderr 与耗时在 `.runtime/validation/foreground-consensus-runners-20261012-002/bracketed-occupancy-sprint-20261014-001-{normal,evaluate}.*`。

## 全部 50 行变化

R/P 为固定容差下恢复率/精度（%），不是完整连通杆比例。比较容差 `1e-12`。

| 指标 | 上升 | 下降 | 相同 |
| --- | ---: | ---: | ---: |
| R | 0 | 0 | 50 |
| P | 0 | 2 | 48 |
| 段数 | 0 | 12 | 38 |

50/50 所选候选、TLS 模型完全相同；50/50 native 状态、几何、完整物理指标与评分错误完全相同。直接对保存结果检查：50/50 新端点都是旧端点，50/50 整体两端坐标逐值未变。这个检查不等于段内几何完全相同——12 行新连接的短孔洞是有意新增的几何，不能叫作纯重编号。

### 全部 10 个 base

R/P 均与旧版相同；下表 P 全为 100%。细/粗 camera-span fraction 为 0.0025/0.005。

| 案例 | 尺度 | R % | 段数旧 → 新 | 长度旧 → 新 m |
| --- | --- | ---: | ---: | ---: |
| r01 | 细 | 95.900 | 10 → 10 | 1.616153 → 1.616153 |
| r01 | 粗 | 98.800 | 6 → 6 | 1.823704 → 1.823704 |
| r02 | 细 | 100 | 9 → 9 | 1.543052 → 1.543052 |
| r02 | 粗 | 100 | 3 → 3 | 1.634148 → 1.634148 |
| r03 | 细 | 82.700 | 14 → 14 | 1.195340 → 1.195340 |
| r03 | 粗 | 87.400 | 9 → 9 | 1.506246 → 1.506246 |
| chair01 | 细 | 96.129 | 11 → 9 | 1.340018 → 1.352143 |
| chair01 | 粗 | 98.194 | 4 → 2 | 1.443776 → 1.455932 |
| aframe01 | 细 | 33.132 | 5 → 5 | 0.444639 → 0.444639 |
| aframe01 | 粗 | 33.736 | 1 → 1 | 0.523060 → 0.523060 |

chair base 两尺度分别增加 12.125 / 12.156 mm 的内部连接，实际占据造成的剩余碎段没有伪造性填平。A-frame base 两尺度完全未改；粗尺度端点最大误差仍 **1119.142 mm**。其余 9 行 base 仍为 `boundary_count_mismatch`，边界误差是 null，不是 0。

### chair 全部四候选与 A-frame 负结果

chair 四候选两尺度全部 **3 → 1 段，R=P=100%**，边界由数量不匹配变为可评分。每行连接两个孔洞，累计约 11.94–12.19 mm。

| chair 候选 | 细尺度最大端点误差 mm | 粗尺度最大端点误差 mm |
| --- | ---: | ---: |
| TLS | 11.719776 | 10.654459 |
| RANSAC | 12.710810 | 11.455829 |
| baseline | 10.413767 | 10.092885 |
| cylinder | 10.413767 | 10.092885 |

全部两行 P 下降都在 A-frame baseline：

| 尺度 | R 旧 → 新 % | P 旧 → 新 % | 段数 | 新最大端点误差 mm | 新增内部长度 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 细 | 100 → 100 | 98.398779 → 98.000000 | 3 → 1 | 58.531646 | 7.911063 |
| 粗 | 100 → 100 | 98.418523 → 98.000000 | 3 → 1 | 58.539720 | 8.048788 |

两行新增内部几何在固定 GT 容差下不是全部正确；外端点未移动也不保证内部连接正确。不能只报告单段化。A-frame 原生 TLS/RANSAC 仍 R=P=0、cylinder 仍空、baseline 原生 R=100%/P=97.879859%；这些原生失败没有因共同读数变化而消失。

### 真实 gap 的全部 10 行

r02 以下所有行的旧/新 `guarded_gap_false_length_m` 均为 **0**，完整物理指标也不变；本轮未重现第一候选的缺口边缘侵入。

| r02 臂 | 细：旧 → 新假邻近 mm | 粗：旧 → 新假邻近 mm |
| --- | ---: | ---: |
| base | 0 → 0 | 0 → 0 |
| TLS | 0 → 0 | 0 → 0 |
| RANSAC | 0 → 0 | 0 → 0 |
| baseline | 0 → 0 | 0 → 0 |
| cylinder | 0 → 0 | 0 → 0 |

该指标是在原声明 gap 的 guarded 内部、处于预测有限曲线容差内的长度，不是一般拓扑证明。其他案例无声明 gap，其零值不是缺口保持证据。

## 留下的主线工作

主代理依据减少碎段、chair 候选得到正确单段、保留原外端点、r02 真 gap 无回退，决定统一采用第二版；不按案例/分支挑新旧版。本轮解决一类内部 RGB 裂缝，但 r03 实际占据碎段、A-frame 基础支持不足和候选边界误差仍在。新增短连接同时保留 A-frame 两行精度负收益；不能据此宣称 reader qualified、真实实拍泛化或 G1 完成。G1 进度由主代理在交接记录。

完整 50 行新旧物理指标、逐项变化、桥接计数/长度、25 个唯一 native、源 SHA 和实际 capture 见[紧凑 JSON](results/2026-10-14-bracketed-occupancy.json)。助手动作记录 `.runtime/work-logs/20261014-readout-support.md`，两个导出脚本同目录；导出只整理已有评分，没有第二套重算。本轮无新 DeepSeek 任务。

## 可运行入口

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_bracketed_occupancy_readout.py -v
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py normal --adapter creator_eval.rgb_bracketed_occupancy_readout:BracketedOccupancyReadout --run-id NEW_UNIQUE_RUN_ID
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py evaluate --run-id NEW_UNIQUE_RUN_ID
```

已有 run ID 拒绝覆盖；本次实际 ID 是 `bracketed-occupancy-sprint-20261014-001`。第一失败候选与本候选各自仅跑一次完整快筛。
