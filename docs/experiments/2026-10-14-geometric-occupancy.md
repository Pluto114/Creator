# 10月14日：移除第二次 RGB 裁剪的完整开发对照

**第一版已拒绝；后来统一开发版本采用 `bracketed-occupancy-sprint-20261014-001`，见[第二版独立报告](2026-10-14-bracketed-occupancy.md)。** 本报告的失败第一版 `geometric-occupancy-sprint-20261014-001` 完成唯一一次 50 配置 normal → evaluate。相对原统一旧版 `occupied-axis-sprint-20261012-001`，恢复率 50 行全相同，段数减少 12 行、边界可配对由 26 增至 36 行；但精度下降 7 行，r02 全部 10 配置在真实 gap 内新增假邻近长度。不按案例保留有利分支，不构成共同读取资格或 G1 验收。

## 改了什么

先做 normal-only 几何诊断：chair baseline 与 A-frame baseline 在两尺度均为选中实际占据候选 1 段、最终 RGB 裁剪后 3 段。chair 两个裂缝约 6.57 / 5.55–5.62 mm，A-frame 约 1.06–1.19 / 6.85–6.86 mm，均不是浮点误差。

新读数完整调用旧 occupied reader，保持同一个 RGB 准入点池、种子扩展、前景带排名和所选 TLS 轴，然后直接输出所选候选的实际三维占据区间，不再对输出进行第二次 RGB cells 裁剪。没有 refit、外推、图像构造几何或新增标注；base/curve 采用同一输入筛选与采样流程。旧源码和旧结果不变。

这是明确的支持策略变化：短未知 RGB 行不再否决已有实际投影占据。输出仍是各实际点投影 ±1v 邻域的并集，截在实际投影 min/max 内。代码 `minimum_resolvable_geometric_gap_voxels=2` 指**相邻实际点轴向投影间距的连接阈值**：间距 2.01v 留下 0.01v 空隙；不是宣称能分辨某个尺寸的真实物理缺口。简单的 >2v 投影缺口测试通过，不代表真实 gap 的边缘扩张安全。

## 实际运行与对照来源

- 新增 6 项机制测试全部通过（NumPy 1.26，0.445 s）；专属测试及新核心 Ruff 通过。仅验证选轴不变、精确候选输出/端点、短未知行的新语义、>2v 投影缺口、点重排重复、base/curve 同坐标同读数。
- 唯一一次完整开发集：5 案 × 5 臂 × 2 尺度 = 50 行。先完整 normal 落盘，再独立进程 evaluate 固定完整原 GT。normal、读数评分、native 评分错误各 0，两阶段退出码均 0。
- normal 捕获墙钟 **125.608 s**，evaluate **2.758 s**，合计 **128.366 s**。旧版对应 112.610 / 2.436 / 115.046 s；不是性能加速结果，也不是独占重复的性能基准。
- 复用原 RGB、相机、DA3 点快照及已保存候选，无新 DA3、端到端重跑、重复快筛、旧 control 重算、全仓回归或祖先审计。
- 50/50 所选候选及 TLS 模型完全相同；50/50 native 状态、几何、完整物理指标、评分错误完全相同。对照读取旧保存结果，不重新评分。
- 本轮是已见开发集；助手在新源和测试固定、normal 已启动后，为复用报告格式读了旧公开报告，随后未改源码/参数。normal 进程未读 GT，以上均不作盲测声明。

原始 normal/evaluation/manifest/stage logs 位于 `.runtime/experiments/geometric-occupancy-sprint-20261014-001/`，实际命令与 stdout/stderr/退出码位于 `.runtime/validation/foreground-consensus-runners-20261012-002/geometric-occupancy-sprint-20261014-001-{normal,evaluate}.*`。

## 完整结果概览

R/P 为固定容差下恢复率/精度（%），不是完整连通杆比例。比较容差为 `1e-12`。

| 全部 50 行 | 上升 | 下降 | 相同 |
| --- | ---: | ---: | ---: |
| R | 0 | 0 | 50 |
| P | 0 | 7 | 43 |
| 段数 | 0 | 12 | 38 |

12 行段数减少由 chair 全 10 行和 A-frame baseline 两行构成。chair 四候选两尺度均 3 → 1 段，R/P 保持 100/100，边界由数量不匹配变为可评分，最大误差约 10.09–13 mm。A-frame baseline 虽 3 → 1 段，P 下降至 98%，两尺度最大边界误差约 58.53 mm。段数和边界可评分数量改善不能抵消真实缺口表现倒退。

### 全部 10 个 base

R 均与旧版相同；除 r02 粗尺度外，新旧 P 均 100%。细/粗对应 camera-span fraction 0.0025/0.005。

| 案例 | 尺度 | R % | P 新 % | 段数旧 → 新 | 长度旧 → 新 m |
| --- | --- | ---: | ---: | ---: | ---: |
| r01 | 细 | 95.900 | 100 | 10 → 10 | 1.616153 → 1.624256 |
| r01 | 粗 | 98.800 | 100 | 6 → 6 | 1.823704 → 1.831057 |
| r02 | 细 | 100 | 100 | 9 → 9 | 1.543052 → 1.562737 |
| r02 | 粗 | 100 | 99.763 | 3 → 3 | 1.634148 → 1.685908 |
| r03 | 细 | 82.700 | 100 | 14 → 14 | 1.195340 → 1.195340 |
| r03 | 粗 | 87.400 | 100 | 9 → 9 | 1.506246 → 1.506246 |
| chair01 | 细 | 96.129 | 100 | 11 → 9 | 1.340018 → 1.355168 |
| chair01 | 粗 | 98.194 | 100 | 4 → 2 | 1.443776 → 1.458912 |
| aframe01 | 细 | 33.132 | 100 | 5 → 5 | 0.444639 → 0.444639 |
| aframe01 | 粗 | 33.736 | 100 | 1 → 1 | 0.523060 → 0.523060 |

A-frame base 仍未解决：粗尺度虽 1 段，最大端点误差仍 1119.142 mm。其余 9 行 base 仍是边界数量不匹配，最大边界误差为 null，不是 0。

### 所有精度下降行

| 案例/臂 | 尺度 | P 旧 → 新 % |
| --- | --- | ---: |
| r02 base | 粗 | 100 → 99.763215 |
| r02 TLS | 粗 | 100 → 99.176434 |
| r02 RANSAC | 粗 | 100 → 99.411806 |
| r02 baseline | 粗 | 100 → 99.647030 |
| r02 cylinder | 粗 | 100 → 99.647023 |
| aframe01 baseline | 细 | 98.398779 → 98.000000 |
| aframe01 baseline | 粗 | 98.418523 → 98.000000 |

### 真实 gap：全部 r02 行

旧版以下 10 行 `guarded_gap_false_length_m` 均为 0，新版全部上升。这里是 GT gap 加端点 guard 后内部、落入预测有限线段容差邻域的长度，不等同于整段被连通；真实中央空隙仍在，但缺口边缘已出现错误侵入，必须保留这个失败。

| r02 臂 | 细：假邻近 mm | 粗：假邻近 mm |
| --- | ---: | ---: |
| base | 17.880795 | 51.655629 |
| TLS | 21.854305 | 49.668874 |
| RANSAC | 17.880795 | 49.668874 |
| baseline | 21.854305 | 53.642384 |
| cylinder | 21.854305 | 53.642384 |

其他案例没有声明物理 gap，零值不是缺口保持证据。r02 粗尺度四候选最大端点误差也变差，例如 baseline 8.201 → 29.326 mm。

## 原生失败与下一步

所有 native 原样保留。A-frame TLS/RANSAC 原生 R=P=0，cylinder 原生为空；baseline 原生 R=100%、P=97.879859%。新共同读数的某项改善不能宣称原生补丁因此改好。

主代理据完整结果拒绝此路线，本报告不提高 G1 估算。下一独立候选只处理同一实际占据区间内被两侧旧输出夹住的短 RGB 孔洞，保留旧输出外边界；不得跨实际占据缺口。该候选使用新源码和新 run ID，本失败版源/结果不覆盖。

全部 50 行新旧完整物理指标、差值、25 个唯一 native 指标、源标识及真实 capture 记录见[紧凑结果 JSON](results/2026-10-14-geometric-occupancy.json)。助手关键动作日志为 `.runtime/work-logs/20261014-readout-support.md`；两份 normal-only 诊断及导出脚本同目录。本轮是本地助手执行，没有新增 DeepSeek 会话或虚构助手活动。

## 可运行入口

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_geometric_occupancy_readout.py -v
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py normal --adapter creator_eval.rgb_geometric_occupancy_readout:GeometricOccupancyReadout --run-id NEW_UNIQUE_RUN_ID
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_readout_sprint.py evaluate --run-id NEW_UNIQUE_RUN_ID
```

本次实际使用 `geometric-occupancy-sprint-20261014-001`；已有运行拒绝覆盖。以上最后两条用于显式复现，不表示本轮又执行了第二次。
