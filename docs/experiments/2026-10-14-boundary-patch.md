# 10月14日续轮：真实 baseline 边界补丁

**统一采用新增研究候选 `baseline_boundary`，保留原 baseline 冻结对照，共同 reader 仍为 bracketed。** 这是 `adopted_new_native_candidate_not_reader_upgrade`：不是生产切换、共同读取资格或 G1 验收，也不按案例/GT选择启用。

实际五个 native 补丁中，A-frame 最大端点误差 **59.356 → 47.484 mm**；同一 bracketed reader 下两尺度 **58.532/58.540 → 46.451/46.492 mm**，其他四案例不变。收益约 12 mm，不是此前被拒绝的 reader-only 试验中约 17 mm 的最终误差。五个 native、十个新共同读数均无 R、段数、gap 回退，但 A-frame 边界问题仍未解决。

## 实现、验证与实际执行

前一[consensus 共同 reader 路线](2026-10-14-consensus-boundary.md)因18行恢复下降而整体拒绝。本轮不再动读取规则，而在所有五案例原生 baseline 自身的有限轴上，应用同一无 padding、同链至少4个不同视图的正支撑外包络；只取交集，不 refit、不延长、不填内部 gap、不抑制 base 点。所有原 baseline 都保留，新方法统一是第六臂。

- 主代理实现 `rod_boundary_patch.py`；endpoint 助手实现运行入口和保存/撤回；readout 助手写5个契约测试与全量结果整理。本轮没有新增 DeepSeek 任务。
- 只跑新5项测试，5/5通过（0.396s）及新核心/测试 Ruff；覆盖原生轴/内部gap、不延长、输入/context不变、空输出拒绝、完整context、不同native共享规则和noGT元数据。没有重跑前两组12项测试。
- `boundary-patch-sprint-20261014-001` normal先产生**5个实际保存的补丁、10个新共同读数，加50个明确复用的旧读数，共60行**；完整normal落盘后，独立evaluate对60行一次完整原GT评分。
- normal退出0，**95.388338s**；evaluate退出0，**2.538977s**；合计 **97.927315s**。stderr均空，normal/共同评分/native评分错误均0。
- 旧50的共同几何/readout没有重算；evaluate确实为全部60行计算物理指标，随后与旧保存分数作精确比较。不能把旧读数说成新运行，也不声称旧physical评分没有重算。
- 没有新DA3、新RGB采集、独立端到端重复、第二遍快筛、全仓测试、双环境或祖先审计。是已见合成开发集，normal未读GT；不作盲测声明。

## 五个唯一 native：不是十次独立补丁

每个真实补丁在两个reader尺度重复使用，native只按五案例计一次。R/P为原固定容差下恢复率/精度（%）；E为最大端点误差（mm）。

| 案例 | R 旧 → 新 | P 旧 → 新 | 段数 | E 旧 → 新 mm |
| --- | ---: | ---: | ---: | ---: |
| r01 | 100 → 100 | 100 → 100 | 1 → 1 | 8.406950 → 8.406950 |
| r02 | 100 → 100 | 100 → 100 | 2 → 2 | 8.925372 → 8.925372 |
| r03 | 100 → 100 | 100 → 100 | 1 → 1 | 5.771170 → 5.771170 |
| chair01 | 99.225806 → 99.225806 | 100 → 100 | 1 → 1 | 36.070032 → 36.070032 |
| aframe01 | 100 → 100 | 97.879859 → 98.574822 | 1 → 1 | 59.356235 → 47.484091 |

A-frame 原生长度 **1.697579243 → 1.682842441m**，收短14.736802mm。现有物理摘要没有独立“误补线长”字段；只从已有 `L × (1−P)` 派生固定容差外长度代理：**35.991079 → 23.983503mm**，减少 **12.007577mm**。没有重评分；这不是另行测得的拓扑假杆长度，不能等同于所有错误几何长度。

r02 新native的 guarded-gap 假邻近长度仍为0，真实内部gap保留。chair原生仍有36.070mm端点误差且R不足100%，不能以共同读数更好掩盖原生失败。

## 新10行共同读数与旧baseline

共同reader固定为 `BracketedOccupancyReadout`，不是本轮新边界reader。

- 新10行R、段数、gap均与旧baseline相同；R全100%。P改善2行、下降0、相同8；最大端点误差改善2行、变差0、相同8。
- 前四案例8行完整物理指标不变；只有A-frame两尺度改变：P **98 → 98.696682%**，最大端点误差分别 **58.531646 → 46.450961mm**、**58.539720 → 46.491996mm**。
- 所有旧50行完整graph、native几何/状态、共同/native物理评分及错误记录，均与旧bracketed记录**逐值完全一致**。旧base没有被替换或重新筛选。
- r02完整六臂两尺度共12行，guarded-gap假邻近长度全为0；其他案例没有声明物理gap，零值不是gap保持证据。

## 全六臂60配置

每格为 **R% / P% / 段数 / 最大端点误差mm**。细/粗fraction为0.0025/0.005；30行×两尺度覆盖全部60配置。`null`是边界数量不匹配，绝不是0。

| 案例 | 臂 | 细 | 粗 |
| --- | --- | --- | --- |
| r01 | base | 95.900/100.000/10/null | 98.800/100.000/6/null |
| r01 | depth_tls_single_span | 99.600/100.000/1/31.998 | 99.700/100.000/1/31.547 |
| r01 | depth_ransac_single_span | 97.700/100.000/2/null | 99.700/100.000/2/null |
| r01 | baseline | 100.000/100.000/1/7.640 | 100.000/100.000/1/12.335 |
| r01 | cylinder_support | 100.000/100.000/1/7.640 | 100.000/100.000/1/12.335 |
| r01 | baseline_boundary | 100.000/100.000/1/7.640 | 100.000/100.000/1/12.335 |
| r02 | base | 100.000/100.000/9/null | 100.000/100.000/3/null |
| r02 | depth_tls_single_span | 100.000/100.000/2/19.402 | 100.000/100.000/2/20.296 |
| r02 | depth_ransac_single_span | 100.000/100.000/2/13.480 | 100.000/100.000/2/15.309 |
| r02 | baseline | 100.000/100.000/2/8.879 | 100.000/100.000/2/8.201 |
| r02 | cylinder_support | 100.000/100.000/2/8.803 | 100.000/100.000/2/8.515 |
| r02 | baseline_boundary | 100.000/100.000/2/8.879 | 100.000/100.000/2/8.201 |
| r03 | base | 82.700/100.000/14/null | 87.400/100.000/9/null |
| r03 | depth_tls_single_span | 98.300/100.000/1/59.861 | 98.200/100.000/1/60.065 |
| r03 | depth_ransac_single_span | 98.300/100.000/1/59.840 | 98.200/100.000/1/60.033 |
| r03 | baseline | 100.000/100.000/1/5.775 | 100.000/100.000/1/6.726 |
| r03 | cylinder_support | 100.000/100.000/1/5.775 | 100.000/100.000/1/6.726 |
| r03 | baseline_boundary | 100.000/100.000/1/5.775 | 100.000/100.000/1/6.726 |
| chair01 | base | 96.129/100.000/9/null | 98.194/100.000/2/null |
| chair01 | depth_tls_single_span | 100.000/100.000/1/11.720 | 100.000/100.000/1/10.654 |
| chair01 | depth_ransac_single_span | 100.000/100.000/1/12.711 | 100.000/100.000/1/11.456 |
| chair01 | baseline | 100.000/100.000/1/10.414 | 100.000/100.000/1/10.093 |
| chair01 | cylinder_support | 100.000/100.000/1/10.414 | 100.000/100.000/1/10.093 |
| chair01 | baseline_boundary | 100.000/100.000/1/10.414 | 100.000/100.000/1/10.093 |
| aframe01 | base | 33.132/100.000/5/null | 33.736/100.000/1/1119.142 |
| aframe01 | depth_tls_single_span | 33.132/100.000/5/null | 33.736/100.000/1/1119.142 |
| aframe01 | depth_ransac_single_span | 33.132/100.000/5/null | 33.736/100.000/1/1119.142 |
| aframe01 | baseline | 100.000/98.000/1/58.532 | 100.000/98.000/1/58.540 |
| aframe01 | cylinder_support | 33.132/100.000/5/null | 33.736/100.000/1/1119.142 |
| aframe01 | baseline_boundary | 100.000/98.697/1/46.451 | 100.000/98.697/1/46.492 |

A-frame base仍只恢复约33%；TLS/RANSAC原生R=P=0、cylinder原生为空。这些旧失败原样保留，新候选改善一个端点不等于项目全面达标。

## 五个实际补丁与成本

全部五个补丁均为accepted。每案都写出新的patch及enabled/withdrawn视图，从磁盘重开，验证基础points/IDs不变、关闭patch精确恢复base、enabled段与新native一致；无点抑制。不是仅在内存里换段。

| 案例 | refine秒 | patch+两视图新增bytes | 独立base复制bytes | 重开/基础不变/精确撤回 |
| --- | ---: | ---: | ---: | --- |
| r01 | 0.023156 | 2897 | 34294493 | 全通过 |
| r02 | 0.017524 | 2987 | 34294493 | 全通过 |
| r03 | 0.014150 | 2897 | 34294493 | 全通过 |
| chair01 | 0.014555 | 2897 | 34294493 | 全通过 |
| aframe01 | 0.163430 | 2897 | 34294493 | 全通过 |

总refine **0.232816s**，十个新共同readout调用合计 **7.066580s**；完整normal仍花95.388338s，不能把局部计算时长当端到端时长。新patch/两视图合计 **14,575bytes**，另有独立base复制 **171,472,465bytes**（约171.47MB），两者分别披露；这些不是整个实验目录（还含日志/评分）的总占用。

## 交付与限制

[完整JSON](results/2026-10-14-boundary-patch.json)包含全部60共同配置、30个唯一native（五案六臂）、5个新native对照、10个新共同对照、50个旧control一致性、patch标识/保存路径、方法/源码SHA、真实capture及派生量说明。导出只整理既有评分，不重复评分。

原始产物 `.runtime/experiments/boundary-patch-sprint-20261014-001/`；命令/退出码/stdout/stderr在 `.runtime/validation/foreground-consensus-runners-20261012-002/boundary-patch-sprint-20261014-001-{normal,evaluate}.*`。助手动作日志 `.runtime/work-logs/20261014-boundary-readout-support.md` 与endpoint助手日志均记录实际分工。

共同读取资格、基础支持、剩余约47mm原生边界误差、新补丁版本端到端重复仍未关闭；本轮保持G1工程估算80%，不是验收。

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_rod_boundary_patch.py -v
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_boundary_patch_sprint.py normal --run-id NEW_UNIQUE_RUN_ID
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_boundary_patch_sprint.py evaluate --run-id NEW_UNIQUE_RUN_ID
```

本次实际ID为`boundary-patch-sprint-20261014-001`；必须新ID，既有结果拒绝覆盖。
