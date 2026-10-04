# 训练采样尺度的稀疏裸杆读取（2026-10-04）

## 决策与范围

用户明确“恢复研发”后，从 `828aff72d34db244601f480b9a6d394470af592e` 继续，实际项目 `D:/Creator-newage`。该基点的 [Windows/Linux CI](https://github.com/Pluto114/Creator/actions/runs/37201075754) 已成功；不能据此声称本轮提交的 CI 已通过。

主线解决裸杆约13mm采样与固定1.75voxel连接/voxel占据计数的冲突。原 `context` 所有源码、已评分结果、输入、GT和快照保留。新实现不改变截面角门、径向内管、物理最短长度、双折覆盖门和RGB测量域。

- `ridge_sampling_evidence.py`：只用训练近轴投影估计重复采样pitch。合并0.05voxel近重复投影，至少6层；短间距25%分位附近±20%形成至少3项且占20%的重复簇；至少80%间距满足整数倍pitch±15%；pitch最多8voxel。密集/不足/不规则时保留原尺度，预算不足显式unmeasurable。
- `common_readout_ridge_sampling.py`：采用该pitch进行轴向连接和采样站占据计数。稀疏站使用中心化bins避免微小轴向扰动合并相邻站；输出端点仍来自实际支持，不生成跨缺口点。新增逐候选及合并后gate诊断。
- `ridge_array_sampling_evidence.py`：平行排证据使用同样训练采样尺度；三条非共线横截面排中心不再被旧“三根平行即否决”无条件拒绝，实际否决仍需训练提出的共线重复排和独立验证确认。旧triplet计数保留为诊断。
- `ridge_sampling_proposals.py`：训练局部6/12邻居候选，缓解短少数支杆被12邻居混入主杆；仅提议，不是接受证据。预算仍统一约束。

这些是有条件的研究假设，不能证明小于报告allowance的物理断口不存在。此轮同时改变轴向采样度量、方向提议与平行排判定，不是单因素消融。

## 协作与预验证

DeepSeek Harness通过独立文件交付工作区参与16项采样机制测试；只提供两份源快照，未提供真实输入/GT/评分，不允许直接写主仓库或执行命令。实际文件允许清单和control SHA校验通过，但回复JSON前多出说明，strict schema失败，原回执保留为失败。Codex人工审查后修正默认字段位置、dense reason和合法层预算3处规格假设，再集成测试。实际测试由Codex执行，不归功于DeepSeek。

106项新增/复用契约已在NumPy1.26和NumPy2通过，涵盖原几何拒绝、稀疏双尺度、真实大缺口、平行排/三非共线杆、训练/验证隔离、预算、逐gate、runner所有正常记录先于GT及原mask/完整候选绑定。旧私有`_ridge_runs`测试仅适配新增第三个audit返回值，保留原几何与断言。CI新增ridge sampling测试入口。

冻结前对已见旧条件做normal-only开发预览：0164–0167均输出两杆；0168/0169/0171输出三主杆但缺短支杆，0170输出四段。该预览不是正式评分，更不是盲测；保留其发生事实。

## 冻结协议

- 解析 `mixed-readout-ridge-sampling-v1-20261004`，入口 `scripts/run_mixed_readout_ridge_sampling.py prepare|infer|evaluate|post`。
- 全部184条件为已见开发回放，原五组44/36/40/32/32分开报告，没有新构造、独立对象或holdout。
- 736正常行＝368新读取执行＋368逐输入/策略/字节核验的原context引用。缓存elapsed=0不是速度基准。
- 继承闭合source inventory并显式纳入新依赖；153份源、1601份pre收据。完整736正常行及来源核验后才可访问隔离GT；不覆盖失败或评分后改源。
- fixture `fixture-ridge-sampling-readout-v1-20261004`，入口 `scripts/run_fixture_ridge_sampling_readout.py`。18条原完整base/candidate，冻结mask、相机、meter frame、curve采样、投票计数不变。解析正常完整性仅作先决条件，不用其分数挑选或推广。

## 正式结果：有恢复，也有拒绝退化，不推广

解析四阶段完成，153源/1601份pre收据不变，736正常行均complete、无方法/评分错误，368表示对全同，post精确复算通过。正常耗时536.596s；与全量测试部分并行，不作受控性能对比。新小邻域候选增加了工作量，后续需同时约束候选规模，不以缓存引用的零耗时声称加速。

| native正例 | context R/P≥90% / 有限结构门 | ridge_sampling R/P≥90% / 有限结构门 |
| --- | --- | --- |
| 旧混合24 | 24 / 24 | 24 / 24 |
| 旧截面20 | 20 / 20 | 20 / 20 |
| 旧采样28 | 28 / 28 | 28 / 28 |
| 旧支持16 | 12 / 12 | 12 / 12 |
| 旧局部20 | 11 / 11 | 19 / 16 |

有限结构门仍同时要求R/P≥90%、精确段数、边界双射≤25mm、受保护缺口误填为0；不是一般拓扑指标。4项双裸杆完整恢复，非共面三杆+短支路4项中1项完整恢复、3项只恢复三根主杆。后三项虽然R/P达到90%，仍判有限结构失败。全表96条带缺口记录保护后误填均为0。

必须保留的坏结果：

- 28个native负例从0项误输出变为3项：`0148/0149/0150` noisy_plane各输出1段。
- 48个native未决从0项误输出变为6项：`0032..0035` sparse_cylinder及`0036/0037` sparse_half_cylinder各输出4段。
- 4项渐变半径正例仍失败：`0132/0133`从空变为各30碎段，`0134/0135`仍空。真缺口半壳`0160`仍空；短支杆`0168/0169/0171`仍缺失。

正常诊断显示，稀疏圆柱4个训练横截面排中心不共线，现有array全部以`no_local_collinear_repeated_spacing`退出，旧triplet诊断会标记4段。这说明“只检查共线排”不能覆盖稀疏曲面歧义，不能把旧veto降为诊断后当作普遍可靠改进。渐变半径两项的旧triplet均标记30段，现有局部模型多为不足3排或heldout未确认。noisy_plane中0148/0149未获heldout确认，0150仅2个持续训练排；这只是正常证据定位，不是单因素因果消融。

[全部736行](results/2026-10-04-mixed-readout-ridge-sampling.json)、[后审计](results/2026-10-04-mixed-readout-ridge-sampling-audit.json)。所有坏行原样保存，不评分后修改方法。

## 完整候选对照与验证

fixture四阶段完成：174源、5296份pre收据、18行、46.373s，全部complete、无评分错误、post精确复算通过。12候选仍R=P=100%、正确段数、最大端点误差8.407mm、最大绝对长度误差22.765mm、缺口误填0；6纯base仍空。与context配对物理结果18/18相同，所有完整输入/原mask/采样投票约束不变；不声称全场景或独立对象改进。[18行](results/2026-10-04-fixture-ridge-sampling-readout.json)、[后审计](results/2026-10-04-fixture-ridge-sampling-readout-audit.json)。

实际验证：106项新增/复用契约双NumPy通过；全量1100项通过（435.379s）；核心16项/15子检查、Ruff、315源/三CLI、evaluation sdist/wheel及Blender包通过。协作桥41项Node测试通过，原协调会话`research-resume-ack-001`严格JSON确认成功，0次工具调用。机制测试通过不代表研究方法合格，本次9项反例/未决退化明确阻止推广。

## 下一步

1. 保留已验证的训练cadence模块，下一版本补稀疏曲面/渐变曲面歧义证据，区分共线平面排与非共线多母线；不要恢复“一律三杆否决”，也不能用只共线代替所有歧义。对小邻域候选和veto变更分别消融，并处理噪声平面残余，先清除本轮9项误输出。
2. 再处理短杆训练/验证采样不足及折分失衡：0168/0169正常候选没有得到足够训练层，0171明确记录验证覆盖不足；不把3主杆高R/P当完整支路恢复，不直接降低已有双折门。
3. 渐变半径和0160训练横截面分组仍是原问题，后续新源/新run处理。当前184条件全部已见开发材料，不再扩类似构造赚通过率；反例与纯base未解决前不切主流程、不宣称G1。

完整实现入口：[读取器](../../experiments/src/creator_eval/common_readout_ridge_sampling.py)、[训练采样模型](../../experiments/src/creator_eval/ridge_sampling_evidence.py)、[局部阵列](../../experiments/src/creator_eval/ridge_array_sampling_evidence.py)、[候选提议](../../experiments/src/creator_eval/ridge_sampling_proposals.py)。两个run四阶段已用完，不可覆盖重跑。

提交前复核5296份最终pre收据及174依赖源未变，16份新增冻结源/配置/测试与Git暂存字节完全一致，空白检查通过。协作配置留在本地独立目录，不将账户配置/凭据上传仓库。
