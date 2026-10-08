# 10月8日：完整候选池的跨视图同链支持

结论：新归属实现显著改善 r01/r02 的共同读出，但 r03 和椅背 base 回退，Aframe 仍不能形成唯一主轴。**G1:55%，主流程不替换。** 这是已见合成开发验证，不是实拍、盲测或新的 DA3 端到端重复。

## 决策与实现

只改共同 RGB 支持，不改原 single-axis TLS、实际点/补丁、采样密度、相机或完整 GT 域。原 raw-union 与新 chain-mask 同时运行，五臂为 base、TLS、RANSAC、baseline、cylinder_support；不把不同结构的像素证据拼成三票。

完整实现：[关联与支持](../../experiments/src/creator_eval/rgb_chain_support.py)、[适配器](../../experiments/src/creator_eval/rgb_chain_axis_readout.py)。完整原图采样池不 cap8；两投影平面交线生成种子，sine≥0.05，所有视图内9个等距y的水平中位残差≤1.5px，至少4个视图形成一条可行链。中点必要条件只加速，不替代完整残差判断。原始行保留全部垂直距离≤1.2px的实际边对，宽窄同中心对也保留，不填候选之间的 envelope。点必须在**同一条链**内得到至少3个不同视图支持，然后才对链取并集。

只对完全相同的成员签名去重，不按排名丢分支。50万配对/10万链预算，超限则该案所有臂共同退回原 mask；不会把搜索缺失当成负证据。10万链预算在首次正式冻结及评分前确定：正常性能检查证实2万不足以覆盖Aframe。固定4096点分块和位集合计数降低内存，结果与朴素布尔运算等价。

这只穷举固定、已采样图像池中的非退化 pair proposals，不穷举所有噪声最小二乘解，也不证明目标身份。多条可能链的并集仍可能含多个物体；其 TLS 输出是实际3D几何假设，可能平均竞争结构，不能叫已确认的前景杆。

## 完整实验与审计

| run | 源 / pre收据 | 正常行 | 正常耗时 | 独立进程重复 |
| --- | ---: | ---: | ---: | ---: |
| fixture-rgb-chain-readout-v1-20261008 | 282 / 9888 | 120 | 63.433s | 60/60一致 |
| g1-object-rgb-chain-readout-v1-20261008 | 385 / 1340 | 80，另10 native | 74.923s | 40/40一致 |

两个 run 的全部正常记录、关联文件及源收据均核验后，才进入本轮任何 GT 评分；外部调度屏障见 `.runtime/validation/rgb-chain-runners-20261008-001/dual-normal-barrier.json`。两套 prepare/infer/evaluate/post 全部成功，0运行/评分错误；所有空、错位和碎段都保留。

结果：[fixture完整120行](results/2026-10-08-fixture-rgb-chain-readout.json)、[后审计](results/2026-10-08-fixture-rgb-chain-readout-audit.json)、[对象完整80行及10 native](results/2026-10-08-g1-object-rgb-chain-readout.json)、[后审计](results/2026-10-08-g1-object-rgb-chain-readout-audit.json)。200行包含两个读取策略及两次重复，只有5个已见案例，不是200个独立样本。

独立正常审计重建五案 mask 并确认真子集，全部两重复关联内容相同、完整且未回退：

| 案例 | 原raw支持点 → chain支持点 | 配对数 | 唯一链数 |
| --- | ---: | ---: | ---: |
| r01 | 569 → 472 | 2286 | 32 |
| r02 | 555 → 390 | 631 | 18 |
| r03 | 120 → 103 | 622 | 1 |
| chair01 | 271 → 221 | 1866 | 5 |
| aframe01 | 13245 → 9041 | 256021 | 28774 |

完整base每案仍952560点，表中是共同支持域内计数，不是删除源快照。100对原/new native几何、完整输入及原始RGB票数均不变。60条fixture raw控制与旧single-axis核心图相同，40条object raw控制与10月6/7日global核心图相同（忽略计时及新增元数据）。

## Base：收益与回退一起报告

R/P为百分比；N为段数；长度误差与边界误差单位mm。以下只列repeat 0，repeat 1完全一致。所有非空base都仍有多段/边界数量不匹配，不能用高R/P冒称完整恢复。

| 案例/尺度 | 原 R/P/N | 新 R/P/N | 新长度误差 |
| --- | --- | --- | ---: |
| r01细 | 39.3 / 46.22 / 5 | 55.0 / 64.95 / 6 | -530.288 |
| r01粗 | 44.4 / 49.52 / 4 | 69.7 / 79.21 / 4 | -392.788 |
| r02细 | 0 / 0 / 3 | 95.64 / 100 / 7 | -257.993 |
| r02粗 | 0 / 0 / 3 | 100 / 100 / 3 | -56.319 |
| r03细 | 47.0 / 100 / 4 | 43.5 / 100 / 4 | -1328.187 |
| r03粗 | 68.9 / 100 / 4 | 53.0 / 100 / 3 | -1088.240 |
| chair细 | 97.42 / 100 / 5 | 91.74 / 100 / 6 | -318.060 |
| chair粗 | 99.74 / 100 / 2 | 95.87 / 100 / 2 | -142.078 |
| aframe细 | 0 / 不适用 / 0 | 0 / 不适用 / 0 | -1653.028 |
| aframe粗 | 0 / 不适用 / 0 | 0 / 不适用 / 0 | -1653.028 |

旧fixture新策略60行均非空；对同run原raw策略，40行R上升、8行下降、12行R不变，物理结果均有变化。下降项为r03两尺度base的4重复行及粗TLS/RANSAC的4重复行（98.3→98.2）。无受保护缺口误填，但不能抵消碎段、错位与缺失。

候选方面：r02全部4候选两尺度R=P100、正确2段，最大边界19.402mm；r03 baseline/cylinder两尺度R=P100、正确1段、最大边界6.726mm。r01 baseline/cylinder改善至细80.6/80.95、粗89.4/89.92，仍不足90/90；TLS/RANSAC也未恢复完整。r03朴素TLS/RANSAC仍约98.2–98.3覆盖，边界约60mm。

新对象40条chain行对raw没有R上升，4条chair base行下降；20条Aframe行物理不变且全空。chair四候选两尺度均R=P100、1段，新边界最大20.377mm；朴素方法同样达到，因此不能声称研究方法全面超过朴素对照。

## 关键阻断与原生错误

Aframe已有28774条链、9041支持点，失败不是无观测或预算截断。所有20条chain重复行的正常原因都是 `not_a_unique_dominant_axis`，次特征值比约0.43231–0.52991，高于原0.2门；完整RGB链并集仍混合多个结构。baseline曲线细213/粗107个支持采样全部保留，也未解除整场单轴不成立。不会降低该门来制造“成功”。

原生几何未被新读取器裁剪隐去：Aframe TLS/RANSAC分别添加1.790420m/1.079930m错误杆，原生R=P0，边界误差462.911/872.758mm；baseline原生R100/P97.88、边界59.356mm；cylinder仍拒绝、无新增。两新对象没有真gap，gap=0不能证明缺口安全。

下一核心决策：将前景身份选择和几何重建分开，复用现有 `rod_foreground_identity` 输入契约，推进可审计的双视图前景选择/分链支持入口；所有对照共用，新增选择来源、操作量与失败必须明示。粗guide不自动充当前景点击，不以GT/native最近轴选目标。Aframe需要明确归属，r03/chair则须保住稀疏base的原始覆盖；不能继续将全链并集直接当作一根杆。完成这一点后才值得做真正DA3端到端重复。本轮没有新增点击输入，也没有执行这项后续实验。

## 成本、测试与日志

正式正常耗时包含关联、整base支持、读取和收据验证，并与全量测试共享CPU，不作性能基准。逐臂reader调用时间不含context构建/base缓存：fixture raw/new合计0.287/0.442s；对象raw/new合计0.437/9.843s，new单次最大0.842s。冻结前Aframe单次正常性能检查context约3.98s/base-mask约5.94s，单列为诊断而非正式完整成本。没有新增人工提示或DA3运行；原输入/标定成本依旧保留。

新增55契约双NumPy通过（16核心、6DeepSeek、33runner）；一次完整全量1827/1827通过，0失败/错误/跳过，测试runner117.991s；core pytest16+15 subtests、Ruff、374源/3CLI通过。662源/config/workflow前后SHA相同。离线三包4.177/1.676/1.911s成功，6产物；eval wheel新核心模块字节匹配。日志分别在 `.runtime/validation/numpy126-full-20261008-003`、`packages-20261008-003`。

早期正式阶段的PowerShell transcript只捕获命令与起止，未捕获原生stdout；保留该限制，不重写历史日志或补造输出。实际阶段退出与清单SHA另存commands/outcome，详细每行产物在records。DeepSeek只交付6项机械测试和本人日志，Codex独立执行；固定MD/JSON总览18任务，验收关联原/集成SHA。开发详见[当天日志](../journal/2026-10-08.md)。

关键收据：

- fixture pre `0032e8a6c1e7d35b84e345294cff1a013a30df84ebf6d705dceedecc87eb8654`；normal `e54264bf1355b6890aa528ee1f6e1b22be766307e3f3c61042d7e4db45a38929`；public `fc7ff9bdd5907e9d2b511e472973af5d2be0af7889055cea86539fa66b5ce002`。
- object pre `a9768d87134b7425237705101f9f93b5d2019ee55f4db6a1b4e9aa98d07b9c7d`；normal `68095163013a45be277ecdc4f95f7921e2cc2794257883fd2ed70e5f5a5c3dd4`；public `6e0d10b942c630bcdad1289d6d8d7f508c2d7e2c51847160536405dc3375abe9`。
- 全量收集 `0492b204d66acbebdb75706f71a269bb9b1c24cdc2b07b57738d606447af5b5c`；源清单 `bf0d1090dd42499baf02073ea570ce01eb013884fefa9ea6b809c75fadc024c6`。

入口为 `scripts/run_fixture_rgb_chain_readout.py` 和 `scripts/run_g1_object_rgb_chain_readout.py`，先加载 `scripts/Enter-CreatorEnvironment.ps1`，使用既有DA3环境。两run全部阶段已占用并冻结，禁止覆盖重跑或原地修改源/config/tests；新研究另建版本。
