# 10月8日：原始支持保真与独立竞争轴

本日两版研究都使用既有完整点云、真实已保存补丁、原RGB体积、估计相机和两档尺度；没有重建DA3、重拟合补丁、换输入域或读GT选轴。全部是已见合成开发数据，不是盲测、实拍或端到端重复。G1结论与当前进度见[清单](../G1-2026-10-11.md)。

## 1. 原始点支持版 support

旧体素均值先把同格内不同结构混合，再以体素数量选主轴。新版保留unique原坐标，对每个轴假设取每格最近的真实支持点做TLS，在同一完整点池重算支持；以实测轴向区间并集长度排名，不按点数排名。最终用实际3D拟合轴，不用RGB提示轴投影修正几何；几何占据与原RGB支持同时满足才输出，原断口不闭合。

固定参数：垂距0.5v、至少6格、物理最短4v、轴向支持半径v、输出采样0.5v。候选包含全局TLS、坐标极点对、512次seed0体素平衡随机点对，以及最多512个固定lex位置anchor的4近邻局部方向；局部点对≥0.5v只作提议，最终物理门不下降。候选搜索不是穷举。

独立复审发现并在freeze前修复两项：61点长1m杆在8000随机杂点中被随机提议漏掉，加入局部提议后契约通过；候选逐点数组缓存增长失控，改为固定大小模型/标量、代表索引SHA去重、只重算赢家支撑。设2e8点-假设检查预算，超限明确unmeasurable，不静默抽样。固定轴支持的集合保真不等于任意加点后赢家不变，真正第二结构仍应拒绝。

### 完整实测

| run | 源 / pre | 完整评价 | 独立reader重复 | normal时间 |
| --- | --- | --- | --- | --- |
| fixture-support-axis-readout-v1-20261008 | 245 / 9814 | 60行，native60不变 | 30/30 | 152.485s |
| g1-object-support-readout-v1-20261008 | 348 / 1266 | 80共同+10native | 40/40 | 160.114s |

两run prepare/infer/evaluate/post全部成功，运行/评分错误0；这不是算法通过。fixture34/60空，所有base全空，粗档除r03 TLS外全部空。细档baseline/cylinder三案R=P100%、段数1/2/1、边界最大8.432mm。r01 TLS/RANSAC细档分别R43.4/0、P43.207/0；r02 TLS/RANSAC细档R0/50.061，r03 TLS/RANSAC细档R98.3/98、P100。旧案例全部guarded-gap误补0，但空输出和偏轴错误仍保留。

新对象support40行32空：chair只fine baseline/cylinder各1段，R98.9677/P100、长度少50.842mm、边界41.123mm；aframe只baseline两尺度各1段，R100/P97.8824、长度多46.836/46.938mm、边界59.305/59.202mm。40条global控制与10月7日正常图除计时外逐字段相同，10条native整行相同。Aframe原生TLS/RANSAC的整条误补1.790420/1.079930m仍单列，不因共同读出空而隐藏；无新增几何的base-native空不是点云质量结论。

support reader成本：旧60次合计40.034s、单次最大1.070s；新40次85.072s、最大5.662s；同run新global40次0.741s。共享CPU条件下运行，不将不同日总耗时差宣称速度收益。原DA3+patch的280.711s未重跑。

## 2. 第二版 independent 的依据与变化

support将共享同一批真实点的不同TLS估计误称为两根结构。正常输入诊断：chair fine TLS竞争对独立覆盖仅9.41%/9.53%，coarse baseline/cylinder仅19.62%/27.24%；chair base两档仍有81.42/62.06和58.09/76.54，Aframe拒绝对均100%/100%。只是首个存档pair，不以此预记整场正式结果。

新版本保留完整池覆盖≥80%及空间分离>v的门，再要求双方分别有≥50%自身全覆盖来自另一轴管外的独立真实支持。两轴对称计算，不删除第一轴点再拟合第二轴，6格/4v门不降。必须遍历所有合格候选，不能去掉首个假竞争者便认定成功。独立支持计算额外成本计入同一个工作预算。这确实改变竞争资格，不能称拒绝语义与旧版完全一样。

机制验证：同一噪声轴的两种微偏估计不被误判第二结构；.03m平行杆及斜率.0201/.021/.022/.025/.04的1m浅交叉，在v=.01下仍拒绝；支持保真、局部稀疏探索、缺口、点/等价曲线、输入/资源/失败契约继续通过。

### independent完整实测及结论

| run | 源 / pre | 完整评价 | 重复 / native | normal时间 |
| --- | --- | --- | --- | --- |
| fixture-independent-axis-readout-v1-20261008 | 252 / 9828 | 60行 | 30/30重复、60/60 native不变 | 158.017s |
| g1-object-independent-readout-v1-20261008 | 355 / 1280 | 80共同+10native | 40/40重复、40global及10native精确同10/7 | 160.772s |

四阶段全部通过，方法/评分错误0；对support并非全面变好，而是26条原空结果恢复且其他物理保持相同（包含两个repeat）。

旧60行：38物理完全不变、22从空恢复、无新增空，空34→12。baseline全部12条R=P100%、正确段数，边界最大18.205mm；cylinder10条R=P100%，但r02粗两个repeat仍空。r01/r02的base全部空。r03 base细/粗R12.4%/63.9%、P100%，仅2/5碎段、长度短缺1848.374/944.021mm，不能称完整杆恢复，仍弱于此前global读取的R47%/68.9%。全部旧例guarded-gap误补仍0。

新对象：仅chair细档TLS/RANSAC两配置（四重复行）从空恢复，均1段R98.8387/P100，边界41.873/41.866mm；长度分别少50.837/58.857mm。其余18配置physical逐字段相同，新reader空32→28/40。chair基础与全部粗档仍空；aframe仍只有baseline双尺度R100/P97.8824、边界59.305/59.202mm。后续其他独立竞争者仍会拒绝chair粗档，不能只处理第一个共享假竞争者就报告成功。所有global与native控制不变，错误新增几何完整保留。

independent reader旧60次39.008s、最大1.269s；新40次84.812s、最大5.081s，global40次0.749s。原DA3及补丁成本另列，不混同本次reader-only成本。

**结论：G1仍未通过，估算维持55%。** 支持保真、候选探索和独立竞争资格确有可复现机制改进，但不能用试验数/测试数代替共同读取公平性；不同对象的可重复净收益及真正端到端重复尚未关闭。下一步停止继续松动竞争票数，先解决原RGB union含其他结构和基础噪声支持的任务归属：可研究原始guide/估计相机仅用于固定公共方向或分组，输出仍须来自实际3D支持、所有臂同域；不得使用GT/候选身份挑轴或将点直接投回guide。之后才固定方法做DA3+补丁端到端重复，不把这些reader重放冒充。

## 3. 实际验证与协作

support阶段47项新增/复用双NumPy通过；完整DA3环境unittest1624/1624通过（297.979s，0失败/错误/跳过），622份源码/config前后SHA不变。core pytest在既有reconstruction环境16 tests+15 subtests通过；DA3缺pytest的首次命令失败原样保存，不安装库。Ruff、358源/三CLI、三Python包离线隔离sdist/wheel通过。

第二版另49项双NumPy通过，不拼成一次1673全量。最终Ruff、362源/三CLI、evaluation离线隔离wheel/sdist通过，10份第二版源/config/tests前后SHA一致。验证收据在.runtime/validation/numpy126-full-20261008-001、packages-20261008-001、independent-final-20261008-001。

DeepSeek交付6测试与本人work-log.json，总览现16任务。Codex独立运行测试，修正两处断口端点精确假设为既定1v占据容差，原交付保留。第二版6项复用文件由Codex适配，不冒称DeepSeek新任务。主代理负责两个核心实现/集成；子助手负责四组runner、正常诊断、完整四阶段运行、独立反例复审和回归。

## 4. 完整结果索引

- [support旧对象完整结果](results/2026-10-08-fixture-support-axis-readout.json) / [审计](results/2026-10-08-fixture-support-axis-readout-audit.json)：public SHA `8d6c8eef725b23bf562c0d7433223ed192cad5a207d7dabb26e2bb64cb37ec42`。
- [support新对象完整结果](results/2026-10-08-g1-object-support-readout.json) / [审计](results/2026-10-08-g1-object-support-readout-audit.json)：public SHA `fd25f7eeb6dedfe6d436f0d635733ae3c3a6d7022115908ea7366dd33f9d69ae`。
- [independent旧对象完整结果](results/2026-10-08-fixture-independent-axis-readout.json) / [审计](results/2026-10-08-fixture-independent-axis-readout-audit.json)：public SHA `9b91373ab12eec011181dbeee082a23f773546ea8b32503c75efc026886c4e47`。
- [independent新对象完整结果](results/2026-10-08-g1-object-independent-readout.json) / [审计](results/2026-10-08-g1-object-independent-readout-audit.json)：public SHA `b84553222f9005363215a0340b953b7edfb56aa640332886f9421a5ffd9396bc`。

每个runner均使用prepare→infer→evaluate→post，正常完整清单与SHA先于GT。已freeze的源、配置、正常产物、评分和审计不可原地修改；新变更用新版本/run。所有原拒绝、未决、错误、真缺口、端点及原生误补保留，不用测试通过或局部R/P高分代替共同读取资格。
