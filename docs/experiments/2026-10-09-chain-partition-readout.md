# 10月9日晚间：固定前景支持内的分链实际几何读取

状态：两run四阶段、300行/150重复、独立正常屏障与2029全量验证完成。**G1:55%，未通过，生产主流程不切。** 旧结果、源、输入均保留，不替换此前失败。

## 固定决定

新文件 `chain_partition_readout.py`、`rgb_chain_partition_axis_readout.py`、`chain_segment_union.py`。两旧控制为 stroke-only 与 required-view；新臂与 required-view 使用逐坐标完全相同的输入正支持域，不新增选择、不缩评分域。固定30个助手原图标记沿用，额外标注成本为0；原标注仍不是用户点击或盲标。相机、原图候选、原始票数、五臂和两尺度不变。

1. 每条已有赋值单独要求所有标注视图及至少三个视图正支持；相同实际坐标点集只拟合一次，保留全部赋值来源。
2. 每组仍用实际3D坐标、固定体素平衡TLS与原门限；输出采样由本组赋值支持，不用其他组补洞。坏组不否决好组，预算不完整则显式unmeasurable，不输出部分结果冒充完整。
3. 重叠集合会产生近似重复假设。按实际受支持有限段总长排序，只有共享较小点集至少50%、整段到已保留输出的连续距离上界≤1v才认定**分辨率alias**。保留者坐标不移动；只允许已保留组抑制，不传递。所有拟合和被抑制假设仍保存在审计。
4. 连续上界由0.5v采样最大距离加0.25v得到，来自距离函数1-Lipschitz性；未能证明覆盖就保留。它不是几何严格等价、真实身份或排除全部潜在轴的证明。
5. 最终只合并机器精度共线、重叠或接触的区间，端点选自原输出；不平均不同轴，不跨正轴向缺口，不按GT/native/方法标签选择。

这仍是局部条件正支持读取器，不是全场景重建或身份识别。不同保留组仍可能部分重复；保留的全部几何进入同一完整评分，不能把它们声称为唯一物体数。

## 冻结前证据与反例

无GT逐链正常诊断：r01/r02/r03/chair/Aframe的14/8/1/3/68 assignments只对应3/2/1/1/17个实际点集。Aframe较小集合共享率中位数0.942，说明直接输出17根杆不成立。58项源/输入SHA不变，诊断及原始输出留在 `.runtime/diagnostics/per-chain-normal-20261009-001/`。

独立反例：v=1时，旧段[0,10]与[12.2,22.2]，新段[8.85,13.35]。0.5v离散样本最大距离0.85v，却在缺口中心达到1.10v；仅采样判定会吞2.2v缺口。冻结前引入Lipschitz余量，反例加入测试，失败草案不作为正式算法运行。

主机制23项双NumPy通过，覆盖实际双轴、坏链隔离、不吸附、真实/RGB缺口、重复幂等、base/curve同坐标公平、联合平移、预算和alias反例。DeepSeek6项合并契约双环境由Codex实跑通过，其本人日志明确未运行测试。完整回归及正式源收据随后记录，不能将分批相加冒充全量。

## 复现入口与隔离

实际仓库 `D:/Creator-newage`，先载入 `scripts/Enter-CreatorEnvironment.ps1`，使用既有 `backends/da3/.venv/Scripts/python.exe -B`。

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_fixture_chain_partition_readout.py prepare
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_object_chain_partition_readout.py prepare
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_fixture_chain_partition_readout.py infer
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_object_chain_partition_readout.py infer
# 两套normal完整并经独立无GT屏障后，才允许运行evaluate和post。
```

首次执行入口；已存在产物会拒绝覆盖。后续新算法必须新文件、新run。fixture 180、object 120正常/评价行，合计300含150重复对；它们不是300独立物体或重新推理DA3的端到端重复。两旧控制共200行必须精确重放上一轮，新臂100行，不选最佳尺度/案例。Aframe原生TLS/RANSAC错误、空补丁、全部native和整个GT域继续保留。

## 正式实测

两套正常运行完整成功：fixture321源/10952收据、180行，normal记录410.303s、外层infer686.323s；object424源/2482收据、120行，normal记录332.300s、外层infer470.413s。prepare外层分别433.293s和133.091s。300行均complete，0运行错误；新partition的100次adapter调用合计17.288739s，object单次最大0.970856s。以上均非冷启动DA3或端到端重建时间。

正常独立屏障271.345s通过：300行、150重复对、200条旧stroke/required控制、100个required/partition计数配对、300条原生几何不变。独立布尔逻辑重算50个非重复partition图的点集，183个alias有限段连续证书、保留组非传递、实际端点和输出正支持全部通过。13218份去重收据前后SHA不变；无GT或评分读取。两正式infer均退出0后，主核实屏障并放行评分。

屏障 `.runtime/validation/partition-normal-review-20261009-001/outcome.json` SHA `02b49d1cd347d660249664cdaee0ac4884ee178dfab9008d2a803870d4fb3adf`；审计脚本SHA `9038a09e21528ff6ab8803dd0502ed2c2dd2d62cd0f9e70bb159c178739a9cba`。首次审计把fixture三字段环境与object完整环境混比而失败；修正未落盘时中止了第二次旧脚本尝试，审批超时后确认修正实际落盘，第三次完整成功。`attempt-1.log`、`attempt-3.log`和`execution-history.json`保留经过；未修改任何正式数据来迁就审计。

两run evaluate/post均退出0、完整精确评分重放通过，300行0评分错误、150/150物理重复相同。fixture180条原生几何逐行不变；object完整10个native对照保留。以下只统计repeat0的50个case/尺度/arm配置，不当独立对象数。

### 覆盖提高，但重复和精确率退化阻止过关

相对同轮required控制，fixture30配置R上升3/下降0/相同27，P上升0/下降2/相同28；object20配置R上升10/下降0/相同10，P上升2/下降4/相同14。合计R↑13/↓0/=37，P↑2/↓6/=42，比较容差1e-12。不得只报“无覆盖下降”：新增重叠段增加长度和边界错误，部分精确率明显下降。

| base案例 | R细 / 粗（%） | P细 / 粗（%） | 段数细 / 粗 |
| --- | ---: | ---: | ---: |
| r01 | 55.600 / 75.400 | 100 / 100 | 9 / 4 |
| r02 | 68.485 / 96.000 | 100 / 100 | 5 / 5 |
| r03 | 34.200 / 52.900 | 100 / 100 | 3 / 4 |
| chair01 | 72.000 / 76.903 | 100 / 100 | 5 / 3 |
| aframe01 | 14.873 / 27.570 | 76.344 / 100 | 4 / 2 |

- **Aframe baseline**：R由44.256/43.652升至89.480/74.849，P由77.679/78.182升至95.300/87.375。但新输出13/4段，长3.449043/2.094728m，相对GT1.653028m多1.796016/0.441700m；边界数不匹配，不能报完整正确杆。实际保留4/2组，仍有部分重叠竞争，整组alias策略不能解决局部重复。
- **Aframe base**：R由7.739/12.455升至14.873/27.570，细尺度P从100降到76.344，预测到GT的p95达89.648mm。TLS/RANSAC/cylinder共同输出与base相同，不能据公共裁剪隐藏它们的native问题。
- **r01细尺度退化**：base R仅55.5→55.6，却从5段变9段；TLS R99.6→99.7但P100→87.334，总长3.912542m；RANSAC R77.8→77.9但P100→82.650，总长2.922609m。baseline和cylinder虽R=P100，却由1段变4段、总长3.380622m（GT2m），不能称成功。对应粗尺度没有这项新增重复，不能只选粗尺度报告。
- **r02/r03/chair**：全部新评分与required控制相同（浮点末位除外），没有恢复其旧raw/stroke已有覆盖。r02细TLS仍R82.667/P82.843；r03朴素TLS/RANSAC仍R98.3/98高于base；chair四候选仍R99.226–99.355、P100，边界35.825–36.134mm。新算法不能消除单点集下的支持缺口。
- **原生轨道**完整保留：Aframe baseline R100/P97.880、1段、边界59.356mm；TLS/RANSAC R=P0、错误长度1790.420/1079.930mm；cylinder拒绝空补丁。两新对象没有真gap，gap0不是新对象缺口安全证据；解析反例单独报告。

完整逐行数据：[fixture结果](results/2026-10-09-fixture-chain-partition-readout.json)、[fixture审计](results/2026-10-09-fixture-chain-partition-readout-audit.json)、[object结果](results/2026-10-09-g1-object-chain-partition-readout.json)、[object审计](results/2026-10-09-g1-object-chain-partition-readout-audit.json)。所有失败、两尺度和两遍都保留。

公开结果SHA：fixture `4f6c1142df813d153bd661581f4fa426c1acba63d5a59cd0b98fd80bc3c14ad3`；object `40000d628137315bce0bf0626af669637881cefbb18c9ea54447a68588e0132c`。post audit SHA：fixture `0fb492cab22a03b26c761d34ca8346233784a725b0ce9281698c22426cc3b121`；object `7511e684c1e14c7f6600350e371adbcee0dffe588d2e21441fc183a363c5ed98`。

eval/post外层实耗：fixture93.008/107.080s、object90.975/91.172s。两套并行，各自评价成功才post；真实stdout/stderr、外层阶段时间、正常计算成本和四份公开产物SHA在 `.runtime/validation/chain-partition-runners-20261009-001/two-run-delivery.json`及同目录日志。

发布后独立只读复核通过：300行实际几何/metadata对normal、150重复、200旧控制（除明确耗时外完整图及评分）、300 native、100新图的组集合/有限并集及366个alias实际几何证书全同；714源SHA及11新源＋4输入的两run共30个freeze关系全部吻合。只读517个直接文件、约8.01s，无递归producer、未读取GT文件。审计 `.runtime/validation/chain-partition-result-crosscheck-20261009-001/outcome.json` SHA `a93e3bfb06f65b018b95712b0991829c732637b1b7582960e2d13f7ecafaa0a4`；独立统计与上表完全一致。

## 完整实现验证

一次全量 **2029/2029**，0失败/错误/跳过/预期失败/意外成功，loader errors=0；不是分批测试相加。unittest runner 700.787s，含收集709.962s，监测包装714.992s。core 16 tests＋15 subtests、Ruff、389 Python源/三个CLI通过。与正式实验共享CPU，仅记录执行成本，不作为算法性能基准。

714项源码/config/build-input及原四份标注输入，前后集合与SHA完全相同，执行中无变化。源清单SHA `b2805e0fdcb335b49d1f0d4de9e7d9aed9c627899192679f0ed3c081720fcb1f`；全量outcome SHA `5da21ebb9b79b1314c077c873763fbf5b8834c0e4e115ef56d35f87c1c33856e`。真实stdout/stderr与收集清单位于 `.runtime/validation/numpy126-full-20261009-002/`。

三包离线构建6产物，180个wheel Python模块（含全部3个新增模块）与冻结源码逐字节相同。构建outcome SHA `e47af4fe65c26f8aef294bbc20e00c57dfd166ec9ae76b8035c48f2c3800b00d`，wheel核验SHA `2295c225ffc09f8ef22ed406190aacf6f7e5787feaa31988a21b8aa9456e2e43`，目录 `.runtime/validation/packages-20261009-002/`。保留001不覆盖。

## 执行成本与下一轮提速点

fixture实际三个reader各60次调用：新partition合计2.772905s、required合计1.726684s、stroke合计1.745833s；不含完整base准备、图像context构建、进程启动和来源核验，不能冒充端到端时间。当前infer外层686.323s，记录内normal elapsed410.303s，二者也不是上述纯reader时间。

助手只读定位：infer对新完整pre集合共扫描6次（外层前后及两worker各前后）；`parent_normal`经两个worker、新normal核验、stroke旧control和rgb-chain祖父control共触发5次naive核验，还递归扫描更旧metric闭包。没有分段profile，不能给每个调用虚构耗时占比。

本轮所有冻结代码和检查照常完成。独立屏障直接对两份完整pre闭包去重核SHA，并重算正常支持/alias证书，不再调用递归producer验证函数；producer原有最终验证仍须退出0。下一版可新建typed direct-parent-index：prepare完整祖先核验一次并冻结直接消费索引，worker完整SHA核验自身实际源/输入，末尾独立完整祖先＋源＋normal＋控制语义屏障后才解锁GT。不得mtime代替SHA、不得省略失败/实际消费文件/全套控制，不能改本轮或旧run。这只是下轮执行器方案，尚未实现，不计进度。

## 决定与下一核心任务

分链确实缓解Aframe一次union TLS的偏移，但“保留全部非alias假设”仍会重复表示同一物理长度，也没有恢复r03/chair的支持缺失。**不靠放宽alias阈值或选择最好尺度签G1**。下一主线分开解决：基于正常观测与实际点的归属/局部重复，及前景必选支持过严丢失真实点；未知不当反证，不能把图像轴投回三维伪造修复。新方案仍须新文件、新run、共享五臂和完整域，真正端到端重复待共同读取资格成立后执行。

G1估算仍 **6+8+15+10+8+8=55%**；实现、复现闭环完成不是共同读取已合格。DeepSeek任务 `chain-segment-union-contracts-001` 的原始request/receipt、本人work-log、Codex实际验收分开留存，固定 `D:/deepseek/collaboration/WORK_LOG.md` / `work-log.json` 现20项任务；提交后关联真实commit。
