# 2026-10-05：曲面/薄片歧义与小邻域候选对照

## 决策和范围

在10月4日训练cadence版本上增加“训练提出、验证确认”的圆周排及局部平面歧义证据；同时固定两套guard、只改变提议邻域，比较12邻居与6/12邻居。旧冻结方法、失败记录、门限及run不改。

- `ridge_bundle_evidence.py`：沿用训练cadence提取持续排；至少4个训练排落在同一圆周、角覆盖至少150°，再用heldout确认原训练排。三根非共线杆本身不构成该证据。
- `ridge_plane_evidence.py`：训练点拟合候选轴附近的薄平面，要求沿轴重复出现横向宽度；固定模型和站点后独立验证，补持续排计数遗漏的薄片。两根杆没有三档横向宽度。
- `common_readout_bundle.py`：12邻居比较臂；`common_readout_bundle_small.py`：6/12邻居臂。两者只有`local_proposal_small_neighbors`默认值不同，共用上述guard与原section/context和ridge cadence。
- 仍然拒绝预算不可测、证据不足，不补点、不读取GT、不改物理长度/角门。上述是几何歧义，不能当语义类别识别。

新增circle和plane作为整体干预，与旧ridge_sampling比较；两个新臂隔离小邻域提议的影响。没有circle/plane单独消融，不能单独归因二者的全局收益。

## 冻结协议

解析run：`mixed-readout-bundle-v1-20261005`。旧184条件全部已见、已评分，五组44/36/40/32/32，**没有新盲测或独立对象**。三个读取器×两种表示，共1104行：736新执行，368旧ridge_sampling正常result按输入/策略/SHA精确引用。源167份，pre收据1629份。正常进程拒读GT/评分；完整正常清单及所有SHA核验后才评分。

fixture run预定：`fixture-bundle-readout-v1-20261005`。在解析评分前已选定`bundle_small`，读取原3 fixture×2尺度×3完整base/candidate，18行，沿用旧mask、相机、采样和支持域，不重跑DA3或重建补丁。解析完整性前置不是按分数挑选方案。这个旧局部域测试不代表完整场景质量。

可执行入口（每阶段只执行一次，禁止覆盖）：

```powershell
Set-Location D:/Creator-newage
. ./scripts/Enter-CreatorEnvironment.ps1
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_bundle.py prepare
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_bundle.py infer
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_bundle.py evaluate
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_bundle.py post
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_bundle_readout.py prepare
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_bundle_readout.py infer
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_bundle_readout.py evaluate
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_bundle_readout.py post
```

## 验证与协作

135项新增/复用契约已在NumPy1.26与NumPy2通过；包含DeepSeek独立交付的6条机制测试。Codex审查修正两条反向断言后集成，其原始交付未改。其余方法、runner及测试由Codex实现并执行。

关键动作见[协作工作日志](../collaboration/DEEPSEEK-WORK-LOG.md)；DeepSeek固定总览`D:/deepseek/collaboration/WORK_LOG.md`和`work-log.json`已自动更新。单测通过不等于算法合格，所有正例失败和反例误输出都纳入正式比较。

全量1235项（476.970秒）、核心16项/15子检查、Ruff、322源/三CLI、evaluation sdist/wheel及Blender包通过。新增14份解析冻结源与Git暂存内容逐字节一致。旧继承源仍与原本地冻结一致，不为了Git换行规范改历史文件。

## 正式解析结果

[完整结果](results/2026-10-05-mixed-readout-bundle.json)、[后审计](results/2026-10-05-mixed-readout-bundle-audit.json)。1104行全部complete、无方法/评分错误，552表示对完全一致；正常耗时888.604秒，1629份pre收据不变，评分精确复算通过。

下表是native正例有限结构门：R/P均≥90%、正确段数、端点一一对应且最大误差≤25mm、保护缺口误填0。只适用于当前分段约定，不是通用拓扑/G1指标。

| 开发组 | 旧ridge_sampling | 新12邻居bundle | 新6/12邻居bundle_small |
| --- | --- | --- | --- |
| 旧混合 | 24/24 | 24/24 | 24/24 |
| 旧截面 | 20/20 | 20/20 | 20/20 |
| 旧采样 | 28/28 | 28/28 | 28/28 |
| 旧支持 | 12/16 | 12/16 | 12/16 |
| 旧局部 | 16/20 | 15/20 | 16/20 |
| 合计 | 100/108 | 99/108 | 100/108 |

- **bundle_small消除9项旧误输出，全部108正例物理结果与旧版一致**。28负例和48未决均无输出；其中0032..0035稀疏圆柱、0036/0037半圆柱原各4段，0148/0149/0150噪声薄片原各1段，新版均为0。这里是构造条件下拒绝，不是语义分类准确率。
- 两个新臂都消除这9项误输出。12邻居臂还使0170从4段回退为3段；6/12臂保留4段。这次邻域对照支持保留小邻域提议，不采用12邻居作为后续主候选。
- 三臂R/P90均103/108，明显高于有限结构通过数，说明仅看R/P会掩盖缺短支路。两种表示结果一致。每臂48条带保护缺口评分行，合计144条，误填长度全部0。
- `bundle_small`仍有8项正例失败：0132/0133渐变半径各30碎段，0134/0135及0160真缺口半壳为空；0168/0169/0171非共面结构各缺短支路（只有3主杆）。4个双裸杆仍完整，0170仍完整。没有把空正例算拒绝成功。

因此后续研究基点为`bundle_small`，**不切生产主流程，G1未过**；184条件全是开发回归，未获得独立对象/全场景资格。

## 后续定位

仅从旧正常记录核对：0168/0169各只有3主杆通过，其他候选训练层只有3–4层，cadence未建立并被拆为短run；0171已有190mm/13mm cadence候选，但独立验证覆盖0.25低于0.3。下一版本应补训练方向/采样支持而非一律降物理长度或验证门。渐变半径与0160横截面分组另立新源/run，不能评分后修本轮冻结源。

## 正式fixture结果

[完整结果](results/2026-10-05-fixture-bundle-readout.json)、[后审计](results/2026-10-05-fixture-bundle-readout-audit.json)。192源/6803份pre收据，18行全部complete、无方法/评分错误，44.943秒，精确post通过。12候选全部R=P100%、段数正确、最大端点误差8.407mm、最大绝对长度误差22.765mm、保护缺口误填0；6纯base仍空，不能作为成功拒绝。

18/18物理结果与10月4日ridge_sampling完全相同；完整输入、相机、mask、曲线采样和投票计数不变。完整base各952560点，局部mask内仅569/555/120点。只说明旧局部共同读取域不退化，不代表完整场景或独立对象合格。

最终6803份pre收据再次逐条SHA核验通过，本轮18份新增冻结文件与Git暂存字节完全一致。两个run的prepare/infer/evaluate/post均已执行完毕，上述命令仅为入口说明，不得在原run覆盖重跑。
