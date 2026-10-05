# 2026-10-05第二阶段：固定分折有限段一致性补检

基点`eeaf41e`，[该提交CI](https://github.com/Pluto114/Creator/actions/runs/37278911567)已成功。上一阶段[圆周/薄片歧义报告](2026-10-05-ridge-bundle.md)及所有源/运行/失败保持不变。

## 方法决策

正常记录表明短支路受到小样本分折不平衡影响：0168/0169支路残余训练点仅4/3点，旧采样模型未建立；0171已有190mm轴和13mm cadence，但验证占据0.25不足0.3。不能靠降低覆盖门或物理长度解决。

新`common_readout_consensus`先执行并原样保留`bundle_small`的几何，再运行两个固定附加ridge分折（salt索引1、2）。`common_readout_fold`保留旧section所有权和遮除范围；ridge全局/局部方向、拟合与cadence仅由该分折自己的训练点提出，验证点不重拟合。原长度、覆盖、径向、圆周排/薄片歧义门不变。

`ridge_fold_consensus`要求两折都输出、两个有限段的无向端点距离≤0.5voxel才补入。保留第一折实际几何，不平均、不延长、不跨缺口；既有成员上的重叠拟合不重复添加。单折出现不足以补检。新增预算不可测保留为失败。

这是**有固定上游section范围的ridge补检**，不是完整新三折section算法。分折数据互相重叠，不能当独立试验或置信度倍增；也不是把三个分折成绩挑最好。没有salt搜索。冻结前normal-only预览中0169/0171恢复、0170保持，0168两附加折分别4/3段仍未恢复，故保持失败。

## 冻结与执行

- `mixed-readout-consensus-v1-20261005`：184已见开发条件，无新对象/盲测；两读取器×两表示=736行，368新执行+368旧bundle_small逐字节绑定引用。冻结176源、2015份pre收据，先核验父run全部1104行才复制隔离真值。
- 新runner采用3个独立进程按输入执行，每个worker启用GT/评分拒读审计；结果按固定清单落盘，完整normal与所有SHA核验后才能评分。并行wall time和每行elapsed受调度影响，不与缓存0秒或旧串行run做速度比。
- 预先选择consensus接同一旧18项完整base/candidate fixture；沿用原输入、相机、mask和曲线采样。解析前置是正常记录完整性，不是按分数选择方法。

每个阶段一次，不覆盖旧run：

```powershell
Set-Location D:/Creator-newage
. ./scripts/Enter-CreatorEnvironment.ps1
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_consensus.py prepare
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_consensus.py infer
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_consensus.py evaluate
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_mixed_readout_consensus.py post
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_consensus_readout_v2.py prepare
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_consensus_readout_v2.py infer
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_consensus_readout_v2.py evaluate
& ./backends/da3/.venv/Scripts/python.exe -B scripts/run_fixture_consensus_readout_v2.py post
```

## 验证与分工

73项新增/复用契约双NumPy通过，含进程GT拒读、并行与顺序几何一致、每折同体素不可泄露、提议API只收到本折训练点。核心16项/15子检查、Ruff、328源/三CLI、evaluation包与Blender包通过。

DeepSeek交付6项纯有限段一致性测试及本人动作日志，Codex审查无需改断言；Codex负责算法、其余测试、runner和实际运行。原始交付、验收SHA与实际测试分开保存，见[协作日志](../collaboration/DEEPSEEK-WORK-LOG.md)。

全量1308项通过（553.066秒）。静态检查发现7份新增文件末尾多一个空行；因已冻结而原样保留，没有为格式检查修改冻结字节。除此之外暂存差异检查通过。

## 正式解析结果

[完整结果](results/2026-10-05-mixed-readout-consensus.json)、[后审计](results/2026-10-05-mixed-readout-consensus-audit.json)。736行全部complete、无方法/评分错误，368表示对全部一致，2015份pre收据不变，post精确复算通过。3进程正常wall time403.839秒；不是与旧串行run的受控性能比较。

| native开发组 | 旧bundle_small有限结构门 | consensus有限结构门 |
| --- | --- | --- |
| 旧混合 | 24/24 | 24/24 |
| 旧截面 | 20/20 | 20/20 |
| 旧采样 | 28/28 | 28/28 |
| 旧支持 | 12/16 | 12/16 |
| 旧局部 | 16/20 | 18/20 |
| 合计 | 100/108 | 102/108 |

有限结构门仍要求R/P均≥90%、精确段数、端点一一对应且最大误差≤25mm、保护缺口误填0；只适用于当前分段约定。

- 仅0169和0171物理结果改变：3主杆补为4段，缺失短支路恢复并通过有限门。其余182条件物理结果与旧版相同，旧0170及4个双裸杆保持。
- 28负例、48未决全部无输出，前一阶段消除的9项误输出没有回来。96条带保护缺口评分行误填0。R/P通过数仍103/108，进一步说明R/P不能代替支路完整性。
- 仍有6项正例失败：0132/0133渐变半径各30碎段，0134/0135/0160空，0168只3主杆。空正例不计拒绝成功。
- 0168附加fold1恢复190mm短段（训练覆盖0.6875、验证0.375），fold2只3主杆，其他拟合仅3–4训练层并产生47次短run拒绝。因此不满足双折补检；没有继续换salt、补点或降覆盖门。

全部184条件仍是已见开发回归；不切主流程或升级G1。下一步应处理真正有限训练支持/横截面分组，避免继续搜索分折来追分。

## fixture路径修复与失败保留

首次`fixture-consensus-readout-v1-20261005`冻结205源/7959收据后在infer失败：runner迁移把原始`bundle/base`与`bundle/*-enabled/withdrawn.json`路径误替换为`consensus/...`。原始存储没有变更，0条正常记录、未调用fixture读取器、未评分。失败源、pre和空records目录原样保留，[机器可读失败记录](results/2026-10-05-fixture-consensus-readout-failure.json)单列，不将失败改写为成功。

新`run_fixture_consensus_readout_v2.py`/config v2/run v2只修复存储路径，并冻结失败v1的pre及evaluation-freeze哈希。读取器、输入、相机、mask、采样、评分均未改变；增加专用`bundle_paths`契约，15项修复/继承测试双NumPy通过。最终Ruff、329源/三CLI通过。上述1308全量测试在路径修复前完成；修复后另跑15项，不冒称又跑过1323项全量。

## fixture v2正式结果

[完整结果](results/2026-10-05-fixture-consensus-readout-v2.json)、[后审计](results/2026-10-05-fixture-consensus-readout-v2-audit.json)。208源/7967份pre收据，18行全部complete、无方法/评分错误，81.592秒，post精确复算通过。12候选R=P100%、正确段数、端点最大8.407mm、绝对长度误差最大22.765mm、保护缺口误填0；6纯base仍为空。

18/18物理结果与上一阶段bundle_small相同，完整base/candidate、相机、旧mask、采样及投票计数均不变。仍只是旧局部共同读取域，不代表整场景或独立对象成功。

最终7967份pre收据再次逐条SHA通过，16份新增冻结文件与Git暂存字节一致。解析v1及fixture v2四阶段均已结束，上述命令仅说明入口，不可覆盖重跑。失败fixture v1没有被修改或冒充完成。
