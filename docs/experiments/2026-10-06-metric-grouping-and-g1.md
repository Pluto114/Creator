# 10月6日：训练真实距离修复与G1收敛

## 结论

真实距离分组修复0160半壳漏读，有限结构开发门102→103/108；其余183条件物理结果完全相同，28负例/48未决仍空。不是G1通过：主线转向完整base公平读取、不同对象和朴素对照，不再把解析108/108当G1前置。

本周目标见[10月11日验收清单](../G1-2026-10-11.md)，动作见[开发日志](../journal/2026-10-06.md)。基准`e26df4126df6dc1b18a4a6854668e1ed76f2a71b`的Windows/Linux CI已成功；本轮提交CI需另查。

## 核心实现

旧section分组按横向网格整数坐标连接。0160两个断开的训练组实际最近点距只有1.1721/1.1803voxel，低于1.5门，却因取整拆组，导致每组实际薄片角覆盖不足。Codex实现`section_metric_grouping`：保留原边，补查相邻cell内实际训练点距离，仅原1.5voxel半径内允许连接；KD-tree只由训练点建立，heldout不提议组、不架桥，仅附着唯一既定组。

`section_metric_evidence`复用原context算法，仅替换分组。`common_readout_metric_base`保留既有6/12提议邻居，`metric_fold`保持固定salt；`common_readout_metric`沿用两折有限端点一致补检。角门、训练窗口、验证一次原则、采样、最短物理段、缺口及歧义门均未放宽。不是从多组参数中按得分择优。

旧源和所有旧失败run保留。独立审查未发现阻断性问题；80组随机几何/6种合法半径与暴力连通及heldout归属一致。该检查是机制验证，不是对象性能证据。

## 正式解析结果

运行`mixed-readout-metric-v1-20261006`，186源、1667份pre收据。184个已评分开发条件，736行=368新metric执行+368旧consensus精确引用。三进程正常推理184.373秒（与其他测试并发，不是受控速度对比）；无方法/评分错误，368表示对全部一致，post精确复算通过。

| 开发组 | 旧consensus有限门 | 新metric有限门 |
| --- | --- | --- |
| mixed | 24/24 | 24/24 |
| section | 20/20 | 20/20 |
| sampling | 28/28 | 28/28 |
| support | 12/16 | 12/16 |
| local | 18/20 | 19/20 |
| 合计 | 102/108 | 103/108 |

0160恢复两段，R=P100%，总长1.5200013m，长度误差约0.0013mm，边界最大误差0.0192mm，保护缺口误填0。只有0160的physical字段改变；其余183条件全部不变。96条带保护缺口评分行误填均0。R/P门104/108仍不能替代有限结构门。

剩余5项：0132/0133渐变半径各30碎段、0134/0135空、0168缺支路。全部保留为开发失败；不再新增分折或一般骨架能力来追满分。

[全部解析结果](results/2026-10-06-mixed-readout-metric.json) · [后审计](results/2026-10-06-mixed-readout-metric-audit.json)

## 完整fixture回放

`fixture-metric-readout-v1-20261006`四阶段完成，9151份pre收据最终SHA核验通过，18行全部complete且无方法/评分错误，normal34.012秒，post精确复算通过。12候选R=P100%、段数正确，最大边界8.407mm、最大绝对长度误差22.765mm、保护缺口误填0；6纯base仍空。18/18物理结果与consensus v2完全一致，原完整base/candidate、相机、mask、投票和采样计数不变。仍是旧局部域，不是不同对象或G1证明。

[全部fixture结果](results/2026-10-06-fixture-metric-readout.json) · [后审计](results/2026-10-06-fixture-metric-readout-audit.json)

## G1纯base支持证据

新`diagnose_g1_base_support.py`独占生成[normal-only诊断](results/2026-10-06-g1-base-support.json)，7989份输入/源收据前后核验，59.853秒；GT未读，不运行新readout、不改深度/相机/mask/policy。逐view分块投票与原joint函数精确一致；三案全场各952560点，来源ID是frame,row,column，各源190512点，没有删除原点。

| 对象 | view00…04来源的三票支持点 | 源内最近邻中位数mm（同顺序） |
| --- | --- | --- |
| r01 | 108,164,149,148,0 | 11.206,10.307,9.057,17.658,无 |
| r02 | 96,156,141,145,17 | 11.219,10.986,12.075,12.011,9.445 |
| r03 | 18,48,36,18,0 | 24.599,15.536,18.979,15.124,无 |

六个纯base正常结果的三个折都没有measured ridge runs，主要是below_minimum_physical_length。r01/r03来源view04没有三票支持点，r03源内NN p95最高115.457mm；但不能从这些数据断言基础无杆或确定读取器漏检。深度几何、估计相机/投影、mask筛选、读取模型/有限支持四类原因仍未决。

这份诊断锁定可复核来源与支持量，不把空base当作候选收益证明。下一步把正常RGB限定域内的基础几何和已有TLS/RANSAC对照接入相同完整评价，先判读取上限；新对象和实拍素材并行准备。

## 分工与验证

Codex负责上述核心修复、集成、冻结和结论；Codex子助手负责两runner机械迁移、批量测试/构建、base诊断、独立代码审查。DeepSeek真实交付6条机制测试及本人work-log.json，Codex修正一个辅助函数调用错误，原交付不动；本人未执行命令，实际验收分列。

- 42核心/继承契约、6条DeepSeek契约、35条runner契约均双NumPy通过。
- 1406项全量通过（265.316秒），后新增9项base诊断契约另行双环境通过，不宣称1415全量已跑。
- 核心pytest16项、Ruff、最终338源/三CLI、evaluation隔离sdist/wheel、Blender包通过。
- DeepSeek固定可见日志：`D:/deepseek/collaboration/WORK_LOG.md`和`work-log.json`，本轮共12任务；[项目协作记录](../collaboration/DEEPSEEK-WORK-LOG.md)。

入口均要求先加载`scripts/Enter-CreatorEnvironment.ps1`。两个正式runner各有prepare/infer/evaluate/post阶段；完成run不得覆盖重跑。新算法另建源/run。
