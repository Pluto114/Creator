# 10月12日：实际种子扩展、有限支持竞争与投影区间裁剪

本轮为此前已看分数的合成开发验证，不是盲测、实拍或重新运行 DA3。原代码、原30个助手 RGB 标记、全部旧失败保留。**两套完整流程及独立审计通过；去重有所改善，但共同读取资格仍未过，G1:55%，未通过，不切生产主流程。**

## 固定算法

1. 输入仍为完整原点快照与实际补丁按固定间隔采样后的坐标；两者使用同一 stroke 正支持域，不按方法身份选点。
2. 同链全部标注视图及至少三个视图支持的实际点只作为种子。种子用唯一坐标、固定世界体素均衡 TLS；在原种子轴的固定 **2v** 管内加入同链 stroke 实际点，同时保留全部原种子。只扩展一次，不吸附图像轴、不迭代扩域。
3. 每个候选从实际扩展点拟合，以有限输出能解释的实际占用体素数、实际拟合残差及确定性几何键排序。后续从已保留有限输出 **2v** 邻域外的实际点重拟合，不将高度重叠假设直接叠加成多根物理杆。
4. 后续输出采样增加 h/2 排斥余量（h=0.5v）；依据距离函数的1-Lipschitz性，证成不同保留组之间连续距离>2v。同组真实/RGB缺口不受此跨组证书替代。近杆、交点附近结构可能被分辨率竞争丢弃，不能声称身份唯一或穷尽所有杆。
5. 新 `rgb_segment_cells.py` 沿实际拟合有限段枚举相机正深度、取整行边界和测得水平带边界的全部投影参数根。在支持恒定区间内检查同链票数，只输出连续正支持区间，开放端点向内微调并验证；最终应用原4v最短段规则。原 sampled 图、区间裁剪和 final 图独立保存。
6. 任何搜索/几何/距离/投影事件预算不完整，整次读取显式 unmeasurable，不把部分输出冒充完整成功。局部退化种子不否决其他正常组。

完整实现：`experiments/src/creator_eval/foreground_support.py`、`foreground_consensus_readout.py`、`rgb_foreground_consensus_axis_readout.py`、`rgb_segment_cells.py`。这是固定局部任务的共同读取候选，不是新增前景身份或通用骨架化方案。

## 冻结前正常证据

59项源/输入正常诊断显示1v管会切掉许多原种子，因此采用固定2v并保留种子；并非从本轮真值选半径。细尺度相对required增加同链实际点 r01/r02/r03/chair/Aframe=35/61/20/50/189；粗尺度158/196/30/99/646。成本和尺度敏感性一并保留。

首版正常 smoke 在v/8密采样发现 chair粗2/353与Aframe粗1/174个点落在离散v/2 callback漏掉的未知RGB区间。已在冻结前加入投影事件裁剪，原失败结果不覆盖。最终新 smoke 64项源/输入SHA不变，10配置3818密采样点全部正支持；该抽查与独立事件区间验证分开，不能只靠加密采样声称连续证明。

独立解析/机制契约包括真实双杆、交叉臂、近邻假设、短斜种子不迭代扩张、几何/RGB gap、透视变深、横向带切换、不同链不能拼票、base/curve同坐标公平、重复幂等与预算。DeepSeek只写纯函数8项测试，Codex实际双环境运行；本人日志和验收分开。

## 执行器与可复现入口

实际仓库 `D:/Creator-newage`；先载入 `scripts/Enter-CreatorEnvironment.ps1`，使用已有 `backends/da3/.venv/Scripts/python.exe -B`。

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_fixture_foreground_consensus_readout.py prepare
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_object_foreground_consensus_readout.py prepare
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_fixture_foreground_consensus_readout.py infer
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_object_foreground_consensus_readout.py infer
# 原barrier存在旧guard哈希接口兼容问题，本轮使用下面单独冻结的补充入口。
# 两套完整正常屏障、独立正常语义审核和精确SHA绑定授权后，才可evaluate/post。
```

已存在产物拒绝覆盖；新算法必须新文件/新run。两控制为原stroke与10/9 partition，新臂为foreground consensus；保持五案×五方法臂×两尺度×两全新进程，fixture180行、object120行，共300行/150重复对，200旧控制全部重算。不把这些配置当300独立物体。

新执行器 prepare 一次完整递归祖先核验后冻结 typed direct-parent index。fixture337源/614直接文件/11154祖先收据；object440源/603直接文件/2618祖先收据。worker逐字节核验实际直接输入/源并拒绝未声明仓库输入，末尾完整祖先SHA和正常语义屏障不可省略。mtime不能替代SHA。两套源、完整base、enabled/withdrawn补丁、RGB/相机、上下文、原生几何和全部控制均绑定。

所有阶段真实输出和外层成本在 `.runtime/validation/foreground-consensus-runners-20261012-001/`；各run在 `.runtime/experiments/*-foreground-consensus-readout-v1-20261012/`。运行成本与同时执行的全量测试共享CPU/IO，不作为独立性能基准，也不等于冷启动DA3或端到端重建。

### 独立补充屏障

原object barrier因通用digest被旧generation-config guard拒绝而退出1；fixture在用户中断时退出1073807364。原正常300行及冻结源不改，两个失败/中断的退出码和真实日志SHA完整保留。

补充 `scripts/run_foreground_consensus_barrier_supplement.py` 调用原已批准的 `run_g1_object_rgb.digest`；两套旧truth guard保持开启。该provider SHA在原object pre认证，跨suite manifest绑定两个原pre/inference/index及原失败记录。所有own/direct/ancestor/normal文件前后逐字节SHA核验，原控制/重复的完整语义也核验，不缩成mtime检查，不重跑normal。

补充两源独立冻结于 `.runtime/validation/foreground-consensus-runners-20261012-002/supplement.json`，SHA `a94ff36abdfbb52f2c4f8b88d264f6d350cacc6fdda71aef2c313b93d6825b92`。这是正常来源验证的补充，不把新核验脚本塞回旧pre，也不修改已冻结算法。

```powershell
$manifest = '.runtime/validation/foreground-consensus-runners-20261012-002/supplement.json'
$manifestSha = 'a94ff36abdfbb52f2c4f8b88d264f6d350cacc6fdda71aef2c313b93d6825b92'
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_foreground_consensus_barrier_supplement.py barrier --kind fixture --manifest $manifest --manifest-sha256 $manifestSha
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_foreground_consensus_barrier_supplement.py barrier --kind g1_object --manifest $manifest --manifest-sha256 $manifestSha
```

这些命令是本地冻结运行的核验入口，已有成功产物拒绝覆盖；它们不绕过独立审核和评分授权，也不是无需原始运行文件的独立示例。

### 实现验证

原730源/输入冻结下全量 **2089/2089**（0失败/错误/跳过），core pytest16项+15subtests、Ruff和397源/3CLI通过；3包6产物、184 wheel Python成员匹配冻结源。补充入口加入后，另跑原60+新增12共 **72项**，NumPy1.26.4/2.4.6均通过；全库Ruff通过，原730+补充2共732项SHA不变，既有184包成员重新逐字节核对通过。不是2161个唯一测试或一次含新入口的2089全量。

全量/构建/补充专项 outcome SHA依次为 `85b1fb4874c1144b64bf68d0d3af3af080d0b888139a0bd282346d67e1df4c7f` / `1f2c3028b69f7854a517f5219a602d5eecec2e3d54311335152215cfe979dc8f` / `1ad3ab438e96a79f0f24df3fd79b5a2c15f2b51d417ec2f22869daefa1a4139e`。真实stdout/stderr和前后清单分别在 `numpy126-full-20261012-001`、`packages-20261012-001`、`numpy126-full-20261012-002`。

## 正式全表结果

两run的prepare/infer/补充barrier/evaluate/post全部完成。共300行、150完整图及评分重复对、200旧stroke/partition控制精确相同，0运行记录错误/评分错误，所有reader行均complete。这里的complete只表示算法完成，并非几何正确。25个原生臂完整保留；其中base没有原生curve，native为空不等于完整base点云为空。

评分前独立审核14,336唯一文件收据、300/150/200与100原stroke输入/native配对，50新非重复图；246支持组、60保留组、111段。独立检查71,419离散轴查询、360,961距离运算、56,386投影cells、225,236区间/边界探针及27,816额外v/8点通过，未读GT/score。SHA `76e9ab6cb5f32f460bfa4bcd401e0828f1b941575083745c19767b2cf3bd6c0f`；真实捕获370.136秒。主核验后才生成精确双suite授权。

评分后另一独立脚本未运行生产runner，也未导入主统计脚本，检查全300记录、150完整重复、200旧图/评分、300行原生几何/评分绑定及730+2源/补充manifest/原失败日志/收据绑定，0失败。其直接复核552文件，完整祖先字节闭包由此前14,336项normal审核负责。SHA `16bdf5930e4c6e186520e323dfca411aebfec614cecf6a9c44bec344ae7a212d`，15.045秒。

所有结果而非只列改善项：[fixture完整JSON](results/2026-10-12-fixture-foreground-consensus-readout.json)、[fixture post](results/2026-10-12-fixture-foreground-consensus-readout-audit.json)、[object完整JSON](results/2026-10-12-g1-object-foreground-consensus-readout.json)、[object post](results/2026-10-12-g1-object-foreground-consensus-readout-audit.json)。下表均repeat0；R=覆盖率、P=精确率，百分数，细/粗为voxel fraction 0.0025/0.005。

### 完整base对照

| 案例 | 尺度 | 原partition R | 新R | 新P | 新段数 | 新长度m | 原stroke R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r01 | 细 | 55.600 | 60.100 | 100 | 7 | 1.109428 | 80.200 |
| r01 | 粗 | 75.400 | 88.600 | 100 | 4 | 1.607398 | 88.700 |
| r02 | 细 | 68.485 | 87.030 | 100 | 8 | 1.182608 | 95.636 |
| r02 | 粗 | 96.000 | 100.000 | 100 | 3 | 1.593749 | 100.000 |
| r03 | 细 | 34.200 | 42.700 | 100 | 4 | 0.654370 | 43.500 |
| r03 | 粗 | 52.900 | 53.000 | 100 | 3 | 0.911760 | 53.000 |
| chair01 | 细 | 72.000 | 73.935 | 100 | 5 | 0.944021 | 91.742 |
| chair01 | 粗 | 76.903 | 95.355 | 100 | 2 | 1.394236 | 95.871 |
| aframe01 | 细 | 14.873 | 26.360 | 54.188 | 4 | 0.574769 | 0 |
| aframe01 | 粗 | 27.570 | 26.360 | 62.604 | 2 | 0.642290 | 0 |

所有新base都仍为碎段，边界数不匹配不能写边界误差为0。r03/chair仍不及此前raw已有覆盖（47/68.9、97.419/99.742）；仅与更差的partition比并不足以证明公平。Aframe细P从76.344跌54.188，粗P从100跌62.604：RGB正支持、连续间距证书不代表实际前景身份正确。

### 去重收益与候选退化必须一起看

- r01细baseline/cylinder：原4段/3.380622m→1段/1.988402m，R=P100，边界17.280mm。TLS由2段/3.912542m、P87.334→1段/1.958075m、P100，但R99.7→99.5；RANSAC由5段/2.922609m→3段/2.038499m，R77.9→95.3、P82.650→100，仍不是正确单段。
- r01粗TLS R99.7/P100→95.8/96.527；RANSAC虽R87→91.8，P100→95.570。baseline/cylinder R100→99.7、边界12.189→28.828mm，不能只报细尺度去重。
- r02细TLS R82.667/P82.843→76.970/77.195。其余r02候选R=P100；粗TLS也100，边界18.024mm。r02全10个新配置guarded gap错误长度0，真实guarded gap为0.3m；新两个对象无真实gap，不能拿其gap0证明安全。
- r03候选大致保留：TLS细98.3/粗98.2、RANSAC细98/粗98.2，P100；baseline/cylinder两尺度R=P100、单段。
- chair四候选R99.226–99.484、P100、单段，但仍损伤原生TLS/RANSAC的R100；base缺失仍在。
- **Aframe细baseline**：R89.480→100，但P95.300→87.263；13段/3.449043m→4段/1.890965m，真值1.653028m，仍多0.237937m且边界数不匹配。**粗尺度**R74.849→60.580、P87.375→55.866；4段/2.094728m→5段/1.785828m。短了不等于正确，细尺度覆盖100也不等于验收通过。
- Aframe其他三个候选与base同结果；被拒绝的cylinder仍原生为空，不能把共同reader产生的base段冒充补丁成功。TLS/RANSAC原生仍R=P0、错误长度1.790420/1.079930m、边界462.911/872.758mm。baseline原生R100/P97.880、1段/1.697579m、边界59.356mm：好native仍被共同读取损伤。

50个非重复新配置相对partition：R上升21/下降11/相同18，P上升2/下降13/相同35；相对stroke：R上升10/下降21/相同19，P上升10/下降3/相同37。只有两种新合成装配，旧三个fixture同对象；不能把这些配置当独立物体样本。

## 成本、结论与下一主线

prepare外层fixture/object=571.457/217.198秒；infer外层217.267/426.173秒（内部204.021/411.158秒）；补充barrier431.437/130.884秒；evaluate36.213/16.257秒，post48.957/16.647秒。原失败/中断额外32.145/43.189秒另记，未从成功成本抹去。

本轮100次新reader调用合计113.647秒（fixture60次10.188、object40次103.459，单次最大5.606秒）；同轮旧partition100次23.935秒、stroke5.780秒。共享CPU/IO下只报告实际成本，不当独立速度基准。新算法的投影事件与多候选检查明显增加成本；worker路径减少重复递归不等于本轮总体更快。

结论：局部重复、原种子外支持恢复、连续RGB未知区间漏检三个问题都有实测推进；但Aframe错误支撑可获得更多实际体素票并主导竞争，粗尺度和base精度显著退化。**G1维持55%，六项6+8+15+10+8+8，不升级共同读取或不同对象收益，不切生产。**

下一核心优先级为实际点的局部前景归属与噪声权重、保住base原有支持；先从正常输入诊断Aframe错误高票组为何获胜及尺度翻转，建立新的真实双杆/前后景竞争反例，再冻结新版本。禁止按本轮GT或native挑赢家、直接扩大2v、只调alias、把连续RGB正支持当身份确认。真正DA3＋补丁端到端独立重复仍未完成；未提供独立实拍包，之前已询问，不阻塞算法、不冒充已有。

DeepSeek 8项测试及本人关键动作在固定MD/JSON；主/执行器/审计真实日志均保存。现有CI显式patterns没有执行本轮新增72项，证据为本地双NumPy；下轮冻结前补CI入口，远程CI成功不冒称新增测试已在CI执行。
