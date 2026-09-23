# Creator Newage

**Multi-view thin-structure reconstruction research, with a Blender interface.**

利用同一静态物体的多张照片，研究如何改善基础重建中细杆的遗漏、断裂和错误连接。基础深度与相机估计采用独立的 DA3 后端；自研部分是局部多视图几何恢复及其独立评测。

## 当前进度

**G1：方法可行性开发中，尚未通过。** 新增15张高度变化的严格配对杆图：主方法断杆曲线R/P由82%/84%提高至100%/100%，但缺口边缘仍有2.65%容差覆盖；细杆约69%/70%，尚未合格。全真实相机诊断中的新断杆仍因候选歧义拒绝，下一步检查候选/身份稳定性。已接通6份明确绑定相机和深度的新点云版本、12补丁及精确撤回；深度未修复，共同读取器和独立对象验证仍待完成。旧输入/结果保留，无自动应用，9月27日仍是冲刺检查点。

| 已可用 | 尚未实现 |
| --- | --- |
| 固定模型推理、NPZ结果、点云及GLB演示 | Creator正式DA3任务适配器与完整文件协议 |
| 六条件、30张同步RGB与独立深度/ID/相机/几何GT | 独立物体与真实照片验证 |
| 原生基线、6组裁剪、3组oracle、TLS/RANSAC与RGB交会对照 | 完整候选的共同曲线评分与跨对象验证 |
| 原图双边缘、逐段证据与拒绝原型，6张评测专用新视角 | 跨场景稳健的估计相机恢复、正式补丁及可靠接受/拒绝策略 |
| 中心偏差材质干预、22种亮度反例、12种解析关联及8种新Blender身份布局 | 独立对象保留集与稳定的物理轴验收层 |
| 新增10种场景/50帧、30组提示偏移、条件式有限杆段及源码/输入/GT冻结检查 | 稳定的候选保留、目标身份和跨物体几何验证 |
| 7份真实模型快照、约582万点、来源ID补丁、精确撤回与保存重开 | 正式深度契约迁移及实际抑制算法 |
| 前两版完整基础/候选读取、配对尺度回放、圆柱上限与反例 | 历史读取器未合格，保留失败结果供回归 |
| 第三版读取器、396项配对控制；4份新估计相机预测、18个身份补丁、72项特权相机诊断 | 局部噪声/部分表面读取、跨对象重复收益 |
| 固定镜头下RGB轨迹联合校正原型、两轮18项优化及错配压力对照 | 稳健的假设失配检测与可信自动应用 |
| 20张新参考纹理杆图、16相机/32曲线条件、24条特权诊断；完整杆取得曲线收益 | 断杆/细杆的残余误差、跨采集路径稳定性与独立对象验证 |
| 15张高度配对图、24曲线/18特权诊断；6份新几何版本、约572万点和12补丁 | 候选/身份稳定性、深度准确性、合格共同评分与独立对象 |
| Blender配置侧栏、骨架检查、插件打包和CI | 完整导入、增强、比较与撤回流程 |

最新[高度配对与相机点云版本](docs/experiments/2026-09-23-height-and-camera-versions.md)：断杆曲线p95降至14.08mm，细杆30.35mm仍差；圆柱法在新完整杆/断杆上拒绝，高度变化不是通用解法。6基础、12补丁、24保存视图通过撤回/重开，旧补丁跨版本被拒绝；未改原深度。295项诊断、15,400条独立射线、30,720次往返及3,840次独立来源像素核验通过。见[今日日志](docs/journal/2026-09-23.md)、[ADR0015](docs/decisions/0015-height-is-partial-and-camera-versions-are-explicit.md)。

此前[参考纹理辅助的新杆实验](docs/experiments/2026-09-23-reference-assisted-rods.md)：完整杆两阶段校正后R/P100%，断杆82%/84%、细杆68%/68%，尚未全面达标。断杆保留两段却偏移并侵入真实缺口；只换真实位姿的评价诊断恢复正确。变焦三项校正均扣留。289项测试、20,600条独立射线检查通过；参考板是新增采集条件，三种杆布局不是独立对象。见[ADR0014](docs/decisions/0014-residual-pose-before-rod-extent-changes.md)。

此前[相机联合优化与固定镜头](docs/experiments/2026-09-23-camera-bundle-and-fixed-lens.md)：两轮18项优化，6组证据不足保持原样。共享焦距两阶段方法在已知解析正例上得到保留像素与三维的同步改善；错误匹配压力下比单次鲁棒拟合更稳定。该轮没有GT初始化、没有新的杆收益、没有重标旧点云；285项测试通过。见[ADR0013](docs/decisions/0013-shared-calibration-and-training-outlier-control.md)。

此前[相机分支与对应入口](docs/experiments/2026-09-22-camera-heads-and-track-availability.md)：新增4份ray pose预测、18补丁和80对RGB轨迹审计。当前SIFT闭环56/60物理对应正确，但关键视角只剩1–2条；ORB闭环仍大量错误。15张解析RGB正反例揭示并修复密集轨迹划分问题，旧版保留；校正入口可区分这些可观测/不足条件，但不是相机准确性保证。见[日志](docs/journal/2026-09-22.md)、[ADR0012](docs/decisions/0012-correspondence-availability-before-camera-optimization.md)。

此前[读取器修复与估计相机](docs/experiments/2026-09-22-readout-split-and-estimated-cameras.md)：旧128项读取控制均达标、新64项58达标，6挑战仍有3项失败；真实点云评分仍不稳。前景身份已接4份新快照共3,810,240点，但估计相机两个接受结果都错误。单换真实K无正确输出，换真实位姿普通3正确/1错误、圆柱2正确；特权结果只作诊断，未写回模型输入/补丁。见[当天追加日志](docs/journal/2026-09-22.md)、[ADR0011](docs/decisions/0011-estimated-camera-error-before-more-rod-gates.md)。

此前[点补丁与共同读取器实验](docs/experiments/2026-09-22-point-patch-and-common-readout.md)：7份历史预测共5,821,200点已完成绑定、应用、保存、逐字节撤回及重开；56份协议文件和71,680次重复来源像素抽查通过。第一版读取器漏厚/噪声表面，第二版改善上限却会合成假中间轴；两版均未通过评价资格。4份历史估计相机仍无修改，本轮不提供独立对象收益证据。见[日志与本周排期](docs/journal/2026-09-22.md)、[复现](docs/point-patch-pilot.md)。

上一轮[前景点身份实验](docs/experiments/2026-09-21-foreground-identity.md)：两视图显式前景点身份层已实现，完成旧开发560项和冻结新4布局/20帧的720项回放；这些是重复条件，不是独立物体数。相同点击预算下，旧开发普通/圆柱各12次正确，新主条件普通13次、圆柱8次，均未扰动时零误收。新双杆可按点击切换目标；圆柱退出标注视图导致新缺口/邻杆未知，普通版4px错点产生一次假中间杆。一致错点邻杆仍会误收。20,800条独立射线核验通过，GT与推理隔离，旧数据完整保留。G1仍未通过；下一步验证局部身份观测与视角退出、跨杆假边缘带，再做独立对象/估计相机、共同读取器与实际补丁。不扩UI。 见[日志](docs/journal/2026-09-21.md)。

上一轮[圆柱筛选与冻结新布局](docs/experiments/2026-09-20-cylinder-screen-and-new-layouts.md)：圆柱轮廓像素筛选与单次视角退出已实现。旧60开发条件从4正确/1误收变为16正确/0误收，实际对应两个物体的重复条件；严格版缺口回退完整保留。冻结后新增6布局/30帧/72条件，基线12正确/0错误，新方法24正确/8错目标，错误来自同一近邻负例；最细约1.2px杆和遮挡仍拒绝。新正确有限轴p95约1.09–1.32mm，真缺口保留两段，但保护区容差接近覆盖为0.87%，非零。圆柱自洽不解决目标身份，G1仍未通过。下一步明确目标身份输入/验收，再做独立物体、估计相机与实拍验证；不扩UI，不发布自动补丁。

上一轮[最终支持修复与边缘宽度诊断](docs/experiments/2026-09-20-refit-support-and-edge-radius.md)：最终支持检查已修复并跑完60条件配对回放（3进程约7分33秒）。合成真缺口s04在两个窗口、两个身份提示下正确恢复两段，2.5cm容差内恢复/精确率100%，有限段p95约0.9mm，缺口内部未误填；四次回放只对应一个物体。干净杆原正确最佳模型仍在，但新增竞争解释导致拒绝；高光误输出被拒绝，近邻杆误收与遮挡拒绝仍在，cap16仍全拒。3009条最终候选支持契约违规为0。新增64个假设的无GT边缘射线宽度诊断，尚未启用新接受门槛。下一步测试考虑像素定位误差的左右边缘联合约束，再冻结规则做独立对象验证；G1开发中未通过。 210项本地诊断通过，记录见[日志](docs/journal/2026-09-20.md)。

此前[候选截断与窗口消融](docs/experiments/2026-09-20-candidate-window-ablation.md)：9月20日完成固定观测池的60条件开发消融：窗口0/8、候选8/16、身份提示0/8/16独立控制；10组旧条件完整回归通过。cap8有限段5次输出中，正确2次来自同一干净杆的提示回放，另有2次高光偏轴、1次错收近杆；cap16全部拒绝。扩大池没有恢复收益。另定位重拟合后漏验支持契约：前8有18/390、前16有23/736条候选不合格。下一步先在排序去重前补最终支持检查，再验证左右边缘共同几何；G1开发中未通过。 缓存关联与旧版5份完整输出SHA一致，单次历史计时约2.5–3.9倍加速。见[本周开工日志](docs/journal/2026-09-20.md)。

前轮[提示偏差与有限杆段测试](docs/experiments/2026-09-19-identity-stress-and-finite-segments.md)：30次运行仅2次正确输出，另有1次高光偏轴、1次错收近邻杆，26次拒绝；真实缺口与遮挡均未恢复。50帧独立射线核验通过，但所有运行均触及图像候选上限。下一步先拆开搜索窗口与guide变化，检查候选截断及物理边缘约束；不以放宽阈值刷分。

第一版离线证据原型已跑通：真实相机下最细杆的缺口改善，估计相机下全部目标被拒绝，中杆在新视角失败。仍不能宣称普通照片修复有效。设计背景见 [算法决定](docs/decisions/0004-evidence-guided-line-recovery.md)；经典几何组件本身不作为新算法贡献。

9月17日已确认：中杆检测会选中内部亮条，砖块背景又能产生极线自洽但物理错误的对应。材质干预支持外观导致偏差；新增拒绝对照仅减少部分错判，未提高准确覆盖。最新见 [中心偏差与背景对应审计](docs/experiments/2026-09-17-center-and-correspondence-audit.md)、[22种解析反例](docs/experiments/2026-09-17-profile-controls.md)、[后续算法决定](docs/decisions/0005-observed-bands-and-correspondence-validation.md)及[今日日志](docs/journal/2026-09-17.md)。

9月19日已实现多图像线保留与跨视图候选关联。它能排除不一致杂纹、保留双线歧义，但一致表面亮条和一致背景线仍会被误认；四视图门槛又会误拒只在三视图可见的真杆。因此它定位为候选关联层，尚不是物理身份验收。见[多视图候选实验](docs/experiments/2026-09-19-multiview-candidate-association.md)、[ADR 0006](docs/decisions/0006-multiview-consistency-is-not-identity.md)与[9月19日日志](docs/journal/2026-09-19.md)。

同日追加的新Blender身份包已冻结8种布局、40帧配对RGB/深度/ID/相机/网格并通过投影轴复核。几何层正确处理干净、遮挡、斜杆和目标旁背景杆，对亮条、高光和相邻双杆保持歧义；空目标ROI中的另一根真实杆仍会被接受。粗guide身份门在开发回放中消除了该假物体但挡不住近guide错误轴，该旧包使用理想轴投影guide，后续压力测试已暴露其局限。见[Blender身份实验](docs/experiments/2026-09-19-blender-identity-and-guide-guard.md)与[ADR 0007](docs/decisions/0007-guide-is-target-identity-input.md)。

此前[原图证据与新视角验收](docs/experiments/2026-09-16-rod-evidence.md)、[运行说明](docs/rod-evidence-controls.md)、[裁剪/真实相机/直线对照](docs/experiments/2026-09-14-crop-oracle-line-controls.md)、[首轮基线](docs/experiments/2026-09-14-native-baseline.md)、[配对数据包](docs/experiments/2026-09-14-paired-thin-pack-v2.md)均保留。安装与实测记录见 [环境验证](docs/environment-verification.md)。

未实现的计算入口会明确返回非零，不生成空的“成功结果”。运行现成模型、提供插件界面本身不作为研究贡献。

## 开始使用

需要 Python 3.11 和 uv。以下从仓库根目录执行：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
uv sync --project reconstruction --locked
uv run --project reconstruction --locked creator --help
uv run --project reconstruction --locked creator status
```

三个项目分别管理环境，DA3 不加入核心的环境：

```powershell
uv sync --project backends/da3 --locked --extra inference
uv run --project backends/da3 --locked --extra inference creator-da3 status
uv sync --project experiments --locked
uv run --project experiments --locked creator-eval status
```

DA3 的 `inference` extra 包含固定版本的上游模型运行栈；模型与下载缓存存放在项目目录。完整安装、健康检查和 Blender 启动见 [环境说明](docs/environment.md)，实现进度见 [开发说明](docs/development.md)。

## Blender 插件

支持目标为 Blender 5.2 LTS；当前只提供配置与状态面板。

```powershell
python scripts/build_addon.py
```

在 Blender 偏好设置的 Add-ons 菜单选择 **Install from Disk**，安装生成的 `dist/creator_recon-0.1.0.zip`，启用后打开 3D Viewport 侧栏的 **Creator**。具体操作见 [插件说明](blender_addon/README.md)。

## 项目结构

```text
reconstruction/        独立核心：契约、应用、几何、修复、基础设施
backends/da3/          DA3 的隔离适配环境
blender_addon/         Blender 薄插件
experiments/          独立数据、对齐、指标和报告模块
schemas/v1/           未来从核心导出的正式文件协议
schemas/pilot/        已实现的点快照、曲线补丁与保存视图试行协议
configs/              可分享配置的位置
tests/fixtures/       小型测试样例
scripts/              独立推理、配对数据导出、核验、骨架检查与插件打包
docs/                 架构、决策和开发说明
```

模型权重、照片、运行日志、缓存和本机配置不进入 Git。主开发目录为 `D:/Creator-newage`，克隆到其他目录也可使用。原 C++ 原型由 Git 历史保留。

## 设计与研究边界

- [文档索引](docs/README.md)
- [系统设计](docs/architecture/01-system-design.md)
- [数据契约](docs/architecture/02-data-contracts.md)
- [重建与修复方法](docs/architecture/03-reconstruction-and-refinement.md)
- [任务与 Blender](docs/architecture/04-runtime-and-blender.md)
- [实验、测试与三个月计划](docs/architecture/05-evaluation-and-delivery.md)

首版限定具有足够观测证据的静态、不透明杆状结构，允许人工 ROI。研究输出以中心线为主，显示圆管的粗细不代表真实表面。当前没有已证明优于现有模型的结果。
