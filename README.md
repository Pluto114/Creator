# Creator Newage

**Multi-view thin-structure reconstruction research, with a Blender interface.**

利用同一静态物体的多张照片，研究如何改善基础重建中细杆的遗漏、断裂和错误连接。基础深度与相机估计采用独立的 DA3 后端；自研部分是局部多视图几何恢复及其独立评测。

## 当前进度

**G1：方法可行性开发中，尚未通过。** G0实验链路已收口；多候选关联已经接到有限杆段，但新压力测试仍有大面积拒绝、偏轴与错目标，尚无跨对象重复收益和正式补丁。

| 已可用 | 尚未实现 |
| --- | --- |
| 固定模型推理、NPZ结果、点云及GLB演示 | Creator正式DA3任务适配器与完整文件协议 |
| 六条件、30张同步RGB与独立深度/ID/相机/几何GT | 独立物体与真实照片验证 |
| 原生基线、6组裁剪、3组oracle、TLS/RANSAC与RGB交会对照 | 完整候选的共同曲线评分与跨对象验证 |
| 原图双边缘、逐段证据与拒绝原型，6张评测专用新视角 | 普通估计相机下有效恢复、正式补丁及可靠接受/拒绝策略 |
| 中心偏差材质干预、22种亮度反例、12种解析关联及8种新Blender身份布局 | 独立对象保留集与稳定的物理轴验收层 |
| 新增10种场景/50帧、30组提示偏移、条件式有限杆段及源码/输入/GT冻结检查 | 稳定的候选保留、目标身份和跨物体几何验证 |
| Blender配置侧栏、骨架检查、插件打包和CI | 完整导入、增强、比较与撤回流程 |

最新[候选截断与窗口消融](docs/experiments/2026-09-20-candidate-window-ablation.md)：9月20日完成固定观测池的60条件开发消融：窗口0/8、候选8/16、身份提示0/8/16独立控制；10组旧条件完整回归通过。cap8有限段5次输出中，正确2次来自同一干净杆的提示回放，另有2次高光偏轴、1次错收近杆；cap16全部拒绝。扩大池没有恢复收益。另定位重拟合后漏验支持契约：前8有18/390、前16有23/736条候选不合格。下一步先在排序去重前补最终支持检查，再验证左右边缘共同几何；G1开发中未通过。 缓存关联与旧版5份完整输出SHA一致，单次历史计时约2.5–3.9倍加速。见[本周开工日志](docs/journal/2026-09-20.md)。

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
