# Creator Newage

**Multi-view thin-structure reconstruction research, with a Blender interface.**

利用同一静态物体的多张照片，研究如何改善基础重建中细杆的遗漏、断裂和错误连接。基础深度与相机估计采用独立的 DA3 后端；自研部分是局部多视图几何恢复及其独立评测。

## 当前进度

当前处于 **G0：题目与实验链路验证**。独立脚本已能运行DA3、导出点云演示和带真值的合成数据；正式任务链路与自研修复算法仍在开发。

| 已可用 | 尚未实现 |
| --- | --- |
| 固定模型推理、NPZ结果、点云及GLB演示 | Creator正式DA3任务适配器与完整文件协议 |
| 六条件、30张同步RGB与独立深度/ID/相机/几何GT | 独立物体与真实照片验证 |
| 原生基线、6组裁剪、3组oracle、TLS/RANSAC与RGB交会对照 | 完整候选的共同曲线评分与跨对象验证 |
| 原图双边缘、逐段证据与拒绝原型，6张评测专用新视角 | 普通估计相机下有效恢复、正式补丁及可靠接受/拒绝策略 |
| 中心偏差的材质干预、22种亮度反例及背景物理对应审计 | 新几何布局、遮挡反例与可用的多候选观测方法 |
| 坐标、像素采样、数据哈希和输入/GT隔离检查 | 完整跨物体几何验证 |
| Blender配置侧栏、骨架检查、插件打包和CI | 完整导入、增强、比较与撤回流程 |

第一版离线证据原型已跑通：真实相机下最细杆的缺口改善，估计相机下全部目标被拒绝，中杆在新视角失败。仍不能宣称普通照片修复有效。设计背景见 [算法决定](docs/decisions/0004-evidence-guided-line-recovery.md)；经典几何组件本身不作为新算法贡献。

9月17日已确认：中杆检测会选中内部亮条，砖块背景又能产生极线自洽但物理错误的对应。材质干预支持外观导致偏差；新增拒绝对照仅减少部分错判，未提高准确覆盖。最新见 [中心偏差与背景对应审计](docs/experiments/2026-09-17-center-and-correspondence-audit.md)、[22种解析反例](docs/experiments/2026-09-17-profile-controls.md)、[后续算法决定](docs/decisions/0005-observed-bands-and-correspondence-validation.md)及[今日日志](docs/journal/2026-09-17.md)。

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
