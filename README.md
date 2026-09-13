# Creator Newage

**Multi-view thin-structure reconstruction research, with a Blender interface.**

利用同一静态物体的多张照片，研究如何改善基础重建中细杆的遗漏、断裂和错误连接。基础深度与相机估计采用独立的 DA3 后端；自研部分是局部多视图几何恢复及其独立评测。

## 当前进度

当前版本是 **项目骨架**，还不能生成或修复三维模型。

| 已可用 | 尚未实现 |
| --- | --- |
| 三个独立环境、两档 DA3 的真实 GPU 冒烟 | Creator 正式 DA3 任务适配器 |
| 命令行帮助、版本、能力状态 | 完整文件协议及不可变快照 |
| Blender 配置侧栏与注册/卸载 | 细杆匹配、拟合、补丁和预览导入 |
| 骨架检查、插件打包、持续集成配置 | 真实数据集、几何指标与性能结论 |

安装与实测记录见 [环境验证](docs/environment-verification.md)。上游模型已可独立运行，项目的正式任务链路仍待实现。

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
scripts/              骨架检查与插件打包
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
