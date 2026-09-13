# Creator 细结构重建技术栈决策

日期：2026-09-13

状态：确定首版的技术边界和验证顺序；具体依赖组合、模型显存和算法收益待实测。本次仅更新设计文档，未安装依赖、下载权重或编写实现。

取代：[0001：旧场景生成技术栈](0001-technology-stack.md)。配套设计：[详细系统架构](../architecture/01-system-design.md)。新项目根目录与包边界见 [ADR 0003](0003-newage-architecture.md)。

## 目标

交付一个 Blender 研究插件，利用同一静态对象的多张照片，改善初步重建中细杆的遗漏、断裂和错误连接。首版允许用户指定目标区域。输出包括保留的初步重建、可撤回的局部改动和改动依据。

适用范围先限定为静止、不透明、具有足够多视角观测的杆状结构。研究问题是实际几何质量与额外计算成本的改进；插件界面和调用现成模型本身不作为创新。

## 需要重新论证什么

Blender 交付入口、独立 Python 计算和文件交换保留。原有 Scene IR、摆放约束、方向观察调度已退出主线。现在的数据核心是照片、相机、重建快照和局部几何修改记录。

| 部分 | 本次决定 | 理由与边界 |
| --- | --- | --- |
| Blender 宿主 | 采用本机的 Blender 5.2.1 LTS，记录构建号 | 已验证可执行；不为沿用旧文档降到 4.5。Steam 更新后需重验宿主兼容性。 |
| 研究核心 | 独立 Python 3.11，uv 管理项目环境 | Python 版本是目标选择；本机可用解释器尚未确认。核心不导入 bpy。 |
| 数值与图像 | NumPy 2.x、SciPy、OpenCV、Pillow | 分别用于数组、邻域/数值求解、相机和图像几何、图像读取及方向处理；补丁版本经验证后锁定。 |
| 契约与验证 | Pydantic 2、pytest | 校验任务及元数据；测试只覆盖坐标、补丁、执行生命周期等关键行为。结构校验不能证明几何正确。 |
| 首个重建后端 | DA3，独立 Python 3.11 环境 | DA3-BASE 用于工程冒烟；DA3-LARGE-1.1 是后续质量对照候选。不得以较弱的 BASE 作为唯一研究对手。 |
| 张量/GPU | PyTorch 2.x，优先官方预编译 CUDA wheel | DA3 后端必需；核心中只有实际需要的张量运算才启用 GPU。具体 torch/CUDA/xformers 组合由后端环境验证决定。 |
| 三维文件工具 | trimesh 作为文件交换和网格工具 | 算法不能依赖“展示文件看起来正确”；评测读取完整数值产物。 |
| 合成测试与预览 | Blender；EEVEE 先用于诊断，正式测试固定渲染配置 | 真值来自已知几何和明确的相机定义。后续是否用 Cycles 生成某类外观，由实验需要决定。 |
| 数据格式 | JSON 元数据、NPY/NPZ 数值、PNG 掩码、固定布局二进制预览块 | 详细架构将插件预览细化为有预算的点/线段块，避免通用大文件导入；PLY/GLB 可作为后续交换导出。BLEND 是用户工作产物，不是算法唯一数据源。 |

OpenCV 核心环境优先使用无桌面窗口的发行包；第三方后端按其兼容要求选择发行包，同一环境不同时安装多个提供 cv2 的变体。

Shapely、python-fcl、场景布局求解器不再是首版依赖。Open3D、pycolmap 等若由第三方后端要求，限制在该后端环境；研究核心不提前依赖它们。自写 C++、CUDA 或 Vulkan 仅在测量证明有必要后讨论。

## 环境必须隔离

本次核对的上游声明存在直接冲突：DA3 要求 NumPy 小于 2，MoGe-3 要求 NumPy 至少为 2。因此不用一套环境容纳全部模型，也不把机器学习依赖安装进 Blender 自带 Python。

首版需要两套环境：

1. **核心环境**：数据准备、几何运算、任务控制和评测。
2. **DA3 环境**：官方模型及薄适配程序，输出约定的文件产物。

MoGe-3 在需要运行强对照时增加第三套独立实验环境。它目前已有代码和权重，不能按“尚未发布”处理；其稀疏运算依赖 FlexGEMM/Triton，需要另外验证 Windows、JIT 和显存。Windows 支持不能仅凭 Triton 名称否定。Linux/WSL 是具体后端确有阻塞时的备选，不是现在的先决条件。

MoGe-3 原生输出是单图几何，不提供跨照片的相机外参，因此首期作为单图细节对照，不能直接替换 DA3 的完整多视图流程。若组合 DA3 相机与 MoGe-3 几何，需要把尺度对齐、融合和全部计算成本定义为一条独立的复合基线。

Python 包代码可以由不同环境运行，但互相只交换版本化的文件，不传递 Python 对象或绑定某个 torch 版本的运行状态。每个已启用环境分别记录依赖锁定结果；上游仓库提交、模型权重 revision/哈希独立记录。

DA3 后端不能机械地同时接受所有依赖的最新版本。例如其 NumPy 限制还可能影响 OpenCV 等依赖选择；兼容结果必须来自实际安装和推理，不能以 README 安装命令代替验证。

## 首版模型与内存策略

- 先用 DA3-BASE 和少量视图验证数据链路，再尝试 DA3-LARGE-1.1。BASE 能运行只证明链路可用。
- 首阶段只请求深度、相机、置信评分及对应处理图像，关闭不需要的高斯分支和特征导出。
- 从公开 API 的 504 像素处理尺度附近建立资源基线；视图数量、实际图像尺寸、精度模式和峰值显存均需记录。这是冒烟起点，不是正式实验固定分辨率。
- 原始照片完整保留。细杆在降采样后可能只剩很少像素，分辨率与局部裁剪必须成为明确实验变量。
- 从 API 保存原始预测，不从已过滤、已抽样的展示 GLB 反推研究数据。
- 首版一次只运行一个本项目 GPU 作业。重建后端写出快照并退出后，再做后续处理；合成渲染与模型推理串行安排。
- 插件视口仍会占用 GPU。预算以实际可用显存为准，不把 8188 MiB 全部视为模型可用空间。
- 显存不足时明确失败并提供降低输入规模的建议；变更分辨率或视图集后产生新运行记录，不能静默改配置后仍计作原实验。
- 实验分别报告首次加载、完整运行和缓存快照上的局部修复耗时。

## 插件与计算进程

Blender 插件负责选择照片/区域、提交任务、展示状态、导入候选版本、比较和撤回。其宿主 Python 与外部 Python 可以不同版本。

插件启动独立的核心任务进程；任务进程按需调用选定后端环境。所有计算通过文件产物连接，首版不需要 HTTP 服务、数据库或消息中间件。

Blender 的数据修改在主线程执行。定时器仅轮询轻量状态和进度；大量推理、图像处理和优化在外部执行。任务归属、取消、文件切换和结果导入约定详见架构文档。

## 当前本机事实

| 项目 | 本次检查结果 |
| --- | --- |
| 宿主路径 | `D:/CloudMusic/steam/steamapps/common/Blender/blender.exe` |
| Blender | `--version` 成功：5.2.1 LTS，构建哈希 `9e2066aef7ef`，构建日期 2026-08-25 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 595.97 |
| Python | 启动器列出 3.11 与 3.13，但本次两个版本的启动命令都失败；尚不能认定原因或环境可用 |
| uv | 本次 PATH 查询没有找到；不据此断定磁盘任何位置都未安装 |
| 现有仓库 | C++ 原型已有用户修改；本次不修改这些代码 |
| 实际推理 | 尚未安装新环境、下载模型或运行推理；没有质量、速度或显存实测结论 |

这些发现不阻止确定架构，但在实施前必须完成运行环境验证。

## 锁定版本前的验证

1. 确认独立 Python 3.11 可执行并建立隔离环境；确认 Blender 的数值数组读取能力及外部进程启动方式。
2. 在 DA3 专属环境找到可安装、可推理的 torch、NumPy、OpenCV 与注意力依赖组合；记录确切版本。
3. 使用 3–5 张测试图跑通 DA3-BASE，核对输出字段、图像变换和相机投影，记录显存和耗时。
4. 独立验证核心加载后端产物，并在 Blender 中显示对应图像、相机和几何。
5. 再验证质量对照模型和正式实验输入规模。若某强基线不能运行，报告限制并选择可复现的替代对照，不宣称已胜过未运行的方法。

以上均为下一阶段工作，没有在本次设计阶段执行。

## 依据

- [Blender 5.2 LTS](https://www.blender.org/releases/5-2/)
- [Blender Python 集成](https://docs.blender.org/api/5.2/info_overview.html)
- [Blender Python 线程限制](https://docs.blender.org/api/5.0/info_gotchas_threading.html)
- [DA3 仓库与模型说明](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [DA3 依赖声明](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/pyproject.toml)
- [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py)
- [MoGe 当前依赖声明](https://github.com/microsoft/MoGe/blob/main/pyproject.toml)
- [MoGe-3 ViT-L 权重](https://huggingface.co/Ruicheng/moge-3-vitl/tree/main)
- [FlexGEMM 固定版本的 Windows 依赖](https://github.com/JeffreyXiang/FlexGEMM/blob/b2fadb29d41846c7981ade6801ffc689fae119cf/pyproject.toml)
- [PyTorch 官方安装选择](https://pytorch.org/get-started/locally/)
- [OpenCV 相机几何](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)

上游 main 和模型列表会变化，正式实验必须使用已记录的提交与权重版本。
