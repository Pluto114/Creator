# 环境落地验证：2026-09-13

本页记录一次本机验证，不是算法质量报告。测试输入为 DA3 官方仓库固定提交 `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` 中 `assets/examples/robot_unitree.mp4` 的第 35、70、105 帧；它们只用于验证真实推理链路，不进入研究评测集。

## 机器与版本

- Windows，NVIDIA GeForce RTX 4060 Laptop GPU，8 GB；驱动 595.97。
- Python 3.11.4；uv 0.12.13；Blender 5.2.1 LTS，构建 `9e2066aef7ef`。
- 核心/评测：NumPy 2.4.6、SciPy 1.17.1、OpenCV headless 4.14.0.94、Pydantic 2.13.5、Pillow 12.3.0、trimesh 4.12.2。
- DA3：torch 2.10.0+cu128、torchvision 0.25.0+cu128、xformers 0.0.34、NumPy 1.26.4、OpenCV 4.11.0.86、Open3D 0.19.0、pycolmap 4.2.0。完整间接依赖以对应 `uv.lock` 为准。
- 本机已有 CUDA Toolkit 13.3，本次没有重复安装；实际 PyTorch 推理使用 wheel 内的 CUDA 12.8 运行库。

## 已通过的验证

核心与评测环境通过数值求解、相机旋转和基本几何库检查；CUDA 环境通过真实 GPU 矩阵运算与 torchvision NMS。Blender 插件在专用 D 盘 profile 中完成启用、设置、保存和再次启动验证。代码静态检查、4 个现有核心测试、骨架入口检查和后端打包通过。

DA3 使用已校验的本地模型目录并禁止 Hub 联网加载。三个输入视图经 `upper_bound_resize` 得到 **504×280**；采用上游默认混合精度，`infer_gs=False`、`use_ray_pose=False`、`ref_view_strategy=saddle_balanced`。

| 模型 | 权重加载 | 推理调用 | PyTorch 分配峰值 | PyTorch 保留峰值 |
| --- | ---: | ---: | ---: | ---: |
| BASE | 7.567 秒 | 2.846 秒 | 1188.3 MiB | 1744.0 MiB |
| LARGE-1.1 | 9.165 秒 | 2.243 秒 | 3025.3 MiB | 3756.0 MiB |

上述是各一次调用，没有预热、重复统计或控制全部机器负载；不能根据它宣称 LARGE 比 BASE 快。耗时不包含 Python 导入、首次字体缓存构建、下载或 Blender 启动。显存统计仅覆盖该 PyTorch 进程的分配器，不包括所有驱动/桌面占用，也不代表更多视图或更高分辨率一定可运行。

两档输出均通过视图数、形状、有限值和正深度检查：

- 深度与置信度：`[3, 280, 504]`，float32。
- 相机内参：`[3, 3, 3]`；外参：`[3, 3, 4]`，float32。
- 处理后的图像：`[3, 280, 504, 3]`，uint8。

完整本机证据（不提交 Git）：

- `.runtime/environment.json`
- `.runtime/smoke/da3-base/20260913T105041.337372Z/report.json`
- `.runtime/smoke/da3-large/20260913T105245.975614Z/report.json`
- 同目录 `prediction.npz`，以及 `.runtime/smoke/inputs/source.json`。

## 安装中发现并处理的问题

- Open3D 0.19.0 的 Windows wheel 比解析所采用的 Linux metadata 多要求 `ipywidgets`。已在 DA3 `inference` extra 中显式声明 Windows 依赖，并在恢复脚本中增加 `uv pip check`。
- 大型下载发生过连接停滞；增加低速超时/重试、保留断点文件，PyTorch 使用官方直连。一个已终止安装器遗留的约 1.9 GB 临时目录已清理。
- 工具环境中上一轮用于验证打包的旧 Creator wheel 已移除，避免误调用不含正式依赖的副本。
- 未设置项目缓存时，环境检查会在导入第三方依赖前失败；已实际验证该保护。

上游可能输出缺少 `Triton` 优化或 `gsplat` 的提示。本次 BASE/LARGE 深度与相机路径已在不安装这些可选功能的情况下跑通；没有修改第三方源码去伪造导入或吞掉推理异常。

## 依赖与磁盘检查

三个环境的 `uv pip check` 均通过；DA3 离线 `uv sync --locked --extra inference --dry-run --offline` 显示无需变更。最终 DA3 环境含 137 个包。磁盘可用空间本轮从 C 盘约 13.02 / D 盘约 219.41 GiB，变为 C 盘约 12.94 / D 盘约 211.25 GiB；这是整盘观测，包含同期其他进程活动。新增虚拟环境、权重、下载缓存和本机报告位于 D 盘项目目录。

## 项目能力边界

环境与独立上游推理已经可用。Creator 正式 DA3 适配器、文件协议、任务运行器、细杆算法与预览导入仍是骨架。因此 `creator-da3 status` 中的 `inference_ready: false` 描述的是尚未实现的项目入口，不否认独立环境冒烟成功。后续应先完成真实输入/输出契约与相机坐标验证，再接入系统。