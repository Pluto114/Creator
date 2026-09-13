# 本机环境与空间管理

本项目在 Windows 上使用 Python 3.11、三个独立 uv 环境和 Blender 自带的独立 Python。核心环境使用 NumPy 2，DA3 使用 NumPy 1.26.4，不能混装。主开发根目录是 `D:/Creator-newage`。本机验证已通过，确切版本、三视图推理和显存记录见 [环境验证](environment-verification.md)。

## 每次开始开发

从项目根目录打开 PowerShell：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
uv run --project reconstruction --locked creator status
uv run --project experiments --locked creator-eval status
# 保留 inference extra；省略它执行 uv sync/run 会移除模型依赖。
uv run --project backends/da3 --locked --extra inference creator-da3 status
```

需要本机代理时，在入口附加 `-ProxyUrl http://127.0.0.1:7897`。代理地址不是通用项目要求，不写入全局 Git/系统代理。大型 PyTorch 文件可直连时，可以在当前终端设置：

```powershell
$env:NO_PROXY = 'download-r2.pytorch.org,download.pytorch.org,127.0.0.1,localhost'
```

上述入口只改变当前 PowerShell 进程及其子进程，不修改系统环境变量。现有 Python 基础解释器仍使用已经安装的 3.11；新增虚拟环境和数据都在项目目录。

## 重装与恢复

```powershell
.\scripts\Setup-Creator.ps1
# 或：.\scripts\Setup-Creator.ps1 -ProxyUrl http://127.0.0.1:7897
# 只恢复依赖、不下载权重：追加 -SkipModels
```

需要预先安装 Python 3.11 和 Git。脚本固定工具版本，使用三个已提交的 `uv.lock` 安装依赖；DA3 额外启用 `inference`。不修改 Blender 的 Python，也不安装另一个 CUDA Toolkit。GPU 包采用 PyTorch 官方 CUDA 12.8 索引；本机已有 NVIDIA 驱动用于运行这些预编译包。

| 用途 | Python 路径 |
| --- | --- |
| 工具 | `.venv-tools/Scripts/python.exe` |
| 核心与几何开发 | `reconstruction/.venv/Scripts/python.exe` |
| DA3 推理 | `backends/da3/.venv/Scripts/python.exe` |
| 独立实验与评测 | `experiments/.venv/Scripts/python.exe` |

## 缓存与磁盘

| 内容 | 项目内目录 |
| --- | --- |
| uv 下载、构建及共享依赖 | `.local/uv-cache/` |
| pip 缓存、构建临时文件 | `.local/pip-cache/`、`.local/tmp/` |
| 固定 DA3 权重 | `models/da3-base/<revision>/`、`models/da3-large/<revision>/` |
| Hugging Face 与 Torch 备用缓存 | `models/huggingface/`、`models/torch/` |
| CUDA、编译扩展、Matplotlib、ImageIO 等缓存 | `.local/` 下各专用目录 |
| Blender 插件、偏好设置和扩展 | `.local/blender-profile/` |
| 环境检查与模型冒烟结果 | `.runtime/` |
| 后续原始数据、研究运行 | `data/`、`runs/` |

同一 D 盘上的 uv 环境采用硬链接共享缓存中的依赖；逐目录相加会重复计算这些文件，不能当实际磁盘占用。停止安装任务后如需回收下载/构建缓存，优先使用 `uv cache prune`；不要手动编辑环境中的库文件，也不要在正在安装时清理缓存。模型、虚拟环境、缓存和本机报告均已被 Git 忽略。

## 模型与验证

`configs/models.lock.json` 记录模型仓库、不可变 revision、字节数和 SHA256；权重、配置、模型卡保存在同一个版本目录，下载成功才把 `.part` 发布为最终文件。

```powershell
.\scripts\Download-CreatorModels.ps1
.venv-tools\Scripts\python.exe scripts/check_environment.py
```

环境检查会确认缓存位于项目目录、核心数值库可执行、DA3 API 可导入，以及 PyTorch 矩阵运算和 torchvision NMS 真正在 CUDA 上运行。结果写入 `.runtime/environment.json`，失败返回非零。

真实模型冒烟测试需要至少两张照片，使用本地已校验的模型，禁用网络加载，不启用 GS：

```powershell
backends\da3\.venv\Scripts\python.exe scripts/smoke_da3.py --model base --images <image1.png> <image2.png> <image3.png>
backends\da3\.venv\Scripts\python.exe scripts/smoke_da3.py --model large --images <image1.png> <image2.png> <image3.png>
```

每次保存到 `.runtime/smoke/da3-<model>/<UTC时间>/`，包含深度、置信度、相机内外参、处理后图像，以及版本、输入哈希、耗时、CUDA 显存峰值。运行中和失败状态同样记录；输出通过只证明这组输入可运行，不证明几何准确、细杆修复成功或普遍不会显存不足。Creator 正式文件协议和任务执行器仍未接入。

## Blender

```powershell
.\scripts\Start-CreatorBlender.ps1 -Check   # 后台配置与验证
.\scripts\Start-CreatorBlender.ps1          # 打开项目专用 Blender
```

默认路径为本机已经安装的 `D:/CloudMusic/steam/steamapps/common/Blender/blender.exe`，其他机器用 `-BlenderPath` 指定。启动脚本同步源码中的插件副本，填好项目目录及核心 Python 路径，并在 D 盘专用 profile 中保存。打开后在 3D Viewport 侧栏看到 Creator 配置面板。当前插件只具备配置能力。

## 上游来源与范围

- [固定 DA3 源码与依赖](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4)
- [PyTorch 官方 CUDA 12.8 包](https://download.pytorch.org/whl/cu128/)、[uv 的 PyTorch 配置说明](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [DA3-BASE 模型卡](https://huggingface.co/depth-anything/DA3-BASE)、[DA3-LARGE-1.1 模型卡](https://huggingface.co/depth-anything/DA3-LARGE-1.1)

LARGE 的固定 Hugging Face 模型卡标记 Apache 2.0，但上游仓库模型表列 CC BY-NC 4.0；锁文件保留两种来源的冲突，没有据此声明可宽松再分发。本项目不会将权重提交 Git。

当前只准备基础深度/相机预测路线。GS、Gradio 应用、MoGe 对照环境、自定义 CUDA 编译和训练工具尚未安装；采用它们时单独评估依赖与空间。DA3 上游 basic dependencies 均由本次独立环境管理。